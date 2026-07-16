from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from fugue.bench.files import terminate_async_process_group as _terminate_process
from fugue.weave_support import trace_async_operation


@dataclass(frozen=True)
class ConversationRequest:
    messages: tuple[dict[str, str], ...]

    @classmethod
    def normalized(cls, messages: tuple[dict[str, str], ...]) -> ConversationRequest:
        if not messages:
            raise ValueError("a non-empty text message history is required")
        if len(messages) > 128:
            raise ValueError("message history exceeds the 128-message limit")
        normalized: list[dict[str, str]] = []
        total = 0
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role not in {"system", "developer", "user", "assistant"}:
                raise ValueError(f"unsupported message role: {role or 'missing'}")
            if not content.strip():
                raise ValueError("messages must contain non-empty text")
            total += len(content.encode("utf-8"))
            normalized.append({"role": role, "content": content})
        if total > 256 * 1024:
            raise ValueError("message text exceeds the 256 KiB limit")
        return cls(messages=tuple(normalized))

    def render_json(self) -> str:
        return (
            json.dumps(
                {
                    "instruction": (
                        "Continue this stateless conversation and return only the "
                        "assistant's next response."
                    ),
                    "messages": list(self.messages),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


@dataclass(frozen=True)
class WorkerRequest:
    request_id: str
    protocol: str
    conversation: ConversationRequest

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        return self.conversation.messages


class WorkerBackend(Protocol):
    deployment: dict[str, Any]

    def readiness(self) -> tuple[bool, tuple[str, ...]]: ...

    async def run(self, request: WorkerRequest) -> str: ...


def load_deployment(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"deployment spec is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict) or not value.get("deployment_id"):
        raise RuntimeError(f"deployment spec is invalid: {path}")
    return value


class HarborWorkerBackend:
    """Launch one isolated process that uses Harbor's Python API per request."""

    def __init__(
        self,
        deployment_path: Path,
        *,
        runtime_dir: Path | None = None,
        env: dict[str, str] | None = None,
        python: str | None = None,
    ) -> None:
        self.deployment_path = deployment_path.resolve()
        self.deployment = load_deployment(self.deployment_path)
        self.env = dict(os.environ if env is None else env)
        self.runtime_dir = (
            runtime_dir
            or Path(self.env.get("FUGUE_SERVE_RUNTIME_DIR", "/var/lib/fugue/requests"))
        ).resolve()
        self.python = python or sys.executable

    def readiness(self) -> tuple[bool, tuple[str, ...]]:
        missing = tuple(
            name
            for name in self.deployment.get("required_env") or []
            if not self.env.get(str(name), "").strip()
        )
        if not self.env.get("FUGUE_SERVE_API_KEY", "").strip():
            missing = (*missing, "FUGUE_SERVE_API_KEY")
        probe_url = self.env.get("FUGUE_SERVE_HARBOR_PROBE_URL", "").strip()
        if probe_url:
            try:
                response = httpx.get(probe_url, timeout=2.0)
                response.raise_for_status()
            except Exception:
                missing = (*missing, "harbor-remote")
        elif self._harbor_environment() == "docker":
            if shutil.which("docker") is None:
                missing = (*missing, "docker")
            else:
                try:
                    subprocess.run(
                        ["docker", "info"],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=2,
                    )
                except (OSError, subprocess.SubprocessError):
                    missing = (*missing, "docker-daemon")
        return not missing, missing

    async def run(self, request: WorkerRequest) -> str:
        async def operation() -> str:
            return await self._run_isolated(request)

        return await trace_async_operation(
            "fugue.serve.request",
            {
                "fugue.deployment_id": self.deployment["deployment_id"],
                "fugue.source_candidate_id": self.deployment["candidate_id"],
                "fugue.serve_request_id": request.request_id,
                "fugue.serve_protocol": request.protocol,
            },
            self.env,
            operation,
            lambda answer: {
                "status": "completed",
                "response_bytes": len(answer.encode()),
            },
        )

    async def _run_isolated(self, request: WorkerRequest) -> str:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        request_dir = self.runtime_dir / request.request_id
        request_dir.mkdir(mode=0o700)
        result_path = request_dir / "worker-result.json"
        try:
            task_dir = request_dir / "task"
            task_dir.mkdir()
            (task_dir / "instruction.md").write_text(
                request.conversation.render_json(), encoding="utf-8"
            )
            (task_dir / "task.toml").write_text(self._task_toml(), encoding="utf-8")
            config_path = request_dir / "job-config.json"
            config_path.write_text(
                json.dumps(
                    self._job_config(request, task_dir),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            config_path.chmod(0o600)
            child_env = self._request_env(request)
            process = await asyncio.create_subprocess_exec(
                self.python,
                "-m",
                "fugue.serve.worker",
                config_path.as_posix(),
                result_path.as_posix(),
                cwd=request_dir,
                env=child_env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                returncode = await process.wait()
            except asyncio.CancelledError:
                await _terminate_process(process)
                raise
            result = _read_worker_result(result_path)
            if returncode != 0 or result.get("status") != "completed":
                detail = str(result.get("error") or f"worker exited {returncode}")
                raise RuntimeError(detail)
            answer = str(result.get("answer") or "")
            if not answer.strip():
                raise RuntimeError("Harbor worker returned an empty final answer")
            return answer
        finally:
            shutil.rmtree(request_dir, ignore_errors=True)

    def _job_config(self, request: WorkerRequest, task_dir: Path) -> dict[str, Any]:
        candidate = self.deployment["candidate"]
        agent = _resolve_runtime_env(candidate["agent"], self.env)
        agent_env = dict(agent.get("env") or {})
        agent_env.update(
            {
                "FUGUE_RUN_ID": request.request_id,
                "FUGUE_RUN_NAME": f"serve-{self.deployment['deployment_id'][:12]}",
                "FUGUE_RUN_GROUP": f"serve-{self.deployment['deployment_id'][:12]}",
                "FUGUE_EXPERIMENT_ID": self.deployment["experiment_id"],
                "FUGUE_WORKLOAD_ID": "serve",
                "FUGUE_VARIANT_ID": self.deployment["variant_id"],
                "FUGUE_CONTEXT_SYSTEM_ID": self.deployment["context_system_id"],
                "FUGUE_CONTEXT_VERSION": str(
                    self.deployment.get("context_version") or ""
                ),
                "FUGUE_CONTEXT_CONFIG_HASH": str(
                    self.deployment.get("context_config_hash") or ""
                ),
                "FUGUE_AGENT_CONFIG_HASH": str(
                    self.deployment.get("agent_config_hash") or ""
                ),
                "FUGUE_CANDIDATE_ID": self.deployment["candidate_id"],
                "FUGUE_HARNESS": self.deployment["harness"],
                "FUGUE_MODEL": self.deployment["model"],
                "FUGUE_MODEL_PROVIDER": self.deployment["model_provider"],
                "FUGUE_TASK_NAME": request.request_id,
                "FUGUE_TRIAL_INDEX": "1",
                "FUGUE_COMPARISON_EXAMPLE_ID": request.request_id,
                "FUGUE_CONVERSATION_KEY": request.request_id,
                "FUGUE_TRACE_CONTENT": str(candidate.get("trace_content") or "full"),
                "FUGUE_SERVE_DEPLOYMENT_ID": self.deployment["deployment_id"],
                "FUGUE_SERVE_REQUEST_ID": request.request_id,
                "FUGUE_SERVE_PROTOCOL": request.protocol,
            }
        )
        agent_env.update(_model_route_env(self.deployment))
        agent["env"] = agent_env
        agent["override_timeout_sec"] = int(self.deployment["resources"]["timeout_sec"])
        environment = _resolve_runtime_env(candidate.get("environment") or {}, self.env)
        environment.update(
            {
                "type": self._harbor_environment(),
                "delete": True,
                "cpu_enforcement_policy": "limit",
                "memory_enforcement_policy": "limit",
                "override_cpus": self.deployment["resources"]["cpus"],
                "override_memory_mb": self.deployment["resources"]["memory_mb"],
                "override_storage_mb": self.deployment["resources"]["storage_mb"],
            }
        )
        return {
            "job_name": request.request_id,
            "jobs_dir": (task_dir.parent / "jobs").as_posix(),
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "quiet": True,
            "agents": [agent],
            "tasks": [{"path": task_dir.as_posix()}],
            "environment": environment,
            "verifier": {"disable": True},
            "retry": {"max_retries": 0},
            "extra_instruction_paths": candidate.get("extra_instruction_paths") or [],
        }

    def _task_toml(self) -> str:
        resources = self.deployment["resources"]
        required_env = sorted(self.deployment.get("required_env") or [])
        allowed = [
            *self.deployment.get("network_allowed_hosts", []),
            *_csv(self.env.get("FUGUE_SERVE_ALLOWED_HOSTS")),
        ]
        image = self.env.get("FUGUE_SERVE_WORKER_IMAGE") or self.deployment["image"]
        lines = [
            'version = "1.0"',
            "",
            "[metadata]",
            "",
            "[agent]",
            f"timeout_sec = {int(resources['timeout_sec'])}",
            "",
            "[verifier]",
            "timeout_sec = 1",
            "",
            "[environment]",
            f"docker_image = {json.dumps(image)}",
            'workdir = "/workspace"',
            f"cpus = {int(resources['cpus'])}",
            f"memory_mb = {int(resources['memory_mb'])}",
            f"storage_mb = {int(resources['storage_mb'])}",
            'network_mode = "allowlist"',
            f"allowed_hosts = {json.dumps(list(dict.fromkeys(allowed)))}",
            "",
            "[environment.env]",
            *[
                f"{json.dumps(name)} = {json.dumps('${' + name + '}')}"
                for name in required_env
            ],
            "",
        ]
        return "\n".join(lines)

    def _request_env(self, request: WorkerRequest) -> dict[str, str]:
        env = dict(self.env)
        env.update(
            {
                "FUGUE_RUN_ID": request.request_id,
                "FUGUE_RUN_NAME": f"serve-{self.deployment['deployment_id'][:12]}",
                "FUGUE_RUN_GROUP": f"serve-{self.deployment['deployment_id'][:12]}",
                "FUGUE_EXPERIMENT_ID": self.deployment["experiment_id"],
                "FUGUE_WORKLOAD_ID": "serve",
                "FUGUE_VARIANT_ID": self.deployment["variant_id"],
                "FUGUE_CONTEXT_SYSTEM_ID": self.deployment["context_system_id"],
                "FUGUE_CONTEXT_VERSION": str(
                    self.deployment.get("context_version") or ""
                ),
                "FUGUE_CONTEXT_CONFIG_HASH": str(
                    self.deployment.get("context_config_hash") or ""
                ),
                "FUGUE_AGENT_CONFIG_HASH": str(
                    self.deployment.get("agent_config_hash") or ""
                ),
                "FUGUE_CANDIDATE_ID": self.deployment["candidate_id"],
                "FUGUE_HARNESS": self.deployment["harness"],
                "FUGUE_MODEL": self.deployment["model"],
                "FUGUE_MODEL_PROVIDER": self.deployment["model_provider"],
                "FUGUE_TASK_NAME": request.request_id,
                "FUGUE_TRIAL_INDEX": "1",
                "FUGUE_COMPARISON_EXAMPLE_ID": request.request_id,
                "FUGUE_CONVERSATION_KEY": request.request_id,
                "FUGUE_TRACE_CONTENT": str(
                    self.deployment["candidate"].get("trace_content") or "full"
                ),
                "FUGUE_SERVE_DEPLOYMENT_ID": self.deployment["deployment_id"],
                "FUGUE_SERVE_REQUEST_ID": request.request_id,
                "FUGUE_SERVE_PROTOCOL": request.protocol,
            }
        )
        env.update(_model_route_env(self.deployment))
        env.pop("FUGUE_SERVE_API_KEY", None)
        return env

    def _harbor_environment(self) -> str:
        return (
            self.env.get("FUGUE_SERVE_HARBOR_ENVIRONMENT", "docker").strip() or "docker"
        )


def render_conversation(messages: tuple[dict[str, str], ...]) -> str:
    return ConversationRequest.normalized(messages).render_json()


def _read_worker_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _csv(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _model_route_env(deployment: dict[str, Any]) -> dict[str, str]:
    route = (deployment.get("candidate") or {}).get("model_route") or {}
    provider = route.get("provider")
    mapping = {
        "wandb": ("WANDB_INFERENCE_BASE_URL", route.get("chat_base_url")),
        "openai": ("OPENAI_BASE_URL", route.get("responses_base_url")),
        "anthropic": ("ANTHROPIC_BASE_URL", route.get("messages_base_url")),
    }
    name, value = mapping.get(provider, (None, None))
    return {name: str(value)} if name and value else {}


def _resolve_runtime_env(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_resolve_runtime_env(item, env) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _resolve_runtime_env(item, env) for key, item in value.items()
        }
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
        resolved = env.get(name)
        if resolved is None:
            raise RuntimeError(
                f"required runtime environment variable is missing: {name}"
            )
        return resolved
    return value


def new_request_id(prefix: str = "req") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
