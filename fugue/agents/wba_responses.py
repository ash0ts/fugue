from __future__ import annotations

import json
import os
import shlex
from pathlib import PurePosixPath
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from fugue.agents.model_plane import (
    _CONTAINER_SECRET_ROOT,
    _bridge_key,
    _chat_base_url,
    _require_model_key,
    _require_trace_key,
    _TrialMetaMixin,
    _weave_entity_project,
    _weave_project_slug,
)
from fugue.bridge import bridge_container_base_url
from fugue.model_plane import (
    normalize_wba_transport_profile,
    provider_client_env,
    provider_request_headers,
    resolve_model_route,
    resolve_wba_transport_receipt,
)

_RUNTIME_ROOT = PurePosixPath("/opt/fugue-agent-runtime")
_RUNNER = _RUNTIME_ROOT / "bin/wba-runner"
_RUNTIME_LIB = _RUNTIME_ROOT / "lib"
_STATE_ROOT = PurePosixPath("/tmp/fugue-wba-responses")
_LOG_ROOT = PurePosixPath("/logs/agent")


class _WBAExecutionBase(BaseAgent):
    """Minimal Harbor command boundary used by the independent WBA loop."""

    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        return await environment.exec(
            command=command,
            env=env,
            cwd=cwd,
            timeout_sec=timeout_sec,
            user=environment.default_user,
        )

    async def exec_as_root(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        return await environment.exec(
            command=command,
            env=env,
            cwd=cwd,
            timeout_sec=timeout_sec,
            user="root",
        )


class FugueWBAResponses(_TrialMetaMixin, _WBAExecutionBase):
    """Task-neutral WBA-style loop with locked transport-profile ablations."""

    TRACE_HARNESS = "wba-responses"
    _VERSION = "0.1.4"

    @staticmethod
    @override
    def name() -> str:
        return "fugue-wba-responses"

    @override
    def version(self) -> str:
        return self._VERSION

    def __init__(
        self,
        *args: Any,
        model_name: str | None = None,
        transport_profile: str | None = None,
        context_window: int = 131_072,
        **kwargs: Any,
    ) -> None:
        supported = {"logs_dir", "logger", "mcp_servers", "skills_dir", "extra_env"}
        unknown = sorted(set(kwargs) - supported)
        if unknown:
            raise ValueError(
                "unknown WBA agent configuration field(s): " + ", ".join(unknown)
            )
        if kwargs.get("mcp_servers"):
            raise ValueError(
                "wba-responses does not support native MCP servers in contract v1"
            )
        self.model_route = resolve_model_route(model_name)
        _require_model_key(self.model_route)
        _require_trace_key()
        _weave_entity_project()
        self.transport_profile = normalize_wba_transport_profile(transport_profile)
        if not 4_096 <= context_window <= 1_000_000:
            raise ValueError("WBA context_window must be between 4096 and 1000000")
        self.context_window = context_window
        super().__init__(*args, model_name=self.model_route.model_id, **kwargs)

    @property
    def transport_receipt(self) -> dict[str, object]:
        return resolve_wba_transport_receipt(
            self.model_route,
            self.transport_profile,
        )

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        result = await self.exec_as_agent(
            environment,
            command=(
                f"test -x {_RUNNER.as_posix()} && "
                f"PYTHONPATH={_RUNTIME_LIB.as_posix()} "
                f"{_RUNNER.as_posix()} --version | grep -F {self._VERSION}"
            ),
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "WBA prepared runtime is missing or does not match its lock; "
                "run fugue setup --prepare"
            )
        await self._capture_runtime_fingerprint(environment, "verified")

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        interaction = self._task_interaction(instruction)
        if interaction.plan.kind == "model":
            raise RuntimeError(
                "wba-responses supports locked scripted follow-ups but not "
                "model-generated interactor follow-ups"
            )
        turns = [instruction, *interaction.plan.scripted_turns]
        await self._begin_trial("wba-responses", self.model_route, environment)

        state_root = _STATE_ROOT / self.trace_conversation_id
        config_path = state_root / "config.json"
        skills_root = state_root / "skills"
        events_path = _LOG_ROOT / "wba-responses-events.jsonl"
        summary_path = _LOG_ROOT / "wba-responses-summary.json"
        workspace = await self._container_repo_root(environment)
        entity, project = _weave_entity_project()
        runner_config = {
            "profile": self.transport_profile,
            "model_id": self.model_route.model_id,
            "display_model": self.model_route.display_model,
            "litellm_model": self.model_route.litellm_model.replace(
                "*", self.model_route.model_id
            ),
            "provider": self.model_route.provider,
            "provider_base_url": _chat_base_url(self.model_route),
            "bridge_base_url": f"{bridge_container_base_url(os.environ)}/v1",
            "provider_key_env": self.model_route.api_key_env,
            "bridge_key_env": "LITELLM_MASTER_KEY",
            "provider_headers": provider_request_headers(
                self.model_route,
                os.environ,
            ),
            "turns": turns,
            "session_id": self.trace_conversation_id,
            "conversation_id": self.trace_conversation_id,
            "conversation_name": self.job_name,
            "weave_project": f"{entity}/{project}",
            "trace_content": self.trace_content,
            "trace_attributes": {
                key: value
                for key, value in self._trace_attributes(
                    "wba-responses",
                    self.model_route,
                ).items()
                if value not in (None, "")
            },
            "workspace": workspace,
            "events_path": events_path.as_posix(),
            "summary_path": summary_path.as_posix(),
            "skills_dir": skills_root.as_posix() if self.skills_dir else None,
            "max_steps": 20,
            "context_window": self.context_window,
        }
        config_text = json.dumps(runner_config, sort_keys=True)
        env = {
            **provider_client_env(self.model_route, os.environ),
            **self._trace_environment("wba-responses", self.model_route),
            self.model_route.api_key_env: _require_model_key(self.model_route),
            "LITELLM_MASTER_KEY": _bridge_key(),
            "WANDB_API_KEY": _require_trace_key(),
            "WEAVE_PROJECT": _weave_project_slug(),
            "PYTHONPATH": _RUNTIME_LIB.as_posix(),
            "PATH": (
                f"{(_RUNTIME_ROOT / 'bin').as_posix()}:"
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
        }
        setup_commands = [
            f"mkdir -p {shlex.quote(state_root.as_posix())} {_LOG_ROOT.as_posix()}",
            (
                f"cat > {shlex.quote(config_path.as_posix())} <<'FUGUEWBACONFIG'\n"
                f"{config_text}\nFUGUEWBACONFIG"
            ),
        ]
        if self.skills_dir:
            setup_commands.extend(
                [
                    f"mkdir -p {shlex.quote(skills_root.as_posix())}",
                    f"cp -R {shlex.quote(str(self.skills_dir))}/. "
                    f"{shlex.quote(skills_root.as_posix())}/",
                ]
            )
        setup_result = await self.exec_as_agent(
            environment,
            command="\n".join(setup_commands),
            env=env,
            timeout_sec=30,
        )
        if setup_result.return_code != 0:
            detail = (
                setup_result.stderr or setup_result.stdout or "WBA setup failed"
            ).strip()
            raise RuntimeError(detail[-2_000:])
        await self._verify_skill_registration(environment, skills_root.as_posix())
        await self._lock_trial_mutators(environment)

        result = None
        try:
            try:
                result = await self.exec_as_agent(
                    environment,
                    command=(
                        "set -o pipefail; "
                        f"rm -rf {_CONTAINER_SECRET_ROOT.as_posix()}; "
                        f"{_RUNNER.as_posix()} --config "
                        f"{shlex.quote(config_path.as_posix())} "
                        "2>&1 | tee /logs/agent/wba-responses.txt"
                    ),
                    env=env,
                )
            finally:
                self._fugue_secret_files.clear()
            if result.return_code != 0:
                detail = (result.stderr or result.stdout or "WBA Agent failed").strip()
                raise RuntimeError(detail[-4_000:])
            summary = _summary_from_output(result.stdout or "")
            outputs = [str(item) for item in summary.get("outputs") or []]
            if len(outputs) != len(turns):
                raise RuntimeError(
                    "WBA runner did not return exactly one answer per scripted turn"
                )
            for index, output in enumerate(outputs):
                interaction.observe_agent(output or "No response text captured.")
                if index < interaction.plan.follow_up_count:
                    expected = interaction.plan.scripted_turns[index]
                    observed = await self._interaction_follow_up(interaction, index)
                    if observed != expected:
                        raise RuntimeError("locked scripted follow-up drifted")
            context.n_input_tokens = _optional_int(summary.get("input_tokens"))
            context.n_output_tokens = _optional_int(summary.get("output_tokens"))
            context.metadata = {
                "transport_receipt": self.transport_receipt,
                "session_id": summary.get("session_id"),
                "tool_calls": summary.get("tool_calls"),
                "tool_errors": summary.get("tool_errors"),
                "orphan_tool_outputs": summary.get("orphan_tool_outputs"),
                "normalization_errors": summary.get("normalization_errors"),
                "stream_events": summary.get("stream_events"),
                "retries": summary.get("retries"),
                "transport_errors": summary.get("transport_errors"),
                "compactions": summary.get("compactions"),
                "stop_reason": summary.get("stop_reason"),
            }
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command=f"rm -rf {shlex.quote(state_root.as_posix())}",
                    env=env,
                    timeout_sec=30,
                )
            except Exception:
                pass
            self._set_task_interaction_summary(interaction)
            await self._finish_trial(environment)

    @override
    def _extract_session_ids(self) -> list[str]:
        path = self.logs_dir / "wba-responses-summary.json"
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        session_id = str(value.get("session_id") or "").strip()
        return [session_id] if session_id else []


def _summary_from_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("event") == "session_summary":
            return value
    raise RuntimeError("WBA runner emitted no terminal session summary")


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None
