"""Harbor agent subclasses: provider-neutral model plane + Weave tracing.

Fugue always traces to W&B Weave, while model calls can bill through W&B
Inference, OpenAI, or Anthropic. The shared ``ModelRoute`` determines whether
each harness can talk to the provider natively or should use the local LiteLLM
bridge.

Setup builds each harness and tracing integration into a locked runtime image.
Trials mount that image read-only and only write cell-specific configuration.

Every trial also writes ``/logs/agent/fugue-meta.json`` (host side)
with the run key, harness, model, experiment, variant, context system,
timestamps, and harness session ids so Weave traces can be joined back to
Harbor trials.

All four accept one canonical model string:
``wandb/...``, ``openai/...``, or ``anthropic/...``.
"""

import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

try:
    from typing import override
except ImportError:  # pragma: no cover - Python 3.11 fallback

    def override(func):
        return func


from harbor.agents.installed.base import BaseInstalledAgent, CliFlag
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.hermes import Hermes
from harbor.agents.installed.openclaw import OpenClaw
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from fugue.agent_tracing import (
    agent_conversation_id,
    agent_conversation_name,
    codex_skill_instruction,
    conversation_id,
    normalize_trace_content,
    openclaw_agent_id,
    skill_invocation_evidence,
    stable_agent_name,
)
from fugue.artifacts import artifact_recoveries
from fugue.codex_mcp import render_codex_mcp_toml
from fugue.model_plane import (
    BRIDGE_BASE_URL_CONTAINER,
    ModelRoute,
    bridge_master_key,
    provider_client_env,
    resolve_model_route,
    trace_entity_project,
)
from fugue.registration import (
    context_registration_digest,
    skill_registration_probe_command,
)
from fugue.tool_policy import (
    HarnessToolPolicy,
    tool_result_guard_cli_flags,
    tool_result_guard_install_command,
)
from fugue.weave_support import (
    WEAVE_AGENTS_OTEL_ENDPOINT,
    weave_agents_otel_headers,
)

# Local working tree of the hermes-otel plugin (uploaded into Hermes
# containers; see README "Trace plane").
HERMES_OTEL_CHECKOUT = Path(
    os.environ.get(
        "HERMES_OTEL_CHECKOUT",
        str(Path.home() / "Documents" / "GitHub" / "hermes-otel"),
    )
)

# Weave node SDK built from the OTel-2.x migration branch
# (wandb/weave ashah/node-sdk-otel-2x, sdks/node -> `pnpm pack`). The published
# weave SDK (<=0.16.2) ships the OTel 1.x trace stack, which crashes at load
# under OpenClaw's managed override @opentelemetry/core@2.8.0
# ("TracesSamplerValues.AlwaysOn"). Drop once the branch is released to npm.
# vendor/ sits at the repo root, two levels above this module (fugue/agents/).
WEAVE_NODE_SDK_TGZ = Path(
    os.environ.get(
        "WEAVE_NODE_SDK_TGZ",
        Path(__file__).resolve().parent.parent.parent / "vendor" / "weave-node-sdk.tgz",
    )
)

# Plugin runtime files (per hermes-otel README "File structure" + packaging
# needs). Everything else in the checkout (website/ is 746MB, tests, video,
# dashboard) stays on the host.
_HERMES_OTEL_FILES = (
    "plugin.yaml",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "config.yaml.example",
)
_HERMES_OTEL_DIRS = ("skills",)


def _require_env(key_name: str, purpose: str) -> str:
    key = os.environ.get(key_name, "").strip()
    if not key:
        raise ValueError(f"{key_name} is not set. Source the repo .env for {purpose}.")
    return key


def _require_trace_key() -> str:
    return _require_env("WANDB_API_KEY", "Weave tracing")


def _require_model_key(route: ModelRoute) -> str:
    return _require_env(route.api_key_env, f"{route.display_model} model calls")


def _weave_entity_project() -> tuple[str, str]:
    return trace_entity_project(os.environ)


def _weave_project_slug() -> str:
    entity, project = _weave_entity_project()
    return f"{entity}/{project}"


def _experiment_name() -> str:
    return os.environ.get("FUGUE_RUN_NAME", "").strip() or "manual"


def _run_group() -> str:
    return os.environ.get("FUGUE_RUN_GROUP", "").strip() or _experiment_name()


def _split_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _json_env(key: str) -> Any:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _dedupe_tags(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        tag = str(value).strip()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _experiment_tags(
    harness: str, route: ModelRoute, context_system_id: str
) -> list[str]:
    prompt_ids = _split_tags(os.environ.get("FUGUE_PROMPT_ID"))
    skill_ids = _split_tags(os.environ.get("FUGUE_SKILL_IDS"))
    integration_ids = _split_tags(os.environ.get("FUGUE_INTEGRATION_IDS"))
    return _dedupe_tags(
        [
            *_split_tags(os.environ.get("FUGUE_TAGS")),
            "fugue",
            f"experiment-id:{os.environ.get('FUGUE_EXPERIMENT_ID', 'manual')}",
            f"run:{_experiment_name()}",
            f"group:{_run_group()}",
            f"harness:{harness}",
            f"variant:{os.environ.get('FUGUE_VARIANT_ID', 'baseline')}",
            f"context-system:{context_system_id}",
            *[f"prompt:{item_id}" for item_id in prompt_ids],
            *[f"skill:{item_id}" for item_id in skill_ids],
            *[f"integration:{item_id}" for item_id in integration_ids],
            f"provider:{route.provider}",
            f"model:{route.display_model}",
        ]
    )


def _bridge_url_v1() -> str:
    return f"{BRIDGE_BASE_URL_CONTAINER}/v1"


def _bridge_key() -> str:
    return bridge_master_key(os.environ)


def _chat_base_url(route: ModelRoute) -> str:
    return route.chat_base_url or _bridge_url_v1()


def _chat_key_env(route: ModelRoute) -> str:
    return route.api_key_env if route.chat_base_url else "LITELLM_MASTER_KEY"


def _chat_key(route: ModelRoute) -> str:
    return _require_model_key(route) if route.chat_base_url else _bridge_key()


def _messages_base_url(route: ModelRoute) -> str:
    return route.messages_base_url or BRIDGE_BASE_URL_CONTAINER


def _messages_key(route: ModelRoute) -> str:
    return _require_model_key(route) if route.provider == "anthropic" else _bridge_key()


def _responses_base_url(route: ModelRoute) -> str:
    return route.responses_base_url or _bridge_url_v1()


def _responses_key(route: ModelRoute) -> str:
    return _require_model_key(route) if route.provider == "openai" else _bridge_key()


def _codex_provider_name(route: ModelRoute) -> str:
    return "openai" if route.provider == "openai" else "fugue"


_STAGED_HERMES_OTEL: Path | None = None


def stage_hermes_otel_checkout() -> Path:
    """Copy the plugin's runtime files from the local checkout to a temp dir.

    Cached per-process so concurrent trials share one staging copy.
    """
    global _STAGED_HERMES_OTEL
    if _STAGED_HERMES_OTEL is not None and _STAGED_HERMES_OTEL.exists():
        return _STAGED_HERMES_OTEL

    src = HERMES_OTEL_CHECKOUT
    if not (src / "plugin.yaml").exists():
        raise FileNotFoundError(
            f"hermes-otel checkout not found at {src} (set HERMES_OTEL_CHECKOUT)"
        )

    staged = Path(tempfile.mkdtemp(prefix="hermes-otel-staged-"))
    for name in _HERMES_OTEL_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, staged / name)
    for py in src.glob("*.py"):
        shutil.copy2(py, staged / py.name)
    for dirname in _HERMES_OTEL_DIRS:
        if (src / dirname).is_dir():
            shutil.copytree(
                src / dirname,
                staged / dirname,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    tracer_path = staged / "tracer.py"
    source = tracer_path.read_text()
    needle = "attrs = dict(attributes or {})"
    replacement = (
        "attrs = {**(self.config.resource_attributes or {}), **dict(attributes or {})}"
    )
    if needle not in source and replacement not in source:
        raise RuntimeError("hermes-otel span attribute patch target was not found")
    tracer_path.write_text(source.replace(needle, replacement))
    _STAGED_HERMES_OTEL = staged
    return staged


class _TrialMetaMixin:
    """Writes /logs/agent/fugue-meta.json (host side) per trial.

    The run key contains the immutable execution, workload, record type, task,
    harness, context system, variant, and Fugue trial index. Weave traces are
    joined back to trials through that key and the local metadata file.
    """

    logs_dir: Path  # provided by BaseAgent
    TRACE_HARNESS: ClassVar[str]

    @property
    def context_system_id(self) -> str:
        return os.environ.get("FUGUE_CONTEXT_SYSTEM_ID", "none")

    @property
    def run_key(self) -> str:
        coordinates = (
            os.environ.get("FUGUE_RUN_ID"),
            os.environ.get("FUGUE_WORKLOAD_ID"),
            "trial",
            os.environ.get("FUGUE_TASK_NAME"),
            os.environ.get("FUGUE_HARNESS"),
            os.environ.get("FUGUE_CONTEXT_SYSTEM_ID"),
            os.environ.get("FUGUE_VARIANT_ID"),
            f"t{int(os.environ.get('FUGUE_TRIAL_INDEX', '1')):03d}",
        )
        return ":".join(value for value in coordinates if value)

    @property
    def conversation_key(self) -> str:
        return os.environ.get("FUGUE_CONVERSATION_KEY", "").strip() or self.run_key

    @property
    def conversation_id(self) -> str:
        return conversation_id(self.conversation_key)

    @property
    def trace_conversation_id(self) -> str:
        return agent_conversation_id(self.TRACE_HARNESS, self.conversation_key)

    @property
    def trace_content(self) -> str:
        return normalize_trace_content(os.environ.get("FUGUE_TRACE_CONTENT"))

    @property
    def capture_content(self) -> bool:
        return self.trace_content == "full"

    @property
    def job_name(self) -> str:
        return self.logs_dir.parent.parent.name

    def _meta_path(self) -> Path:
        return self.logs_dir / "fugue-meta.json"

    async def _detect_home(self, environment: BaseEnvironment) -> str:
        result = await self.exec_as_agent(
            environment, command='printf %s "$HOME"', timeout_sec=10
        )
        return (result.stdout or "").strip() or "/root"

    async def _install_tool_result_guard(
        self,
        environment: BaseEnvironment,
        harness: HarnessToolPolicy,
        config_path: PurePosixPath,
    ) -> None:
        command = tool_result_guard_install_command(
            self.model_route, harness, config_path
        )
        if command is None:
            return
        result = await self.exec_as_agent(
            environment,
            command=command,
            timeout_sec=30,
        )
        if result.return_code != 0:
            detail = (
                result.stderr or result.stdout or "tool-result guard failed"
            ).strip()
            raise RuntimeError(detail[-2_000:])

    async def _begin_trial(
        self, harness: str, route: ModelRoute, environment: BaseEnvironment
    ) -> None:
        os.environ["FUGUE_WEAVE_CONVERSATION_ID"] = self.trace_conversation_id
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = self._otel_resource_attributes(
            harness, route
        )
        os.environ["FUGUE_TRACE_ATTRIBUTES_JSON"] = json.dumps(
            {
                key: str(value)
                for key, value in self._trace_attributes(harness, route).items()
            },
            sort_keys=True,
        )
        await self._capture_runtime_fingerprint(environment, "pre_execution")
        registration_error: Exception | None = None
        try:
            self._context_registration_meta = self._context_registration(
                await self._install_context_runtime(environment)
            )
        except Exception as exc:
            registration_error = exc
            self._context_registration_meta = {
                "status": "failed",
                "delivery": os.environ.get("FUGUE_CONTEXT_DELIVERY", "portable"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        self._context_artifact_meta = await self._inject_context_artifact(environment)
        self._meta_begin(harness, route)
        if registration_error is not None:
            raise RuntimeError(
                "context registration probe failed before agent execution: "
                f"{registration_error}"
            ) from registration_error

    async def _capture_runtime_fingerprint(
        self, environment: BaseEnvironment, stage: str
    ) -> None:
        command = _runtime_fingerprint_command(stage)
        result = await self.exec_as_agent(environment, command=command, timeout_sec=30)
        output = (result.stdout or "").strip().splitlines()
        try:
            fingerprint = json.loads(output[-1]) if output else {}
        except json.JSONDecodeError:
            fingerprint = {"stage": stage, "status": "unavailable"}
        fingerprints = getattr(self, "_runtime_fingerprints", {})
        fingerprints[stage] = fingerprint
        self._runtime_fingerprints = fingerprints

    async def _lock_trial_mutators(self, environment: BaseEnvironment) -> None:
        command = """
set -eu
if [ -S /var/run/docker.sock ]; then
  echo 'trial policy rejected a mounted Docker socket' >&2
  exit 86
fi
for name in apt apt-get apk dnf yum microdnf pip pip3 uv uvx npm npx pnpm yarn \
            cargo rustup curl wget docker podman buildah; do
  for directory in /bin /sbin /usr/bin /usr/sbin /usr/local/bin /usr/local/sbin; do
    candidate="$directory/$name"
    if [ -e "$candidate" ]; then
      resolved="$(readlink -f "$candidate" 2>/dev/null || printf %s "$candidate")"
      case "$resolved" in
        /opt/fugue-agent-runtime/*) ;;
        *) chmod 000 "$resolved" 2>/dev/null || true ;;
      esac
    fi
  done
done
""".strip()
        result = await self.exec_as_root(environment, command=command, timeout_sec=30)
        if result.return_code != 0:
            detail = (
                result.stderr or result.stdout or "trial mutation policy failed"
            ).strip()
            raise RuntimeError(detail[-2_000:])

    async def _install_context_runtime(
        self, environment: BaseEnvironment
    ) -> dict[str, Any]:
        delivery = os.environ.get("FUGUE_CONTEXT_DELIVERY", "portable")
        if self.context_system_id == "none":
            return {"status": "not_assigned", "delivery": delivery}
        portable_command = os.environ.get("FUGUE_CONTEXT_COMMAND", "").strip()
        if delivery == "portable" and portable_command:
            result = await self.exec_as_agent(
                environment,
                command=(
                    f"command -v {shlex.quote(portable_command)} >/dev/null && "
                    f"{shlex.quote(portable_command)} probe"
                ),
                timeout_sec=30,
            )
            if result.return_code != 0:
                detail = (result.stderr or result.stdout or "probe failed").strip()
                raise RuntimeError(detail[:1_000])
            output = (result.stdout or "").strip().splitlines()
            try:
                payload = json.loads(output[-1]) if output else {}
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "portable context probe returned invalid JSON"
                ) from exc
            if payload.get("ok") is not True:
                raise RuntimeError(str(payload.get("error") or "probe was not ready"))
            return {
                "status": "registered",
                "delivery": delivery,
                "command": portable_command,
                "context_system_id": payload.get("context_system_id"),
            }
        servers = getattr(self, "mcp_servers", []) or []
        commands = " ".join(
            " ".join(
                [
                    str(getattr(server, "command", "") or ""),
                    *[str(item) for item in (getattr(server, "args", None) or [])],
                ]
            )
            for server in servers
        )
        if not commands:
            return {
                "status": ("pending_native_registration" if servers else "static"),
                "delivery": delivery,
                "servers": len(servers),
            }
        required = {"python"}
        if "uvx" in commands:
            required.add("uvx")
        if "npx" in commands:
            required.add("npx")
        if "project-rag" in commands:
            required.add("project-rag")
        checks = " && ".join(
            f"command -v {shlex.quote(command)} >/dev/null"
            for command in sorted(required)
        )
        result = await self.exec_as_root(
            environment,
            command=checks,
            timeout_sec=30,
        )
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "runtime check failed").strip()
            raise RuntimeError(detail[:1_000])
        return {
            "status": "registered",
            "delivery": delivery,
            "servers": len(servers),
        }

    def _meta_begin(self, harness: str, route: ModelRoute) -> None:
        entity, project = _weave_entity_project()
        tags = _experiment_tags(harness, route, self.context_system_id)
        prompt_id = os.environ.get("FUGUE_PROMPT_ID")
        variant_id = os.environ.get("FUGUE_VARIANT_ID") or "baseline"
        assigned_skills = _split_tags(os.environ.get("FUGUE_SKILL_IDS"))
        skill_registration = getattr(
            self,
            "_skill_registration_meta",
            {
                "status": "pending" if assigned_skills else "not_assigned",
                "skills_assigned": assigned_skills,
                "skills_registered": [],
            },
        )
        meta = {
            "run_key": self.run_key,
            "run_id": os.environ.get("FUGUE_RUN_ID"),
            "harbor_trial_id": self.logs_dir.parent.name,
            "trial_index": int(os.environ.get("FUGUE_TRIAL_INDEX", "1")),
            "comparison_example_id": os.environ.get("FUGUE_COMPARISON_EXAMPLE_ID"),
            "candidate_id": os.environ.get("FUGUE_CANDIDATE_ID"),
            "execution_fingerprint": os.environ.get("FUGUE_EXECUTION_FINGERPRINT"),
            "execution_kind": os.environ.get("FUGUE_EXECUTION_KIND", "agent"),
            "identity_schema_version": int(
                os.environ.get("FUGUE_IDENTITY_SCHEMA_VERSION", "1")
            ),
            "job_name": self.job_name,
            "harness": harness,
            "run_name": _experiment_name(),
            "run_group": _run_group(),
            "tags": tags,
            "model_provider": route.provider,
            "model": route.display_model,
            "tool_result_modalities": list(route.tool_result_modalities),
            "builder_model": os.environ.get("FUGUE_BUILDER_MODEL"),
            "judge_model": os.environ.get("FUGUE_JUDGE_MODEL"),
            "experiment_id": os.environ.get("FUGUE_EXPERIMENT_ID"),
            "workload_id": os.environ.get("FUGUE_WORKLOAD_ID"),
            "preset_id": os.environ.get("FUGUE_PRESET_ID"),
            "variant_id": variant_id,
            "context_system_id": self.context_system_id,
            "context_delivery": os.environ.get("FUGUE_CONTEXT_DELIVERY", "portable"),
            "context_version": os.environ.get("FUGUE_CONTEXT_VERSION"),
            "context_support": os.environ.get("FUGUE_CONTEXT_SUPPORT"),
            "context_config_hash": os.environ.get("FUGUE_CONTEXT_CONFIG_HASH"),
            "context_cache_keys": _json_env("FUGUE_CONTEXT_CACHE_KEYS"),
            "context_gateway_events_path": os.environ.get(
                "FUGUE_CONTEXT_GATEWAY_EVENTS_PATH"
            ),
            "expected_artifact_paths": _json_env("FUGUE_EXPECTED_ARTIFACT_PATHS"),
            "prompt_id": prompt_id,
            "prompt_hashes": _json_env("FUGUE_PROMPT_HASHES"),
            "skill_ids": _split_tags(os.environ.get("FUGUE_SKILL_IDS")),
            "skill_hashes": _json_env("FUGUE_SKILL_HASHES"),
            "skill_provenance": _json_env("FUGUE_SKILL_PROVENANCE"),
            "skills_assigned": assigned_skills,
            "skills_registered": skill_registration.get("skills_registered", []),
            "skill_registration": skill_registration,
            "skill_invocation_evidence": (
                {
                    "status": "unavailable",
                    "skills_invoked": [],
                    "reason": "execution has not produced skill-use evidence yet",
                }
                if assigned_skills
                else {"status": "not_applicable", "skills_invoked": []}
            ),
            "integration_ids": _split_tags(os.environ.get("FUGUE_INTEGRATION_IDS")),
            "integration_provenance": _json_env("FUGUE_INTEGRATION_PROVENANCE"),
            "harbor_config": os.environ.get("FUGUE_HARBOR_CONFIG"),
            "harbor_environment": os.environ.get("FUGUE_HARBOR_ENVIRONMENT"),
            "harbor_resources": _json_env("FUGUE_HARBOR_RESOURCES"),
            "agent_config_hash": os.environ.get("FUGUE_AGENT_CONFIG_HASH"),
            "dataset": os.environ.get("FUGUE_DATASET"),
            "repository": os.environ.get("FUGUE_REPOSITORY"),
            "base_commit": os.environ.get("FUGUE_BASE_COMMIT"),
            "manifest_path": os.environ.get("FUGUE_MANIFEST_PATH"),
            "weave_entity": entity,
            "weave_project": project,
            "trace_project": f"{entity}/{project}",
            "weave_agent_name": stable_agent_name(harness),
            "weave_conversation_key": self.conversation_key,
            "weave_conversation_id": self.trace_conversation_id,
            "planned_conversation_id": self.trace_conversation_id,
            "eval_predict_and_score_call_id": os.environ.get(
                "FUGUE_WEAVE_EVAL_PREDICT_AND_SCORE_CALL_ID"
            ),
            "evaluation_scope_id": os.environ.get("FUGUE_EVALUATION_SCOPE_ID"),
            "trace_content": self.trace_content,
            "runtime_fingerprints": getattr(self, "_runtime_fingerprints", {}),
            "context_registration": getattr(
                self, "_context_registration_meta", {"status": "unavailable"}
            ),
            "started_at": datetime.now(UTC).isoformat(),
        }
        if getattr(self, "_context_artifact_meta", None):
            meta["context_artifact"] = self._context_artifact_meta
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path().write_text(json.dumps(meta, indent=2) + "\n")

    async def _finish_trial(self, environment: BaseEnvironment) -> None:
        changed_paths: list[str] = []
        try:
            await self._capture_runtime_fingerprint(environment, "post_execution")
        except Exception:
            pass
        try:
            artifact_normalization = await self._normalize_artifact_paths(environment)
        except Exception as exc:
            artifact_normalization = [
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            ]
        try:
            repo_root = await self._container_repo_root(environment)
            result = await self.exec_as_agent(
                environment,
                command=(
                    f"git -C {shlex.quote(repo_root)} diff --name-only --relative; "
                    f"git -C {shlex.quote(repo_root)} ls-files --others "
                    "--exclude-standard"
                ),
                timeout_sec=30,
            )
            changed_paths = list(
                dict.fromkeys(
                    line.strip()
                    for line in (result.stdout or "").splitlines()
                    if line.strip()
                )
            )
        except Exception:
            pass
        self._meta_end(
            changed_paths=changed_paths,
            artifact_normalization=artifact_normalization,
        )

    async def _normalize_artifact_paths(
        self, environment: BaseEnvironment
    ) -> list[dict[str, str]]:
        expected = _json_env("FUGUE_EXPECTED_ARTIFACT_PATHS")
        if not isinstance(expected, list):
            return []
        repo_root = await self._container_repo_root(environment)
        commands: list[str] = []
        for recovery in artifact_recoveries(expected, repo_root):
            target = PurePosixPath(recovery.target)
            for candidate_text in recovery.candidates:
                candidate = PurePosixPath(candidate_text)
                commands.append(
                    f"if [ ! -e {shlex.quote(target.as_posix())} ] "
                    f"&& [ -f {shlex.quote(candidate.as_posix())} ]; then "
                    f"mkdir -p {shlex.quote(target.parent.as_posix())} && "
                    f"mv {shlex.quote(candidate.as_posix())} "
                    f"{shlex.quote(target.as_posix())} && "
                    "printf 'FUGUE_ARTIFACT_RECOVERED\\t%s\\t%s\\n' "
                    f"{shlex.quote(target.as_posix())} "
                    f"{shlex.quote(candidate.as_posix())}; fi"
                )
        if not commands:
            return []
        result = await self.exec_as_agent(
            environment,
            command="; ".join([*commands, "true"]),
            timeout_sec=30,
        )
        recovered: list[dict[str, str]] = []
        for line in (result.stdout or "").splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[0] == "FUGUE_ARTIFACT_RECOVERED":
                recovered.append(
                    {
                        "status": "recovered",
                        "target": parts[1],
                        "source": parts[2],
                    }
                )
        return recovered

    def _meta_end(
        self,
        *,
        changed_paths: list[str] | None = None,
        artifact_normalization: list[dict[str, str]] | None = None,
    ) -> None:
        try:
            meta = json.loads(self._meta_path().read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta["ended_at"] = datetime.now(UTC).isoformat()
        meta["changed_paths"] = changed_paths or []
        meta["artifact_normalization"] = artifact_normalization or []
        fingerprints = getattr(self, "_runtime_fingerprints", {})
        meta["runtime_fingerprints"] = fingerprints
        before = (fingerprints.get("pre_execution") or {}).get("comparable_digest")
        after = (fingerprints.get("post_execution") or {}).get("comparable_digest")
        meta["runtime_drift"] = (
            before != after if before is not None and after is not None else None
        )
        try:
            native_ids = self._extract_session_ids()
            meta["native_session_ids"] = native_ids
            meta["weave_conversation_ids"] = list(
                dict.fromkeys([self.trace_conversation_id, *native_ids])
            )
        except (OSError, json.JSONDecodeError):
            meta["native_session_ids"] = []
            meta["weave_conversation_ids"] = [self.trace_conversation_id]
        registration = meta.get("skill_registration")
        if isinstance(registration, dict):
            meta["skill_invocation_evidence"] = skill_invocation_evidence(
                self.logs_dir,
                self.TRACE_HARNESS,
                registration,
            )
        self._meta_path().write_text(json.dumps(meta, indent=2) + "\n")

    def _set_context_registration(self, value: dict[str, Any]) -> None:
        value = self._context_registration(value)
        self._context_registration_meta = value
        trace_environment = self._trace_environment(
            self.TRACE_HARNESS, self.model_route
        )
        os.environ.update(trace_environment)
        if hasattr(self, "_resolved_env_vars"):
            self._resolved_env_vars.update(trace_environment)
        try:
            meta = json.loads(self._meta_path().read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta["context_registration"] = value
        self._meta_path().write_text(json.dumps(meta, indent=2) + "\n")

    def _context_registration(self, value: dict[str, Any]) -> dict[str, Any]:
        registration = dict(value)
        if registration.get("status") not in {"registered", "static"}:
            registration.setdefault("registration_digest", None)
            return registration
        servers = []
        for server in getattr(self, "mcp_servers", []) or []:
            servers.append(
                {
                    key: item
                    for key, item in {
                        "name": getattr(server, "name", None),
                        "transport": getattr(server, "transport", None),
                        "url": getattr(server, "url", None),
                        "command": getattr(server, "command", None),
                        "args": list(getattr(server, "args", None) or []),
                    }.items()
                    if item not in (None, "", [])
                }
            )
        registration["context_system_id"] = self.context_system_id
        registration["registration_digest"] = context_registration_digest(
            context_system_id=self.context_system_id,
            delivery=str(
                registration.get("delivery")
                or registration.get("transport")
                or os.environ.get("FUGUE_CONTEXT_DELIVERY", "portable")
            ),
            context_config_hash=os.environ.get("FUGUE_CONTEXT_CONFIG_HASH", ""),
            command=(
                str(registration["command"])
                if registration.get("command")
                else None
            ),
            servers=servers,
        )
        return registration

    def _set_skill_registration(self, value: dict[str, Any]) -> None:
        self._skill_registration_meta = value
        try:
            meta = json.loads(self._meta_path().read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta["skills_assigned"] = value.get("skills_assigned", [])
        meta["skills_registered"] = value.get("skills_registered", [])
        meta["skill_registration"] = value
        self._meta_path().write_text(json.dumps(meta, indent=2) + "\n")

    async def _verify_skill_registration(
        self,
        environment: BaseEnvironment,
        directory: str,
    ) -> None:
        assigned = _split_tags(os.environ.get("FUGUE_SKILL_IDS"))
        if not assigned:
            self._set_skill_registration(
                {
                    "status": "not_assigned",
                    "skills_assigned": [],
                    "skills_registered": [],
                    "registration_digest": None,
                }
            )
            return
        command = skill_registration_probe_command(directory, assigned)
        result = await self.exec_as_agent(
            environment,
            command=command,
            timeout_sec=30,
        )
        lines = (result.stdout or "").strip().splitlines()
        try:
            payload = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            payload = {}
        registration = {
            "status": "registered" if result.return_code == 0 else "failed",
            "skills_assigned": assigned,
            "skills_registered": payload.get("skills_registered", []),
            "registration_digest": payload.get("registration_digest"),
            "directory": payload.get("directory", directory),
        }
        self._set_skill_registration(registration)
        if result.return_code != 0:
            detail = (
                result.stderr or result.stdout or "required skills were not registered"
            ).strip()
            raise RuntimeError(
                "skill registration probe failed before agent execution: "
                f"{detail[-2_000:]}"
            )

    def _extract_session_ids(self) -> list[str]:
        return []

    def _trace_attributes(self, harness: str, route: ModelRoute) -> dict[str, Any]:
        trial_index = int(os.environ.get("FUGUE_TRIAL_INDEX", "1"))
        attributes = {
            "gen_ai.agent.name": stable_agent_name(harness),
            "gen_ai.conversation.id": self.trace_conversation_id,
            "weave.conversation.name": agent_conversation_name(
                run_name=_experiment_name(),
                task_id=os.environ.get("FUGUE_TASK_NAME", ""),
                variant_id=os.environ.get("FUGUE_VARIANT_ID", "baseline"),
                trial_index=trial_index,
            ),
            "fugue.run_key": self.run_key,
            "fugue.run_id": os.environ.get("FUGUE_RUN_ID", ""),
            "fugue.run_name": _experiment_name(),
            "fugue.run_group": _run_group(),
            "fugue.job_name": self.job_name,
            "fugue.experiment_id": os.environ.get("FUGUE_EXPERIMENT_ID", ""),
            "fugue.workload_id": os.environ.get("FUGUE_WORKLOAD_ID", ""),
            "fugue.preset_id": os.environ.get("FUGUE_PRESET_ID", ""),
            "fugue.harness": harness,
            "fugue.variant_id": os.environ.get("FUGUE_VARIANT_ID", "baseline"),
            "fugue.context_system_id": self.context_system_id,
            "fugue.context_delivery": os.environ.get(
                "FUGUE_CONTEXT_DELIVERY", "portable"
            ),
            "fugue.context_registration_status": getattr(
                self, "_context_registration_meta", {}
            ).get("status", "unavailable"),
            "fugue.context_registration_digest": getattr(
                self, "_context_registration_meta", {}
            ).get("registration_digest", ""),
            "fugue.context_support": os.environ.get("FUGUE_CONTEXT_SUPPORT", ""),
            "fugue.task_id": os.environ.get("FUGUE_TASK_NAME", ""),
            "fugue.trial_id": self.logs_dir.parent.name,
            "fugue.trial_index": trial_index,
            "fugue.comparison_example_id": os.environ.get(
                "FUGUE_COMPARISON_EXAMPLE_ID", ""
            ),
            "fugue.candidate_id": os.environ.get("FUGUE_CANDIDATE_ID", ""),
            "fugue.execution_fingerprint": os.environ.get(
                "FUGUE_EXECUTION_FINGERPRINT", ""
            ),
            "fugue.execution_kind": os.environ.get("FUGUE_EXECUTION_KIND", "agent"),
            "fugue.evaluation_scope_id": os.environ.get(
                "FUGUE_EVALUATION_SCOPE_ID", ""
            ),
            "fugue.model_provider": route.provider,
            "fugue.model": route.display_model,
            "fugue.tool_result_modalities": "|".join(route.tool_result_modalities),
            "fugue.prompt_id": os.environ.get("FUGUE_PROMPT_ID", ""),
            "fugue.skill_ids": os.environ.get("FUGUE_SKILL_IDS", "").replace(",", "|"),
            "fugue.integration_ids": os.environ.get(
                "FUGUE_INTEGRATION_IDS", ""
            ).replace(",", "|"),
            "fugue.tags": os.environ.get("FUGUE_TAGS", "").replace(",", "|"),
            "fugue.conversation_id": self.trace_conversation_id,
        }
        attributes.update(
            {
                key: value
                for key, value in {
                    "weave.eval.predict_and_score_call_id": os.environ.get(
                        "FUGUE_WEAVE_EVAL_PREDICT_AND_SCORE_CALL_ID"
                    ),
                    "weave.eval.project_id": os.environ.get(
                        "FUGUE_WEAVE_EVAL_PROJECT_ID"
                    ),
                    "weave.eval.evaluation_name": os.environ.get(
                        "FUGUE_WEAVE_EVAL_NAME"
                    ),
                }.items()
                if value
            }
        )
        return attributes

    def _otel_resource_attributes(self, harness: str, route: ModelRoute) -> str:
        values = self._trace_attributes(harness, route)
        inherited = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "").strip()
        encoded = ",".join(
            f"{key}={str(value).replace(',', '|')}"
            for key, value in values.items()
            if value not in (None, "")
        )
        return ",".join(part for part in (inherited, encoded) if part)

    def _trace_environment(self, harness: str, route: ModelRoute) -> dict[str, str]:
        return {
            "OTEL_RESOURCE_ATTRIBUTES": self._otel_resource_attributes(harness, route),
            "FUGUE_TRACE_ATTRIBUTES_JSON": json.dumps(
                {
                    key: str(value)
                    for key, value in self._trace_attributes(harness, route).items()
                },
                sort_keys=True,
            ),
        }

    def _task_artifact_name(self) -> str:
        return (
            os.environ.get("FUGUE_TASK_NAME")
            or self.logs_dir.parent.name.rsplit("__", 1)[0]
        )

    async def _container_repo_root(self, environment: BaseEnvironment) -> str:
        configured = os.environ.get("FUGUE_CONTAINER_REPO_ROOT", "").strip()
        if configured:
            return configured
        result = await self.exec_as_agent(
            environment, command='printf %s "$PWD"', timeout_sec=10
        )
        return (result.stdout or "").strip() or "/app"

    async def _inject_context_artifact(
        self, environment: BaseEnvironment
    ) -> dict[str, Any] | None:
        system_id = self.context_system_id
        cache_root = os.environ.get("FUGUE_CONTEXT_CACHE_ROOT", "").strip()
        if not cache_root or system_id == "none":
            return None

        task_name = self._task_artifact_name()
        cache_keys = _json_env("FUGUE_CONTEXT_CACHE_KEYS")
        cache_key = cache_keys.get(task_name) if isinstance(cache_keys, dict) else None
        prepared_dir = Path(cache_root) / str(cache_key or "")
        if not cache_key or not prepared_dir.is_dir():
            raise FileNotFoundError(
                f"context artifact not prepared for {system_id}/{task_name}: "
                f"run `fugue setup --prepare-context --systems {system_id}` first"
            )

        if os.environ.get("FUGUE_CONTEXT_COMMAND", "").strip():
            return {
                "context_system_id": system_id,
                "task_name": task_name,
                "cache_key": cache_key,
                "source_dir": prepared_dir.as_posix(),
                "container_root": None,
                "sha256": _hash_dir(prepared_dir),
                "files": [],
                "delivery": "sidecar",
            }

        repo_root = await self._container_repo_root(environment)
        target_dir = f"{repo_root.rstrip('/')}/.fugue-context"
        await environment.upload_dir(source_dir=prepared_dir, target_dir=target_dir)
        files = [
            f".fugue-context/{p.relative_to(prepared_dir).as_posix()}"
            for p in sorted(prepared_dir.rglob("*"))
            if p.is_file()
        ]
        exclude_lines = "\n".join(f"/{path}" for path in files) + "\n"
        await self.exec_as_agent(
            environment,
            command=(
                f"ROOT={shlex.quote(repo_root)}; "
                'GIT_DIR="$(git -C "$ROOT" rev-parse --git-dir 2>/dev/null || true)"; '
                'if [ -n "$GIT_DIR" ]; then '
                'mkdir -p "$GIT_DIR/info"; '
                "cat >> \"$GIT_DIR/info/exclude\" <<'FUGUE_EXCLUDE'\n"
                f"{exclude_lines}"
                "FUGUE_EXCLUDE\n"
                "fi"
            ),
            timeout_sec=30,
        )
        return {
            "context_system_id": system_id,
            "task_name": task_name,
            "cache_key": cache_key,
            "source_dir": prepared_dir.as_posix(),
            "container_root": target_dir,
            "sha256": _hash_dir(prepared_dir),
            "files": files,
        }

    @staticmethod
    def _regex_ids(path: Path, pattern: str) -> list[str]:
        if not path.exists():
            return []
        found = re.findall(pattern, path.read_text(errors="replace"))
        seen: list[str] = []
        for f in found:
            if f not in seen:
                seen.append(f)
        return seen


def _hash_dir(path: Path) -> str:
    hasher = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        hasher.update(file.relative_to(path).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(file.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _runtime_fingerprint_command(stage: str) -> str:
    script = """
import hashlib
import importlib.metadata
import json
import os
import platform
import sys

packages = sorted(
    f"{dist.metadata.get('Name', '')}=={dist.version}"
    for dist in importlib.metadata.distributions()
)
comparable = {
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python_executable": os.path.realpath(sys.executable),
    "python_prefix": os.path.realpath(sys.prefix),
    "python_version": platform.python_version(),
    "packages_sha256": hashlib.sha256("\\n".join(packages).encode()).hexdigest(),
}
payload = {
    "stage": os.environ["FUGUE_FINGERPRINT_STAGE"],
    "status": "available",
    "comparable": comparable,
    "comparable_digest": hashlib.sha256(
        json.dumps(comparable, sort_keys=True).encode()
    ).hexdigest(),
    "path": os.environ.get("PATH", ""),
    "shell": os.environ.get("SHELL", ""),
    "uid": os.getuid(),
}
print(json.dumps(payload, sort_keys=True))
""".strip()
    encoded = shlex.quote(script)
    target = f"/logs/agent/runtime-fingerprint-{stage}.json"
    return (
        "mkdir -p /logs/agent; "
        f"export FUGUE_FINGERPRINT_STAGE={shlex.quote(stage)}; "
        "PY=$(command -v python3 || command -v python || true); "
        'if [ -z "$PY" ]; then '
        f'  printf \'%s\\n\' \'{{"stage":"{stage}","status":"unavailable"}}\' | tee {target}; '
        "else "
        f'  "$PY" -c {encoded} | tee {target}; '
        "fi"
    )


class FugueHermes(_TrialMetaMixin, Hermes):
    """Hermes with provider-routed model calls and local hermes-otel tracing.

    Model plane: hermes's ``openai`` builtin ignores ``OPENAI_BASE_URL`` for
    auth routing (verified 401), but config.yaml supports a ``providers:`` map
    for arbitrary OpenAI-compatible endpoints. Anthropic models therefore use
    the LiteLLM bridge's OpenAI-compatible chat endpoint.

    Tracing: hermes auto-discovers plugins from ``~/.hermes/plugins/<name>/
    plugin.yaml``, and the plugin's own config loader hardcodes
    ``~/.hermes/plugins/hermes_otel/config.yaml`` (DEFAULT_CONFIG_PATH) — so
    this class uses the default ``~/.hermes`` home (not the stock adapter's
    /tmp/hermes): the staged local checkout is uploaded to
    ``~/.hermes/plugins/hermes_otel``, pip-installed editable into the hermes
    venv (README install contract), and given a ``type: weave`` backend with
    run-key resource attributes.
    """

    TRACE_HARNESS = "hermes"
    _HERMES_VERSION = "v2026.6.5"

    @staticmethod
    @override
    def name() -> str:
        return "fugue-hermes"

    def __init__(self, *args, model_name: str | None = None, **kwargs):
        self.model_route = resolve_model_route(model_name)
        _require_model_key(self.model_route)
        _require_trace_key()
        _weave_entity_project()  # fail fast before containers spin up
        kwargs.setdefault("version", self._HERMES_VERSION)
        super().__init__(*args, model_name=self.model_route.model_id, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        runtime = "/opt/fugue-agent-runtime"
        await self.exec_as_root(
            environment,
            command=(
                f"test -x {runtime}/bin/hermes && "
                f"ln -sf {runtime}/bin/hermes /usr/local/bin/hermes"
            ),
            timeout_sec=30,
        )
        result = await self.exec_as_agent(
            environment,
            command=(
                f"PATH={runtime}/bin:$PATH hermes version && "
                f"test -s {runtime}/hermes-otel/fugue-patch-lock.json"
            ),
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "Hermes prepared runtime is missing or does not match its lock; "
                "run fugue setup --prepare"
            )
        await self._capture_runtime_fingerprint(environment, "verified")

    def _provider_name(self) -> str:
        return self.model_route.provider if self.model_route.chat_base_url else "fugue"

    @override
    def _build_register_mcp_servers_command(self) -> str | None:
        if not self.mcp_servers:
            return None
        import yaml

        servers: dict[str, dict[str, Any]] = {}
        for server in self.mcp_servers:
            if server.transport == "stdio":
                servers[server.name] = {
                    "command": server.command,
                    "args": list(server.args),
                }
            else:
                servers[server.name] = {"url": server.url}
        rendered = yaml.dump({"mcp_servers": servers}, default_flow_style=False)
        return (
            'cat >> "$HOME/.hermes/config.yaml" << \'MCPEOF\'\n'
            f"{rendered}MCPEOF"
        )

    @override
    def _build_register_skills_command(self) -> str | None:
        if not self.skills_dir:
            return None
        source = shlex.quote(str(self.skills_dir))
        return (
            'mkdir -p "$HOME/.hermes/skills" && '
            f'cp -R {source}/. "$HOME/.hermes/skills/"'
        )

    def _build_model_config_yaml(self) -> str:
        import yaml

        provider = self._provider_name()
        config: dict[str, Any] = {
            "model": self.model_route.model_id,
            "provider": provider,
            "toolsets": ["hermes-cli"],
            "agent": {
                "max_turns": 90,
                "disabled_toolsets": ["delegation"],
            },
            "memory": {"memory_enabled": False, "user_profile_enabled": False},
            "compression": {"enabled": True, "threshold": 0.85},
            "terminal": {"backend": "local", "timeout": 180},
            "delegation": {"max_iterations": 50},
            "checkpoints": {"enabled": False},
            "providers": {
                provider: {
                    "name": f"Fugue {self.model_route.provider}",
                    "api": _chat_base_url(self.model_route),
                    "key_env": _chat_key_env(self.model_route),
                    "models": [self.model_route.model_id],
                },
            },
            # Plugin enablement lives in this same file (`hermes plugins
            # enable` rewrites config.yaml in place — verified by diffing
            # before/after). Declare it here so the config overwrite can't
            # wipe it; mirrors the exact block the CLI writes.
            "plugins": {
                "enabled": ["hermes_otel"],
                "disabled": [],
                "entries": {"hermes_otel": {"allow_tool_override": False}},
            },
        }
        return yaml.dump(config, default_flow_style=False)

    def _build_otel_plugin_config_yaml(self) -> str:
        import yaml

        entity, project = _weave_entity_project()
        config: dict[str, Any] = {
            "backends": [
                {
                    "type": "otlp",
                    "name": "W&B Weave",
                    "endpoint": WEAVE_AGENTS_OTEL_ENDPOINT,
                    "metrics": False,
                },
            ],
            "resource_attributes": self._trace_attributes("hermes", self.model_route),
            "capture_previews": self.capture_content,
            "force_flush_on_session_end": True,
        }
        return yaml.dump(config, default_flow_style=False)

    async def _configure_hermes_otel(
        self, environment: BaseEnvironment, home: str
    ) -> None:
        plugin_dir = f"{home}/.hermes/plugins/hermes_otel"
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {shlex.quote(plugin_dir)} && "
                "cp -R /opt/fugue-agent-runtime/hermes-otel/. "
                f"{shlex.quote(plugin_dir)}/"
            ),
            timeout_sec=30,
        )

        otel_config = self._build_otel_plugin_config_yaml()
        await self.exec_as_agent(
            environment,
            command=(
                f"cat > {shlex.quote(plugin_dir + '/config.yaml')} << 'OTELEOF'\n"
                f"{otel_config}OTELEOF"
            ),
            timeout_sec=10,
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        instruction = self.render_instruction(instruction)
        await self._begin_trial("hermes", self.model_route, environment)

        entity, project = _weave_entity_project()
        trace_key = _require_trace_key()
        env: dict[str, str] = {
            **provider_client_env(self.model_route, os.environ),
            **self._trace_environment("hermes", self.model_route),
            "TERMINAL_ENV": "local",
            "WANDB_API_KEY": trace_key,
            "WANDB_ENTITY": entity,
            "WANDB_PROJECT": project,
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS": weave_agents_otel_headers(
                f"{entity}/{project}", trace_key
            ),
            "HARBOR_INSTRUCTION": instruction,
            # Per-span detail lands in the plugin dir's debug.log — the
            # fastest signal when validating trace delivery.
            "HERMES_OTEL_DEBUG": "true",
            _chat_key_env(self.model_route): _chat_key(self.model_route),
        }

        home = await self._detect_home(environment)
        await self._configure_hermes_otel(environment, home)

        config_yaml = self._build_model_config_yaml()
        await self.exec_as_agent(
            environment,
            command=(
                'mkdir -p "$HOME/.hermes/sessions" "$HOME/.hermes/skills" '
                '"$HOME/.hermes/memories" && '
                f"cat > \"$HOME/.hermes/config.yaml\" << 'EOF'\n{config_yaml}EOF"
            ),
            env=env,
            timeout_sec=10,
        )

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            result = await self.exec_as_agent(
                environment, command=mcp_command, env=env, timeout_sec=10
            )
            if result.return_code != 0:
                detail = (result.stderr or result.stdout or "MCP setup failed").strip()
                raise RuntimeError(detail[-2_000:])
            self._set_context_registration(
                {
                    "status": "registered",
                    "delivery": "native_mcp",
                    "servers": sorted(server.name for server in self.mcp_servers),
                    "probe": "$HOME/.hermes/config.yaml",
                }
            )
            env.update(self._trace_environment("hermes", self.model_route))
        skills_command = self._build_register_skills_command()
        if skills_command:
            result = await self.exec_as_agent(
                environment, command=skills_command, env=env, timeout_sec=10
            )
            if result.return_code != 0:
                detail = (
                    result.stderr or result.stdout or "skill setup failed"
                ).strip()
                raise RuntimeError(detail[-2_000:])
        await self._verify_skill_registration(
            environment,
            "$HOME/.hermes/skills",
        )
        await self._lock_trial_mutators(environment)

        run_cmd = (
            'export PATH="$HOME/.local/bin:$PATH" && '
            'hermes --yolo chat -q "$HARBOR_INSTRUCTION" -Q '
            f"--provider {shlex.quote(self._provider_name())} "
            f"--model {shlex.quote(self.model_route.model_id)} "
            "2>&1 | stdbuf -oL tee /logs/agent/hermes.txt"
        )

        try:
            await self.exec_as_agent(environment, command=run_cmd, env=env)
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        'export PATH="$HOME/.local/bin:$PATH" && '
                        "hermes sessions export /logs/agent/hermes-session.jsonl "
                        "--source cli 2>/dev/null; "
                        'cp "$HOME/.hermes/plugins/hermes_otel/debug.log" '
                        "/logs/agent/hermes-otel-debug.log 2>/dev/null; true"
                    ),
                    timeout_sec=30,
                )
            except Exception:
                pass
            await self._finish_trial(environment)

    @override
    def _extract_session_ids(self) -> list[str]:
        ids = self._regex_ids(
            self.logs_dir / "hermes-session.jsonl",
            r'"session_id"\s*:\s*"([^"]+)"',
        )
        if ids:
            return ids
        return self._regex_ids(self.logs_dir / "hermes.txt", r"session_id:\s*(\S+)")


class FugueOpenClaw(_TrialMetaMixin, OpenClaw):
    """OpenClaw with provider-routed chat calls and the weave-openclaw plugin.

    OpenClaw discovers external plugins through ``plugins.load.paths`` or its
    mutable extensions directory. Fugue points at the read-only setup-built
    tree and verifies the resolved plugin id, version, source, and loaded
    state before starting a turn. A warning that merely mentions Weave is not
    registration evidence.

    Plugin services start only in gateway mode. The loopback gateway wraps
    exactly one turn and receives SIGTERM after the final exporter flush.

    Spans land in the Weave *Agents* store (``/agents/otel/v1/traces`` ->
    query via ``POST /agents/spans/query``), not the calls table. The stable
    agent name groups all OpenClaw trials; Fugue attributes identify a trial.
    """

    TRACE_HARNESS = "openclaw"
    CLI_FLAGS = [
        CliFlag("openclaw_agent_id", cli="--agent", type="str", default="main"),
        CliFlag("thinking", cli="--thinking", type="str", default="off"),
        CliFlag("timeout", cli="--timeout", type="int"),
    ]

    _GATEWAY_TOKEN = "fugue-gateway"
    _GATEWAY_PORT = 18789
    _GATEWAY_LOG = "/logs/agent/openclaw-gateway.log"
    _GATEWAY_PID_FILE = "/tmp/openclaw-gateway.pid"
    _WEAVE_PLUGIN_ROOT = "/opt/fugue-agent-runtime/node_modules/weave-openclaw"
    _WEAVE_PLUGIN_VERSION = "0.1.1"
    _OPENCLAW_VERSION = "2026.7.1"
    _HEADLESS_TOOL_DENY = (
        "message",
        "browser",
        "canvas",
        "nodes",
        "cron",
        "gateway",
        "sessions_spawn",
        "sessions_send",
        "web_search",
        "web_fetch",
        "image",
        "memory_search",
        "memory_get",
    )

    @staticmethod
    @override
    def name() -> str:
        return "fugue-openclaw"

    def __init__(self, *args, model_name: str | None = None, **kwargs):
        self.model_route = resolve_model_route(model_name)
        _require_model_key(self.model_route)
        _require_trace_key()
        os.environ["OPENAI_API_KEY"] = _chat_key(self.model_route)
        os.environ["OPENAI_BASE_URL"] = _chat_base_url(self.model_route)
        _weave_entity_project()
        kwargs.setdefault("version", self._OPENCLAW_VERSION)
        super().__init__(
            *args, model_name=f"openai/{self.model_route.model_id}", **kwargs
        )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        runtime = "/opt/fugue-agent-runtime"
        await self.exec_as_root(
            environment,
            command=(
                f"test -x {runtime}/bin/node && test -x {runtime}/bin/openclaw && "
                f"ln -sf {runtime}/bin/node /usr/local/bin/node && "
                f"ln -sf {runtime}/bin/openclaw /usr/local/bin/openclaw"
            ),
            timeout_sec=30,
        )
        result = await self.exec_as_agent(
            environment,
            command=(
                f"PATH={runtime}/bin:$PATH openclaw --version | "
                f"grep -F {shlex.quote(self._OPENCLAW_VERSION)} && "
                f"test -s {runtime}/openclaw-patch-lock.json"
            ),
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "OpenClaw prepared runtime is missing or does not match its lock; "
                "run fugue setup --prepare"
            )
        await self._capture_runtime_fingerprint(environment, "verified")

    @override
    def _normalize_provider_models_schema(self, cfg: dict[str, Any]) -> None:
        models_root = cfg.setdefault("models", {})
        providers = models_root.setdefault("providers", {})
        prov_cfg = providers.setdefault("openai", {})
        raw_models = prov_cfg.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            prov_cfg["models"] = [
                {"id": self.model_route.model_id, "name": self.model_route.model_id}
            ]

    @override
    def _build_full_openclaw_config(self) -> dict[str, Any]:
        cfg = super()._build_full_openclaw_config()
        entity, project = _weave_entity_project()
        plugins = cfg.setdefault("plugins", {})
        load = plugins.setdefault("load", {})
        paths = load.setdefault("paths", [])
        if self._WEAVE_PLUGIN_ROOT not in paths:
            paths.append(self._WEAVE_PLUGIN_ROOT)
        allow = plugins.setdefault("allow", [])
        if "weave" not in allow:
            allow.append("weave")
        entries = plugins.setdefault("entries", {})
        entries["weave"] = {
            "enabled": True,
            "config": {
                "entity": entity,
                "project": project,
                "serviceName": "fugue",
                "agentName": stable_agent_name("openclaw"),
                "agentDescription": (
                    f"fugue {_experiment_name()} {self.context_system_id} / "
                    f"{self.model_route.display_model}"
                ),
                "apiKey": {
                    "source": "env",
                    "provider": "default",
                    "id": "WANDB_API_KEY",
                },
                "captureContent": self.capture_content,
                "flushIntervalMs": 1000,
            },
            "hooks": {"allowConversationAccess": True},
        }
        return cfg

    def _verify_plugin_command(self) -> str:
        """Require the immutable plugin tree to resolve as an enabled plugin."""
        return (
            "set -e; RUNTIME=/opt/fugue-agent-runtime; "
            "test -s $RUNTIME/openclaw-patch-lock.json; "
            f"test -s {self._WEAVE_PLUGIN_ROOT}/openclaw.plugin.json; "
            "openclaw plugins list --json "
            "> /logs/agent/weave-openclaw-registration.json; "
            "node -e 'const fs=require(\"fs\"); "
            "const value=JSON.parse(fs.readFileSync("
            "\"/logs/agent/weave-openclaw-registration.json\",\"utf8\")); "
            "const plugin=value.plugins.find((item)=>item.id===\"weave\"); "
            f"const root=\"{self._WEAVE_PLUGIN_ROOT}/dist/index.js\"; "
            f"if (!plugin || plugin.status!==\"loaded\" || "
            f"plugin.version!==\"{self._WEAVE_PLUGIN_VERSION}\" || "
            "plugin.source!==root) { console.error(JSON.stringify(plugin)); "
            "process.exit(1); }'"
        )

    def _verify_mcp_config_command(self) -> str:
        expected = {
            str(server.name): {
                key: value
                for key, value in {
                    "transport": getattr(server, "transport", None),
                    "url": getattr(server, "url", None),
                    "command": getattr(server, "command", None),
                    "args": list(getattr(server, "args", None) or []),
                }.items()
                if value not in (None, "", [])
            }
            for server in self.mcp_servers
        }
        probe = (
            "const fs=require('fs');"
            "const actual=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));"
            "const expected=JSON.parse(process.argv[2]);"
            "const names=(value)=>Object.keys(value).sort();"
            "if(JSON.stringify(names(actual))!==JSON.stringify(names(expected)))"
            "{console.error(JSON.stringify({expected:names(expected),"
            "registered:names(actual)}));process.exit(1);}"
            "for(const [name,want] of Object.entries(expected)){"
            "for(const [key,value] of Object.entries(want)){"
            "if(JSON.stringify(actual[name]?.[key])!==JSON.stringify(value))"
            "{console.error(JSON.stringify({name,key,expected:value,"
            "registered:actual[name]?.[key]}));process.exit(1);}}}"
        )
        return (
            "openclaw config validate --json "
            "> /logs/agent/openclaw-config-validation.json && "
            "openclaw config get mcp.servers --json "
            "> /logs/agent/openclaw-mcp-list.json && "
            f"node -e {shlex.quote(probe)} "
            "/logs/agent/openclaw-mcp-list.json "
            f"{shlex.quote(json.dumps(expected, sort_keys=True))}"
        )

    def _start_gateway_command(self) -> str:
        """Start a loopback gateway in the background and wait until ready.

        Plugin services (the Weave exporter among them) are only started by
        the gateway; ``openclaw agent --local`` runs never trace.
        """
        return (
            f"nohup openclaw gateway --bind loopback --port {self._GATEWAY_PORT} "
            f"> {self._GATEWAY_LOG} 2>&1 & echo $! > {self._GATEWAY_PID_FILE}; "
            "for i in $(seq 1 60); do "
            f"  grep -q '\\[gateway\\] ready' {self._GATEWAY_LOG} 2>/dev/null && break; "
            "  sleep 1; "
            "done; "
            f"grep -q '\\[gateway\\] ready' {self._GATEWAY_LOG} "
            "|| { echo 'gateway did not become ready'; "
            f"tail -50 {self._GATEWAY_LOG}; exit 1; }}; "
            f"grep -E 'plugin|weave' {self._GATEWAY_LOG} | head -5"
        )

    def _stop_gateway_command(self) -> str:
        # SIGTERM lets the plugin's shutdown hook force-flush the exporter.
        return (
            f"if [ -f {self._GATEWAY_PID_FILE} ]; then "
            f'kill "$(cat {self._GATEWAY_PID_FILE})" 2>/dev/null || true; '
            f"rm -f {self._GATEWAY_PID_FILE}; fi; sleep 3"
        )

    def _register_trial_agent_command(self) -> str:
        agent_id = openclaw_agent_id(self.conversation_id)
        return (
            "openclaw agents add "
            f"{shlex.quote(agent_id)} --workspace . "
            f"--model {shlex.quote(self.model_name)} "
            "--non-interactive --json"
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # The gateway owns plugin lifecycle, so it must bracket the one turn.
        await self._begin_trial("openclaw", self.model_route, environment)
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        provider, _ = self.model_name.split("/", 1)
        self._validate_provider(provider)

        env: dict[str, str] = {
            **provider_client_env(self.model_route, os.environ),
            **self._trace_environment("openclaw", self.model_route),
            "WANDB_API_KEY": _require_trace_key(),
            "OPENCLAW_GATEWAY_TOKEN": self._GATEWAY_TOKEN,
            "OPENCLAW_GATEWAY_PORT": str(self._GATEWAY_PORT),
            "OPENAI_API_KEY": _chat_key(self.model_route),
            "OPENAI_BASE_URL": _chat_base_url(self.model_route),
            "FUGUE_WEAVE_CONVERSATION_ID": self.trace_conversation_id,
        }
        for key in self._provider_env_keys(provider):
            val = self._get_env(key)
            if val:
                env[key] = val

        upload_path = self.logs_dir / self._UPLOAD_CONFIG_FILENAME
        upload_path.write_text(
            json.dumps(self._build_full_openclaw_config(), indent=2) + "\n",
            encoding="utf-8",
        )

        try:
            (self.logs_dir / "instruction.txt").write_text(instruction)
        except OSError:
            pass

        gateway_started = False
        try:
            await self.exec_as_agent(
                environment,
                command=(
                    "openclaw onboard --non-interactive --accept-risk "
                    "--auth-choice skip --workspace . --skip-daemon "
                    "--skip-channels --skip-skills --skip-hooks --skip-search "
                    "--skip-ui --skip-health --json"
                ),
                env=env,
            )

            copy_upload = (
                "mkdir -p ~/.openclaw && cp "
                f"{shlex.quote(f'{self._CONTAINER_LOGS_AGENT}/{self._UPLOAD_CONFIG_FILENAME}')} "
                "~/.openclaw/openclaw.json"
            )
            await self.exec_as_agent(environment, command=copy_upload, env=env)

            await self.exec_as_agent(
                environment,
                command=self._verify_plugin_command(),
                env=env,
                timeout_sec=60,
            )

            if self.mcp_servers:
                registration = await self.exec_as_agent(
                    environment,
                    command=self._verify_mcp_config_command(),
                    env=env,
                    timeout_sec=60,
                )
                if registration.return_code != 0:
                    detail = (
                        registration.stderr
                        or registration.stdout
                        or "OpenClaw MCP registration failed"
                    ).strip()
                    raise RuntimeError(detail[-2_000:])
                self._set_context_registration(
                    {
                        "status": "registered",
                        "delivery": "native_mcp",
                        "servers": sorted(
                            server.name for server in self.mcp_servers
                        ),
                        "probe": "openclaw config validate + mcp.servers",
                    }
                )
                env.update(self._trace_environment("openclaw", self.model_route))

            await self.exec_as_agent(
                environment,
                command=self._register_trial_agent_command(),
                env=env,
                timeout_sec=120,
            )

            skills_command = self._build_register_skills_command()
            if skills_command:
                result = await self.exec_as_agent(
                    environment,
                    command=skills_command,
                    env=env,
                )
                if result.return_code != 0:
                    detail = (
                        result.stderr or result.stdout or "skill setup failed"
                    ).strip()
                    raise RuntimeError(detail[-2_000:])
            home = await self._detect_home(environment)
            await self._verify_skill_registration(
                environment,
                f"{home}/.openclaw/skills",
            )
            await self._lock_trial_mutators(environment)

            await self.exec_as_agent(
                environment,
                command=self._start_gateway_command(),
                env=env,
                timeout_sec=180,
            )
            gateway_started = True

            self._resolved_flags["openclaw_agent_id"] = openclaw_agent_id(
                self.conversation_id
            )
            cli_flags = self.build_cli_flags()
            cli_flags_arg = (cli_flags + " ") if cli_flags else ""
            command = (
                f"openclaw agent --json {cli_flags_arg}"
                f"--model {shlex.quote(self.model_name)} "
                f"--message {escaped_instruction} "
                f"2>&1 </dev/null | stdbuf -oL tee /logs/agent/openclaw.txt"
            )
            await self.exec_as_agent(environment, command, env=env)
            await self._copy_openclaw_session_file_to_agent_logs(environment, env)

            # The plugin flushes on a 1s cadence; give the last batch a beat
            # before the gateway teardown force-flushes the rest.
            await self.exec_as_agent(
                environment, command="sleep 5", env=env, timeout_sec=30
            )
        finally:
            if gateway_started:
                try:
                    await self.exec_as_agent(
                        environment,
                        command=self._stop_gateway_command(),
                        env=env,
                        timeout_sec=60,
                    )
                except Exception:
                    pass
            await self._finish_trial(environment)

    @override
    def _extract_session_ids(self) -> list[str]:
        native_ids = self._regex_ids(
            self.logs_dir / "openclaw.txt", r'"sessionId"\s*:\s*"([^"]+)"'
        )
        return list(dict.fromkeys([self.trace_conversation_id, *native_ids]))


class FugueClaudeCode(_TrialMetaMixin, ClaudeCode):
    """Claude Code with provider-routed Messages calls and Weave tracing.

    Model plane: Anthropic models use Claude Code's native Messages path.
    Other providers use the local LiteLLM bridge, which exposes the Messages
    API and translates downstream.

    Tracing: ``weave-claude-code install --non-interactive --source=local``
    (the documented container-sandbox path) must run with the same
    ``CLAUDE_CONFIG_DIR`` the stock run() uses (/logs/agent/sessions), because
    ``claude plugin ...`` registers the marketplace/plugin inside that dir.
    """

    TRACE_HARNESS = "claude-code"
    _CLAUDE_CONFIG_DIR = (EnvironmentPaths.agent_dir / "sessions").as_posix()
    _CLAUDE_CODE_VERSION = "2.1.210"
    _WEAVE_PLUGIN_VERSION = "0.2.12"

    @staticmethod
    @override
    def name() -> str:
        return "fugue-claude-code"

    def __init__(self, *args, model_name: str | None = None, **kwargs):
        self.model_route = resolve_model_route(model_name)
        _require_model_key(self.model_route)
        _require_trace_key()
        _weave_entity_project()
        os.environ["ANTHROPIC_BASE_URL"] = _messages_base_url(self.model_route)
        os.environ["ANTHROPIC_API_KEY"] = _messages_key(self.model_route)
        os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        # Bridged models can round-trip provider-specific reasoning content as
        # Anthropic thinking blocks and fail validation on the next turn.
        os.environ["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
        kwargs.setdefault("version", self._CLAUDE_CODE_VERSION)
        super().__init__(*args, model_name=self.model_route.model_id, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        runtime = "/opt/fugue-agent-runtime"
        await self.exec_as_root(
            environment,
            command=(
                f"test -x {runtime}/bin/node && test -x {runtime}/bin/claude && "
                f"test -x {runtime}/bin/weave-claude-code && "
                f"test -x {runtime}/bin/npm && "
                f"ln -sf {runtime}/bin/node /usr/local/bin/node && "
                f"ln -sf {runtime}/bin/claude /usr/local/bin/claude && "
                f"ln -sf {runtime}/bin/weave-claude-code "
                "/usr/local/bin/weave-claude-code"
            ),
            timeout_sec=30,
        )
        result = await self.exec_as_agent(
            environment,
            command=(
                f"PATH={runtime}/bin:$PATH claude --version | "
                f"grep -F {shlex.quote(self._CLAUDE_CODE_VERSION)} && "
                f"PATH={runtime}/bin:$PATH weave-claude-code --version && "
                f"test -s {runtime}/claude-code-patch-lock.json && "
                f"test -s {runtime}/lib/node_modules/weave-claude-code/"
                ".claude-plugin/marketplace.json"
            ),
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "Claude Code prepared runtime is missing or does not match its lock; "
                "run fugue setup --prepare"
            )
        await self._capture_runtime_fingerprint(environment, "verified")

    async def _install_weave_plugin(self, environment: BaseEnvironment) -> None:
        env = {
            **self._trace_environment("claude-code", self.model_route),
            "CLAUDE_CONFIG_DIR": self._CLAUDE_CONFIG_DIR,
            "WEAVE_PROJECT": _weave_project_slug(),
            "WANDB_API_KEY": _require_trace_key(),
            "IS_SANDBOX": "1",
            "FUGUE_WEAVE_CONVERSATION_ID": self.trace_conversation_id,
            "NPM_CONFIG_PREFIX": "/opt/fugue-agent-runtime",
        }
        # Three container gotchas, all verified empirically:
        # 1. `--source=local` registers the npm tree as a *directory
        #    marketplace*, but the plugin entry inside the shipped
        #    marketplace.json still points at github ("source": "github"), so
        #    `claude plugin install` attempts a git/SSH clone and dies in the
        #    sandbox. Patch the plugin source to "./" — the npm package root
        #    is itself a valid plugin (.claude-plugin/plugin.json ships).
        # 2. The daemon validates transcript_path against os.homedir() only,
        #    but Harbor's CLAUDE_CONFIG_DIR (/logs/agent/sessions) puts
        #    transcripts outside home -> every session dies with "transcript
        #    _path outside home dir" and no spans are ever exported. Patch
        #    the check to also accept CLAUDE_CONFIG_DIR as a base (upstream
        #    bug in weave-claude-code; it should honor CLAUDE_CONFIG_DIR).
        # 3. The non-interactive installer registers the plugin but does NOT
        #    persist env credentials to settings.json — and the trace daemon
        #    is spawned later with claude's env, which has no W&B credentials
        #    (ANTHROPIC_* point at the bridge). Persist them explicitly; the
        #    daemon resolves env > settings.json. The agent name is stable;
        #    conversation and Fugue resource attributes identify the trial.
        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="/opt/fugue-agent-runtime/bin:$HOME/.local/bin:$PATH"; '
                "export NPM_CONFIG_PREFIX=/opt/fugue-agent-runtime; "
                'export npm_config_prefix="$NPM_CONFIG_PREFIX"; '
                'mkdir -p "$CLAUDE_CONFIG_DIR"; '
                "{ test -s /opt/fugue-agent-runtime/claude-code-patch-lock.json && "
                "weave-claude-code install --non-interactive --source=local && "
                'weave-claude-code config set weave_project "$WEAVE_PROJECT" && '
                'weave-claude-code config set wandb_api_key "$WANDB_API_KEY" && '
                "weave-claude-code config set agent_name claude-code && "
                "{ weave-claude-code status || true; }; } "
                "2>&1 | tee /logs/agent/weave-claude-code-install.txt"
            ),
            env=env,
            timeout_sec=300,
        )

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await self._begin_trial("claude-code", self.model_route, environment)
        self._resolved_env_vars.update(
            {
                **self._trace_environment("claude-code", self.model_route),
                "FUGUE_WEAVE_CONVERSATION_ID": self.trace_conversation_id,
            }
        )
        try:
            await self._install_weave_plugin(environment)
            await self._install_tool_result_guard(
                environment,
                "claude-code",
                PurePosixPath(self._CLAUDE_CONFIG_DIR) / "settings.json",
            )
            setup_commands = [f"mkdir -p {shlex.quote(self._CLAUDE_CONFIG_DIR)}/skills"]
            skills_command = self._build_register_skills_command()
            if skills_command:
                setup_commands.append(skills_command)
            mcp_command = self._build_register_mcp_servers_command()
            if mcp_command:
                setup_commands.append(mcp_command)
            registration = await self.exec_as_agent(
                environment,
                command=" && ".join(setup_commands),
                env={"CLAUDE_CONFIG_DIR": self._CLAUDE_CONFIG_DIR},
                timeout_sec=30,
            )
            if registration.return_code != 0:
                detail = (
                    registration.stderr
                    or registration.stdout
                    or "Claude registration failed"
                ).strip()
                raise RuntimeError(detail[-2_000:])
            await self._verify_skill_registration(
                environment,
                f"{self._CLAUDE_CONFIG_DIR}/skills",
            )
            if mcp_command:
                self._set_context_registration(
                    {
                        "status": "registered",
                        "delivery": "native_mcp",
                        "servers": sorted(server.name for server in self.mcp_servers),
                        "probe": f"{self._CLAUDE_CONFIG_DIR}/.claude.json",
                    }
                )
            await self._lock_trial_mutators(environment)
            await super().run(instruction, environment, context)
            # Let the plugin daemon flush the final turn before teardown.
            await self.exec_as_agent(
                environment,
                command=(
                    "sleep 5; "
                    "cp -R ~/.weave-claude-code/logs /logs/agent/weave-claude-code-logs "
                    "2>/dev/null || true"
                ),
                timeout_sec=60,
            )
        finally:
            await self._finish_trial(environment)

    @override
    def _extract_session_ids(self) -> list[str]:
        ids = self._regex_ids(
            self.logs_dir / "claude-code.txt", r'"session_id"\s*:\s*"([^"]+)"'
        )
        if ids:
            return ids
        sessions = self.logs_dir / "sessions" / "projects"
        if sessions.exists():
            return sorted({p.stem for p in sessions.rglob("*.jsonl")})
        return []


class FugueCodex(_TrialMetaMixin, Codex):
    """Codex CLI with provider-routed Responses calls and weave-codex tracing.

    Model plane: OpenAI models use the native Responses API. Other providers
    go through the LiteLLM bridge's Responses endpoint.

    Tracing: weave-codex's headless ``run`` wrapper collects the rollout after
    ``codex exec`` exits. Codex does not fire its interactive Stop hook for a
    failed turn, so wrapper ownership is required to trace both outcomes and
    avoids racing a detached collector during cleanup.

    The setup-built runtime patches weave-codex to honor the stable Fugue Agent
    identity and correlation attributes. Trials only verify that locked runtime;
    they never install or mutate Codex packages.
    """

    TRACE_HARNESS = "codex"
    # Bridged providers may reject OpenAI reasoning params; drop the stock
    # default of `-c model_reasoning_effort=high`.
    CLI_FLAGS = [flag for flag in Codex.CLI_FLAGS if flag.kwarg != "reasoning_effort"]
    _CODEX_VERSION = "0.143.0"
    _WEAVE_PLUGIN_VERSION = "0.1.1"
    _BRIDGED_DISABLED_FEATURES = (
        "apps",
        "browser_use",
        "browser_use_external",
        "browser_use_full_cdp_access",
        "code_mode_host",
        "computer_use",
        "goals",
        "image_generation",
        "in_app_browser",
        "multi_agent",
        "plugins",
        "remote_plugin",
        "tool_suggest",
        "unified_exec",
        "workspace_dependencies",
    )

    @staticmethod
    @override
    def name() -> str:
        return "fugue-codex"

    def __init__(self, *args, model_name: str | None = None, **kwargs):
        self.model_route = resolve_model_route(model_name)
        _require_model_key(self.model_route)
        _require_trace_key()
        _weave_entity_project()
        kwargs.setdefault("version", self._CODEX_VERSION)
        super().__init__(*args, model_name=self.model_route.model_id, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        runtime = "/opt/fugue-agent-runtime"
        await self.exec_as_root(
            environment,
            command=(
                f"test -x {runtime}/bin/node && "
                f"ln -sf {runtime}/bin/node /usr/local/bin/node && "
                f"ln -sf {runtime}/bin/codex /usr/local/bin/codex && "
                f"ln -sf {runtime}/bin/weave-codex /usr/local/bin/weave-codex"
            ),
            timeout_sec=30,
        )
        result = await self.exec_as_agent(
            environment,
            command=(
                f"test -x {runtime}/bin/codex && "
                f"test -x {runtime}/bin/weave-codex && "
                f"PATH={runtime}/bin:$PATH codex --version | "
                f"grep -F {shlex.quote(self._CODEX_VERSION)} && "
                f"PATH={runtime}/bin:$PATH weave-codex --help >/dev/null"
            ),
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "Codex prepared runtime is missing or does not match its lock; "
                "run fugue setup --prepare"
            )
        await self._capture_runtime_fingerprint(environment, "verified")

    def _build_model_config_toml(self) -> str:
        provider = _codex_provider_name(self.model_route)
        return (
            f'model = "{self.model_route.model_id}"\n'
            f'model_provider = "{provider}"\n'
            f"[model_providers.{provider}]\n"
            f'name = "Fugue {self.model_route.provider}"\n'
            f'base_url = "{_responses_base_url(self.model_route)}"\n'
            'env_key = "OPENAI_API_KEY"\n'
            'wire_api = "responses"\n'
        )

    def _codex_home(self) -> PurePosixPath:
        cell_digest = hashlib.sha256(self.run_key.encode()).hexdigest()[:16]
        return PurePosixPath("/tmp/fugue-codex") / cell_digest

    @override
    def _build_register_mcp_servers_command(self) -> str | None:
        config = render_codex_mcp_toml(self.mcp_servers)
        if not config:
            return None
        return f'printf %s {shlex.quote(config)} >> "$CODEX_HOME/config.toml"'

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await self._begin_trial("codex", self.model_route, environment)
        instruction = self.render_instruction(instruction)

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""
        feature_flags = (
            "--enable unified_exec "
            if self.model_route.provider == "openai"
            else "".join(
                f"--disable {feature} " for feature in self._BRIDGED_DISABLED_FEATURES
            )
        )
        hook_flags = "".join(
            f"{flag} "
            for flag in tool_result_guard_cli_flags(self.model_route, "codex")
        )

        codex_home = self._codex_home()
        remote_codex_home = codex_home.as_posix()
        instruction = codex_skill_instruction(
            instruction,
            skills=_split_tags(os.environ.get("FUGUE_SKILL_IDS")),
            directory=f"{remote_codex_home}/home/.agents/skills",
        )
        escaped_instruction = shlex.quote(instruction)
        weave_project = _weave_project_slug()
        env: dict[str, str] = {
            **provider_client_env(self.model_route, os.environ),
            **self._trace_environment("codex", self.model_route),
            "CODEX_HOME": remote_codex_home,
            "HOME": f"{remote_codex_home}/home",
            # Codex reads the configured provider's key from OPENAI_API_KEY
            # regardless of whether it is native OpenAI or the local bridge.
            "OPENAI_API_KEY": _responses_key(self.model_route),
            # The wrapper inherits these values; the key never enters a command.
            "WANDB_API_KEY": _require_trace_key(),
            "WEAVE_PROJECT": weave_project,
            # Consumed by the emit.js patch from install(); keeps all trials
            # grouped under the stable Codex agent.
            "WEAVE_CODEX_AGENT_NAME": stable_agent_name("codex"),
            "FUGUE_WEAVE_CONVERSATION_ID": self.trace_conversation_id,
            "PATH": (
                "/opt/fugue-agent-runtime/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin"
            ),
        }

        config_toml = self._build_model_config_toml()
        settings_json = json.dumps(
            {
                "weave_project": weave_project,
                "capture_content": self.capture_content,
            }
        )
        setup_command = (
            f'mkdir -p "$CODEX_HOME" "$HOME" {shlex.quote(EnvironmentPaths.agent_dir.as_posix())}\n'
            f"cat >>\"$CODEX_HOME/config.toml\" <<'TOML'\n{config_toml}TOML\n"
            "mkdir -p ~/.weave-codex\n"
            f"cat > ~/.weave-codex/settings.json <<'JSON'\n{settings_json}\nJSON\n"
            "weave-codex status --json "
            "2>&1 | tee /logs/agent/weave-codex-status.json\n"
        )

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"
        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            setup_command += f"\n{mcp_command}"
        expected_servers = json.dumps(
            sorted(str(server.name) for server in self.mcp_servers)
        )
        probe_script = (
            "import json,sys;"
            "rows=json.load(open(sys.argv[1]));"
            "expected=set(json.loads(sys.argv[2]));"
            "actual={str(row.get('name')) for row in rows "
            "if row.get('enabled') is not False};"
            "missing=sorted(expected-actual);"
            "unexpected=sorted(actual-expected);"
            "print(json.dumps({'expected':sorted(expected),"
            "'registered':sorted(actual),'missing':missing,"
            "'unexpected':unexpected}));"
            "sys.exit(bool(missing or unexpected))"
        )
        setup_command += (
            "\ncodex mcp list --json > /logs/agent/codex-mcp-list.json\n"
            f"python3 -c {shlex.quote(probe_script)} "
            "/logs/agent/codex-mcp-list.json "
            f"{shlex.quote(expected_servers)}\n"
        )

        setup_result = await self.exec_as_agent(
            environment, command=setup_command, env=env, timeout_sec=600
        )
        if setup_result.return_code != 0:
            if mcp_command:
                self._set_context_registration(
                    {
                        "status": "failed",
                        "transport": "native_mcp",
                        "servers": len(self.mcp_servers),
                        "probe": "codex mcp list --json",
                    }
                )
            detail = (
                setup_result.stderr or setup_result.stdout or "Codex setup failed"
            ).strip()
            raise RuntimeError(detail[-2_000:])
        await self._verify_skill_registration(
            environment,
            f"{remote_codex_home}/home/.agents/skills",
        )
        await self._install_tool_result_guard(
            environment,
            "codex",
            codex_home / "hooks.json",
        )
        if mcp_command:
            self._set_context_registration(
                {
                    "status": "registered",
                    "transport": "native_mcp",
                    "servers": len(self.mcp_servers),
                    "probe": "codex mcp list --json",
                }
            )
            env.update(self._trace_environment("codex", self.model_route))
        await self._lock_trial_mutators(environment)

        codex_output = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        codex_sessions = (EnvironmentPaths.agent_dir / "sessions").as_posix()

        try:
            await self.exec_as_agent(
                environment,
                command=(
                    "set -o pipefail; "
                    "weave-codex run -- codex exec "
                    "--dangerously-bypass-approvals-and-sandbox "
                    "--skip-git-repo-check "
                    "--json "
                    f"{feature_flags}"
                    f"{hook_flags}"
                    f"{cli_flags_arg}"
                    "-- "
                    f"{escaped_instruction} "
                    f"2>&1 </dev/null | tee {codex_output}"
                ),
                env=env,
            )
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()}\n"
                        'if [ -d "$CODEX_HOME/sessions" ]; then\n'
                        f"  rm -rf {codex_sessions}\n"
                        f'  cp -R "$CODEX_HOME/sessions" {codex_sessions}\n'
                        "fi"
                    ),
                    env=env,
                )
            except Exception:
                pass
            try:
                await self.exec_as_agent(
                    environment, command='rm -rf "$CODEX_HOME"', env=env
                )
            except Exception:
                pass
            await self._finish_trial(environment)

    @override
    def _extract_session_ids(self) -> list[str]:
        sessions = self.logs_dir / "sessions"
        ids: list[str] = []
        if sessions.exists():
            for p in sorted(sessions.rglob("rollout-*.jsonl")):
                m = re.search(
                    r"rollout-.*-([0-9a-f]{8}-[0-9a-f-]{27,})\.jsonl$", p.name
                )
                ids.append(m.group(1) if m else p.stem)
        return ids


class FugueLetta(_TrialMetaMixin, BaseInstalledAgent):
    """Pinned Letta Code harness with isolated local state per Harbor trial.

    Letta is intentionally an agent harness, not a portable context provider.
    Its MemFS and conversation state are collected under ``/logs/agent`` so
    stateful results remain separate from the four-harness context matrix.
    """

    TRACE_HARNESS = "letta"
    LETTA_VERSION = "0.26.2"

    @staticmethod
    @override
    def name() -> str:
        return "fugue-letta"

    def __init__(self, *args, model_name: str | None = None, **kwargs):
        self.model_route = resolve_model_route(model_name)
        _require_model_key(self.model_route)
        _require_trace_key()
        _weave_entity_project()
        kwargs.pop("version", None)
        super().__init__(
            *args,
            model_name=self.model_route.display_model,
            version=self.LETTA_VERSION,
            **kwargs,
        )

    def get_version_command(self) -> str:
        return "letta --version"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "apt-get update && apt-get install -y --no-install-recommends "
                "nodejs npm ca-certificates && rm -rf /var/lib/apt/lists/* && "
                f"npm install -g @letta-ai/letta-code@{self.LETTA_VERSION}"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
            timeout_sec=600,
        )

    def _connection(self) -> tuple[str, str, str]:
        route = self.model_route
        if route.provider == "anthropic":
            return "anthropic", route.display_model, route.messages_base_url or ""
        if route.provider == "openai":
            return "openai", route.display_model, route.chat_base_url or ""
        return (
            "openai",
            f"openai/{route.model_id}",
            f"{BRIDGE_BASE_URL_CONTAINER}/v1",
        )

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await self._begin_trial("letta", self.model_route, environment)
        provider, letta_model, base_url = self._connection()
        model_key = (
            bridge_master_key()
            if self.model_route.provider == "wandb"
            else _require_model_key(self.model_route)
        )
        env = {
            **provider_client_env(self.model_route, os.environ),
            "FUGUE_LETTA_API_KEY": model_key,
            "LETTA_LOCAL_BACKEND_DIR": "/logs/agent/letta-state",
            "WANDB_API_KEY": _require_trace_key(),
            "WEAVE_PROJECT": _weave_project_slug(),
        }
        connect = (
            f"letta --backend local connect {shlex.quote(provider)} "
            '--api-key "$FUGUE_LETTA_API_KEY"'
        )
        if base_url:
            connect += f" --base-url {shlex.quote(base_url)}"
        skills = ""
        if self.skills_dir:
            skills = (
                "mkdir -p .agents/skills; "
                f"cp -R {shlex.quote(str(self.skills_dir))}/. .agents/skills/; "
            )
        command = (
            'mkdir -p "$LETTA_LOCAL_BACKEND_DIR" /logs/agent; '
            f"{skills}"
            f"{connect} > /logs/agent/letta-connect.log 2>&1 && "
            "letta --backend local --new-agent "
            f"--model {shlex.quote(letta_model)} "
            f"-p {shlex.quote(self.render_instruction(instruction))} "
            "--output-format text --no-system-info-reminder "
            "2>&1 | tee /logs/agent/letta-output.txt"
        )
        try:
            await self.exec_as_agent(environment, command=command, env=env)
        finally:
            await self._finish_trial(environment)

    @override
    def _extract_session_ids(self) -> list[str]:
        root = self.logs_dir / "letta-state" / "memfs"
        if not root.is_dir():
            return []
        return sorted(path.name for path in root.iterdir() if path.is_dir())
