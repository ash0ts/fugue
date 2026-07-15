"""Harbor agent subclasses: provider-neutral model plane + Weave tracing.

Fugue always traces to W&B Weave, while model calls can bill through W&B
Inference, OpenAI, or Anthropic. The shared ``ModelRoute`` determines whether
each harness can talk to the provider natively or should use the local LiteLLM
bridge.

Tracing plane — every harness ships its Weave plugin inside the container:

- Hermes     -> local hermes-otel checkout (HERMES_OTEL_CHECKOUT, default
                ~/Documents/GitHub/hermes-otel) uploaded + pip-installed into
                the hermes venv; W&B Agents OTLP backend with Fugue span
                attributes.
- OpenClaw   -> ``openclaw plugins install weave-openclaw``; config entry
                with entity/project + ``hooks.allowConversationAccess``.
- Claude Code-> ``npm i -g weave-claude-code`` + non-interactive
                ``--source=local`` install against the run's CLAUDE_CONFIG_DIR.
- Codex      -> ``npm i -g weave-codex`` + Stop hook merged into
                ``$CODEX_HOME/hooks.json`` and explicit headless hook trust.

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
from pathlib import Path
from typing import Any

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
    conversation_id,
    normalize_trace_content,
    openclaw_agent_id,
    openclaw_conversation_id,
    stable_agent_name,
)
from fugue.model_plane import (
    BRIDGE_BASE_URL_CONTAINER,
    ModelRoute,
    bridge_master_key,
    provider_client_env,
    resolve_model_route,
    trace_entity_project,
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
        Path(__file__).resolve().parent.parent.parent
        / "vendor"
        / "weave-node-sdk.tgz",
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
        "attrs = {**(self.config.resource_attributes or {}), "
        "**dict(attributes or {})}"
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

    async def _begin_trial(
        self, harness: str, route: ModelRoute, environment: BaseEnvironment
    ) -> None:
        os.environ["FUGUE_WEAVE_CONVERSATION_ID"] = self.conversation_id
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
            self._context_registration_meta = await self._install_context_runtime(
                environment
            )
        except Exception as exc:
            registration_error = exc
            self._context_registration_meta = {
                "status": "failed",
                "transport": os.environ.get("FUGUE_CONTEXT_TRANSPORT", "portable"),
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

    async def _install_context_runtime(
        self, environment: BaseEnvironment
    ) -> dict[str, Any]:
        transport = os.environ.get("FUGUE_CONTEXT_TRANSPORT", "portable")
        if self.context_system_id == "none":
            return {"status": "not_assigned", "transport": transport}
        portable_command = os.environ.get("FUGUE_CONTEXT_COMMAND", "").strip()
        if transport == "portable" and portable_command:
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
                raise RuntimeError("portable context probe returned invalid JSON") from exc
            if payload.get("ok") is not True:
                raise RuntimeError(str(payload.get("error") or "probe was not ready"))
            return {
                "status": "registered",
                "transport": transport,
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
                "status": (
                    "pending_native_registration" if servers else "static"
                ),
                "transport": transport,
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
            "transport": transport,
            "servers": len(servers),
        }

    def _meta_begin(self, harness: str, route: ModelRoute) -> None:
        entity, project = _weave_entity_project()
        tags = _experiment_tags(harness, route, self.context_system_id)
        prompt_id = os.environ.get("FUGUE_PROMPT_ID")
        variant_id = os.environ.get("FUGUE_VARIANT_ID") or "baseline"
        meta = {
            "run_key": self.run_key,
            "run_id": os.environ.get("FUGUE_RUN_ID"),
            "harbor_trial_id": self.logs_dir.parent.name,
            "trial_index": int(os.environ.get("FUGUE_TRIAL_INDEX", "1")),
            "comparison_example_id": os.environ.get(
                "FUGUE_COMPARISON_EXAMPLE_ID"
            ),
            "candidate_id": os.environ.get("FUGUE_CANDIDATE_ID"),
            "job_name": self.job_name,
            "harness": harness,
            "run_name": _experiment_name(),
            "run_group": _run_group(),
            "tags": tags,
            "model_provider": route.provider,
            "model": route.display_model,
            "builder_model": os.environ.get("FUGUE_BUILDER_MODEL"),
            "judge_model": os.environ.get("FUGUE_JUDGE_MODEL"),
            "experiment_id": os.environ.get("FUGUE_EXPERIMENT_ID"),
            "workload_id": os.environ.get("FUGUE_WORKLOAD_ID"),
            "preset_id": os.environ.get("FUGUE_PRESET_ID"),
            "variant_id": variant_id,
            "context_system_id": self.context_system_id,
            "context_transport": os.environ.get(
                "FUGUE_CONTEXT_TRANSPORT", "portable"
            ),
            "context_version": os.environ.get("FUGUE_CONTEXT_VERSION"),
            "context_config_hash": os.environ.get("FUGUE_CONTEXT_CONFIG_HASH"),
            "context_cache_keys": _json_env("FUGUE_CONTEXT_CACHE_KEYS"),
            "expected_evidence_paths": _json_env(
                "FUGUE_EXPECTED_EVIDENCE_PATHS"
            ),
            "prompt_id": prompt_id,
            "prompt_hashes": _json_env("FUGUE_PROMPT_HASHES"),
            "skill_ids": _split_tags(os.environ.get("FUGUE_SKILL_IDS")),
            "skill_hashes": _json_env("FUGUE_SKILL_HASHES"),
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
            "weave_conversation_id": self.conversation_id,
            "planned_conversation_id": self.conversation_id,
            "eval_predict_and_score_call_id": os.environ.get(
                "FUGUE_WEAVE_EVAL_PREDICT_AND_SCORE_CALL_ID"
            ),
            "evaluation_scope_id": os.environ.get(
                "FUGUE_EVALUATION_SCOPE_ID"
            ),
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
        self._meta_end(changed_paths=changed_paths)

    def _meta_end(self, *, changed_paths: list[str] | None = None) -> None:
        try:
            meta = json.loads(self._meta_path().read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta["ended_at"] = datetime.now(UTC).isoformat()
        meta["changed_paths"] = changed_paths or []
        try:
            native_ids = self._extract_session_ids()
            meta["native_session_ids"] = native_ids
            meta["weave_conversation_ids"] = list(
                dict.fromkeys([self.conversation_id, *native_ids])
            )
        except (OSError, json.JSONDecodeError):
            meta["native_session_ids"] = []
            meta["weave_conversation_ids"] = [self.conversation_id]
        self._meta_path().write_text(json.dumps(meta, indent=2) + "\n")

    def _set_context_registration(self, value: dict[str, Any]) -> None:
        self._context_registration_meta = value
        try:
            meta = json.loads(self._meta_path().read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta["context_registration"] = value
        self._meta_path().write_text(json.dumps(meta, indent=2) + "\n")

    def _extract_session_ids(self) -> list[str]:
        return []

    def _trace_attributes(self, harness: str, route: ModelRoute) -> dict[str, Any]:
        attributes = {
            "gen_ai.agent.name": stable_agent_name(harness),
            "gen_ai.conversation.id": self.conversation_id,
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
            "fugue.context_transport": os.environ.get(
                "FUGUE_CONTEXT_TRANSPORT", "portable"
            ),
            "fugue.context_registration_status": getattr(
                self, "_context_registration_meta", {}
            ).get("status", "unavailable"),
            "fugue.task_id": os.environ.get("FUGUE_TASK_NAME", ""),
            "fugue.trial_id": self.logs_dir.parent.name,
            "fugue.trial_index": int(os.environ.get("FUGUE_TRIAL_INDEX", "1")),
            "fugue.comparison_example_id": os.environ.get(
                "FUGUE_COMPARISON_EXAMPLE_ID", ""
            ),
            "fugue.candidate_id": os.environ.get("FUGUE_CANDIDATE_ID", ""),
            "fugue.evaluation_scope_id": os.environ.get(
                "FUGUE_EVALUATION_SCOPE_ID", ""
            ),
            "fugue.model_provider": route.provider,
            "fugue.model": route.display_model,
            "fugue.prompt_id": os.environ.get("FUGUE_PROMPT_ID", ""),
            "fugue.skill_ids": os.environ.get("FUGUE_SKILL_IDS", "").replace(",", "|"),
            "fugue.tags": os.environ.get("FUGUE_TAGS", "").replace(",", "|"),
            "fugue.conversation_id": self.conversation_id,
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
        return os.environ.get("FUGUE_TASK_NAME") or self.logs_dir.parent.name.rsplit(
            "__", 1
        )[0]

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
        "if [ -z \"$PY\" ]; then "
        f"  printf '%s\\n' '{{\"stage\":\"{stage}\",\"status\":\"unavailable\"}}' | tee {target}; "
        "else "
        f"  \"$PY\" -c {encoded} | tee {target}; "
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
        await self._capture_runtime_fingerprint(environment, "pre_install")
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl git ripgrep xz-utils",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        installer = (
            "https://raw.githubusercontent.com/NousResearch/hermes-agent/"
            f"{self._HERMES_VERSION}/scripts/install.sh"
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"curl -fsSL {shlex.quote(installer)} | bash -s -- "
                "--skip-setup --skip-browser --no-skills --non-interactive "
                f"--branch {shlex.quote(self._HERMES_VERSION)} && "
                'export PATH="$HOME/.local/bin:$PATH" && '
                'export HERMES_HOME="${HERMES_HOME:-/tmp/hermes}" && '
                'mkdir -p "$HERMES_HOME" "$HERMES_HOME/sessions" '
                '"$HERMES_HOME/skills" "$HERMES_HOME/memories" && '
                "hermes version"
            ),
        )

    def _provider_name(self) -> str:
        return self.model_route.provider if self.model_route.chat_base_url else "fugue"

    def _build_model_config_yaml(self) -> str:
        import yaml

        provider = self._provider_name()
        config: dict[str, Any] = {
            "model": self.model_route.model_id,
            "provider": provider,
            "toolsets": ["hermes-cli"],
            "agent": {"max_turns": 90},
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
            "resource_attributes": self._trace_attributes(
                "hermes", self.model_route
            ),
            "capture_previews": self.capture_content,
            "force_flush_on_session_end": True,
        }
        return yaml.dump(config, default_flow_style=False)

    async def _detect_home(self, environment: BaseEnvironment) -> str:
        result = await self.exec_as_agent(
            environment, command='printf %s "$HOME"', timeout_sec=10
        )
        return (result.stdout or "").strip() or "/root"

    async def _install_hermes_otel(
        self, environment: BaseEnvironment, home: str
    ) -> None:
        staged = stage_hermes_otel_checkout()
        plugin_dir = f"{home}/.hermes/plugins/hermes_otel"

        await self.exec_as_agent(
            environment, command=f"mkdir -p {shlex.quote(plugin_dir)}", timeout_sec=10
        )
        await environment.upload_dir(source_dir=staged, target_dir=plugin_dir)

        # The OTel deps must land in the hermes-agent venv (README install
        # contract). The venv is uv-managed and ships no pip, so prefer the
        # hermes-bundled uv ($HOME/.hermes/bin/uv, verified) and fall back to
        # ensurepip. The venv path comes from the hermes launcher shim.
        install_cmd = (
            "set -e; "
            'export PATH="$HOME/.local/bin:$PATH"; '
            'HB="$(command -v hermes)"; '
            "VENV_PY=\"$(sed -n 's|^exec \"\\(.*\\)/bin/hermes\".*|\\1/bin/python|p' \"$HB\")\"; "
            '[ -x "$VENV_PY" ] || VENV_PY=/usr/local/lib/hermes-agent/venv/bin/python; '
            '[ -x "$VENV_PY" ] || { echo "hermes venv python not found" >&2; exit 1; }; '
            'if [ -x "$HOME/.hermes/bin/uv" ]; then '
            f'  "$HOME/.hermes/bin/uv" pip install --quiet --python "$VENV_PY" -e "{plugin_dir}[yaml]"; '
            "else "
            '  "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true; '
            f'  "$VENV_PY" -m pip install --quiet -e "{plugin_dir}[yaml]"; '
            "fi"
        )
        await self.exec_as_agent(environment, command=install_cmd, timeout_sec=600)

        # Discovery alone leaves the plugin "not enabled" (verified via
        # `hermes plugins list`). Enablement is persisted as a `plugins:`
        # block inside ~/.hermes/config.yaml, so this MUST run after the
        # main config write — a later `cat > config.yaml` wipes it and the
        # plugin silently never loads (root cause of the traceless smoke
        # runs). Capture list output as a trial artifact for verification.
        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH" && '
                "{ hermes plugins enable hermes_otel && "
                "hermes plugins list | grep -A2 hermes_otel; } "
                "2>&1 | tee /logs/agent/hermes-otel-install.txt"
            ),
            timeout_sec=60,
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
        await self._install_hermes_otel(environment, home)

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
            await self.exec_as_agent(
                environment, command=mcp_command, env=env, timeout_sec=10
            )
        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(
                environment, command=skills_command, env=env, timeout_sec=10
            )

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

    Model plane fixes over the stock adapter:

    - Provider ``models`` array must hold the id relative to the provider
      (``zai-org/GLM-5.2``), not the full harbor name.
    - ``--thinking high`` is rejected for custom OpenAI-compatible models
      ("Use one of: off").

    Tracing (all three verified empirically, incl. on a fresh host install):

    - The weave entry (entity/project + allowConversationAccess) is baked
      into the generated openclaw.json *before* ``openclaw plugins install
      weave-openclaw`` runs — the plugin manager validates config against
      the plugin schema, which requires ``entity``.
    - OpenClaw's managed npm overrides force ``@opentelemetry/core@2.8.0``
      but the published weave SDK ships the OTel 1.x trace stack
      (``TracesSamplerValues.AlwaysOn`` is gone from core 2.x), so the
      plugin crashes at load. Fix: after the plugin install, override
      ``weave`` in the plugin project to our OTel-2.x SDK build
      (WEAVE_NODE_SDK_TGZ, from the wandb/weave migration branch) and
      reinstall — the whole tree then resolves a consistent 2.x stack.
    - Plugin services only start in **gateway mode** — ``openclaw agent
      --local`` never initializes the exporter. run() starts a loopback
      gateway, waits for it to become ready, runs the turn against it, and
      tears it down after a flush window.

    Spans land in the Weave *Agents* store (``/agents/otel/v1/traces`` ->
    query via ``POST /agents/spans/query``), not the calls table. The stable
    agent name groups all OpenClaw trials; Fugue attributes identify a trial.
    """

    CLI_FLAGS = [
        CliFlag("openclaw_agent_id", cli="--agent", type="str", default="main"),
        CliFlag("thinking", cli="--thinking", type="str", default="off"),
        CliFlag("timeout", cli="--timeout", type="int"),
    ]

    _GATEWAY_TOKEN = "fugue-gateway"
    _GATEWAY_PORT = 18789
    _GATEWAY_LOG = "/logs/agent/openclaw-gateway.log"
    _GATEWAY_PID_FILE = "/tmp/openclaw-gateway.pid"
    _PLUGIN_PROJECT = "~/.openclaw/npm/projects/weave-openclaw"
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
        super().__init__(*args, model_name=f"openai/{self.model_route.model_id}", **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self._capture_runtime_fingerprint(environment, "pre_install")
        await super().install(environment)

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

    _WEAVE_TGZ_UPLOAD = "/logs/agent/weave-node-sdk.tgz"

    def _install_plugin_command(self) -> str:
        """Install weave-openclaw, then swap in the OTel-2.x weave SDK build.

        The published weave SDK (<=0.16.2) uses the OTel 1.x trace stack,
        which crashes at plugin load against OpenClaw's managed override
        ``@opentelemetry/core@2.8.0``. Overriding ``weave`` to our tgz (built
        from the wandb/weave OTel-2.x branch) and reinstalling resolves a
        consistent 2.x tree. Node_modules and the lockfile are removed first
        so npm actually honors the new override.
        """
        override_js = (
            "const fs = require('node:fs');"
            "const p = process.argv[1] + '/package.json';"
            "const j = JSON.parse(fs.readFileSync(p, 'utf8'));"
            "j.overrides = j.overrides || {};"
            f"j.overrides['weave'] = 'file:{self._WEAVE_TGZ_UPLOAD}';"
            "fs.writeFileSync(p, JSON.stringify(j, null, 2) + '\\n');"
            "console.log('override weave ->', j.overrides['weave']);"
        )
        trace_patch_js = (
            "const fs = require('node:fs');"
            "const weaveRoot = process.argv[1] + '/node_modules/weave/dist/genai/';"
            "const needle = 'this.span = span;';"
            "const repl = \"this.span = span; try { this.span.setAttributes(JSON.parse(process.env.FUGUE_TRACE_ATTRIBUTES_JSON || '{}')); } catch {}\";"
            "for (const name of ['spanBase.js', 'spanBase.mjs']) {"
            " const p = weaveRoot + name; let src = fs.readFileSync(p, 'utf8');"
            " const matches = src.split(needle).length - 1;"
            " if (matches !== 1 && !src.includes('FUGUE_TRACE_ATTRIBUTES_JSON'))"
            "  { console.error('trace attrs patch: emitter pattern missing in ' + name); process.exit(1); }"
            " src = src.replace(needle, repl); fs.writeFileSync(p, src);"
            "}"
            "console.log('patched OpenClaw Fugue trace attributes');"
        )
        return (
            f"{{ openclaw plugins install weave-openclaw@{self._WEAVE_PLUGIN_VERSION} && "
            f"PLUGIN_DIR=$(cd {self._PLUGIN_PROJECT} && pwd) && "
            f"node -e {shlex.quote(override_js)} \"$PLUGIN_DIR\" && "
            'rm -rf "$PLUGIN_DIR/node_modules" "$PLUGIN_DIR/package-lock.json" && '
            'npm install --prefix "$PLUGIN_DIR" --no-audit --no-fund && '
            f"node -e {shlex.quote(trace_patch_js)} \"$PLUGIN_DIR\" && "
            "node -e 'console.log(\"resolved weave:\", "
            "require(process.argv[1] + "
            "\"/node_modules/weave/package.json\").version, "
            "\"core:\", require(process.argv[1] + "
            "\"/node_modules/@opentelemetry/core/package.json\").version)' "
            '"$PLUGIN_DIR"; } '
            "2>&1 | tee /logs/agent/weave-openclaw-install.txt"
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
            f'if [ -f {self._GATEWAY_PID_FILE} ]; then '
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
        # Mirrors the stock run() with three insertions: WANDB_API_KEY in the
        # agent env (the plugin reads it per config apiKey.source=env), the
        # plugin install + OTel pin after the config lands, and a loopback
        # gateway wrapped around the agent turn (plugins only trace there).
        from harbor.agents.installed.openclaw import _nvm22

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
            "FUGUE_WEAVE_CONVERSATION_ID": self.conversation_id,
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

        # Ship the OTel-2.x weave SDK build into the container via the logs
        # mount; _install_plugin_command overrides the plugin's `weave` dep
        # to this file.
        if not WEAVE_NODE_SDK_TGZ.exists():
            raise FileNotFoundError(
                f"weave node SDK tarball not found at {WEAVE_NODE_SDK_TGZ} "
                "(build it from the wandb/weave OTel-2.x branch: "
                "`pnpm build && pnpm pack` in sdks/node, or set "
                "WEAVE_NODE_SDK_TGZ)"
            )
        shutil.copy2(WEAVE_NODE_SDK_TGZ, self.logs_dir / "weave-node-sdk.tgz")

        try:
            (self.logs_dir / "instruction.txt").write_text(instruction)
        except OSError:
            pass

        gateway_started = False
        try:
            await self.exec_as_agent(
                environment,
                command=_nvm22(
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
                command=_nvm22(self._install_plugin_command()),
                env=env,
                timeout_sec=600,
            )

            await self.exec_as_agent(
                environment,
                command=_nvm22(self._register_trial_agent_command()),
                env=env,
                timeout_sec=120,
            )

            skills_command = self._build_register_skills_command()
            if skills_command:
                await self.exec_as_agent(environment, command=skills_command, env=env)

            await self.exec_as_agent(
                environment,
                command=_nvm22(self._start_gateway_command()),
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
                ". ~/.nvm/nvm.sh && nvm use 22 && "
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
        return [openclaw_conversation_id(self.conversation_id), *native_ids]


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
        await self._capture_runtime_fingerprint(environment, "pre_install")
        await super().install(environment)
        # The plugin CLI + its daemon need node >= 18.19. Debian may provide an
        # older executable, so checking only for npm is insufficient.
        await self.exec_as_root(
            environment,
            command=(
                "node -e \"const [a,b]=process.versions.node.split('.').map(Number);"
                "process.exit(a>18||(a===18&&b>=19)?0:1)\" 2>/dev/null || { "
                "apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates curl && "
                "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && "
                "apt-get install -y --no-install-recommends nodejs; }"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                f"npm install -g weave-claude-code@{self._WEAVE_PLUGIN_VERSION} && "
                "weave-claude-code --version"
            ),
            timeout_sec=600,
        )

    async def _install_weave_plugin(self, environment: BaseEnvironment) -> None:
        env = {
            **self._trace_environment("claude-code", self.model_route),
            "CLAUDE_CONFIG_DIR": self._CLAUDE_CONFIG_DIR,
            "WEAVE_PROJECT": _weave_project_slug(),
            "WANDB_API_KEY": _require_trace_key(),
            "IS_SANDBOX": "1",
            "FUGUE_WEAVE_CONVERSATION_ID": self.conversation_id,
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
        marketplace_js = (
            "const fs = require('node:fs');"
            "const p = process.argv[1] + '/.claude-plugin/marketplace.json';"
            "const j = JSON.parse(fs.readFileSync(p, 'utf8'));"
            "j.plugins = (j.plugins || []).map(pl => ({...pl, source: './'}));"
            "fs.writeFileSync(p, JSON.stringify(j, null, 2) + '\\n');"
            "console.log('patched marketplace plugin source -> ./');"
        )
        transcript_js = (
            "const fs = require('node:fs');"
            "const p = process.argv[1] + '/dist/transcriptFile.js';"
            "let src = fs.readFileSync(p, 'utf8');"
            "const needle = 'isPathWithinBase(resolved, os.homedir())';"
            "const repl = '[os.homedir(), process.env.CLAUDE_CONFIG_DIR]"
            ".filter(Boolean).some(b => isPathWithinBase(resolved, b))';"
            "if (!src.includes(needle) && !src.includes('CLAUDE_CONFIG_DIR'))"
            " { console.error('transcript patch: pattern missing'); process.exit(1); }"
            "src = src.split(needle).join(repl);"
            "fs.writeFileSync(p, src);"
            "console.log('patched transcript_path base check');"
        )
        trace_attrs_js = (
            "const fs = require('node:fs');"
            "const root = process.argv[1] + '/dist/';"
            "const spansPath = root + 'genaiSpans.js';"
            "let spans = fs.readFileSync(spansPath, 'utf8');"
            "const spansNeedle = \"entries[`${WEAVE_INTEGRATION_META_PREFIX}${key}`] = { value };\";"
            "const spansRepl = \"entries[(key.startsWith('fugue.') || key.startsWith('weave.eval.')) ? key : `${WEAVE_INTEGRATION_META_PREFIX}${key}`] = { value: String(value) };\";"
            "if (!spans.includes(spansNeedle) && !spans.includes(spansRepl))"
            " { console.error('trace attrs patch: baggage pattern missing'); process.exit(1); }"
            "spans = spans.split(spansNeedle).join(spansRepl);"
            "const processorNeedle = 'if (key.startsWith(WEAVE_INTEGRATION_PREFIX)) {';"
            "const processorRepl = \"if (key.startsWith(WEAVE_INTEGRATION_PREFIX) || key.startsWith('fugue.') || key.startsWith('weave.eval.')) {\";"
            "if (!spans.includes(processorNeedle) && !spans.includes(processorRepl))"
            " { console.error('trace attrs patch: processor pattern missing'); process.exit(1); }"
            "spans = spans.split(processorNeedle).join(processorRepl);"
            "fs.writeFileSync(spansPath, spans);"
            "const daemonPath = root + 'daemon.js';"
            "let daemon = fs.readFileSync(daemonPath, 'utf8');"
            "const daemonNeedle = \"meta: { claude_code_app_version: claudeCodeAppVersion },\";"
            "const daemonRepl = \"meta: { claude_code_app_version: claudeCodeAppVersion, ...JSON.parse(process.env.FUGUE_TRACE_ATTRIBUTES_JSON || '{}') },\";"
            "if (!daemon.includes(daemonNeedle) && !daemon.includes(daemonRepl))"
            " { console.error('trace attrs patch: daemon pattern missing'); process.exit(1); }"
            "daemon = daemon.split(daemonNeedle).join(daemonRepl);"
            "fs.writeFileSync(daemonPath, daemon);"
            "console.log('patched Fugue trace attribute baggage');"
        )
        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH"; '
                'mkdir -p "$CLAUDE_CONFIG_DIR"; '
                '{ NPM_ROOT="$(npm root -g)" && '
                f"node -e {shlex.quote(marketplace_js)} \"$NPM_ROOT/weave-claude-code\" && "
                f"node -e {shlex.quote(transcript_js)} \"$NPM_ROOT/weave-claude-code\" && "
                f"node -e {shlex.quote(trace_attrs_js)} \"$NPM_ROOT/weave-claude-code\" && "
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
                "FUGUE_WEAVE_CONVERSATION_ID": self.conversation_id,
            }
        )
        try:
            await self._install_weave_plugin(environment)
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

    Tracing: weave-codex merges a Stop hook into ``$CODEX_HOME/hooks.json``
    (it honors CODEX_HOME, verified in its constants.js). The hook spawns a
    *detached* collector that reads rollout files from ``$CODEX_HOME`` — so
    cleanup waits for the collector log to go quiet before deleting it.

    Hook trust (verified on codex 0.143.0): the ``bypass_hook_trust = true``
    config key from the weave-codex README does NOT unlock headless runs —
    the untrusted hook is silently skipped. The working mechanism is the
    ``--dangerously-bypass-hook-trust`` CLI flag on ``codex exec``.

    weave-codex hardcodes ``agent_name=codex`` in its spans; install() patches
    its emit.js to honor ``WEAVE_CODEX_AGENT_NAME`` so the stable harness
    identity is explicit. Fugue stores both its deterministic conversation id
    and the native Codex session id for export correlation.
    """

    # Bridged providers may reject OpenAI reasoning params; drop the stock
    # default of `-c model_reasoning_effort=high`.
    CLI_FLAGS = [
        flag for flag in Codex.CLI_FLAGS if flag.kwarg != "reasoning_effort"
    ]
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
        await self._capture_runtime_fingerprint(environment, "pre_install")
        await super().install(environment)
        # weave-codex has no --version flag; `command -v` is the install check.
        # The emit.js patch makes the span agent_name configurable via
        # WEAVE_CODEX_AGENT_NAME (upstream hardcodes 'codex'), which run()
        # sets to Fugue's stable harness identity.
        trace_patch_js = (
            "const fs = require('node:fs');"
            "const p = process.argv[1] + '/dist/spans/emit.js';"
            "let src = fs.readFileSync(p, 'utf8');"
            "const needle = \"const AGENT_NAME = 'codex';\";"
            "const repl = \"const AGENT_NAME = "
            "process.env.WEAVE_CODEX_AGENT_NAME || 'codex';\\n"
            "const FUGUE_TRACE_ATTRIBUTES = (() => { try { return JSON.parse(process.env.FUGUE_TRACE_ATTRIBUTES_JSON || '{}'); } catch { return {}; } })();\";"
            "if (!src.includes(needle) && !src.includes('WEAVE_CODEX_AGENT_NAME'))"
            " { console.error('agent-name patch: pattern missing'); process.exit(1); }"
            "src = src.split(needle).join(repl);"
            "const attrsNeedle = 'const spanAttributes = {';"
            "const attrsRepl = 'const spanAttributes = { ...FUGUE_TRACE_ATTRIBUTES,';"
            "const count = src.split(attrsNeedle).length - 1;"
            "if (count && count !== 3)"
            " { console.error('trace attrs patch: expected 3 span objects, got ' + count); process.exit(1); }"
            "if (!count && !src.includes(attrsRepl))"
            " { console.error('trace attrs patch: span pattern missing'); process.exit(1); }"
            "src = src.split(attrsNeedle).join(attrsRepl);"
            "fs.writeFileSync(p, src);"
            "console.log('patched weave-codex identity and Fugue trace attributes');"
        )
        await self.exec_as_agent(
            environment,
            command=(
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                f"npm install -g weave-codex@{self._WEAVE_PLUGIN_VERSION} && "
                "codex --version && command -v weave-codex && "
                'NPM_ROOT="$(npm root -g)" && '
                f"node -e {shlex.quote(trace_patch_js)} \"$NPM_ROOT/weave-codex\""
            ),
            timeout_sec=600,
        )

    def _build_model_config_toml(self) -> str:
        # Hook trust is handled by --dangerously-bypass-hook-trust on the exec
        # invocation; the README's `bypass_hook_trust` config key is a no-op
        # for headless runs on codex 0.143.0 (verified: hook never fires).
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

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await self._begin_trial("codex", self.model_route, environment)
        instruction = self.render_instruction(instruction)
        escaped_instruction = shlex.quote(instruction)

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""
        feature_flags = (
            "--enable unified_exec "
            if self.model_route.provider == "openai"
            else "".join(
                f"--disable {feature} "
                for feature in self._BRIDGED_DISABLED_FEATURES
            )
        )

        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        weave_project = _weave_project_slug()
        env: dict[str, str] = {
            **provider_client_env(self.model_route, os.environ),
            **self._trace_environment("codex", self.model_route),
            "CODEX_HOME": remote_codex_home,
            # Codex reads the configured provider's key from OPENAI_API_KEY
            # regardless of whether it is native OpenAI or the local bridge.
            "OPENAI_API_KEY": _responses_key(self.model_route),
            # The Stop-hook collector inherits codex's env; give it Weave
            # credentials directly without writing the key into a command.
            "WANDB_API_KEY": _require_trace_key(),
            "WEAVE_PROJECT": weave_project,
            # Consumed by the emit.js patch from install(); keeps all trials
            # grouped under the stable Codex agent.
            "WEAVE_CODEX_AGENT_NAME": stable_agent_name("codex"),
            "FUGUE_WEAVE_CONVERSATION_ID": self.conversation_id,
        }

        config_toml = self._build_model_config_toml()
        settings_json = json.dumps(
            {
                "weave_project": weave_project,
                "capture_content": self.capture_content,
            }
        )
        setup_command = (
            f'mkdir -p "$CODEX_HOME" {shlex.quote(EnvironmentPaths.agent_dir.as_posix())}\n'
            f'cat >>"$CODEX_HOME/config.toml" <<\'TOML\'\n{config_toml}TOML\n'
            "mkdir -p ~/.weave-codex\n"
            f"cat > ~/.weave-codex/settings.json <<'JSON'\n{settings_json}\nJSON\n"
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi\n"
            "{ weave-codex install && weave-codex status; } "
            "2>&1 | tee /logs/agent/weave-codex-install.txt\n"
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
                "print(json.dumps({'expected':sorted(expected),"
                "'registered':sorted(actual),'missing':missing}));"
                "sys.exit(bool(missing))"
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

        codex_output = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        codex_sessions = (EnvironmentPaths.agent_dir / "sessions").as_posix()

        try:
            await self.exec_as_agent(
                environment,
                command=(
                    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                    "codex exec "
                    "--dangerously-bypass-approvals-and-sandbox "
                    "--dangerously-bypass-hook-trust "
                    "--skip-git-repo-check "
                    "--json "
                    f"{feature_flags}"
                    f"{cli_flags_arg}"
                    "-- "
                    f"{escaped_instruction} "
                    f"2>&1 </dev/null | tee {codex_output}"
                ),
                env=env,
            )
        finally:
            try:
                # Wait for the detached weave-codex collector to finish
                # exporting (log quiet for 2 consecutive seconds, max ~20s),
                # then snapshot its log for debugging.
                await self.exec_as_agent(
                    environment,
                    command=(
                        "LOG=~/.weave-codex/logs/collector.log; "
                        "for i in $(seq 1 10); do "
                        '  s1=$(stat -c %s "$LOG" 2>/dev/null || echo 0); sleep 2; '
                        '  s2=$(stat -c %s "$LOG" 2>/dev/null || echo 0); '
                        '  [ "$s1" = "$s2" ] && [ "$i" -gt 2 ] && break; '
                        "done; "
                        'cp "$LOG" /logs/agent/weave-codex-collector.log 2>/dev/null || true'
                    ),
                    env=env,
                    timeout_sec=60,
                )
            except Exception:
                pass
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
            "mkdir -p \"$LETTA_LOCAL_BACKEND_DIR\" /logs/agent; "
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
