"""Harbor agent subclasses: provider-neutral model plane + Weave tracing.

Fugue always traces to W&B Weave, while model calls can bill through W&B
Inference, OpenAI, or Anthropic. The shared ``ModelRoute`` determines whether
each harness can talk to the provider natively or should use the local LiteLLM
bridge.

Tracing plane — every harness ships its Weave plugin inside the container:

- Hermes     -> local hermes-otel checkout (HERMES_OTEL_CHECKOUT, default
                ~/Documents/GitHub/hermes-otel) uploaded + pip-installed into
                the hermes venv; ``type: weave`` backend with run-key
                resource attributes.
- OpenClaw   -> ``openclaw plugins install weave-openclaw``; config entry
                with entity/project + ``hooks.allowConversationAccess``.
- Claude Code-> ``npm i -g weave-claude-code`` + non-interactive
                ``--source=local`` install against the run's CLAUDE_CONFIG_DIR.
- Codex      -> ``npm i -g weave-codex`` + Stop hook merged into
                ``$CODEX_HOME/hooks.json`` (+ ``bypass_hook_trust``).

Every trial also writes ``/logs/agent/fugue-meta.json`` (host side)
with the run key, harness, model, condition, timestamps, and harness session
ids so Weave traces can be joined back to Harbor trials.

All four accept one canonical model string:
``wandb/...``, ``openai/...``, or ``anthropic/...``.
"""

import copy
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

from harbor.agents.installed.base import CliFlag
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.hermes import Hermes
from harbor.agents.installed.openclaw import OpenClaw
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from fugue.model_plane import (
    BRIDGE_BASE_URL_CONTAINER,
    ModelRoute,
    bridge_master_key,
    resolve_model_route,
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
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "").strip()
    slug = os.environ.get("WEAVE_PROJECT", "").strip()
    if slug and "/" in slug:
        entity, project = slug.split("/", 1)
    if not entity or not project:
        raise ValueError(
            "WANDB_ENTITY/WANDB_PROJECT (or WEAVE_PROJECT=entity/project) must "
            "be set for Weave tracing. Source the repo .env."
        )
    return entity, project


def _weave_project_slug() -> str:
    entity, project = _weave_entity_project()
    return f"{entity}/{project}"


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
    _STAGED_HERMES_OTEL = staged
    return staged


class _TrialMetaMixin:
    """Writes /logs/agent/fugue-meta.json (host side) per trial.

    The run key is the Harbor trial directory name (e.g.
    ``bridge-check__hmXLrEo``); Weave traces are joined back to trials via
    this file (plus harness session ids extracted from agent output).
    """

    logs_dir: Path  # provided by BaseAgent

    @property
    def condition(self) -> str:
        return os.environ.get("FUGUE_CONDITION", "none")

    @property
    def run_key(self) -> str:
        # logs_dir is <trial_dir>/agent
        return self.logs_dir.parent.name

    @property
    def job_name(self) -> str:
        return self.logs_dir.parent.parent.name

    def _meta_path(self) -> Path:
        return self.logs_dir / "fugue-meta.json"

    async def _begin_trial(
        self, harness: str, route: ModelRoute, environment: BaseEnvironment
    ) -> None:
        self._memory_artifact_meta = await self._inject_memory_artifact(environment)
        self._meta_begin(harness, route)

    def _meta_begin(self, harness: str, route: ModelRoute) -> None:
        entity, project = _weave_entity_project()
        meta = {
            "run_key": self.run_key,
            "job_name": self.job_name,
            "harness": harness,
            "model_provider": route.provider,
            "model": route.display_model,
            "condition": self.condition,
            "weave_entity": entity,
            "weave_project": project,
            "trace_project": f"{entity}/{project}",
            "started_at": datetime.now(UTC).isoformat(),
        }
        if getattr(self, "_memory_artifact_meta", None):
            meta["memory_artifact"] = self._memory_artifact_meta
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path().write_text(json.dumps(meta, indent=2) + "\n")

    def _meta_end(self) -> None:
        try:
            meta = json.loads(self._meta_path().read_text())
        except Exception:
            meta = {}
        meta["ended_at"] = datetime.now(UTC).isoformat()
        try:
            meta["session_ids"] = self._extract_session_ids()
        except Exception:
            meta["session_ids"] = []
        self._meta_path().write_text(json.dumps(meta, indent=2) + "\n")

    def _extract_session_ids(self) -> list[str]:
        return []

    def _task_artifact_name(self) -> str:
        return os.environ.get("FUGUE_TASK_NAME") or self.run_key.rsplit("__", 1)[0]

    async def _container_repo_root(self, environment: BaseEnvironment) -> str:
        configured = os.environ.get("FUGUE_CONTAINER_REPO_ROOT", "").strip()
        if configured:
            return configured
        result = await self.exec_as_agent(
            environment, command='printf %s "$PWD"', timeout_sec=10
        )
        return (result.stdout or "").strip() or "/app"

    async def _inject_memory_artifact(
        self, environment: BaseEnvironment
    ) -> dict[str, Any] | None:
        condition = self.condition
        memory_root = os.environ.get("FUGUE_MEMORY_DIR", "").strip()
        if not memory_root or condition == "none":
            return None

        task_name = self._task_artifact_name()
        artifact_dir = Path(memory_root) / condition / task_name
        if not artifact_dir.is_dir():
            raise FileNotFoundError(
                f"memory artifact not found for {condition}/{task_name}: "
                f"{artifact_dir}"
            )

        repo_root = await self._container_repo_root(environment)
        await environment.upload_dir(source_dir=artifact_dir, target_dir=repo_root)
        files = [
            p.relative_to(artifact_dir).as_posix()
            for p in sorted(artifact_dir.rglob("*"))
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
            "condition": condition,
            "task_name": task_name,
            "source_dir": artifact_dir.as_posix(),
            "container_root": repo_root,
            "sha256": _hash_dir(artifact_dir),
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

    @staticmethod
    @override
    def name() -> str:
        return "fugue-hermes"

    def __init__(self, *args, model_name: str | None = None, **kwargs):
        self.model_route = resolve_model_route(model_name)
        _require_model_key(self.model_route)
        _require_trace_key()
        _weave_entity_project()  # fail fast before containers spin up
        super().__init__(*args, model_name=self.model_route.model_id, **kwargs)

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
                {"type": "weave", "entity": entity, "project": project},
            ],
            "resource_attributes": {
                "fugue.run_key": self.run_key,
                "fugue.harness": "hermes",
                "fugue.condition": self.condition,
                "fugue.model": self.model_route.display_model,
                "fugue.model_provider": self.model_route.provider,
            },
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
        env: dict[str, str] = {
            "TERMINAL_ENV": "local",
            "WANDB_API_KEY": _require_trace_key(),
            "WANDB_ENTITY": entity,
            "WANDB_PROJECT": project,
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
            self._meta_end()

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
    query via ``POST /agents/spans/query``), not the calls table;
    ``agent_name`` carries the run key.
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
        super().__init__(*args, model_name=f"openai/{self.model_route.model_id}", **kwargs)

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
                "agentName": self.run_key,
                "agentDescription": (
                    f"fugue {self.condition} / {self.model_route.display_model}"
                ),
                "apiKey": {
                    "source": "env",
                    "provider": "default",
                    "id": "WANDB_API_KEY",
                },
                "captureContent": True,
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
        fix_js = (
            "const fs = require('node:fs');"
            "const p = process.argv[1] + '/package.json';"
            "const j = JSON.parse(fs.readFileSync(p, 'utf8'));"
            "j.overrides = j.overrides || {};"
            f"j.overrides['weave'] = 'file:{self._WEAVE_TGZ_UPLOAD}';"
            "fs.writeFileSync(p, JSON.stringify(j, null, 2) + '\\n');"
            "console.log('override weave ->', j.overrides['weave']);"
        )
        return (
            "{ openclaw plugins install weave-openclaw && "
            f"PLUGIN_DIR=$(cd {self._PLUGIN_PROJECT} && pwd) && "
            f"node -e {shlex.quote(fix_js)} \"$PLUGIN_DIR\" && "
            'rm -rf "$PLUGIN_DIR/node_modules" "$PLUGIN_DIR/package-lock.json" && '
            'npm install --prefix "$PLUGIN_DIR" --no-audit --no-fund && '
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
            "WANDB_API_KEY": _require_trace_key(),
            "OPENCLAW_GATEWAY_TOKEN": self._GATEWAY_TOKEN,
            "OPENCLAW_GATEWAY_PORT": str(self._GATEWAY_PORT),
            "OPENAI_API_KEY": _chat_key(self.model_route),
            "OPENAI_BASE_URL": _chat_base_url(self.model_route),
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
                command=_nvm22("openclaw setup --workspace ."),
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
            self._meta_end()

    @override
    def _extract_session_ids(self) -> list[str]:
        return self._regex_ids(
            self.logs_dir / "openclaw.txt", r'"sessionId"\s*:\s*"([^"]+)"'
        )


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
        super().__init__(*args, model_name=self.model_route.model_id, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        # The plugin CLI + its daemon need node/npm (engines >= 18.19), which
        # the claude bootstrap installer does not provide on debian.
        await self.exec_as_root(
            environment,
            command=(
                "command -v npm >/dev/null 2>&1 || { "
                "apt-get update && apt-get install -y --no-install-recommends "
                "nodejs npm; }"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command="npm install -g weave-claude-code && weave-claude-code --version",
            timeout_sec=600,
        )

    async def _install_weave_plugin(self, environment: BaseEnvironment) -> None:
        env = {
            "CLAUDE_CONFIG_DIR": self._CLAUDE_CONFIG_DIR,
            "WEAVE_PROJECT": _weave_project_slug(),
            "WANDB_API_KEY": _require_trace_key(),
            "IS_SANDBOX": "1",
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
        #    daemon resolves env > settings.json. agent_name is set to the
        #    run key so Agents-store spans join back to Harbor trials.
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
        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH"; '
                'mkdir -p "$CLAUDE_CONFIG_DIR"; '
                '{ NPM_ROOT="$(npm root -g)" && '
                f"node -e {shlex.quote(marketplace_js)} \"$NPM_ROOT/weave-claude-code\" && "
                f"node -e {shlex.quote(transcript_js)} \"$NPM_ROOT/weave-claude-code\" && "
                "weave-claude-code install --non-interactive --source=local && "
                'weave-claude-code config set weave_project "$WEAVE_PROJECT" && '
                'weave-claude-code config set wandb_api_key "$WANDB_API_KEY" && '
                f'weave-claude-code config set agent_name {shlex.quote(self.run_key)} && '
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
            self._meta_end()

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
    its emit.js to honor ``WEAVE_CODEX_AGENT_NAME`` (set per-run to the run
    key) so Agents-store spans join back to Harbor trials like the other
    harnesses. conversation_id additionally carries the codex session id.
    """

    # Bridged providers may reject OpenAI reasoning params; drop the stock
    # default of `-c model_reasoning_effort=high`.
    CLI_FLAGS = [
        flag for flag in Codex.CLI_FLAGS if flag.kwarg != "reasoning_effort"
    ]

    @staticmethod
    @override
    def name() -> str:
        return "fugue-codex"

    def __init__(self, *args, model_name: str | None = None, **kwargs):
        self.model_route = resolve_model_route(model_name)
        _require_model_key(self.model_route)
        _require_trace_key()
        _weave_entity_project()
        super().__init__(*args, model_name=self.model_route.model_id, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        # weave-codex has no --version flag; `command -v` is the install check.
        # The emit.js patch makes the span agent_name configurable via
        # WEAVE_CODEX_AGENT_NAME (upstream hardcodes 'codex'), which run()
        # sets to the Harbor run key for trial<->trace joins.
        agent_name_js = (
            "const fs = require('node:fs');"
            "const p = process.argv[1] + '/dist/spans/emit.js';"
            "let src = fs.readFileSync(p, 'utf8');"
            "const needle = \"const AGENT_NAME = 'codex';\";"
            "const repl = \"const AGENT_NAME = "
            "process.env.WEAVE_CODEX_AGENT_NAME || 'codex';\";"
            "if (!src.includes(needle) && !src.includes('WEAVE_CODEX_AGENT_NAME'))"
            " { console.error('agent-name patch: pattern missing'); process.exit(1); }"
            "src = src.split(needle).join(repl);"
            "fs.writeFileSync(p, src);"
            "console.log('patched weave-codex agent name env override');"
        )
        await self.exec_as_agent(
            environment,
            command=(
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                "npm install -g weave-codex && command -v weave-codex && "
                'NPM_ROOT="$(npm root -g)" && '
                f"node -e {shlex.quote(agent_name_js)} \"$NPM_ROOT/weave-codex\""
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

        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        weave_project = _weave_project_slug()
        env: dict[str, str] = {
            "CODEX_HOME": remote_codex_home,
            # Codex reads the configured provider's key from OPENAI_API_KEY
            # regardless of whether it is native OpenAI or the local bridge.
            "OPENAI_API_KEY": _responses_key(self.model_route),
            # The Stop-hook collector inherits codex's env; give it Weave
            # credentials directly (settings.json below is the fallback).
            "WANDB_API_KEY": _require_trace_key(),
            "WEAVE_PROJECT": weave_project,
            # Consumed by the emit.js patch from install(); joins spans to
            # this Harbor trial.
            "WEAVE_CODEX_AGENT_NAME": self.run_key,
        }

        config_toml = self._build_model_config_toml()
        settings_json = json.dumps(
            {
                "wandb_api_key": _require_trace_key(),
                "weave_project": weave_project,
                "capture_content": True,
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

        await self.exec_as_agent(
            environment, command=setup_command, env=env, timeout_sec=600
        )

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
                    "--enable unified_exec "
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
            self._meta_end()

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


# copy.deepcopy is used by the stock OpenClaw config builder; keep the import
# referenced so linters don't flag it after subclass edits.
_ = copy
