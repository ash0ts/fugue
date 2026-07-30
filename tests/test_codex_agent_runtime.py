from __future__ import annotations

import asyncio
import inspect
import json
import stat
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor.agents.installed.codex")

from fugue.agents.model_plane import FugueCodex
from fugue.bench.export import export_rows
from fugue.model_plane import resolve_model_route


class _FakeEnvironment:
    default_user = "agent"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.uploads: list[dict[str, object]] = []

    async def exec(self, command: str, **kwargs):
        self.calls.append({"command": command, **kwargs})
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source_path, target_path) -> None:
        self.uploads.append(
            {
                "target": str(target_path),
                "content": source_path.read_text(),
                "mode": stat.S_IMODE(source_path.stat().st_mode),
            }
        )


def test_runtime_evidence_environment_reaches_meta_and_export(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_runtime = {
        "image": "fugue-agent-codex-amd64:locked",
        "image_id": "sha256:" + "a" * 64,
    }
    task_runtime = {
        "image": "fugue-task-task-a-amd64:locked",
        "image_id": "sha256:" + "b" * 64,
    }
    sandbox_attestation = {
        "schema_version": 1,
        "source": "rendered_harbor_config",
        "attestation_digest": "c" * 64,
    }
    for name, value in {
        "FUGUE_AGENT_RUNTIME": agent_runtime,
        "FUGUE_TASK_RUNTIME": task_runtime,
        "FUGUE_SANDBOX_ATTESTATION": sandbox_attestation,
    }.items():
        monkeypatch.setenv(name, json.dumps(value, sort_keys=True))
    for name, value in {
        "FUGUE_RUN_ID": "run-a",
        "FUGUE_WORKLOAD_ID": "coding",
        "FUGUE_TASK_NAME": "task-a",
        "FUGUE_HARNESS": "codex",
        "FUGUE_CONTEXT_SYSTEM_ID": "none",
        "FUGUE_VARIANT_ID": "baseline",
        "FUGUE_TRIAL_INDEX": "1",
        "FUGUE_APPLICABLE": "true",
        "FUGUE_SKIP_REASON": "",
        "FUGUE_WEAVE_PROJECT": "entity/project",
    }.items():
        monkeypatch.setenv(name, value)

    trial_dir = tmp_path / "jobs" / "runtime-evidence" / "trial-1"
    agent = object.__new__(FugueCodex)
    agent.logs_dir = trial_dir / "agent"
    agent._meta_begin("codex", resolve_model_route("openai/gpt-5", {}))

    meta = json.loads((agent.logs_dir / "fugue-meta.json").read_text())
    assert meta["agent_runtime"] == agent_runtime
    assert meta["task_runtime"] == task_runtime
    assert meta["sandbox_attestation"] == sandbox_attestation
    assert meta["applicable"] is True
    assert meta["skip_reason"] is None

    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "fixture/task-a",
                "trial_name": "trial-1",
                "config": {"agent": {"model_name": "openai/gpt-5"}},
                "agent_info": {"name": "codex"},
                "agent_result": {},
                "verifier_result": {"rewards": {"reward": 1.0}},
                "exception_info": None,
                "started_at": "2026-07-29T12:00:00Z",
                "finished_at": "2026-07-29T12:00:01Z",
            }
        )
    )

    [row] = export_rows([trial_dir])
    assert row["agent_runtime"] == agent_runtime
    assert row["task_runtime"] == task_runtime
    assert row["sandbox_attestation"] == sandbox_attestation
    assert row["applicable"] is True
    assert row["skip_reason"] is None
    assert row["agent_runtime_image_id"] == agent_runtime["image_id"]
    assert row["task_runtime_image_id"] == task_runtime["image_id"]
    assert (
        row["sandbox_attestation_digest"]
        == sandbox_attestation["attestation_digest"]
    )
    assert "privacy_scan_complete" not in row
    assert "sandbox_cleanup_verified" not in row


def test_codex_runtime_uses_cell_home_and_structured_mcp_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(FugueCodex)
    agent.mcp_servers = [
        SimpleNamespace(
            name="repo",
            transport="stdio",
            command="python",
            args=["-m", "server"],
        )
    ]
    monkeypatch.setenv("FUGUE_RUN_ID", "run-a")
    monkeypatch.setenv("FUGUE_TASK_NAME", "task-a")
    monkeypatch.setenv("FUGUE_HARNESS", "codex")
    monkeypatch.setenv("FUGUE_CONTEXT_SYSTEM_ID", "gitnexus")
    monkeypatch.setenv("FUGUE_CONTEXT_CONFIG_HASH", "config-a")
    monkeypatch.setenv("FUGUE_VARIANT_ID", "gitnexus")
    monkeypatch.setenv("FUGUE_TRIAL_INDEX", "1")

    home = agent._codex_home()
    command = agent._build_register_mcp_servers_command()

    assert home.as_posix().startswith("/tmp/fugue-codex/")
    assert home.as_posix() != "/tmp/codex-home"
    assert command is not None
    assert 'command = "python"' in command
    assert 'args = ["-m","server"]' in command

    registration = agent._context_registration(
        {"status": "registered", "transport": "native_mcp"}
    )
    assert registration["context_system_id"] == "gitnexus"
    assert registration["registration_digest"].startswith("sha256:")


def test_static_context_registration_has_a_behavioral_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FUGUE_CONTEXT_SYSTEM_ID", "agentsmd")
    monkeypatch.setenv("FUGUE_CONTEXT_DELIVERY", "portable")
    monkeypatch.setenv("FUGUE_CONTEXT_CONFIG_HASH", "config-a")
    agent = object.__new__(FugueCodex)
    agent.mcp_servers = []

    registration = agent._context_registration(
        {"status": "static", "delivery": "portable", "servers": 0}
    )

    assert registration["registration_digest"].startswith("sha256:")


def test_codex_shells_keep_runtime_context_but_exclude_secrets(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "trace-key")
    monkeypatch.setenv("WANDB_ENTITY", "wandb")
    monkeypatch.setenv("WANDB_PROJECT", "fugue-test")
    agent = FugueCodex(logs_dir=tmp_path, model_name="wandb/test-model")

    config = agent._build_model_config_toml()

    assert "[shell_environment_policy]" in config
    assert 'inherit = "all"' in config
    assert '"*KEY*"' in config
    assert '"*TOKEN*"' in config
    assert '"*AUTH*"' in config
    assert '"LD_LIBRARY_PATH"' in config


def test_codex_consumes_staged_secrets_before_starting_agent() -> None:
    source = inspect.getsource(FugueCodex.run)

    assert source.index("rm -rf {_CONTAINER_SECRET_ROOT.as_posix()}") < source.index(
        '"weave-codex run -- codex exec "'
    )
    assert "self._fugue_secret_files.clear()" in source


def test_codex_exec_stages_secrets_outside_process_arguments(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "constructor-trace-key")
    monkeypatch.setenv("WANDB_ENTITY", "wandb")
    monkeypatch.setenv("WANDB_PROJECT", "fugue-test")
    agent = FugueCodex(
        logs_dir=tmp_path,
        model_name="wandb/test-model",
        extra_env={"CUSTOM_TOKEN": "scoped-secret", "SAFE_SETTING": "enabled"},
    )
    environment = _FakeEnvironment()

    asyncio.run(
        agent.exec_as_agent(
            environment,
            command="printenv WANDB_API_KEY",
            env={"WANDB_API_KEY": "per-call-secret", "PUBLIC_VALUE": "visible"},
        )
    )

    assert agent.extra_env == {"SAFE_SETTING": "enabled"}
    assert len(environment.uploads) == 1
    upload = environment.uploads[0]
    assert upload["mode"] == 0o644
    assert str(upload["target"]).endswith(".env.upload")
    assert "WANDB_API_KEY=per-call-secret" in upload["content"]
    assert "CUSTOM_TOKEN=scoped-secret" in upload["content"]
    serialized_calls = repr(environment.calls)
    assert "per-call-secret" not in serialized_calls
    assert "scoped-secret" not in serialized_calls
    assert "install -m 0600" in serialized_calls
    assert "rm -f" in serialized_calls
    invocation = environment.calls[-1]
    assert invocation["env"] == {"PUBLIC_VALUE": "visible"}
    assert ". /tmp/fugue-agent-secrets/" in str(invocation["command"])
