from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.ai import AssetDraft, ExperimentDraft
from fugue.bench.cli import main
from fugue.bench.operator import (
    OperatorService,
    PreviewSummary,
    RunSummary,
    SetupPreparation,
    load_env,
)
from fugue.bench.services import GRAPHITI_SERVICE, ManagedServiceStatus


def test_bare_fugue_is_noninteractive_when_not_attached_to_tty(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    for command in ("plan", "run", "runs", "analyze", "setup", "tui"):
        assert command in output


def test_public_command_surface_is_intentionally_small() -> None:
    from fugue.bench import cli

    subparsers = next(
        action
        for action in cli._parser()._actions
        if isinstance(action, cli.argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {
        "plan",
        "run",
        "runs",
        "analyze",
        "setup",
        "tui",
        "research",
    }
    assert "--env-file" in subparsers.choices["run"].format_help()
    assert "--env-file" in subparsers.choices["setup"].format_help()


@pytest.mark.parametrize(
    ("flag", "method", "ready", "state"),
    (
        ("--start-services", "start_services", True, "healthy"),
        ("--service-status", "service_status", True, "healthy"),
        ("--stop-services", "stop_services", False, "not_created"),
    ),
)
def test_setup_exposes_explicit_managed_service_lifecycle(
    flag: str,
    method: str,
    ready: bool,
    state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[object] = []
    status = ManagedServiceStatus(
        GRAPHITI_SERVICE.id,
        state,  # type: ignore[arg-type]
        ready,
        "lifecycle result",
        GRAPHITI_SERVICE.container_name,
        GRAPHITI_SERVICE.image,
        GRAPHITI_SERVICE.host_uri,
    )

    def lifecycle(self, request):
        captured.append(request)
        return (status,)

    monkeypatch.setattr(OperatorService, method, lifecycle)

    assert (
        main(
            [
                "setup",
                flag,
                "--systems",
                "graphiti",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )
    assert captured[0].systems == ("graphiti",)
    assert GRAPHITI_SERVICE.id in capsys.readouterr().out


def test_setup_service_actions_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["setup", "--start-services", "--stop-services"])


def test_setup_prepare_accepts_exact_plan_selectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[object] = []

    def prepare(self, request, **_kwargs):
        captured.append(request)
        return SetupPreparation(context=(), agent_runtimes=())

    monkeypatch.setattr(OperatorService, "prepare", prepare)
    assert (
        main(
            [
                "setup",
                "--prepare",
                "--variants",
                "none,gitnexus-vector",
                "--harnesses",
                "codex",
                "--n-tasks",
                "1",
                "--n-attempts",
                "2",
                "--n-concurrent",
                "4",
                "--repo-root",
                tmp_path.as_posix(),
            ]
        )
        == 0
    )
    request = captured[0]
    assert request.variants == ("none", "gitnexus-vector")
    assert request.harnesses == ("codex",)
    assert (request.n_tasks, request.n_attempts, request.n_concurrent) == (1, 2, 4)


def test_runs_packages_one_explicit_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    captured = {}

    def package(self, run_id, candidate_id, **kwargs):
        captured.update({"run_id": run_id, "candidate_id": candidate_id, **kwargs})
        return SimpleNamespace(
            candidate_id=candidate_id,
            image=kwargs["image"],
            deployment_id="deployment-1",
            path=tmp_path / ".fugue/runtime/deployments/deployment-1",
        )

    monkeypatch.setattr(OperatorService, "package_candidate", package)

    assert (
        main(
            [
                "runs",
                "run-1",
                "package",
                "candidate-1",
                "--workspace",
                tmp_path.as_posix(),
                "--image",
                "example/fugue:test",
                "--yes",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )
    assert captured == {
        "run_id": "run-1",
        "candidate_id": "candidate-1",
        "workspace": tmp_path,
        "image": "example/fugue:test",
        "platform": "linux/amd64",
        "allow_failed": False,
    }
    assert "deployment-1" in capsys.readouterr().out


def test_runs_status_is_observational_across_supervisor_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: list[bool] = []

    def run_summary(self, run_id: str, *, recover: bool = True) -> RunSummary:
        del self
        observed.append(recover)
        return RunSummary(
            run_id=run_id,
            run_name="Foreign worker",
            experiment_id="demo",
            status="running",
            created_at=None,
            cells=(),
            passed=0,
            failed=0,
            cancelled=0,
            interrupted=0,
            pending=1,
            not_applicable=0,
            candidates=(),
            log_path=tmp_path / "combined.log",
        )

    monkeypatch.setattr(OperatorService, "run_summary", run_summary)

    assert (
        main(
            [
                "runs",
                "run-foreign",
                "--json",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )
    assert observed == [False]
    assert '"status": "running"' in capsys.readouterr().out


def test_rich_command_center_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    from fugue.bench import cli

    parser = cli._parser()
    terminal = SimpleNamespace(isatty=lambda: True)
    monkeypatch.setattr(cli.sys, "stdin", terminal)
    monkeypatch.setattr(cli.sys, "stdout", terminal)
    monkeypatch.setattr(cli.CONSOLE, "clear", lambda: None)
    monkeypatch.setattr(cli, "_print_home", lambda service: None)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: "exit")

    assert cli._command_center(parser) == 0


@pytest.mark.parametrize(
    "command",
    (
        "render",
        "export",
        "preflight",
        "bridge",
        "status",
        "compose",
        "analyses",
        "catalog",
        "prompts",
        "skills",
        "experiments",
        "context",
    ),
)
def test_removed_public_commands_are_rejected(command: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        main([command])


def test_run_preview_is_side_effect_free(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "pilot.yaml"
    manifest.write_text(
        """
dataset:
  ref: swe-bench/swe-bench-verified
harnesses:
  - name: codex
    agent: fugue.agents:FugueCodex
tasks:
  - id: astropy__astropy-12907
"""
    )

    assert (
        main(
            [
                "run",
                "--manifest",
                manifest.as_posix(),
                "--model",
                "openai/gpt-5",
                "--run-name",
                "unit-exp",
                "--tags",
                "nightly,cli",
                "--preview",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "harbor run --config" in out
    assert not (tmp_path / ".fugue").exists()


def test_shell_environment_wins_over_blank_dotenv(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=\nWANDB_API_KEY=dotenv-value\n")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-value")
    monkeypatch.setenv("WANDB_API_KEY", "shell-trace")

    env = load_env(env_file)

    assert env["OPENAI_API_KEY"] == "shell-value"
    assert env["WANDB_API_KEY"] == "shell-trace"


def test_repo_memory_smoke_preview_uses_per_workload_limits(
    tmp_path: Path, capsys
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert (
        main(
            [
                "run",
                "repo-memory-impact",
                "--preset",
                "smoke",
                "--run-name",
                "smoke-preview",
                "--preview",
                "--repo-root",
                repo_root.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "--limit 3" in output
    assert "--limit 1" in output


def test_plan_run_requires_generated_assets_to_be_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_operator import make_operator_repo

    service = make_operator_repo(tmp_path)
    draft = _draft(
        service,
        assets=(AssetDraft("prompt", "new-prompt", "New prompt", "# New\n"),),
    )

    async def compose(*args, **kwargs):
        return draft

    monkeypatch.setattr("fugue.bench.ai.ExperimentComposer.compose", compose)
    with pytest.raises(ValueError, match="save the experiment"):
        main(
            [
                "plan",
                "use a new prompt",
                "--from",
                "demo",
                "--run",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )


def test_plan_save_and_run_launches_saved_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_operator import make_operator_repo

    service = make_operator_repo(tmp_path)
    draft = _draft(service)
    launched = []

    async def compose(*args, **kwargs):
        return draft

    def save(self, draft, *, experiment_id, replace_assets=False):
        return replace(draft.experiment, id=experiment_id)

    def launch(self, request, *, experiment=None):
        launched.append((request.experiment_id, experiment))
        return type("Run", (), {"run_id": "run-saved"})()

    monkeypatch.setattr("fugue.bench.ai.ExperimentComposer.compose", compose)
    monkeypatch.setattr("fugue.bench.ai.ExperimentComposer.save", save)
    monkeypatch.setattr(OperatorService, "launch", launch)

    assert (
        main(
            [
                "plan",
                "save this",
                "--from",
                "demo",
                "--save",
                "saved-demo",
                "--run",
                "--json",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )
    assert launched == [("saved-demo", None)]


def test_run_uses_one_durable_launch_path_and_waits_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_operator import make_operator_repo

    make_operator_repo(tmp_path)
    launched = []
    waited = []

    def launch(self, request, *, experiment=None):
        launched.append((request.experiment_id, experiment))
        return SimpleNamespace(
            run_id="run-managed",
            run_name="Demo",
            log_path=tmp_path / "combined.log",
        )

    monkeypatch.setattr(OperatorService, "launch", launch)
    monkeypatch.setattr(
        "fugue.bench.cli._wait_for_run",
        lambda service, run_id: waited.append(run_id) or 0,
    )

    assert (
        main(
            [
                "run",
                "demo",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )
    assert launched == [("demo", None)]
    assert waited == ["run-managed"]

    assert (
        main(
            [
                "run",
                "demo",
                "--detach",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )
    assert len(launched) == 2
    assert waited == ["run-managed"]


def _draft(
    service: OperatorService,
    *,
    assets: tuple[AssetDraft, ...] = (),
) -> ExperimentDraft:
    return ExperimentDraft(
        experiment=service.experiment("demo"),
        assets=assets,
        rationale="Controlled demo",
        assumptions=(),
        warnings=(),
        diff="",
        preview=PreviewSummary(
            cells=1,
            applicable_cells=1,
            estimated_trials=1,
            harnesses=("codex",),
            variants=("baseline",),
            systems=("none",),
            workloads=("harbor",),
            commands=(),
        ),
        model="openai/gpt-5",
        provider="openai",
        session_id="session",
        input_tokens=1,
        output_tokens=1,
    )
