from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fugue.bench.ai import AssetDraft, ExperimentDraft
from fugue.bench.cli import main
from fugue.bench.comparison import scaffold_comparison
from fugue.bench.operator import (
    OperatorService,
    PreviewSummary,
    RunSummary,
    SetupPreparation,
    load_env,
)
from fugue.bench.services import GRAPHITI_SERVICE, ManagedServiceStatus
from fugue.bench.templates import scaffold_standalone_template


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
        "doctor",
        "init",
        "check",
        "compare",
        "approve",
        "result",
        "publish",
        "study",
        "demo",
        "sandbox",
        "mcp",
        "skills",
        "provider",
        "taskset",
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
    compare_help = " ".join(subparsers.choices["compare"].format_help().split())
    assert "local mode this flag does not trigger hosted evidence hydration" in (
        compare_help
    )
    result_help = subparsers.choices["result"].format_help()
    assert "--authorize-followup" in result_help
    assert "--signoff-by" in result_help
    publish_actions = next(
        action
        for action in subparsers.choices["publish"]._actions
        if isinstance(action, cli.argparse._SubParsersAction)
    )
    assert "weave" in publish_actions.choices
    publish_weave_help = " ".join(
        publish_actions.choices["weave"].format_help().split()
    )
    assert "--project" in publish_weave_help
    assert "Raw local transcript and tool-event artifact files remain local" in (
        publish_weave_help
    )
    assert "inspect each sanitized_answer_excerpt" in publish_weave_help
    publish_index_help = " ".join(
        publish_actions.choices["wandb-index"].format_help().split()
    )
    assert "Table lists the indexed Studies" in publish_index_help
    assert "does not create a native W&B Study or Report" in publish_index_help
    research_actions = next(
        action
        for action in subparsers.choices["research"]._actions
        if isinstance(action, cli.argparse._SubParsersAction)
    )
    assert "publications" in research_actions.choices
    assert "replay" in research_actions.choices["publications"].format_help()
    root_help = " ".join(cli._parser().format_help().split())
    assert "not a native W&B Study" in root_help
    assert "results or Research indexes" in root_help
    study_help = " ".join(subparsers.choices["study"].format_help().split())
    assert "scoped Weave publication receipts" in study_help


def test_comparison_readiness_names_the_canonical_local_destination(capsys) -> None:
    from fugue.bench.cli import _print_comparison_readiness_dict

    _print_comparison_readiness_dict(
        {
            "status": "ready",
            "question": "Does the candidate improve the locked tasks?",
            "evidence_project": None,
            "task_count": 2,
            "actual_changes": ["prompt"],
            "base_failures": 2,
            "gold_passes": 2,
            "judge_evaluators": [],
            "estimated_cells": 4,
            "estimated_cost_usd": 1.25,
            "blockers": [],
            "warnings": [],
        },
        evidence_mode="local",
    )

    output = " ".join(capsys.readouterr().out.split())
    assert "Evidence destination canonical local artifact ledger" in output


@pytest.mark.parametrize(
    ("evidence_project", "expected"),
    (
        (
            "wandb/locked-study",
            "canonical local artifact ledger + required W&B/Weave: wandb/locked-study",
        ),
        (
            None,
            "canonical local artifact ledger + required hosted destination "
            "resolved by the operator",
        ),
    ),
)
def test_comparison_readiness_names_required_hosted_destination(
    evidence_project: str | None,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.bench.cli import _print_comparison_readiness_dict

    _print_comparison_readiness_dict(
        {
            "status": "ready",
            "question": "Does the candidate improve the locked tasks?",
            "evidence_project": evidence_project,
            "task_count": 2,
            "actual_changes": ["prompt"],
            "base_failures": 2,
            "gold_passes": 2,
            "judge_evaluators": [],
            "estimated_cells": 4,
            "estimated_cost_usd": 1.25,
            "blockers": [],
            "warnings": [],
        },
        evidence_mode="weave_required",
    )

    output = " ".join(capsys.readouterr().out.split())
    assert expected in output


def test_local_preview_without_required_credentials_is_not_approvable(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "fugue.bench.comparison._run_custom_scorer",
        lambda _evaluator, *, output, expected, **_kwargs: {
            "score": 1.0 if output == expected else 0.0,
            "reason": "isolated scorer test double",
            "details": {
                "answer_present": output is not None,
                "expected_values": output == expected,
            },
        },
    )
    comparison = scaffold_comparison(tmp_path / "comparison")
    raw = yaml.safe_load(comparison.read_text(encoding="utf-8"))
    raw["execution"]["model"] = "anthropic/claude-sonnet-5"
    raw["execution"]["harnesses"] = ["claude-code"]
    raw["execution"]["preparation_required"] = False
    raw["baseline"]["integrations"] = ["missing-key-a"]
    raw["candidate"]["skills"] = []
    raw["candidate"]["integrations"] = ["missing-key-b"]
    raw["changed"] = ["integrations"]
    comparison.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    root = comparison.parent
    integration_root = root / "configs/fugue/integrations"
    integration_root.mkdir(parents=True, exist_ok=True)
    for integration_id in ("missing-key-a", "missing-key-b"):
        (integration_root / f"{integration_id}.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": integration_id,
                    "version": "v1",
                    "support": "supported",
                    "runtime": {
                        "type": "external",
                        "url": f"https://api.example.com/{integration_id}",
                    },
                    "interfaces": [
                        {
                            "type": "mcp",
                            "name": integration_id,
                            "transport": "streamable-http",
                            "url": f"https://api.example.com/{integration_id}",
                        }
                    ],
                    "required_env": ["ANTHROPIC_API_KEY", "WANDB_API_KEY"],
                    "allowed_hosts": ["api.example.com"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    assert (
        main(
            [
                "compare",
                comparison.as_posix(),
                "--preview",
                "--json",
                "--env-file",
                (tmp_path / "missing.env").as_posix(),
                "--repo-root",
                root.as_posix(),
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload["preview_digest"], str)
    assert payload["approval_eligible"] is False
    assert payload["matrix"]["applicable_cells"] == 0
    assert payload["matrix"]["estimated_trials"] == 0


def test_check_reports_missing_host_only_labels_without_traceback(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    root = tmp_path / "study"
    comparison = scaffold_standalone_template(
        root,
        template_id="prompt-change",
    )
    (root / "private-labels.jsonl").unlink()

    assert (
        main(
            [
                "check",
                comparison.as_posix(),
                "--repo-root",
                root.as_posix(),
                "--json",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert any(
        "host-only private labels are unavailable" in blocker
        for blocker in payload["blockers"]
    )


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


@pytest.mark.parametrize(
    "argv",
    (
        ["runs", "cancel", "--run-id", "run-1"],
        ["runs", "cancel", "run-1"],
    ),
)
def test_runs_cancel_accepts_action_first_recovery_grammar(
    argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.bench.supervisor import RunSupervisor

    cancelled: list[str] = []

    def cancel(self, run_id: str):
        del self
        cancelled.append(run_id)
        return SimpleNamespace(run_id=run_id, status="cancelled")

    monkeypatch.setattr(RunSupervisor, "cancel", cancel)
    assert (
        main(
            [
                *argv,
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )
    assert cancelled == ["run-1"]
    assert "cancelled" in capsys.readouterr().out


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


def test_run_preview_reports_missing_governed_asset_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / ".fugue/loop-engineering/tasksets/discovery.yaml"

    assert (
        main(
            [
                "run",
                "--manifest",
                missing.as_posix(),
                "--preview",
                "--json",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 2
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "asset": ".fugue/loop-engineering/tasksets/discovery.yaml",
        "error_type": "missing_governed_asset",
        "next_action": (
            "Materialize and lock the referenced task or evaluation asset before "
            "previewing or running this experiment."
        ),
        "schema_version": 1,
        "status": "blocked",
    }


def test_shell_environment_wins_over_blank_dotenv(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=\nWANDB_API_KEY=dotenv-value\n")
    env_file.chmod(0o600)
    monkeypatch.setenv("OPENAI_API_KEY", "shell-value")
    monkeypatch.setenv("WANDB_API_KEY", "shell-trace")

    env = load_env(env_file)

    assert env["OPENAI_API_KEY"] == "shell-value"
    assert env["WANDB_API_KEY"] == "shell-trace"


def test_load_env_rejects_group_or_world_readable_credentials(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=secret\n", encoding="utf-8")
    env_file.chmod(0o644)

    with pytest.raises(PermissionError, match="chmod 600"):
        load_env(env_file)


def test_load_env_rejects_a_symlinked_credential_file(tmp_path: Path) -> None:
    target = tmp_path / "private.env"
    target.write_text("ANTHROPIC_API_KEY=credential\n")
    target.chmod(0o600)
    link = tmp_path / ".env"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        load_env(link)


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
