from __future__ import annotations

import json
from pathlib import Path

from fugue.bench.cli import main
from fugue.bench.operator import load_env


def test_bare_fugue_launches_compose_tui(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "fugue.bench.cli._tui",
        lambda args: calls.append((args.screen, args.experiment)) or 0,
    )

    assert main([]) == 0
    assert calls == [("compose", "pilot")]


def test_run_dry_run_uses_cli_model_and_neutral_adapter(
    tmp_path: Path, capsys
) -> None:
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
                "--dry-run",
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
    config_line = next(line for line in out.splitlines() if line.startswith("# config: "))
    config_path = Path(config_line.removeprefix("# config: "))
    config = json.loads(config_path.read_text())
    assert config["agents"][0]["model_name"] == "openai/gpt-5"
    assert config["agents"][0]["import_path"] == "fugue.agents:FugueCodex"
    assert config["job_name"] == (
        "unit-exp-harbor-codex-baseline-astropy-astropy-12907"
    )
    assert config["fugue"]["experiment_id"] == "pilot"
    assert config["fugue"]["variant_id"] == "baseline"
    assert config["fugue"]["context_system_id"] == "none"


def test_shell_environment_wins_over_blank_dotenv(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=\nWANDB_API_KEY=dotenv-value\n")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-value")
    monkeypatch.setenv("WANDB_API_KEY", "shell-trace")

    env = load_env(env_file)

    assert env["OPENAI_API_KEY"] == "shell-value"
    assert env["WANDB_API_KEY"] == "shell-trace"


def test_repo_memory_smoke_render_uses_per_workload_limits(tmp_path: Path, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert (
        main(
            [
                "render",
                "--experiment",
                "repo-memory-impact",
                "--preset",
                "smoke",
                "--run-name",
                "smoke-preview",
                "--repo-root",
                repo_root.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    paths = [
        Path(line.removeprefix("# config: "))
        for line in output.splitlines()
        if line.startswith("# config: ")
    ]
    harbor_configs = [json.loads(path.read_text()) for path in paths if path.suffix == ".json"]
    counts = {
        workload: {
            len(item["datasets"][0]["task_names"])
            for item in harbor_configs
            if item["fugue"]["workload_id"] == workload
        }
        for workload in ("qa", "coding")
    }
    assert counts == {"qa": {1}, "coding": {1}}
    assert "--limit 3" in output
    assert "--limit 1" in output


def test_catalog_cli_refreshes_local_experiment_buckets(tmp_path: Path, capsys) -> None:
    from test_operator import make_operator_repo

    make_operator_repo(tmp_path)
    assert (
        main(
            [
                "catalog",
                "refresh",
                "--source",
                "local",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                (tmp_path / ".env").as_posix(),
            ]
        )
        == 0
    )
    assert "1 experiments" in capsys.readouterr().out
