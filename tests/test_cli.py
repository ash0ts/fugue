from __future__ import annotations

import json
from pathlib import Path

from fugue.bench.cli import main


def test_run_dry_run_uses_cli_model_and_neutral_adapter(
    tmp_path: Path, capsys
) -> None:
    manifest = tmp_path / "pilot.yaml"
    manifest.write_text(
        """
dataset:
  ref: swe-bench/swe-bench-verified
memory_variants: [none]
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
    assert config["job_name"] == "unit-exp-codex-baseline"
    assert config["fugue"]["experiment_id"] == "pilot"
    assert config["fugue"]["variant_id"] == "baseline"
    assert config["fugue"]["feature_memory"] == "none"
