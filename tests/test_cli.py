from __future__ import annotations

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
conditions: [none]
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
    assert "-m openai/gpt-5" in out
    assert "-a fugue.agents:FugueCodex" in out
