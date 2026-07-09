from pathlib import Path

from fugue.bench.export import export_rows, write_jsonl


def test_export_joins_harbor_result_and_fugue_meta(tmp_path: Path) -> None:
    jobs = Path(__file__).parent / "fixtures" / "export" / "jobs"

    rows = export_rows([jobs])

    assert len(rows) == 1
    row = rows[0]
    assert row["run_key"] == "bridge-check__abc123"
    assert row["harness"] == "hermes"
    assert row["condition"] == "none"
    assert row["model_provider"] == "wandb"
    assert row["trace_project"] == "test/fugue"
    assert row["reward"] == 1.0
    assert row["pass"] is True
    assert row["wall_time_sec"] == 5.0

    out = tmp_path / "pilot.jsonl"
    write_jsonl(rows, out)
    assert "bridge-check__abc123" in out.read_text()
