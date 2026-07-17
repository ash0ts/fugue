from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from fugue.bench.candidates import stable_digest
from tools.experiment_atlas import (
    build_index,
    build_public_experiment,
    load_editorial,
    validate_public_experiment,
    write_publication,
)

DATASET_BODY = "id: locked-v1\n"
DATASET_DIGEST = hashlib.sha256(DATASET_BODY.encode()).hexdigest()


def _editorial(**overrides):
    value = {
        "schema_version": 1,
        "id": "controlled-grid",
        "title": "Controlled grid",
        "summary": "A deliberately small controlled grid.",
        "question": "Does the intervention change task resolution?",
        "hypothesis": "The treatment may improve one paired task.",
        "why_it_matters": "The result controls a product decision.",
        "task_selection": "Tasks were locked before execution.",
        "evidence_tier": "directional",
        "decision_value": 80,
        "status": "complete",
        "matrix": {
            "experiment_id": "demo",
            "workload_id": "coding",
            "expected_predictions": 2,
            "attempts": 1,
            "models": ["wandb/example/model"],
            "harnesses": ["codex"],
            "treatments": ["none", "rag-dense"],
            "tasks": ["task-1"],
            "cohorts": [
                {
                    "id": "cohort-1",
                    "label": "Compatible cohort",
                    "models": ["wandb/example/model"],
                    "harnesses": ["codex"],
                    "treatments": ["none", "rag-dense"],
                    "tasks": ["task-1"],
                    "expected_predictions": 2,
                }
            ],
        },
        "provenance": {
            "source_commit": "a" * 40,
            "source_url": f"https://github.com/ash0ts/fugue/commit/{'a' * 40}",
            "dataset_id": "locked-v1",
            "dataset_digest": DATASET_DIGEST,
            "snapshot_digest": "c" * 64,
            "run_ids": ["run-1"],
        },
        "links": {
            "project": "https://wandb.ai/example/project/weave",
            "evaluations": [],
        },
        "findings": ["One paired task changed."],
        "caveats": ["One attempt is directional."],
    }
    value.update(overrides)
    return value


def _row(index: int, *, treatment: str, passed: bool, trial_index: int = 1):
    return {
        "schema_version": 1,
        "prediction_schema_version": 1,
        "record_type": "trial",
        "source_record_type": "harbor",
        "prediction_id": f"prediction-{index}",
        "run_id": "run-1",
        "candidate_id": f"{index:064d}",
        "execution_fingerprint": f"{index + 1:064d}",
        "comparison_example_id": "example-1",
        "trial_index": trial_index,
        "execution_kind": "agent",
        "experiment_id": "demo",
        "preset_id": "study",
        "workload_id": "coding",
        "task_id": "task-1",
        "task_name": "task-1",
        "harness": "codex",
        "variant_id": treatment,
        "context_system_id": treatment,
        "context_delivery": "portable",
        "model_provider": "wandb",
        "model": "wandb/example/model",
        "status": "passed",
        "benchmark_outcome": "passed" if passed else "failed",
        "pass": passed,
        "reward": 1.0 if passed else 0.0,
        "wall_time_sec": 10.0 + index,
        "cost_usd": 0.1,
        "n_input_tokens": 100,
        "n_output_tokens": 20,
        "weave_tool_call_count": 2,
        "weave_turn_count": 1,
        "recoverable_error_count": 0,
        "provider_error_count": 0,
        "harness_error_count": 0,
        "trace_link_status": "linked",
        "agent_link_status": "linked",
        "trace_url": f"https://wandb.ai/example/project/weave/traces/trace-{index}",
        "context_registered": treatment != "none",
        "context_invoked": treatment != "none",
        "context_invocation_count": int(treatment != "none"),
        "recall_at_10": None,
        "mrr": None,
    }


def _run_summary(rows):
    evaluations = []
    for candidate_id in sorted({row["candidate_id"] for row in rows}):
        evaluations.append(
            {
                "active": True,
                "candidate_id": candidate_id,
                "agent_predictions": sum(
                    row["candidate_id"] == candidate_id for row in rows
                ),
                "linked_agent_predictions": sum(
                    row["candidate_id"] == candidate_id for row in rows
                ),
                "linking_failures": [],
                "url": f"https://wandb.ai/example/project/r/call/{candidate_id[:12]}",
            }
        )
    return {"run_id": "run-1", "status": "passed", "evaluation_runs": evaluations}


def _snapshot(rows):
    value = {
        "schema_version": 1,
        "snapshot_sha256": "",
        "lock_sha256": "",
        "run_id": "run-1",
        "request": {"experiment_id": "demo", "manifest": None},
        "experiment": {
            "manifest": "datasets/pilot.yaml",
            "workloads": [
                {"id": "coding", "manifest": "datasets/locked-v1.yaml"}
            ],
        },
        "runtime": {
            "executions": {
                "execution-1": {
                    "fugue_source": {
                        "kind": "git",
                        "commit": "a" * 40,
                        "dirty": False,
                    }
                }
            }
        },
        "planned_prediction_count": len(rows),
        "planned_matrix": [
            {
                "applicable": True,
                "candidate_id": row["candidate_id"],
                "comparison_example_id": row["comparison_example_id"],
                "trial_index": row["trial_index"],
                "execution_kind": row["execution_kind"],
                "workload_id": row["workload_id"],
                "task_id": row["task_id"],
            }
            for row in rows
        ],
    }
    digest = stable_digest(value)
    value["snapshot_sha256"] = digest
    value["lock_sha256"] = digest
    return value


def _build(editorial, rows):
    return build_public_experiment(editorial, rows, [_run_summary(rows)])


def test_public_experiment_recomputes_metrics_and_orders_index() -> None:
    directional = _build(
        _editorial(),
        [_row(1, treatment="none", passed=False), _row(2, treatment="rag-dense", passed=True)],
    )
    contract = _build(
        _editorial(
            id="contract",
            evidence_tier="contract",
            decision_value=99,
        ),
        [_row(3, treatment="none", passed=True), _row(4, treatment="rag-dense", passed=True)],
    )

    assert directional.metrics["passed_predictions"] == 1
    assert directional.metrics["pass_rate"] == 0.5
    assert directional.metrics["total_cost_usd"] == pytest.approx(0.2)
    assert directional.metrics["paired_bootstrap"] is None
    assert [item["id"] for item in build_index([contract, directional]).experiments] == [
        "controlled-grid",
        "contract",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update({"schema_version": 2}), "schema 1"),
        (lambda row: row.update({"trace_link_status": "missing"}), "verified trace"),
        (lambda row: row.update({"task_id": "/Users/alice/private"}), "local path"),
        (lambda row: row.update({"model": "sk-secretsecretsecret"}), "secret-like"),
    ],
)
def test_public_snapshot_rejects_unsafe_or_mixed_rows(mutation, message: str) -> None:
    first = _row(1, treatment="none", passed=False)
    mutation(first)

    with pytest.raises(ValueError, match=message):
        _build(
            _editorial(),
            [first, _row(2, treatment="rag-dense", passed=True)],
        )


def test_public_snapshot_rejects_duplicates_and_incompatible_cohorts() -> None:
    duplicate = _row(1, treatment="none", passed=False)
    second = _row(2, treatment="rag-dense", passed=True)
    second["prediction_id"] = duplicate["prediction_id"]
    with pytest.raises(ValueError, match="duplicate prediction"):
        _build(_editorial(), [duplicate, second])

    duplicate_coordinate = _row(2, treatment="none", passed=True)
    with pytest.raises(ValueError, match="duplicate frozen-matrix coordinate"):
        _build(
            _editorial(),
            [_row(1, treatment="none", passed=False), duplicate_coordinate],
        )

    wrong_model = _row(2, treatment="rag-dense", passed=True)
    wrong_model["model"] = "anthropic/other"
    with pytest.raises(ValueError, match="incompatible models"):
        _build(
            _editorial(),
            [_row(1, treatment="none", passed=False), wrong_model],
        )

    incompatible = _editorial()
    incompatible["matrix"]["models"].append("anthropic/other")
    wrong_cohort = _row(2, treatment="rag-dense", passed=True)
    wrong_cohort["model"] = "anthropic/other"
    with pytest.raises(ValueError, match="compatible cohort"):
        _build(
            incompatible,
            [_row(1, treatment="none", passed=False), wrong_cohort],
        )


def test_editorial_cannot_override_results_or_add_fields(tmp_path: Path) -> None:
    record = _editorial(metrics={"pass_rate": 1.0})
    path = tmp_path / "record.yaml"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected fields"):
        load_editorial(path)


def test_rich_private_rows_are_reduced_to_the_public_allowlist() -> None:
    first = _row(1, treatment="none", passed=False)
    first.update(
        {
            "agent_response": "private response",
            "reasoning": "private chain",
            "trial_dir": "/private/tmp/run",
            "exception_class": "RuntimeError",
            "arbitrary_new_private_field": {"secret": "not copied"},
        }
    )

    public = _build(
        _editorial(),
        [first, _row(2, treatment="rag-dense", passed=True)],
    )

    serialized = json.dumps(public.to_dict())
    assert "private response" not in serialized
    assert "private chain" not in serialized
    assert "trial_dir" not in serialized
    assert "exception_class" not in serialized


def test_public_snapshot_validator_rejects_raw_or_unexpected_content() -> None:
    public = _build(
        _editorial(),
        [
            _row(1, treatment="none", passed=False),
            _row(2, treatment="rag-dense", passed=True),
        ],
    ).to_dict()
    public["prompt"] = "raw prompt"

    with pytest.raises(ValueError, match="unexpected fields"):
        validate_public_experiment(public)


def test_public_snapshot_validator_recomputes_metrics_and_links() -> None:
    public = _build(
        _editorial(),
        [
            _row(1, treatment="none", passed=False),
            _row(2, treatment="rag-dense", passed=True),
        ],
    ).to_dict()
    public["metrics"]["pass_rate"] = 1.0

    with pytest.raises(ValueError, match="metrics do not match"):
        validate_public_experiment(public)

    public = _build(
        _editorial(),
        [
            _row(1, treatment="none", passed=False),
            _row(2, treatment="rag-dense", passed=True),
        ],
    ).to_dict()
    public["links"]["evaluations"] = []
    with pytest.raises(ValueError, match="links do not match"):
        validate_public_experiment(public)


def test_evaluation_links_require_reconciled_run_summaries() -> None:
    rows = [
        _row(1, treatment="none", passed=False),
        _row(2, treatment="rag-dense", passed=True),
    ]
    with pytest.raises(ValueError, match="covered by declared run summaries"):
        build_public_experiment(_editorial(), rows)

    summary = _run_summary(rows)
    summary["evaluation_runs"][0]["linked_agent_predictions"] = 0
    with pytest.raises(ValueError, match="do not verify every Agent"):
        build_public_experiment(_editorial(), rows, [summary])

    record = _editorial()
    record["links"]["evaluations"] = [
        "https://wandb.ai/example/project/r/call/editorial-claim"
    ]
    with pytest.raises(ValueError, match="derived from run summaries"):
        build_public_experiment(record, rows, [_run_summary(rows)])


def test_missing_usage_remains_unavailable() -> None:
    first = _row(1, treatment="none", passed=False)
    first["cost_usd"] = None
    first["n_input_tokens"] = None
    first["n_output_tokens"] = None
    public = _build(
        _editorial(),
        [first, _row(2, treatment="rag-dense", passed=True)],
    )

    assert public.metrics["total_cost_usd"] is None
    assert public.metrics["input_tokens"] is None
    assert public.cells[0]["cost_usd"] is None


def test_public_cost_uses_authoritative_weave_measurement() -> None:
    first = _row(1, treatment="none", passed=False)
    first["cost_usd"] = 0.1
    first["weave_total_cost_usd"] = 0.4
    second = _row(2, treatment="rag-dense", passed=True)
    second["weave_total_cost_usd"] = None

    public = _build(_editorial(), [first, second])

    assert public.cells[0]["cost_usd"] == 0.4
    assert public.metrics["total_cost_usd"] == pytest.approx(0.5)


def test_confirmed_evidence_requires_replication_and_uses_paired_bootstrap() -> None:
    with pytest.raises(ValueError, match="replicated"):
        _build(
            _editorial(evidence_tier="confirmed"),
            [_row(1, treatment="none", passed=False), _row(2, treatment="rag-dense", passed=True)],
        )

    editorial = _editorial(
        evidence_tier="confirmed",
        matrix={
            **_editorial()["matrix"],
            "expected_predictions": 4,
            "attempts": 2,
            "cohorts": [
                {
                    **_editorial()["matrix"]["cohorts"][0],
                    "expected_predictions": 4,
                }
            ],
        },
    )
    public = _build(
        editorial,
        [
            _row(1, treatment="none", passed=False, trial_index=1),
            _row(2, treatment="rag-dense", passed=True, trial_index=1),
            _row(3, treatment="none", passed=False, trial_index=2),
            _row(4, treatment="rag-dense", passed=True, trial_index=2),
        ],
    )

    assert public.metrics["paired_bootstrap"] == [
        {
            "treatment": "rag-dense",
            "baseline": "none",
            "confidence": 0.95,
            "low": 1.0,
            "high": 1.0,
        }
    ]


def test_public_generation_is_reproducible(tmp_path: Path) -> None:
    editorial_dir = tmp_path / "editorial"
    editorial_dir.mkdir()
    dataset = tmp_path / "datasets" / "locked-v1.yaml"
    dataset.parent.mkdir()
    dataset.write_text(DATASET_BODY, encoding="utf-8")
    rows = [
        _row(1, treatment="none", passed=False),
        _row(2, treatment="rag-dense", passed=True),
    ]
    snapshot = _snapshot(rows)
    record = _editorial()
    record["provenance"]["snapshot_digest"] = snapshot["snapshot_sha256"]
    (editorial_dir / "controlled-grid.yaml").write_text(
        yaml.safe_dump(record), encoding="utf-8"
    )
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    summary_path = tmp_path / "run.json"
    summary_path.write_text(
        json.dumps(
            _run_summary(rows)
        ),
        encoding="utf-8",
    )
    snapshot_path = tmp_path / "input-lock.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    left = write_publication(
        list(editorial_dir.glob("*.yaml")),
        {"controlled-grid": rows_path},
        {"controlled-grid": [summary_path]},
        {"controlled-grid": [snapshot_path]},
        first,
        repo_root=tmp_path,
    )
    right = write_publication(
        list(editorial_dir.glob("*.yaml")),
        {"controlled-grid": rows_path},
        {"controlled-grid": [summary_path]},
        {"controlled-grid": [snapshot_path]},
        second,
        repo_root=tmp_path,
    )

    assert left == right
    assert (first / "index.json").read_bytes() == (second / "index.json").read_bytes()
    assert (
        first / "experiments" / "controlled-grid.json"
    ).read_bytes() == (
        second / "experiments" / "controlled-grid.json"
    ).read_bytes()


def test_trusted_publication_rejects_unverified_snapshot_schema(tmp_path: Path) -> None:
    editorial_dir = tmp_path / "editorial"
    editorial_dir.mkdir()
    dataset = tmp_path / "datasets" / "locked-v1.yaml"
    dataset.parent.mkdir()
    dataset.write_text(DATASET_BODY, encoding="utf-8")
    rows = [
        _row(1, treatment="none", passed=False),
        _row(2, treatment="rag-dense", passed=True),
    ]
    snapshot = _snapshot(rows)
    snapshot["schema_version"] = 2
    record = _editorial()
    record["provenance"]["snapshot_digest"] = snapshot["snapshot_sha256"]
    editorial_path = editorial_dir / "controlled-grid.yaml"
    editorial_path.write_text(yaml.safe_dump(record), encoding="utf-8")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    summary_path = tmp_path / "run.json"
    summary_path.write_text(json.dumps(_run_summary(rows)), encoding="utf-8")
    snapshot_path = tmp_path / "input-lock.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(ValueError, match="valid immutable run snapshots"):
        write_publication(
            [editorial_path],
            {"controlled-grid": rows_path},
            {"controlled-grid": [summary_path]},
            {"controlled-grid": [snapshot_path]},
            tmp_path / "public",
            repo_root=tmp_path,
        )
