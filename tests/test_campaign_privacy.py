from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import fugue.bench.run_conformance as run_conformance
from fugue.bench.campaign_evidence import (
    _harbor_conformance_failures,
    apply_campaign_run_conformance,
    safe_prediction_row,
    verified_trace_link_set,
)
from fugue.bench.campaign_lifecycle import CampaignService
from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.library import get_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]


def _verified_trace_row() -> dict[str, object]:
    project = "wandb/fugue-test"
    row: dict[str, object] = {
        "attempt_id": "a" * 64,
        "trace_project": project,
        "trace_receipt": {"app_base_url": "https://wandb.ai"},
        "otel_trace_id": "b" * 32,
        "otel_root_span_id": "c" * 16,
        "weave_dataset_ref": f"weave:///{project}/object/tasks:dataset-v1",
        "weave_dataset_url": (
            f"https://wandb.ai/{project}/weave/objects/tasks/versions/dataset-v1"
        ),
        "dataset_version_object_verified": True,
    }
    for prefix, call_id in (
        ("weave_evaluation_root", "evaluation"),
        ("eval_predict_and_score", "predict-and-score"),
        ("weave_prediction", "prediction"),
        ("weave_agent_root", "agent-evidence"),
    ):
        id_field = (
            "eval_predict_and_score_call_id"
            if prefix == "eval_predict_and_score"
            else f"{prefix}_call_id"
        )
        row[id_field] = call_id
        row[f"{prefix}_ref"] = f"weave:///{project}/call/{call_id}"
        row[f"{prefix}_url"] = (
            f"https://wandb.ai/{project}/weave/calls/{call_id}"
        )
    row.update(
        {
            "evaluation_root_object_verified": True,
            "eval_predict_and_score_object_verified": True,
            "weave_prediction_object_verified": True,
            "agent_graph_verified": True,
        }
    )
    return row


def test_trace_link_set_rejects_unknown_agent_evidence_kind() -> None:
    row = _verified_trace_row()
    row.update(
        {
            "weave_agent_root_evidence_kind": "asserted_agent_root_v0",
            "weave_agent_root_is_native_call": True,
        }
    )

    result = verified_trace_link_set(row)

    assert result["verified"] is False
    assert "does not recognize the Agent evidence kind" in result["failures"]


def test_trace_link_set_reserves_agent_root_for_native_weave_call() -> None:
    row = _verified_trace_row()
    row.update(
        {
            "weave_agent_root_evidence_kind": "native_weave_call_v1",
            "weave_agent_root_is_native_call": True,
            "conversation_correlation_verified": True,
        }
    )

    result = verified_trace_link_set(row)

    assert result["verified"] is True
    agent_link = next(
        link for link in result["links"] if link["slot"] == "agent_root"
    )
    assert agent_link["kind"] == "agent_root"
    assert agent_link["native_trajectory_status"] == "native_weave_call"
    assert agent_link["conversation_correlation_status"] == "verified"


def test_trace_link_set_requires_exact_receipt_cross_transport_edge() -> None:
    row = _verified_trace_row()
    row.update(
        {
            "weave_agent_root_evidence_kind": (
                "native_otel_cross_transport_receipt_v1"
            ),
            "weave_agent_root_is_native_call": False,
            "agent_cross_transport_edge": {
                "schema_version": 1,
                "status": "verified",
                "source_system": "otel",
                "source_trace_id": "wrong-trace",
                "source_span_id": "c" * 16,
                "receipt_system": "weave",
                "receipt_call_id": "agent-evidence",
            },
        }
    )

    result = verified_trace_link_set(row)

    assert result["verified"] is False
    assert "does not verify the Agent receipt cross-transport edge" in result[
        "failures"
    ]

    row["agent_cross_transport_edge"]["source_trace_id"] = "b" * 32  # type: ignore[index]
    result = verified_trace_link_set(row)
    assert result["verified"] is False
    assert "does not verify the Agent conversation correlation" in result["failures"]

    row["conversation_correlation_verified"] = True
    result = verified_trace_link_set(row)
    assert result["verified"] is True
    agent_link = next(
        link
        for link in result["links"]
        if link["slot"] == "agent_evidence_receipt"
    )
    assert agent_link["kind"] == "agent_evidence_receipt"
    assert agent_link["evidence_kind"] == (
        "native_otel_cross_transport_receipt_v1"
    )
    assert agent_link["native_trajectory_status"] == "otel_correlated"
    assert agent_link["conversation_correlation_status"] == "verified"

    row["execution_kind"] = "agent"
    safe_row = safe_prediction_row(row)
    assert "weave_agent_root_call_id" not in safe_row
    assert safe_row["weave_agent_evidence_receipt_call_id"] == "agent-evidence"
    assert safe_row["native_trajectory_status"] == "otel_correlated"


def test_local_harbor_receipt_gates_every_agent_row_not_a_workload_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        "schema_version": run_conformance.PRIVACY_CONTRACT_VERSION,
        "backend": "local_harbor_docker",
        "status": "passed",
        "receipt_sha256": "a" * 64,
        "execution_identity": {"status": "passed"},
        "local_artifact_privacy_scan": {
            "status": "passed",
            "files_with_matches": [],
        },
        "private_label_boundary": {"status": "passed"},
        "docker_cleanup": {
            "status": "passed",
            "matched_containers": [],
        },
    }
    monkeypatch.setattr(
        run_conformance,
        "read_harbor_run_conformance_receipt",
        lambda **_kwargs: receipt,
    )
    monkeypatch.setattr(
        run_conformance,
        "read_hosted_evidence_privacy_receipt",
        lambda **_kwargs: {
            "status": "passed",
            "receipt_sha256": "b" * 64,
            "secret_match_count": 0,
            "private_corpus_match_count": 0,
            "private_structure_match_count": 0,
        },
    )
    rows = [
        {
            "run_id": "local-run",
            "workload_id": "maintenance-task-one",
            "execution_kind": "agent",
            "harbor_environment": "docker",
        },
        {
            "run_id": "local-run",
            "workload_id": "maintenance-task-two",
            "execution_kind": "agent",
        },
        {
            "run_id": "local-run",
            "workload_id": "diagnostic",
            "execution_kind": "diagnostic",
        },
    ]

    apply_campaign_run_conformance(
        rows,
        repo_root=tmp_path,
        run_id="local-run",
    )

    for row in rows[:2]:
        assert row["harbor_environment"] == "local_harbor_docker"
        assert _harbor_conformance_failures(row) == []
    assert "harbor_conformance_status" not in rows[2]

    rows[1]["sandbox_cleanup_verified"] = False
    assert "lacks verified run-scoped Harbor cleanup" in (
        _harbor_conformance_failures(rows[1])
    )


def test_changed_loop_intervention_requires_exact_observed_invocation() -> None:
    row = {
        "execution_kind": "agent",
        "harbor_environment": "local_harbor_docker",
        "harbor_conformance_status": "passed",
        "harbor_policy_attestation_verified": True,
        "privacy_contract_version": 2,
        "local_artifact_privacy_scan_status": "passed",
        "local_artifact_privacy_match_count": 0,
        "hosted_evidence_privacy_scan_status": "passed",
        "hosted_evidence_privacy_match_count": 0,
        "private_label_boundary_verified": True,
        "sandbox_cleanup_verified": True,
        "orphaned_sandbox": False,
        "integration_ids": [
            "optional-generic-mcp",
            "loop-intervention-fixed-mcp",
        ],
        "integration_ids_invoked": [],
        "skills_assigned": [
            "optional-generic-skill",
            "loop-intervention-bounded-evidence",
        ],
        "skill_ids_invoked": [],
    }

    failures = _harbor_conformance_failures(row)

    assert any("loop-intervention-fixed-mcp" in item for item in failures)
    assert any("loop-intervention-bounded-evidence" in item for item in failures)
    assert not any("optional-generic" in item for item in failures)

    row["integration_ids_invoked"] = ["loop-intervention-fixed-mcp"]
    row["skill_ids_invoked"] = ["loop-intervention-bounded-evidence"]

    assert _harbor_conformance_failures(row) == []


def test_local_harbor_without_run_receipt_fails_closed_for_custom_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_receipt(**_kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(
        run_conformance,
        "read_harbor_run_conformance_receipt",
        missing_receipt,
    )
    row = {
        "run_id": "missing-receipt",
        "workload_id": "real-maintenance-task",
        "execution_kind": "agent",
        "harbor_environment": "docker",
    }

    apply_campaign_run_conformance(
        [row],
        repo_root=tmp_path,
        run_id="missing-receipt",
    )

    assert row["harbor_conformance_status"] == "unavailable"
    assert _harbor_conformance_failures(row)


def test_expected_harbor_backend_cannot_be_disabled_by_omitting_row_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_receipt(**_kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(
        run_conformance,
        "read_harbor_run_conformance_receipt",
        missing_receipt,
    )
    attempt_id = "a" * 64
    row = {
        "run_id": "missing-markers",
        "attempt_id": attempt_id,
        "execution_kind": "agent",
    }

    apply_campaign_run_conformance(
        [row],
        repo_root=tmp_path,
        run_id="missing-markers",
        required_attempt_ids=frozenset({attempt_id}),
    )

    assert row["harbor_conformance_status"] == "unavailable"
    assert _harbor_conformance_failures(row, required=True)


@pytest.mark.parametrize(
    "experiment_id",
    ["real-harness-study", "real-memory-study"],
)
def test_real_lane_uses_locked_host_evaluation_corpus(
    tmp_path: Path,
    experiment_id: str,
) -> None:
    experiment = get_experiment(experiment_id, REPO_ROOT)
    assert experiment.workloads[0].runner == "harbor"

    run_id = f"{experiment_id}-run"
    unsigned = {
        "schema_version": 1,
        "run_id": run_id,
        "predictions": {
            "prediction-lock": {
                "task_id": "locked-maintenance-task",
                "expected_evidence_paths": ["locked/private-path.py"],
            }
        },
        "lock_sha256": "",
    }
    lock_sha256 = stable_digest(unsigned)
    lock_path = tmp_path / ".fugue/runtime" / run_id / "evaluation-assets.json"
    atomic_write_json(lock_path, {**unsigned, "lock_sha256": lock_sha256})
    lock_path.chmod(0o600)

    service = CampaignService(tmp_path, operator=SimpleNamespace())
    source = service._hosted_private_corpus(  # noqa: SLF001
        proposal=SimpleNamespace(task_suite_digest=None),
        run_id=run_id,
    )

    assert source == {
        "path": lock_path,
        "applicable": True,
        "source_kind": "run_evaluation_asset_lock",
        "source_lock_sha256": lock_sha256,
        "reason": (
            "run-scoped host evaluation assets contain private prediction material"
        ),
    }


def test_valid_empty_evaluation_lock_marks_private_corpus_not_applicable(
    tmp_path: Path,
) -> None:
    run_id = "no-private-corpus"
    unsigned = {
        "schema_version": 1,
        "run_id": run_id,
        "predictions": {},
        "lock_sha256": "",
    }
    lock_sha256 = stable_digest(unsigned)
    lock_path = tmp_path / ".fugue/runtime" / run_id / "evaluation-assets.json"
    atomic_write_json(lock_path, {**unsigned, "lock_sha256": lock_sha256})

    service = CampaignService(tmp_path, operator=SimpleNamespace())
    source = service._hosted_private_corpus(  # noqa: SLF001
        proposal=SimpleNamespace(task_suite_digest=None),
        run_id=run_id,
    )

    assert source["path"] is None
    assert source["applicable"] is False
    assert source["source_lock_sha256"] == lock_sha256
