from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import analyze_comparison_rows, write_comparison_result
from fugue.bench.files import atomic_write_json
from fugue.research.approvals import ApprovalLedger
from fugue.research.comparisons import (
    COMPARISON_RESULT_ROOT,
    ComparisonControlService,
    ComparisonRegistry,
    _project_result,
    _signed_state,
)
from fugue.research.contracts import ResearchError
from fugue.research.store import StudyStore

_COMPARISON_ID = "source-use-skill-v1"


def _control(
    tmp_path: Path,
    *,
    launcher: Any = None,
) -> tuple[ComparisonControlService, StudyStore, ApprovalLedger]:
    repo_root = Path(__file__).resolve().parents[1]
    store = StudyStore(tmp_path)
    store.create_study(
        study_id="study-1",
        title="Comparison control",
        campaign_id="source-use-skill-v1",
        question="Does the registered comparison improve the result?",
        operation_id="create-study",
    )
    approvals = ApprovalLedger(store.path)
    service = ComparisonControlService(
        repo_root,
        store=store,
        approvals=approvals,
        state_root=tmp_path / "private",
        public_root=tmp_path / "public",
        launch_worker=launcher,
    )
    return service, store, approvals


def test_registry_rejects_unknown_and_drifted_comparisons(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = ComparisonRegistry.from_file(repo_root)

    with pytest.raises(ResearchError, match="not registered"):
        registry.resolve("unreviewed-comparison")

    drifted = tmp_path / "comparisons.yaml"
    drifted.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "comparisons:",
                f"  - id: {_COMPARISON_ID}",
                "    path: examples/comparisons/source-use-skill/comparison.yaml",
                f"    digest: {'0' * 64}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="registered comparison drift"):
        ComparisonRegistry.from_file(repo_root, drifted)


def test_public_preview_and_study_resource_exclude_private_labels(
    tmp_path: Path,
) -> None:
    service, store, _ = _control(tmp_path)

    preview = service.preview("study-1", _COMPARISON_ID)
    serialized = json.dumps(preview, sort_keys=True)

    assert preview["readiness"]["status"] == "ready"
    assert preview["readiness"]["estimated_cells"] > 0
    assert "private_labels_digest" not in serialized
    assert "criteria_digest" not in serialized
    assert "gold_output" not in serialized
    assert "base_output" not in serialized
    unsigned = dict(preview)
    artifact_digest = unsigned.pop("artifact_digest")
    assert stable_digest(unsigned) == artifact_digest

    digest = str(preview["preview_digest"])
    private_path = service.state_root / digest / "preview.json"
    public_path = service.public_root / "study-1" / digest / "preview.json"
    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    assert json.loads(public_path.read_text(encoding="utf-8")) == preview

    requested = service.request_approval(
        "study-1",
        _COMPARISON_ID,
        digest,
        idempotency_key="request-comparison-approval",
    )
    assert requested["approval_state"] == "awaiting_approval"
    resource = store.get_study("study-1").resources[-1]
    assert resource.digest == artifact_digest
    assert resource.version == digest
    assert Path(resource.uri).resolve() == public_path.resolve()
    [event] = [
        item
        for item in store.research_log_events()
        if item.summary.get("experiment_view", {}).get("kind") == "design"
    ]
    view = event.summary["experiment_view"]
    assert event.study_id == f"comparison-{digest[:20]}"
    assert view["question"] == registry_question(service)
    assert view["matrix_size"] == preview["readiness"]["estimated_cells"]
    assert view["taskset"]["digest"] == preview["readiness"]["taskset_digest"]
    assert view["runtime"]["id"] == "harbor"
    assert {item["id"] for item in view["treatment_arms"]} == {
        "baseline",
        "candidate",
    }
    assert "private_labels" not in json.dumps(event.to_dict(), sort_keys=True)


def test_exact_approval_is_required_and_launch_is_idempotent(
    tmp_path: Path,
) -> None:
    launches: list[Path] = []

    def launch(path: Path) -> int:
        launches.append(path)
        return os.getpid()

    service, _, approvals = _control(tmp_path, launcher=launch)
    preview = service.preview("study-1", _COMPARISON_ID)
    digest = str(preview["preview_digest"])

    with pytest.raises(ResearchError) as missing:
        service.start(
            "study-1",
            _COMPARISON_ID,
            digest,
            idempotency_key="start-comparison",
        )
    assert missing.value.code == "approval_required"
    assert not launches

    approvals.approve(
        subject_kind="experiment",
        preview_digest=digest,
        maximum_cost_usd=float(preview["readiness"]["estimated_cost_usd"]),
        maximum_cells=int(preview["readiness"]["estimated_cells"]),
        approved_by="operator",
        operation_id="approve-comparison",
    )
    started = service.start(
        "study-1",
        _COMPARISON_ID,
        digest,
        idempotency_key="start-comparison",
    )
    repeated = service.start(
        "study-1",
        _COMPARISON_ID,
        digest,
        idempotency_key="start-comparison",
    )

    assert started == repeated
    assert started["status"] == "running"
    assert len(launches) == 1
    assert "approval" not in json.dumps(started, sort_keys=True)
    worker_input = json.loads(launches[0].read_text(encoding="utf-8"))
    assert len(str(worker_input["approval_digest"])) == 64
    assert stat.S_IMODE(launches[0].stat().st_mode) == 0o600
    [progress] = [
        item
        for item in service.store.research_log_events()
        if item.summary.get("experiment_view", {}).get("kind") == "progress"
    ]
    assert progress.state == "running"
    assert progress.progress == {
        "completed": 0,
        "total": preview["readiness"]["estimated_cells"],
    }


def test_approval_limits_are_checked_before_worker_launch(tmp_path: Path) -> None:
    launches: list[Path] = []
    service, _, approvals = _control(
        tmp_path,
        launcher=lambda path: launches.append(path) or os.getpid(),
    )
    preview = service.preview("study-1", _COMPARISON_ID)
    digest = str(preview["preview_digest"])
    approvals.approve(
        subject_kind="experiment",
        preview_digest=digest,
        maximum_cost_usd=0,
        maximum_cells=1,
        approved_by="operator",
        operation_id="approve-too-small",
    )

    with pytest.raises(ResearchError) as denied:
        service.start(
            "study-1",
            _COMPARISON_ID,
            digest,
            idempotency_key="start-over-limit",
        )

    assert denied.value.code in {"approval_cell_limit", "approval_cost_limit"}
    assert not launches


def test_result_projection_is_safe_and_result_digest_is_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, store, _ = _control(tmp_path)
    digest = "a" * 64
    result = analyze_comparison_rows(
        comparison_id=_COMPARISON_ID,
        preview_digest=digest,
        source="wandb-artifact://entity/project/comparison:v1",
        rows=(
            {
                "task_id": "task-1",
                "harness": "claude",
                "trial_index": 1,
                "variant_id": "baseline",
                "prediction_id": "prediction-base",
                "pass": False,
                "comparison_required_evaluation_complete": True,
            },
            {
                "task_id": "task-1",
                "harness": "claude",
                "trial_index": 1,
                "variant_id": "candidate",
                "prediction_id": "prediction-candidate",
                "pass": True,
                "comparison_required_evaluation_complete": True,
            },
        ),
    )
    result_dir = tmp_path / COMPARISON_RESULT_ROOT / digest
    result_path, _ = write_comparison_result(result, destination=result_dir)
    _project_result(store, "study-1", result, result_path, tmp_path)
    study = store.get_study("study-1")
    assert study.results[-1].sample_size == 2
    assert study.results[-1].estimate.value == 1
    assert study.resources[-1].digest == result.result_digest
    [evaluation] = [
        item
        for item in store.research_log_events()
        if item.summary.get("experiment_view", {}).get("kind") == "evaluation"
    ]
    view = evaluation.summary["experiment_view"]
    assert evaluation.study_id == f"comparison-{digest[:20]}"
    assert view["matrix_size"] == 2
    assert view["aligned_comparisons"][0]["pairs"] == 1
    assert view["arm_totals"][1]["passed"] == 1
    assert view["infrastructure_health"] == "healthy"
    assert view["evidence_eligible"] is False
    assert any(
        item["id"] == "judge-evidence" and item["status"] == "not_applicable"
        for item in view["outcome_summaries"]
    )
    assert any(item["kind"] == "comparison_result" for item in view["evidence_links"])

    result_service = ComparisonControlService(
        tmp_path,
        store=store,
        approvals=ApprovalLedger(store.path),
        state_root=tmp_path / "private",
        public_root=tmp_path / "public",
    )
    monkeypatch.setattr(
        result_service,
        "_accepted_preview",
        lambda *_: (None, None),
    )
    state_path = result_service.state_root / digest / "state.json"
    state_path.parent.mkdir(parents=True)
    atomic_write_json(
        state_path,
        _signed_state(
            {
                "schema_version": 1,
                "study_id": "study-1",
                "comparison_id": _COMPARISON_ID,
                "spec_digest": "b" * 64,
                "preview_digest": digest,
                "status": "completed",
                "pid": None,
                "created_at": "2026-07-28T00:00:00Z",
                "updated_at": "2026-07-28T00:00:01Z",
                "result_digest": result.result_digest,
                "rows": result.rows,
                "result_path": result_path.relative_to(tmp_path).as_posix(),
                "error": None,
                "operation_id": "start",
            }
        ),
    )
    assert result_service.result("study-1", _COMPARISON_ID, digest)["rows"] == 2

    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["rows"] = 3
    atomic_write_json(result_path, tampered)
    with pytest.raises(ResearchError) as drift:
        result_service.result("study-1", _COMPARISON_ID, digest)
    assert drift.value.code == "comparison_result_drift"


def registry_question(service: ComparisonControlService) -> str:
    return next(
        item["question"] for item in service.catalog() if item["id"] == _COMPARISON_ID
    )
