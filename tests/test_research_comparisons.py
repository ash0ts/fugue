from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fugue.bench.comparison as comparison_module
from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    ComparisonPublicationError,
    ComparisonResultV1,
    analyze_comparison_rows,
    execute_comparison,
    load_comparison,
    materialize_comparison,
    preview_comparison,
    scaffold_comparison,
    write_comparison_result,
)
from fugue.bench.files import atomic_write_json
from fugue.model_plane import trace_destination_identity
from fugue.research.approvals import ApprovalLedger
from fugue.research.comparisons import (
    COMPARISON_RESULT_ROOT,
    ComparisonControlService,
    ComparisonRegistry,
    _project_result,
    _signed_state,
    append_comparison_invalidation,
    project_direct_comparison_result,
    project_direct_comparison_start,
)
from fugue.research.contracts import ResearchError
from fugue.research.experiment_views import experiment_view_from_dict
from fugue.research.store import StudyStore

_COMPARISON_ID = "source-use-skill-v1"
_DIRECT_COMPARISON = Path("examples/comparisons/source-use-replay/comparison.yaml")


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
    rows = (
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
    )
    result = analyze_comparison_rows(
        comparison_id=_COMPARISON_ID,
        preview_digest=digest,
        source="wandb-artifact://entity/project/comparison:v1",
        rows=rows,
    )
    result_dir = tmp_path / COMPARISON_RESULT_ROOT / digest
    result_dir.mkdir(parents=True)
    (result_dir / "attempts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result_path, _ = write_comparison_result(result, destination=result_dir)
    _project_result(store, "study-1", result, result_path, tmp_path)
    study = store.get_study("study-1")
    assert study.results == ()
    assert study.resources[-1].digest == result.result_digest
    assert store.ensure_result_projection_events() == 0
    [evaluation] = [
        item
        for item in store.research_log_events()
        if item.summary.get("experiment_view", {}).get("kind") == "evaluation"
    ]
    view = evaluation.summary["experiment_view"]
    assert evaluation.study_id == f"comparison-{digest[:20]}"
    assert view["matrix_size"] == 2
    assert view["infrastructure_health"] == "unavailable"
    assert view["evidence_eligible"] is False
    assert view["behavioral_summary"]["improved_pairs"] == 1
    assert len(view["paired_cases"]) == 1
    assert not {
        "cells",
        "arm_totals",
        "aligned_comparisons",
        "behavioral_measures",
        "mechanism_funnel",
        "outcome_summaries",
        "score_summaries",
    } & view.keys()
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


def test_v1_result_projection_remains_backward_compatible(tmp_path: Path) -> None:
    store = StudyStore(tmp_path)
    store.create_study(
        study_id="legacy-study",
        title="Legacy comparison",
        campaign_id="legacy-comparison",
        question="Does the legacy comparison remain readable?",
        operation_id="create-legacy-study",
    )
    result_path = tmp_path / "legacy-result.json"
    result_path.write_text("{}\n", encoding="utf-8")
    result = ComparisonResultV1(
        schema_version=1,
        comparison_id="legacy-comparison",
        preview_digest="a" * 64,
        source="legacy-run",
        evidence_project=None,
        rows=2,
        baseline_passed=0,
        candidate_passed=1,
        improved=1,
        regressed=0,
        unchanged=0,
        incomplete=0,
        required_evaluations_incomplete=0,
        deterministic_summary={},
        judge_summary={},
        mechanism_summary={},
        operational_summary={},
        evidence_links=(),
        paired_cases=(),
        limitations=(),
        result_digest="b" * 64,
    )

    _project_result(store, "legacy-study", result, result_path, tmp_path)

    study = store.get_study("legacy-study")
    assert len(study.results) == 1
    assert study.results[0].id == f"comparison-{result.result_digest[:20]}"
    assert store.ensure_result_projection_events() == 1
    evaluations = [
        item
        for item in store.research_log_events()
        if item.summary.get("experiment_view", {}).get("kind") == "evaluation"
    ]
    assert len(evaluations) == 1
    assert evaluations[0].summary["experiment_view"]["schema_version"] == 1


def test_direct_comparison_projection_is_idempotent_and_preserves_v2_pairs(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    research_id = "configured-harbor-research"
    spec = load_comparison(_DIRECT_COMPARISON, repo_root=source_root)
    spec = replace(
        spec,
        spec_digest="",
        execution=replace(
            spec.execution,
            research_id=research_id,
            approval_required=False,
            evidence_project="wandb/projection-test",
        ),
    )
    preview = preview_comparison(spec, repo_root=source_root)
    expected_experiment_id = spec.id

    started = project_direct_comparison_start(tmp_path, research_id, preview)
    repeated_start = project_direct_comparison_start(
        tmp_path,
        research_id,
        preview,
    )

    assert repeated_start == started
    assert started["research_id"] == research_id
    assert started["experiment_id"] == expected_experiment_id

    rows = [
        _direct_attempt(variant="baseline", passed=False),
        _direct_attempt(variant="candidate", passed=True),
    ]
    result = analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=rows,
        source="local-run-1",
        expected_evidence_project="wandb/projection-test",
    )
    result_dir = tmp_path / COMPARISON_RESULT_ROOT / preview.preview_digest
    result_dir.mkdir(parents=True)
    (result_dir / "attempts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    result_path, _ = write_comparison_result(
        result,
        destination=result_dir,
    )

    projected = project_direct_comparison_result(
        tmp_path,
        research_id,
        result,
        result_path,
    )
    repeated_result = project_direct_comparison_result(
        tmp_path,
        research_id,
        result,
        result_path,
    )

    assert repeated_result == projected
    assert projected["research_id"] == research_id
    assert projected["experiment_id"] == expected_experiment_id
    store = StudyStore(tmp_path)
    study = store.get_study(research_id)
    assert study.results == ()
    assert store.ensure_result_projection_events() == 0
    assert len(
        [resource for resource in study.resources if resource.kind == "comparison_result"]
    ) == 1
    view_events = [
        event
        for event in store.research_log_events()
        if event.research_id == research_id
        and event.study_id == expected_experiment_id
        and event.summary.get("experiment_view")
    ]
    assert len(view_events) == 3
    assert {event.summary["experiment_view"]["kind"] for event in view_events} == {
        "design",
        "progress",
        "evaluation",
    }
    evaluation = next(
        event.summary["experiment_view"]
        for event in view_events
        if event.summary["experiment_view"]["kind"] == "evaluation"
    )
    assert evaluation["schema_version"] == 2
    assert evaluation["behavioral_summary"]["status"] == "improved"
    [projected_pair] = evaluation["paired_cases"]
    assert projected_pair["pair_id"] == result.paired_cases[0].pair_id
    assert projected_pair["task_id"] == "task-1"
    assert projected_pair["status"] == "improved"
    assert not {
        "baseline_passed",
        "candidate_passed",
        "baseline_prediction_id",
        "candidate_prediction_id",
        "baseline_evaluation_call_id",
        "candidate_evaluation_call_id",
    } & projected_pair.keys()
    assert projected_pair["dimension_changes"] == [
        {
            "id": "task.correct",
            "status": "improved",
            "baseline": False,
            "candidate": True,
            "critical": True,
        }
    ]
    for arm in ("baseline", "candidate"):
        attempt = projected_pair[arm]
        assert attempt["attempt_id"] == getattr(
            result.paired_cases[0], arm
        ).attempt_id
        assert attempt["identity"]["arm"] == arm
        assert {
            (link["kind"], link["status"])
            for link in attempt["evidence_links"]
        } == {
            ("evaluation_root", "resolved"),
            ("prediction_and_score", "resolved"),
            ("prediction", "resolved"),
            ("agent_root", "resolved"),
            ("dataset", "resolved"),
        }
        assert attempt["infrastructure"]["label_boundary_verified"] is True
        assert "private_label_boundary_verified" not in attempt["infrastructure"]
    assert not {
        "cells",
        "arm_totals",
        "aligned_comparisons",
        "behavioral_measures",
        "mechanism_funnel",
        "outcome_summaries",
        "score_summaries",
    } & evaluation.keys()


def test_execute_comparison_surfaces_result_projection_failure_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "comparison-repo"
    comparison_path = scaffold_comparison(repo_root)
    spec = load_comparison(comparison_path, repo_root=repo_root)
    research_id = "configured-projection-failure"
    spec = replace(
        spec,
        spec_digest="",
        execution=replace(
            spec.execution,
            research_id=research_id,
            approval_required=False,
            evidence_project="wandb/projection-test",
        ),
    )
    spec = comparison_module.comparison_from_dict(
        spec.to_dict(),
        repo_root=repo_root,
        source=repo_root,
    )
    preview = preview_comparison(spec, repo_root=repo_root)
    materialized = materialize_comparison(preview, repo_root=repo_root)
    preview_operator = comparison_module.OperatorService(repo_root)
    approved = materialized[1].approved_comparison
    rows = []
    for cell in approved["expected_cells"]:
        variant = str(cell["variant_id"])
        row = _direct_attempt(
            variant=variant,
            passed=variant == "candidate",
            call_suffix=f"t{cell['trial_index']}",
        )
        row.update(
            {
                "task_id": cell["task_id"],
                "harness": cell["harness"],
                "trial_index": cell["trial_index"],
                "attempt_id": cell["attempt_id"],
                "candidate_id": cell["candidate_id"],
                "execution_fingerprint": cell["execution_fingerprint"],
                "applicable": cell["applicable"],
                "skip_reason": cell["skip_reason"],
                "run_id": "projection-failure-run",
                "integration_provenance_digest": cell[
                    "integration_provenance_digest"
                ],
                "approved_comparison": approved,
            }
        )
        rows.append(row)

    class FakeOperator:
        def __init__(self, _repo_root: Path, _env_file: Path | None) -> None:
            self.env = dict(preview_operator.env)

        def prepare(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def preview_experiment(self, *args: Any, **kwargs: Any) -> Any:
            return preview_operator.preview_experiment(*args, **kwargs)

        def execute_run(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def export_run(
            self,
            _run_id: str,
            *,
            out: Path,
            **_kwargs: Any,
        ) -> SimpleNamespace:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows)
                + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(path=out)

    monkeypatch.setattr(comparison_module, "OperatorService", FakeOperator)
    monkeypatch.setattr(
        comparison_module,
        "materialize_comparison",
        lambda *_args, **_kwargs: materialized,
    )
    monkeypatch.setattr(
        "fugue.bench.execution.new_run_id",
        lambda: "projection-failure-run",
    )
    projection_calls: list[str] = []

    def pass_start(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        projection_calls.append("start")
        return {
            "schema_version": 1,
            "research_id": research_id,
            "experiment_id": "comparison-test",
            "design_event_digest": "d" * 64,
            "progress_event_digest": "p" * 64,
        }

    def fail_result(*_args: Any, **_kwargs: Any) -> None:
        projection_calls.append("result")
        raise RuntimeError("presentation sink unavailable")

    monkeypatch.setattr(
        "fugue.research.comparisons.project_direct_comparison_start",
        pass_start,
    )
    monkeypatch.setattr(
        "fugue.research.comparisons.project_direct_comparison_result",
        fail_result,
    )

    with pytest.raises(ComparisonPublicationError) as raised:
        execute_comparison(
            preview,
            approval_digest="",
            repo_root=repo_root,
            fetch_weave=False,
        )

    assert projection_calls == ["start", "result"]
    assert raised.value.stage == "result"
    assert raised.value.result is not None
    assert raised.value.result_path is not None
    result = raised.value.result
    result_path = raised.value.result_path
    assert result.improved == 2
    assert result.regressed == 0
    assert result.candidate_passed == 2
    assert result.baseline_passed == 0
    assert result.behavioral_summary.status == "improved"
    assert json.loads(result_path.read_text())["result_digest"] == (
        result.result_digest
    )
    publication = json.loads(
        (result_path.parent / "research-publication.json").read_text()
    )
    assert publication == {
        "schema_version": 1,
        "research_id": research_id,
        "experiment_id": "comparison-test",
        "design_event_digest": "d" * 64,
        "progress_event_digest": "p" * 64,
        "publication_complete": False,
        "status": "publication_incomplete",
        "stage": "result",
        "error_type": "RuntimeError",
        "result_digest": result.result_digest,
        "result": result_path.relative_to(repo_root).as_posix(),
    }
    latest = json.loads(
        (repo_root / comparison_module.COMPARISON_RESULT_ROOT / "latest.json").read_text()
    )
    assert latest["research_publication"] == {
        "status": "publication_incomplete",
        "complete": False,
        "receipt": raised.value.receipt_path.relative_to(repo_root).as_posix(),
    }


def test_execute_comparison_requires_research_start_before_preparation_or_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "comparison-repo"
    comparison_path = scaffold_comparison(repo_root)
    spec = load_comparison(comparison_path, repo_root=repo_root)
    research_id = "configured-start-gate"
    spec = replace(
        spec,
        spec_digest="",
        execution=replace(
            spec.execution,
            research_id=research_id,
            approval_required=False,
        ),
    )
    spec = comparison_module.comparison_from_dict(
        spec.to_dict(),
        repo_root=repo_root,
        source=repo_root,
    )
    preview = preview_comparison(spec, repo_root=repo_root)
    preview_operator = comparison_module.OperatorService(repo_root)
    operator_calls: list[str] = []

    class FakeOperator:
        def __init__(self, _repo_root: Path, _env_file: Path | None) -> None:
            self.env = dict(preview_operator.env)

        def prepare(self, *_args: Any, **_kwargs: Any) -> None:
            operator_calls.append("prepare")

        def preview_experiment(self, *args: Any, **kwargs: Any) -> Any:
            return preview_operator.preview_experiment(*args, **kwargs)

        def execute_run(self, *_args: Any, **_kwargs: Any) -> None:
            operator_calls.append("execute")

        def export_run(self, *_args: Any, **_kwargs: Any) -> Any:
            operator_calls.append("export")
            raise AssertionError("export must not run before Research start")

    monkeypatch.setattr(comparison_module, "OperatorService", FakeOperator)
    monkeypatch.setattr(
        comparison_module,
        "materialize_comparison",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        "fugue.research.comparisons.project_direct_comparison_start",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("presentation sink unavailable")
        ),
    )

    with pytest.raises(ComparisonPublicationError) as raised:
        execute_comparison(
            preview,
            approval_digest="",
            repo_root=repo_root,
            fetch_weave=False,
        )

    assert raised.value.stage == "start"
    assert raised.value.result is None
    assert operator_calls == []
    assert json.loads(raised.value.receipt_path.read_text()) == {
        "schema_version": 1,
        "research_id": research_id,
        "publication_complete": False,
        "status": "publication_incomplete",
        "stage": "start",
        "error_type": "RuntimeError",
    }
    assert not (
        repo_root / comparison_module.COMPARISON_RESULT_ROOT / "latest.json"
    ).exists()


def test_historical_invalidation_appends_without_rewriting_result(
    tmp_path: Path,
) -> None:
    store = StudyStore(tmp_path)
    store.create_study(
        study_id="study-1",
        title="Release evidence",
        campaign_id="release-evidence",
        question="Should the package ship?",
        operation_id="create-study",
    )
    result_path = tmp_path / ".fugue/results/historical/result.json"
    attempts_path = result_path.with_name("attempts.jsonl")
    result_path.parent.mkdir(parents=True)
    historical = {
        "schema_version": 1,
        "comparison_id": "historical-release",
        "preview_digest": "a" * 64,
        "source": "historical-run",
        "rows": 2,
        "baseline_passed": 0,
        "candidate_passed": 0,
        "improved": 0,
        "regressed": 0,
        "unchanged": 1,
        "incomplete": 0,
        "required_evaluations_incomplete": 0,
        "deterministic_summary": {
            "baseline": {
                "dimensions": {
                    "answer_present": {"passed": 0, "evaluated": 1}
                }
            },
            "candidate": {
                "dimensions": {
                    "answer_present": {"passed": 0, "evaluated": 1}
                }
            },
        },
        "judge_summary": {"status": "not_used"},
        "mechanism_summary": {},
        "operational_summary": {"observed_cost_usd": 1.25},
        "evidence_links": [
            {
                "label": "Agent root — scope — baseline",
                "url": "https://wandb.ai/entity/project/r/call/not-authoritative",
            }
        ],
        "paired_cases": [
            {
                "task_id": "scope",
                "harness": "claude-code",
                "attempt": 1,
                "baseline_passed": False,
                "candidate_passed": False,
                "baseline_prediction_id": "baseline-prediction",
                "candidate_prediction_id": "candidate-prediction",
                "status": "unchanged",
            }
        ],
        "limitations": [],
        "result_digest": "b" * 64,
    }
    result_path.write_text(json.dumps(historical), encoding="utf-8")
    rows = [
        {
            "variant_id": variant,
            "agent_response": "A terminal answer exists.",
        }
        for variant in ("baseline", "candidate")
    ]
    attempts_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    correction_path = tmp_path / "invalidation.json"
    correction = {
        "schema_version": 1,
        "research_id": "study-1",
        "experiment_id": "historical-release-view",
        "comparison_id": "historical-release",
        "preview_digest": "a" * 64,
        "superseded_candidate_sha": "c" * 40,
        "result": {
            "path": result_path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        },
        "attempts": {
            "path": attempts_path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(attempts_path.read_bytes()).hexdigest(),
        },
        "corrections": {
            "stored_answer_present": {"passed": 0, "evaluated": 2},
            "recomputed_answer_present": {"passed": 2, "evaluated": 2},
            "behavioral_outcome": "invalid",
            "critical_scope_regression": {},
        },
        "mechanism_evidence_only": {"tool_call_reduction": 1},
        "decision": {
            "status": "invalid",
            "recommendation": "INVALID — do not use this release result.",
            "evidence_grade": "invalid",
            "critical_blockers": ["result mismatch"],
            "next_action": "Create a new Study.",
        },
    }
    correction_path.write_text(json.dumps(correction), encoding="utf-8")
    original_result = result_path.read_bytes()
    correction_digest = stable_digest(correction)
    legacy_producer_event_id = (
        "fugue:study-1:historical-release-view:"
        "comparison-invalidation-view-v2-r3-"
        f"{correction_digest}"
    )
    legacy_event = store.record_experiment_view_event(
        research_id="study-1",
        experiment_id="historical-release-view",
        producer_event_id=legacy_producer_event_id,
        classification="limitation",
        state="failed",
        message="Earlier invalidation projection retained legacy outcome fields.",
        progress={"completed": 2, "total": 2},
        observed_cost_usd=1.25,
        view=experiment_view_from_dict(
            {
                "schema_version": 1,
                "kind": "evaluation",
                "matrix_size": 2,
                "phase": "completed",
                "completed_cells": 2,
                "infrastructure_health": "unavailable",
                "evidence_eligible": False,
                "limitations": ["Legacy invalidation projection."],
                "decision": {"status": "invalid"},
            }
        ),
    )

    receipt = append_comparison_invalidation(
        tmp_path,
        correction_path,
        research_id="study-1",
    )
    repeated = append_comparison_invalidation(
        tmp_path,
        correction_path,
        research_id="study-1",
    )

    assert receipt == repeated
    assert receipt["status"] == "invalid"
    assert receipt["projection_revision"] == 4
    assert receipt["producer_event_id"] != legacy_producer_event_id
    assert result_path.read_bytes() == original_result
    study = store.get_study("study-1")
    assert study.notes[-1].kind == "integrity_correction"
    events = [
        item
        for item in store.research_log_events()
        if item.study_id == "historical-release-view"
    ]
    assert len(events) == 2
    assert events[0].event_digest == legacy_event.event_digest
    event = events[-1]
    assert event.producer_event_id == receipt["producer_event_id"]
    assert event.state == "failed"
    expected_view = json.loads(
        (
            Path(__file__).parent
            / "fixtures/experiment-view-v2-invalid-comparison.json"
        ).read_text(encoding="utf-8")
    )
    assert event.summary["experiment_view"] == expected_view
    assert "cells" not in expected_view
    assert "state_counts" not in expected_view
    assert "supported_claim" not in expected_view["behavioral_summary"]


def registry_question(service: ComparisonControlService) -> str:
    return next(
        item["question"] for item in service.catalog() if item["id"] == _COMPARISON_ID
    )


def _direct_attempt(
    *,
    variant: str,
    passed: bool,
    call_suffix: str = "",
) -> dict[str, Any]:
    call = f"{variant}{f'-{call_suffix}' if call_suffix else ''}-call"
    base = "https://wandb.ai/wandb/projection-test/weave"
    return {
        "task_id": "task-1",
        "task_name": "Reconcile one Evaluation",
        "harness": "claude-code",
        "trial_index": 1,
        "variant_id": variant,
        "status": "passed",
        "prediction_id": f"{variant}-prediction-row",
        "pass": passed,
        "comparison_evaluation_status": "scored",
        "comparison_required_evaluation_complete": True,
        "comparison_deterministic_scores": {"task.correct": passed},
        "comparison_deterministic_criticality": {"task.correct": True},
        "trace_project": "wandb/projection-test",
        "trace_receipt": trace_destination_identity(
            {"FUGUE_WEAVE_PROJECT": "wandb/projection-test"}
        ),
        "evaluation_id": f"{variant}-evaluation",
        "weave_evaluation_root_call_id": f"{call}-evaluation",
        "weave_evaluation_root_ref": (
            f"weave:///wandb/projection-test/call/{call}-evaluation"
        ),
        "evaluation_url": f"{base}/calls/{call}-evaluation",
        "evaluation_root_object_verified": True,
        "evaluation_root_dataset_relationship_verified": True,
        "evaluation_root_prediction_relationship_verified": True,
        "dataset_id": (
            "weave:///wandb/projection-test/object/projection-dataset:v1"
        ),
        "dataset_url": f"{base}/datasets/projection-dataset",
        "dataset_version_object_verified": True,
        "eval_predict_and_score_call_id": f"{call}-predict-and-score",
        "eval_predict_and_score_ref": (
            "weave:///wandb/projection-test/call/"
            f"{call}-predict-and-score"
        ),
        "eval_predict_and_score_url": f"{base}/calls/{call}-predict-and-score",
        "eval_predict_and_score_object_verified": True,
        "weave_prediction_call_id": f"{call}-prediction",
        "weave_prediction_ref": (
            f"weave:///wandb/projection-test/call/{call}-prediction"
        ),
        "weave_prediction_url": f"{base}/calls/{call}-prediction",
        "weave_prediction_object_verified": True,
        "prediction_child_relationship_verified": True,
        "evaluation_prediction_graph_verified": True,
        "weave_agent_root_call_id": f"{call}-agent",
        "weave_agent_root_ref": (
            f"weave:///wandb/projection-test/call/{call}-agent"
        ),
        "weave_agent_url": f"{base}/calls/{call}-agent",
        "trace_link_status": "linked",
        "agent_graph_verified": True,
        "otel_root_span_id": f"otel-{call}",
        "execution_fingerprint": f"{variant}-execution",
        "runtime_lock_digest": f"{variant}-runtime",
        "private_label_boundary_verified": True,
    }
