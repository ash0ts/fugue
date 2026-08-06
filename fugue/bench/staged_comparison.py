"""Durable staged execution for an immutable comparison preview.

This module owns stage admission, recovery, budget leasing, row reconciliation,
and host-only finalization.  The comparison module remains the canonical owner
of spec compilation, scoring, analysis, and publication; those operations are
resolved lazily at execution time to avoid a second comparison path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fugue.bench.analysis_contracts import (
    EvidenceDriftCheckV1,
    evidence_drift_check_from_dict,
)
from fugue.bench.campaign_accounting import BudgetLeaseLedger
from fugue.bench.campaign_store import CampaignStore, read_json_object
from fugue.bench.execution import ExecutionScheduleV1, StageSubsetReceiptV1
from fugue.bench.files import latest_jsonl_records
from fugue.bench.library import validate_id
from fugue.model_plane import trace_project_slug
from fugue.research.approvals import ApprovalLedger
from fugue.research.store import StudyStore

if TYPE_CHECKING:
    from fugue.bench.comparison import (
        ComparisonPreviewV1,
        ComparisonResult,
        ComparisonSpecV1,
    )
    from fugue.bench.host_capacity import HostCapacityProbe
    from fugue.bench.operator import OperatorService


class StagedFinalizationPending(RuntimeError):
    """Host-only evidence may be retried without another Agent execution."""


def _comparison_module() -> Any:
    # Deliberately lazy: comparison.py re-exports this module's public surface.
    # It also retains the established monkeypatch seams used by recovery tests.
    from fugue.bench import comparison

    return comparison


def _persist_controller(
    runtime: CampaignStore,
    comparison_id: str,
    state_path: Path,
    *,
    stage_id: str | None = None,
    stage: Mapping[str, Any] | None = None,
    **updates: Any,
) -> dict[str, Any]:
    with runtime.campaign_lock(comparison_id):
        state = read_json_object(state_path)
        if stage is not None:
            state["stages"][stage_id].update(stage)
        state.update(updates)
        runtime.write_json(state_path, state)
    return state


def finalize_staged_comparison(
    *,
    preview: ComparisonPreviewV1,
    spec: ComparisonSpecV1,
    service: OperatorService,
    request: Any,
    rows: Sequence[Mapping[str, Any]],
    controller_id: str,
    repo_root: Path,
    readiness: Mapping[str, Any],
    source_pre_run_drift: EvidenceDriftCheckV1 | None,
    source_checkpoint_drift: EvidenceDriftCheckV1 | None,
    fetch_weave: bool,
    publish_research: bool,
) -> tuple[ComparisonResult, Path, Path]:
    """Finalize terminal staged rows through the canonical analysis pipeline."""

    comparison = _comparison_module()
    destination = repo_root / comparison.COMPARISON_RESULT_ROOT / preview.preview_digest
    approved = comparison._verified_approved_inputs(
        request.approved_comparison,
        repo_root=repo_root,
    )
    infrastructure = comparison._bound_execution_infrastructure_receipt(
        spec,
        readiness=readiness,
        repo_root=repo_root,
    )
    release_coverage = comparison._bound_v3_release_note_coverage(
        spec,
        readiness=readiness,
        repo_root=repo_root,
    )
    source_post_run_drift = comparison._verify_v3_source_drift(
        spec,
        readiness=readiness,
        repo_root=repo_root,
        env=service.env,
    )
    if source_post_run_drift is not None and source_post_run_drift.status != "matched":
        raise ValueError("immutable source evidence changed during staged execution")
    expected_project = trace_project_slug(
        comparison._comparison_evidence_environment(spec, service.env)
    )
    mutable_rows = [dict(row) for row in rows]
    for row in mutable_rows:
        if str(row.get("trace_project") or "") != expected_project:
            raise ValueError("staged row routed to the wrong evidence project")
        links = comparison._attempt_evidence_links(row)
        if any(link.status == "invalid" for link in links):
            raise ValueError("staged row has an invalid evidence identity or link")
        if any(link.status == "missing" for link in links):
            raise StagedFinalizationPending(
                "required Weave evidence is not yet resolvable"
            )
    scored = comparison._score_and_bind_exported_comparison_rows(
        spec=spec,
        rows=mutable_rows,
        repo_root=repo_root,
        env=service.env,
        approved_comparison=request.approved_comparison,
        source_pre_run_drift=source_pre_run_drift,
        source_checkpoint_drift=source_checkpoint_drift,
        source_post_run_drift=source_post_run_drift,
        release_note_coverage=release_coverage,
        infrastructure_receipt=infrastructure,
    )
    study_intent = comparison._comparison_study_intent(spec)
    draft = comparison.analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=scored,
        source=controller_id,
        expected_evidence_project=expected_project,
        approved_comparison=request.approved_comparison,
        decision_policy=spec.decision_policy,
        expected_source_evidence_project=spec.execution.source_evidence_project,
        result_schema_version=3 if spec.schema_version >= 3 else 2,
        study_intent=study_intent,
        release_note_coverage=release_coverage,
        supersedes=spec.supersedes,
    )
    payloads: dict[str, Any] = {"result": draft.to_dict()}
    if publish_research and spec.execution.research_id:
        from fugue.research.experiment_views import build_comparison_evaluation_view

        payloads["study"] = build_comparison_evaluation_view(
            draft.to_dict(),
            result_ref=(destination.resolve() / "result.json")
            .relative_to(repo_root.resolve())
            .as_posix(),
        ).to_dict()
    from fugue.bench.run_conformance import write_hosted_evidence_privacy_receipt

    privacy = write_hosted_evidence_privacy_receipt(
        repo_root=repo_root,
        run_id=controller_id,
        rows=scored,
        env=service.env,
        evidence_project=expected_project,
        private_labels_path=comparison._frozen_private_labels_path(
            repo_root,
            str(approved["private_labels_sha256"]),
        ),
        publication_payloads=payloads,
        fetch_hosted=fetch_weave,
    )
    if privacy.status == "unavailable":
        raise StagedFinalizationPending(
            "hosted evidence/privacy reconciliation is not yet complete"
        )
    if privacy.status != "passed":
        raise ValueError("hosted evidence privacy reconciliation failed")
    comparison._apply_hosted_evidence_privacy(
        scored,
        repo_root=repo_root,
        run_id=controller_id,
        receipt_digest=privacy.sha256,
    )
    attempts_path = destination / "attempts.jsonl"
    comparison._atomic_text(attempts_path, comparison._jsonl(scored))
    result = comparison.analyze_comparison_rows(
        comparison_id=spec.id,
        preview_digest=preview.preview_digest,
        rows=scored,
        source=controller_id,
        expected_evidence_project=expected_project,
        approved_comparison=request.approved_comparison,
        decision_policy=spec.decision_policy,
        expected_source_evidence_project=spec.execution.source_evidence_project,
        result_schema_version=3 if spec.schema_version >= 3 else 2,
        study_intent=study_intent,
        release_note_coverage=release_coverage,
        supersedes=spec.supersedes,
    )
    json_path, markdown_path = comparison.write_comparison_result(
        result, destination=destination
    )
    comparison._publish_comparison_result(
        spec=spec,
        result=result,
        json_path=json_path,
        markdown_path=markdown_path,
        destination=destination,
        publication_path=destination / "research-publication.json",
        projection=None,
        publish_research=publish_research,
        repo_root=repo_root,
    )
    return result, json_path, markdown_path


def execute_comparison_stage(  # noqa: C901 - durable stage transaction
    preview: ComparisonPreviewV1,
    *,
    stage_id: str,
    approval_digest: str,
    repo_root: Path,
    env_file: Path | None = None,
    fetch_weave: bool = True,
    controller_id: str | None = None,
    publish_research: bool = True,
    host_capacity_probe: HostCapacityProbe | None = None,
) -> dict[str, Any]:
    """Execute/resume one approved subset without replaying canonical cells."""

    comparison = _comparison_module()
    comparison._verify_artifact(
        preview.to_dict(), "preview_digest", "comparison preview"
    )
    spec = comparison.comparison_from_dict(
        preview.comparison, repo_root=repo_root, source=repo_root
    )
    service = comparison.OperatorService(repo_root, env_file)
    capacity_receipt = comparison._preview_host_capacity_receipt(preview)
    if capacity_receipt is not None:
        comparison.verify_host_capacity_receipt(
            capacity_receipt,
            repo_root,
            probe=host_capacity_probe,
        )
    current = comparison.preview_comparison(
        spec,
        repo_root=repo_root,
        operator=service,
        host_capacity_receipt=capacity_receipt,
    )
    if current.preview_digest != preview.preview_digest:
        raise ValueError("comparison inputs changed after the staged preview")
    comparison._require_execution_judge_calibrations(spec, repo_root=repo_root)
    if current.readiness["status"] not in {"ready", "needs_review"}:
        raise ValueError("comparison is no longer execution-ready")
    source_pre_run_drift = comparison._verify_v3_source_drift(
        spec,
        readiness=current.readiness,
        repo_root=repo_root,
        env=service.env,
    )
    if source_pre_run_drift is not None and source_pre_run_drift.status != "matched":
        raise RuntimeError("immutable source evidence changed before stage admission")
    subset = comparison.comparison_stage_receipt(preview, stage_id)
    schedule = ExecutionScheduleV1.from_dict(preview.execution_schedule)
    controller_id = validate_id(
        controller_id or f"comparison-{preview.preview_digest[:20]}",
        kind="comparison controller id",
    )
    stage_subject = f"{controller_id}-{stage_id}"
    ledger = ApprovalLedger(StudyStore(repo_root).path)
    runtime = CampaignStore(
        repo_root.resolve(), Path(".fugue/runtime/comparison-stages")
    )
    state_path = runtime.campaign_dir(spec.id) / f"{controller_id}.json"
    persist = partial(
        _persist_controller, runtime, spec.id, state_path, stage_id=stage_id
    )
    with runtime.campaign_lock(spec.id):
        state = (
            read_json_object(state_path)
            if state_path.exists()
            else {
                "schema_version": 1,
                "controller_id": controller_id,
                "comparison_id": spec.id,
                "preview_digest": preview.preview_digest,
                "schedule_digest": schedule.schedule_digest,
                "stages": {item: {"status": "pending"} for item in schedule.stages},
                "source_pre_run_drift": (
                    source_pre_run_drift.to_dict()
                    if source_pre_run_drift is not None
                    else None
                ),
                "fatal_blockers": [],
            }
        )
        if (
            state.get("schema_version") != 1
            or state.get("controller_id") != controller_id
            or state.get("comparison_id") != spec.id
            or state.get("preview_digest") != preview.preview_digest
            or state.get("schedule_digest") != schedule.schedule_digest
        ):
            raise ValueError("staged controller belongs to another exact preview")
        stages_state = comparison._mapping(
            state.get("stages"), "staged controller stages"
        )
        if set(stages_state) != set(schedule.stages):
            raise ValueError("staged controller stage manifest changed")
        for recorded_stage_id, recorded_stage in stages_state.items():
            value = comparison._mapping(recorded_stage, "staged controller stage")
            if value.get("status") not in {
                "pending",
                "running",
                "admission_paused",
                "finalization_pending",
                "complete",
                "fatal",
            }:
                raise ValueError(
                    f"staged controller stage {recorded_stage_id!r} has invalid state"
                )
        completed = {
            key
            for key, value in stages_state.items()
            if isinstance(value, Mapping) and value.get("status") == "complete"
        }
        ordered = list(schedule.stages)
        missing_predecessors = [
            item for item in ordered[: ordered.index(stage_id)] if item not in completed
        ]
        if missing_predecessors:
            raise ValueError(
                "staged execution requires completed predecessor(s): "
                + ", ".join(missing_predecessors)
            )
        if state.get("fatal_blockers"):
            raise RuntimeError("staged controller is blocked by fatal evidence")
        for predecessor_id in ordered[: ordered.index(stage_id)]:
            predecessor = comparison._mapping(
                stages_state.get(predecessor_id), "staged predecessor state"
            )
            predecessor_subset = comparison.comparison_stage_receipt(
                preview, predecessor_id
            )
            predecessor_subject = f"{controller_id}-{predecessor_id}"
            predecessor_approval = str(predecessor.get("approval_digest") or "")
            authorization = comparison._verify_stage_authorization_receipt(
                predecessor.get("authorization"),
                subset=predecessor_subset,
                subject_id=predecessor_subject,
                approval_digest=predecessor_approval,
            )
            claimed_predecessor = ledger.require_claimed_by(
                approval_digest=predecessor_approval,
                subject_kind="experiment",
                preview_digest=predecessor_subset.subset_digest,
                subject_id=predecessor_subject,
            )
            if claimed_predecessor.to_dict() != authorization["approval"]:
                raise ValueError("stored predecessor approval receipt changed")
        prior = comparison._mapping(
            stages_state.get(stage_id), "staged controller stage"
        )
        finalization_only = bool(
            prior
            and prior.get("status") == "complete"
            and stage_id == ordered[-1]
            and not state.get("result_digest")
        )
        stored_authorization = prior.get("authorization")
        if prior.get("status") != "pending":
            authorization = comparison._verify_stage_authorization_receipt(
                stored_authorization,
                subset=subset,
                subject_id=stage_subject,
                approval_digest=approval_digest,
            )
            claimed = ledger.require_claimed_by(
                approval_digest=approval_digest,
                subject_kind="experiment",
                preview_digest=subset.subset_digest,
                subject_id=stage_subject,
            )
            if claimed.to_dict() != authorization["approval"]:
                raise ValueError("stored stage approval receipt changed")
        else:
            claimed = ledger.claim_stage(
                approval_digest=approval_digest,
                subset=subset,
                subject_id=stage_subject,
            )
            authorization = comparison._stage_authorization_receipt(
                subset=subset,
                subject_id=stage_subject,
                approval=claimed,
            )
        if prior.get("status") == "complete" and not finalization_only:
            return comparison._staged_status(state, stage_id, state_path, repo_root)
        child_run_id = str(
            prior.get("run_id")
            or f"{controller_id}-{stage_id}-{subset.subset_digest[:8]}"
        )
        if not finalization_only:
            state["stages"][stage_id] = {
                "status": "running",
                "run_id": child_run_id,
                "subset": subset.to_dict(),
                "approval_digest": approval_digest,
                "authorization": authorization,
            }
            runtime.write_json(state_path, state)

    experiment, request = comparison.materialize_comparison(
        preview,
        repo_root=repo_root,
        operator=service,
        # Stage approvals authorize admission in the host-only controller state.
        # The per-cell lock keeps the unchanged full-preview identity.
        approval_digest="",
    )
    finalize = partial(
        finalize_staged_controller,
        preview=preview,
        spec=spec,
        service=service,
        request=request,
        schedule=schedule,
        runtime=runtime,
        state_path=state_path,
        stage_id=stage_id,
        repo_root=repo_root,
        readiness=current.readiness,
        source_pre_run_drift=source_pre_run_drift,
        fetch_weave=fetch_weave,
        publish_research=publish_research,
    )
    budget = BudgetLeaseLedger(
        runtime.budget_ledger(spec.id, controller_id),
        maximum_total_cost_usd=schedule.total_cost_usd,
        maximum_in_flight_cost_usd=schedule.maximum_in_flight_cost_usd,
        maximum_in_flight_executions=schedule.worker_limit,
        maximum_physical_executions=schedule.maximum_physical_executions,
    )
    coordination_budget = None
    if schedule.coordination is not None:
        coordination = schedule.coordination
        coordination_budget = BudgetLeaseLedger(
            repo_root
            / ".fugue/runtime/comparison-stages/coordination"
            / f"{coordination['group_id']}.json",
            maximum_total_cost_usd=float(coordination["total_cost_usd"]),
            maximum_in_flight_cost_usd=float(
                coordination["maximum_in_flight_cost_usd"]
            ),
            maximum_in_flight_executions=int(coordination["worker_limit"]),
            maximum_physical_executions=int(
                coordination["maximum_physical_executions"]
            ),
        )
    reserve = schedule.per_execution_cost_usd["total"]
    existing_run = None
    try:
        existing_run = service.run_summary(child_run_id)
    except FileNotFoundError:
        pass
    prior_fatal, prior_pending = (
        comparison._staged_run_gate(existing_run, repo_root=repo_root)
        if existing_run is not None
        else ([], [])
    )
    if prior_fatal:
        state = persist(stage={"status": "fatal"}, fatal_blockers=prior_fatal)
        return comparison._staged_status(state, stage_id, state_path, repo_root)
    if not finalization_only and (
        existing_run is None
        or existing_run.status not in {"passed", "failed"}
        or prior_pending
    ):
        admission = schedule.stage_admission[stage_id]
        existing_run = service.execute_run(
            request,
            run_id=child_run_id,
            experiment=experiment,
            selected_attempt_ids=subset.attempt_ids,
            resume=existing_run is not None,
            infrastructure_retries=schedule.infrastructure_retry_limit,
            budget_ledger=budget,
            coordination_budget_ledger=coordination_budget,
            reserved_cost_per_execution_usd=reserve,
            host_scorer_names=comparison._comparison_scorer_names(spec),
            execution_worker_limit=int(admission["worker_limit"]),
            execution_wave_size=int(admission["wave_size"]),
            evidence_checkpoint_cells_override=len(subset.attempt_ids),
        )
    if finalization_only:
        return finalize()
    assert existing_run is not None
    fatal, pending = comparison._staged_run_gate(existing_run, repo_root=repo_root)
    if fatal:
        state = persist(stage={"status": "fatal"}, fatal_blockers=fatal)
        return comparison._staged_status(state, stage_id, state_path, repo_root)
    if pending:
        stage_status = (
            "admission_paused"
            if any("budget admission" in reason for reason in pending)
            else "finalization_pending"
        )
        state = persist(
            stage={"status": stage_status, "pending_reasons": pending},
        )
        return comparison._staged_status(state, stage_id, state_path, repo_root)

    stage_rows = runtime.campaign_dir(spec.id) / "rows" / f"{stage_id}.jsonl"
    exported = service.export_run(
        child_run_id,
        out=stage_rows,
        fetch_weave=fetch_weave,
        to_weave=False,
    )
    try:
        rows = comparison._read_jsonl(exported.path, "staged comparison rows")
    except (FileNotFoundError, ValueError) as exc:
        blocker = f"staged child export is invalid: {type(exc).__name__}: {exc}"
        state = persist(stage={"status": "fatal"}, fatal_blockers=[blocker])
        return comparison._staged_status(state, stage_id, state_path, repo_root)
    comparison._apply_harbor_conformance(rows, repo_root=repo_root, run_id=child_run_id)
    observed_attempts = {str(row.get("attempt_id") or "") for row in rows}
    if len(rows) != len(observed_attempts) or observed_attempts != set(
        subset.attempt_ids
    ):
        raise RuntimeError("staged export does not match its logical subset")
    comparison._atomic_text(stage_rows, comparison._jsonl(rows))
    try:
        comparison._validate_staged_rows(
            rows,
            expected_project=trace_project_slug(
                comparison._comparison_evidence_environment(spec, service.env)
            ),
        )
    except StagedFinalizationPending as exc:
        pending.append(str(exc))
    except Exception as exc:
        state = persist(
            stage={"status": "fatal"},
            fatal_blockers=[f"{type(exc).__name__}: {exc}"],
        )
        return comparison._staged_status(state, stage_id, state_path, repo_root)
    source_checkpoint_drift = None
    if source_pre_run_drift is not None:
        source_checkpoint_drift = comparison._verify_v3_source_drift(
            spec,
            readiness=current.readiness,
            repo_root=repo_root,
            env=service.env,
        )
        if (
            source_checkpoint_drift is None
            or source_checkpoint_drift.status != "matched"
        ):
            state = persist(
                stage={"status": "fatal"},
                fatal_blockers=[
                    "immutable source evidence changed at the stage checkpoint"
                ],
            )
            return comparison._staged_status(state, stage_id, state_path, repo_root)
    state = persist(
        stage={
            "status": "finalization_pending" if pending else "complete",
            "rows_path": stage_rows.relative_to(repo_root).as_posix(),
            "rows_digest": comparison._sha256_path(stage_rows),
            "pending_reasons": pending,
            "source_checkpoint_drift": (
                source_checkpoint_drift.to_dict()
                if source_checkpoint_drift is not None
                else None
            ),
        },
    )
    if pending:
        return comparison._staged_status(state, stage_id, state_path, repo_root)
    return finalize()


def staged_run_gate(run: Any, *, repo_root: Path) -> tuple[list[str], list[str]]:
    """Separate behavioral failures from fatal or host-repairable failures."""

    records = latest_jsonl_records(
        repo_root / ".fugue/runtime" / run.run_id / "cells.jsonl", "cell_id"
    )
    fatal_kinds = {
        "configuration_failure",
        "evidence_initialization_failure",
        "runner_start_failure",
        "admission_aborted",
        "prestart_cancelled",
        "cancelled",
    }
    fatal = [
        f"{row.get('cell_id')}: {row.get('terminal_kind')}"
        for row in records
        if row.get("terminal_kind") in fatal_kinds
    ]
    pending: list[str] = []
    for row in records:
        if row.get("terminal_kind") == "admission_paused":
            pending.append(
                f"{row.get('cell_id')}: budget admission awaits measured cost"
            )
            continue
        if row.get("terminal_kind") != "evidence_failure":
            continue
        message = str(row.get("error") or "").lower()
        if any(
            marker in message
            for marker in (
                "wrong project",
                "cross-project",
                "identity mismatch",
                "unrelated call",
                "privacy",
                "secret",
                "duplicate",
                "cleanup failed",
                "evidence destination receipt",
                "five-link evidence",
                "authoritatively verified",
                "agent evidence call was not linked",
            )
        ):
            fatal.append(f"{row.get('cell_id')}: fatal evidence mismatch")
        else:
            pending.append(f"{row.get('cell_id')}: host evidence finalization pending")
    for failure in getattr(run, "evaluation_failures", ()):
        message = str(failure).lower()
        if any(
            marker in message
            for marker in (
                "wrong project",
                "cross-project",
                "privacy",
                "identity",
                "five-link evidence",
                "trace link",
                "cleanup",
            )
        ):
            fatal.append(str(failure))
        else:
            pending.append(str(failure))
    if run.status == "interrupted":
        pending.append(f"child run {run.run_id} is resumable after interruption")
    elif run.status == "cancelled" and not pending:
        fatal.append(f"child run {run.run_id} ended {run.status}")
    return sorted(set(fatal)), sorted(set(pending))


def validate_staged_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_project: str
) -> None:
    """Fail closed on routing, evidence, cleanup, or privacy disagreement."""

    comparison = _comparison_module()
    for row in rows:
        if str(row.get("trace_project") or "") != expected_project:
            raise ValueError("staged row routed to the wrong evidence project")
        links = comparison._attempt_evidence_links(row)
        if any(link.status == "invalid" for link in links):
            raise ValueError("staged row contains an invalid evidence relationship")
        if any(link.status == "missing" for link in links):
            raise StagedFinalizationPending(
                "required Weave relationships are not yet resolvable"
            )
        if row.get("harbor_conformance_status") == "failed":
            raise ValueError("local Harbor conformance or cleanup failed")
        if row.get("harbor_conformance_status") == "unavailable":
            raise StagedFinalizationPending(
                "local Harbor conformance is not yet available"
            )
        if int(row.get("local_artifact_privacy_match_count") or 0):
            raise ValueError("local artifact privacy scan found protected content")


def finalize_staged_controller(  # noqa: PLR0913
    *,
    preview: ComparisonPreviewV1,
    spec: ComparisonSpecV1,
    service: OperatorService,
    request: Any,
    schedule: ExecutionScheduleV1,
    runtime: CampaignStore,
    state_path: Path,
    stage_id: str,
    repo_root: Path,
    readiness: Mapping[str, Any],
    source_pre_run_drift: EvidenceDriftCheckV1 | None,
    fetch_weave: bool,
    publish_research: bool,
) -> dict[str, Any]:
    """Assemble exact stage rows and resume host-only finalization safely."""

    comparison = _comparison_module()
    persist = partial(_persist_controller, runtime, spec.id, state_path)
    with runtime.campaign_lock(spec.id):
        state = read_json_object(state_path)
    if not all(
        isinstance(item, Mapping) and item.get("status") == "complete"
        for item in state["stages"].values()
    ):
        return comparison._staged_status(state, stage_id, state_path, repo_root)
    verify_staged_controller_authorizations(
        state,
        preview=preview,
        schedule=schedule,
        repo_root=repo_root,
    )
    by_attempt: dict[str, dict[str, Any]] = {}
    source_checkpoint_drift: EvidenceDriftCheckV1 | None = None
    for current_stage in schedule.stages:
        stage = comparison._mapping(state["stages"].get(current_stage), "staged state")
        raw_checkpoint = stage.get("source_checkpoint_drift")
        if source_pre_run_drift is not None:
            if not isinstance(raw_checkpoint, Mapping):
                raise ValueError("staged source checkpoint evidence is missing")
            checkpoint = evidence_drift_check_from_dict(raw_checkpoint)
            if checkpoint.status != "matched":
                raise ValueError("staged source checkpoint did not match")
            if (
                source_checkpoint_drift is not None
                and checkpoint != source_checkpoint_drift
            ):
                raise ValueError("staged source checkpoints disagree across stages")
            source_checkpoint_drift = checkpoint
        elif raw_checkpoint is not None:
            raise ValueError("unexpected staged source checkpoint evidence")
        path = repo_root / str(stage.get("rows_path") or "")
        if comparison._sha256_path(path) != str(stage.get("rows_digest") or ""):
            raise ValueError("staged row artifact changed after reconciliation")
        for row in comparison._read_jsonl(path, "staged comparison rows"):
            attempt = str(row.get("attempt_id") or "")
            if not attempt or attempt in by_attempt:
                raise ValueError("staged rows contain a missing or duplicate attempt")
            by_attempt[attempt] = row
    expected = schedule.logical_attempt_ids
    if set(by_attempt) != set(expected):
        raise ValueError("staged rows do not match the immutable logical matrix")
    ordered_rows = [by_attempt[attempt] for attempt in expected]
    try:
        result, result_path, markdown_path = comparison._finalize_staged_comparison(
            preview=preview,
            spec=spec,
            service=service,
            request=request,
            rows=ordered_rows,
            controller_id=str(state["controller_id"]),
            repo_root=repo_root,
            readiness=readiness,
            source_pre_run_drift=source_pre_run_drift,
            source_checkpoint_drift=source_checkpoint_drift,
            fetch_weave=fetch_weave,
            publish_research=publish_research,
        )
    except (comparison.ComparisonPublicationError, StagedFinalizationPending) as exc:
        state = persist(
            finalization={
                "status": "pending",
                "reason": f"{type(exc).__name__}: {exc}",
            },
        )
        return comparison._staged_status(state, stage_id, state_path, repo_root)
    except Exception as exc:
        state = persist(
            fatal_blockers=[f"{type(exc).__name__}: {exc}"],
            finalization={"status": "fatal"},
        )
        return comparison._staged_status(state, stage_id, state_path, repo_root)
    state = persist(
        result_digest=result.result_digest,
        result_path=result_path.relative_to(repo_root).as_posix(),
        markdown_path=markdown_path.relative_to(repo_root).as_posix(),
        finalization={"status": "complete"},
    )
    return comparison._staged_status(state, stage_id, state_path, repo_root)


def verify_staged_controller_authorizations(
    state: Mapping[str, Any],
    *,
    preview: ComparisonPreviewV1,
    schedule: ExecutionScheduleV1,
    repo_root: Path,
) -> None:
    """Reconcile every signed stage approval before canonical finalization."""

    comparison = _comparison_module()
    controller_id = validate_id(
        str(state.get("controller_id") or ""), kind="comparison controller id"
    )
    stages = comparison._mapping(state.get("stages"), "staged controller stages")
    ledger = ApprovalLedger(StudyStore(repo_root).path)
    for stage_id in schedule.stages:
        stage = comparison._mapping(stages.get(stage_id), "staged controller stage")
        subset = comparison.comparison_stage_receipt(preview, stage_id)
        subject_id = f"{controller_id}-{stage_id}"
        approval_digest = str(stage.get("approval_digest") or "")
        authorization = comparison._verify_stage_authorization_receipt(
            stage.get("authorization"),
            subset=subset,
            subject_id=subject_id,
            approval_digest=approval_digest,
        )
        if (
            StageSubsetReceiptV1.from_dict(
                comparison._mapping(stage.get("subset"), "stored stage subset")
            )
            != subset
        ):
            raise ValueError("stored stage subset changed before finalization")
        approval = ledger.require_claimed_by(
            approval_digest=approval_digest,
            subject_kind="experiment",
            preview_digest=subset.subset_digest,
            subject_id=subject_id,
        )
        if approval.to_dict() != authorization["approval"]:
            raise ValueError("stored stage approval changed before finalization")


def staged_status(
    state: Mapping[str, Any], stage_id: str, state_path: Path, repo_root: Path
) -> dict[str, Any]:
    """Return the canonical persisted controller status."""

    comparison = _comparison_module()
    stages = comparison._mapping(state.get("stages"), "staged controller stages")
    stages_complete = bool(stages) and all(
        isinstance(item, Mapping) and item.get("status") == "complete"
        for item in stages.values()
    )
    finalization = comparison._mapping_or_empty(state.get("finalization"))
    complete = bool(
        stages_complete
        and finalization.get("status") == "complete"
        and state.get("result_digest")
    )
    fatal = bool(state.get("fatal_blockers"))
    if complete:
        status = "complete"
    elif fatal:
        status = "fatal"
    elif stages_complete:
        status = "finalization_pending"
    elif stages[stage_id]["status"] == "complete":
        status = "stage_complete"
    else:
        status = str(stages[stage_id]["status"])
    return {
        "schema_version": 1,
        "controller_id": state["controller_id"],
        "comparison_id": state["comparison_id"],
        "preview_digest": state["preview_digest"],
        "schedule_digest": state["schedule_digest"],
        "stage_id": stage_id,
        "status": status,
        "stages": dict(stages),
        "finalization": dict(finalization),
        "fatal_blockers": list(state.get("fatal_blockers") or []),
        "controller_path": state_path.relative_to(repo_root).as_posix(),
        **(
            {
                "result_digest": state["result_digest"],
                "result_path": state["result_path"],
                "markdown_path": state["markdown_path"],
            }
            if complete
            else {}
        ),
    }
