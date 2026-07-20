from __future__ import annotations

import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from fugue.bench.campaigns import (
    CampaignError,
    CampaignService,
    build_experiment_proposal,
)
from fugue.bench.candidates import stable_digest
from fugue.bench.task_authoring import (
    scoring_revision_from_dict,
    task_suite_draft_from_dict,
    task_suite_preview_from_dict,
)
from fugue.research.contracts import (
    RESEARCH_SCHEMA_VERSION,
    TERMINAL_EXPERIMENT_STATES,
    ExperimentDraftV1,
    ExperimentPreviewV1,
    ExperimentRecordV1,
    ResearchError,
    experiment_draft_from_dict,
    experiment_preview_from_dict,
    now,
    sign_preview,
    sign_record,
)
from fugue.research.store import StudyStore

_RUN_TERMINAL = frozenset({"passed", "failed", "cancelled", "interrupted"})


class ResearchService:
    """Governed Study and Experiment façade over Fugue's campaign lifecycle."""

    def __init__(
        self,
        repo_root: Path,
        env_file: Path | None = None,
        *,
        campaign_service: CampaignService | None = None,
        store: StudyStore | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.campaign = campaign_service or CampaignService(self.repo_root, env_file)
        self.store = store or StudyStore(self.repo_root)

    def preview_experiment(
        self, study_id: str, draft: ExperimentDraftV1
    ) -> ExperimentPreviewV1:
        """Validate and estimate an experiment without writing Fugue state."""
        try:
            study = self.store.get_study(study_id)
            draft = experiment_draft_from_dict(draft.to_dict())
            if draft.study_id != study.id or draft.campaign_id != study.campaign_id:
                raise ResearchError(
                    "study_mismatch",
                    "experiment draft does not belong to the requested Study",
                )
            self._validate_parent_refs(study.id, draft)
            catalog = self.campaign.catalog(draft.campaign_id)
            task_preview = None
            plan = None
            blockers: list[str] = []
            estimated_calls: dict[str, int] = {}
            if draft.scoring_revision:
                scoring_revision_from_dict(draft.scoring_revision)
            if draft.task_suite_draft is not None:
                self._validate_inline_selection(catalog, draft)
                self.campaign.validate_proposal(
                    self._proposal(draft, catalog.catalog_digest)
                )
                task_draft = task_suite_draft_from_dict(draft.task_suite_draft)
                if task_draft.stage_id != draft.stage_id:
                    raise ResearchError(
                        "stage_mismatch",
                        "task suite and experiment must use the same campaign stage",
                    )
                task_preview = self.campaign.preview_task_suite(
                    draft.campaign_id, catalog.catalog_digest, task_draft
                )
                blockers.extend(task_preview.failures)
                multiplier = self._inline_coordinate_multiplier(catalog, draft)
                estimated_cells = task_preview.task_count * multiplier
                estimated_calls = {
                    key: value * multiplier
                    for key, value in task_preview.estimated_calls.items()
                }
            else:
                proposal = self._proposal(draft, catalog.catalog_digest)
                plan = self.campaign.preview(proposal)
                estimated_cells = plan.cell_count
        except CampaignError as exc:
            raise self._research_error(exc) from exc
        record_id = self._record_id(study.id, draft.proposal_id)
        unsigned = ExperimentPreviewV1(
            schema_version=RESEARCH_SCHEMA_VERSION,
            study_id=study.id,
            experiment_id=record_id,
            campaign_id=draft.campaign_id,
            catalog_digest=catalog.catalog_digest,
            policy_digest=catalog.policy_digest,
            draft=draft.to_dict(),
            task_suite_preview=task_preview.to_dict() if task_preview else None,
            plan_receipt=plan.to_dict() if plan else None,
            estimated_cells=estimated_cells,
            estimated_calls=estimated_calls,
            eligible=not blockers,
            blockers=tuple(blockers),
        )
        return sign_preview(unsigned)

    def start_experiment(
        self,
        preview: ExperimentPreviewV1,
        *,
        idempotency_key: str,
    ) -> ExperimentRecordV1:
        """Accept an exact preview at the explicit spend boundary."""
        preview = experiment_preview_from_dict(preview.to_dict())
        if not preview.eligible:
            raise ResearchError(
                "preview_ineligible",
                "an ineligible experiment preview cannot be started",
                category="policy",
                details={"blockers": list(preview.blockers)},
            )
        study = self.store.get_study(preview.study_id)
        draft = experiment_draft_from_dict(preview.draft)
        if study.campaign_id != preview.campaign_id:
            raise ResearchError(
                "study_mismatch", "preview campaign does not match Study"
            )
        try:
            catalog = self.campaign.catalog(preview.campaign_id)
        except CampaignError as exc:
            raise self._research_error(exc) from exc
        if (
            catalog.catalog_digest != preview.catalog_digest
            or catalog.policy_digest != preview.policy_digest
        ):
            raise ResearchError(
                "preview_drift",
                "the accepted preview no longer matches campaign policy or catalog",
                category="policy",
            )
        timestamp = now()
        record = sign_record(
            ExperimentRecordV1(
                schema_version=RESEARCH_SCHEMA_VERSION,
                id=preview.experiment_id,
                study_id=preview.study_id,
                campaign_id=preview.campaign_id,
                state="queued",
                draft=draft.to_dict(),
                preview=preview.to_dict(),
                parent_experiment_ids=draft.parent_experiment_ids,
                proposal=None,
                plan=preview.plan_receipt,
                task_suite_lock=None,
                prepared_plan=None,
                admission=None,
                run_id=None,
                outcome=None,
                evaluation=None,
                analysis=None,
                error=None,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        input_digest = stable_digest(
            {
                "action": "start_experiment",
                "preview_digest": preview.preview_digest,
            }
        )
        return self.store.insert_experiment(
            record,
            operation_id=idempotency_key,
            input_digest=input_digest,
        )

    def cancel_experiment(
        self, experiment_id: str, *, idempotency_key: str, reason: str
    ) -> ExperimentRecordV1:
        record = self.store.get_experiment(experiment_id)
        input_digest = stable_digest(
            {
                "action": "cancel_experiment",
                "experiment_id": experiment_id,
                "reason": reason,
            }
        )
        if record.run_id:
            try:
                self.campaign.cancel(record.run_id, idempotency_key, reason)
            except CampaignError as exc:
                raise self._research_error(exc) from exc
            return self.store.record_cancellation(
                record,
                operation_id=idempotency_key,
                input_digest=input_digest,
                prelaunch=False,
            )
        return self.store.record_cancellation(
            record,
            operation_id=idempotency_key,
            input_digest=input_digest,
            prelaunch=True,
        )

    def run_once(self, worker_id: str | None = None) -> ExperimentRecordV1 | None:
        worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        record = self.store.claim_experiment(worker_id)
        if record is None:
            return None
        try:
            return self._advance(record, worker_id)
        except CampaignError as exc:
            current = self.store.get_experiment(record.id)
            if current.state in TERMINAL_EXPERIMENT_STATES:
                return current
            return self._fail(current, worker_id, self._research_error(exc))
        except ResearchError as exc:
            current = self.store.get_experiment(record.id)
            if current.state in TERMINAL_EXPERIMENT_STATES:
                return current
            return self._fail(current, worker_id, exc)
        except (OSError, RuntimeError, ValueError) as exc:
            error = ResearchError(
                "research_worker_failed",
                str(exc) or type(exc).__name__,
                category="execution",
                details={"exception_type": type(exc).__name__},
            )
            current = self.store.get_experiment(record.id)
            if current.state in TERMINAL_EXPERIMENT_STATES:
                return current
            return self._fail(current, worker_id, error)

    def run_until_idle(
        self, worker_id: str | None = None, *, max_steps: int = 1000
    ) -> tuple[ExperimentRecordV1, ...]:
        worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        changed: list[ExperimentRecordV1] = []
        for _ in range(max_steps):
            record = self.run_once(worker_id)
            if record is None:
                break
            changed.append(record)
            if record.state == "running":
                break
        return tuple(changed)

    def _advance(
        self, record: ExperimentRecordV1, worker_id: str
    ) -> ExperimentRecordV1:
        draft = experiment_draft_from_dict(record.draft)
        preview = experiment_preview_from_dict(record.preview)
        if record.state == "cancelling":
            return self._reconcile_run(record, worker_id, cancelled=True)
        if record.run_id and record.outcome is None:
            return self._reconcile_run(record, worker_id)

        task_suite_digest = draft.task_suite_digest
        if draft.task_suite_draft is not None and record.task_suite_lock is None:
            task_preview = task_suite_preview_from_dict(
                preview.task_suite_preview or {}
            )
            lock = self.campaign.lock_task_suite(
                task_preview, self._operation(record.id, "lock-task-suite")
            )
            record = self._save(
                record,
                worker_id=worker_id,
                state="planning",
                task_suite_lock=lock.to_dict(),
                event_type="task_suite_locked",
                message="Inline tasks were validated and locked as immutable assets.",
                artifact_type="TaskSuiteLockV1",
                artifact_digest=lock.suite_digest,
            )
            task_suite_digest = lock.suite_digest
        elif record.task_suite_lock:
            task_suite_digest = str(record.task_suite_lock["suite_digest"])

        if record.proposal is None:
            proposal = self._proposal(
                draft,
                preview.catalog_digest,
                task_suite_digest=task_suite_digest,
            )
            record = self._save(
                record,
                worker_id=worker_id,
                state="planning",
                proposal=proposal.to_dict(),
                event_type="proposal_compiled",
                message="The accepted research draft was compiled into a campaign proposal.",
                artifact_type="ExperimentProposalV1",
                artifact_digest=proposal.proposal_digest,
            )
        else:
            proposal = self._proposal(
                draft,
                preview.catalog_digest,
                task_suite_digest=task_suite_digest,
            )

        if record.plan is None or draft.task_suite_draft is not None:
            plan = self.campaign.preview(proposal)
            if plan.cell_count != preview.estimated_cells:
                raise ResearchError(
                    "preview_size_drift",
                    "the locked experiment does not match the accepted cell estimate",
                    category="policy",
                    details={
                        "accepted_cells": preview.estimated_cells,
                        "resolved_cells": plan.cell_count,
                    },
                )
            if record.plan != plan.to_dict():
                record = self._save(
                    record,
                    worker_id=worker_id,
                    state="preparing",
                    plan=plan.to_dict(),
                    event_type="plan_resolved",
                    message="The exact Fugue run plan was resolved without starting work.",
                    artifact_type="PlanReceiptV1",
                    artifact_digest=plan.plan_digest,
                )
        else:
            plan = self.campaign.preview(proposal)
            if plan.to_dict() != record.plan:
                raise ResearchError(
                    "plan_drift",
                    "the accepted exact run plan changed before execution",
                    category="policy",
                )

        if record.prepared_plan is None:
            prepared = self.campaign.prepare(
                plan, self._operation(record.id, "prepare")
            )
            record = self._save(
                record,
                worker_id=worker_id,
                state="admitting",
                prepared_plan=prepared.to_dict(),
                event_type="plan_prepared",
                message="Runtimes and routes were prepared and locked before admission.",
                artifact_type="PreparedPlanV1",
                artifact_digest=prepared.prepared_plan_digest,
            )
        else:
            from fugue.bench.campaigns import prepared_plan_from_dict

            prepared = prepared_plan_from_dict(record.prepared_plan)

        if record.admission is None:
            admission = self.campaign.admit(
                prepared, self._operation(record.id, "admit")
            )
            record = self._save(
                record,
                worker_id=worker_id,
                state="launching",
                admission=admission.to_dict(),
                event_type="plan_admitted",
                message="Campaign policy admitted the exact prepared plan and reserved spend.",
                artifact_type="AdmissionReceiptV1",
                artifact_digest=admission.admission_digest,
            )
        else:
            from fugue.bench.campaigns import admission_receipt_from_dict

            admission = admission_receipt_from_dict(record.admission)

        status = self.campaign.launch(admission, self._operation(record.id, "launch"))
        run_id = self._run_id(status.to_dict(), draft.proposal_id)
        return self._save(
            record,
            worker_id=worker_id,
            state="running",
            run_id=run_id,
            event_type="run_started",
            message="The admitted plan is executing through Fugue's canonical runner.",
            release=True,
        )

    def _reconcile_run(
        self,
        record: ExperimentRecordV1,
        worker_id: str,
        *,
        cancelled: bool = False,
    ) -> ExperimentRecordV1:
        assert record.run_id is not None
        status = self.campaign.status(record.run_id).to_dict()
        run_state = self._run_state(status, record.run_id)
        if run_state not in _RUN_TERMINAL:
            self.store.release_lease(record.id, worker_id)
            return record
        outcome = self.campaign.finalize(
            record.run_id, self._operation(record.id, "finalize")
        )
        record = self._save(
            record,
            worker_id=worker_id,
            state="scoring" if record.draft.get("scoring_revision") else "analyzing",
            outcome=outcome.to_dict(),
            event_type="evidence_finalized",
            message="Terminal rows and evidence were reconciled into an immutable outcome.",
            artifact_type="OutcomePacketV1",
            artifact_digest=outcome.outcome_digest,
        )
        if not outcome.eligible:
            return self._save(
                record,
                worker_id=worker_id,
                state="blocked",
                error=ResearchError(
                    "outcome_ineligible",
                    "the run finished but its evidence is not eligible for interpretation",
                    category="evidence",
                    details={
                        "eligibility_failures": list(outcome.eligibility_failures)
                    },
                ).to_dict(),
                event_type="experiment_blocked",
                message="Run evidence failed the campaign eligibility contract.",
                release=True,
            )
        if cancelled or run_state in {"cancelled", "interrupted"}:
            terminal_state = (
                "cancelled" if cancelled or run_state == "cancelled" else "interrupted"
            )
            return self._save(
                record,
                worker_id=worker_id,
                state=terminal_state,
                event_type=f"experiment_{terminal_state}",
                message="The stopped run was reconciled into terminal evidence.",
                release=True,
            )
        draft = experiment_draft_from_dict(record.draft)
        evaluation = None
        if draft.scoring_revision:
            task_digest = draft.task_suite_digest or str(
                (record.task_suite_lock or {}).get("suite_digest") or ""
            )
            if not task_digest:
                raise ResearchError(
                    "scoring_without_task_suite",
                    "authored scoring requires a locked task suite",
                )
            revision = scoring_revision_from_dict(draft.scoring_revision)
            evaluation = self.campaign.score_task_suite(
                record.run_id,
                task_digest,
                revision,
                self._operation(record.id, "score"),
            )
            record = self._save(
                record,
                worker_id=worker_id,
                state="analyzing",
                evaluation=evaluation.to_dict(),
                event_type="evaluation_scored",
                message="Authored criteria were scored against normalized run evidence.",
                artifact_type="TaskEvaluationV1",
                artifact_digest=evaluation.evaluation_digest,
            )
        if draft.task_analysis_id:
            analysis = self.campaign.analyze_task_study(
                record.run_id,
                draft.task_analysis_id,
                self._operation(record.id, "analyze"),
                evaluation_digest=(
                    evaluation.evaluation_digest if evaluation else None
                ),
            )
            record = self._save(
                record,
                worker_id=worker_id,
                state="completed",
                analysis=analysis.to_dict(),
                event_type="analysis_completed",
                message="The registered task study analysis completed.",
                artifact_type="TaskStudyAnalysisV1",
                artifact_digest=analysis.analysis_digest,
                release=True,
            )
        else:
            record = self._save(
                record,
                worker_id=worker_id,
                state="completed",
                event_type="experiment_completed",
                message=(
                    "Cancellation reconciled into terminal evidence."
                    if cancelled
                    else "Experiment completed with immutable outcome evidence."
                ),
                release=True,
            )
        return record

    def _proposal(
        self,
        draft: ExperimentDraftV1,
        catalog_digest: str,
        *,
        task_suite_digest: str | None = None,
    ) -> Any:
        return build_experiment_proposal(
            proposal_id=draft.proposal_id,
            campaign_id=draft.campaign_id,
            catalog_digest=catalog_digest,
            stage_id=draft.stage_id,
            research_question=draft.question,
            hypothesis=draft.hypothesis,
            fixed_dimensions=draft.fixed_dimensions,
            varied_dimensions=draft.varied_dimensions,
            measured_dimensions=draft.measured_dimensions,
            experiment_id=draft.experiment_id,
            model=draft.model,
            n_attempts=draft.n_attempts,
            n_concurrent=draft.n_concurrent,
            preset_id=draft.preset_id,
            workloads=draft.workloads,
            harnesses=draft.harnesses,
            context_systems=draft.context_systems,
            variants=draft.variants,
            n_tasks=draft.n_tasks,
            trace_content=draft.trace_content,
            task_suite_digest=task_suite_digest or draft.task_suite_digest,
            analysis_ids=draft.analysis_ids,
            parent_outcome_id=draft.parent_outcome_id,
            decision_rationale=draft.decision_rationale,
        )

    def _save(
        self,
        record: ExperimentRecordV1,
        *,
        state: str,
        event_type: str,
        message: str,
        worker_id: str | None = None,
        artifact_type: str | None = None,
        artifact_digest: str | None = None,
        release: bool = False,
        **changes: Any,
    ) -> ExperimentRecordV1:
        updated = sign_record(replace(record, state=state, updated_at=now(), **changes))
        return self.store.update_experiment(
            updated,
            worker_id=worker_id,
            event_type=event_type,
            message=message,
            artifact_type=artifact_type,
            artifact_digest=artifact_digest,
            release=release,
        )

    def _fail(
        self, record: ExperimentRecordV1, worker_id: str, error: ResearchError
    ) -> ExperimentRecordV1:
        terminal = "interrupted" if error.category == "execution" else "blocked"
        return self._save(
            record,
            worker_id=worker_id,
            state=terminal,
            error=error.to_dict(),
            event_type="experiment_interrupted"
            if terminal == "interrupted"
            else "experiment_blocked",
            message=str(error),
            release=True,
        )

    def _validate_parent_refs(self, study_id: str, draft: ExperimentDraftV1) -> None:
        if self._record_id(study_id, draft.proposal_id) in draft.parent_experiment_ids:
            raise ResearchError("lineage_cycle", "an experiment cannot parent itself")
        known = {item.id for item in self.store.list_experiments(study_id)}
        missing = set(draft.parent_experiment_ids) - known
        if missing:
            raise ResearchError(
                "unknown_parent",
                f"parent experiments are not in the Study: {', '.join(sorted(missing))}",
            )

    @staticmethod
    def _validate_inline_selection(catalog: Any, draft: ExperimentDraftV1) -> None:
        experiments = {
            str(item.get("id")): item for item in getattr(catalog, "experiments", ())
        }
        experiment = experiments.get(draft.experiment_id)
        if experiment is None:
            raise ResearchError(
                "unregistered_component",
                f"campaign does not expose experiment {draft.experiment_id}",
            )
        if draft.preset_id is not None:
            raise ResearchError(
                "task_suite_preset_unsupported",
                "authored task suites cannot inherit a registered preset",
            )
        if draft.workloads not in {(), ("harbor",)}:
            raise ResearchError(
                "task_suite_workload_unsupported",
                "authored task suites execute through the Harbor workload",
            )
        allowed = {
            "model": {str(item.get("id")) for item in catalog.models},
            "harness": set(catalog.harnesses),
            "context system": {str(item.get("id")) for item in catalog.context_systems},
            "analysis": {str(item.get("id")) for item in catalog.analyses},
            "workload": {
                str(item.get("id")) for item in experiment.get("workloads", [])
            },
            "variant": {str(item.get("id")) for item in experiment.get("variants", [])},
            "preset": {str(item.get("id")) for item in experiment.get("presets", [])},
        }
        selections = {
            "model": (draft.model,),
            "harness": draft.harnesses,
            "context system": draft.context_systems,
            "analysis": draft.analysis_ids,
            "workload": () if draft.workloads == ("harbor",) else draft.workloads,
            "variant": draft.variants,
            "preset": (draft.preset_id,) if draft.preset_id else (),
        }
        for kind, values in selections.items():
            missing = set(values) - allowed[kind]
            if missing:
                raise ResearchError(
                    "unregistered_component",
                    f"campaign does not expose {kind}: {', '.join(sorted(missing))}",
                )

    @staticmethod
    def _inline_coordinate_multiplier(catalog: Any, draft: ExperimentDraftV1) -> int:
        experiment = next(
            item
            for item in catalog.experiments
            if item.get("id") == draft.experiment_id
        )
        harnesses = draft.harnesses or tuple(experiment.get("harnesses", ()))
        variants = [
            item
            for item in experiment.get("variants", [])
            if item.get("enabled", True)
            and (not draft.variants or item.get("id") in draft.variants)
            and (
                not draft.context_systems
                or item.get("context_system_id") in draft.context_systems
            )
        ]
        if not harnesses or not variants:
            raise ResearchError(
                "empty_plan",
                "the selected authored-task coordinates contain no harness or variant",
            )
        return len(harnesses) * len(variants) * draft.n_attempts

    @staticmethod
    def _record_id(study_id: str, proposal_id: str) -> str:
        return f"{study_id}.{proposal_id}"

    @staticmethod
    def _operation(experiment_id: str, stage: str) -> str:
        return f"research-{experiment_id}-{stage}"

    @staticmethod
    def _run_id(status: dict[str, Any], proposal_id: str) -> str:
        matches = [
            str(item.get("run_id") or "")
            for item in status.get("runs", [])
            if item.get("proposal_id") == proposal_id and item.get("run_id")
        ]
        if len(matches) != 1:
            raise ResearchError(
                "run_identity_missing",
                "launch did not reconcile to exactly one run identity",
                category="evidence",
            )
        return matches[0]

    @staticmethod
    def _run_state(status: dict[str, Any], run_id: str) -> str:
        for item in status.get("runs", []):
            if item.get("run_id") == run_id:
                return str(item.get("status") or "unknown")
        raise ResearchError(
            "run_identity_missing",
            "campaign status no longer contains the experiment run",
            category="evidence",
        )

    @staticmethod
    def _research_error(error: CampaignError) -> ResearchError:
        return ResearchError(
            error.code,
            str(error),
            category=error.category,
            retryable=error.retryable,
            details=error.details,
        )


class ResearchWorker:
    def __init__(
        self,
        service: ResearchService,
        *,
        worker_id: str | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        self.service = service
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        self.poll_interval = max(0.05, poll_interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> ResearchWorker:
        if self._thread and self._thread.is_alive():
            return self
        self._thread = threading.Thread(
            target=self.run_forever,
            name=f"fugue-research-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def run_forever(self) -> None:
        while not self._stop.is_set():
            record = self.service.run_once(self.worker_id)
            if record is None or record.state == "running":
                self._stop.wait(self.poll_interval)


class ExperimentHandle:
    def __init__(self, service: ResearchService, experiment_id: str) -> None:
        self.service = service
        self.id = experiment_id

    def status(self) -> ExperimentRecordV1:
        return self.service.store.get_experiment(self.id)

    def events(self, *, after: int = 0) -> tuple[Any, ...]:
        return self.service.store.events(self.id, after=after)

    def watch(self, *, after: int = 0, poll_interval: float = 1.0) -> Any:
        cursor = after
        while True:
            events = self.events(after=cursor)
            for event in events:
                cursor = event.sequence
                yield event
            if self.status().state in TERMINAL_EXPERIMENT_STATES and not events:
                return
            time.sleep(max(0.05, poll_interval))

    def wait(
        self, *, timeout: float | None = None, poll_interval: float = 1.0
    ) -> ExperimentRecordV1:
        started = time.monotonic()
        while True:
            record = self.status()
            if record.state in TERMINAL_EXPERIMENT_STATES:
                return record
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(f"experiment {self.id} did not finish in time")
            time.sleep(max(0.05, poll_interval))

    def cancel(self, *, idempotency_key: str, reason: str) -> ExperimentRecordV1:
        return self.service.cancel_experiment(
            self.id, idempotency_key=idempotency_key, reason=reason
        )

    def links(self) -> dict[str, Any]:
        record = self.status()
        outcome = record.outcome or {}
        return {
            "proposal": record.proposal,
            "plan": record.plan,
            "task_suite_lock": record.task_suite_lock,
            "prepared_plan": record.prepared_plan,
            "admission": record.admission,
            "run_id": record.run_id,
            "outcome_id": outcome.get("outcome_id"),
            "rows": outcome.get("row_refs", []),
            "evidence": outcome.get("evidence_refs", []),
            "analyses": outcome.get("analysis_results", []),
            "evaluation": record.evaluation,
            "task_analysis": record.analysis,
        }

    def result(self) -> dict[str, Any]:
        record = self.status()
        if record.state not in TERMINAL_EXPERIMENT_STATES:
            raise ResearchError(
                "experiment_not_terminal", "experiment does not have a final result"
            )
        if record.outcome is None:
            raise ResearchError(
                "outcome_unavailable",
                "experiment ended before a trustworthy run outcome was produced",
                details={"state": record.state, "error": record.error},
            )
        return {
            "experiment_id": record.id,
            "state": record.state,
            "run_refs": [
                {
                    "kind": "run",
                    "ref": record.run_id,
                    "digest": record.outcome.get("run_snapshot_sha256"),
                }
            ],
            "outcome": record.outcome,
            "evaluation": record.evaluation,
            "analysis": record.analysis,
        }
