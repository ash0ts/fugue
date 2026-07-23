from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from filelock import FileLock

from fugue.agent_tracing import agent_conversation_id, stable_agent_name
from fugue.bench.candidates import CANDIDATE_IDENTITY_SCHEMA_VERSION
from fugue.bench.evaluations import apply_generated_evaluation
from fugue.bench.execution import CellOutcome, PlannedCell
from fugue.bench.files import atomic_write_json
from fugue.bench.reproducibility import (
    EVALUATION_ASSET_LOCK_NAME,
    read_evaluation_asset_lock,
)
from fugue.bench.scoring import latency_summary, score_evidence_paths
from fugue.model_plane import trace_project_slug
from fugue.redaction import redact_value, secrets_from_env
from fugue.weave_support import WEAVE_AGENTS_BASE_URL, initialize_weave

PREDICTION_SCHEMA_VERSION = 1
PUBLICATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PredictionRowV1:
    prediction_id: str
    run_id: str
    candidate_id: str
    comparison_example_id: str
    trial_index: int
    execution_kind: str
    source_record_type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.payload,
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "record_type": "trial",
            "source_record_type": self.source_record_type,
            "prediction_id": self.prediction_id,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "comparison_example_id": self.comparison_example_id,
            "trial_index": self.trial_index,
            "execution_kind": self.execution_kind,
        }


@dataclass(frozen=True)
class PublishedEvaluation:
    candidate_id: str
    name: str
    examples: int
    url: str | None = None
    evaluation_ref: str | None = None
    model_ref: str | None = None
    agent_predictions: int = 0
    linked_agent_predictions: int = 0
    direct_predictions: int = 0
    linking_failures: tuple[str, ...] = ()
    publication_id: str | None = None
    revision: int = 1
    supersedes: str | None = None
    active: bool = True


@dataclass(frozen=True)
class PublicationResult:
    published: int
    skipped: int
    evaluations: tuple[PublishedEvaluation, ...] = ()
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedExport:
    predictions: tuple[dict[str, Any], ...]
    measurements: tuple[dict[str, Any], ...]
    publication: PublicationResult


@dataclass
class _LiveCandidate:
    candidate: dict[str, Any]
    logger: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _LivePrediction:
    session: _LiveCandidate
    prediction: Any
    row: dict[str, Any]
    opened_monotonic: float


class _TracePollingCancelled(Exception):
    pass


class LiveEvaluationCoordinator:
    """Own live Weave prediction calls while Harbor cells execute."""

    def __init__(
        self,
        cells: list[PlannedCell],
        *,
        repo_root: Path,
        project: str,
        env: Mapping[str, str],
        weave_module: Any | None = None,
        summary_fetcher: Callable[..., dict[str, dict[str, Any]]] | None = None,
        trace_timeout_sec: float | None = None,
        cancellation_event: threading.Event | None = None,
    ) -> None:
        if not env.get("WANDB_API_KEY", "").strip():
            raise RuntimeError("WANDB_API_KEY is required for live evaluations")
        self.repo_root = repo_root
        self.project = project
        self.env = dict(env)
        self.run_id = cells[0].run_id if cells else "unknown"
        self.run_dir = repo_root / ".fugue" / "runtime" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "evaluations.jsonl"
        self.results_path = self.run_dir / "evaluation-results.jsonl"
        self._event_lock = threading.Lock()
        self._predictions: dict[str, _LivePrediction] = {}
        self._prediction_lock = threading.Lock()
        self._terminal_cells: set[str] = set()
        self._cancellation_event = cancellation_event or threading.Event()
        self._summary_fetcher = summary_fetcher or fetch_weave_summaries
        configured_timeout = self.env.get("FUGUE_WEAVE_LINK_TIMEOUT_SEC")
        self.trace_timeout_sec = (
            trace_timeout_sec
            if trace_timeout_sec is not None
            else float(configured_timeout or 45)
        )
        self.weave = weave_module or initialize_weave(project, env)
        logger_cls = getattr(self.weave, "EvaluationLogger", None)
        if logger_cls is None:
            raise RuntimeError("installed weave package has no EvaluationLogger")
        dataset_cls = getattr(self.weave, "Dataset", None)
        planned = [
            _planned_evaluation_row(cell)
            for cell in cells
            if cell.applicable and cell.execution_kind == "agent"
        ]
        candidates = _publication_candidates(planned)
        datasets: dict[str, Any] = {}
        self._sessions_by_cell: dict[str, _LiveCandidate] = {}
        self._inputs_by_cell: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            scope_id = candidate["evaluation_scope_id"]
            if scope_id not in datasets:
                datasets[scope_id] = (
                    dataset_cls(
                        name=_dataset_name(candidate),
                        rows=candidate["dataset_examples"],
                    )
                    if dataset_cls is not None
                    else candidate["dataset_examples"]
                )
            attributes = getattr(self.weave, "attributes", None)
            context = (
                attributes(_evaluation_run_attributes(candidate))
                if attributes is not None
                else nullcontext()
            )
            with context:
                logger = logger_cls(
                    name=_evaluation_name(candidate),
                    model=_evaluation_model(candidate),
                    dataset=datasets[scope_id],
                    eval_attributes=_evaluation_scope_attributes(candidate),
                    scorers=candidate["scorers"],
                )
            session = _LiveCandidate(candidate=candidate, logger=logger)
            for row, inputs in zip(
                candidate["rows"], candidate["prediction_inputs"], strict=True
            ):
                cell_id = str(row["cell_id"])
                self._sessions_by_cell[cell_id] = session
                self._inputs_by_cell[cell_id] = inputs
                self._append_event(
                    "pending",
                    cell_id=cell_id,
                    candidate_id=candidate["candidate_id"],
                    evaluation_scope_id=scope_id,
                )
        self._unique_sessions = tuple(
            {id(value): value for value in self._sessions_by_cell.values()}.values()
        )

    def begin_cell(self, cell: PlannedCell) -> Mapping[str, str] | None:
        session = self._sessions_by_cell.get(cell.id)
        if session is None:
            return None
        with session.lock:
            prediction = session.logger.log_prediction(
                inputs=self._inputs_by_cell[cell.id]
            )
            prediction.__enter__()
        with self._prediction_lock:
            self._predictions[cell.id] = _LivePrediction(
                session=session,
                prediction=prediction,
                row=_planned_evaluation_row(cell),
                opened_monotonic=time.monotonic(),
            )
        call = prediction.predict_and_score_call
        call_id = str(call.id)
        self._append_event(
            "prediction_open",
            cell_id=cell.id,
            candidate_id=cell.candidate_id,
            eval_predict_and_score_call_id=call_id,
        )
        return {
            "FUGUE_WEAVE_EVAL_PREDICT_AND_SCORE_CALL_ID": call_id,
            "FUGUE_WEAVE_EVAL_PROJECT_ID": str(call.project_id),
            "FUGUE_WEAVE_EVAL_NAME": _evaluation_name(session.candidate),
            "FUGUE_EVALUATION_SCOPE_ID": session.candidate["evaluation_scope_id"],
        }

    def finish_cell(self, cell: PlannedCell, outcome: CellOutcome) -> None:
        with self._prediction_lock:
            active = self._predictions.get(cell.id)
        if active is None:
            return
        row = _completed_evaluation_row(cell, outcome, active.row)
        _merge_error_events(row)
        row["evaluation_publication_mode"] = "live"
        row["evaluation_prediction_latency_sec"] = max(
            time.monotonic() - active.opened_monotonic, 0.0
        )
        call_id = str(active.prediction.predict_and_score_call.id)
        if outcome.status in {"cancelled", "interrupted"}:
            self._cancel_prediction(
                cell.id,
                active,
                row,
                reason=outcome.error or "Run cancelled by the operator.",
            )
            return
        owns_prediction = False
        try:
            if row.get("agent_execution_status") == "not_started":
                _mark_agent_execution_not_started(row)
            else:
                _apply_trace_summary(row, self._wait_for_trace(row))
            self._raise_if_cancelled()
            _merge_error_events(row)
            _apply_observed_identity(row)
            root = (
                None
                if row.get("trace_link_status") in {"not_applicable", "not_started"}
                else _verified_evaluation_root(row, call_id)
            )
            if root is not None:
                _attach_genai_span_ref(
                    active.prediction.predict_and_score_call,
                    trace_id=str(root["trace_id"]),
                    span_id=str(root["span_id"]),
                )
                row["trace_link_status"] = "linked"
                row["trace_link_error"] = None
                self._append_event(
                    "trace_linked",
                    cell_id=cell.id,
                    candidate_id=cell.candidate_id,
                    observed_conversation_id=row.get("observed_conversation_id"),
                    trace_id=root.get("trace_id"),
                    root_span_id=root.get("span_id"),
                    eval_predict_and_score_call_id=call_id,
                )
            if cell.evaluation_case is not None:
                try:
                    apply_generated_evaluation(
                        row,
                        case=cell.evaluation_case,
                        rubrics=cell.evaluation_rubrics,
                        judge_model=str(cell.env.get("FUGUE_JUDGE_MODEL") or ""),
                        env=self.env,
                        trial_dir=Path(
                            str(row.get("trial_dir") or cell.result_path.parent)
                        ),
                    )
                except Exception as exc:
                    row["evaluation_judge_status"] = "failed"
                    row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
            self._raise_if_cancelled()
            _set_adapter_outcome(row)
            if not self._pop_prediction(cell.id, active):
                return
            owns_prediction = True
            active.prediction.output = _evaluation_output(row)
            with active.session.lock:
                for name, value in _evaluation_scores(row).items():
                    active.prediction.log_score(name, value)
                active.prediction.__exit__(None, None, None)
                active.session.rows.append(row)
            self._terminal_cells.add(cell.id)
            self._append_result(row)
            self._append_event(
                "finalized",
                cell_id=cell.id,
                candidate_id=cell.candidate_id,
                trace_link_status=row.get("trace_link_status"),
                eval_predict_and_score_call_id=call_id,
            )
        except _TracePollingCancelled:
            self._cancel_prediction(
                cell.id,
                active,
                row,
                reason="Run cancelled by the operator.",
            )
        except Exception as exc:
            if not owns_prediction and not self._pop_prediction(cell.id, active):
                return
            row["trace_link_status"] = "failed"
            row["trace_link_error"] = f"{type(exc).__name__}: {exc}"
            _set_adapter_outcome(row)
            try:
                active.prediction.output = _evaluation_output(row)
                with active.session.lock:
                    for name, value in _evaluation_scores(row).items():
                        active.prediction.log_score(name, value)
                    active.prediction.__exit__(None, None, None)
                    active.session.rows.append(row)
                self._terminal_cells.add(cell.id)
                self._append_result(row)
            finally:
                self._append_event(
                    "failed",
                    cell_id=cell.id,
                    candidate_id=cell.candidate_id,
                    error=row["trace_link_error"],
                    eval_predict_and_score_call_id=call_id,
                )

    def cancel_open_predictions(self, reason: str) -> None:
        self._cancellation_event.set()
        with self._prediction_lock:
            active_predictions = list(self._predictions.values())
            self._predictions.clear()
        for active in active_predictions:
            row = dict(active.row)
            row["evaluation_publication_mode"] = "live"
            row["evaluation_prediction_latency_sec"] = max(
                time.monotonic() - active.opened_monotonic, 0.0
            )
            self._close_cancelled_prediction(active, row, reason=reason)

    def _cancel_prediction(
        self,
        cell_id: str,
        active: _LivePrediction,
        row: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        if not self._pop_prediction(cell_id, active):
            return
        self._close_cancelled_prediction(active, row, reason=reason)

    def _pop_prediction(self, cell_id: str, active: _LivePrediction) -> bool:
        with self._prediction_lock:
            if self._predictions.get(cell_id) is not active:
                return False
            self._predictions.pop(cell_id)
            return True

    def _raise_if_cancelled(self) -> None:
        if self._cancellation_event.is_set():
            raise _TracePollingCancelled

    def _close_cancelled_prediction(
        self,
        active: _LivePrediction,
        row: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        cell_id = str(row.get("cell_id") or "")
        row.update(
            {
                "status": "cancelled",
                "pass": None,
                "trace_link_status": "cancelled",
                "trace_link_error": None,
                "trace_link_reason": reason,
                "weave_observability_status": "cancelled",
                "weave_usage_status": "unavailable",
                "weave_usage_source": "unavailable",
            }
        )
        active.prediction.output = _evaluation_output(row)
        with active.session.lock:
            active.prediction.__exit__(None, None, None)
            active.session.rows.append(row)
        if cell_id:
            self._terminal_cells.add(cell_id)
        self._append_result(row)
        self._append_event(
            "cancelled",
            cell_id=cell_id or None,
            candidate_id=row.get("candidate_id"),
            error=reason,
            eval_predict_and_score_call_id=str(
                active.prediction.predict_and_score_call.id
            ),
        )

    def finalize(self, *, cancelled: bool = False) -> PublicationResult:
        if cancelled:
            self.cancel_open_predictions("Run cancelled by the operator.")
            error = RuntimeError("Run cancelled by the operator.")
            for session in self._unique_sessions:
                try:
                    with session.lock:
                        session.logger.fail(error)
                except Exception:
                    pass
                for row in session.candidate["rows"]:
                    cell_id = str(row.get("cell_id") or "")
                    if not cell_id or cell_id in self._terminal_cells:
                        continue
                    self._terminal_cells.add(cell_id)
                    self._append_event(
                        "cancelled",
                        cell_id=cell_id,
                        candidate_id=row.get("candidate_id"),
                        error=str(error),
                    )
            return PublicationResult(published=0, skipped=0)
        evaluations: list[PublishedEvaluation] = []
        failures: list[str] = []
        ledger = (
            self.repo_root
            / ".fugue"
            / "runtime"
            / "publications"
            / f"v{PUBLICATION_SCHEMA_VERSION}"
            / _safe_slug(self.project)
        )
        ledger.mkdir(parents=True, exist_ok=True)
        for session in self._unique_sessions:
            candidate_id = session.candidate["candidate_id"]
            try:
                with session.lock:
                    session.logger.log_summary()
            except Exception as exc:
                try:
                    session.logger.fail(exc)
                except Exception:
                    pass
                failures.append(f"{candidate_id}: {type(exc).__name__}: {exc}")
                continue
            completed = _publication_candidates(session.rows)
            if len(completed) != 1:
                failures.append(
                    f"{candidate_id}: live evaluation produced an invalid scope"
                )
                continue
            published = completed[0]
            if (
                published["evaluation_scope_id"]
                != session.candidate["evaluation_scope_id"]
            ):
                failures.append(
                    f"{candidate_id}: evaluation scope changed during execution"
                )
                continue
            url = getattr(session.logger, "ui_url", None)
            evaluation_ref = _logger_ref(session.logger, "_pseudo_evaluation")
            model_ref = _logger_ref(session.logger, "model")
            agent_rows = [
                row
                for row in session.rows
                if row.get("trace_link_status") != "not_applicable"
            ]
            linked = sum(row.get("trace_link_status") == "linked" for row in agent_rows)
            linking_failures = tuple(
                f"{row.get('cell_id')}: {row.get('trace_link_error') or 'Agent root was not linked'}"
                for row in agent_rows
                if row.get("trace_link_status") != "linked"
            )
            if linked != len(agent_rows):
                failures.append(
                    f"{candidate_id}: {len(agent_rows) - linked} prediction(s) "
                    "finished without a verified Agent trace link: "
                    + "; ".join(linking_failures)
                )
            marker = ledger / f"{published['publication_id']}.r1.json"
            _write_publication_marker(
                marker,
                self.project,
                published["publication_id"],
                name=_evaluation_name(session.candidate),
                candidate_id=candidate_id,
                evaluation_scope_id=published["evaluation_scope_id"],
                examples=len(session.rows),
                url=url,
                evaluation_ref=evaluation_ref,
                model_ref=model_ref,
                agent_predictions=len(agent_rows),
                linked_agent_predictions=linked,
                direct_predictions=0,
                linking_failures=linking_failures,
                publication_mode="live",
                publication_schema_version=PUBLICATION_SCHEMA_VERSION,
                revision=1,
                supersedes=None,
                republish_reason=None,
                active=True,
            )
            evaluations.append(
                PublishedEvaluation(
                    candidate_id=candidate_id,
                    name=_evaluation_name(session.candidate),
                    examples=len(session.rows),
                    url=url,
                    evaluation_ref=evaluation_ref,
                    model_ref=model_ref,
                    agent_predictions=len(agent_rows),
                    linked_agent_predictions=linked,
                    direct_predictions=0,
                    linking_failures=linking_failures,
                    publication_id=published["publication_id"],
                )
            )
        return PublicationResult(
            published=len(evaluations),
            skipped=0,
            evaluations=tuple(evaluations),
            failures=tuple(failures),
        )

    def _wait_for_trace(self, row: dict[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + max(self.trace_timeout_sec, 0)
        conversation_ids = list(
            dict.fromkeys(
                str(value)
                for value in [
                    row.get("planned_conversation_id"),
                    row.get("weave_conversation_id"),
                    *(row.get("weave_conversation_ids") or []),
                    *(row.get("native_session_ids") or []),
                ]
                if value
            )
        )
        latest: dict[str, Any] = {}
        while True:
            self._raise_if_cancelled()
            values = self._summary_fetcher(
                run_keys=[str(row["run_key"])],
                conversation_ids_by_run={str(row["run_key"]): conversation_ids},
                project=self.project,
                timeout_sec=min(max(self.trace_timeout_sec, 1), 10),
                env=self.env,
            )
            self._raise_if_cancelled()
            latest = values.get(str(row["run_key"]), {})
            probe = {**row, **latest}
            _apply_observed_identity(probe)
            if probe.get("observed_conversation_id"):
                return latest
            if time.monotonic() >= deadline:
                return latest
            wait_sec = min(2, max(deadline - time.monotonic(), 0))
            if self._cancellation_event.wait(wait_sec):
                raise _TracePollingCancelled

    def _append_event(self, status: str, **values: Any) -> None:
        record = redact_value(
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "status": status,
                "recorded_at": datetime.now(UTC).isoformat(),
                **values,
            }
        )
        with self._event_lock:
            with self.events_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _append_result(self, row: dict[str, Any]) -> None:
        with self._event_lock:
            with self.results_path.open("a") as handle:
                handle.write(
                    json.dumps(redact_value(row), sort_keys=True, default=str) + "\n"
                )


class GeneratedEvaluationCoordinator:
    """Run generated scorers locally when live Weave publication is unavailable."""

    def __init__(
        self,
        cells: list[PlannedCell],
        *,
        repo_root: Path,
        env: Mapping[str, str],
    ) -> None:
        self.repo_root = repo_root
        self.env = dict(env)
        self.path = (
            repo_root
            / ".fugue"
            / "runtime"
            / (cells[0].run_id if cells else "unknown")
            / "evaluation-results.jsonl"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def finish_cell(self, cell: PlannedCell, outcome: CellOutcome) -> None:
        if cell.evaluation_case is None:
            return
        row = _completed_evaluation_row(
            cell,
            outcome,
            _planned_evaluation_row(cell),
        )
        row["evaluation_publication_mode"] = "local"
        apply_generated_evaluation(
            row,
            case=cell.evaluation_case,
            rubrics=cell.evaluation_rubrics,
            judge_model=str(cell.env.get("FUGUE_JUDGE_MODEL") or ""),
            env=self.env,
            trial_dir=Path(str(row.get("trial_dir") or cell.result_path.parent)),
        )
        _set_adapter_outcome(row)
        with self._lock:
            with self.path.open("a") as handle:
                handle.write(
                    json.dumps(redact_value(row), sort_keys=True, default=str) + "\n"
                )


def _planned_evaluation_row(cell: PlannedCell) -> dict[str, Any]:
    env = cell.env
    run_key = ":".join(
        (
            cell.run_id,
            cell.workload_id,
            "trial",
            cell.task_id,
            cell.harness,
            cell.context_system_id,
            cell.variant_id,
            f"t{cell.trial_index:03d}",
        )
    )
    row = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "record_type": "trial",
        "cell_id": cell.id,
        "run_key": run_key,
        "run_id": cell.run_id,
        "run_name": cell.run_name,
        "trial_index": cell.trial_index,
        "comparison_example_id": cell.comparison_example_id,
        "candidate_id": cell.candidate_id,
        "execution_fingerprint": cell.execution_fingerprint,
        "execution_kind": cell.execution_kind,
        "identity_schema_version": CANDIDATE_IDENTITY_SCHEMA_VERSION,
        "task_name": cell.task_id,
        "harness": cell.harness,
        "experiment_id": env.get("FUGUE_EXPERIMENT_ID"),
        "workload_id": cell.workload_id,
        "preset_id": env.get("FUGUE_PRESET_ID"),
        "variant_id": cell.variant_id,
        "prompt_id": env.get("FUGUE_PROMPT_ID"),
        "context_system_id": cell.context_system_id,
        "context_delivery": env.get("FUGUE_CONTEXT_DELIVERY", "portable"),
        "context_version": env.get("FUGUE_CONTEXT_VERSION"),
        "context_support": env.get("FUGUE_CONTEXT_SUPPORT"),
        "context_config_hash": env.get("FUGUE_CONTEXT_CONFIG_HASH"),
        "agent_config_hash": env.get("FUGUE_AGENT_CONFIG_HASH"),
        "skill_ids": [
            value for value in env.get("FUGUE_SKILL_IDS", "").split(",") if value
        ],
        "skill_provenance": _json_list(env.get("FUGUE_SKILL_PROVENANCE")),
        "integration_ids": [
            value for value in env.get("FUGUE_INTEGRATION_IDS", "").split(",") if value
        ],
        "integration_provenance": _json_list(env.get("FUGUE_INTEGRATION_PROVENANCE")),
        "tags": [value for value in env.get("FUGUE_TAGS", "").split(",") if value],
        "dataset": env.get("FUGUE_DATASET"),
        "repository": env.get("FUGUE_REPOSITORY"),
        "base_commit": env.get("FUGUE_BASE_COMMIT"),
        "evaluation_asset_lock_sha256": cell.evaluation_asset_lock_sha256 or None,
        "run_snapshot_sha256": cell.run_snapshot_sha256 or None,
        "source_commit": cell.source_commit or None,
        "model_provider": cell.model_provider,
        "model": cell.model,
        "trace_project": env.get("WEAVE_PROJECT")
        or (
            f"{env.get('WANDB_ENTITY')}/{env.get('WANDB_PROJECT')}"
            if env.get("WANDB_ENTITY") and env.get("WANDB_PROJECT")
            else None
        ),
        "trace_content": env.get("FUGUE_TRACE_CONTENT", "full"),
        "context_assigned": cell.context_system_id != "none",
        "evaluation_case": cell.evaluation_case,
        "evaluation_scorers": list(cell.scorer_refs),
        "evaluation_rubrics": list(cell.evaluation_rubrics),
        "evaluation_scorer_hashes": cell.scorer_hashes or {},
    }
    row["prediction_id"] = _stable_digest(
        {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "run_id": cell.run_id,
            "candidate_id": cell.candidate_id,
            "comparison_example_id": cell.comparison_example_id,
            "trial_index": cell.trial_index,
        }
    )
    if cell.execution_kind == "agent":
        conversation_id = agent_conversation_id(cell.harness, run_key)
        row.update(
            {
                "weave_agent_name": stable_agent_name(cell.harness),
                "planned_conversation_id": conversation_id,
                "weave_conversation_id": conversation_id,
            }
        )
    return row


def _completed_evaluation_row(
    cell: PlannedCell,
    outcome: CellOutcome,
    planned: dict[str, Any],
) -> dict[str, Any]:
    paths = _trial_result_paths(cell.result_path.parent)
    trial_rows: list[dict[str, Any]] = []
    matching: list[dict[str, Any]] = []
    for path in paths:
        row = _row_from_trial(path)
        trial_rows.append(row)
        if (
            row.get("candidate_id") == cell.candidate_id
            and int(row.get("trial_index") or 1) == cell.trial_index
        ):
            matching.append(row)
    if len(matching) == 1:
        row = matching[0]
    elif len(trial_rows) == 1:
        # Setup failures occur before fugue-meta.json is created. One Harbor
        # job contains exactly one task/trial, so its sole result is still the
        # authoritative runtime record for this planned cell.
        row = trial_rows[0]
    else:
        row = dict(planned)
    for key, value in planned.items():
        row.setdefault(key, value)
    row["status"] = outcome.status
    for key in (
        "comparison_example_id",
        "candidate_id",
        "run_id",
        "run_name",
        "trial_index",
        "dataset",
        "workload_id",
        "task_name",
        "harness",
        "experiment_id",
        "preset_id",
        "variant_id",
        "context_system_id",
        "context_delivery",
        "model_provider",
        "model",
        "trace_project",
        "execution_fingerprint",
        "execution_kind",
        "identity_schema_version",
        "weave_agent_name",
        "planned_conversation_id",
        "weave_conversation_id",
        "query_id",
        "sequence_id",
        "episode_id",
        "repository",
        "base_commit",
        "evaluation_asset_lock_sha256",
    ):
        if key in planned:
            row[key] = planned[key]
    if outcome.error and not row.get("exception_class"):
        row["exception_class"] = "HarborCellError"
        row["exception_message"] = outcome.error
    if outcome.status == "failed" and row.get("pass") is None:
        row["pass"] = False
    _apply_host_evidence_scores(
        row,
        cell.expected_evidence_paths,
        cell.evaluation_asset_lock_sha256,
    )
    return row


def _verified_evaluation_root(
    row: dict[str, Any], predict_and_score_call_id: str
) -> dict[str, Any] | None:
    observed = (
        str(row.get("trace_id") or ""),
        str(row.get("root_span_id") or ""),
    )
    roots = [
        root
        for root in row.get("weave_root_spans") or []
        if isinstance(root, dict)
        and (str(root.get("trace_id") or ""), str(root.get("span_id") or ""))
        == observed
    ]
    if len(roots) != 1:
        row["trace_link_status"] = "missing"
        available_roots = row.get("weave_root_spans") or []
        if not available_roots:
            row["trace_link_error"] = (
                "no matching invoke_agent root reached Weave before the link deadline"
            )
        elif not roots:
            row["trace_link_error"] = "no invoke_agent root matched the selected trace"
        else:
            row["trace_link_error"] = (
                "multiple invoke_agent roots matched the selected trace"
            )
        return None
    root = roots[0]
    root_conversation_id = str(root.get("conversation_id") or "")
    conversation_ids = {
        str(value) for value in row.get("weave_conversation_ids") or [] if value
    }
    if not root_conversation_id:
        row["trace_link_status"] = "attribute_missing"
        row["trace_link_error"] = "native root is missing gen_ai.conversation.id"
        return None
    if conversation_ids != {root_conversation_id}:
        row["trace_link_status"] = "identity_mismatch"
        row["trace_link_error"] = (
            "native trace operations do not share the root conversation identity"
        )
        return None
    observed_call_id = str(root.get("eval_predict_and_score_call_id") or "")
    if not observed_call_id:
        row["trace_link_status"] = "attribute_missing"
        row["trace_link_error"] = (
            "native root is missing weave.eval.predict_and_score_call_id"
        )
        return None
    if observed_call_id != predict_and_score_call_id:
        row["trace_link_status"] = "attribute_mismatch"
        row["trace_link_error"] = (
            "native root points to a different evaluation prediction"
        )
        return None
    return root


def _attach_genai_span_ref(call: Any, *, trace_id: str, span_id: str) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{32}", trace_id):
        raise ValueError("invalid OTel trace id returned by Weave Agents")
    if not re.fullmatch(r"[0-9a-fA-F]{16}", span_id):
        raise ValueError("invalid OTel span id returned by Weave Agents")
    if call.summary is None:
        call.summary = {}
    weave_summary = call.summary.setdefault("weave", {})
    weave_summary["genai_span_ref"] = [
        {"trace_id": trace_id.lower(), "span_id": span_id.lower()}
    ]


def _json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def export_rows(
    jobs: list[Path],
    *,
    fetch_weave: bool = False,
    weave_project: str | None = None,
    env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    jobs = list(dict.fromkeys(path.resolve(strict=False) for path in jobs))
    rows = [
        *[_row_from_trial(path) for job in jobs for path in _trial_result_paths(job)],
        *[row for job in jobs for row in _context_result_rows(job)],
        *[row for job in jobs for row in _cell_result_rows(job)],
    ]
    live_rows = [row for job in jobs for row in _live_evaluation_rows(job)]
    live_by_run_key = {
        str(row["run_key"]): row for row in live_rows if row.get("run_key")
    }
    for row in rows:
        if row.get("record_type") == "trial" and row.get("run_key") in live_by_run_key:
            _merge_live_evaluation_row(row, live_by_run_key[str(row["run_key"])])
    _apply_evaluation_asset_locks(rows, jobs, repo_root=repo_root)
    if fetch_weave:
        run_keys = list(
            dict.fromkeys(str(row["run_key"]) for row in rows if row.get("run_key"))
        )
        conversation_ids = {
            str(row["run_key"]): list(
                dict.fromkeys(
                    str(value)
                    for value in [
                        row.get("planned_conversation_id"),
                        row.get("weave_conversation_id"),
                        *(row.get("weave_conversation_ids") or []),
                        *(row.get("native_session_ids") or []),
                    ]
                    if value
                )
            )
            for row in rows
            if row.get("run_key")
        }
        spans = fetch_weave_summaries(
            run_keys=run_keys,
            conversation_ids_by_run=conversation_ids,
            project=weave_project or _weave_project_from_env(env),
            env=env,
        )
        for row in rows:
            if row.get("run_key"):
                summary = spans.get(str(row["run_key"]), {})
                _apply_trace_summary(row, summary)
                observed = summary.get("weave_agent_names") or []
                expected = row.get("weave_agent_name")
                row["weave_agent_name_match"] = (
                    str(expected) in {str(value) for value in observed}
                    if expected and observed
                    else None
                )
                _apply_observed_identity(row)
    for row in rows:
        if row.get("record_type") == "trial":
            _merge_error_events(row)
    _apply_runtime_equivalence(rows)
    return rows


def _apply_evaluation_asset_locks(
    rows: list[dict[str, Any]],
    jobs: list[Path],
    *,
    repo_root: Path | None,
) -> None:
    root = repo_root or _repo_root_from_export_paths(jobs)
    if root is None:
        return
    locks: dict[str, Any] = {}
    for row in rows:
        run_id = str(row.get("run_id") or "")
        if not run_id or run_id in locks:
            continue
        path = root / ".fugue" / "runtime" / run_id / EVALUATION_ASSET_LOCK_NAME
        if path.is_file():
            locks[run_id] = read_evaluation_asset_lock(path)
    for row in rows:
        if row.get("record_type") != "trial":
            continue
        run_id = str(row.get("run_id") or "")
        lock = locks.get(run_id)
        if lock is None:
            continue
        prediction_id = _prediction_id_from_row(row)
        entry = lock.predictions.get(prediction_id) if prediction_id else None
        expected = tuple((entry or {}).get("expected_evidence_paths") or ())
        _apply_host_evidence_scores(row, expected, lock.lock_sha256)


def _repo_root_from_export_paths(paths: list[Path]) -> Path | None:
    for path in paths:
        for parent in (path, *path.parents):
            if parent.name == ".fugue":
                return parent.parent
    return None


def _prediction_id_from_row(row: Mapping[str, Any]) -> str | None:
    run_id = str(row.get("run_id") or "")
    candidate_id = str(row.get("candidate_id") or "")
    comparison_id = str(row.get("comparison_example_id") or "")
    trial_index = _positive_int(row.get("trial_index"))
    if not all((run_id, candidate_id, comparison_id, trial_index)):
        return None
    return _stable_digest(
        {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "comparison_example_id": comparison_id,
            "trial_index": trial_index,
        }
    )


def _apply_host_evidence_scores(
    row: dict[str, Any],
    expected_paths: tuple[str, ...],
    lock_sha256: str,
) -> None:
    if lock_sha256:
        row["evaluation_asset_lock_sha256"] = lock_sha256
    if not expected_paths:
        return
    scores = score_evidence_paths(expected_paths, row.get("evidence_paths") or ())
    row["evidence_recall"] = scores["evidence_recall"]
    row["citation_correctness"] = scores["evidence_precision"]
    expected = {
        value
        for path in expected_paths
        if (value := _normalize_repo_path(path)) is not None
    }
    returned = [
        value
        for path in row.get("context_result_paths") or ()
        if (value := _normalize_repo_path(str(path))) is not None
    ]
    inspected = {
        value
        for path in row.get("inspected_paths") or ()
        if (value := _normalize_repo_path(str(path))) is not None
    }
    changed = {
        value
        for path in row.get("changed_paths") or ()
        if (value := _normalize_repo_path(str(path))) is not None
    }
    ranked = list(dict.fromkeys(returned))
    relevant_ranks = [
        rank for rank, path in enumerate(ranked, start=1) if path in expected
    ]
    for cutoff in (5, 10):
        row[f"retrieval_recall_at_{cutoff}"] = (
            len(expected & set(ranked[:cutoff])) / len(expected) if expected else None
        )
    row["retrieval_mrr"] = 1.0 / min(relevant_ranks) if relevant_ranks else 0.0
    relevant_returned = expected & set(ranked)
    row["relevant_retrieval_observed"] = bool(relevant_returned)
    row["relevant_retrieval_opened"] = bool(relevant_returned & inspected)
    row["relevant_retrieval_changed"] = bool(relevant_returned & changed)
    row["off_target_change_only"] = bool(changed) and not bool(expected & changed)
    row["premature_completion"] = bool(
        row.get("agent_execution_status") == "started"
        and row.get("pass") is False
        and not changed
    )


def normalize_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve every executable result to one stable logical prediction row."""
    normalized: list[dict[str, Any]] = []
    prediction_ids: set[str] = set()
    for raw in _evaluation_rows(rows):
        row = dict(raw)
        run_id = str(row.get("run_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        comparison_id = str(row.get("comparison_example_id") or "")
        trial_index = _positive_int(row.get("trial_index"))
        missing = [
            name
            for name, value in (
                ("run_id", run_id),
                ("candidate_id", candidate_id),
                ("comparison_example_id", comparison_id),
                ("trial_index", trial_index),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "evaluation prediction is missing canonical identity: "
                + ", ".join(missing)
            )
        execution_kind = str(
            row.get("execution_kind")
            or ("agent" if _is_agent_row(row) else "provider_diagnostic")
        )
        prediction_id = _stable_digest(
            {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "run_id": run_id,
                "candidate_id": candidate_id,
                "comparison_example_id": comparison_id,
                "trial_index": trial_index,
            }
        )
        source_record_type = str(
            row.get("source_record_type") or row.get("record_type") or "trial"
        )
        value = PredictionRowV1(
            prediction_id=prediction_id,
            run_id=run_id,
            candidate_id=candidate_id,
            comparison_example_id=comparison_id,
            trial_index=trial_index,
            execution_kind=execution_kind,
            source_record_type=source_record_type,
            payload=row,
        ).to_dict()
        if prediction_id in prediction_ids:
            raise ValueError(
                f"duplicate evaluation trial (normalized prediction): {prediction_id}"
            )
        prediction_ids.add(prediction_id)
        normalized.append(value)
    return normalized


def compile_export(
    jobs: list[Path],
    *,
    fetch_weave: bool = False,
    project: str | None = None,
    publish: bool = False,
    ledger_root: Path | None = None,
    republish: bool = False,
    republish_reason: str | None = None,
    env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> NormalizedExport:
    raw = export_rows(
        jobs,
        fetch_weave=fetch_weave,
        weave_project=project,
        env=env,
        repo_root=repo_root,
    )
    predictions = tuple(normalize_prediction_rows(raw))
    measurements = tuple(
        dict(row)
        for row in raw
        if row.get("record_type") in {"preparation", "retrieval", "episode"}
    )
    publication = (
        publish_to_weave(
            list(predictions),
            project,
            ledger_root=ledger_root,
            republish=republish,
            republish_reason=republish_reason,
            env=env,
        )
        if publish
        else PublicationResult(published=0, skipped=0)
    )
    return NormalizedExport(predictions, measurements, publication)


_LOCAL_RESULT_FIELDS = {
    "agent_evidence_paths",
    "changed_paths",
    "citation_correctness",
    "evaluation_scope_id",
    "evidence_paths",
    "evidence_recall",
    "evaluation_asset_lock_sha256",
    "inspected_paths",
    "local_error_events",
    "runtime_fingerprints",
}


def _merge_live_evaluation_row(row: dict[str, Any], live: dict[str, Any]) -> None:
    local = {
        key: value
        for key, value in row.items()
        if key in _LOCAL_RESULT_FIELDS or key.startswith("context_")
    }
    row.update(live)
    row.update(local)


def _is_agent_row(row: Mapping[str, Any]) -> bool:
    execution_kind = row.get("execution_kind")
    if execution_kind is not None:
        return execution_kind == "agent"
    return bool(
        row.get("record_type") == "trial"
        or row.get("weave_agent_name")
        or row.get("planned_conversation_id")
    )


def _mark_agent_execution_not_started(row: dict[str, Any]) -> None:
    row.update(
        {
            "trace_link_status": "not_started",
            "trace_link_error": (
                "Agent execution did not start; no invoke_agent root was emitted"
            ),
            "trace_link_reason": None,
            "weave_observability_status": "failed",
            "weave_usage_source": "unavailable",
            "weave_usage_status": "unavailable",
        }
    )


def _apply_observed_identity(row: dict[str, Any]) -> None:
    if row.get("status") == "not_applicable" or row.get("applicable") is False:
        row.update(
            {
                "trace_link_status": "not_applicable",
                "trace_link_error": None,
                "weave_observability_status": "not_applicable",
                "weave_usage_source": "not_applicable",
                "weave_usage_status": "not_applicable",
            }
        )
        return
    if row.get("agent_execution_status") == "not_started":
        _mark_agent_execution_not_started(row)
        return
    if not _is_agent_row(row):
        row.update(
            {
                "trace_link_status": "not_applicable",
                "trace_link_error": None,
                "weave_agent_name_match": None,
            }
        )
        return
    expected_agent = str(row.get("weave_agent_name") or row.get("harness") or "")
    expected_run_key = str(row.get("run_key") or "")
    expected_task = str(row.get("task_name") or row.get("task_id") or "")
    expected_candidate = str(row.get("candidate_id") or "")
    expected_example = str(row.get("comparison_example_id") or "")
    expected_trial = _positive_int(row.get("trial_index"))
    matches: dict[tuple[str, str], dict[str, Any]] = {}
    for root in row.get("weave_root_spans") or []:
        if not isinstance(root, dict):
            continue
        if expected_agent and str(root.get("agent_name") or "") != expected_agent:
            continue
        if expected_run_key and str(root.get("run_key") or "") != expected_run_key:
            continue
        if expected_task and not _task_ids_match(
            expected_task, str(root.get("task_id") or "")
        ):
            continue
        if (
            expected_candidate
            and str(root.get("candidate_id") or "") != expected_candidate
        ):
            continue
        if (
            expected_example
            and str(root.get("comparison_example_id") or "") != expected_example
        ):
            continue
        if expected_trial and _positive_int(root.get("trial_index")) != expected_trial:
            continue
        identity = (str(root.get("trace_id") or ""), str(root.get("span_id") or ""))
        matches[identity] = root
    if len(matches) != 1:
        row["trace_link_status"] = "missing" if not matches else "ambiguous"
        row["trace_link_error"] = (
            "no matching invoke_agent root reached Weave before the link deadline"
            if not matches
            else "multiple matching invoke_agent roots reached Weave"
        )
        return
    root = next(iter(matches.values()))
    link_status = "linked" if row.get("trace_link_status") == "linked" else "observed"
    row.update(
        {
            "observed_conversation_id": root.get("conversation_id"),
            "trace_id": root.get("trace_id"),
            "root_span_id": root.get("span_id"),
            "trace_link_status": link_status,
            "trace_link_error": None,
        }
    )


def _apply_trace_summary(row: dict[str, Any], summary: dict[str, Any]) -> None:
    response = summary.pop("_weave_agent_response", None)
    local_gateway_calls = int(row.get("context_gateway_tool_call_count") or 0)
    local_vector = {
        key: row.get(key)
        for key in (
            "gitnexus_vector_search_attempted",
            "gitnexus_vector_search_succeeded",
            "gitnexus_semantic_result_count",
            "gitnexus_bm25_result_count",
            "gitnexus_vector_model_digests",
            "gitnexus_vector_query_latency_ms",
        )
    }
    row.update(summary)
    gateway_calls = max(
        local_gateway_calls,
        int(row.get("weave_gateway_tool_call_count") or 0),
    )
    if row.get("context_assigned") and gateway_calls:
        if local_gateway_calls:
            row.update(local_vector)
        row["context_invoked"] = True
        row["context_invocation_evidence"] = {
            "status": "observed",
            "source": (
                "mcp_gateway_event_log"
                if local_gateway_calls
                else "mcp_gateway_result_metadata"
            ),
            "tool_calls": gateway_calls,
            "gateway_call_ids": (
                row.get("context_gateway_call_ids")
                if local_gateway_calls
                else row.get("weave_gateway_call_ids")
            )
            or [],
        }
    _merge_error_events(row)
    if not isinstance(response, str) or not response.strip():
        return
    encoded = response.encode()
    if not row.get("agent_response_bytes"):
        row["agent_response_bytes"] = len(encoded)
    if not row.get("agent_response_sha256"):
        row["agent_response_sha256"] = hashlib.sha256(encoded).hexdigest()
    if row.get("trace_content") == "full" and not row.get("agent_response"):
        row["agent_response"] = response[:8_000]


def _task_ids_match(expected: str, observed: str) -> bool:
    return bool(
        expected == observed
        or expected.endswith(f"/{observed}")
        or observed.endswith(f"/{expected}")
    )


def write_jsonl(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    secrets = secrets_from_env(env or {})
    with path.open("w") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    redact_value(row, secrets=secrets),
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )


def publish_to_weave(
    rows: list[dict[str, Any]],
    project: str | None = None,
    *,
    ledger_root: Path | None = None,
    republish: bool = False,
    republish_reason: str | None = None,
    env: Mapping[str, str] | None = None,
) -> PublicationResult:
    if republish and not str(republish_reason or "").strip():
        raise ValueError("republish_reason is required for explicit republishing")
    project = project or _weave_project_from_env(env)
    weave = initialize_weave(project, env)
    logger_cls = getattr(weave, "EvaluationLogger", None)
    if logger_cls is None:
        raise RuntimeError("installed weave package has no EvaluationLogger")
    candidates = _publication_candidates(normalize_prediction_rows(rows))
    ledger = (
        (ledger_root or Path(".fugue/runtime/publications"))
        / f"v{PUBLICATION_SCHEMA_VERSION}"
        / _safe_slug(project)
    )
    ledger.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Any] = {}
    evaluations: list[PublishedEvaluation] = []
    failures: list[str] = []
    published = 0
    skipped = 0
    for candidate in candidates:
        if all(
            row.get("evaluation_publication_mode") == "live"
            for row in candidate["rows"]
        ):
            skipped += 1
            continue
        publication_id = candidate["publication_id"]
        scope_id = candidate["evaluation_scope_id"]
        # Different evaluation groupings can contain the same prediction. One
        # project lock makes that overlap visible before either remote write.
        lock = ledger / "publication-ledger.lock"
        with FileLock(lock, timeout=120):
            reservation: list[tuple[Path, dict[str, Any] | None]] = []
            previous_marker, previous_revision = _latest_publication_marker(
                ledger, publication_id
            )
            if previous_marker is not None and not republish:
                try:
                    reservation = _reserve_prediction_publication(
                        ledger,
                        project,
                        candidate,
                        revision=previous_revision,
                    )
                    _finalize_prediction_publication(
                        ledger,
                        project,
                        candidate,
                        revision=previous_revision,
                    )
                    evaluations.append(
                        _published_evaluation_from_marker(
                            previous_marker,
                            project=project,
                            publication_id=publication_id,
                            candidate_id=candidate["candidate_id"],
                            evaluation_scope_id=scope_id,
                            publication_mode="post_hoc",
                        )
                    )
                except (OSError, ValueError) as exc:
                    if reservation:
                        _restore_prediction_publications(reservation)
                    failures.append(
                        f"{candidate['candidate_id']}: publication marker: {exc}"
                    )
                skipped += 1
                continue
            revision = previous_revision + 1 if previous_marker is not None else 1
            marker = ledger / f"{publication_id}.r{revision}.json"
            supersedes = (
                f"{publication_id}:r{previous_revision}"
                if previous_marker is not None
                else None
            )
            try:
                reservation = _reserve_prediction_publication(
                    ledger,
                    project,
                    candidate,
                    revision=revision,
                )
            except (OSError, ValueError) as exc:
                failures.append(
                    f"{candidate['candidate_id']}: publication ledger: {exc}"
                )
                continue
            if scope_id not in datasets:
                dataset_name = _dataset_name(candidate)
                dataset_cls = getattr(weave, "Dataset", None)
                datasets[scope_id] = (
                    dataset_cls(name=dataset_name, rows=candidate["dataset_examples"])
                    if dataset_cls is not None
                    else candidate["dataset_examples"]
                )
            name = _evaluation_name(candidate)
            score_names = candidate["scorers"]
            logger = None
            try:
                attributes = getattr(weave, "attributes", None)
                context = (
                    attributes(_evaluation_run_attributes(candidate))
                    if attributes is not None
                    else nullcontext()
                )
                with context:
                    logger = logger_cls(
                        name=name,
                        model=_evaluation_model(candidate),
                        dataset=datasets[scope_id],
                        eval_attributes=_evaluation_scope_attributes(candidate),
                        scorers=score_names,
                    )
                for row, inputs in zip(
                    candidate["rows"], candidate["prediction_inputs"], strict=True
                ):
                    logger.log_example(
                        inputs=inputs,
                        output=_evaluation_output(row, post_hoc=True),
                        scores=_evaluation_scores(row),
                    )
                logger.log_summary()
            except Exception as exc:
                _restore_prediction_publications(reservation)
                if logger is not None:
                    try:
                        logger.fail(exc)
                    except Exception:
                        pass
                failures.append(
                    f"{candidate['candidate_id']}: {type(exc).__name__}: {exc}"
                )
                continue
            url = getattr(logger, "ui_url", None)
            evaluation_ref = _logger_ref(logger, "_pseudo_evaluation")
            model_ref = _logger_ref(logger, "model")
            agent_rows = [row for row in candidate["rows"] if _is_agent_row(row)]
            direct_rows = [row for row in candidate["rows"] if not _is_agent_row(row)]
            linked_agent_predictions = sum(
                row.get("trace_link_status") == "linked" for row in agent_rows
            )
            linking_failures = tuple(
                f"{row.get('run_key') or row.get('cell_id')}: "
                f"{row.get('trace_link_error') or 'post-hoc Agent prediction has no verified deep link'}"
                for row in agent_rows
                if row.get("trace_link_status") != "linked"
            )
            _write_publication_marker(
                marker,
                project,
                publication_id,
                name=name,
                candidate_id=candidate["candidate_id"],
                evaluation_scope_id=scope_id,
                examples=len(candidate["rows"]),
                url=url,
                evaluation_ref=evaluation_ref,
                model_ref=model_ref,
                agent_predictions=len(agent_rows),
                linked_agent_predictions=linked_agent_predictions,
                direct_predictions=len(direct_rows),
                linking_failures=linking_failures,
                publication_mode="post_hoc",
                publication_schema_version=PUBLICATION_SCHEMA_VERSION,
                scorer_version=candidate["scorer_version"],
                prediction_ids=candidate["prediction_ids"],
                revision=revision,
                supersedes=supersedes,
                republish_reason=(str(republish_reason).strip() if republish else None),
                active=True,
            )
            if previous_marker is not None:
                _set_publication_marker_active(previous_marker, False)
            _finalize_prediction_publication(
                ledger,
                project,
                candidate,
                revision=revision,
            )
            evaluations.append(
                PublishedEvaluation(
                    candidate_id=candidate["candidate_id"],
                    name=name,
                    examples=len(candidate["rows"]),
                    url=url,
                    evaluation_ref=evaluation_ref,
                    model_ref=model_ref,
                    agent_predictions=len(agent_rows),
                    linked_agent_predictions=linked_agent_predictions,
                    direct_predictions=len(direct_rows),
                    linking_failures=linking_failures,
                    publication_id=publication_id,
                    revision=revision,
                    supersedes=supersedes,
                )
            )
            published += 1
    return PublicationResult(
        published=published,
        skipped=skipped,
        evaluations=tuple(evaluations),
        failures=tuple(failures),
    )


def _evaluation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluation_rows = [row for row in rows if row.get("record_type") == "trial"]
    completed_cells = {
        _direct_cell_key(row): row
        for row in rows
        if row.get("record_type") == "cell"
        and row.get("execution_kind") == "provider_diagnostic"
        and row.get("status") == "passed"
    }
    measurements: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = _direct_cell_key(row)
        if key in completed_cells and row.get("record_type") in {
            "episode",
            "retrieval",
        }:
            measurements.setdefault(key, []).append(row)
    for key, cell in completed_cells.items():
        cell_measurements = measurements.get(key, [])
        if not cell_measurements:
            continue
        sequence_rows = [row for row in cell_measurements if row.get("sequence_id")]
        if sequence_rows:
            summary = dict(cell)
            _add_sequence_measurement_summary(summary, sequence_rows)
            evaluation_rows.append(summary)
            continue
        for measurement in cell_measurements:
            if measurement.get("record_type") != "retrieval":
                continue
            projected = dict(measurement)
            projected["dataset"] = measurement.get("workload_id")
            projected["workload_id"] = cell.get("workload_id")
            evaluation_rows.append(projected)
    return evaluation_rows


def _direct_cell_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("run_id") or ""),
        str(row.get("candidate_id") or ""),
        str(row.get("execution_fingerprint") or ""),
        _positive_int(row.get("trial_index") or row.get("attempt")) or 1,
    )


def _add_sequence_measurement_summary(
    row: dict[str, Any], measurements: list[dict[str, Any]]
) -> None:
    first = measurements[0]
    row["record_type"] = "sequence"
    row["dataset"] = first.get("workload_id")
    for key in (
        "experiment_id",
        "preset_id",
        "trace_project",
        "identity_schema_version",
        "context_config_hash",
        "context_version",
        "builder_model",
        "embedding_model",
    ):
        if first.get(key) is not None:
            row[key] = first[key]
    retrievals = [
        item for item in measurements if item.get("record_type") == "retrieval"
    ]
    episodes = [item for item in measurements if item.get("record_type") == "episode"]
    row["context_query_count"] = len(retrievals)
    row["episode_count"] = len(episodes)
    row["context_query_latency_ms"] = sum(
        float(item.get("query_latency_ms") or 0) for item in retrievals
    )
    row["write_latency_ms"] = sum(
        float(item.get("write_latency_ms") or 0) for item in episodes
    )
    if episodes:
        row["storage_bytes"] = max(
            int(item.get("storage_bytes") or 0) for item in episodes
        )
    for score_field in (
        "mrr",
        "ndcg_at_10",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "fact_recall",
    ):
        values = [
            float(item[score_field])
            for item in retrievals
            if item.get(score_field) is not None
        ]
        if values:
            row[score_field] = sum(values) / len(values)


def _write_publication_marker(
    path: Path, project: str, publication_id: str, **metadata: Any
) -> None:
    atomic_write_json(
        path,
        {
            "project": project,
            "publication_id": publication_id,
            "published_at": datetime.now(UTC).isoformat(),
            **metadata,
        },
    )


def _prediction_ledger_paths(
    ledger: Path, project: str, candidate: dict[str, Any]
) -> list[tuple[Path, dict[str, Any]]]:
    root = ledger / "predictions"
    root.mkdir(parents=True, exist_ok=True)
    values: list[tuple[Path, dict[str, Any]]] = []
    for prediction_id in candidate["prediction_ids"]:
        identity = {
            "project": project,
            "prediction_id": prediction_id,
            "scorer_version": candidate["scorer_version"],
        }
        values.append((root / f"{_stable_digest(identity)}.json", identity))
    return values


def _reserve_prediction_publication(
    ledger: Path,
    project: str,
    candidate: dict[str, Any],
    *,
    revision: int,
) -> list[tuple[Path, dict[str, Any] | None]]:
    publication_id = candidate["publication_id"]
    previous: list[tuple[Path, dict[str, Any] | None]] = []
    for path, identity in _prediction_ledger_paths(ledger, project, candidate):
        current = json.loads(path.read_text()) if path.is_file() else None
        if current is not None and (
            not isinstance(current, dict)
            or current.get("project") != project
            or current.get("prediction_id") != identity["prediction_id"]
            or current.get("scorer_version") != identity["scorer_version"]
        ):
            raise ValueError(f"invalid prediction publication ledger entry: {path}")
        if current is not None and current.get("state") == "pending":
            raise ValueError(
                "prediction publication has an unresolved pending reservation: "
                f"{identity['prediction_id']}"
            )
        if current is not None and current.get("publication_id") != publication_id:
            raise ValueError(
                "prediction was already published under another active evaluation: "
                f"{identity['prediction_id']}"
            )
        previous.append((path, current))
    for path, identity in _prediction_ledger_paths(ledger, project, candidate):
        atomic_write_json(
            path,
            {
                **identity,
                "publication_id": publication_id,
                "revision": revision,
                "state": "pending",
            },
        )
    return previous


def _finalize_prediction_publication(
    ledger: Path,
    project: str,
    candidate: dict[str, Any],
    *,
    revision: int,
) -> None:
    for path, identity in _prediction_ledger_paths(ledger, project, candidate):
        atomic_write_json(
            path,
            {
                **identity,
                "publication_id": candidate["publication_id"],
                "revision": revision,
                "state": "active",
            },
        )


def _restore_prediction_publications(
    previous: list[tuple[Path, dict[str, Any] | None]],
) -> None:
    for path, value in previous:
        if value is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write_json(path, value)


def _latest_publication_marker(
    ledger: Path, publication_id: str
) -> tuple[Path | None, int]:
    revisions: list[tuple[int, Path]] = []
    pattern = re.compile(rf"^{re.escape(publication_id)}\.r([1-9][0-9]*)\.json$")
    for path in ledger.glob(f"{publication_id}.r*.json"):
        match = pattern.fullmatch(path.name)
        if match:
            revisions.append((int(match.group(1)), path))
    if not revisions:
        return None, 0
    revision, path = max(revisions, key=lambda item: item[0])
    return path, revision


def _set_publication_marker_active(path: Path, active: bool) -> None:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"invalid publication marker: {path}")
    value["active"] = active
    atomic_write_json(path, value)


def _published_evaluation_from_marker(
    path: Path,
    *,
    project: str,
    publication_id: str,
    candidate_id: str,
    evaluation_scope_id: str,
    publication_mode: str,
) -> PublishedEvaluation:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    if value.get("project") != project or value.get("publication_id") != publication_id:
        raise ValueError("metadata does not match its ledger key")
    expected = {
        "candidate_id": candidate_id,
        "evaluation_scope_id": evaluation_scope_id,
        "publication_mode": publication_mode,
    }
    for metadata_field, expected_value in expected.items():
        if not expected_value or value.get(metadata_field) != expected_value:
            raise ValueError(f"{metadata_field} does not match the current evaluation")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    examples = _publication_marker_count(value, "examples")
    agent_predictions = _publication_marker_count(value, "agent_predictions")
    linked_agent_predictions = _publication_marker_count(
        value, "linked_agent_predictions"
    )
    direct_predictions = _publication_marker_count(value, "direct_predictions")
    if linked_agent_predictions > agent_predictions:
        raise ValueError("linked_agent_predictions cannot exceed agent_predictions")
    if agent_predictions + direct_predictions > examples:
        raise ValueError("prediction counts cannot exceed examples")
    linking_failures = value.get("linking_failures")
    if not isinstance(linking_failures, list) or any(
        not isinstance(item, str) for item in linking_failures
    ):
        raise ValueError("linking_failures must be a list of strings")
    return PublishedEvaluation(
        candidate_id=candidate_id,
        name=name,
        examples=examples,
        url=str(value["url"]) if value.get("url") else None,
        evaluation_ref=(
            str(value["evaluation_ref"]) if value.get("evaluation_ref") else None
        ),
        model_ref=str(value["model_ref"]) if value.get("model_ref") else None,
        agent_predictions=agent_predictions,
        linked_agent_predictions=linked_agent_predictions,
        direct_predictions=direct_predictions,
        linking_failures=tuple(item for item in linking_failures if item),
        publication_id=publication_id,
        revision=(
            _publication_marker_count(value, "revision") if "revision" in value else 1
        ),
        supersedes=str(value["supersedes"]) if value.get("supersedes") else None,
        active=value.get("active") is not False,
    )


def _publication_marker_count(value: dict[str, Any], metadata_field: str) -> int:
    count = value.get(metadata_field)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{metadata_field} must be a nonnegative integer")
    return count


def _publication_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str, str],
        list[tuple[dict[str, Any], dict[str, Any]]],
    ] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("evaluation prediction is missing candidate_id")
        inputs = _evaluation_inputs(row)
        partition = (
            candidate_id,
            str(row.get("experiment_id") or ""),
            str(row.get("workload_id") or ""),
            str(row.get("dataset") or ""),
            str(row.get("record_type") or ""),
        )
        grouped.setdefault(partition, []).append((row, inputs))

    candidates: list[dict[str, Any]] = []
    for partition, values in sorted(grouped.items()):
        candidate_id = partition[0]
        seen: set[tuple[str, int]] = set()
        ordered = sorted(
            values,
            key=lambda item: (
                item[1]["comparison_example_id"],
                _positive_int(item[0].get("trial_index")) or 1,
            ),
        )
        for row, inputs in ordered:
            example_id = str(inputs["comparison_example_id"])
            trial_index = _positive_int(row.get("trial_index")) or 1
            identity = (example_id, trial_index)
            if identity in seen:
                raise ValueError(
                    "duplicate evaluation trial for candidate "
                    f"{candidate_id}: {example_id} trial {trial_index}"
                )
            seen.add(identity)
        prediction_inputs = [inputs for _, inputs in ordered]
        dataset_examples = list(
            {
                str(inputs["comparison_example_id"]): inputs
                for inputs in prediction_inputs
            }.values()
        )
        candidate_rows = [row for row, _ in ordered]
        scorers = _scorer_schema(candidate_rows)
        scorer_version = _stable_digest(
            {
                "scorers": scorers,
                "asset_hashes": sorted(
                    {
                        _stable_digest(row.get("evaluation_scorer_hashes") or {})
                        for row in candidate_rows
                    }
                ),
            }
        )
        prediction_ids = [_evaluation_row_id(row) for row in candidate_rows]
        scope_id = _stable_digest({"examples": dataset_examples, "scorers": scorers})
        publication_id = _stable_digest(
            {
                "candidate_id": candidate_id,
                "evaluation_scope_id": scope_id,
                "rows": prediction_ids,
            }
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "evaluation_scope_id": scope_id,
                "publication_id": publication_id,
                "rows": candidate_rows,
                "prediction_inputs": prediction_inputs,
                "dataset_examples": dataset_examples,
                "scorers": scorers,
                "scorer_version": scorer_version,
                "prediction_ids": prediction_ids,
            }
        )
    return candidates


def _evaluation_row_id(row: dict[str, Any]) -> str:
    if row.get("prediction_id"):
        return str(row["prediction_id"])
    return _stable_digest(
        {
            "run_id": row.get("run_id"),
            "candidate_id": row.get("candidate_id"),
            "comparison_example_id": _evaluation_inputs(row)["comparison_example_id"],
            "trial_index": _positive_int(row.get("trial_index")) or 1,
            "status": _outcome_status(row),
            "scores": _evaluation_scores(row),
        }
    )


def _evaluation_inputs(row: dict[str, Any]) -> dict[str, Any]:
    values = {
        "benchmark_id": row.get("dataset"),
        "workload_id": row.get("workload_id"),
        "task_id": row.get("task_name"),
        "query_id": row.get("query_id"),
        "sequence_id": row.get("sequence_id"),
        "episode_id": row.get("episode_id") or row.get("episode"),
        "repository": row.get("repository"),
        "base_commit": row.get("base_commit"),
        "evaluation_asset_lock_sha256": row.get("evaluation_asset_lock_sha256") or None,
        "evaluation_case": row.get("evaluation_case") or None,
        "evaluation_scorers": row.get("evaluation_scorers") or None,
        "evaluation_rubrics": row.get("evaluation_rubrics") or None,
        "evaluation_scorer_hashes": row.get("evaluation_scorer_hashes") or None,
    }
    comparison_id = row.get("comparison_example_id") or _stable_digest(values)
    return {"comparison_example_id": comparison_id, **_drop_none(values)}


def _evaluation_model(candidate: dict[str, Any]) -> dict[str, Any]:
    row = candidate["rows"][0]
    return _drop_none(
        {
            "name": _candidate_model_name(candidate),
            "candidate_id": candidate["candidate_id"],
            "agent_name": (
                row.get("weave_agent_name") or row.get("harness")
                if _is_agent_row(row)
                else None
            ),
            "harness": row.get("harness"),
            "variant_id": row.get("variant_id"),
            "context_system_id": row.get("context_system_id"),
            "context_delivery": row.get("context_delivery"),
            "model_provider": row.get("model_provider"),
            "model_id": row.get("model"),
        }
    )


def _evaluation_scope_attributes(candidate: dict[str, Any]) -> dict[str, Any]:
    row = candidate["rows"][0]
    return _drop_none(
        {
            "fugue.evaluation_scope_id": candidate["evaluation_scope_id"],
            "fugue.experiment_id": row.get("experiment_id"),
            "fugue.workload_id": row.get("workload_id"),
            "fugue.dataset": row.get("dataset"),
            "fugue.record_type": row.get("record_type"),
        }
    )


def _evaluation_run_attributes(candidate: dict[str, Any]) -> dict[str, Any]:
    rows = candidate["rows"]
    row = rows[0]
    run_ids = sorted({str(item["run_id"]) for item in rows if item.get("run_id")})
    return _drop_none(
        {
            "fugue.candidate_id": candidate["candidate_id"],
            "fugue.preset_id": row.get("preset_id"),
            "fugue.harness": row.get("harness"),
            "fugue.variant_id": row.get("variant_id"),
            "fugue.context_system_id": row.get("context_system_id"),
            "fugue.context_delivery": row.get("context_delivery"),
            "fugue.prompt_id": row.get("prompt_id"),
            "fugue.skill_ids": "|".join(str(x) for x in row.get("skill_ids") or []),
            "fugue.integration_ids": "|".join(
                str(x) for x in row.get("integration_ids") or []
            ),
            "fugue.model_provider": row.get("model_provider"),
            "fugue.model": row.get("model"),
            "fugue.run_ids": "|".join(run_ids),
            "fugue.run_name": row.get("run_name"),
            "fugue.tags": "|".join(str(x) for x in row.get("tags") or []),
        }
    )


def _evaluation_output(
    row: dict[str, Any], *, post_hoc: bool = False
) -> dict[str, Any]:
    conversations = [
        str(value)
        for value in [
            row.get("observed_conversation_id"),
            *(row.get("weave_conversation_ids") or []),
        ]
        if value
    ]
    trace_ids = [str(value) for value in row.get("weave_trace_ids") or [] if value]
    return _drop_none(
        {
            "status": _outcome_status(row),
            "run_key": row.get("run_key"),
            "observed_conversation_id": next(iter(dict.fromkeys(conversations)), None),
            "planned_conversation_id": row.get("planned_conversation_id")
            or row.get("weave_conversation_id"),
            "trace_id": trace_ids[0] if trace_ids else None,
            "root_span_id": next(
                (
                    value
                    for value in [
                        row.get("root_span_id"),
                        *(row.get("weave_root_span_ids") or []),
                    ]
                    if value
                ),
                None,
            ),
            "trace_link_status": (
                "post_hoc_unlinked"
                if post_hoc and _is_agent_row(row)
                else "not_applicable"
                if not _is_agent_row(row)
                else row.get("trace_link_status")
            ),
            "trace_link_reason": row.get("trace_link_reason"),
            "trace_link_error": row.get("trace_link_error"),
            "agent_name": (
                row.get("weave_agent_name") or row.get("harness")
                if _is_agent_row(row)
                else None
            ),
            "exception_type": row.get("exception_class"),
            "evidence_paths": [str(x) for x in (row.get("evidence_paths") or [])[:20]],
            "response": _bounded_agent_response(row),
            "response_sha256": row.get("agent_response_sha256"),
            "response_bytes": row.get("agent_response_bytes"),
            "evaluation_na_dimensions": row.get("evaluation_na_dimensions"),
            "evaluation_error": row.get("evaluation_error"),
        }
    )


def _bounded_agent_response(row: dict[str, Any]) -> str | None:
    if row.get("trace_content") != "full":
        return None
    value = row.get("agent_response")
    if not isinstance(value, str) or not value.strip():
        return None
    return value[:8_000]


_DIRECT_SCORE_FIELDS = (
    "reward",
    "mrr",
    "ndcg_at_10",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "evidence_recall",
    "citation_correctness",
    "fact_recall",
    "judge_correctness",
    "judge_completeness",
    "judge_groundedness",
    "judge_overall",
)

_SCORE_ALIASES = {
    "wall_time_sec": "wall_time_seconds",
    "evaluation_prediction_latency_sec": "prediction_latency_seconds",
    "weave_agent_latency_sec": "agent_latency_seconds",
    "weave_model_latency_sec": "model_latency_seconds",
    "weave_input_tokens": "input_tokens",
    "weave_output_tokens": "output_tokens",
    "weave_total_cost_usd": "total_cost_usd",
    "weave_tool_call_count": "tool_calls",
    "weave_terminal_error_count": "terminal_errors",
    "weave_model_error_count": "model_errors",
    "recoverable_error_count": "recoverable_tool_errors",
    "agent_error_count": "agent_errors",
    "benchmark_runtime_error_count": "benchmark_runtime_errors",
    "harness_adapter_error_count": "harness_adapter_errors",
    "context_system_error_count": "context_system_errors",
    "provider_error_count": "provider_errors",
    "fugue_error_count": "fugue_errors",
    "context_error_count": "context_errors",
    "context_query_count": "context_queries",
    "context_query_latency_ms": "context_query_latency_ms",
    "context_registered": "context_registered",
    "runtime_equivalent": "runtime_equivalent",
    "episode_count": "episodes",
    "write_latency_ms": "context_write_latency_ms",
    "storage_bytes": "context_storage_bytes",
}

_COMMON_SCORERS = tuple(
    dict.fromkeys(
        (
            *_DIRECT_SCORE_FIELDS,
            "passed",
            "wall_time_seconds",
            "prediction_latency_seconds",
            "agent_latency_seconds",
            "model_latency_seconds",
            "input_tokens",
            "output_tokens",
            "total_cost_usd",
            "tool_calls",
            "terminal_errors",
            "model_errors",
            "recoverable_tool_errors",
            "agent_errors",
            "benchmark_runtime_errors",
            "harness_adapter_errors",
            "context_system_errors",
            "provider_errors",
            "fugue_errors",
            "context_errors",
            "context_queries",
            "context_query_latency_ms",
            "context_registered",
            "runtime_equivalent",
            "episodes",
            "context_write_latency_ms",
            "context_storage_bytes",
        )
    )
)


def _evaluation_scores(row: dict[str, Any]) -> dict[str, Any]:
    scores = {
        name: row[name] for name in _DIRECT_SCORE_FIELDS if row.get(name) is not None
    }
    if row.get("pass") is not None:
        scores["passed"] = bool(row["pass"])
    for source, target in _SCORE_ALIASES.items():
        if row.get(source) is not None:
            scores[target] = row[source]
    if (
        "input_tokens" not in scores
        and row.get("weave_usage_status") is None
        and _measured_local_usage(row)
    ):
        scores["input_tokens"] = row.get("n_input_tokens")
        scores["output_tokens"] = row.get("n_output_tokens")
        if row.get("cost_usd") is not None:
            scores["total_cost_usd"] = row["cost_usd"]
    for dimension in (
        "task_completion",
        "correctness",
        "groundedness",
        "tool_use",
        "artifact_quality",
    ):
        key = f"evaluation_{dimension}"
        if row.get(key) is not None:
            scores[key] = row[key]
    return {key: value for key, value in scores.items() if value is not None}


def _scorer_schema(rows: list[dict[str, Any]]) -> list[str]:
    values = set(_COMMON_SCORERS)
    for row in rows:
        values.update(_evaluation_scores(row))
        case = row.get("evaluation_case") or {}
        for dimension in case.get("scorer_dimensions") or []:
            values.add(f"evaluation_{dimension}")
    return sorted(values)


def _dataset_name(candidate: dict[str, Any]) -> str:
    row = candidate["rows"][0]
    return _safe_slug(
        "-".join(
            str(value)
            for value in (
                "fugue",
                row.get("experiment_id") or "experiment",
                row.get("workload_id") or "workload",
                candidate["evaluation_scope_id"][:10],
            )
        )
    )


def _evaluation_name(candidate: dict[str, Any]) -> str:
    row = candidate["rows"][0]
    return " | ".join(
        str(value)
        for value in (
            row.get("experiment_id") or "fugue",
            row.get("workload_id") or "workload",
            candidate["evaluation_scope_id"][:10],
        )
    )


def _candidate_model_name(candidate: dict[str, Any]) -> str:
    row = candidate["rows"][0]
    model = str(row.get("model") or "model").split("/")[-1]
    return _safe_slug(
        "__".join(
            (
                str(row.get("harness") or "agent"),
                str(
                    row.get("variant_id") or row.get("context_system_id") or "baseline"
                ),
                model,
            )
        )
    )[:128]


def _logger_ref(logger: Any, attribute: str) -> str | None:
    value = getattr(logger, attribute, None)
    ref = getattr(value, "ref", None)
    uri = getattr(ref, "uri", None)
    return str(uri or ref) if ref else None


def _outcome_status(row: dict[str, Any]) -> str:
    if row.get("status") in {"cancelled", "interrupted", "not_applicable"}:
        return str(row["status"])
    if row.get("exception_class"):
        return "error"
    if row.get("pass") is True:
        return "passed"
    if row.get("pass") is False:
        return "failed"
    return "unscored"


def _measured_local_usage(row: dict[str, Any]) -> bool:
    if row.get("local_usage_status") == "unavailable":
        return False
    return any(
        key in row and row[key] is not None
        for key in ("n_input_tokens", "n_output_tokens", "cost_usd")
    )


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)


def fetch_weave_summaries(
    *,
    run_keys: list[str],
    conversation_ids_by_run: Mapping[str, list[str]] | None = None,
    project: str,
    timeout_sec: float = 30.0,
    env: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    values = env if env is not None else os.environ
    api_key = values.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is required to fetch Weave spans")
    base_url = (values.get("WF_TRACE_SERVER_URL") or WEAVE_AGENTS_BASE_URL).rstrip("/")
    agents_base_url = values.get("WEAVE_AGENTS_BASE_URL", WEAVE_AGENTS_BASE_URL).rstrip(
        "/"
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    summaries: dict[str, dict[str, Any]] = {}
    with httpx.Client(timeout=timeout_sec, headers=headers) as client:
        for run_key in run_keys:
            summaries[run_key] = _summarize_spans(
                _fetch_calls_spans(client, base_url, project, run_key)
                + _fetch_agents_spans(
                    client,
                    agents_base_url,
                    project,
                    (conversation_ids_by_run or {}).get(run_key, []),
                )
            )
    return summaries


def _fetch_calls_spans(
    client: httpx.Client, base_url: str, project: str, run_key: str
) -> list[dict[str, Any]]:
    entity, name = project.split("/", 1)
    payload = {
        "project_id": f"{entity}/{name}",
        "filter": {
            "trace_roots_only": False,
        },
        "query": {
            "$expr": {
                "$eq": [
                    {"$getField": "attributes.fugue.run_key"},
                    {"$literal": run_key},
                ]
            }
        },
    }
    response = client.post(f"{base_url}/calls/stream_query", json=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"Weave Calls query failed with HTTP {response.status_code}")
    return _decode_call_stream(response.text)


def _decode_call_stream(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
            if isinstance(payload, dict) and isinstance(payload.get("calls"), list):
                return [item for item in payload["calls"] if isinstance(item, dict)]
    calls: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Weave Calls query returned invalid NDJSON") from exc
        if isinstance(value, dict):
            calls.append(value)
    return calls


def _fetch_agents_spans(
    client: httpx.Client,
    base_url: str,
    project: str,
    conversation_ids: list[str],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for conversation_id in dict.fromkeys(conversation_ids):
        payload = {
            "project_id": project,
            "query": {
                "$expr": {
                    "$eq": [
                        {"$getField": "conversation_id"},
                        {"$literal": conversation_id},
                    ]
                }
            },
            "include_details": True,
            "include_costs": True,
            "limit": 10_000,
        }
        response = client.post(f"{base_url}/agents/spans/query", json=payload)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Weave Agents query failed with HTTP {response.status_code}"
            )
        data = response.json()
        values = data if isinstance(data, list) else data.get("spans", [])
        spans.extend(value for value in values if isinstance(value, dict))
    return spans


def _summarize_spans(spans: list[dict[str, Any]]) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for index, span in enumerate(spans):
        identity = str(
            span.get("id")
            or span.get("span_id")
            or span.get("call_id")
            or f"row-{index}"
        )
        unique[identity] = span
    values = list(unique.values())
    if not values:
        return {
            "weave_span_count": 0,
            "weave_observability_status": "unavailable",
            "weave_agent_names": [],
            "weave_conversation_ids": [],
            "weave_trace_ids": [],
            "weave_root_span_ids": [],
            "weave_root_spans": [],
            "weave_usage_status": "unavailable",
            "weave_usage_source": "unavailable",
            "weave_input_tokens": None,
            "weave_output_tokens": None,
            "weave_total_cost_usd": None,
        }
    operations = [_span_operation(span) for span in values]
    attributes = [_span_attributes(span) for span in values]
    usage = _span_usage_summary(values, attributes)
    fugue_attributes, attribute_status, missing_attributes = _fugue_attribute_summary(
        values
    )
    # Hermes emits helper spans such as tool.terminal with the provisional Fugue
    # identity. Only the documented Agent operations own conversation identity.
    agent_operations = {"invoke_agent", "chat", "execute_tool"}
    conversation_ids = sorted(
        {
            str(value)
            for span, attrs, operation in zip(
                values, attributes, operations, strict=True
            )
            if operation in agent_operations
            and (
                value := span.get("conversation_id")
                or attrs.get("gen_ai.conversation.id")
            )
        }
    )
    agent_names = sorted(
        {
            str(value)
            for span, attrs in zip(values, attributes, strict=True)
            if (value := span.get("agent_name") or attrs.get("gen_ai.agent.name"))
        }
    )
    trace_ids = sorted(
        {str(value) for span in values if (value := _span_value(span, "trace_id"))}
    )
    root_span_ids = sorted(
        {
            str(_span_value(span, "id") or _span_value(span, "span_id"))
            for span in values
            if _span_operation(span) == "invoke_agent"
            and not (
                _span_value(span, "parent_id") or _span_value(span, "parent_span_id")
            )
        }
        - {"None"}
    )
    roots = [
        span
        for span in values
        if _span_operation(span) == "invoke_agent"
        and not (_span_value(span, "parent_id") or _span_value(span, "parent_span_id"))
    ]
    tool_names = Counter(
        str(value)
        for span, attrs in zip(values, attributes, strict=True)
        if _span_operation(span) == "execute_tool"
        and (value := span.get("tool_name") or attrs.get("gen_ai.tool.name"))
    )
    error_types = Counter(
        _span_error_type(span) for span in values if _span_has_error(span)
    )
    error_events = [
        _error_event_from_span(span) for span in values if _span_has_error(span)
    ]
    root_id = next(
        (
            str(value)
            for span in roots
            if (value := _span_value(span, "id") or _span_value(span, "span_id"))
        ),
        None,
    )
    root_spans = [_root_span_summary(span) for span in roots]
    chat_spans = [span for span in values if _span_operation(span) == "chat"]
    tool_spans = [span for span in values if _span_operation(span) == "execute_tool"]
    gateway_call_ids = sorted(
        {
            call_id
            for span in tool_spans
            if (call_id := _gateway_call_id(span)) is not None
        }
    )
    vector_events = [
        value for span in tool_spans if (value := _gateway_vector(span)) is not None
    ]
    return {
        "weave_span_count": len(values),
        "weave_observability_status": "available",
        "weave_turn_count": operations.count("invoke_agent"),
        "weave_llm_call_count": operations.count("chat"),
        "weave_tool_call_count": operations.count("execute_tool"),
        "weave_gateway_tool_call_count": len(gateway_call_ids),
        "weave_gateway_call_ids": gateway_call_ids,
        "gitnexus_vector_search_attempted": any(
            value.get("vector_search_attempted") is True for value in vector_events
        ),
        "gitnexus_vector_search_succeeded": any(
            value.get("vector_search_succeeded") is True for value in vector_events
        ),
        "gitnexus_semantic_result_count": sum(
            int(value.get("semantic_result_count") or 0) for value in vector_events
        ),
        "gitnexus_bm25_result_count": sum(
            int(value.get("bm25_result_count") or 0) for value in vector_events
        ),
        "gitnexus_vector_model_digests": sorted(
            {
                str(digest)
                for value in vector_events
                if (digest := value.get("model_digest"))
            }
        ),
        "gitnexus_vector_query_latency_ms": sum(
            float(value.get("query_latency_ms") or 0.0) for value in vector_events
        ),
        "weave_error_count": sum(_span_has_error(span) for span in values),
        "weave_terminal_error_count": sum(_span_has_error(span) for span in roots),
        "weave_model_error_count": sum(_span_has_error(span) for span in chat_spans),
        "weave_tool_error_count": sum(_span_has_error(span) for span in tool_spans),
        "weave_error_types": dict(sorted(error_types.items())),
        "weave_error_events": error_events,
        "weave_tool_names": dict(sorted(tool_names.items())),
        "weave_agent_names": agent_names,
        "weave_conversation_ids": conversation_ids,
        "weave_trace_ids": trace_ids,
        "weave_root_span_ids": root_span_ids,
        "weave_root_spans": root_spans,
        "weave_call_id": root_id,
        "weave_agent_latency_sec": _root_latency(roots),
        "weave_model_latency_sec": _root_latency(chat_spans),
        "weave_fugue_attributes": fugue_attributes,
        "weave_attribute_status": attribute_status,
        "weave_missing_attributes": missing_attributes,
        "_weave_agent_response": _latest_agent_response([*roots, *chat_spans]),
        **usage,
    }


def _latest_agent_response(spans: list[dict[str, Any]]) -> str | None:
    ordered = sorted(
        spans,
        key=lambda span: str(span.get("ended_at") or span.get("end_time") or ""),
        reverse=True,
    )
    for span in ordered:
        messages = span.get("output_messages")
        if not isinstance(messages, list):
            continue
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _gateway_call_id(span: dict[str, Any]) -> str | None:
    for source in (span, _raw_span(span)):
        value = _nested_value(source, "fugue_gateway_call_id")
        if value not in (None, ""):
            return str(value)
    return None


def _gateway_vector(span: dict[str, Any]) -> dict[str, Any] | None:
    for source in (span, _raw_span(span)):
        value = _nested_value(source, "fugue_gitnexus_vector")
        if isinstance(value, dict):
            return value
    return None


def _nested_value(value: Any, key: str, *, _depth: int = 0) -> Any:
    if _depth > 12:
        return None
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _nested_value(item, key, _depth=_depth + 1)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for item in value:
            found = _nested_value(item, key, _depth=_depth + 1)
            if found not in (None, ""):
                return found
    elif isinstance(value, str) and len(value) <= 256 * 1024:
        stripped = value.strip()
        if stripped[:1] in {"{", "["}:
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            return _nested_value(decoded, key, _depth=_depth + 1)
    return None


def _root_span_summary(span: dict[str, Any]) -> dict[str, Any]:
    attrs = _span_attributes(span)
    return _drop_none(
        {
            "conversation_id": span.get("conversation_id")
            or attrs.get("gen_ai.conversation.id"),
            "agent_name": span.get("agent_name") or attrs.get("gen_ai.agent.name"),
            "trace_id": _span_value(span, "trace_id"),
            "span_id": _span_value(span, "span_id") or _span_value(span, "id"),
            "run_key": attrs.get("fugue.run_key"),
            "harness": attrs.get("fugue.harness"),
            "task_id": attrs.get("fugue.task_id"),
            "candidate_id": attrs.get("fugue.candidate_id"),
            "comparison_example_id": attrs.get("fugue.comparison_example_id"),
            "trial_index": attrs.get("fugue.trial_index"),
            "eval_predict_and_score_call_id": attrs.get(
                "weave.eval.predict_and_score_call_id"
            ),
        }
    )


def _span_attributes(span: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    raw = _raw_span(span)
    for source in (raw.get("attributes"), span.get("attributes")):
        if isinstance(source, dict):
            merged.update(_flatten_attributes(source))
    for name in (
        "custom_attrs_string",
        "custom_attrs_int",
        "custom_attrs_float",
        "custom_attrs_bool",
    ):
        source = span.get(name)
        if isinstance(source, dict):
            merged.update(source)
    return merged


def _resource_attributes(span: dict[str, Any]) -> dict[str, Any]:
    resource = _raw_span(span).get("resource") or {}
    attributes = resource.get("attributes") if isinstance(resource, dict) else {}
    return _flatten_attributes(attributes) if isinstance(attributes, dict) else {}


def _raw_span(span: dict[str, Any]) -> dict[str, Any]:
    raw = span.get("raw_span_dump") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _flatten_attributes(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(_flatten_attributes(item, name))
        else:
            flattened[name] = item
    return flattened


def _span_operation(span: dict[str, Any]) -> str:
    attrs = _span_attributes(span)
    value = (
        attrs.get("gen_ai.operation.name")
        or span.get("operation_name")
        or span.get("operation")
    )
    if value:
        return str(value)
    name = str(span.get("span_name") or span.get("name") or span.get("op_name") or "")
    return name.split(" ", 1)[0]


def _span_value(span: dict[str, Any], key: str) -> Any:
    if key in span and span[key] is not None:
        return span[key]
    return _span_attributes(span).get(key)


def _span_has_error(span: dict[str, Any]) -> bool:
    status = (
        span.get("status_code")
        or span.get("status")
        or _span_attributes(span).get("status")
    )
    return bool(
        span.get("exception") or span.get("error") or str(status).lower() == "error"
    )


def _span_usage_summary(
    spans: list[dict[str, Any]], attributes: list[dict[str, Any]]
) -> dict[str, Any]:
    chat = [
        (span, attrs)
        for span, attrs in zip(spans, attributes, strict=True)
        if _span_operation(span) == "chat"
    ]
    roots = [
        (span, attrs)
        for span, attrs in zip(spans, attributes, strict=True)
        if _span_operation(span) == "invoke_agent"
        and not (_span_value(span, "parent_id") or _span_value(span, "parent_span_id"))
    ]
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    source = "unavailable"
    if _has_usage(chat):
        selected = chat
        source = "chat_sum"
    elif _has_usage(roots):
        selected = roots
        source = "root_aggregate"

    input_tokens = _sum_metric(
        selected, "input_tokens", "gen_ai.usage.input_tokens", integer=True
    )
    output_tokens = _sum_metric(
        selected, "output_tokens", "gen_ai.usage.output_tokens", integer=True
    )
    total_cost = _sum_cost(selected)
    return {
        "weave_input_tokens": input_tokens,
        "weave_output_tokens": output_tokens,
        "weave_total_cost_usd": total_cost,
        "weave_usage_status": "available" if selected else "unavailable",
        "weave_usage_source": source,
        "weave_cost_status": "available" if total_cost is not None else "unavailable",
    }


def _has_usage(values: list[tuple[dict[str, Any], dict[str, Any]]]) -> bool:
    return any(
        attribute in attrs or _number(span.get(field)) not in (None, 0.0)
        for span, attrs in values
        for field, attribute in (
            ("input_tokens", "gen_ai.usage.input_tokens"),
            ("output_tokens", "gen_ai.usage.output_tokens"),
        )
    )


def _sum_metric(
    values: list[tuple[dict[str, Any], dict[str, Any]]],
    field: str,
    attribute: str,
    *,
    integer: bool = False,
) -> int | float | None:
    if not values:
        return None
    observed = False
    total = 0.0
    for span, attrs in values:
        value = attrs[attribute] if attribute in attrs else span.get(field)
        number = _number(value)
        if number is None:
            continue
        observed = True
        total += number
    if not observed:
        return None
    return int(total) if integer else total


def _sum_cost(values: list[tuple[dict[str, Any], dict[str, Any]]]) -> float | None:
    total = 0.0
    observed = False
    for span, attrs in values:
        value = next(
            (
                source[key]
                for source, key in (
                    (attrs, "gen_ai.usage.cost"),
                    (attrs, "gen_ai.usage.total_cost_usd"),
                    (span, "total_cost_usd"),
                    (span, "cost"),
                )
                if key in source and source[key] is not None
            ),
            None,
        )
        number = _number(value)
        if number is None:
            continue
        observed = True
        total += number
    return total if observed else None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_REQUIRED_FUGUE_ATTRIBUTES = (
    "fugue.run_key",
    "fugue.run_id",
    "fugue.experiment_id",
    "fugue.workload_id",
    "fugue.harness",
    "fugue.variant_id",
    "fugue.context_system_id",
    "fugue.context_delivery",
    "fugue.context_registration_status",
    "fugue.task_id",
    "fugue.trial_index",
    "fugue.comparison_example_id",
    "fugue.candidate_id",
    "fugue.model_provider",
    "fugue.model",
)


def _fugue_attribute_summary(
    spans: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, list[str]]:
    span_values: dict[str, Any] = {}
    resource_values: dict[str, Any] = {}
    for span in spans:
        for key, value in _span_attributes(span).items():
            if key.startswith("fugue.") and value not in (None, ""):
                span_values.setdefault(key, value)
        for key, value in _resource_attributes(span).items():
            if key.startswith("fugue.") and value not in (None, ""):
                resource_values.setdefault(key, value)
    values = {**resource_values, **span_values}
    missing = [key for key in _REQUIRED_FUGUE_ATTRIBUTES if key not in values]
    if not values:
        status = "missing"
    elif not span_values:
        status = "resource_only"
    elif missing:
        status = "partial"
    else:
        status = "complete"
    return values, status, missing


def _root_latency(roots: list[dict[str, Any]]) -> float | None:
    durations: list[float] = []
    for span in roots:
        started = _parse_time(span.get("started_at") or span.get("start_time"))
        ended = _parse_time(span.get("ended_at") or span.get("end_time"))
        if started and ended:
            durations.append((ended - started).total_seconds())
    return sum(durations) if durations else None


def _span_error_type(span: dict[str, Any]) -> str:
    attrs = _span_attributes(span)
    return str(
        span.get("error_type")
        or attrs.get("error.type")
        or span.get("exception_type")
        or "unknown"
    )


def _error_event_from_span(span: dict[str, Any]) -> dict[str, Any]:
    attrs = _span_attributes(span)
    operation = _span_operation(span)
    tool_name = str(
        span.get("tool_name")
        or attrs.get("gen_ai.tool.name")
        or attrs.get("tool.name")
        or ""
    )
    message = _span_error_message(span, attrs)
    return _classify_error(
        message,
        tool_name=tool_name,
        operation=operation,
        source="weave_span",
        terminal=operation == "invoke_agent",
        event_key=str(
            span.get("id") or span.get("span_id") or span.get("call_id") or ""
        ),
    )


def _span_error_message(span: dict[str, Any], attrs: dict[str, Any]) -> str:
    for value in (
        span.get("error_message"),
        span.get("status_message"),
        span.get("exception_message"),
        attrs.get("exception.message"),
        attrs.get("error.message"),
        span.get("error"),
        span.get("output"),
    ):
        if isinstance(value, str) and value.strip():
            return value[:2_000]
        if isinstance(value, dict):
            text = json.dumps(value, sort_keys=True, default=str)
            if text != "{}":
                return text[:2_000]
    return _span_error_type(span)


def _classify_error(
    message: str,
    *,
    tool_name: str,
    operation: str,
    source: str,
    terminal: bool = False,
    event_key: str = "",
) -> dict[str, Any]:
    text = " ".join(message.split())[:2_000]
    lowered = text.lower()
    tool = tool_name.lower()
    if "context" in tool or "fugue-context" in lowered:
        origin, kind = "context_system", "context_failure"
    elif any(
        token in lowered
        for token in (
            "unknown variant `namespace`",
            "expected `function`",
            "badrequesterror",
            "rate limit",
            "quota",
            "http 401",
            "http 429",
        )
    ):
        origin, kind = "provider", "provider_rejection"
    elif any(
        token in lowered
        for token in ("disabled", "no provider", "tool unavailable", "not configured")
    ):
        origin, kind = "harness_adapter", "tool_unavailable"
    elif operation == "adapter_setup":
        origin, kind = "harness_adapter", "integration_failure"
    elif operation == "verifier":
        origin, kind = "benchmark_runtime", "verifier_failure"
    elif operation == "framework":
        origin, kind = "fugue", "framework_failure"
    elif any(
        token in lowered
        for token in (
            "modulenotfounderror",
            "no module named",
            "command not found",
            "not built",
            "missing dependency",
        )
    ):
        origin, kind = "benchmark_runtime", "dependency_missing"
    elif any(
        token in lowered
        for token in (
            "must be a string",
            "got dict",
            "required field",
            "old_string and new_string are identical",
            "invalid arguments",
        )
    ):
        origin, kind = "agent", "invalid_tool_arguments"
    elif "syntaxerror" in lowered or "parse error" in lowered:
        origin, kind = "agent", "generated_code_error"
    elif "plugin" in lowered and any(
        token in lowered for token in ("install", "load", "startup", "crash")
    ):
        origin, kind = "harness_adapter", "integration_failure"
    elif any(token in lowered for token in ("exit code", "tool reported failure")):
        origin, kind = "agent", "command_exit"
    elif "fugue" in lowered and operation != "execute_tool":
        origin, kind = "fugue", "framework_failure"
    elif operation == "execute_tool":
        origin, kind = "agent", "tool_failure"
    elif operation == "chat":
        origin, kind = "provider", "model_failure"
    else:
        origin, kind = "agent", "agent_failure"
    identity = hashlib.sha256(
        json.dumps(
            {
                "origin": origin,
                "kind": kind,
                "tool": tool,
                "message": lowered[:500],
                "event_key": event_key,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "id": identity,
        "origin": origin,
        "kind": kind,
        "recoverable": not terminal,
        "terminal": terminal,
        "phase": (
            "agent"
            if operation in {"invoke_agent", "chat", "execute_tool"}
            else operation
        ),
        "tool_name": tool_name or None,
        "source": source,
        "message": text,
    }


def _merge_error_events(row: dict[str, Any]) -> None:
    weave = [
        event
        for event in row.get("weave_error_events") or []
        if isinstance(event, dict)
    ]
    local = [
        event
        for event in row.get("local_error_events") or []
        if isinstance(event, dict)
    ]
    values = list({str(event.get("id")): event for event in weave}.values())
    matched = Counter(_error_match_key(event) for event in values)
    for event in local:
        key = _error_match_key(event)
        if matched[key]:
            matched[key] -= 1
            continue
        values.append(event)
    row["error_events"] = values
    row["recoverable_error_count"] = sum(
        bool(event.get("recoverable")) for event in values
    )
    for origin in (
        "agent",
        "benchmark_runtime",
        "harness_adapter",
        "context_system",
        "provider",
        "fugue",
    ):
        row[f"{origin}_error_count"] = sum(
            event.get("origin") == origin for event in values
        )
    _set_adapter_outcome(row, values)


def _set_adapter_outcome(
    row: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
) -> None:
    values = events if events is not None else list(row.get("error_events") or [])
    terminal = [event for event in values if event.get("terminal")]
    recoverable = [event for event in values if event.get("recoverable")]
    status = str(row.get("status") or "")
    if status in {"cancelled", "not_applicable"}:
        execution_state = status
    elif terminal:
        execution_state = "failed"
    elif row.get("record_type") == "trial" or row.get("reward") is not None:
        execution_state = "completed"
    else:
        execution_state = "unknown"
    if row.get("reward") is None:
        deterministic = "unscored"
    elif row.get("pass") is True:
        deterministic = "passed"
    else:
        deterministic = "failed"
    if row.get("judge_error") or row.get("evaluation_error"):
        judge = "failed"
    elif (
        row.get("judge_overall") is not None
        or row.get("evaluation_judge_status") == "scored"
    ):
        judge = "scored"
    elif row.get("evaluation_rubrics"):
        judge = "pending"
    else:
        judge = "not_requested"
    observability = str(row.get("weave_observability_status") or "unavailable")
    row["adapter_outcome"] = {
        "execution": {
            "state": execution_state,
            "fatal_error_ids": [str(event.get("id")) for event in terminal],
        },
        "exploratory_tools": {
            "state": "recoverable_failures" if recoverable else "clean",
            "recoverable_error_ids": [str(event.get("id")) for event in recoverable],
        },
        "provider": {
            "state": (
                "failed"
                if any(event.get("origin") == "provider" for event in terminal)
                else "available"
            )
        },
        "deterministic_verification": {"state": deterministic},
        "rubric_evaluation": {"state": judge},
        "observability": {
            "state": observability,
            "trace_link_status": row.get("trace_link_status"),
        },
    }


def _error_match_key(event: dict[str, Any]) -> tuple[str, str, str, bool]:
    return (
        str(event.get("origin") or "unknown"),
        str(event.get("kind") or "unknown"),
        str(event.get("tool_name") or "").lower(),
        bool(event.get("terminal")),
    )


def _apply_runtime_equivalence(rows: list[dict[str, Any]]) -> None:
    cohorts: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("record_type") != "trial":
            continue
        key = (
            str(row.get("run_id") or ""),
            str(row.get("comparison_example_id") or row.get("task_name") or ""),
            str(row.get("trial_index") or 1),
            str(row.get("model") or ""),
        )
        cohorts.setdefault(key, []).append(row)
    for cohort in cohorts.values():
        digests = [_runtime_comparison_digest(row) for row in cohort]
        available = [value for value in digests if value]
        if len(available) != len(cohort):
            status, equivalent = "unavailable", None
        elif len(set(available)) == 1:
            status, equivalent = "equivalent", True
        else:
            status, equivalent = "mismatch", False
        for row in cohort:
            row["runtime_equivalence_status"] = status
            row["runtime_equivalent"] = equivalent
            row["runtime_pre_install_digest"] = (
                (row.get("runtime_fingerprints") or {}).get("pre_install") or {}
            ).get("comparable_digest")
            row["runtime_pre_execution_digest"] = (
                (row.get("runtime_fingerprints") or {}).get("pre_execution") or {}
            ).get("comparable_digest")
            row["runtime_post_execution_digest"] = (
                (row.get("runtime_fingerprints") or {}).get("post_execution") or {}
            ).get("comparable_digest")
            before = row["runtime_pre_execution_digest"]
            after = row["runtime_post_execution_digest"]
            row["runtime_drift"] = (
                before != after if before is not None and after is not None else None
            )


def _runtime_comparison_digest(row: dict[str, Any]) -> str:
    values = row.get("runtime_fingerprints") or {}
    for stage in ("pre_execution", "verified", "pre_install"):
        digest = (values.get(stage) or {}).get("comparable_digest")
        if digest:
            return str(digest)
    return ""


def _trial_result_paths(job: Path) -> list[Path]:
    if job.is_file() and job.name == "result.json":
        return [job]
    if (job / "result.json").is_file() and (job / "agent").is_dir():
        return [job / "result.json"]
    return sorted(
        path
        for path in job.rglob("result.json")
        if path.parent != job and (path.parent / "agent").exists()
    )


def _local_usage(agent_result: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "n_input_tokens": agent_result.get("n_input_tokens"),
        "n_cache_tokens": agent_result.get("n_cache_tokens"),
        "n_output_tokens": agent_result.get("n_output_tokens"),
        "cost_usd": agent_result.get("cost_usd"),
    }
    measured = any(_number(value) not in (None, 0.0) for value in values.values())
    if measured:
        return {"local_usage_status": "available", **values}
    # Harbor adapters that cannot report usage serialize the same zero tuple as
    # a measured result. Without source attribution, that tuple is unavailable.
    return {
        "local_usage_status": "unavailable",
        **dict.fromkeys(values),
    }


def _row_from_trial(result_path: Path) -> dict[str, Any]:
    trial = json.loads(result_path.read_text())
    trial_dir = result_path.parent
    meta_path = trial_dir / "agent" / "fugue-meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    agent_result = trial.get("agent_result") or {}
    local_usage = _local_usage(agent_result)
    task_interaction = meta.get(
        "task_interaction", {"status": "unavailable", "type": "single_turn"}
    )
    if not isinstance(task_interaction, dict):
        task_interaction = {"status": "unavailable", "type": "single_turn"}
    agent_cost = local_usage.get("cost_usd")
    interactor_cost = task_interaction.get("accounted_interactor_cost_usd")
    if isinstance(agent_cost, (int, float)) and isinstance(
        interactor_cost, (int, float)
    ):
        local_usage["agent_cost_usd"] = float(agent_cost)
        local_usage["cost_usd"] = float(agent_cost) + float(interactor_cost)
    verifier_result = trial.get("verifier_result") or {}
    exception = trial.get("exception_info") or {}
    reward = (verifier_result.get("rewards") or {}).get("reward")
    started = _parse_time(trial.get("started_at"))
    finished = _parse_time(trial.get("finished_at"))
    wall_time = (finished - started).total_seconds() if started and finished else None
    context_events = _context_event_summary(
        trial_dir,
        gateway_event_path=meta.get("context_gateway_events_path"),
        expected_identity=meta,
    )
    evidence = _evidence_summary(
        trial_dir,
        changed_paths=meta.get("changed_paths") or [],
    )
    trajectory_activity = _trajectory_activity(trial_dir)
    inspected_paths = trajectory_activity["inspected_paths"]
    changed_paths = list(
        dict.fromkeys(
            [
                *evidence.get("changed_paths", []),
                *trajectory_activity["changed_paths"],
            ]
        )
    )
    retrieval_activity = _retrieval_to_action_activity(
        context_events["context_result_paths"],
        inspected_paths,
        changed_paths,
    )
    terminal_error = _terminal_exception_event(exception)
    agent_response = _agent_response(trial_dir)
    context_system_id = meta.get("context_system_id", "none")
    context_assigned = context_system_id != "none"
    context_registration = meta.get("context_registration") or {}
    registration_status = context_registration.get("status")
    context_registered = registration_status in {"registered", "static"}
    if "agent_execution" not in trial:
        agent_execution_status = "unknown"
    elif trial.get("agent_execution") is None:
        agent_execution_status = "not_started"
    else:
        agent_execution_status = "started"
    if registration_status is None:
        context_registered = bool(
            context_events["context_telemetry_available"]
            or meta.get("context_artifact")
        )
    return {
        "schema_version": 1,
        "record_type": "trial",
        "run_key": meta.get("run_key") or trial.get("trial_name") or trial_dir.name,
        "run_id": meta.get("run_id"),
        "trial_index": _positive_int(meta.get("trial_index")) or 1,
        "comparison_example_id": meta.get("comparison_example_id"),
        "candidate_id": meta.get("candidate_id"),
        "execution_fingerprint": meta.get("execution_fingerprint"),
        "execution_kind": meta.get("execution_kind", "agent"),
        "agent_execution_status": agent_execution_status,
        "identity_schema_version": meta.get("identity_schema_version"),
        "evaluation_scope_id": meta.get("evaluation_scope_id"),
        "job_name": meta.get("job_name") or trial_dir.parent.name,
        "task_name": trial.get("task_name"),
        "trial_name": trial.get("trial_name") or trial_dir.name,
        "harness": meta.get("harness") or (trial.get("agent_info") or {}).get("name"),
        "experiment_id": meta.get("experiment_id"),
        "workload_id": meta.get("workload_id") or "harbor",
        "preset_id": meta.get("preset_id"),
        "run_name": meta.get("run_name"),
        "run_group": meta.get("run_group"),
        "variant_id": meta.get("variant_id"),
        "prompt_id": meta.get("prompt_id"),
        "context_system_id": context_system_id,
        "context_delivery": meta.get("context_delivery", "portable"),
        "context_version": meta.get("context_version"),
        "context_support": meta.get("context_support"),
        "context_config_hash": meta.get("context_config_hash"),
        "context_cache_keys": meta.get("context_cache_keys", {}),
        "expected_artifact_paths": meta.get("expected_artifact_paths", []),
        "artifact_normalization": meta.get("artifact_normalization", []),
        "prompt_hashes": meta.get("prompt_hashes", {}),
        "skill_ids": meta.get("skill_ids", []),
        "skill_hashes": meta.get("skill_hashes", {}),
        "skill_provenance": meta.get("skill_provenance", []),
        "skills_assigned": meta.get("skills_assigned", meta.get("skill_ids", [])),
        "skills_registered": meta.get("skills_registered", []),
        "skill_registration": meta.get("skill_registration", {}),
        "skill_registration_status": (
            meta.get("skill_registration", {}).get("status")
            if isinstance(meta.get("skill_registration"), dict)
            else "unavailable"
        ),
        "skill_invocation_evidence": meta.get(
            "skill_invocation_evidence",
            {"status": "unavailable"},
        ),
        "integration_ids": meta.get("integration_ids", []),
        "integration_provenance": meta.get("integration_provenance", []),
        "harbor_config": meta.get("harbor_config"),
        "harbor_environment": meta.get("harbor_environment"),
        "harbor_resources": meta.get("harbor_resources", {}),
        "agent_config_hash": meta.get("agent_config_hash"),
        "tags": meta.get("tags", []),
        "dataset": meta.get("dataset"),
        "repository": meta.get("repository"),
        "base_commit": meta.get("base_commit"),
        "manifest_path": meta.get("manifest_path"),
        "model_provider": meta.get("model_provider"),
        "model_transport": meta.get("model_transport"),
        "builder_model": meta.get("builder_model"),
        "judge_model": meta.get("judge_model"),
        "model": meta.get("model")
        or ((trial.get("config") or {}).get("agent") or {}).get("model_name"),
        "trace_project": meta.get("trace_project")
        or (
            f"{meta.get('weave_entity')}/{meta.get('weave_project')}"
            if meta.get("weave_entity") and meta.get("weave_project")
            else None
        ),
        "reward": reward,
        "pass": reward == 1.0 if reward is not None else None,
        "wall_time_sec": wall_time,
        **local_usage,
        "exception_class": exception.get("exception_type"),
        "runtime_fingerprints": _runtime_fingerprints(trial_dir, meta),
        "context_registration": context_registration,
        "context_registration_status": registration_status or "unavailable",
        "context_registration_digest": context_registration.get("registration_digest"),
        "context_registered": context_registered if context_assigned else None,
        "context_artifact": meta.get("context_artifact"),
        "context_assigned": context_assigned,
        "context_available": context_assigned and context_registered,
        "context_invoked": context_events["context_query_count"] > 0,
        "context_invocation_evidence": {
            "status": (
                "observed"
                if context_events["context_query_count"] > 0
                else "not_observed"
            ),
            "source": (
                "mcp_gateway_event_log"
                if context_events["context_gateway_tool_call_count"] > 0
                else "local_context_events"
            ),
            "tool_calls": context_events["context_query_count"],
            "gateway_call_ids": context_events["context_gateway_call_ids"],
        },
        **context_events,
        **retrieval_activity,
        **evidence,
        "inspected_paths": inspected_paths,
        "changed_paths": changed_paths,
        "local_error_events": [
            *trajectory_activity["error_events"],
            *([terminal_error] if terminal_error else []),
        ],
        "weave_agent_name": meta.get("weave_agent_name"),
        "weave_conversation_key": meta.get("weave_conversation_key"),
        "weave_conversation_id": meta.get("weave_conversation_id"),
        "planned_conversation_id": meta.get("planned_conversation_id")
        or meta.get("weave_conversation_id"),
        "weave_conversation_ids": meta.get("weave_conversation_ids", []),
        "native_session_ids": meta.get("native_session_ids", []),
        "task_interaction": task_interaction,
        "trace_content": meta.get("trace_content", "full"),
        "agent_response": (
            agent_response if meta.get("trace_content", "full") == "full" else None
        ),
        "agent_response_sha256": (
            hashlib.sha256(agent_response.encode()).hexdigest()
            if agent_response
            else None
        ),
        "agent_response_bytes": len(agent_response.encode()) if agent_response else 0,
        "trial_dir": trial_dir.as_posix(),
    }


def _agent_response(trial_dir: Path) -> str | None:
    path = trial_dir / "agent" / "trajectory.json"
    try:
        trajectory = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    steps = trajectory.get("steps", []) if isinstance(trajectory, dict) else []
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        if str(step.get("source") or "").lower() not in {"agent", "assistant"}:
            continue
        message = step.get("message") or step.get("content") or step.get("text")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


def _runtime_fingerprints(trial_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    values = dict(meta.get("runtime_fingerprints") or {})
    for stage in ("pre_install", "verified", "pre_execution", "post_execution"):
        if stage in values:
            continue
        path = trial_dir / "agent" / f"runtime-fingerprint-{stage}.json"
        try:
            fingerprint = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(fingerprint, dict):
            values[stage] = fingerprint
    return values


def _context_result_rows(path: Path) -> list[dict[str, Any]]:
    candidates: list[Path]
    if path.is_file() and path.name == "context-results.jsonl":
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("context-results.jsonl"))
    else:
        candidates = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for line in candidate.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("trial_dir", candidate.parent.as_posix())
            rows.append(row)
    return rows


def _live_evaluation_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_file() and path.name == "evaluation-results.jsonl":
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("evaluation-results.jsonl"))
    else:
        candidates = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        lifecycle_times: dict[str, dict[str, datetime]] = {}
        events_path = candidate.with_name("evaluations.jsonl")
        if events_path.is_file():
            for line in events_path.read_text(errors="replace").splitlines():
                try:
                    event = json.loads(line)
                    recorded_at = datetime.fromisoformat(event["recorded_at"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if event.get("status") in {
                    "prediction_open",
                    "finalized",
                    "failed",
                } and event.get("cell_id"):
                    lifecycle_times.setdefault(str(event["cell_id"]), {})[
                        str(event["status"])
                    ] = recorded_at
        for line in candidate.read_text(errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                value.setdefault("evaluation_publication_mode", "live")
                times = lifecycle_times.get(str(value.get("cell_id") or ""), {})
                opened = times.get("prediction_open")
                closed = times.get("finalized") or times.get("failed")
                if (
                    value.get("evaluation_prediction_latency_sec") is None
                    and opened is not None
                    and closed is not None
                ):
                    value["evaluation_prediction_latency_sec"] = max(
                        (closed - opened).total_seconds(), 0.0
                    )
                rows.append(value)
    return rows


def _cell_result_rows(path: Path) -> list[dict[str, Any]]:
    candidates: list[Path]
    if path.is_file() and path.name == "cells.jsonl":
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("cells.jsonl"))
    else:
        candidates = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        latest: dict[str, dict[str, Any]] = {}
        for line in candidate.read_text(errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("cell_id"):
                latest[str(item["cell_id"])] = item
        for item in latest.values():
            rows.append(
                {
                    **item,
                    "record_type": "cell",
                    "task_name": item.get("task_id"),
                    "applicable": item.get("status") != "not_applicable",
                    "run_key": (
                        f"{item.get('run_id')}:{item.get('workload_id')}:cell:"
                        f"{item.get('task_id')}:{item.get('harness')}:"
                        f"{item.get('context_system_id')}:{item.get('variant_id')}:"
                        f"t{int(item.get('trial_index') or 1):03d}"
                    ),
                    "trial_dir": candidate.parent.as_posix(),
                }
            )
    return rows


def _context_event_summary(
    trial_dir: Path,
    *,
    gateway_event_path: Any = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = sorted(trial_dir.rglob("fugue-context-events.jsonl"))
    events: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    gateway_path, gateway_log_status = _safe_gateway_event_path(gateway_event_path)
    gateway_events: list[dict[str, Any]] = []
    mismatched_gateway_events = 0
    if gateway_path is not None and gateway_path.is_file():
        gateway_log_status = "available"
        for line in gateway_path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if _gateway_identity_matches(event, expected_identity or {}):
                gateway_events.append(event)
            else:
                mismatched_gateway_events += 1
    elif gateway_path is not None:
        gateway_log_status = "missing"
    proxy_responses = [
        event for event in events if event.get("event") == "mcp_tool_response"
    ]
    provider_retrievals = [
        event for event in events if event.get("event") == "retrieve"
    ]
    logical_events = proxy_responses or provider_retrievals
    gateway_calls_by_id = {
        str(event.get("gateway_call_id")): event
        for event in gateway_events
        if event.get("event") in {"tool_end", "tool_failed", "tool_cancelled"}
        and event.get("gateway_call_id")
    }
    gateway_calls = list(gateway_calls_by_id.values())
    gateway_call_ids = sorted(gateway_calls_by_id)
    vector_events = [
        value
        for event in gateway_calls
        if isinstance((value := event.get("vector")), dict)
    ]
    latencies = [
        float(
            (event.get("metrics") or {}).get("query_latency_ms")
            if (event.get("metrics") or {}).get("query_latency_ms") is not None
            else event.get("latency_ms")
        )
        for event in logical_events
        if (event.get("metrics") or {}).get("query_latency_ms") is not None
        or event.get("latency_ms") is not None
    ]
    latencies.extend(
        float(event["duration_ms"])
        for event in gateway_calls
        if event.get("duration_ms") is not None
    )
    latency_percentiles = latency_summary(latencies)
    first_context = [
        float(event["elapsed_ms"])
        for event in events
        if event.get("event") in {"retrieve", "mcp_tool_request"}
        and event.get("elapsed_ms") is not None
    ]
    result_counts = [
        int((event.get("metrics") or {}).get("result_count") or 0)
        for event in logical_events
    ]
    result_tokens = [
        int((event.get("metrics") or {}).get("result_tokens") or 0)
        for event in logical_events
    ]
    vector_result_count = sum(
        int(value.get("semantic_result_count") or 0)
        + int(value.get("bm25_result_count") or 0)
        for value in vector_events
    )
    context_result_paths = _ordered_context_result_paths(events)
    return {
        "context_telemetry_available": bool(paths or gateway_events),
        "context_event_count": len(events) + len(gateway_events),
        "context_call_count": len(logical_events) + len(gateway_calls),
        "context_query_count": len(logical_events) + len(gateway_calls),
        "context_proxy_event_count": sum(
            1 for event in events if event.get("layer") == "proxy"
        ),
        "context_upstream_event_count": sum(
            1 for event in events if event.get("layer") == "upstream"
        ),
        "context_provider_event_count": sum(
            1 for event in events if event.get("layer") == "provider"
        ),
        "context_error_count": sum(1 for event in events if event.get("error"))
        + sum(
            event.get("event") in {"tool_failed", "tool_cancelled"}
            or event.get("is_error") is True
            for event in gateway_calls
        )
        + mismatched_gateway_events,
        "context_result_count": max(sum(result_counts), len(context_result_paths))
        + vector_result_count,
        "context_result_paths": context_result_paths,
        "context_result_path_count": len(context_result_paths),
        "context_result_tokens": sum(result_tokens),
        "context_query_latency_ms": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "context_query_latency_p50_ms": latency_percentiles["p50_ms"],
        "context_query_latency_p95_ms": latency_percentiles["p95_ms"],
        "time_to_first_context_ms": min(first_context) if first_context else None,
        "context_gateway_event_log_status": gateway_log_status,
        "context_gateway_event_count": len(gateway_events),
        "context_gateway_tool_call_count": len(gateway_calls),
        "context_gateway_call_ids": gateway_call_ids,
        "context_gateway_identity_mismatch_count": mismatched_gateway_events,
        "gitnexus_vector_search_attempted": any(
            value.get("vector_search_attempted") is True for value in vector_events
        ),
        "gitnexus_vector_search_succeeded": any(
            value.get("vector_search_succeeded") is True for value in vector_events
        ),
        "gitnexus_semantic_result_count": sum(
            int(value.get("semantic_result_count") or 0) for value in vector_events
        ),
        "gitnexus_bm25_result_count": sum(
            int(value.get("bm25_result_count") or 0) for value in vector_events
        ),
        "gitnexus_vector_model_digests": sorted(
            {
                str(digest)
                for value in vector_events
                if (digest := value.get("model_digest"))
            }
        ),
        "gitnexus_vector_query_latency_ms": sum(
            float(value.get("query_latency_ms") or 0.0) for value in vector_events
        ),
    }


def _ordered_context_result_paths(events: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for event in events:
        hits = event.get("hits")
        if not isinstance(hits, list):
            continue
        for hit in hits:
            raw_path = hit.get("path") if isinstance(hit, dict) else None
            if not isinstance(raw_path, str):
                continue
            path = _normalize_repo_path(raw_path)
            if path and path not in paths:
                paths.append(path)
                if len(paths) == 200:
                    return paths
    return paths


def _retrieval_to_action_activity(
    returned_paths: list[str],
    inspected_paths: list[str],
    changed_paths: list[str],
) -> dict[str, Any]:
    returned = list(
        dict.fromkeys(
            value
            for path in returned_paths
            if (value := _normalize_repo_path(path)) is not None
        )
    )
    inspected = {
        value
        for path in inspected_paths
        if (value := _normalize_repo_path(path)) is not None
    }
    changed = {
        value
        for path in changed_paths
        if (value := _normalize_repo_path(path)) is not None
    }
    opened = [path for path in returned if path in inspected]
    modified = [path for path in returned if path in changed]
    return {
        "context_result_opened_paths": opened,
        "context_result_changed_paths": modified,
        "context_result_opened_count": len(opened),
        "context_result_changed_count": len(modified),
        "context_result_open_rate": len(opened) / len(returned) if returned else None,
        "context_result_change_rate": (
            len(modified) / len(returned) if returned else None
        ),
    }


def _safe_gateway_event_path(value: Any) -> tuple[Path | None, str]:
    if not isinstance(value, str) or not value.strip():
        return None, "not_configured"
    path = Path(value)
    parts = path.parts
    try:
        fugue_index = parts.index(".fugue")
    except ValueError:
        return None, "rejected"
    if (
        not path.is_absolute()
        or path.name != "context-gateway.jsonl"
        or parts[fugue_index + 1 : fugue_index + 2] != ("runtime",)
    ):
        return None, "rejected"
    runtime_root = Path(*parts[: fugue_index + 2]).resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(runtime_root):
        return None, "rejected"
    return resolved, "configured"


def _gateway_identity_matches(
    event: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    fields = {
        "fugue_run_id": "run_id",
        "fugue_candidate_id": "candidate_id",
        "fugue_comparison_example_id": "comparison_example_id",
        "fugue_trial_index": "trial_index",
        "fugue_execution_fingerprint": "execution_fingerprint",
        "fugue_context_system_id": "context_system_id",
    }
    for event_key, expected_key in fields.items():
        expected_value = expected.get(expected_key)
        if expected_value in (None, ""):
            continue
        if str(event.get(event_key) or "") != str(expected_value):
            return False
    return True


def _evidence_summary(
    trial_dir: Path,
    *,
    changed_paths: list[str],
) -> dict[str, Any]:
    authored: list[str] = []
    for path in trial_dir.rglob("fugue-evidence.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        values = (
            payload.get("paths", [])
            if isinstance(payload, dict)
            else payload
            if isinstance(payload, list)
            else []
        )
        for value in values[:100]:
            item = value.get("path") if isinstance(value, dict) else value
            if item:
                authored.append(str(item)[:1_000])
    activity = _trajectory_activity(trial_dir)
    changed = [
        value
        for value in (_normalize_repo_path(item) for item in changed_paths)
        if value
    ]
    observed = list(dict.fromkeys([*activity["inspected_paths"], *changed, *authored]))
    return {
        "evidence_paths": observed,
        "agent_evidence_paths": list(dict.fromkeys(authored)),
        "changed_paths": list(dict.fromkeys(changed)),
    }


_PATH_ARGUMENTS = {"path", "file_path", "filepath", "filename"}
_READ_TOOLS = {"read", "read_file", "grep", "search", "search_files", "glob"}
_WRITE_TOOLS = {"write", "write_file", "edit", "patch", "apply_patch"}
_COMMAND_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:/testbed/|/workspace/repo/|\./)?"
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
)


def _trajectory_activity(trial_dir: Path) -> dict[str, Any]:
    path = trial_dir / "agent" / "trajectory.json"
    try:
        trajectory = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"inspected_paths": [], "changed_paths": [], "error_events": []}
    inspected: list[str] = []
    changed: list[str] = []
    errors: list[dict[str, Any]] = []
    steps = trajectory.get("steps", []) if isinstance(trajectory, dict) else []
    for step in steps:
        if not isinstance(step, dict):
            continue
        results = {
            str(result.get("source_call_id") or ""): result
            for result in ((step.get("observation") or {}).get("results") or [])
            if isinstance(result, dict)
        }
        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            tool_name = str(
                call.get("function_name") or call.get("tool_name") or "unknown"
            )
            arguments = call.get("arguments") or {}
            paths = _paths_from_tool_arguments(arguments)
            normalized_name = tool_name.lower()
            if normalized_name in _WRITE_TOOLS:
                changed.extend(paths)
            elif normalized_name in _READ_TOOLS:
                inspected.extend(paths)
            if isinstance(arguments, dict):
                command = arguments.get("command")
                if isinstance(command, str):
                    inspected.extend(_paths_from_command(command))
            call_id = str(call.get("tool_call_id") or call.get("id") or "")
            result = results.get(call_id)
            if result and _local_tool_result_failed(result):
                errors.append(
                    _classify_error(
                        str(result.get("content") or "tool call failed"),
                        tool_name=tool_name,
                        operation="execute_tool",
                        source="local_trajectory",
                        event_key=call_id,
                    )
                )
    return {
        "inspected_paths": list(dict.fromkeys(inspected)),
        "changed_paths": list(dict.fromkeys(changed)),
        "error_events": errors,
    }


def _terminal_exception_event(exception: dict[str, Any]) -> dict[str, Any] | None:
    exception_type = str(exception.get("exception_type") or "").strip()
    message = str(exception.get("exception_message") or "").strip()
    traceback = str(exception.get("exception_traceback") or "")
    if not exception_type and not message:
        return None
    lowered_traceback = traceback.lower()
    if "_setup_agent" in lowered_traceback or (
        "harbor/agents/installed/" in lowered_traceback
        and " in install" in lowered_traceback
    ):
        operation = "adapter_setup"
    elif "verifier" in lowered_traceback:
        operation = "verifier"
    elif "fugue/" in lowered_traceback and "invoke" not in lowered_traceback:
        operation = "framework"
    else:
        operation = "invoke_agent"
    return _classify_error(
        f"{exception_type}: {message}".strip(": "),
        tool_name="",
        operation=operation,
        source="harbor_trial",
        terminal=True,
        event_key=exception_type,
    )


def _paths_from_tool_arguments(arguments: Any) -> list[str]:
    if not isinstance(arguments, dict):
        return []
    paths: list[str] = []
    for key, value in arguments.items():
        if key.lower() not in _PATH_ARGUMENTS or not isinstance(value, str):
            continue
        normalized = _normalize_repo_path(value)
        if normalized:
            paths.append(normalized)
    return paths


def _paths_from_command(command: str) -> list[str]:
    return list(
        dict.fromkeys(
            value
            for value in (
                _normalize_repo_path(match.group(0))
                for match in _COMMAND_PATH_RE.finditer(command)
            )
            if value
        )
    )


def _normalize_repo_path(value: str) -> str | None:
    path = value.strip().strip("'\"")
    for prefix in ("/testbed/", "/workspace/repo/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    path = path.removeprefix("./")
    if not path or path.startswith(("/", "../", "/logs/", ".fugue-context/")):
        return None
    return path[:1_000]


def _local_tool_result_failed(result: dict[str, Any]) -> bool:
    extra = result.get("extra") or {}
    metadata = extra.get("tool_result_metadata") or {}
    raw = metadata.get("raw_tool_result") or {}
    return bool(
        extra.get("tool_result_is_error")
        or metadata.get("tool_result_is_error")
        or raw.get("is_error")
        or "[error] tool reported failure" in str(result.get("content") or "").lower()
    )


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _weave_project_from_env(env: Mapping[str, str] | None = None) -> str:
    return trace_project_slug(env if env is not None else os.environ)
