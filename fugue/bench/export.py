from __future__ import annotations

import hashlib
import json
import keyword
import math
import os
import re
import secrets
import threading
import time
import urllib.parse
import uuid
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
from fugue.mcp_evidence import (
    safe_graphql_query_shape,
    safe_structured_error_code,
    validated_graphql_query_shape,
)
from fugue.model_plane import (
    inference_project_slug,
    trace_api_key,
    trace_destination_identity,
    trace_project_environment,
    trace_project_slug,
)
from fugue.redaction import redact_value, secrets_from_env
from fugue.weave_support import WEAVE_AGENTS_BASE_URL, initialize_weave

PREDICTION_SCHEMA_VERSION = 1
PUBLICATION_SCHEMA_VERSION = 1
_PROMPT_INJECTION_REWARDS = (
    "safe_and_useful",
    "safe_but_failed_or_refused",
    "compromised",
    "incorrect",
    "task_complete",
    "false_positive_refusal",
    "evidence_preserved",
)
_PROMPT_INJECTION_OPTIONAL_REWARDS = (
    "attack_encountered",
    "sensitive_action_attempted",
    "action_gate_blocked",
    "action_gate_allowed",
)
_EVIDENCE_USE_REWARDS = (
    "artifact_schema_valid",
    "answer_facts_correct",
    "current_document_cited",
    "current_document_used",
    "unsupported_claims_absent",
)


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
    project: str | None = None
    url: str | None = None
    evaluation_ref: str | None = None
    dataset_ref: str | None = None
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
    bridge_call: Any | None
    bridge_client: Any | None
    row: dict[str, Any]
    opened_monotonic: float
    bridge_finished: bool = False


class _TracePollingCancelled(Exception):
    pass


_WEAVE_CALL_SOURCE = "weave_calls"
_WEAVE_AGENT_SPAN_SOURCE = "weave_agents"


def _eager_call_op(weave_module: Any, name: str) -> Any:
    """Create a public Weave op whose Call start is sent immediately."""

    op = getattr(weave_module, "op", None)
    if not callable(op):
        # Lightweight test doubles may only accept anonymous op names.
        return name

    def evidence_boundary(**values: Any) -> dict[str, Any]:
        return values

    return op(
        name=name,
        enable_code_capture=False,
        eager_call_start=True,
    )(evidence_boundary)


def _enable_eager_evaluation_starts(logger: Any) -> bool:
    """Expose the exact open prediction Calls without closing their scores."""

    pseudo_evaluation = getattr(logger, "_pseudo_evaluation", None)
    values = (
        getattr(pseudo_evaluation, "predict_and_score", None),
        getattr(logger, "_context_predict_method", None),
    )
    configured = 0
    for value in values:
        operation = getattr(value, "__func__", value)
        if operation is None or not hasattr(operation, "eager_call_start"):
            continue
        operation.eager_call_start = True
        if operation.eager_call_start is True:
            configured += 1
    # Test doubles without Weave internals do not publish Calls. A real
    # EvaluationLogger must expose both predict-and-score and prediction ops.
    if type(logger).__module__.startswith("weave."):
        return configured == 2
    return True


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
        host_evaluator: Callable[[dict[str, Any]], None] | None = None,
        host_scorer_names: tuple[str, ...] = (),
        evidence_checkpoint_cells: int = 0,
        checkpoint_conformance: (
            Callable[[PlannedCell], Mapping[str, Any]] | None
        ) = None,
    ) -> None:
        if not trace_api_key(env):
            raise RuntimeError(
                "FUGUE_WEAVE_API_KEY is required for live evaluations"
            )
        self.repo_root = repo_root
        self.project = project
        self.env = trace_project_environment(project, env)
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
        self._host_evaluator = host_evaluator
        if evidence_checkpoint_cells < 0:
            raise ValueError("evidence checkpoint cells must be non-negative")
        self._evidence_checkpoint_cells = evidence_checkpoint_cells
        self._evidence_checkpoint_terminal = 0
        self._checkpoint_conformance = checkpoint_conformance
        self._evidence_destination = trace_destination_identity(self.env)
        if self._evidence_destination["project_slug"] != project:
            raise RuntimeError(
                "live Evaluation destination disagrees with its project"
            )
        self._summary_fetcher = summary_fetcher or fetch_weave_summaries
        configured_timeout = self.env.get("FUGUE_WEAVE_LINK_TIMEOUT_SEC")
        self.trace_timeout_sec = (
            trace_timeout_sec
            if trace_timeout_sec is not None
            else float(configured_timeout or 45)
        )
        self.weave = weave_module or initialize_weave(project, env)
        self._agent_bridge_op = _eager_call_op(
            self.weave,
            "fugue.agent_execution_bridge",
        )
        self._native_agent_root_op = _eager_call_op(
            self.weave,
            "fugue.claude_code_native_agent_root",
        )
        logger_cls = getattr(self.weave, "EvaluationLogger", None)
        if logger_cls is None:
            raise RuntimeError("installed weave package has no EvaluationLogger")
        dataset_cls = getattr(self.weave, "Dataset", None)
        planned = [
            _planned_evaluation_row(cell)
            for cell in cells
            if cell.applicable and cell.execution_kind == "agent"
        ]
        for row in planned:
            row["host_scorer_names"] = list(host_scorer_names)
        candidates = _publication_candidates(planned)
        datasets: dict[str, Any] = {}
        self._datasets = datasets
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
                    scorers=_weave_predefined_scorer_names(
                        candidate["scorers"]
                    ),
                )
            if not _enable_eager_evaluation_starts(logger):
                raise RuntimeError(
                    "installed Weave EvaluationLogger cannot expose exact "
                    "prediction Call starts before terminal scoring"
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
        row = _planned_evaluation_row(cell)
        prediction = None
        prediction_entered = False
        bridge_call = None
        bridge_client = None
        attributes = getattr(self.weave, "attributes", None)
        context = (
            attributes(
                {
                    "fugue.attempt_id": cell.attempt_id,
                    "fugue.cell_id": cell.id,
                    "fugue.variant_id": cell.variant_id,
                    "fugue.candidate_id": cell.candidate_id,
                    "fugue.execution_fingerprint": cell.execution_fingerprint,
                    "fugue.experiment_id": cell.env.get("FUGUE_EXPERIMENT_ID", ""),
                    "wandb.research_id": cell.env.get(
                        "FUGUE_WANDB_RESEARCH_ID", ""
                    ),
                    "wandb.study_id": cell.env.get("FUGUE_WANDB_STUDY_ID", ""),
                    "fugue.source_evidence_project": cell.env.get(
                        "FUGUE_SOURCE_EVIDENCE_PROJECT", ""
                    ),
                    "fugue.result_evidence_project": cell.env.get(
                        "FUGUE_RESULT_EVIDENCE_PROJECT", ""
                    ),
                    "fugue.study_console_backlink": cell.env.get(
                        "FUGUE_STUDY_CONSOLE_BACKLINK", ""
                    ),
                }
            )
            if attributes is not None
            else nullcontext()
        )
        try:
            with session.lock:
                with context:
                    prediction = session.logger.log_prediction(
                        inputs=self._inputs_by_cell[cell.id]
                    )
                    prediction.__enter__()
                    prediction_entered = True
            dataset = self._datasets[session.candidate["evaluation_scope_id"]]
            _apply_evaluation_evidence(
                row,
                logger=session.logger,
                dataset=dataset,
                project=self.project,
            )
            predict_call = getattr(prediction, "predict_call", None)
            if predict_call is not None:
                _apply_call_evidence(
                    row,
                    prefix="weave_prediction",
                    call=predict_call,
                    project=self.project,
                )
            call = prediction.predict_and_score_call
            _apply_call_evidence(
                row,
                prefix="eval_predict_and_score",
                call=call,
                project=self.project,
            )
            _verify_live_evaluation_graph(
                row,
                logger=session.logger,
                dataset=dataset,
                prediction=prediction,
                project=self.project,
            )
            if (
                cell.harness == "claude-code"
                and row.get("evaluation_prediction_graph_verified") is not True
            ):
                raise RuntimeError(
                    "live Evaluation graph is unresolved before Claude execution"
                )
            traceparent = ""
            if cell.harness == "claude-code":
                bridge_call, bridge_client, traceparent = self._open_agent_bridge(
                    cell=cell,
                    row=row,
                    prediction=prediction,
                )
        except Exception as exc:
            if self._evidence_checkpoint_cells:
                self._cancellation_event.set()
                self._append_event(
                    "evidence_checkpoint_failed",
                    cell_id=cell.id,
                    candidate_id=cell.candidate_id,
                    failures=[
                        "live Evaluation could not open its authoritative graph"
                    ],
                )
            if bridge_call is not None and bridge_client is not None:
                failed_active = _LivePrediction(
                    session=session,
                    prediction=prediction,
                    bridge_call=bridge_call,
                    bridge_client=bridge_client,
                    row=row,
                    opened_monotonic=time.monotonic(),
                )
                self._finish_agent_bridge(failed_active, status="start_failed")
            if prediction is not None and prediction_entered:
                row.update(
                    {
                        "status": "failed",
                        "pass": False,
                        "trace_link_status": "failed",
                        "trace_link_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                try:
                    prediction.output = _evaluation_output(row)
                except Exception:
                    pass
                try:
                    with session.lock:
                        prediction.__exit__(type(exc), exc, exc.__traceback__)
                except Exception:
                    # The original start failure remains authoritative. A
                    # broken SDK close must not leave the cell eligible to run.
                    pass
            self._append_event(
                "prediction_start_failed",
                cell_id=cell.id,
                candidate_id=cell.candidate_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        with self._prediction_lock:
            self._predictions[cell.id] = _LivePrediction(
                session=session,
                prediction=prediction,
                bridge_call=bridge_call,
                bridge_client=bridge_client,
                row=row,
                opened_monotonic=time.monotonic(),
            )
        call_id = str(call.id)
        self._append_event(
            "prediction_open",
            cell_id=cell.id,
            candidate_id=cell.candidate_id,
            eval_predict_and_score_call_id=call_id,
        )
        overlay = {
            "FUGUE_ATTEMPT_ID": cell.attempt_id,
            "FUGUE_WEAVE_EVAL_PREDICT_AND_SCORE_CALL_ID": call_id,
            "FUGUE_WEAVE_EVAL_PROJECT_ID": str(call.project_id),
            "FUGUE_WEAVE_EVAL_NAME": _evaluation_name(session.candidate),
            "FUGUE_EVALUATION_SCOPE_ID": session.candidate["evaluation_scope_id"],
        }
        if traceparent:
            overlay["FUGUE_WEAVE_TRACEPARENT"] = traceparent
        return overlay

    def _open_agent_bridge(
        self,
        *,
        cell: PlannedCell,
        row: dict[str, Any],
        prediction: Any,
    ) -> tuple[Any, Any, str]:
        """Create the real Call bridge used to parent Claude's remote OTel root."""

        get_client = getattr(self.weave, "get_client", None)
        client = get_client() if callable(get_client) else None
        if client is None or not callable(getattr(client, "create_call", None)):
            raise RuntimeError(
                "installed Weave runtime cannot create the Agent ancestry bridge"
            )
        parent = getattr(prediction, "predict_call", None)
        if parent is None:
            raise RuntimeError("live Evaluation prediction Call is unavailable")
        bridge_id = secrets.token_hex(8)
        bridge = client.create_call(
            getattr(
                self,
                "_agent_bridge_op",
                "fugue.agent_execution_bridge",
            ),
            {
                "attempt_id": cell.attempt_id,
                "cell_id": cell.id,
                "candidate_id": cell.candidate_id,
                "execution_fingerprint": cell.execution_fingerprint,
            },
            parent=parent,
            attributes={
                "fugue.attempt_id": cell.attempt_id,
                "fugue.cell_id": cell.id,
                "fugue.run_key": str(row.get("run_key") or ""),
                "fugue.harness": cell.harness,
                "fugue.task_id": cell.task_id,
                "fugue.candidate_id": cell.candidate_id,
                "fugue.execution_fingerprint": cell.execution_fingerprint,
                "fugue.comparison_example_id": cell.comparison_example_id,
                "fugue.trial_index": cell.trial_index,
                "fugue.experiment_id": cell.env.get("FUGUE_EXPERIMENT_ID", ""),
                "wandb.research_id": cell.env.get(
                    "FUGUE_WANDB_RESEARCH_ID", ""
                ),
                "wandb.study_id": cell.env.get("FUGUE_WANDB_STUDY_ID", ""),
                "fugue.source_evidence_project": cell.env.get(
                    "FUGUE_SOURCE_EVIDENCE_PROJECT", ""
                ),
                "fugue.result_evidence_project": cell.env.get(
                    "FUGUE_RESULT_EVIDENCE_PROJECT", ""
                ),
                "fugue.study_console_backlink": cell.env.get(
                    "FUGUE_STUDY_CONSOLE_BACKLINK", ""
                ),
                "fugue.agent_bridge": True,
                "fugue.agent_bridge_version": 1,
                "weave.eval.predict_and_score_call_id": str(
                    row.get("eval_predict_and_score_call_id") or ""
                ),
            },
            display_name=f"Fugue Agent bridge · {cell.task_id}",
            use_stack=False,
            _call_id_override=bridge_id,
        )
        parent_id = _live_call_value(bridge, "parent_id")
        project_id = _live_call_value(bridge, "project_id")
        trace_id = _live_call_value(bridge, "trace_id")
        expected_parent = _live_call_value(parent, "id")
        expected_trace = _live_call_value(parent, "trace_id")
        otel_trace_id = _w3c_trace_id(trace_id)
        if (
            _live_call_value(bridge, "id") != bridge_id
            or parent_id != expected_parent
            or project_id != self.project
            or not _same_nonempty_trace(trace_id, expected_trace)
            or otel_trace_id is None
        ):
            try:
                client.finish_call(
                    bridge,
                    output={"status": "invalid_bridge"},
                )
            finally:
                raise RuntimeError(
                    "Weave Agent ancestry bridge did not preserve prediction ownership"
                )
        try:
            row.update(
                {
                    "weave_agent_bridge_call_id": bridge_id,
                    "weave_agent_bridge_parent_id": parent_id,
                    "weave_agent_bridge_project_id": project_id,
                    "weave_agent_bridge_trace_id": trace_id,
                    "weave_agent_bridge_otel_trace_id": otel_trace_id,
                }
            )
            _apply_call_evidence(
                row,
                prefix="weave_agent_bridge",
                call=bridge,
                project=self.project,
            )
            row["weave_agent_bridge_object_verified"] = True
        except Exception:
            try:
                client.finish_call(
                    bridge,
                    output={"status": "start_failed"},
                )
            except Exception:
                pass
            raise
        return (
            bridge,
            client,
            f"00-{otel_trace_id}-{bridge_id}-01",
        )

    @staticmethod
    def _finish_agent_bridge(
        active: _LivePrediction,
        *,
        status: str,
        native_root: Mapping[str, Any] | None = None,
        terminal_row: dict[str, Any] | None = None,
    ) -> None:
        def sync_terminal_row() -> None:
            if terminal_row is None or terminal_row is active.row:
                return
            for key in (
                "weave_agent_bridge_close_status",
                "weave_agent_bridge_closed_verified",
                "weave_agent_bridge_close_error",
                "weave_agent_root_call_otel_trace_id",
                "weave_agent_root_call_otel_span_id",
            ):
                if key in active.row:
                    terminal_row[key] = active.row[key]
                else:
                    terminal_row.pop(key, None)

        if (
            active.bridge_finished
            or active.bridge_call is None
            or active.bridge_client is None
        ):
            sync_terminal_row()
            return
        active.bridge_finished = True
        active.row["weave_agent_bridge_close_status"] = status
        output = {"status": status}
        if native_root is not None:
            output.update(
                {
                    "native_conversation_id": native_root.get(
                        "conversation_id"
                    ),
                    "native_otel_trace_id": native_root.get("trace_id"),
                    "native_otel_span_id": native_root.get("span_id"),
                }
            )
        try:
            active.bridge_client.finish_call(
                active.bridge_call,
                output=output,
            )
        except Exception as exc:
            active.row["weave_agent_bridge_closed_verified"] = False
            active.row["weave_agent_bridge_close_error"] = type(exc).__name__
            sync_terminal_row()
            return
        active.row["weave_agent_bridge_closed_verified"] = True
        if native_root is not None:
            active.row["weave_agent_root_call_otel_trace_id"] = str(
                native_root.get("trace_id") or ""
            )
            active.row["weave_agent_root_call_otel_span_id"] = str(
                native_root.get("span_id") or ""
            )
        active.row.pop("weave_agent_bridge_close_error", None)
        sync_terminal_row()

    def _materialize_native_agent_call(
        self,
        *,
        active: _LivePrediction,
        row: dict[str, Any],
        root: Mapping[str, Any],
    ) -> None:
        """Create one durable Weave Call receipt for a verified Claude OTel root."""

        client = active.bridge_client
        bridge = active.bridge_call
        if (
            client is None
            or bridge is None
            or not callable(getattr(client, "create_call", None))
            or not callable(getattr(client, "finish_call", None))
        ):
            raise RuntimeError(
                "installed Weave runtime cannot materialize the native Agent Call"
            )
        call_id = _native_agent_call_id(row, root)
        attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.conversation.id": str(root["conversation_id"]),
            "fugue.native_agent_root_receipt": True,
            "fugue.native_agent_root_receipt_version": 1,
            "fugue.run_key": str(row["run_key"]),
            "fugue.harness": str(row["harness"]),
            "fugue.task_id": str(row["task_id"]),
            "fugue.candidate_id": str(row["candidate_id"]),
            "fugue.attempt_id": str(row["attempt_id"]),
            "fugue.execution_fingerprint": str(
                row["execution_fingerprint"]
            ),
            "fugue.comparison_example_id": str(
                row["comparison_example_id"]
            ),
            "fugue.trial_index": int(row["trial_index"]),
            "fugue.experiment_id": row.get("experiment_id"),
            "wandb.research_id": row.get("wandb_research_id"),
            "wandb.study_id": row.get("wandb_study_id"),
            "fugue.source_evidence_project": row.get(
                "source_evidence_project"
            ),
            "fugue.result_evidence_project": row.get(
                "result_evidence_project"
            ),
            "fugue.study_console_backlink": row.get(
                "study_console_backlink"
            ),
            "weave.eval.predict_and_score_call_id": str(
                row["eval_predict_and_score_call_id"]
            ),
            "fugue.native_otel_trace_id": str(root["trace_id"]),
            "fugue.native_otel_span_id": str(root["span_id"]),
        }
        call = client.create_call(
            getattr(
                self,
                "_native_agent_root_op",
                "fugue.claude_code_native_agent_root",
            ),
            {
                "attempt_id": str(row["attempt_id"]),
                "conversation_id": str(root["conversation_id"]),
                "native_otel_trace_id": str(root["trace_id"]),
                "native_otel_span_id": str(root["span_id"]),
            },
            parent=bridge,
            attributes=attributes,
            display_name=f"Claude Code Agent root · {row['task_id']}",
            use_stack=False,
            _call_id_override=call_id,
        )
        if (
            _live_call_value(call, "id") != call_id
            or _live_call_value(call, "parent_id")
            != _live_call_value(bridge, "id")
            or _live_call_value(call, "project_id") != self.project
            or not _same_nonempty_trace(
                _live_call_value(call, "trace_id"),
                _live_call_value(bridge, "trace_id"),
            )
        ):
            try:
                client.finish_call(
                    call,
                    output={"status": "invalid_native_root_receipt"},
                )
            finally:
                raise RuntimeError(
                    "native Agent Call did not preserve the Evaluation ancestry"
                )
        client.finish_call(
            call,
            output={
                "status": str(row.get("status") or "completed"),
                "response_sha256": row.get("agent_response_sha256"),
                "native_otel_trace_id": str(root["trace_id"]),
                "native_otel_span_id": str(root["span_id"]),
            },
        )
        row.update(
            {
                "weave_agent_root_call_id": call_id,
                "weave_agent_root_call_materialization_source": (
                    "verified_native_otel_root_v1"
                ),
                "weave_agent_root_call_otel_trace_id": str(root["trace_id"]),
                "weave_agent_root_call_otel_span_id": str(root["span_id"]),
                "weave_agent_root_call_object_created": True,
            }
        )

    def _record_verified_agent_root(
        self,
        *,
        active: _LivePrediction,
        cell: PlannedCell,
        row: dict[str, Any],
        root: Mapping[str, Any],
        predict_and_score_call_id: str,
        attach_span_ref: bool,
    ) -> None:
        _apply_verified_agent_evidence(
            row,
            root,
            project=self.project,
        )
        if attach_span_ref:
            _attach_genai_span_ref(
                active.prediction.predict_and_score_call,
                trace_id=str(root["trace_id"]),
                span_id=str(root["span_id"]),
            )
        row["trace_link_status"] = "linked"
        row["trace_link_error"] = None
        _apply_agent_graph_verification(row, root)
        self._append_event(
            "trace_linked",
            cell_id=cell.id,
            candidate_id=cell.candidate_id,
            observed_conversation_id=row.get("observed_conversation_id"),
            trace_id=root.get("trace_id"),
            root_span_id=root.get("span_id"),
            eval_predict_and_score_call_id=predict_and_score_call_id,
        )

    def _prepare_terminal_agent_trace(
        self,
        *,
        active: _LivePrediction,
        cell: PlannedCell,
        row: dict[str, Any],
        predict_and_score_call_id: str,
    ) -> None:
        """Resolve final native trace evidence before persisting Evaluation output."""

        if row.get("agent_execution_status") == "not_started":
            self._finish_agent_bridge(
                active,
                status="not_started",
                terminal_row=row,
            )
            _mark_agent_execution_not_started(row)
            return
        if cell.harness != "claude-code":
            self._finish_agent_bridge(
                active,
                status="agent_completed",
                terminal_row=row,
            )
            _apply_trace_summary(row, self._wait_for_trace(row))
            return
        # Claude exports a native OTel root to the Agents endpoint, not a
        # Weave Call. Verify it first, then materialize one durable child Call
        # under the still-open Evaluation bridge. OTel IDs remain diagnostic.
        _apply_trace_summary(
            row,
            self._wait_for_trace(
                row,
                require_authoritative_graph=False,
            ),
        )
        _apply_observed_identity(row)
        native_root = _verified_native_otel_root(
            row,
            predict_and_score_call_id,
        )
        if native_root is None:
            raise RuntimeError(
                row.get("trace_link_error")
                or "native Claude OTel root did not reconcile"
            )
        if not native_root.get("weave_call_id"):
            self._materialize_native_agent_call(
                active=active,
                row=row,
                root=native_root,
            )
        _attach_genai_span_ref(
            active.prediction.predict_and_score_call,
            trace_id=str(native_root["trace_id"]),
            span_id=str(native_root["span_id"]),
        )
        self._finish_agent_bridge(
            active,
            status="agent_completed",
            terminal_row=row,
        )
        # Fugue initializes the host Weave client with separate start/end
        # publication, so the still-open Evaluation Calls become queryable
        # without persisting a provisional prediction output first.
        _apply_trace_summary(row, self._wait_for_trace(row))
        _merge_error_events(row)
        _apply_observed_identity(row)
        _verify_authoritative_agent_graph(row)
        verified_root = _verified_evaluation_root(
            row,
            predict_and_score_call_id,
        )
        if verified_root is None:
            raise RuntimeError(
                row.get("trace_link_error")
                or "native Claude Weave Call did not reconcile"
            )
        self._record_verified_agent_root(
            active=active,
            cell=cell,
            row=row,
            root=verified_root,
            predict_and_score_call_id=predict_and_score_call_id,
            attach_span_ref=False,
        )

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
        row["eval_predict_and_score_call_id"] = call_id
        if outcome.status in {"cancelled", "interrupted"}:
            self._cancel_prediction(
                cell.id,
                active,
                row,
                reason=outcome.error or "Run cancelled by the operator.",
            )
            return
        owns_prediction = False
        prediction_closed = False
        session_row_appended = False
        try:
            self._prepare_terminal_agent_trace(
                active=active,
                cell=cell,
                row=row,
                predict_and_score_call_id=call_id,
            )
            self._raise_if_cancelled()
            _merge_error_events(row)
            _apply_observed_identity(row)
            if row.get("trace_link_status") != "linked":
                _verify_authoritative_agent_graph(row)
                root = (
                    None
                    if row.get("trace_link_status")
                    in {"not_applicable", "not_started"}
                    else _verified_evaluation_root(row, call_id)
                )
                if root is not None:
                    self._record_verified_agent_root(
                        active=active,
                        cell=cell,
                        row=row,
                        root=root,
                        predict_and_score_call_id=call_id,
                        attach_span_ref=True,
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
            self._apply_host_evaluator(row)
            _set_adapter_outcome(row)
            if not self._pop_prediction(cell.id, active):
                return
            owns_prediction = True
            active.prediction.output = _evaluation_output(row)
            with active.session.lock:
                for name, value in _evaluation_scores(row).items():
                    active.prediction.log_score(name, value)
                active.prediction.__exit__(None, None, None)
                prediction_closed = True
            with active.session.lock:
                active.session.rows.append(row)
                session_row_appended = True
            self._terminal_cells.add(cell.id)
            self._apply_evidence_checkpoint(cell, row)
            self._append_result(row)
            self._append_event(
                "finalized",
                cell_id=cell.id,
                candidate_id=cell.candidate_id,
                trace_link_status=row.get("trace_link_status"),
                eval_predict_and_score_call_id=call_id,
            )
        except _TracePollingCancelled:
            self._finish_agent_bridge(
                active,
                status="cancelled",
                terminal_row=row,
            )
            self._cancel_prediction(
                cell.id,
                active,
                row,
                reason="Run cancelled by the operator.",
            )
        except Exception as exc:
            self._finish_agent_bridge(
                active,
                status="failed",
                terminal_row=row,
            )
            if not owns_prediction and not self._pop_prediction(cell.id, active):
                return
            row["trace_link_status"] = "failed"
            row["trace_link_error"] = f"{type(exc).__name__}: {exc}"
            if not row.get("host_evaluator_status"):
                self._apply_host_evaluator(row)
            _set_adapter_outcome(row)
            try:
                if not prediction_closed:
                    active.prediction.output = _evaluation_output(row)
                with active.session.lock:
                    if not prediction_closed:
                        for name, value in _evaluation_scores(row).items():
                            active.prediction.log_score(name, value)
                        active.prediction.__exit__(None, None, None)
                        prediction_closed = True
                    if not session_row_appended:
                        active.session.rows.append(row)
                        session_row_appended = True
                self._terminal_cells.add(cell.id)
                self._apply_evidence_checkpoint(cell, row)
                self._append_result(row)
            finally:
                self._append_event(
                    "failed",
                    cell_id=cell.id,
                    candidate_id=cell.candidate_id,
                    error=row["trace_link_error"],
                    eval_predict_and_score_call_id=call_id,
                )

    def _apply_host_evaluator(self, row: dict[str, Any]) -> None:
        if self._host_evaluator is None:
            row["host_evaluator_status"] = "not_required"
            return
        try:
            self._host_evaluator(row)
            row["host_evaluator_status"] = "passed"
        except Exception as exc:
            row.update(
                {
                    "host_evaluator_status": "failed",
                    "comparison_evaluation_status": "unavailable",
                    "comparison_evaluation_reason": (
                        "host evaluation failed: "
                        f"{type(exc).__name__}"
                    ),
                    "comparison_required_evaluation_complete": False,
                }
            )

    def _apply_evidence_checkpoint(
        self,
        cell: PlannedCell,
        row: dict[str, Any],
    ) -> None:
        if (
            not self._evidence_checkpoint_cells
            or self._evidence_checkpoint_terminal
            >= self._evidence_checkpoint_cells
        ):
            return
        self._evidence_checkpoint_terminal += 1
        if self._checkpoint_conformance is None:
            row["local_cell_conformance"] = {
                "status": "unavailable",
                "reason": "no exact per-cell Harbor conformance callback was configured",
            }
        else:
            try:
                row["local_cell_conformance"] = dict(
                    self._checkpoint_conformance(cell)
                )
            except Exception as exc:
                row["local_cell_conformance"] = {
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        row["negative_routing_receipt"] = self._negative_routing_receipt(row)
        failures = _live_evidence_checkpoint_failures(
            row,
            expected_destination=self._evidence_destination,
            host_evaluator_required=self._host_evaluator is not None,
        )
        if failures:
            self._cancellation_event.set()
            row["evidence_checkpoint_status"] = "failed"
            row["evidence_checkpoint_failures"] = failures
            self._append_event(
                "evidence_checkpoint_failed",
                cell_id=row.get("cell_id"),
                candidate_id=row.get("candidate_id"),
                checkpoint_index=self._evidence_checkpoint_terminal,
                failures=failures,
            )
            return
        row["evidence_checkpoint_status"] = "passed"
        self._append_event(
            "evidence_checkpoint_passed",
            cell_id=row.get("cell_id"),
            candidate_id=row.get("candidate_id"),
            checkpoint_index=self._evidence_checkpoint_terminal,
        )

    def _negative_routing_receipt(self, row: Mapping[str, Any]) -> dict[str, Any]:
        configured = [
            value.strip()
            for value in self.env.get(
                "FUGUE_EVIDENCE_NEGATIVE_PROJECTS", ""
            ).split(",")
            if value.strip()
        ]
        projects = sorted(
            {
                "wandb/news-research-agent",
                "wandb/fugue-aria-loop-engineering-v1",
                *configured,
            }
            - {self.project}
        )
        run_key = str(row.get("run_key") or "")
        conversations = list(
            dict.fromkeys(
                str(value)
                for value in (
                    row.get("planned_conversation_id"),
                    row.get("observed_conversation_id"),
                    *(row.get("native_session_ids") or ()),
                )
                if value
            )
        )
        agent_call_id = str(row.get("weave_agent_root_call_id") or "")
        matches: list[str] = []
        errors: list[str] = []
        for project in projects:
            try:
                result = self._summary_fetcher(
                    run_keys=[run_key],
                    conversation_ids_by_run={run_key: conversations},
                    call_ids_by_run={
                        run_key: [agent_call_id] if agent_call_id else []
                    },
                    project=project,
                    timeout_sec=min(max(self.trace_timeout_sec, 1), 10),
                    env=self.env,
                ).get(run_key, {})
            except Exception as exc:
                errors.append(f"{project}: {type(exc).__name__}")
                continue
            if int(result.get("weave_span_count") or 0) > 0 or (
                result.get("weave_authoritative_call_graph")
            ):
                matches.append(project)
        return {
            "schema_version": 1,
            "status": (
                "failed"
                if matches
                else "unavailable"
                if errors
                else "passed"
            ),
            "target_project": self.project,
            "queried_projects": projects,
            "query_identity": {
                "run_key": run_key,
                "conversation_ids": conversations,
                "agent_call_id": agent_call_id or None,
            },
            "matching_projects": matches,
            "query_errors": errors,
        }

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
        self._finish_agent_bridge(
            active,
            status="cancelled",
            terminal_row=row,
        )
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
            dataset_ref = _object_ref(
                self._datasets[session.candidate["evaluation_scope_id"]]
            )
            model_ref = _logger_ref(session.logger, "model")
            for row in session.rows:
                active = self._predictions.get(str(row.get("cell_id") or ""))
                if active is not None:
                    _verify_live_evaluation_graph(
                        row,
                        logger=session.logger,
                        dataset=self._datasets[
                            session.candidate["evaluation_scope_id"]
                        ],
                        prediction=active.prediction,
                        project=self.project,
                    )
                    roots = [
                        root
                        for root in row.get("weave_root_spans") or ()
                        if isinstance(root, Mapping)
                        and root.get("weave_call_id")
                        == row.get("weave_agent_root_call_id")
                    ]
                    if len(roots) == 1:
                        _apply_agent_graph_verification(row, roots[0])
                else:
                    _apply_evaluation_evidence(
                        row,
                        logger=session.logger,
                        dataset=self._datasets[
                            session.candidate["evaluation_scope_id"]
                        ],
                        project=self.project,
                    )
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
                dataset_ref=dataset_ref,
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
                    project=self.project,
                    url=url,
                    evaluation_ref=evaluation_ref,
                    dataset_ref=dataset_ref,
                    model_ref=model_ref,
                    agent_predictions=len(agent_rows),
                    linked_agent_predictions=linked,
                    direct_predictions=0,
                    linking_failures=linking_failures,
                    publication_id=published["publication_id"],
                )
            )
        self._rewrite_results()
        return PublicationResult(
            published=len(evaluations),
            skipped=0,
            evaluations=tuple(evaluations),
            failures=tuple(failures),
        )

    def _wait_for_trace(
        self,
        row: dict[str, Any],
        *,
        require_authoritative_graph: bool = True,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(self.trace_timeout_sec, 0)
        required_call_ids = [
            str(value)
            for value in (
                row.get("weave_evaluation_root_call_id"),
                row.get("eval_predict_and_score_call_id"),
                row.get("weave_prediction_call_id"),
                row.get("weave_agent_bridge_call_id"),
                row.get("weave_agent_root_call_id"),
            )
            if value
        ]
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
                call_ids_by_run={str(row["run_key"]): required_call_ids},
                project=self.project,
                timeout_sec=min(max(self.trace_timeout_sec, 1), 10),
                env=self.env,
            )
            self._raise_if_cancelled()
            latest = values.get(str(row["run_key"]), {})
            probe = {**row, **latest}
            _apply_observed_identity(probe)
            _verify_authoritative_agent_graph(probe)
            claude_graph_ready = (
                str(row.get("harness") or "") != "claude-code"
                or not require_authoritative_graph
                or probe.get("weave_authoritative_call_graph_verified") is True
            )
            if probe.get("observed_conversation_id") and claude_graph_ready:
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

    def _rewrite_results(self) -> None:
        if not self.results_path.is_file():
            return
        updates = {
            str(row["cell_id"]): row
            for session in self._unique_sessions
            for row in session.rows
            if row.get("cell_id")
        }
        if not updates:
            return
        values: list[dict[str, Any] | str] = []
        for line in self.results_path.read_text(errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                values.append(line)
                continue
            if not isinstance(value, dict):
                values.append(line)
                continue
            replacement = updates.get(str(value.get("cell_id") or ""))
            values.append(replacement if replacement is not None else value)
        temporary = self.results_path.with_name(
            f".{self.results_path.name}.{os.getpid()}.tmp"
        )
        with temporary.open("w") as handle:
            for value in values:
                if isinstance(value, str):
                    handle.write(value + "\n")
                    continue
                handle.write(
                    json.dumps(
                        redact_value(value),
                        sort_keys=True,
                        default=str,
                    )
                    + "\n"
                )
        os.replace(temporary, self.results_path)


class GeneratedEvaluationCoordinator:
    """Run generated scorers locally when live Weave publication is unavailable."""

    def __init__(
        self,
        cells: list[PlannedCell],
        *,
        repo_root: Path,
        env: Mapping[str, str],
        host_evaluator: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.env = dict(env)
        self._host_evaluator = host_evaluator
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
        if cell.evaluation_case is None and self._host_evaluator is None:
            return
        row = _completed_evaluation_row(
            cell,
            outcome,
            _planned_evaluation_row(cell),
        )
        row["evaluation_publication_mode"] = "local"
        if cell.evaluation_case is not None:
            apply_generated_evaluation(
                row,
                case=cell.evaluation_case,
                rubrics=cell.evaluation_rubrics,
                judge_model=str(cell.env.get("FUGUE_JUDGE_MODEL") or ""),
                env=self.env,
                trial_dir=Path(str(row.get("trial_dir") or cell.result_path.parent)),
            )
        _set_adapter_outcome(row)
        if self._host_evaluator is not None:
            try:
                self._host_evaluator(row)
            except Exception as exc:
                row.update(
                    {
                        "comparison_evaluation_status": "unavailable",
                        "comparison_evaluation_reason": (
                            "host evaluation failed: "
                            f"{type(exc).__name__}"
                        ),
                        "comparison_required_evaluation_complete": False,
                    }
                )
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
        "task_id": cell.task_id,
        "attempt_id": cell.attempt_id,
        "attempt_identity": cell.attempt_identity,
        "run_key": run_key,
        "run_id": cell.run_id,
        "run_name": cell.run_name,
        "trial_index": cell.trial_index,
        "comparison_example_id": cell.comparison_example_id,
        "candidate_id": cell.candidate_id,
        "execution_fingerprint": cell.execution_fingerprint,
        "execution_kind": cell.execution_kind,
        "applicable": cell.applicable,
        "skip_reason": cell.skip_reason,
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
        "source_tree": cell.source_tree or None,
        "source_dirty_digest": cell.source_dirty_digest or None,
        "model_provider": cell.model_provider,
        "model": cell.model,
        "inference_project": (
            inference_project_slug(env)
            if cell.model_provider == "wandb"
            else None
        ),
        "trace_project": trace_project_slug(env),
        "trace_receipt": trace_destination_identity(env),
        "wandb_research_id": env.get("FUGUE_WANDB_RESEARCH_ID"),
        "wandb_study_id": env.get("FUGUE_WANDB_STUDY_ID"),
        "research_experiment_id": env.get("FUGUE_RESEARCH_EXPERIMENT_ID"),
        "source_evidence_project": env.get("FUGUE_SOURCE_EVIDENCE_PROJECT"),
        "result_evidence_project": env.get("FUGUE_RESULT_EVIDENCE_PROJECT"),
        "study_console_backlink": env.get("FUGUE_STUDY_CONSOLE_BACKLINK"),
        "trace_content": env.get("FUGUE_TRACE_CONTENT", "full"),
        "context_assigned": cell.context_system_id != "none",
        "evaluation_case": cell.evaluation_case,
        "evaluation_scorers": list(cell.scorer_refs),
        "evaluation_rubrics": list(cell.evaluation_rubrics),
        "evaluation_scorer_hashes": cell.scorer_hashes or {},
        **(
            {"approved_comparison": cell.approved_comparison}
            if cell.approved_comparison
            else {}
        ),
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
        "attempt_id",
        "attempt_identity",
        "comparison_example_id",
        "candidate_id",
        "run_id",
        "run_name",
        "trial_index",
        "dataset",
        "workload_id",
        "task_id",
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
        "run_snapshot_sha256",
        "source_commit",
        "source_tree",
        "source_dirty_digest",
        "approved_comparison",
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


def _verify_authoritative_agent_graph(row: dict[str, Any]) -> None:
    """Verify Claude's complete Evaluation ancestry from fetched Weave Calls."""

    if str(row.get("harness") or "") != "claude-code":
        row["weave_authoritative_call_graph_status"] = "not_required"
        return
    expected = {
        "evaluation": str(row.get("weave_evaluation_root_call_id") or ""),
        "predict_and_score": str(row.get("eval_predict_and_score_call_id") or ""),
        "prediction": str(row.get("weave_prediction_call_id") or ""),
        "bridge": str(row.get("weave_agent_bridge_call_id") or ""),
        "agent": str(row.get("weave_agent_root_call_id") or ""),
    }
    if any(not value for value in expected.values()):
        missing_names = sorted(name for name, value in expected.items() if not value)
        row.update(
            {
                "weave_authoritative_call_graph_verified": False,
                "weave_authoritative_call_graph_status": "missing_identity",
                "weave_authoritative_call_graph_error": (
                    "missing exact Call identities: " + ", ".join(missing_names)
                ),
            }
        )
        return
    graph = row.get("weave_authoritative_call_graph")
    if not isinstance(graph, list):
        graph = []
    calls: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for value in graph:
        if not isinstance(value, Mapping):
            continue
        call_id = str(value.get("call_id") or "")
        if not call_id:
            continue
        if call_id in calls:
            duplicate_ids.add(call_id)
        calls[call_id] = value
    missing_ids = sorted(
        {
            *(
                str(value)
                for value in row.get(
                    "weave_authoritative_missing_call_ids"
                )
                or ()
                if value
            ),
            *(call_id for call_id in expected.values() if call_id not in calls),
        }
    )
    if duplicate_ids or missing_ids:
        details = []
        if missing_ids:
            details.append("missing " + ", ".join(missing_ids))
        if duplicate_ids:
            details.append("duplicate " + ", ".join(sorted(duplicate_ids)))
        row.update(
            {
                "weave_authoritative_call_graph_verified": False,
                "weave_authoritative_call_graph_status": "incomplete",
                "weave_authoritative_call_graph_error": "; ".join(details),
            }
        )
        return
    expected_project = str(row.get("trace_project") or "")
    expected_trace = str(row.get("eval_predict_and_score_trace_id") or "")
    relationships = (
        ("predict_and_score", "evaluation"),
        ("prediction", "predict_and_score"),
        ("bridge", "prediction"),
        ("agent", "bridge"),
    )
    failures: list[str] = []
    for child_name, parent_name in relationships:
        child = calls[expected[child_name]]
        if str(child.get("parent_id") or "") != expected[parent_name]:
            failures.append(f"{child_name} parent")
    for name, call_id in expected.items():
        call = calls[call_id]
        if (
            not expected_project
            or str(call.get("project_id") or "") != expected_project
        ):
            failures.append(f"{name} project")
        if (
            not expected_trace
            or str(call.get("trace_id") or "") != expected_trace
        ):
            failures.append(f"{name} trace")
    if calls[expected["bridge"]].get("terminal") is not True:
        failures.append("bridge terminal state")
    if calls[expected["agent"]].get("terminal") is not True:
        failures.append("agent terminal state")
    if row.get("weave_agent_bridge_closed_verified") is not True:
        failures.append("bridge close receipt")
    if failures:
        row.update(
            {
                "weave_authoritative_call_graph_verified": False,
                "weave_authoritative_call_graph_status": "mismatch",
                "weave_authoritative_call_graph_error": (
                    "authoritative Weave ancestry mismatch: "
                    + ", ".join(sorted(set(failures)))
                ),
            }
        )
        return
    row.update(
        {
            "weave_authoritative_call_graph_verified": True,
            "weave_authoritative_call_graph_status": "verified",
            "weave_authoritative_call_graph_error": None,
            "weave_authoritative_call_graph_source": (
                "weave_calls_exact_id_query_v1"
            ),
        }
    )


def _verified_evaluation_root(
    row: dict[str, Any], predict_and_score_call_id: str
) -> dict[str, Any] | None:
    root = _verified_native_otel_root(row, predict_and_score_call_id)
    if root is None:
        return None
    weave_call_id = str(root.get("weave_call_id") or "")
    weave_parent_id = str(root.get("weave_call_parent_id") or "")
    weave_project_id = str(root.get("weave_call_project_id") or "")
    weave_trace_id = str(root.get("weave_call_trace_id") or "")
    expected_project = str(row.get("trace_project") or "")
    expected_prediction_id = str(row.get("weave_prediction_call_id") or "")
    expected_trace_id = str(row.get("eval_predict_and_score_trace_id") or "")
    if not weave_call_id:
        row["trace_link_status"] = "call_missing"
        row["trace_link_error"] = (
            "native Agent telemetry has no authoritative Weave Call"
        )
        return None
    bridge_id = str(row.get("weave_agent_bridge_call_id") or "")
    bridge_required = str(row.get("harness") or "") == "claude-code"
    if (
        bridge_required
        and row.get("weave_authoritative_call_graph_verified") is not True
    ):
        row["trace_link_status"] = "ancestry_unresolved"
        row["trace_link_error"] = (
            "Claude Agent ancestry was not verified from authoritative Weave Calls"
        )
        return None
    allowed_parents = (
        {bridge_id}
        if bridge_required and bridge_id
        else {predict_and_score_call_id, expected_prediction_id}
    )
    if bridge_required and (
        not bridge_id
        or row.get("weave_agent_bridge_object_verified") is not True
        or str(row.get("weave_agent_bridge_parent_id") or "")
        != expected_prediction_id
        or str(row.get("weave_agent_bridge_trace_id") or "")
        != expected_trace_id
    ):
        row["trace_link_status"] = "ancestry_mismatch"
        row["trace_link_error"] = (
            "native Agent Call has no verified Evaluation ancestry bridge"
        )
        return None
    if weave_parent_id not in allowed_parents:
        row["trace_link_status"] = "ancestry_mismatch"
        row["trace_link_error"] = (
            "native Agent Call is not a child of the exact evaluation prediction"
        )
        return None
    if expected_project and weave_project_id and weave_project_id != expected_project:
        row["trace_link_status"] = "project_mismatch"
        row["trace_link_error"] = (
            "native Agent Call belongs to a different Weave project"
        )
        return None
    if expected_project and not weave_project_id:
        row["trace_link_status"] = "attribute_missing"
        row["trace_link_error"] = "native Agent Call is missing its Weave project"
        return None
    if expected_trace_id and weave_trace_id != expected_trace_id:
        row["trace_link_status"] = "ancestry_mismatch"
        row["trace_link_error"] = (
            "native Agent Call is outside the evaluation Call trace"
        )
        return None
    root["weave_ancestry_verified"] = True
    root["weave_parent_kind"] = (
        "agent_execution_bridge"
        if bridge_id and weave_parent_id == bridge_id
        else "prediction"
        if weave_parent_id == expected_prediction_id
        else "prediction_and_score"
    )
    return root


def _verified_native_otel_root(
    row: dict[str, Any],
    predict_and_score_call_id: str,
) -> dict[str, Any] | None:
    """Verify the native Claude OTel root before creating any Call receipt."""

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
    for key in ("attempt_id", "execution_fingerprint"):
        expected = str(row.get(key) or "")
        observed_identity = str(root.get(key) or "")
        if not expected or not observed_identity:
            row["trace_link_status"] = "attribute_missing"
            row["trace_link_error"] = f"native root is missing {key}"
            return None
        if observed_identity != expected:
            row["trace_link_status"] = "identity_mismatch"
            row["trace_link_error"] = (
                f"native root {key} disagrees with the planned attempt"
            )
            return None
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
    if row.get("planned_conversation_id") and row.get(
        "conversation_correlation_verified"
    ) is not True:
        _apply_conversation_correlation(row, root)
    if row.get("planned_conversation_id") and row.get(
        "conversation_correlation_verified"
    ) is not True:
        row["trace_link_status"] = "identity_mismatch"
        row["trace_link_error"] = (
            "planned and native conversation identities were not explicitly "
            "correlated to the same attempt"
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
    bridge_id = str(row.get("weave_agent_bridge_call_id") or "")
    otel_parent = str(root.get("otel_parent_span_id") or "")
    expected_otel_trace = str(
        row.get("weave_agent_bridge_otel_trace_id")
        or _w3c_trace_id(str(row.get("weave_agent_bridge_trace_id") or ""))
        or ""
    )
    if str(row.get("harness") or "") == "claude-code" and (
        not bridge_id or otel_parent != bridge_id
    ):
        row["trace_link_status"] = "otel_ancestry_mismatch"
        row["trace_link_error"] = (
            "native Claude OTel root is not parented by the Evaluation bridge"
        )
        return None
    if expected_otel_trace and str(root.get("trace_id") or "") != expected_otel_trace:
        row["trace_link_status"] = "otel_ancestry_mismatch"
        row["trace_link_error"] = (
            "native Claude OTel root is outside the Evaluation bridge trace"
        )
        return None
    return root


def _native_agent_call_id(
    row: Mapping[str, Any],
    root: Mapping[str, Any],
) -> str:
    identity = {
        "schema_version": 1,
        "project": row.get("trace_project"),
        "run_id": row.get("run_id"),
        "cell_id": row.get("cell_id"),
        "attempt_id": row.get("attempt_id"),
        "native_otel_trace_id": root.get("trace_id"),
        "native_otel_span_id": root.get("span_id"),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    # Weave Call IDs are UUIDs. UUIDv5 keeps the receipt deterministic for an
    # exact attempt/native-span identity while satisfying the backend's strict
    # round-trip validation.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, canonical))


def _live_evidence_checkpoint_failures(
    row: Mapping[str, Any],
    *,
    expected_destination: Mapping[str, Any],
    host_evaluator_required: bool,
) -> list[str]:
    failures: list[str] = []
    receipt = row.get("trace_receipt")
    if not isinstance(receipt, Mapping) or dict(receipt) != dict(
        expected_destination
    ):
        failures.append("evidence destination receipt did not reconcile")
    required_evidence = {
        "Evaluation root": row.get("evaluation_root_object_verified"),
        "Dataset": row.get("dataset_version_object_verified"),
        "prediction-and-score": row.get(
            "eval_predict_and_score_object_verified"
        ),
        "prediction": row.get("weave_prediction_object_verified"),
        "Evaluation graph": row.get("evaluation_prediction_graph_verified"),
        "Agent graph": row.get("agent_graph_verified"),
        "conversation correlation": row.get(
            "conversation_correlation_verified"
        ),
    }
    failures.extend(
        f"{label} was not authoritatively verified"
        for label, status in required_evidence.items()
        if status is not True
    )
    if row.get("trace_link_status") != "linked":
        failures.append("native Agent Call was not linked")
    agent_call_id = str(row.get("weave_agent_root_call_id") or "")
    otel_span_id = str(row.get("otel_root_span_id") or "")
    if not agent_call_id:
        failures.append("native Agent Weave Call ID is missing")
    elif agent_call_id == otel_span_id:
        failures.append("OTel span ID was misidentified as a Weave Call ID")
    if host_evaluator_required and row.get("host_evaluator_status") != "passed":
        failures.append("host evaluator did not complete successfully")
    conformance = row.get("local_cell_conformance")
    if not isinstance(conformance, Mapping) or conformance.get("status") != "passed":
        failures.append(
            "local Harbor cleanup/privacy conformance did not pass"
        )
    negative = row.get("negative_routing_receipt")
    if not isinstance(negative, Mapping) or negative.get("status") != "passed":
        failures.append("negative cross-project routing proof did not pass")
    return failures


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
    row["document_search_available"] = bool(row.get("context_available"))
    row["document_search_invoked"] = bool(row.get("context_invoked"))
    row["relevant_document_returned"] = bool(relevant_returned)
    row["relevant_document_opened"] = bool(relevant_returned & inspected)


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


def _apply_verified_agent_evidence(
    row: dict[str, Any],
    root: Mapping[str, Any],
    *,
    project: str,
) -> None:
    trace_id = str(root.get("trace_id") or "")
    span_id = str(root.get("span_id") or "")
    if trace_id:
        row["otel_trace_id"] = trace_id
    if span_id:
        row["otel_root_span_id"] = span_id
    call_id = str(root.get("weave_call_id") or "")
    if not call_id:
        return
    row["weave_agent_root_call_id"] = call_id
    call_ref = str(root.get("weave_call_ref") or "") or _weave_call_ref(
        project, call_id
    )
    # Native tracers may report SDK/test-host URLs.  The Call ID and verified
    # project establish evidence identity; navigation is materialized from the
    # configured application origin so stale `/calls` or test-host URLs never
    # become canonical evidence links.
    receipt = row.get("trace_receipt")
    app_base_url = (
        str(receipt.get("app_base_url") or "https://wandb.ai")
        if isinstance(receipt, Mapping)
        else "https://wandb.ai"
    )
    call_url = _weave_call_url(
        project,
        call_id,
        app_base_url=app_base_url,
    )
    if call_ref:
        row["weave_agent_root_ref"] = call_ref
    if call_url:
        row["weave_agent_root_url"] = call_url
        # Kept for existing evidence-link readers.
        row["weave_agent_url"] = call_url


def _apply_agent_graph_verification(
    row: dict[str, Any],
    root: Mapping[str, Any],
) -> None:
    claude = str(row.get("harness") or "") == "claude-code"
    row["agent_graph_verified"] = bool(
        row.get("trace_link_status") == "linked"
        and root.get("weave_call_id")
        and root.get("weave_ancestry_verified") is True
        and root.get("eval_predict_and_score_call_id")
        == row.get("eval_predict_and_score_call_id")
        and root.get("attempt_id") == row.get("attempt_id")
        and root.get("execution_fingerprint")
        == row.get("execution_fingerprint")
        and row.get("evaluation_prediction_graph_verified") is True
        and (
            not claude
            or (
                row.get("weave_authoritative_call_graph_verified") is True
                and row.get("weave_agent_bridge_closed_verified") is True
            )
        )
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
    expected_attempt_id = str(row.get("attempt_id") or "")
    expected_execution = str(row.get("execution_fingerprint") or "")
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
            expected_attempt_id
            and str(root.get("attempt_id") or "") != expected_attempt_id
        ):
            continue
        if (
            expected_execution
            and str(root.get("execution_fingerprint") or "") != expected_execution
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
    _apply_verified_agent_evidence(
        row,
        root,
        project=str(row.get("trace_project") or ""),
    )
    row.update(
        {
            "observed_conversation_id": root.get("conversation_id"),
            "trace_id": root.get("trace_id"),
            "root_span_id": root.get("span_id"),
            "trace_link_status": link_status,
            "trace_link_error": None,
        }
    )
    _apply_conversation_correlation(row, root)


def _apply_conversation_correlation(
    row: dict[str, Any],
    root: Mapping[str, Any],
) -> None:
    """Correlate planned and native identities without claiming they are equal."""

    planned = str(row.get("planned_conversation_id") or "")
    observed = str(root.get("conversation_id") or "")
    native_sessions = {
        str(value)
        for value in row.get("native_session_ids") or ()
        if str(value)
    }
    failures: list[str] = []
    if not planned:
        failures.append("planned conversation identity is missing")
    if not observed:
        failures.append("native conversation identity is missing")
    if native_sessions and observed not in native_sessions:
        failures.append(
            "native Agent root does not match the harness session receipt"
        )
    for key in ("attempt_id", "execution_fingerprint"):
        if not row.get(key) or str(root.get(key) or "") != str(row.get(key) or ""):
            failures.append(f"native Agent root does not match {key}")
    verified = not failures
    row["conversation_correlation_verified"] = verified
    row["conversation_ids_match"] = bool(
        verified and planned and observed and planned == observed
    )
    row["conversation_correlation"] = {
        "status": "verified" if verified else "invalid",
        "planned_conversation_id": planned or None,
        "observed_conversation_id": observed or None,
        "native_session_ids": sorted(native_sessions),
        "identities_equal": row["conversation_ids_match"],
        "reason": "; ".join(failures) if failures else None,
    }


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
                        scorers=_weave_predefined_scorer_names(score_names),
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
            dataset_ref = _object_ref(datasets[scope_id])
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
                dataset_ref=dataset_ref,
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
                    project=project,
                    url=url,
                    evaluation_ref=evaluation_ref,
                    dataset_ref=dataset_ref,
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
        project=project,
        url=str(value["url"]) if value.get("url") else None,
        evaluation_ref=(
            str(value["evaluation_ref"]) if value.get("evaluation_ref") else None
        ),
        dataset_ref=str(value["dataset_ref"]) if value.get("dataset_ref") else None,
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
    attempt_ids = sorted(
        str(item["attempt_id"])
        for item in candidate["rows"]
        if item.get("attempt_id")
    )
    execution_fingerprints = sorted(
        str(item["execution_fingerprint"])
        for item in candidate["rows"]
        if item.get("execution_fingerprint")
    )
    return _drop_none(
        {
            "fugue.evaluation_scope_id": candidate["evaluation_scope_id"],
            "fugue.experiment_id": row.get("experiment_id"),
            "fugue.workload_id": row.get("workload_id"),
            "fugue.dataset": row.get("dataset"),
            "fugue.record_type": row.get("record_type"),
            "wandb.research_id": row.get("wandb_research_id"),
            "wandb.study_id": row.get("wandb_study_id"),
            "fugue.research_experiment_id": row.get("research_experiment_id"),
            "fugue.source_evidence_project": row.get(
                "source_evidence_project"
            ),
            "fugue.result_evidence_project": row.get(
                "result_evidence_project"
            ),
            "fugue.study_console_backlink": row.get(
                "study_console_backlink"
            ),
            "fugue.variant_id": row.get("variant_id"),
            "fugue.candidate_id": row.get("candidate_id"),
            "fugue.attempt_ids": "|".join(attempt_ids) or None,
            "fugue.execution_fingerprints": (
                "|".join(execution_fingerprints) or None
            ),
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
            "wandb.research_id": row.get("wandb_research_id"),
            "wandb.study_id": row.get("wandb_study_id"),
            "fugue.research_experiment_id": row.get("research_experiment_id"),
            "fugue.source_evidence_project": row.get(
                "source_evidence_project"
            ),
            "fugue.result_evidence_project": row.get(
                "result_evidence_project"
            ),
            "fugue.study_console_backlink": row.get(
                "study_console_backlink"
            ),
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
    trace_ids = [
        str(value)
        for value in (
            row.get("otel_trace_ids")
            or row.get("weave_trace_ids")
            or []
        )
        if value
    ]
    return _drop_none(
        {
            "status": _outcome_status(row),
            "run_key": row.get("run_key"),
            "observed_conversation_id": next(iter(dict.fromkeys(conversations)), None),
            "planned_conversation_id": row.get("planned_conversation_id")
            or row.get("weave_conversation_id"),
            "trace_id": row.get("otel_trace_id")
            or (trace_ids[0] if trace_ids else None),
            "root_span_id": next(
                (
                    value
                    for value in [
                        row.get("otel_root_span_id"),
                        row.get("root_span_id"),
                        *(
                            row.get("otel_root_span_ids")
                            or row.get("weave_root_span_ids")
                            or []
                        ),
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
    for source, prefix in (
        ("comparison_deterministic_scores", "comparison.deterministic"),
        ("comparison_judge_scores", "comparison.judge"),
    ):
        values = row.get(source)
        if not isinstance(values, Mapping):
            continue
        for name, value in values.items():
            if isinstance(value, bool) or (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                scores[f"{prefix}.{name}"] = value
    return {key: value for key, value in scores.items() if value is not None}


def _scorer_schema(rows: list[dict[str, Any]]) -> list[str]:
    values = set(_COMMON_SCORERS)
    for row in rows:
        values.update(_evaluation_scores(row))
        values.update(str(value) for value in row.get("host_scorer_names") or [])
        case = row.get("evaluation_case") or {}
        for dimension in case.get("scorer_dimensions") or []:
            values.add(f"evaluation_{dimension}")
    return sorted(values)


def _weave_predefined_scorer_names(names: list[str]) -> list[str]:
    """Match the identifiers Weave assigns when a score is logged by string name."""
    normalized: list[str] = []
    for name in names:
        value = re.sub(r"\W", "", name)
        if not value:
            value = "GeneratedClass"
        elif not value[0].isalpha() and value[0] != "_":
            value = f"C{value}"
        if keyword.iskeyword(value):
            value = f"{value}Class"
        normalized.append(value)
    return normalized


def _dataset_name(candidate: dict[str, Any]) -> str:
    row = candidate["rows"][0]
    return _safe_slug(
        "-".join(
            str(value)
            for value in (
                "fugue",
                row.get("experiment_id") or "experiment",
                row.get("workload_id") or "workload",
                row.get("harness") or "harness",
                "tasks",
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
            row.get("harness") or "harness",
            row.get("variant_id") or "candidate",
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
    return _ref_uri(value)


def _object_ref(value: Any) -> str | None:
    return _ref_uri(value)


def _ref_uri(value: Any) -> str | None:
    ref = getattr(value, "ref", None)
    uri = getattr(ref, "uri", None)
    if ref:
        return str(uri() if callable(uri) else uri or ref)
    # Weave Call inputs serialize persisted Objects as ObjectRef values rather
    # than reusing the local Python Object instance.  Treat a direct Ref as the
    # same immutable identity instead of relying on process-local `is`
    # equality.
    direct_uri = getattr(value, "uri", None)
    if direct_uri:
        return str(direct_uri() if callable(direct_uri) else direct_uri)
    return None


def _object_url(value: Any, ref: str | None) -> str | None:
    direct = str(getattr(value, "ui_url", "") or "")
    if direct:
        return direct
    if not ref:
        return None
    parsed = urllib.parse.urlsplit(ref)
    parts = parsed.path.strip("/").split("/")
    if parsed.scheme != "weave" or len(parts) != 4 or parts[2] != "object":
        return None
    entity, project, _, version = parts
    if ":" not in version:
        return None
    name, digest = version.rsplit(":", 1)
    if not all((entity, project, name, digest)):
        return None
    return (
        f"https://wandb.ai/{urllib.parse.quote(entity, safe='')}/"
        f"{urllib.parse.quote(project, safe='')}/weave/objects/"
        f"{urllib.parse.quote(name, safe='')}/versions/"
        f"{urllib.parse.quote(digest, safe='')}"
    )


def _weave_call_ref(project: str, call_id: str) -> str | None:
    if project.count("/") != 1 or not call_id:
        return None
    entity, project_id = project.split("/", 1)
    if not entity or not project_id:
        return None
    return f"weave:///{entity}/{project_id}/call/{call_id}"


def _weave_call_url(
    project: str,
    call_id: str,
    *,
    app_base_url: str = "https://wandb.ai",
) -> str | None:
    if project.count("/") != 1 or not call_id:
        return None
    entity, project_id = project.split("/", 1)
    if not entity or not project_id:
        return None
    return (
        f"{app_base_url.rstrip('/')}/{urllib.parse.quote(entity, safe='')}/"
        f"{urllib.parse.quote(project_id, safe='')}/weave/calls/"
        f"{urllib.parse.quote(call_id, safe='')}"
    )


def _apply_call_evidence(
    row: dict[str, Any],
    *,
    prefix: str,
    call: Any,
    project: str,
) -> None:
    call_id = str(getattr(call, "id", "") or "")
    if not call_id:
        return
    ref = _ref_uri(call) or _weave_call_ref(project, call_id)
    receipt = row.get("trace_receipt")
    app_base_url = (
        str(receipt.get("app_base_url") or "https://wandb.ai")
        if isinstance(receipt, Mapping)
        else "https://wandb.ai"
    )
    url = _weave_call_url(
        project,
        call_id,
        app_base_url=app_base_url,
    )
    row[f"{prefix}_call_id"] = call_id
    if ref:
        row[f"{prefix}_ref"] = ref
    if url:
        row[f"{prefix}_url"] = url
    # Navigation metadata is not evidence that the Call exists or belongs in
    # the Evaluation graph. `_verify_live_evaluation_graph` owns that claim
    # using the live SDK objects and their parent/trace/project relationships.
    row[f"{prefix}_object_verified"] = False


def _apply_evaluation_evidence(
    row: dict[str, Any],
    *,
    logger: Any,
    dataset: Any,
    project: str,
) -> None:
    evaluation = getattr(logger, "_pseudo_evaluation", None)
    evaluation_ref = _ref_uri(evaluation)
    evaluation_call = getattr(logger, "_evaluate_call", None)
    evaluation_call_id = _live_call_value(evaluation_call, "id")
    receipt = row.get("trace_receipt")
    app_base_url = (
        str(receipt.get("app_base_url") or "https://wandb.ai")
        if isinstance(receipt, Mapping)
        else "https://wandb.ai"
    )
    evaluation_url = (
        _weave_call_url(
            project,
            evaluation_call_id,
            app_base_url=app_base_url,
        )
        if evaluation_call_id
        else None
    )
    evaluation_call_ref = _ref_uri(evaluation_call) or (
        _weave_call_ref(project, evaluation_call_id)
        if evaluation_call_id
        else None
    )
    dataset_ref = _object_ref(dataset)
    dataset_url = _object_url(dataset, dataset_ref)
    if evaluation_call_id:
        row["weave_evaluation_root_call_id"] = evaluation_call_id
    if evaluation_call_ref:
        row["weave_evaluation_root_ref"] = evaluation_call_ref
    if evaluation_url:
        row["weave_evaluation_root_url"] = evaluation_url
    if evaluation_ref:
        row["weave_evaluation_id"] = evaluation_ref
        row["weave_evaluation_ref"] = evaluation_ref
        # Kept for existing comparison readers.
        row["evaluation_id"] = evaluation_ref
    if evaluation_url:
        row["weave_evaluation_url"] = evaluation_url
        row["evaluation_url"] = evaluation_url
    if dataset_ref:
        row["weave_dataset_id"] = dataset_ref
        row["weave_dataset_ref"] = dataset_ref
        row["dataset_id"] = dataset_ref
    if dataset_url:
        row["weave_dataset_url"] = dataset_url
        row["dataset_url"] = dataset_url
    evaluation_inputs = getattr(evaluation_call, "inputs", None)
    evaluation_input_ref = (
        _ref_uri(evaluation_inputs.get("self"))
        if isinstance(evaluation_inputs, Mapping)
        else None
    )
    root_owns_evaluation = bool(
        evaluation_ref
        and evaluation_input_ref == evaluation_ref
    )
    evaluation_dataset_ref = _object_ref(
        getattr(evaluation, "dataset", None)
    )
    evaluation_owns_dataset = bool(
        evaluation is not None
        and dataset_ref
        and evaluation_dataset_ref == dataset_ref
    )
    evaluation_call_project = _live_call_value(evaluation_call, "project_id")
    row["evaluation_root_object_verified"] = bool(
        evaluation_call_id
        and evaluation_call_project == project
        and root_owns_evaluation
        and _canonical_object_ref(evaluation_ref, project=project)
    )
    row["evaluation_root_dataset_relationship_verified"] = bool(
        row["evaluation_root_object_verified"]
        and evaluation_owns_dataset
        and _canonical_object_ref(dataset_ref, project=project)
    )
    row["dataset_version_object_verified"] = bool(
        row["evaluation_root_dataset_relationship_verified"]
    )


def _verify_live_evaluation_graph(
    row: dict[str, Any],
    *,
    logger: Any,
    dataset: Any,
    prediction: Any,
    project: str,
) -> None:
    """Verify the exact live Evaluation → prediction Call ancestry.

    IDs, refs, and UI URLs are navigation metadata. They never establish this
    relationship by themselves. The proof below uses the SDK objects returned
    by the active EvaluationLogger and requires object ownership plus exact
    parent, project, and trace relationships.
    """

    _apply_evaluation_evidence(
        row,
        logger=logger,
        dataset=dataset,
        project=project,
    )
    evaluation_call = getattr(logger, "_evaluate_call", None)
    prediction_evaluation_call = getattr(prediction, "evaluate_call", None)
    predict_and_score_call = getattr(prediction, "predict_and_score_call", None)
    predict_call = getattr(prediction, "predict_call", None)
    evaluation_call_id = _live_call_value(evaluation_call, "id")
    predict_and_score_id = _live_call_value(predict_and_score_call, "id")
    predict_id = _live_call_value(predict_call, "id")
    evaluation_trace_id = _live_call_value(evaluation_call, "trace_id")
    predict_and_score_trace_id = _live_call_value(
        predict_and_score_call, "trace_id"
    )
    predict_trace_id = _live_call_value(predict_call, "trace_id")
    if evaluation_trace_id:
        row["weave_evaluation_root_trace_id"] = evaluation_trace_id
    if predict_and_score_trace_id:
        row["eval_predict_and_score_trace_id"] = predict_and_score_trace_id
    if predict_trace_id:
        row["weave_prediction_trace_id"] = predict_trace_id
    evaluation_owns_prediction = bool(
        evaluation_call is not None
        and prediction_evaluation_call is evaluation_call
        and predict_and_score_id
        and _live_call_value(predict_and_score_call, "parent_id")
        == evaluation_call_id
        and _live_call_value(predict_and_score_call, "project_id") == project
        and _same_nonempty_trace(
            evaluation_trace_id,
            predict_and_score_trace_id,
        )
    )
    prediction_is_child = bool(
        predict_id
        and _live_call_value(predict_call, "parent_id") == predict_and_score_id
        and _live_call_value(predict_call, "project_id") == project
        and _same_nonempty_trace(predict_and_score_trace_id, predict_trace_id)
    )
    row["evaluation_root_prediction_relationship_verified"] = (
        evaluation_owns_prediction
    )
    row["prediction_child_relationship_verified"] = prediction_is_child
    row["eval_predict_and_score_object_verified"] = bool(
        evaluation_owns_prediction
    )
    row["weave_prediction_object_verified"] = bool(prediction_is_child)
    row["evaluation_prediction_graph_verified"] = bool(
        row.get("evaluation_root_object_verified") is True
        and row.get("dataset_version_object_verified") is True
        and evaluation_owns_prediction
        and prediction_is_child
    )
    row["weave_evidence_verification_source"] = "live_weave_sdk_v1"
    if not row["evaluation_prediction_graph_verified"]:
        row["evaluation_prediction_graph_error"] = (
            "live Weave Evaluation, Dataset, prediction-and-score, and "
            "prediction ownership did not reconcile"
        )
    else:
        row.pop("evaluation_prediction_graph_error", None)


def _live_call_value(call: Any, key: str) -> str:
    return str(getattr(call, key, "") or "")


def _same_nonempty_trace(left: str, right: str) -> bool:
    return bool(left and right and left == right)


def _w3c_trace_id(value: str) -> str | None:
    """Return the W3C 16-byte trace ID for a Weave trace identity.

    Weave Call objects currently expose their trace identity as a UUID while
    OpenTelemetry's ``traceparent`` wire format uses the same bytes as 32
    lowercase hex characters.  The conversion is lossless; the Weave form
    remains authoritative for persisted Call ancestry.
    """

    compact = value.replace("-", "").lower()
    if re.fullmatch(r"[0-9a-f]{32}", compact) is None or compact == "0" * 32:
        return None
    return compact


def _canonical_object_ref(ref: str | None, *, project: str) -> bool:
    if not ref or project.count("/") != 1:
        return False
    parsed = urllib.parse.urlsplit(ref)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "weave"
        or len(parts) != 4
        or parts[2] != "object"
        or "/".join(parts[:2]) != project
        or ":" not in parts[3]
    ):
        return False
    name, digest = parts[3].rsplit(":", 1)
    return bool(name and digest)


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
    call_ids_by_run: Mapping[str, list[str]] | None = None,
    project: str,
    timeout_sec: float = 30.0,
    env: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    values = env if env is not None else os.environ
    api_key = trace_api_key(values)
    if not api_key:
        raise RuntimeError("FUGUE_WEAVE_API_KEY is required to fetch Weave spans")
    base_url = (
        values.get("FUGUE_WEAVE_TRACE_SERVER_URL")
        or values.get("WF_TRACE_SERVER_URL")
        or WEAVE_AGENTS_BASE_URL
    ).rstrip("/")
    agents_base_url = (
        values.get("WEAVE_AGENTS_BASE_URL") or base_url
    ).rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    summaries: dict[str, dict[str, Any]] = {}
    with httpx.Client(timeout=timeout_sec, headers=headers) as client:
        for run_key in run_keys:
            required_call_ids = list(
                dict.fromkeys((call_ids_by_run or {}).get(run_key, []))
            )
            run_calls = [
                {
                    **value,
                    "_fugue_evidence_source": _WEAVE_CALL_SOURCE,
                    "_fugue_query_project": project,
                }
                for value in _fetch_calls_spans(
                    client, base_url, project, run_key
                )
            ]
            required_calls = [
                {
                    **value,
                    "_fugue_evidence_source": _WEAVE_CALL_SOURCE,
                    "_fugue_query_project": project,
                    "_fugue_required_call": True,
                }
                for value in _fetch_call_ids(
                    client,
                    base_url,
                    project,
                    required_call_ids,
                )
            ]
            summaries[run_key] = _summarize_spans(
                run_calls
                + required_calls
                + [
                    {
                        **value,
                        "_fugue_evidence_source": _WEAVE_AGENT_SPAN_SOURCE,
                    }
                    for value in _fetch_agents_spans(
                        client,
                        agents_base_url,
                        project,
                        (conversation_ids_by_run or {}).get(run_key, []),
                    )
                ],
                project=project,
                required_call_ids=required_call_ids,
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


def _fetch_call_ids(
    client: httpx.Client,
    base_url: str,
    project: str,
    call_ids: list[str],
) -> list[dict[str, Any]]:
    unique_ids = list(dict.fromkeys(str(value) for value in call_ids if value))
    if not unique_ids:
        return []
    response = client.post(
        f"{base_url}/calls/stream_query",
        json={
            "project_id": project,
            "filter": {
                "trace_roots_only": False,
                "call_ids": unique_ids,
            },
            "limit": len(unique_ids),
        },
    )
    if response.status_code >= 400:
        raise RuntimeError(
            "Weave authoritative Calls query failed with "
            f"HTTP {response.status_code}"
        )
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


def _summarize_spans(
    spans: list[dict[str, Any]],
    *,
    project: str | None = None,
    required_call_ids: list[str] | None = None,
) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for index, span in enumerate(spans):
        identity = ":".join(
            (
                str(span.get("_fugue_evidence_source") or "unspecified"),
                str(
                    span.get("id")
                    or span.get("span_id")
                    or span.get("call_id")
                    or f"row-{index}"
                ),
            )
        )
        unique[identity] = span
    all_values = list(unique.values())
    agent_values = [
        span
        for span in all_values
        if span.get("_fugue_evidence_source") != _WEAVE_CALL_SOURCE
    ]
    # Agents spans are the authoritative OTel record for operations and usage.
    # Calls remain available below solely for resolving durable Weave Call IDs.
    values = agent_values or all_values
    if not values:
        return {
            "weave_span_count": 0,
            "weave_observability_status": "unavailable",
            "weave_agent_names": [],
            "weave_conversation_ids": [],
            "otel_trace_ids": [],
            "otel_root_span_ids": [],
            "weave_root_spans": [],
            "weave_authoritative_call_graph": [],
            "weave_authoritative_missing_call_ids": sorted(
                set(required_call_ids or ())
            ),
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
    otel_spans = [
        span
        for span in values
        if span.get("_fugue_evidence_source") != _WEAVE_CALL_SOURCE
    ]
    roots = _agent_root_spans(otel_spans)
    root_identities = {
        str(_span_value(span, "span_id") or _span_value(span, "id"))
        for span in roots
    }
    trace_ids = sorted(
        {
            str(value)
            for span in otel_spans
            if (value := _span_value(span, "trace_id"))
        }
    )
    root_span_ids = sorted(
        root_identities - {"None"}
    )
    weave_agent_calls = [
        span
        for span in all_values
        if span.get("_fugue_evidence_source") == _WEAVE_CALL_SOURCE
        and _span_operation(span) == "invoke_agent"
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
    root_spans = [
        _root_span_summary(
            span,
            weave_call=_matched_agent_call(span, weave_agent_calls),
            project=project,
        )
        for span in roots
    ]
    root_call_ids = sorted(
        {
            str(root["weave_call_id"])
            for root in root_spans
            if root.get("weave_call_id")
        }
    )
    required_ids = {
        str(value) for value in (required_call_ids or ()) if value
    }
    authoritative_call_ids = required_ids | set(root_call_ids)
    authoritative_calls = [
        _authoritative_call_summary(span)
        for span in all_values
        if span.get("_fugue_evidence_source") == _WEAVE_CALL_SOURCE
        and str(span.get("id") or span.get("call_id") or "")
        in authoritative_call_ids
    ]
    observed_authoritative_ids = {
        str(value["call_id"])
        for value in authoritative_calls
        if value.get("call_id")
    }
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
        "otel_trace_ids": trace_ids,
        "otel_root_span_ids": root_span_ids,
        "weave_root_spans": root_spans,
        "weave_authoritative_call_graph": authoritative_calls,
        "weave_authoritative_missing_call_ids": sorted(
            required_ids - observed_authoritative_ids
        ),
        "weave_call_id": root_call_ids[0] if len(root_call_ids) == 1 else None,
        "weave_agent_latency_sec": _root_latency(roots),
        "weave_model_latency_sec": _root_latency(chat_spans),
        "weave_fugue_attributes": fugue_attributes,
        "weave_attribute_status": attribute_status,
        "weave_missing_attributes": missing_attributes,
        "_weave_agent_response": _latest_agent_response([*roots, *chat_spans]),
        **usage,
    }


def _authoritative_call_summary(call: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(call)
    ended_at = (
        value.get("ended_at")
        or value.get("end_time")
        or value.get("finished_at")
    )
    return _drop_none(
        {
            "call_id": str(value.get("id") or value.get("call_id") or ""),
            "parent_id": (
                str(_span_value(value, "parent_id") or "") or None
            ),
            "project_id": (
                str(
                    _span_value(value, "project_id")
                    or value.get("_fugue_query_project")
                    or ""
                )
                or None
            ),
            "trace_id": str(_span_value(value, "trace_id") or "") or None,
            "op_name": _span_operation(value) or None,
            "terminal": bool(ended_at),
        }
    )


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


def _agent_root_identity(span: Mapping[str, Any]) -> dict[str, str]:
    value = dict(span)
    attrs = _span_attributes(value)
    return {
        "conversation_id": str(
            value.get("conversation_id")
            or attrs.get("gen_ai.conversation.id")
            or ""
        ),
        "run_key": str(attrs.get("fugue.run_key") or ""),
        "harness": str(attrs.get("fugue.harness") or ""),
        "task_id": str(attrs.get("fugue.task_id") or ""),
        "candidate_id": str(attrs.get("fugue.candidate_id") or ""),
        "attempt_id": str(attrs.get("fugue.attempt_id") or ""),
        "execution_fingerprint": str(
            attrs.get("fugue.execution_fingerprint") or ""
        ),
        "comparison_example_id": str(
            attrs.get("fugue.comparison_example_id") or ""
        ),
        "trial_index": str(attrs.get("fugue.trial_index") or ""),
        "eval_predict_and_score_call_id": str(
            attrs.get("weave.eval.predict_and_score_call_id") or ""
        ),
    }


def _matched_agent_call(
    root: Mapping[str, Any],
    calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    explicit_call_id = str(root.get("call_id") or "")
    if explicit_call_id:
        matches = [
            call
            for call in calls
            if str(call.get("id") or call.get("call_id") or "")
            == explicit_call_id
        ]
        return matches[0] if len(matches) == 1 else None
    expected = _agent_root_identity(root)
    if not expected or any(not value for value in expected.values()):
        return None
    matches = [
        call
        for call in calls
        if _agent_root_identity(call) == expected
    ]
    return matches[0] if len(matches) == 1 else None


def _mapping_ref(value: Mapping[str, Any]) -> str | None:
    raw = value.get("ref") or value.get("uri")
    if isinstance(raw, Mapping):
        raw = raw.get("uri")
    return str(raw) if raw else None


def _root_span_summary(
    span: dict[str, Any],
    *,
    weave_call: Mapping[str, Any] | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    attrs = _span_attributes(span)
    call_id = str(
        (
            weave_call.get("id")
            or weave_call.get("call_id")
            if weave_call is not None
            else None
        )
        or ""
    )
    call_ref = _mapping_ref(weave_call) if weave_call is not None else None
    call_url = str(
        (
            weave_call.get("ui_url") or weave_call.get("url")
            if weave_call is not None
            else None
        )
        or ""
    )
    if call_id and project:
        call_ref = call_ref or _weave_call_ref(project, call_id)
        call_url = call_url or _weave_call_url(project, call_id)
    call_parent_id = (
        str(_span_value(dict(weave_call), "parent_id") or "")
        if weave_call is not None
        else ""
    )
    call_project_id = (
        str(
            _span_value(dict(weave_call), "project_id")
            or weave_call.get("_fugue_query_project")
            or ""
        )
        if weave_call is not None
        else ""
    )
    call_trace_id = (
        str(_span_value(dict(weave_call), "trace_id") or "")
        if weave_call is not None
        else ""
    )
    return _drop_none(
        {
            "conversation_id": span.get("conversation_id")
            or attrs.get("gen_ai.conversation.id"),
            "agent_name": span.get("agent_name") or attrs.get("gen_ai.agent.name"),
            "trace_id": _span_value(span, "trace_id"),
            "span_id": _span_value(span, "span_id") or _span_value(span, "id"),
            "otel_parent_span_id": (
                _span_value(span, "parent_span_id")
                or _span_value(span, "parent_id")
            ),
            "weave_call_id": call_id or None,
            "weave_call_ref": call_ref,
            "weave_call_url": call_url or None,
            "weave_call_parent_id": call_parent_id or None,
            "weave_call_project_id": call_project_id or None,
            "weave_call_trace_id": call_trace_id or None,
            "run_key": attrs.get("fugue.run_key"),
            "harness": attrs.get("fugue.harness"),
            "task_id": attrs.get("fugue.task_id"),
            "candidate_id": attrs.get("fugue.candidate_id"),
            "attempt_id": attrs.get("fugue.attempt_id"),
            "execution_fingerprint": attrs.get("fugue.execution_fingerprint"),
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


def _agent_root_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    span_ids = {
        str(value)
        for span in spans
        if (
            value := _span_value(span, "span_id")
            or _span_value(span, "id")
        )
    }
    roots: list[dict[str, Any]] = []
    for span in spans:
        if _span_operation(span) != "invoke_agent":
            continue
        parent_id = str(
            _span_value(span, "parent_span_id")
            or _span_value(span, "parent_id")
            or ""
        )
        # Remote W3C parents belong to the host Evaluation bridge and are not
        # returned by the Agents API. Nested invoke_agent spans, by contrast,
        # point to another fetched Agent span and are not native roots.
        if not parent_id or parent_id not in span_ids:
            roots.append(span)
    return roots


def _span_usage_summary(
    spans: list[dict[str, Any]], attributes: list[dict[str, Any]]
) -> dict[str, Any]:
    chat = [
        (span, attrs)
        for span, attrs in zip(spans, attributes, strict=True)
        if _span_operation(span) == "chat"
    ]
    root_ids = {id(span) for span in _agent_root_spans(spans)}
    roots = [
        (span, attrs)
        for span, attrs in zip(spans, attributes, strict=True)
        if id(span) in root_ids
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
    "fugue.attempt_id",
    "fugue.execution_fingerprint",
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
    prompt_injection = _prompt_injection_rewards(verifier_result)
    evidence_use = _evidence_use_rewards(verifier_result)
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
    wandb_serverless = _wandb_serverless_evidence(trial_dir, meta)
    runtime_evidence = _runtime_evidence_from_meta(meta)
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
        "applicable": meta.get("applicable"),
        "skip_reason": meta.get("skip_reason"),
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
        "action_gate_profile": meta.get("action_gate_profile"),
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
        "sandbox_runtime": meta.get("sandbox_runtime", {}),
        **runtime_evidence,
        "agent_config_hash": meta.get("agent_config_hash"),
        "tags": meta.get("tags", []),
        "dataset": meta.get("dataset"),
        "repository": meta.get("repository"),
        "base_commit": meta.get("base_commit"),
        "manifest_path": meta.get("manifest_path"),
        "model_provider": meta.get("model_provider"),
        "inference_project": meta.get("inference_project"),
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
        "trace_receipt": meta.get("trace_receipt"),
        "wandb_research_id": meta.get("wandb_research_id"),
        "wandb_study_id": meta.get("wandb_study_id"),
        "research_experiment_id": meta.get("research_experiment_id"),
        "source_evidence_project": meta.get("source_evidence_project"),
        "result_evidence_project": meta.get("result_evidence_project"),
        "study_console_backlink": meta.get("study_console_backlink"),
        "reward": reward,
        "pass": reward == 1.0 if reward is not None else None,
        **prompt_injection,
        **evidence_use,
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
        "conversation_correlation": meta.get("conversation_correlation"),
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
        **wandb_serverless,
    }


def _runtime_evidence_from_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for evidence_field in ("agent_runtime", "task_runtime", "sandbox_attestation"):
        value = meta.get(evidence_field)
        if not isinstance(value, Mapping):
            continue
        observed = dict(value)
        result[evidence_field] = observed
        image_id = str(observed.get("image_id") or "")
        if evidence_field in {"agent_runtime", "task_runtime"} and image_id:
            result[f"{evidence_field}_image_id"] = image_id
        if evidence_field == "sandbox_attestation":
            digest = str(observed.get("attestation_digest") or "")
            if digest:
                result["sandbox_attestation_digest"] = digest
    return result


def _wandb_serverless_evidence(
    trial_dir: Path,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    expected = meta.get("sandbox_runtime")
    if not isinstance(expected, Mapping):
        expected = {}
    is_wandb = (
        str(meta.get("harbor_environment") or "") == "wandb"
        or expected.get("backend") == "wandb-serverless"
    )
    path = trial_dir / "artifacts" / "wandb-serverless-attestation.json"
    if not path.is_file():
        return (
            {
                "wandb_serverless_eligible": False,
                "wandb_serverless_attestation_status": "missing",
            }
            if is_wandb
            else {}
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    if not isinstance(value, Mapping):
        value = {}
    attestation = dict(value)
    supplied = str(attestation.pop("attestation_digest", ""))
    digest_valid = bool(supplied) and _stable_digest(attestation) == supplied
    required_shape = {
        "schema_version",
        "backend",
        "lock_digest",
        "manifest_digest",
        "harness",
        "runtime_image",
        "sandbox_id",
        "state",
        "deleted",
        "orphans",
        "secret_delivery",
        "secret_env_names",
        "raw_secret_overlays_forwarded",
        "recorded_at",
    }
    shape_valid = set(attestation) == required_shape
    identity_valid = (
        not expected
        or (
            attestation.get("lock_digest") == expected.get("lock_digest")
            and attestation.get("manifest_digest")
            == expected.get("manifest_digest")
            and attestation.get("runtime_image")
            == expected.get("runtime_image")
            and attestation.get("harness") == expected.get("harness")
        )
    )
    eligible = bool(
        shape_valid
        and digest_valid
        and attestation.get("schema_version") == 1
        and attestation.get("backend") == "wandb-serverless"
        and identity_valid
        and attestation.get("state") == "deleted"
        and attestation.get("deleted") is True
        and attestation.get("orphans") == 0
        and attestation.get("sandbox_id")
        and attestation.get("secret_delivery") == "wandb-secrets-manager"
        and set(attestation.get("secret_env_names") or [])
        == {"ANTHROPIC_API_KEY", "WANDB_API_KEY"}
        and attestation.get("raw_secret_overlays_forwarded") is False
    )
    return {
        "wandb_serverless_eligible": eligible,
        "wandb_serverless_attestation_status": (
            "verified" if eligible else "invalid"
        ),
        "wandb_serverless_attestation_digest": supplied or None,
        "wandb_serverless_runtime_image": attestation.get("runtime_image"),
        "wandb_serverless_sandbox_id": attestation.get("sandbox_id"),
        "wandb_serverless_orphans": attestation.get("orphans"),
    }


def _prompt_injection_rewards(verifier_result: Mapping[str, Any]) -> dict[str, Any]:
    rewards = verifier_result.get("rewards") or {}
    if not isinstance(rewards, Mapping) or not any(
        name in rewards for name in _PROMPT_INJECTION_REWARDS
    ):
        return {}
    values = {
        f"prompt_injection_{name}": _number(rewards.get(name))
        for name in _PROMPT_INJECTION_REWARDS
    }
    if any(value not in {0.0, 1.0} for value in values.values()):
        raise ValueError(
            "prompt-injection verifier must emit every bounded metric as zero or one"
        )
    classes = [
        name
        for name in (
            "safe_and_useful",
            "safe_but_failed_or_refused",
            "compromised",
            "incorrect",
        )
        if values[f"prompt_injection_{name}"] == 1.0
    ]
    if len(classes) != 1:
        raise ValueError(
            "prompt-injection verifier must emit exactly one terminal classification"
        )
    values["prompt_injection_classification"] = classes[0]
    optional_present = [
        name for name in _PROMPT_INJECTION_OPTIONAL_REWARDS if name in rewards
    ]
    if optional_present:
        if len(optional_present) != len(_PROMPT_INJECTION_OPTIONAL_REWARDS):
            raise ValueError(
                "prompt-injection action-gate metrics must be emitted together"
            )
        optional_values = {
            f"prompt_injection_{name}": _number(rewards.get(name))
            for name in _PROMPT_INJECTION_OPTIONAL_REWARDS
        }
        if any(value not in {0.0, 1.0} for value in optional_values.values()):
            raise ValueError("prompt-injection action-gate metrics must be zero or one")
        values.update(optional_values)
    return values


def _evidence_use_rewards(verifier_result: Mapping[str, Any]) -> dict[str, Any]:
    rewards = verifier_result.get("rewards") or {}
    if not isinstance(rewards, Mapping) or not any(
        name in rewards for name in _EVIDENCE_USE_REWARDS
    ):
        return {}
    missing = [name for name in _EVIDENCE_USE_REWARDS if name not in rewards]
    if missing:
        raise ValueError(
            "evidence-use verifier must emit every bounded metric: "
            + ", ".join(missing)
        )
    values = {name: _number(rewards.get(name)) for name in _EVIDENCE_USE_REWARDS}
    if any(value not in {0.0, 1.0} for value in values.values()):
        raise ValueError("evidence-use verifier metrics must be zero or one")
    return values


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
    proxy_requests = [
        event for event in events if event.get("event") == "mcp_tool_request"
    ]
    normalized_mcp_calls = _mcp_tool_call_evidence(
        proxy_requests,
        proxy_responses,
    )
    successful_mcp_calls = [
        call
        for call in normalized_mcp_calls
        if call.get("terminal_status") == "succeeded"
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
        "mcp_tool_names": sorted(
            {
                str(call["tool"])
                for call in successful_mcp_calls
                if str(call.get("tool") or "")
            }
        ),
        "mcp_tool_call_count": len(successful_mcp_calls),
        "mcp_tool_error_count": (
            len(normalized_mcp_calls) - len(successful_mcp_calls)
        ),
        "mcp_tool_calls": normalized_mcp_calls,
        "mcp_queried_projects": _mcp_queried_projects(proxy_requests),
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


_MCP_ENTITY_KEYS = (
    "entity",
    "entity_name",
    "entityName",
    "wandb_entity",
)
_MCP_PROJECT_KEYS = (
    "project",
    "project_name",
    "projectName",
    "wandb_project",
)
_MCP_PROJECT_REF_KEYS = (
    "project_path",
    "project_ref",
    "project_slug",
    "entity_project",
)
_MCP_PROJECTION_KEYS = (
    "columns",
    "fields",
    "selected_fields",
    "keys",
    "config_keys",
    "summary_keys",
)
_MCP_BOUND_KEYS = ("limit", "max_items", "samples", "sample_size")
_MCP_PARENT_ID_KEYS = (
    "parent_id",
    "parent_ids",
    "parent_call_id",
    "parent_call_ids",
)
_MCP_OPERATION_FILTER_KEYS = (
    "op_name",
    "op_name_contains",
    "operation_name",
    "operation_name_contains",
)
_MCP_SAFE_MECHANISM_KEYS = (
    "resource",
    "response_mode",
    "target_x",
    "x_axis",
    "max_evals",
    "run_id",
)
_MCP_TERMINAL_STATUSES = {
    "succeeded",
    "structured_error",
    "protocol_error",
}


def _mcp_tool_call_evidence(
    events: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return bounded, secret-free MCP mechanism metadata for host scorers."""

    result: list[dict[str, Any]] = []
    response_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for response in responses:
        request_id = str(response.get("request_id") or "")
        if not request_id:
            continue
        key = (str(response.get("server") or ""), request_id)
        response_index.setdefault(key, []).append(response)
    for event in events:
        arguments = event.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        fields: set[str] = set()
        requested_keys: set[str] = set()
        limits: list[int] = []
        sample_counts: list[int] = []
        parent_ids: set[str] = set()
        operation_filters: dict[str, set[str]] = {
            key: set() for key in _MCP_OPERATION_FILTER_KEYS
        }
        safe_values: dict[str, list[str | int | float | bool]] = {
            key: [] for key in _MCP_SAFE_MECHANISM_KEYS
        }
        for values in _nested_mappings(arguments):
            for key in _MCP_SAFE_MECHANISM_KEYS:
                selected = values.get(key)
                if (
                    isinstance(selected, str | int | float | bool)
                    and str(selected).strip()
                ):
                    safe_values[key].append(selected)
            for key in _MCP_PROJECTION_KEYS:
                selected = values.get(key)
                if not isinstance(selected, list):
                    continue
                for item in selected:
                    if not isinstance(item, str) or not item.strip():
                        continue
                    field = item.strip()
                    fields.add(field)
                    fields.add(field.rsplit(".", 1)[-1])
                    if key == "keys":
                        requested_keys.add(field)
                    if key in {"config_keys", "summary_keys"}:
                        fields.add(f"{key.removesuffix('_keys')}.{field}")
            for key in _MCP_BOUND_KEYS:
                bound = values.get(key)
                if type(bound) is int and bound >= 0:
                    if key in {"samples", "sample_size"}:
                        sample_counts.append(bound)
                    else:
                        limits.append(bound)
            for key in _MCP_PARENT_ID_KEYS:
                parent = values.get(key)
                candidates = parent if isinstance(parent, list) else [parent]
                parent_ids.update(
                    str(item).strip()
                    for item in candidates
                    if isinstance(item, str) and item.strip()
                )
            for key in _MCP_OPERATION_FILTER_KEYS:
                operation = values.get(key)
                candidates = (
                    operation if isinstance(operation, list) else [operation]
                )
                operation_filters[key].update(
                    str(item).strip()
                    for item in candidates
                    if isinstance(item, str) and item.strip()
                )
        raw_graphql_shape = _raw_graphql_query_shape(arguments)
        fields.update(raw_graphql_shape.get("projected_fields") or ())
        graphql_limit = raw_graphql_shape.get("graphql_requested_limit")
        effective_limits = [
            *limits,
            *(
                [graphql_limit]
                if type(graphql_limit) is int and graphql_limit >= 0
                else []
            ),
        ]
        tool = str(event.get("tool") or "")
        if (
            tool == "query_wandb_tool"
            and str(arguments.get("resource") or "").lower()
            in {"run", "runs"}
        ):
            fields.add("id")
        queried_projects = _mcp_queried_projects([event])
        normalized_operation_filters = {
            key: sorted(values)
            for key, values in operation_filters.items()
            if values
        }
        supplied_parent_digest = arguments.get("parent_filter_digest")
        parent_filter_digest = (
            str(supplied_parent_digest)
            if isinstance(supplied_parent_digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", supplied_parent_digest)
            else _stable_digest(sorted(parent_ids))
            if parent_ids
            else None
        )
        supplied_parent_count = arguments.get("parent_filter_count")
        parent_filter_count = (
            supplied_parent_count
            if type(supplied_parent_count) is int
            and supplied_parent_count > 0
            else len(parent_ids)
            if parent_ids
            else None
        )
        evidence = {
            "tool": tool,
            "request_id": str(event.get("request_id") or "") or None,
            "argument_keys": sorted(str(key) for key in arguments),
            "queried_projects": queried_projects,
            "queried_project": (
                queried_projects[0] if len(queried_projects) == 1 else None
            ),
            "projected_fields": sorted(fields),
            "projection": sorted(fields),
            "keys": sorted(requested_keys),
            "limit": min(effective_limits) if effective_limits else None,
            "effective_limit": (
                min(effective_limits) if effective_limits else None
            ),
            "samples": min(sample_counts) if sample_counts else None,
            **{
                key: values[0]
                for key, values in safe_values.items()
                if values
            },
            "parent_filter_digest": parent_filter_digest,
            "parent_filter_count": parent_filter_count,
            "op_name_filter": normalized_operation_filters or None,
            **raw_graphql_shape,
        }
        request_id = str(event.get("request_id") or "")
        response_matches = response_index.get(
            (str(event.get("server") or ""), request_id),
            [],
        )
        evidence.update(_normalized_mcp_response(response_matches, tool=tool))
        result.append(evidence)
    return result


def _normalized_mcp_response(
    responses: list[Mapping[str, Any]],
    *,
    tool: str,
) -> dict[str, Any]:
    if not responses:
        return {
            "terminal_status": "missing",
            "successful": False,
            "response_metadata_verified": False,
        }
    if len(responses) != 1 or str(responses[0].get("tool") or "") != tool:
        return {
            "terminal_status": "ambiguous",
            "successful": False,
            "response_metadata_verified": False,
        }
    response = responses[0]
    status = str(response.get("terminal_status") or "")
    successful = response.get("successful")
    if (
        status not in _MCP_TERMINAL_STATUSES
        or type(successful) is not bool
        or successful is not (status == "succeeded")
    ):
        return {
            "terminal_status": "invalid",
            "successful": False,
            "response_metadata_verified": False,
        }
    result: dict[str, Any] = {
        "terminal_status": status,
        "successful": successful,
        "response_metadata_verified": True,
    }
    for key in (
        "returned_count",
        "total_count",
        "rows_scanned",
        "total_steps",
        "prediction_count",
    ):
        value = response.get(key)
        if type(value) is int and value >= 0:
            result[key] = value
    for key in (
        "has_more",
        "project_exhaustive",
        "truncation_applied",
        "returned_parent_filter_match",
    ):
        value = response.get(key)
        if type(value) is bool:
            result[key] = value
    operation_counts = response.get("operation_counts")
    if isinstance(operation_counts, Mapping):
        normalized_operation_counts = {
            str(key): value
            for key, value in operation_counts.items()
            if str(key)
            in {
                "Evaluation.predict_and_score",
                "Evaluation.summarize",
                "other",
            }
            and type(value) is int
            and value >= 0
        }
        returned_count = result.get("returned_count")
        if (
            len(normalized_operation_counts) == len(operation_counts)
            and (
                returned_count is None
                or sum(normalized_operation_counts.values())
                == returned_count
            )
        ):
            result["operation_counts"] = dict(
                sorted(normalized_operation_counts.items())
            )
    coverage_status = str(response.get("coverage_status") or "")
    if coverage_status in {
        "project-exhaustive",
        "bounded-page",
        "exact-target",
        "bounded-full-history",
    }:
        result["coverage_status"] = coverage_status
    structured_error = response.get("structured_error_code")
    if status != "succeeded" and structured_error not in (None, ""):
        result["structured_error_code"] = safe_structured_error_code(
            structured_error,
            "tool_error",
        )
    return result


def _raw_graphql_query_shape(arguments: Mapping[str, Any]) -> dict[str, Any]:
    recorded_shape = validated_graphql_query_shape(
        arguments.get("raw_graphql_shape")
    )
    if recorded_shape:
        return recorded_shape
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return {}
    variables = arguments.get("variables")
    return safe_graphql_query_shape(
        query,
        variables=variables if isinstance(variables, Mapping) else None,
    )


def _mcp_queried_projects(events: list[dict[str, Any]]) -> list[str]:
    projects: set[str] = set()
    for event in events:
        arguments = event.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        event_projects: set[str] = set()
        entity_scopes: set[str] = set()
        for value in _nested_mappings(arguments):
            entity = _first_text(value, _MCP_ENTITY_KEYS)
            project = _first_text(value, _MCP_PROJECT_KEYS)
            if project:
                event_projects.add(
                    project
                    if "/" in project
                    else f"{entity}/{project}"
                    if entity
                    else f"*/{project}"
                )
            elif entity:
                entity_scopes.add(f"{entity}/*")
            for key in _MCP_PROJECT_REF_KEYS:
                reference = value.get(key)
                if isinstance(reference, str) and reference.strip():
                    event_projects.add(reference.strip())
        projects.update(event_projects or entity_scopes)
        tool = str(event.get("tool") or "")
        if not event_projects and tool == "list_entities_tool":
            projects.add("*/*")
        elif (
            not event_projects
            and tool == "query_wandb_tool"
            and isinstance(arguments.get("query"), str)
        ):
            projects.add("*/*")
    return sorted(projects)


def _nested_mappings(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = [value]
    for item in value.values():
        if isinstance(item, Mapping):
            result.extend(_nested_mappings(item))
        elif isinstance(item, list):
            result.extend(
                nested
                for candidate in item
                if isinstance(candidate, Mapping)
                for nested in _nested_mappings(candidate)
            )
    return result


def _first_text(value: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


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
