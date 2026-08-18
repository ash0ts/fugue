from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock

from fugue.bench.candidates import attempt_id as canonical_attempt_id
from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json

LOCAL_EVIDENCE_SCHEMA_VERSION = 1
LOCAL_EVIDENCE_LAYOUT_VERSION = 1
LOCAL_EVIDENCE_FORMAT = "fugue-evidence"

EvidenceNodeKind = Literal[
    "evaluation_root",
    "prediction_and_score",
    "prediction",
    "agent_root",
    "dataset",
]
EvidenceStatus = Literal["resolved", "missing", "invalid"]
EdgeStatus = Literal["verified", "invalid"]
AttemptTerminalStatus = Literal[
    "passed",
    "failed",
    "cancelled",
    "interrupted",
    "not_applicable",
]
AttemptIntegrityStatus = Literal["resolved", "incomplete", "invalid"]
ManifestStatus = Literal["complete", "incomplete", "invalid"]
RunConformanceStatus = Literal[
    "passed", "failed", "unavailable", "not_applicable"
]

_NODE_KINDS = frozenset(
    {
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_root",
        "dataset",
    }
)
_EDGE_RELATIONSHIPS = (
    "evaluation_has_dataset",
    "evaluation_has_prediction_and_score",
    "prediction_and_score_has_prediction",
    "prediction_has_agent_root",
)
_EVIDENCE_STATUSES = frozenset({"resolved", "missing", "invalid"})
_EDGE_STATUSES = frozenset({"verified", "invalid"})
_TERMINAL_STATUSES = frozenset(
    {"passed", "failed", "cancelled", "interrupted", "not_applicable"}
)
_INTEGRITY_STATUSES = frozenset({"resolved", "incomplete", "invalid"})
_MANIFEST_STATUSES = frozenset({"complete", "incomplete", "invalid"})
_EVENT_TYPES = frozenset(
    {
        "run_initialized",
        "attempt_opened",
        "attempt_finalized",
        "manifest_finalized",
        "publication_receipt_written",
    }
)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_PRIVATE_FIELDS = frozenset(
    {
        "answer_key",
        "expected",
        "gold",
        "gold_output",
        "private",
        "private_expected_values",
        "private_labels",
        "reference_answer",
    }
)
_SECRET_FIELD = re.compile(
    r"(?:^|_)(?:api_?key|token|secret|password|credential|private_?key)(?:$|_)",
    re.IGNORECASE,
)


def local_result_row_projection_v1(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the decision-bearing V3 attempt projection for one prediction.

    The projection is deliberately smaller than the full prediction row, but
    includes every attempt field that can change a behavioral or efficiency
    conclusion.  Local attempt evidence commits to its digest before the
    convenience JSONL export exists.  Comparison construction and reload use
    the same projection, so copied digest strings cannot authorize altered
    pass/fail, score, explanation, excerpt, or cost data.
    """

    deterministic = _projection_mapping(row.get("comparison_deterministic_scores"))
    judge = {
        f"comparison.judge.{dimension}": value
        for dimension, value in _projection_mapping(
            row.get("comparison_judge_scores")
        ).items()
        if _projection_number(value) is not None
    }
    scores = {**deterministic, **judge}
    queried_projects = tuple(sorted(_projection_queried_projects(row)))
    tools, tool_calls = _projection_tool_activity(row)
    usage = _projection_mapping(row.get("usage"))
    return {
        "schema_version": 1,
        "attempt_id": str(row.get("attempt_id") or ""),
        "prediction_id": str(row.get("prediction_id") or "") or None,
        "passed": _projection_bool(row.get("pass")),
        "execution_status": _projection_execution_status(row),
        "evaluation_status": str(
            row.get("comparison_evaluation_status") or "unknown"
        ),
        "cost_usd": _projection_first_number(
            row, "cost_usd", "observed_cost_usd", "total_cost_usd"
        ),
        "latency_sec": _projection_latency_sec(row),
        "input_tokens": _projection_number(
            usage.get("input_tokens") or row.get("input_tokens")
        ),
        "output_tokens": _projection_number(
            usage.get("output_tokens") or row.get("output_tokens")
        ),
        "tool_calls": tool_calls,
        "tools": list(tools),
        "queried_projects": list(queried_projects),
        "scores": scores,
        "score_explanations": {
            dimension: _projection_score_explanation(
                dimension,
                _projection_bool(value),
                row=row,
            )
            for dimension, value in scores.items()
        },
        "sanitized_answer_excerpt": _projection_answer_excerpt(row),
        "actual_query_scope": list(queried_projects),
        "reported_project_identity": _projection_reported_project(row),
        "execution_fingerprint": str(row.get("execution_fingerprint") or "")
        or None,
        "runtime_lock_digest": str(
            row.get("runtime_lock_digest") or row.get("runtime_digest") or ""
        )
        or None,
    }


def local_result_attempt_projection_v1(
    *,
    attempt_id: str,
    prediction_id: str | None,
    passed: bool | None,
    execution_status: str,
    evaluation_status: str,
    cost_usd: float | None,
    latency_sec: float | None,
    input_tokens: float | None,
    output_tokens: float | None,
    tool_calls: int,
    tools: Sequence[str],
    queried_projects: Sequence[str],
    scores: Mapping[str, Any],
    score_explanations: Mapping[str, str],
    sanitized_answer_excerpt: str | None,
    actual_query_scope: Sequence[str],
    reported_project_identity: str | None,
    execution_fingerprint: str | None,
    runtime_lock_digest: str | None,
) -> dict[str, Any]:
    """Project an already-normalized V3 attempt using the canonical shape."""

    return {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "prediction_id": prediction_id,
        "passed": passed,
        "execution_status": execution_status,
        "evaluation_status": evaluation_status,
        "cost_usd": cost_usd,
        "latency_sec": latency_sec,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_calls": tool_calls,
        "tools": list(tools),
        "queried_projects": list(queried_projects),
        "scores": dict(scores),
        "score_explanations": dict(score_explanations),
        "sanitized_answer_excerpt": sanitized_answer_excerpt,
        "actual_query_scope": list(actual_query_scope),
        "reported_project_identity": reported_project_identity,
        "execution_fingerprint": execution_fingerprint,
        "runtime_lock_digest": runtime_lock_digest,
    }


def local_result_row_projection_digest(row: Mapping[str, Any]) -> str:
    return stable_digest(local_result_row_projection_v1(row))


def _projection_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _projection_number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _projection_first_number(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _projection_number(row.get(key))
        if value is not None:
            return value
    return None


def _projection_latency_sec(row: Mapping[str, Any]) -> float | None:
    direct = _projection_first_number(row, "latency_sec", "wall_time_sec")
    if direct is not None:
        return direct
    milliseconds = _projection_first_number(row, "latency_ms")
    return round(milliseconds / 1000, 6) if milliseconds is not None else None


def _projection_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, Mapping):
        passed = value.get("passed")
        return passed if isinstance(passed, bool) else None
    return None


def _projection_execution_status(row: Mapping[str, Any]) -> str:
    status = str(row.get("status") or row.get("execution_status") or "").lower()
    if status in {"passed", "success", "succeeded", "completed"}:
        return "completed"
    if status in {
        "failed",
        "error",
        "infrastructure_failed",
        "timed_out",
        "timeout",
    }:
        return "failed"
    if status in {"cancelled", "interrupted", "not_applicable"}:
        return status
    return str(row.get("status") or row.get("execution_status") or "unknown")


def _projection_queried_projects(row: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in ("queried_projects", "mcp_queried_projects", "project_reads"):
        raw = row.get(key) or ()
        if isinstance(raw, str):
            result.add(raw)
        elif isinstance(raw, Sequence):
            result.update(str(item) for item in raw if str(item))
    return result


def _projection_tool_activity(row: Mapping[str, Any]) -> tuple[tuple[str, ...], int]:
    local_names: list[str] = []
    local_count = 0
    for item in row.get("tool_calls") or ():
        if isinstance(item, str):
            local_names.append(item)
            local_count += 1
        elif isinstance(item, Mapping):
            name = str(item.get("name") or item.get("tool_name") or "")
            if name:
                local_names.append(name)
            local_count += 1
    normalized_mcp_count = 0
    for item in row.get("mcp_tool_calls") or ():
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("tool") or item.get("name") or item.get("tool_name") or "")
        if name:
            local_names.append(name)
        normalized_mcp_count += 1
    explicit = row.get("mcp_tool_names") or ()
    if isinstance(explicit, str):
        local_names.append(explicit)
    elif isinstance(explicit, Sequence):
        local_names.extend(str(item) for item in explicit if str(item))
    traced = row.get("weave_tool_names")
    traced_names: list[str] = []
    traced_count = 0
    if isinstance(traced, Mapping):
        for raw_name, raw_count in traced.items():
            name = str(raw_name)
            count = (
                max(raw_count, 0)
                if isinstance(raw_count, int) and not isinstance(raw_count, bool)
                else 0
            )
            if name and count:
                traced_names.append(name)
            traced_count += count
    declared = row.get("tool_call_count")
    if isinstance(declared, int) and not isinstance(declared, bool):
        local_count = max(local_count, declared)
    if normalized_mcp_count:
        return tuple(sorted(set(local_names))), normalized_mcp_count
    if traced_names:
        return tuple(sorted(set(traced_names))), traced_count
    return tuple(sorted(set(local_names))), local_count


def _projection_structured_result(output: Any) -> Any:
    if not isinstance(output, str):
        return output
    text = output.strip()
    if not text:
        return output
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    fenced: list[Any] = []
    index = 0
    while True:
        start = text.find("```", index)
        if start < 0:
            break
        line_end = text.find("\n", start + 3)
        if line_end < 0:
            break
        fence_end = text.find("```", line_end + 1)
        if fence_end < 0:
            break
        language = text[start + 3 : line_end].strip().lower()
        body = text[line_end + 1 : fence_end].strip()
        if language in {"", "json"}:
            try:
                parsed, end = decoder.raw_decode(body)
            except json.JSONDecodeError:
                pass
            else:
                if not body[end:].strip():
                    fenced.append(parsed)
        index = fence_end + 3
    return fenced[0] if len(fenced) == 1 else output


def _projection_answer(row: Mapping[str, Any]) -> Any:
    for key in ("agent_response", "final_output", "answer"):
        if row.get(key) is not None:
            return row.get(key)
    return None


def _projection_answer_excerpt(row: Mapping[str, Any]) -> str | None:
    raw = _projection_answer(row)
    if raw is None:
        return None
    structured = _projection_structured_result(raw)
    if isinstance(structured, Mapping | list | tuple):
        text = json.dumps(structured, sort_keys=True, separators=(",", ":"))
    else:
        text = str(raw)
    text = " ".join(text.strip().split())
    if not text:
        return None
    return text.encode()[:1000].decode("utf-8", errors="ignore").rstrip()


def _projection_reported_project(row: Mapping[str, Any]) -> str | None:
    raw = _projection_structured_result(_projection_answer(row))
    if not isinstance(raw, Mapping):
        return None
    value = raw.get("source_project")
    if value is None:
        value = raw.get("project")
    return str(value).strip() or None if value is not None else None


def _projection_score_explanation(
    dimension: str,
    passed: bool | None,
    *,
    row: Mapping[str, Any],
) -> str:
    label = dimension.rsplit(".", 1)[-1].replace("_", " ")
    if dimension.startswith("comparison.judge."):
        return "Blind judge score; no rationale or private truth is published."
    if passed is None:
        return f"{label}: the required evidence was unavailable."
    if dimension.endswith(("locked_project_scope", "locked_source_scope")):
        actual = sorted(_projection_queried_projects(row))
        reported = _projection_reported_project(row)
        if passed:
            return (
                "locked project scope passed; actual MCP reads stayed in "
                f"{', '.join(actual) if actual else 'the locked scope'}."
            )
        if actual and reported:
            return (
                "locked project scope failed in the serialized answer; "
                f"actual MCP scope was {', '.join(actual)}, while the answer "
                f"reported {reported}."
            )
        return (
            "locked project scope failed; the answer or normalized MCP evidence "
            "did not prove the exact locked source project."
        )
    return (
        f"{label} {'passed' if passed else 'failed'} under the pinned "
        "deterministic scorer."
    )


class LocalEvidenceIntegrityError(ValueError):
    """Raised when immutable local evidence cannot be reconciled."""


@dataclass(frozen=True)
class LocalEvidenceDestinationV1:
    """Portable identity for Fugue's canonical local evidence layout.

    The destination deliberately excludes an absolute path. A run may move
    between machines without changing its evidence identity.
    """

    kind: Literal["local"] = "local"
    format: Literal["fugue-evidence"] = LOCAL_EVIDENCE_FORMAT
    layout_version: Literal[1] = LOCAL_EVIDENCE_LAYOUT_VERSION
    schema_version: Literal[1] = LOCAL_EVIDENCE_SCHEMA_VERSION
    destination_digest: str = ""

    def __post_init__(self) -> None:
        if self.kind != "local":
            raise ValueError("local evidence destination kind must be local")
        if self.format != LOCAL_EVIDENCE_FORMAT:
            raise ValueError("unsupported local evidence format")
        if self.layout_version != LOCAL_EVIDENCE_LAYOUT_VERSION:
            raise ValueError("unsupported local evidence layout version")
        if self.schema_version != LOCAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported local evidence destination schema")
        computed = self.computed_digest()
        if self.destination_digest and self.destination_digest != computed:
            raise ValueError("local evidence destination digest does not match")
        if not self.destination_digest:
            object.__setattr__(self, "destination_digest", computed)

    def computed_digest(self) -> str:
        return stable_digest(
            {
                "schema_version": self.schema_version,
                "kind": self.kind,
                "format": self.format,
                "layout_version": self.layout_version,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "format": self.format,
            "layout_version": self.layout_version,
            "destination_digest": self.computed_digest(),
        }


@dataclass(frozen=True)
class LocalArtifactRefV1:
    path: str
    sha256: str
    size_bytes: int
    media_type: str = "application/json"

    def __post_init__(self) -> None:
        _portable_path(self.path, "local evidence artifact path")
        _digest(self.sha256, "local evidence artifact digest")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("local evidence artifact size must be nonnegative")
        _required_text(self.media_type, "local evidence artifact media type")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentEvidenceReceiptV1:
    """Honest local correlation for one native Agent execution.

    This is not a Weave Call and must never be presented as one. The receipt
    proves which isolated local session artifacts belong to the Fugue attempt.
    """

    attempt_id: str
    planned_conversation_id: str
    primary_session_id: str | None
    child_session_ids: tuple[str, ...]
    artifacts: tuple[LocalArtifactRefV1, ...]
    transcript_artifact: LocalArtifactRefV1 | None
    transcript_session_id: str | None
    correlation_verified: bool
    tool_event_count: int | None
    tool_events_sha256: str | None
    response_sha256: str | None
    correlation_method: Literal["isolated_trial_directory_v1"] = (
        "isolated_trial_directory_v1"
    )
    status: EvidenceStatus = "resolved"
    native_weave_call: Literal[False] = False
    reason: str | None = None

    def __post_init__(self) -> None:
        _digest(self.attempt_id, "Agent evidence attempt id")
        _required_text(
            self.planned_conversation_id,
            "Agent evidence planned conversation id",
        )
        if self.primary_session_id is not None:
            _required_text(self.primary_session_id, "Agent evidence primary session id")
        for value in self.child_session_ids:
            _required_text(value, "Agent evidence child session id")
        if len(set(self.child_session_ids)) != len(self.child_session_ids):
            raise ValueError("Agent evidence child session ids must be unique")
        if self.primary_session_id in set(self.child_session_ids):
            raise ValueError("Agent primary session cannot also be a child session")
        if self.transcript_session_id is not None:
            _required_text(
                self.transcript_session_id,
                "Agent evidence transcript session id",
            )
        if not isinstance(self.correlation_verified, bool):
            raise ValueError("Agent evidence correlation_verified must be boolean")
        if self.tool_event_count is not None and (
            isinstance(self.tool_event_count, bool) or self.tool_event_count < 0
        ):
            raise ValueError("Agent evidence tool event count must be nonnegative")
        if self.tool_events_sha256 is not None:
            _digest(self.tool_events_sha256, "Agent tool-event digest")
        if self.response_sha256 is not None:
            _digest(self.response_sha256, "Agent response digest")
        if self.correlation_method != "isolated_trial_directory_v1":
            raise ValueError("unsupported Agent evidence correlation method")
        if self.native_weave_call is not False:
            raise ValueError("local Agent evidence cannot be a native Weave Call")
        if self.status not in _EVIDENCE_STATUSES:
            raise ValueError("unsupported Agent evidence status")
        if self.status == "resolved":
            if (
                not self.primary_session_id
                or not self.artifacts
                or self.transcript_artifact is None
                or self.transcript_session_id != self.primary_session_id
                or not self.correlation_verified
                or self.tool_event_count is None
                or self.tool_events_sha256 is None
            ):
                raise ValueError(
                    "resolved Agent evidence requires a verified primary-session "
                    "transcript and tool-event receipt"
                )
            if self.transcript_artifact not in self.artifacts:
                raise ValueError(
                    "Agent transcript artifact must be included in Agent artifacts"
                )
            if self.reason:
                raise ValueError("resolved Agent evidence cannot have a reason")
        elif not self.reason:
            raise ValueError("unresolved Agent evidence requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "planned_conversation_id": self.planned_conversation_id,
            "primary_session_id": self.primary_session_id,
            "child_session_ids": list(self.child_session_ids),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "transcript_artifact": (
                self.transcript_artifact.to_dict()
                if self.transcript_artifact is not None
                else None
            ),
            "transcript_session_id": self.transcript_session_id,
            "correlation_verified": self.correlation_verified,
            "tool_event_count": self.tool_event_count,
            "tool_events_sha256": self.tool_events_sha256,
            "response_sha256": self.response_sha256,
            "correlation_method": self.correlation_method,
            "status": self.status,
            "native_weave_call": self.native_weave_call,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LocalEvidenceNodeV1:
    kind: EvidenceNodeKind
    status: EvidenceStatus
    ref: str
    artifact: LocalArtifactRefV1 | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _NODE_KINDS:
            raise ValueError(f"unsupported local evidence node kind: {self.kind}")
        if self.status not in _EVIDENCE_STATUSES:
            raise ValueError("unsupported local evidence node status")
        _local_ref(self.ref, "local evidence node ref")
        if self.status == "resolved":
            if self.artifact is None:
                raise ValueError("resolved local evidence node requires an artifact")
            if self.reason:
                raise ValueError("resolved local evidence node cannot have a reason")
        elif not self.reason:
            raise ValueError("unresolved local evidence node requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "ref": self.ref,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LocalEvidenceEdgeV1:
    relationship: Literal[
        "evaluation_has_dataset",
        "evaluation_has_prediction_and_score",
        "prediction_and_score_has_prediction",
        "prediction_has_agent_root",
    ]
    source_ref: str
    target_ref: str
    status: EdgeStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.relationship not in _EDGE_RELATIONSHIPS:
            raise ValueError(
                f"unsupported local evidence relationship: {self.relationship}"
            )
        _local_ref(self.source_ref, "local evidence edge source")
        _local_ref(self.target_ref, "local evidence edge target")
        if self.source_ref == self.target_ref:
            raise ValueError("local evidence edge cannot reference itself")
        if self.status not in _EDGE_STATUSES:
            raise ValueError("unsupported local evidence edge status")
        if self.status == "verified" and self.reason:
            raise ValueError("verified local evidence edge cannot have a reason")
        if self.status == "invalid" and not self.reason:
            raise ValueError("invalid local evidence edge requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalAttemptPlanV1:
    run_id: str
    cell_id: str
    attempt_id: str
    attempt_identity: dict[str, Any]
    prediction_id: str
    evaluation_scope_id: str
    dataset_id: str

    def __post_init__(self) -> None:
        _run_id(self.run_id)
        _required_text(self.cell_id, "local evidence cell id")
        _digest(self.attempt_id, "local evidence attempt id")
        _digest(self.prediction_id, "local evidence prediction id")
        _digest(self.evaluation_scope_id, "local evidence evaluation scope id")
        _digest(self.dataset_id, "local evidence dataset id")
        expected_identity = _canonical_attempt_identity(self.attempt_identity)
        if self.attempt_identity != expected_identity:
            raise ValueError("local evidence attempt identity is not canonical")
        if canonical_attempt_id(**expected_identity) != self.attempt_id:
            raise ValueError("local evidence attempt id disagrees with its identity")

    @property
    def candidate_id(self) -> str:
        return str(self.attempt_identity["candidate"])

    @property
    def evaluation_root_id(self) -> str:
        return stable_digest(
            {
                "schema_version": LOCAL_EVIDENCE_SCHEMA_VERSION,
                "run_id": self.run_id,
                "evaluation_scope_id": self.evaluation_scope_id,
                "candidate_id": self.candidate_id,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "attempt_identity": dict(self.attempt_identity),
            "prediction_id": self.prediction_id,
            "evaluation_scope_id": self.evaluation_scope_id,
            "dataset_id": self.dataset_id,
        }


@dataclass(frozen=True)
class LocalEvidenceRunPlanV1:
    run_id: str
    destination: LocalEvidenceDestinationV1
    run_snapshot_sha256: str
    evaluation_asset_lock_sha256: str
    attempts: tuple[LocalAttemptPlanV1, ...]
    require_run_conformance: bool = True
    schema_version: Literal[1] = LOCAL_EVIDENCE_SCHEMA_VERSION
    plan_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != LOCAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported local evidence run-plan schema")
        _run_id(self.run_id)
        _digest(self.run_snapshot_sha256, "run snapshot digest")
        _digest(self.evaluation_asset_lock_sha256, "evaluation asset lock digest")
        if not self.attempts:
            raise ValueError("local evidence run plan requires at least one attempt")
        if tuple(sorted(self.attempts, key=lambda item: item.attempt_id)) != self.attempts:
            raise ValueError("local evidence run-plan attempts must be sorted")
        attempt_ids = [item.attempt_id for item in self.attempts]
        cell_ids = [item.cell_id for item in self.attempts]
        prediction_ids = [item.prediction_id for item in self.attempts]
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("local evidence run-plan attempt ids must be unique")
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("local evidence run-plan cell ids must be unique")
        if len(set(prediction_ids)) != len(prediction_ids):
            raise ValueError("local evidence run-plan prediction ids must be unique")
        if any(item.run_id != self.run_id for item in self.attempts):
            raise ValueError("local evidence attempt belongs to another run")
        computed = self.computed_digest()
        if self.plan_digest and self.plan_digest != computed:
            raise ValueError("local evidence run-plan digest does not match")
        if not self.plan_digest:
            object.__setattr__(self, "plan_digest", computed)

    def computed_digest(self) -> str:
        return stable_digest(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "destination": self.destination.to_dict(),
                "run_snapshot_sha256": self.run_snapshot_sha256,
                "evaluation_asset_lock_sha256": (
                    self.evaluation_asset_lock_sha256
                ),
                "attempts": [item.to_dict() for item in self.attempts],
                "require_run_conformance": self.require_run_conformance,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "destination": self.destination.to_dict(),
            "run_snapshot_sha256": self.run_snapshot_sha256,
            "evaluation_asset_lock_sha256": self.evaluation_asset_lock_sha256,
            "attempts": [item.to_dict() for item in self.attempts],
            "require_run_conformance": self.require_run_conformance,
            "plan_digest": self.computed_digest(),
        }


@dataclass(frozen=True)
class LocalAttemptEvidenceV1:
    attempt: LocalAttemptPlanV1
    terminal_status: AttemptTerminalStatus
    nodes: tuple[LocalEvidenceNodeV1, ...]
    edges: tuple[LocalEvidenceEdgeV1, ...]
    prediction_row_sha256: str | None
    result_row_projection_digest: str | None
    agent_receipt: AgentEvidenceReceiptV1
    receipts: dict[str, dict[str, Any]]
    recorded_at: str
    schema_version: Literal[1] = LOCAL_EVIDENCE_SCHEMA_VERSION
    record_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != LOCAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported local attempt-evidence schema")
        _timestamp(self.recorded_at, "local attempt evidence timestamp")
        if self.terminal_status not in _TERMINAL_STATUSES:
            raise ValueError("unsupported local evidence terminal status")
        if self.prediction_row_sha256 is not None:
            _digest(self.prediction_row_sha256, "prediction row digest")
        if self.result_row_projection_digest is not None:
            _digest(
                self.result_row_projection_digest,
                "result row projection digest",
            )
        if self.terminal_status in {"passed", "failed"} and not self.prediction_row_sha256:
            raise ValueError("executed attempts require a prediction row digest")
        if self.agent_receipt.attempt_id != self.attempt.attempt_id:
            raise ValueError("Agent evidence receipt belongs to another attempt")
        if set(self.receipts) != {"privacy", "policy", "usage", "cleanup"}:
            raise ValueError(
                "local attempt evidence requires privacy, policy, usage, and "
                "cleanup receipts"
            )
        _assert_public_payload(self.receipts)
        receipt_statuses: set[str] = set()
        for name, receipt in self.receipts.items():
            if not isinstance(receipt, Mapping):
                raise ValueError(f"local {name} receipt must be an object")
            status = str(receipt.get("status") or "")
            if status not in {
                "passed",
                "failed",
                "unavailable",
                "not_applicable",
            }:
                raise ValueError(f"local {name} receipt has an unsupported status")
            receipt_statuses.add(status)
        nodes = {item.kind: item for item in self.nodes}
        if len(self.nodes) != 5 or set(nodes) != _NODE_KINDS:
            raise ValueError(
                "local attempt evidence requires exactly five unique nodes"
            )
        expected_refs = local_attempt_refs(self.attempt)
        if any(nodes[kind].ref != expected_refs[kind] for kind in _NODE_KINDS):
            raise ValueError("local evidence node ref disagrees with its attempt")
        edges = {item.relationship: item for item in self.edges}
        if len(self.edges) != 4 or tuple(sorted(edges)) != tuple(
            sorted(_EDGE_RELATIONSHIPS)
        ):
            raise ValueError(
                "local attempt evidence requires exactly four unique edges"
            )
        expected_edges = _expected_edges(expected_refs)
        for relationship, (source_ref, target_ref) in expected_edges.items():
            edge = edges[relationship]
            if (edge.source_ref, edge.target_ref) != (source_ref, target_ref):
                raise ValueError(
                    f"local evidence edge {relationship} has the wrong endpoints"
                )
            if edge.status == "verified" and (
                nodes[_ref_kind(source_ref, expected_refs)].status != "resolved"
                or nodes[_ref_kind(target_ref, expected_refs)].status != "resolved"
            ):
                raise ValueError(
                    "verified local evidence edge requires resolved endpoints"
                )
        computed = self.computed_digest()
        if self.record_digest and self.record_digest != computed:
            raise ValueError("local attempt evidence digest does not match")
        if not self.record_digest:
            object.__setattr__(self, "record_digest", computed)

    @property
    def integrity_status(self) -> AttemptIntegrityStatus:
        statuses = {
            *(item.status for item in self.nodes),
            *(item.status for item in self.edges),
            self.agent_receipt.status,
        }
        if "invalid" in statuses:
            return "invalid"
        receipt_statuses = {
            str(receipt.get("status") or "unavailable")
            for receipt in self.receipts.values()
        }
        if "failed" in receipt_statuses:
            return "invalid"
        if statuses <= {"resolved", "verified"} and receipt_statuses <= {
            "passed",
            "not_applicable",
        }:
            return "resolved"
        return "incomplete"

    def computed_digest(self) -> str:
        return stable_digest(self._unsigned_dict())

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt": self.attempt.to_dict(),
            "terminal_status": self.terminal_status,
            "integrity_status": self.integrity_status,
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "prediction_row_sha256": self.prediction_row_sha256,
            **(
                {
                    "result_row_projection_digest": (
                        self.result_row_projection_digest
                    )
                }
                if self.result_row_projection_digest is not None
                else {}
            ),
            "agent_receipt": self.agent_receipt.to_dict(),
            "receipts": self.receipts,
            "recorded_at": self.recorded_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned_dict(), "record_digest": self.computed_digest()}


@dataclass(frozen=True)
class LocalAttemptRecordRefV1:
    attempt_id: str
    path: str
    sha256: str
    record_digest: str
    integrity_status: AttemptIntegrityStatus
    terminal_status: AttemptTerminalStatus
    prediction_row_sha256: str | None
    result_row_projection_digest: str | None = None

    def __post_init__(self) -> None:
        _digest(self.attempt_id, "attempt record attempt id")
        _portable_path(self.path, "attempt record path")
        _digest(self.sha256, "attempt record file digest")
        _digest(self.record_digest, "attempt record digest")
        if self.integrity_status not in _INTEGRITY_STATUSES:
            raise ValueError("unsupported local attempt integrity status")
        if self.terminal_status not in _TERMINAL_STATUSES:
            raise ValueError("unsupported local attempt terminal status")
        if self.prediction_row_sha256 is not None:
            _digest(self.prediction_row_sha256, "attempt prediction row digest")
        if self.result_row_projection_digest is not None:
            _digest(
                self.result_row_projection_digest,
                "attempt result row projection digest",
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.result_row_projection_digest is None:
            value.pop("result_row_projection_digest")
        return value


@dataclass(frozen=True)
class LocalRunConformanceRefV1:
    """Digest-bound final Harbor privacy/policy/cleanup receipt."""

    path: str
    sha256: str
    receipt_sha256: str
    status: RunConformanceStatus
    enforced: bool

    def __post_init__(self) -> None:
        _portable_path(self.path, "run conformance path")
        _digest(self.sha256, "run conformance file digest")
        _digest(self.receipt_sha256, "run conformance receipt digest")
        if self.status not in {
            "passed",
            "failed",
            "unavailable",
            "not_applicable",
        }:
            raise ValueError("unsupported run conformance status")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalEvidenceManifestV1:
    run_id: str
    destination: LocalEvidenceDestinationV1
    run_snapshot_sha256: str
    evaluation_asset_lock_sha256: str
    plan_digest: str
    planned_attempt_ids: tuple[str, ...]
    terminal_attempt_ids: tuple[str, ...]
    attempt_records: tuple[LocalAttemptRecordRefV1, ...]
    attempt_record_set_digest: str
    prediction_row_set_digest: str
    run_conformance: LocalRunConformanceRefV1 | None
    run_conformance_required: bool
    status: ManifestStatus
    created_at: str
    schema_version: Literal[1] = LOCAL_EVIDENCE_SCHEMA_VERSION
    manifest_digest: str = ""
    result_row_projection_set_digest: str | None = None

    def __post_init__(self) -> None:  # noqa: C901 - validates one immutable graph
        if self.schema_version != LOCAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported local evidence manifest schema")
        _run_id(self.run_id)
        for label, value in (
            ("run snapshot digest", self.run_snapshot_sha256),
            ("evaluation asset lock digest", self.evaluation_asset_lock_sha256),
            ("local evidence plan digest", self.plan_digest),
            ("attempt record set digest", self.attempt_record_set_digest),
            ("prediction row set digest", self.prediction_row_set_digest),
        ):
            _digest(value, label)
        if self.result_row_projection_set_digest is not None:
            _digest(
                self.result_row_projection_set_digest,
                "result row projection set digest",
            )
        _timestamp(self.created_at, "local evidence manifest timestamp")
        if self.status not in _MANIFEST_STATUSES:
            raise ValueError("unsupported local evidence manifest status")
        if tuple(sorted(set(self.planned_attempt_ids))) != self.planned_attempt_ids:
            raise ValueError("planned local evidence attempt ids must be sorted/unique")
        if tuple(sorted(set(self.terminal_attempt_ids))) != self.terminal_attempt_ids:
            raise ValueError("terminal local evidence attempt ids must be sorted/unique")
        if not set(self.terminal_attempt_ids) <= set(self.planned_attempt_ids):
            raise ValueError("terminal local evidence attempt was not planned")
        for attempt_id in (*self.planned_attempt_ids, *self.terminal_attempt_ids):
            _digest(attempt_id, "local evidence manifest attempt id")
        if tuple(sorted(self.attempt_records, key=lambda item: item.attempt_id)) != (
            self.attempt_records
        ):
            raise ValueError("local evidence attempt records must be sorted")
        record_ids = tuple(item.attempt_id for item in self.attempt_records)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("local evidence attempt record ids must be unique")
        if record_ids != self.terminal_attempt_ids:
            raise ValueError(
                "local evidence terminal attempts disagree with record references"
            )
        if self.attempt_record_set_digest != stable_digest(
            [item.to_dict() for item in self.attempt_records]
        ):
            raise ValueError("local evidence attempt record set digest does not match")
        if self.prediction_row_set_digest != stable_digest(
            [
                [item.attempt_id, item.prediction_row_sha256]
                for item in self.attempt_records
            ]
        ):
            raise ValueError("local evidence prediction row set digest does not match")
        projection_digests = tuple(
            item.result_row_projection_digest for item in self.attempt_records
        )
        if self.result_row_projection_set_digest is None:
            if any(projection_digests):
                raise ValueError(
                    "local evidence result row projections require a set digest"
                )
        elif not all(projection_digests) or (
            self.result_row_projection_set_digest
            != stable_digest(
                [
                    [item.attempt_id, item.result_row_projection_digest]
                    for item in self.attempt_records
                ]
            )
        ):
            raise ValueError(
                "local evidence result row projection set digest does not match"
            )
        if self.run_conformance_required and self.run_conformance is None:
            conformance_status = "missing"
        elif self.run_conformance is None:
            conformance_status = "not_required"
        elif (
            self.run_conformance.status == "passed"
            and self.run_conformance.enforced
        ):
            conformance_status = "resolved"
        elif self.run_conformance.status == "failed":
            conformance_status = "invalid"
        else:
            conformance_status = "incomplete"
        expected_status: ManifestStatus = (
            "invalid"
            if any(item.integrity_status == "invalid" for item in self.attempt_records)
            or conformance_status == "invalid"
            else "complete"
            if self.terminal_attempt_ids == self.planned_attempt_ids
            and all(
                item.integrity_status == "resolved" for item in self.attempt_records
            )
            and conformance_status in {"resolved", "not_required"}
            else "incomplete"
        )
        if self.status != expected_status:
            raise ValueError("local evidence manifest status does not match its records")
        computed = self.computed_digest()
        if self.manifest_digest and self.manifest_digest != computed:
            raise ValueError("local evidence manifest digest does not match")
        if not self.manifest_digest:
            object.__setattr__(self, "manifest_digest", computed)

    def computed_digest(self) -> str:
        return stable_digest(self._unsigned_dict())

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "destination": self.destination.to_dict(),
            "run_snapshot_sha256": self.run_snapshot_sha256,
            "evaluation_asset_lock_sha256": self.evaluation_asset_lock_sha256,
            "plan_digest": self.plan_digest,
            "planned_attempt_ids": list(self.planned_attempt_ids),
            "terminal_attempt_ids": list(self.terminal_attempt_ids),
            "attempt_records": [item.to_dict() for item in self.attempt_records],
            "attempt_record_set_digest": self.attempt_record_set_digest,
            "prediction_row_set_digest": self.prediction_row_set_digest,
            **(
                {
                    "result_row_projection_set_digest": (
                        self.result_row_projection_set_digest
                    )
                }
                if self.result_row_projection_set_digest is not None
                else {}
            ),
            "run_conformance": (
                self.run_conformance.to_dict()
                if self.run_conformance is not None
                else None
            ),
            "run_conformance_required": self.run_conformance_required,
            "status": self.status,
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned_dict(), "manifest_digest": self.computed_digest()}


@dataclass(frozen=True)
class LocalEvidencePublicationReceiptV1:
    """Immutable receipt for the verified, provider-neutral local manifest."""

    run_id: str
    destination: LocalEvidenceDestinationV1
    plan_digest: str
    manifest_digest: str
    manifest_path: str
    manifest_file_sha256: str
    attempt_record_set_digest: str
    prediction_row_set_digest: str
    published_at: str
    schema_version: Literal[1] = LOCAL_EVIDENCE_SCHEMA_VERSION
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != LOCAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported local publication-receipt schema")
        _run_id(self.run_id)
        for label, value in (
            ("publication plan digest", self.plan_digest),
            ("publication manifest digest", self.manifest_digest),
            ("publication manifest file digest", self.manifest_file_sha256),
            ("publication attempt-record-set digest", self.attempt_record_set_digest),
            ("publication prediction-row-set digest", self.prediction_row_set_digest),
        ):
            _digest(value, label)
        _portable_path(self.manifest_path, "publication manifest path")
        _timestamp(self.published_at, "local publication timestamp")
        computed = self.computed_digest()
        if self.receipt_digest and self.receipt_digest != computed:
            raise ValueError("local publication receipt digest does not match")
        if not self.receipt_digest:
            object.__setattr__(self, "receipt_digest", computed)

    def computed_digest(self) -> str:
        return stable_digest(self._unsigned_dict())

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "destination": self.destination.to_dict(),
            "plan_digest": self.plan_digest,
            "manifest_digest": self.manifest_digest,
            "manifest_path": self.manifest_path,
            "manifest_file_sha256": self.manifest_file_sha256,
            "attempt_record_set_digest": self.attempt_record_set_digest,
            "prediction_row_set_digest": self.prediction_row_set_digest,
            "published_at": self.published_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned_dict(), "receipt_digest": self.computed_digest()}


@dataclass(frozen=True)
class LocalEvidenceEventV1:
    """One best-effort lifecycle journal entry.

    Events aid operator diagnostics and recovery, but they are not part of the
    canonical evidence chain. The immutable plan, attempt records, artifact
    digests, manifest, and publication receipt are the authoritative ledger.
    """

    run_id: str
    event: Literal[
        "run_initialized",
        "attempt_opened",
        "attempt_finalized",
        "manifest_finalized",
        "publication_receipt_written",
    ]
    recorded_at: str
    attempt_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    schema_version: Literal[1] = LOCAL_EVIDENCE_SCHEMA_VERSION
    event_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != LOCAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported local evidence event schema")
        _run_id(self.run_id)
        _timestamp(self.recorded_at, "local evidence event timestamp")
        if self.event not in _EVENT_TYPES:
            raise ValueError("unsupported local evidence event")
        if self.attempt_id is not None:
            _digest(self.attempt_id, "local evidence event attempt id")
        attempt_event = self.event in {"attempt_opened", "attempt_finalized"}
        if attempt_event != (self.attempt_id is not None):
            raise ValueError(
                "local evidence event attempt identity does not match its type"
            )
        _assert_public_payload(self.details)
        computed = self.computed_digest()
        if self.event_digest and self.event_digest != computed:
            raise ValueError("local evidence event digest does not match")
        if not self.event_digest:
            object.__setattr__(self, "event_digest", computed)

    def computed_digest(self) -> str:
        return stable_digest(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "event": self.event,
                "recorded_at": self.recorded_at,
                "attempt_id": self.attempt_id,
                "details": self.details,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "event": self.event,
            "recorded_at": self.recorded_at,
            "attempt_id": self.attempt_id,
            "details": self.details,
            "event_digest": self.computed_digest(),
        }


class LocalEvidenceStore:
    """Digest-verified canonical evidence owned by one Fugue run.

    ``events.jsonl`` is a non-authoritative diagnostic journal. It is parsed
    only when explicitly requested and cannot create, erase, or invalidate a
    behavioral claim; canonical integrity is recomputed from immutable records
    and their artifacts.
    """

    def __init__(self, repo_root: Path, run_id: str) -> None:
        _run_id(run_id)
        self.repo_root = repo_root.resolve()
        self.run_id = run_id
        self.root = self.repo_root / ".fugue" / "runtime" / run_id / "evidence"
        self.run_conformance_path = self.root.parent / "harbor-conformance.json"
        self.plan_path = self.root / "plan.json"
        self.events_path = self.root / "events.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.publication_receipt_path = self.root / "publication-receipt.json"
        self.lock_path = self.root / ".evidence.lock"

    def initialize(self, plan: LocalEvidenceRunPlanV1) -> Path:
        if plan.run_id != self.run_id:
            raise ValueError("local evidence plan belongs to another run")
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(self.lock_path, timeout=120):
            created = not self.plan_path.exists()
            _write_immutable_json(self.plan_path, plan.to_dict())
            if created:
                self._append_event_unlocked(
                    LocalEvidenceEventV1(
                        run_id=self.run_id,
                        event="run_initialized",
                        recorded_at=_now(),
                        details={"plan_digest": plan.plan_digest},
                    )
                )
        return self.plan_path

    def read_plan(self) -> LocalEvidenceRunPlanV1:
        if not self.plan_path.is_file():
            raise FileNotFoundError(f"local evidence plan not found: {self.plan_path}")
        return local_evidence_run_plan_from_dict(_read_mapping(self.plan_path))

    def begin_attempt(self, attempt_id: str) -> dict[str, str]:
        _digest(attempt_id, "local evidence attempt id")
        plan = self.read_plan()
        attempt = _attempt_from_plan(plan, attempt_id)
        if self._attempt_record_path(attempt_id).is_file():
            raise LocalEvidenceIntegrityError(
                f"local evidence attempt is already terminal: {attempt_id}"
            )
        marker = self.root / "opened" / f"{attempt_id}.json"
        with FileLock(self.lock_path, timeout=120):
            if not marker.exists():
                opened_at = _now()
                _write_immutable_json(
                    marker,
                    {
                        "schema_version": LOCAL_EVIDENCE_SCHEMA_VERSION,
                        "run_id": self.run_id,
                        "attempt_id": attempt_id,
                        "plan_digest": plan.plan_digest,
                        "opened_at": opened_at,
                    },
                )
                self._append_event_unlocked(
                    LocalEvidenceEventV1(
                        run_id=self.run_id,
                        event="attempt_opened",
                        recorded_at=opened_at,
                        attempt_id=attempt_id,
                    )
                )
        return {
            "FUGUE_ATTEMPT_ID": attempt.attempt_id,
            "FUGUE_EVALUATION_SCOPE_ID": attempt.evaluation_scope_id,
            "FUGUE_LOCAL_PREDICTION_ID": attempt.prediction_id,
            "FUGUE_LOCAL_EVIDENCE_ROOT": self.root.as_posix(),
            "FUGUE_LOCAL_EVIDENCE_PLAN_DIGEST": plan.plan_digest,
        }

    def write_artifact(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        secret_values: Sequence[str] = (),
    ) -> LocalArtifactRefV1:
        relative = _portable_path(path, "local evidence artifact path")
        value = _public_payload(payload, secret_values=secret_values)
        target = self._target(relative)
        with FileLock(self.lock_path, timeout=120):
            _write_immutable_json(target, value)
        raw = target.read_bytes()
        return LocalArtifactRefV1(
            path=relative.as_posix(),
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
        )

    def write_attempt(self, record: LocalAttemptEvidenceV1) -> Path:
        plan = self.read_plan()
        planned = _attempt_from_plan(plan, record.attempt.attempt_id)
        if record.attempt != planned:
            raise ValueError("local evidence record disagrees with its run plan")
        self._verify_record_artifacts(record)
        path = self._attempt_record_path(record.attempt.attempt_id)
        with FileLock(self.lock_path, timeout=120):
            created = not path.exists()
            _write_immutable_json(path, record.to_dict())
            if created:
                self._append_event_unlocked(
                    LocalEvidenceEventV1(
                        run_id=self.run_id,
                        event="attempt_finalized",
                        recorded_at=record.recorded_at,
                        attempt_id=record.attempt.attempt_id,
                        details={
                            "record_digest": record.record_digest,
                            "terminal_status": record.terminal_status,
                            "integrity_status": record.integrity_status,
                        },
                    )
                )
        return path

    def read_attempt(self, attempt_id: str) -> LocalAttemptEvidenceV1:
        path = self._attempt_record_path(attempt_id)
        if not path.is_file():
            raise FileNotFoundError(f"local attempt evidence not found: {attempt_id}")
        return local_attempt_evidence_from_dict(_read_mapping(path))

    def build_manifest(self) -> LocalEvidenceManifestV1:
        plan = self.read_plan()
        record_refs: list[LocalAttemptRecordRefV1] = []
        for attempt in plan.attempts:
            path = self._attempt_record_path(attempt.attempt_id)
            if not path.is_file():
                continue
            record = local_attempt_evidence_from_dict(_read_mapping(path))
            if record.attempt != attempt:
                raise LocalEvidenceIntegrityError(
                    f"attempt record changed identity: {attempt.attempt_id}"
                )
            self._verify_record_artifacts(record)
            raw = path.read_bytes()
            record_refs.append(
                LocalAttemptRecordRefV1(
                    attempt_id=attempt.attempt_id,
                    path=path.relative_to(self.root).as_posix(),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    record_digest=record.record_digest,
                    integrity_status=record.integrity_status,
                    terminal_status=record.terminal_status,
                    prediction_row_sha256=record.prediction_row_sha256,
                    result_row_projection_digest=(
                        record.result_row_projection_digest
                    ),
                )
            )
        records = tuple(sorted(record_refs, key=lambda item: item.attempt_id))
        planned_ids = tuple(item.attempt_id for item in plan.attempts)
        terminal_ids = tuple(item.attempt_id for item in records)
        existing_created_at = None
        if self.manifest_path.is_file():
            existing_created_at = str(
                _read_mapping(self.manifest_path).get("created_at") or ""
            )
        run_conformance = self._read_run_conformance_ref()
        conformance_invalid = bool(
            run_conformance is not None and run_conformance.status == "failed"
        )
        conformance_complete = bool(
            not plan.require_run_conformance
            or (
                run_conformance is not None
                and run_conformance.status == "passed"
                and run_conformance.enforced
            )
        )
        status: ManifestStatus = (
            "invalid"
            if any(item.integrity_status == "invalid" for item in records)
            or conformance_invalid
            else "complete"
            if terminal_ids == planned_ids
            and all(item.integrity_status == "resolved" for item in records)
            and conformance_complete
            else "incomplete"
        )
        return LocalEvidenceManifestV1(
            run_id=self.run_id,
            destination=plan.destination,
            run_snapshot_sha256=plan.run_snapshot_sha256,
            evaluation_asset_lock_sha256=plan.evaluation_asset_lock_sha256,
            plan_digest=plan.plan_digest,
            planned_attempt_ids=planned_ids,
            terminal_attempt_ids=terminal_ids,
            attempt_records=records,
            attempt_record_set_digest=stable_digest(
                [item.to_dict() for item in records]
            ),
            prediction_row_set_digest=stable_digest(
                [
                    [item.attempt_id, item.prediction_row_sha256]
                    for item in records
                ]
            ),
            result_row_projection_set_digest=(
                stable_digest(
                    [
                        [item.attempt_id, item.result_row_projection_digest]
                        for item in records
                    ]
                )
                if records
                and all(
                    item.result_row_projection_digest is not None
                    for item in records
                )
                else None
            ),
            run_conformance=run_conformance,
            run_conformance_required=plan.require_run_conformance,
            status=status,
            created_at=existing_created_at or _now(),
        )

    def finalize(self) -> LocalEvidenceManifestV1:
        manifest = self.build_manifest()
        if manifest.status != "complete":
            raise LocalEvidenceIntegrityError(
                "local evidence cannot finalize: "
                f"manifest status is {manifest.status}"
            )
        with FileLock(self.lock_path, timeout=120):
            created = not self.manifest_path.exists()
            _write_immutable_json(self.manifest_path, manifest.to_dict())
            if created:
                self._append_event_unlocked(
                    LocalEvidenceEventV1(
                        run_id=self.run_id,
                        event="manifest_finalized",
                        recorded_at=manifest.created_at,
                        details={"manifest_digest": manifest.manifest_digest},
                    )
                )
            manifest_raw = self.manifest_path.read_bytes()
            published_at = _now()
            if self.publication_receipt_path.is_file():
                published_at = str(
                    _read_mapping(self.publication_receipt_path).get("published_at")
                    or ""
                )
            receipt = LocalEvidencePublicationReceiptV1(
                run_id=self.run_id,
                destination=manifest.destination,
                plan_digest=manifest.plan_digest,
                manifest_digest=manifest.manifest_digest,
                manifest_path="manifest.json",
                manifest_file_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                attempt_record_set_digest=manifest.attempt_record_set_digest,
                prediction_row_set_digest=manifest.prediction_row_set_digest,
                published_at=published_at,
            )
            receipt_created = not self.publication_receipt_path.exists()
            _write_immutable_json(
                self.publication_receipt_path,
                receipt.to_dict(),
            )
            if receipt_created:
                self._append_event_unlocked(
                    LocalEvidenceEventV1(
                        run_id=self.run_id,
                        event="publication_receipt_written",
                        recorded_at=receipt.published_at,
                        details={"receipt_digest": receipt.receipt_digest},
                    )
                )
        return manifest

    def read_manifest(self) -> LocalEvidenceManifestV1:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"local evidence manifest not found: {self.manifest_path}"
            )
        manifest = local_evidence_manifest_from_dict(
            _read_mapping(self.manifest_path)
        )
        recomputed = self.build_manifest()
        if manifest.to_dict() != recomputed.to_dict():
            raise LocalEvidenceIntegrityError(
                "local evidence manifest disagrees with run artifacts"
            )
        return manifest

    def read_publication_receipt(self) -> LocalEvidencePublicationReceiptV1:
        if not self.publication_receipt_path.is_file():
            raise FileNotFoundError(
                "local evidence publication receipt not found: "
                f"{self.publication_receipt_path}"
            )
        receipt = local_evidence_publication_receipt_from_dict(
            _read_mapping(self.publication_receipt_path)
        )
        manifest = self.read_manifest()
        manifest_raw = self.manifest_path.read_bytes()
        expected = (
            receipt.run_id == manifest.run_id
            and receipt.destination == manifest.destination
            and receipt.plan_digest == manifest.plan_digest
            and receipt.manifest_digest == manifest.manifest_digest
            and receipt.manifest_path == "manifest.json"
            and receipt.manifest_file_sha256
            == hashlib.sha256(manifest_raw).hexdigest()
            and receipt.attempt_record_set_digest
            == manifest.attempt_record_set_digest
            and receipt.prediction_row_set_digest
            == manifest.prediction_row_set_digest
        )
        if not expected:
            raise LocalEvidenceIntegrityError(
                "local evidence publication receipt disagrees with its manifest"
            )
        return receipt

    def read_events(self) -> tuple[LocalEvidenceEventV1, ...]:
        if not self.events_path.is_file():
            return ()
        events: list[LocalEvidenceEventV1] = []
        for index, line in enumerate(
            self.events_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LocalEvidenceIntegrityError(
                    f"invalid local evidence event at line {index}"
                ) from exc
            if not isinstance(raw, Mapping):
                raise LocalEvidenceIntegrityError(
                    f"local evidence event at line {index} is not an object"
                )
            event = local_evidence_event_from_dict(raw)
            if event.run_id != self.run_id:
                raise LocalEvidenceIntegrityError(
                    "local evidence event belongs to another run"
                )
            events.append(event)
        return tuple(events)

    def _verify_record_artifacts(self, record: LocalAttemptEvidenceV1) -> None:
        for node in record.nodes:
            if node.artifact is None:
                continue
            target = self._target(Path(node.artifact.path))
            if not target.is_file():
                raise LocalEvidenceIntegrityError(
                    f"local evidence artifact is missing: {node.artifact.path}"
                )
            raw = target.read_bytes()
            if (
                len(raw) != node.artifact.size_bytes
                or hashlib.sha256(raw).hexdigest() != node.artifact.sha256
            ):
                raise LocalEvidenceIntegrityError(
                    f"local evidence artifact digest changed: {node.artifact.path}"
                )
            if node.kind == "prediction" and record.result_row_projection_digest:
                try:
                    prediction = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise LocalEvidenceIntegrityError(
                        "local prediction artifact is not valid JSON"
                    ) from exc
                if not isinstance(prediction, Mapping) or (
                    local_result_row_projection_digest(prediction)
                    != record.result_row_projection_digest
                ):
                    raise LocalEvidenceIntegrityError(
                        "local prediction result projection digest changed"
                    )
        for artifact in record.agent_receipt.artifacts:
            target = (self.repo_root / artifact.path).resolve()
            if self.repo_root != target and self.repo_root not in target.parents:
                raise LocalEvidenceIntegrityError(
                    f"Agent evidence artifact escapes the repository: {artifact.path}"
                )
            if not target.is_file():
                raise LocalEvidenceIntegrityError(
                    f"Agent evidence artifact is missing: {artifact.path}"
                )
            raw = target.read_bytes()
            if (
                len(raw) != artifact.size_bytes
                or hashlib.sha256(raw).hexdigest() != artifact.sha256
            ):
                raise LocalEvidenceIntegrityError(
                    f"Agent evidence artifact digest changed: {artifact.path}"
                )

    def _append_event_unlocked(self, event: LocalEvidenceEventV1) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    event.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def _attempt_record_path(self, attempt_id: str) -> Path:
        _digest(attempt_id, "local evidence attempt id")
        return self.root / "attempt-records" / f"{attempt_id}.json"

    def _read_run_conformance_ref(self) -> LocalRunConformanceRefV1 | None:
        path = self.run_conformance_path
        if not path.is_file():
            return None
        raw = path.read_bytes()
        value = _read_mapping(path)
        if str(value.get("run_id") or "") != self.run_id:
            raise LocalEvidenceIntegrityError(
                "run conformance receipt belongs to another run"
            )
        status = str(value.get("status") or "")
        if status not in {
            "passed",
            "failed",
            "unavailable",
            "not_applicable",
        }:
            raise LocalEvidenceIntegrityError(
                "run conformance receipt has an unsupported status"
            )
        receipt_sha256 = str(value.get("receipt_sha256") or "")
        unsigned = {**value, "receipt_sha256": ""}
        if receipt_sha256 != stable_digest(unsigned):
            raise LocalEvidenceIntegrityError(
                "run conformance receipt digest does not match"
            )
        return LocalRunConformanceRefV1(
            path=path.relative_to(self.repo_root).as_posix(),
            sha256=hashlib.sha256(raw).hexdigest(),
            receipt_sha256=receipt_sha256,
            status=status,  # type: ignore[arg-type]
            enforced=bool(value.get("enforced")),
        )

    def _target(self, relative: Path) -> Path:
        relative = _portable_path(relative.as_posix(), "local evidence path")
        target = (self.root / relative).resolve()
        if self.root.resolve() not in target.parents:
            raise ValueError("local evidence path escapes its run root")
        return target


class LocalEvidenceCoordinator:
    """Provider-neutral lifecycle API ready for OperatorService integration."""

    def __init__(
        self,
        store: LocalEvidenceStore,
        plan: LocalEvidenceRunPlanV1,
        *,
        secret_values: Sequence[str] = (),
    ) -> None:
        if store.run_id != plan.run_id:
            raise ValueError("local evidence store and plan disagree on run id")
        self.store = store
        self.plan = plan
        self.secret_values = tuple(
            value for value in secret_values if isinstance(value, str) and len(value) >= 8
        )
        self.store.initialize(plan)

    def begin_attempt(self, attempt_id: str) -> dict[str, str]:
        return self.store.begin_attempt(attempt_id)

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        terminal_status: AttemptTerminalStatus,
        prediction_row: Mapping[str, Any],
        scores: Mapping[str, Any],
        agent_receipt: AgentEvidenceReceiptV1,
        evaluation_payload: Mapping[str, Any] | None = None,
        dataset_payload: Mapping[str, Any] | None = None,
        attempt_payload: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
    ) -> LocalAttemptEvidenceV1:
        attempt = _attempt_from_plan(self.plan, attempt_id)
        _verify_prediction_identity(prediction_row, attempt)
        safe_prediction = _public_payload(
            prediction_row,
            secret_values=self.secret_values,
        )
        try:
            existing = self.store.read_attempt(attempt_id)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            expected_prediction_digest = (
                stable_digest(safe_prediction)
                if terminal_status in {"passed", "failed"}
                else None
            )
            expected_projection_digest = (
                local_result_row_projection_digest(safe_prediction)
                if terminal_status in {"passed", "failed"}
                else None
            )
            if (
                existing.terminal_status != terminal_status
                or existing.prediction_row_sha256 != expected_prediction_digest
                or existing.result_row_projection_digest
                != expected_projection_digest
                or existing.agent_receipt != agent_receipt
            ):
                raise LocalEvidenceIntegrityError(
                    f"conflicting terminal evidence for attempt: {attempt_id}"
                )
            return existing
        safe_scores = _public_payload(scores, secret_values=self.secret_values)
        safe_evaluation = _public_payload(
            evaluation_payload
            or {
                "run_id": attempt.run_id,
                "evaluation_root_id": attempt.evaluation_root_id,
                "evaluation_scope_id": attempt.evaluation_scope_id,
                "candidate_id": attempt.candidate_id,
            },
            secret_values=self.secret_values,
        )
        safe_dataset = _public_payload(
            dataset_payload
            or {
                "dataset_id": attempt.dataset_id,
                "evaluation_asset_lock_sha256": (
                    self.plan.evaluation_asset_lock_sha256
                ),
            },
            secret_values=self.secret_values,
        )
        safe_attempt = _public_payload(
            attempt_payload
            or {
                "attempt_id": attempt.attempt_id,
                "terminal_status": terminal_status,
            },
            secret_values=self.secret_values,
        )
        receipts_raw = safe_attempt.get("receipts")
        if not isinstance(receipts_raw, Mapping):
            receipts_raw = {
                name: {"status": "unavailable"}
                for name in ("privacy", "policy", "usage", "cleanup")
            }
        receipts = {
            str(name): dict(value)
            for name, value in receipts_raw.items()
            if isinstance(value, Mapping)
        }
        _assert_public_payload(agent_receipt.to_dict(), self.secret_values)

        artifacts = {
            "evaluation_root": self.store.write_artifact(
                (
                    f"evaluations/{attempt.evaluation_root_id}/"
                    "evaluation-root.json"
                ),
                safe_evaluation,
                secret_values=self.secret_values,
            ),
            "dataset": self.store.write_artifact(
                f"datasets/{attempt.dataset_id}.json",
                safe_dataset,
                secret_values=self.secret_values,
            ),
            "prediction": self.store.write_artifact(
                f"predictions/{attempt.prediction_id}.json",
                safe_prediction,
                secret_values=self.secret_values,
            ),
            "agent_root": self.store.write_artifact(
                f"agents/{attempt.attempt_id}.json",
                agent_receipt.to_dict(),
                secret_values=self.secret_values,
            ),
        }
        prediction_row_sha256 = stable_digest(safe_prediction)
        result_row_projection_digest = local_result_row_projection_digest(
            safe_prediction
        )
        artifacts["prediction_and_score"] = self.store.write_artifact(
            f"attempts/{attempt.attempt_id}/prediction-and-score.json",
            {
                "attempt_id": attempt.attempt_id,
                "prediction_id": attempt.prediction_id,
                "prediction_row_sha256": prediction_row_sha256,
                "scores": safe_scores,
                "attempt": safe_attempt,
            },
            secret_values=self.secret_values,
        )
        refs = local_attempt_refs(attempt)
        agent_status = agent_receipt.status
        nodes = tuple(
            LocalEvidenceNodeV1(
                kind=kind,  # type: ignore[arg-type]
                status=(agent_status if kind == "agent_root" else "resolved"),
                ref=refs[kind],
                artifact=artifacts[kind],
                reason=(
                    agent_receipt.reason
                    if kind == "agent_root" and agent_status != "resolved"
                    else None
                ),
            )
            for kind in sorted(_NODE_KINDS)
        )
        edges = tuple(
            LocalEvidenceEdgeV1(
                relationship=relationship,  # type: ignore[arg-type]
                source_ref=source_ref,
                target_ref=target_ref,
                status=(
                    "invalid"
                    if relationship == "prediction_has_agent_root"
                    and agent_status != "resolved"
                    else "verified"
                ),
                reason=(
                    agent_receipt.reason or "Agent evidence did not reconcile"
                    if relationship == "prediction_has_agent_root"
                    and agent_status != "resolved"
                    else None
                ),
            )
            for relationship, (source_ref, target_ref) in _expected_edges(refs).items()
        )
        record = LocalAttemptEvidenceV1(
            attempt=attempt,
            terminal_status=terminal_status,
            nodes=nodes,
            edges=edges,
            prediction_row_sha256=(
                prediction_row_sha256
                if terminal_status in {"passed", "failed"}
                else None
            ),
            result_row_projection_digest=(
                result_row_projection_digest
                if terminal_status in {"passed", "failed"}
                else None
            ),
            agent_receipt=agent_receipt,
            receipts=receipts,
            recorded_at=recorded_at or _now(),
        )
        self.store.write_attempt(record)
        return record

    def finalize(self) -> LocalEvidenceManifestV1:
        return self.store.finalize()

    def publication_receipt(self) -> LocalEvidencePublicationReceiptV1:
        return self.store.read_publication_receipt()


def build_local_evidence_run_plan(
    *,
    run_id: str,
    run_snapshot_sha256: str,
    evaluation_asset_lock_sha256: str,
    attempts: Sequence[LocalAttemptPlanV1],
    destination: LocalEvidenceDestinationV1 | None = None,
    require_run_conformance: bool = True,
) -> LocalEvidenceRunPlanV1:
    return LocalEvidenceRunPlanV1(
        run_id=run_id,
        destination=destination or LocalEvidenceDestinationV1(),
        run_snapshot_sha256=run_snapshot_sha256,
        evaluation_asset_lock_sha256=evaluation_asset_lock_sha256,
        attempts=tuple(sorted(attempts, key=lambda item: item.attempt_id)),
        require_run_conformance=require_run_conformance,
    )


def local_attempt_refs(attempt: LocalAttemptPlanV1) -> dict[EvidenceNodeKind, str]:
    root = f"fugue://local-evidence/{attempt.run_id}"
    return {
        "evaluation_root": f"{root}/evaluation/{attempt.evaluation_root_id}",
        "prediction_and_score": (
            f"{root}/attempt/{attempt.attempt_id}/prediction-and-score"
        ),
        "prediction": f"{root}/prediction/{attempt.prediction_id}",
        "agent_root": f"{root}/agent/{attempt.attempt_id}",
        "dataset": f"{root}/dataset/{attempt.dataset_id}",
    }


def local_evidence_destination_from_dict(
    raw: Mapping[str, Any],
) -> LocalEvidenceDestinationV1:
    value = _strict_mapping(
        raw,
        {
            "schema_version",
            "kind",
            "format",
            "layout_version",
            "destination_digest",
        },
        "local evidence destination",
    )
    return LocalEvidenceDestinationV1(
        schema_version=_literal_one(value.get("schema_version"), "destination"),
        kind=str(value.get("kind") or ""),  # type: ignore[arg-type]
        format=str(value.get("format") or ""),  # type: ignore[arg-type]
        layout_version=_literal_one(value.get("layout_version"), "layout"),
        destination_digest=_required_digest(
            value.get("destination_digest"), "destination digest"
        ),
    )


def local_attempt_plan_from_dict(raw: Mapping[str, Any]) -> LocalAttemptPlanV1:
    value = _strict_mapping(
        raw,
        {
            "run_id",
            "cell_id",
            "attempt_id",
            "attempt_identity",
            "prediction_id",
            "evaluation_scope_id",
            "dataset_id",
        },
        "local attempt plan",
    )
    identity = value.get("attempt_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("local attempt identity must be an object")
    return LocalAttemptPlanV1(
        run_id=str(value.get("run_id") or ""),
        cell_id=str(value.get("cell_id") or ""),
        attempt_id=str(value.get("attempt_id") or ""),
        attempt_identity=dict(identity),
        prediction_id=str(value.get("prediction_id") or ""),
        evaluation_scope_id=str(value.get("evaluation_scope_id") or ""),
        dataset_id=str(value.get("dataset_id") or ""),
    )


def local_evidence_run_plan_from_dict(
    raw: Mapping[str, Any],
) -> LocalEvidenceRunPlanV1:
    value = _strict_mapping(
        raw,
        {
            "schema_version",
            "run_id",
            "destination",
            "run_snapshot_sha256",
            "evaluation_asset_lock_sha256",
            "attempts",
            "require_run_conformance",
            "plan_digest",
        },
        "local evidence run plan",
    )
    return LocalEvidenceRunPlanV1(
        schema_version=_literal_one(value.get("schema_version"), "run plan"),
        run_id=str(value.get("run_id") or ""),
        destination=local_evidence_destination_from_dict(
            _required_mapping(value.get("destination"), "local destination")
        ),
        run_snapshot_sha256=str(value.get("run_snapshot_sha256") or ""),
        evaluation_asset_lock_sha256=str(
            value.get("evaluation_asset_lock_sha256") or ""
        ),
        attempts=tuple(
            local_attempt_plan_from_dict(
                _required_mapping(item, "local attempt plan")
            )
            for item in _required_sequence(value.get("attempts"), "attempts")
        ),
        require_run_conformance=_required_bool(
            value.get("require_run_conformance"),
            "require run conformance",
        ),
        plan_digest=_required_digest(value.get("plan_digest"), "plan digest"),
    )


def local_artifact_ref_from_dict(raw: Mapping[str, Any]) -> LocalArtifactRefV1:
    value = _strict_mapping(
        raw,
        {"path", "sha256", "size_bytes", "media_type"},
        "local artifact reference",
    )
    return LocalArtifactRefV1(
        path=str(value.get("path") or ""),
        sha256=str(value.get("sha256") or ""),
        size_bytes=_nonnegative_int(value.get("size_bytes"), "artifact size"),
        media_type=str(value.get("media_type") or ""),
    )


def agent_evidence_receipt_from_dict(
    raw: Mapping[str, Any],
) -> AgentEvidenceReceiptV1:
    value = _strict_mapping(
        raw,
        {
            "attempt_id",
            "planned_conversation_id",
            "primary_session_id",
            "child_session_ids",
            "artifacts",
            "transcript_artifact",
            "transcript_session_id",
            "correlation_verified",
            "tool_event_count",
            "tool_events_sha256",
            "response_sha256",
            "correlation_method",
            "status",
            "native_weave_call",
            "reason",
        },
        "Agent evidence receipt",
    )
    native_weave_call = value.get("native_weave_call")
    if native_weave_call is not False:
        raise ValueError("local Agent evidence native_weave_call must be false")
    correlation_verified = value.get("correlation_verified")
    if not isinstance(correlation_verified, bool):
        raise ValueError("Agent evidence correlation_verified must be boolean")
    transcript_artifact = value.get("transcript_artifact")
    return AgentEvidenceReceiptV1(
        attempt_id=str(value.get("attempt_id") or ""),
        planned_conversation_id=str(
            value.get("planned_conversation_id") or ""
        ),
        primary_session_id=(
            str(value["primary_session_id"])
            if value.get("primary_session_id") is not None
            else None
        ),
        child_session_ids=tuple(
            str(item)
            for item in _required_sequence(
                value.get("child_session_ids"), "child session ids"
            )
        ),
        artifacts=tuple(
            local_artifact_ref_from_dict(
                _required_mapping(item, "Agent evidence artifact")
            )
            for item in _required_sequence(value.get("artifacts"), "Agent artifacts")
        ),
        transcript_artifact=(
            local_artifact_ref_from_dict(
                _required_mapping(
                    transcript_artifact,
                    "Agent transcript artifact",
                )
            )
            if transcript_artifact is not None
            else None
        ),
        transcript_session_id=(
            str(value["transcript_session_id"])
            if value.get("transcript_session_id") is not None
            else None
        ),
        correlation_verified=correlation_verified,
        tool_event_count=(
            _nonnegative_int(value.get("tool_event_count"), "tool event count")
            if value.get("tool_event_count") is not None
            else None
        ),
        tool_events_sha256=(
            str(value["tool_events_sha256"])
            if value.get("tool_events_sha256") is not None
            else None
        ),
        response_sha256=(
            str(value["response_sha256"])
            if value.get("response_sha256") is not None
            else None
        ),
        correlation_method=str(value.get("correlation_method") or ""),  # type: ignore[arg-type]
        status=_evidence_status(value.get("status")),
        native_weave_call=False,
        reason=(str(value["reason"]) if value.get("reason") is not None else None),
    )


def local_evidence_node_from_dict(raw: Mapping[str, Any]) -> LocalEvidenceNodeV1:
    value = _strict_mapping(
        raw,
        {"kind", "status", "ref", "artifact", "reason"},
        "local evidence node",
    )
    artifact = value.get("artifact")
    return LocalEvidenceNodeV1(
        kind=str(value.get("kind") or ""),  # type: ignore[arg-type]
        status=_evidence_status(value.get("status")),
        ref=str(value.get("ref") or ""),
        artifact=(
            local_artifact_ref_from_dict(
                _required_mapping(artifact, "local evidence artifact")
            )
            if artifact is not None
            else None
        ),
        reason=(str(value["reason"]) if value.get("reason") is not None else None),
    )


def local_evidence_edge_from_dict(raw: Mapping[str, Any]) -> LocalEvidenceEdgeV1:
    value = _strict_mapping(
        raw,
        {"relationship", "source_ref", "target_ref", "status", "reason"},
        "local evidence edge",
    )
    status = str(value.get("status") or "")
    if status not in {"verified", "invalid"}:
        raise ValueError("unsupported local evidence edge status")
    return LocalEvidenceEdgeV1(
        relationship=str(value.get("relationship") or ""),  # type: ignore[arg-type]
        source_ref=str(value.get("source_ref") or ""),
        target_ref=str(value.get("target_ref") or ""),
        status=status,  # type: ignore[arg-type]
        reason=(str(value["reason"]) if value.get("reason") is not None else None),
    )


def local_attempt_evidence_from_dict(
    raw: Mapping[str, Any],
) -> LocalAttemptEvidenceV1:
    raw = {**raw, "result_row_projection_digest": raw.get("result_row_projection_digest")}
    value = _strict_mapping(
        raw,
        {
            "schema_version",
            "attempt",
            "terminal_status",
            "integrity_status",
            "nodes",
            "edges",
            "prediction_row_sha256",
            "result_row_projection_digest",
            "agent_receipt",
            "receipts",
            "recorded_at",
            "record_digest",
        },
        "local attempt evidence",
    )
    terminal_status = str(value.get("terminal_status") or "")
    if terminal_status not in {
        "passed",
        "failed",
        "cancelled",
        "interrupted",
        "not_applicable",
    }:
        raise ValueError("unsupported local attempt terminal status")
    record = LocalAttemptEvidenceV1(
        schema_version=_literal_one(value.get("schema_version"), "attempt evidence"),
        attempt=local_attempt_plan_from_dict(
            _required_mapping(value.get("attempt"), "local attempt")
        ),
        terminal_status=terminal_status,  # type: ignore[arg-type]
        nodes=tuple(
            local_evidence_node_from_dict(
                _required_mapping(item, "local evidence node")
            )
            for item in _required_sequence(value.get("nodes"), "evidence nodes")
        ),
        edges=tuple(
            local_evidence_edge_from_dict(
                _required_mapping(item, "local evidence edge")
            )
            for item in _required_sequence(value.get("edges"), "evidence edges")
        ),
        prediction_row_sha256=(
            str(value["prediction_row_sha256"])
            if value.get("prediction_row_sha256") is not None
            else None
        ),
        result_row_projection_digest=(
            str(value["result_row_projection_digest"])
            if value.get("result_row_projection_digest") is not None
            else None
        ),
        agent_receipt=agent_evidence_receipt_from_dict(
            _required_mapping(value.get("agent_receipt"), "Agent evidence receipt")
        ),
        receipts={
            str(name): dict(_required_mapping(receipt, f"local {name} receipt"))
            for name, receipt in _required_mapping(
                value.get("receipts"), "local attempt receipts"
            ).items()
        },
        recorded_at=str(value.get("recorded_at") or ""),
        record_digest=_required_digest(
            value.get("record_digest"), "attempt record digest"
        ),
    )
    if value.get("integrity_status") != record.integrity_status:
        raise ValueError("local attempt integrity status does not match")
    return record


def local_attempt_record_ref_from_dict(
    raw: Mapping[str, Any],
) -> LocalAttemptRecordRefV1:
    raw = {**raw, "result_row_projection_digest": raw.get("result_row_projection_digest")}
    value = _strict_mapping(
        raw,
        {
            "attempt_id",
            "path",
            "sha256",
            "record_digest",
            "integrity_status",
            "terminal_status",
            "prediction_row_sha256",
            "result_row_projection_digest",
        },
        "local attempt record reference",
    )
    integrity = str(value.get("integrity_status") or "")
    if integrity not in {"resolved", "incomplete", "invalid"}:
        raise ValueError("unsupported local attempt integrity status")
    terminal = str(value.get("terminal_status") or "")
    if terminal not in {
        "passed",
        "failed",
        "cancelled",
        "interrupted",
        "not_applicable",
    }:
        raise ValueError("unsupported local attempt terminal status")
    return LocalAttemptRecordRefV1(
        attempt_id=str(value.get("attempt_id") or ""),
        path=str(value.get("path") or ""),
        sha256=str(value.get("sha256") or ""),
        record_digest=str(value.get("record_digest") or ""),
        integrity_status=integrity,  # type: ignore[arg-type]
        terminal_status=terminal,  # type: ignore[arg-type]
        prediction_row_sha256=(
            str(value["prediction_row_sha256"])
            if value.get("prediction_row_sha256") is not None
            else None
        ),
        result_row_projection_digest=(
            str(value["result_row_projection_digest"])
            if value.get("result_row_projection_digest") is not None
            else None
        ),
    )


def local_run_conformance_ref_from_dict(
    raw: Mapping[str, Any],
) -> LocalRunConformanceRefV1:
    value = _strict_mapping(
        raw,
        {"path", "sha256", "receipt_sha256", "status", "enforced"},
        "local run conformance reference",
    )
    status = str(value.get("status") or "")
    if status not in {"passed", "failed", "unavailable", "not_applicable"}:
        raise ValueError("unsupported local run conformance status")
    return LocalRunConformanceRefV1(
        path=str(value.get("path") or ""),
        sha256=str(value.get("sha256") or ""),
        receipt_sha256=str(value.get("receipt_sha256") or ""),
        status=status,  # type: ignore[arg-type]
        enforced=_required_bool(value.get("enforced"), "run conformance enforced"),
    )


def local_evidence_manifest_from_dict(
    raw: Mapping[str, Any],
) -> LocalEvidenceManifestV1:
    raw = {
        **raw,
        "result_row_projection_set_digest": raw.get(
            "result_row_projection_set_digest"
        ),
    }
    value = _strict_mapping(
        raw,
        {
            "schema_version",
            "run_id",
            "destination",
            "run_snapshot_sha256",
            "evaluation_asset_lock_sha256",
            "plan_digest",
            "planned_attempt_ids",
            "terminal_attempt_ids",
            "attempt_records",
            "attempt_record_set_digest",
            "prediction_row_set_digest",
            "result_row_projection_set_digest",
            "run_conformance",
            "run_conformance_required",
            "status",
            "created_at",
            "manifest_digest",
        },
        "local evidence manifest",
    )
    status = str(value.get("status") or "")
    if status not in {"complete", "incomplete", "invalid"}:
        raise ValueError("unsupported local evidence manifest status")
    return LocalEvidenceManifestV1(
        schema_version=_literal_one(value.get("schema_version"), "manifest"),
        run_id=str(value.get("run_id") or ""),
        destination=local_evidence_destination_from_dict(
            _required_mapping(value.get("destination"), "local destination")
        ),
        run_snapshot_sha256=str(value.get("run_snapshot_sha256") or ""),
        evaluation_asset_lock_sha256=str(
            value.get("evaluation_asset_lock_sha256") or ""
        ),
        plan_digest=str(value.get("plan_digest") or ""),
        planned_attempt_ids=tuple(
            str(item)
            for item in _required_sequence(
                value.get("planned_attempt_ids"), "planned attempt ids"
            )
        ),
        terminal_attempt_ids=tuple(
            str(item)
            for item in _required_sequence(
                value.get("terminal_attempt_ids"), "terminal attempt ids"
            )
        ),
        attempt_records=tuple(
            local_attempt_record_ref_from_dict(
                _required_mapping(item, "attempt record reference")
            )
            for item in _required_sequence(
                value.get("attempt_records"), "attempt record references"
            )
        ),
        attempt_record_set_digest=str(
            value.get("attempt_record_set_digest") or ""
        ),
        prediction_row_set_digest=str(
            value.get("prediction_row_set_digest") or ""
        ),
        result_row_projection_set_digest=(
            str(value["result_row_projection_set_digest"])
            if value.get("result_row_projection_set_digest") is not None
            else None
        ),
        run_conformance=(
            local_run_conformance_ref_from_dict(
                _required_mapping(
                    value.get("run_conformance"),
                    "local run conformance reference",
                )
            )
            if value.get("run_conformance") is not None
            else None
        ),
        run_conformance_required=_required_bool(
            value.get("run_conformance_required"),
            "run conformance required",
        ),
        status=status,  # type: ignore[arg-type]
        created_at=str(value.get("created_at") or ""),
        manifest_digest=_required_digest(
            value.get("manifest_digest"), "manifest digest"
        ),
    )


def local_evidence_publication_receipt_from_dict(
    raw: Mapping[str, Any],
) -> LocalEvidencePublicationReceiptV1:
    value = _strict_mapping(
        raw,
        {
            "schema_version",
            "run_id",
            "destination",
            "plan_digest",
            "manifest_digest",
            "manifest_path",
            "manifest_file_sha256",
            "attempt_record_set_digest",
            "prediction_row_set_digest",
            "published_at",
            "receipt_digest",
        },
        "local evidence publication receipt",
    )
    return LocalEvidencePublicationReceiptV1(
        schema_version=_literal_one(value.get("schema_version"), "publication receipt"),
        run_id=str(value.get("run_id") or ""),
        destination=local_evidence_destination_from_dict(
            _required_mapping(value.get("destination"), "local destination")
        ),
        plan_digest=_required_digest(value.get("plan_digest"), "plan digest"),
        manifest_digest=_required_digest(
            value.get("manifest_digest"), "manifest digest"
        ),
        manifest_path=str(value.get("manifest_path") or ""),
        manifest_file_sha256=_required_digest(
            value.get("manifest_file_sha256"), "manifest file digest"
        ),
        attempt_record_set_digest=_required_digest(
            value.get("attempt_record_set_digest"), "attempt record set digest"
        ),
        prediction_row_set_digest=_required_digest(
            value.get("prediction_row_set_digest"), "prediction row set digest"
        ),
        published_at=str(value.get("published_at") or ""),
        receipt_digest=_required_digest(
            value.get("receipt_digest"), "publication receipt digest"
        ),
    )


def local_evidence_event_from_dict(
    raw: Mapping[str, Any],
) -> LocalEvidenceEventV1:
    value = _strict_mapping(
        raw,
        {
            "schema_version",
            "run_id",
            "event",
            "recorded_at",
            "attempt_id",
            "details",
            "event_digest",
        },
        "local evidence event",
    )
    event = str(value.get("event") or "")
    if event not in {
        "run_initialized",
        "attempt_opened",
        "attempt_finalized",
        "manifest_finalized",
        "publication_receipt_written",
    }:
        raise ValueError("unsupported local evidence event")
    return LocalEvidenceEventV1(
        schema_version=_literal_one(value.get("schema_version"), "event"),
        run_id=str(value.get("run_id") or ""),
        event=event,  # type: ignore[arg-type]
        recorded_at=str(value.get("recorded_at") or ""),
        attempt_id=(
            str(value["attempt_id"])
            if value.get("attempt_id") is not None
            else None
        ),
        details=dict(_required_mapping(value.get("details"), "event details")),
        event_digest=_required_digest(value.get("event_digest"), "event digest"),
    )


def _expected_edges(
    refs: Mapping[EvidenceNodeKind, str],
) -> dict[str, tuple[str, str]]:
    return {
        "evaluation_has_dataset": (
            refs["evaluation_root"],
            refs["dataset"],
        ),
        "evaluation_has_prediction_and_score": (
            refs["evaluation_root"],
            refs["prediction_and_score"],
        ),
        "prediction_and_score_has_prediction": (
            refs["prediction_and_score"],
            refs["prediction"],
        ),
        "prediction_has_agent_root": (
            refs["prediction"],
            refs["agent_root"],
        ),
    }


def _ref_kind(
    ref: str, refs: Mapping[EvidenceNodeKind, str]
) -> EvidenceNodeKind:
    matches = [kind for kind, value in refs.items() if value == ref]
    if len(matches) != 1:
        raise ValueError("local evidence ref does not identify one node")
    return matches[0]


def _verify_prediction_identity(
    row: Mapping[str, Any], attempt: LocalAttemptPlanV1
) -> None:
    checks = {
        "run_id": attempt.run_id,
        "attempt_id": attempt.attempt_id,
        "prediction_id": attempt.prediction_id,
        "candidate_id": attempt.candidate_id,
    }
    for key, expected in checks.items():
        observed = row.get(key)
        if observed is not None and str(observed) != expected:
            raise ValueError(f"prediction row {key} disagrees with its run plan")
    identity = row.get("attempt_identity")
    if identity is not None and (
        not isinstance(identity, Mapping)
        or dict(identity) != attempt.attempt_identity
    ):
        raise ValueError("prediction row attempt identity disagrees with its plan")


def _attempt_from_plan(
    plan: LocalEvidenceRunPlanV1, attempt_id: str
) -> LocalAttemptPlanV1:
    matches = [item for item in plan.attempts if item.attempt_id == attempt_id]
    if len(matches) != 1:
        raise ValueError(f"attempt is not present exactly once in run plan: {attempt_id}")
    return matches[0]


def _canonical_attempt_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {"task_id", "arm", "harness", "attempt", "candidate", "runtime"}
    unknown = sorted(set(raw) - required)
    missing = sorted(required - set(raw))
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        if missing:
            details.append("missing " + ", ".join(missing))
        raise ValueError("invalid attempt identity: " + "; ".join(details))
    attempt = raw.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt identity attempt must be a positive integer")
    return {
        "task_id": _required_text(raw.get("task_id"), "attempt task id"),
        "arm": _required_text(raw.get("arm"), "attempt arm"),
        "harness": _required_text(raw.get("harness"), "attempt harness"),
        "attempt": attempt,
        "candidate": _required_text(raw.get("candidate"), "attempt candidate"),
        "runtime": _required_text(raw.get("runtime"), "attempt runtime"),
    }


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _read_mapping(path)
        if existing != dict(value):
            raise LocalEvidenceIntegrityError(
                f"immutable local evidence changed: {path}"
            )
        return
    atomic_write_json(path, dict(value))


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalEvidenceIntegrityError(
            f"local evidence is not valid JSON: {path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise LocalEvidenceIntegrityError(
            f"local evidence is not an object: {path}"
        )
    return dict(value)


def _strict_mapping(
    raw: Mapping[str, Any], allowed: set[str], label: str
) -> dict[str, Any]:
    value = dict(raw)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")
    missing = sorted(allowed - set(value))
    if missing:
        raise ValueError(f"missing {label} field(s): {', '.join(missing)}")
    return value


def _required_mapping(raw: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    return raw


def _required_sequence(raw: Any, label: str) -> Sequence[Any]:
    if not isinstance(raw, list | tuple):
        raise ValueError(f"{label} must be a list")
    return raw


def _required_text(raw: Any, label: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"{label} is required")
    return value


def _required_bool(raw: Any, label: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{label} must be a boolean")
    return raw


def _required_digest(raw: Any, label: str) -> str:
    value = str(raw or "")
    _digest(value, label)
    return value


def _digest(value: str, label: str) -> None:
    if not _HEX_SHA256.fullmatch(str(value or "")):
        raise ValueError(f"{label} must be a SHA-256 digest")


def _run_id(value: str) -> None:
    if not _SAFE_RUN_ID.fullmatch(str(value or "")):
        raise ValueError("local evidence run id is invalid")


def _portable_path(value: str, label: str) -> Path:
    path = Path(str(value or ""))
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a portable relative path")
    return path


def _local_ref(value: str, label: str) -> str:
    ref = str(value or "")
    if not ref.startswith("fugue://local-evidence/") or any(
        part in {"", ".", ".."} for part in ref.removeprefix("fugue://").split("/")
    ):
        raise ValueError(f"{label} must be a safe local Fugue ref")
    return ref


def _timestamp(value: str, label: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _literal_one(raw: Any, label: str) -> Literal[1]:
    if raw != 1:
        raise ValueError(f"unsupported local evidence {label} schema version")
    return 1


def _nonnegative_int(raw: Any, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return raw


def _evidence_status(raw: Any) -> EvidenceStatus:
    value = str(raw or "")
    if value not in {"resolved", "missing", "invalid"}:
        raise ValueError("unsupported local evidence status")
    return value  # type: ignore[return-value]


def _public_payload(
    raw: Mapping[str, Any], *, secret_values: Sequence[str] = ()
) -> dict[str, Any]:
    value = json.loads(json.dumps(dict(raw), sort_keys=True, default=str))
    _assert_public_payload(value, secret_values)
    return value


def _assert_public_payload(
    value: Any,
    secret_values: Sequence[str] = (),
    *,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            normalized = name.lower()
            if normalized in _PRIVATE_FIELDS or _SECRET_FIELD.search(normalized):
                location = ".".join((*path, name))
                raise ValueError(
                    f"private or credential-bearing field is not allowed in "
                    f"local evidence: {location}"
                )
            _assert_public_payload(
                item,
                secret_values,
                path=(*path, name),
            )
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_public_payload(
                item,
                secret_values,
                path=(*path, str(index)),
            )
        return
    if isinstance(value, str):
        for secret in secret_values:
            if len(secret) >= 8 and secret in value:
                location = ".".join(path) or "payload"
                raise ValueError(
                    f"configured secret value is not allowed in local evidence: "
                    f"{location}"
                )
