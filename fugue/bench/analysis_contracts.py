from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from fugue.bench.candidates import stable_digest
from fugue.model_plane import (
    EvidenceDestinationV1,
    evidence_destination_from_dict,
)

DimensionRole = Literal[
    "outcome",
    "mechanism",
    "safety_gate",
    "infrastructure",
    "efficiency",
]
TaskValidityStatus = Literal[
    "valid",
    "non_discriminating",
    "drifted",
    "invalid",
    "inconclusive",
]
DriftStatus = Literal["matched", "drifted", "unavailable"]

_DIGEST_LENGTH = 64
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class EvidenceDriftCheckV1:
    status: DriftStatus
    expected_digest: str
    observed_digest: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"matched", "drifted", "unavailable"}:
            raise ValueError("unsupported evidence drift status")
        _digest(self.expected_digest, "expected drift digest")
        if self.observed_digest is not None:
            _digest(self.observed_digest, "observed drift digest")
        if self.status == "matched":
            if self.observed_digest != self.expected_digest:
                raise ValueError(
                    "matched source drift evidence requires equal digests"
                )
            if self.reason:
                raise ValueError("matched source drift evidence cannot have a reason")
        elif self.status == "drifted":
            if (
                self.observed_digest is None
                or self.observed_digest == self.expected_digest
            ):
                raise ValueError(
                    "drifted source evidence requires a different observed digest"
                )
        elif not self.reason:
            raise ValueError("unavailable source drift evidence requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class EvidenceTopologyV1:
    source_destination: EvidenceDestinationV1
    result_destination: EvidenceDestinationV1
    source_lock_digest: str
    pre_run_drift: EvidenceDriftCheckV1
    post_run_drift: EvidenceDriftCheckV1
    execution_identity: str
    schema_version: Literal[1] = 1
    topology_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported evidence topology schema version")
        _digest(self.source_lock_digest, "source lock digest")
        _digest(self.execution_identity, "evidence execution identity")
        computed = self.computed_digest()
        if self.topology_digest and self.topology_digest != computed:
            raise ValueError("evidence topology digest does not match")
        if not self.topology_digest:
            object.__setattr__(self, "topology_digest", computed)

    def computed_digest(self) -> str:
        return stable_digest(
            {
                "schema_version": self.schema_version,
                "source_destination": self.source_destination.to_dict(),
                "result_destination": self.result_destination.to_dict(),
                "source_lock_digest": self.source_lock_digest,
                "pre_run_drift": self.pre_run_drift.to_dict(),
                "post_run_drift": self.post_run_drift.to_dict(),
                "execution_identity": self.execution_identity,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_destination": self.source_destination.to_dict(),
            "result_destination": self.result_destination.to_dict(),
            "source_lock_digest": self.source_lock_digest,
            "pre_run_drift": self.pre_run_drift.to_dict(),
            "post_run_drift": self.post_run_drift.to_dict(),
            "execution_identity": self.execution_identity,
            "topology_digest": self.computed_digest(),
        }


@dataclass(frozen=True)
class TaskValidityV1:
    task_id: str
    status: TaskValidityStatus
    reasons: tuple[str, ...] = ()
    discriminating_dimensions: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.task_id, "task validity task_id")
        if self.status == "valid" and self.blockers:
            raise ValueError("valid tasks cannot have validity blockers")
        if self.status != "valid" and not (self.reasons or self.blockers):
            raise ValueError(
                "non-valid tasks require a safe reason or named blocker"
            )

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class LockDescriptorV1:
    id: str
    label: str
    digest: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.id, "lock descriptor id")
        _required_text(self.label, "lock descriptor label")
        _digest(self.digest, "lock descriptor digest", allow_sha256=True)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class AlignedDimensionV1:
    id: str
    label: str
    role: DimensionRole
    critical: bool

    def __post_init__(self) -> None:
        _required_text(self.id, "aligned dimension id")
        _required_text(self.label, "aligned dimension label")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignedArmV1:
    id: str
    label: str
    source_revision: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "aligned arm id")
        _required_text(self.label, "aligned arm label")

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class AlignedContrastV1:
    id: str
    reference_arm: str
    treatment_arms: tuple[str, ...]
    dimensions: tuple[AlignedDimensionV1, ...]

    def __post_init__(self) -> None:
        _required_text(self.id, "aligned contrast id")
        _required_text(self.reference_arm, "aligned contrast reference arm")
        if not self.treatment_arms:
            raise ValueError("aligned contrast requires a treatment arm")
        if self.reference_arm in self.treatment_arms:
            raise ValueError(
                "aligned contrast reference cannot also be a treatment arm"
            )
        if len(set(self.treatment_arms)) != len(self.treatment_arms):
            raise ValueError("aligned contrast treatment arms must be unique")
        if not self.dimensions:
            raise ValueError("aligned contrast requires declared dimensions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "reference_arm": self.reference_arm,
            "treatment_arms": list(self.treatment_arms),
            "dimensions": [item.to_dict() for item in self.dimensions],
        }


@dataclass(frozen=True)
class AlignedAttemptSetV1:
    alignment_id: str
    task_id: str
    harness: str
    attempt: int
    attempt_ids_by_arm: dict[str, str]
    task_label: str | None = None

    def __post_init__(self) -> None:
        _digest(self.alignment_id, "alignment id")
        _required_text(self.task_id, "aligned task id")
        _required_text(self.harness, "aligned harness")
        if self.attempt < 1:
            raise ValueError("aligned attempt index must be positive")
        if len(self.attempt_ids_by_arm) < 2:
            raise ValueError("aligned attempt set requires at least two arms")
        for arm, attempt_id in self.attempt_ids_by_arm.items():
            _required_text(arm, "aligned attempt arm")
            _digest(attempt_id, "aligned attempt id")

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class TaskStratifiedSummaryV1:
    task_id: str
    validity: TaskValidityStatus
    pair_counts: dict[str, int]
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.task_id, "task summary task_id")
        for key, count in self.pair_counts.items():
            _required_text(key, "task summary pair status")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("task summary pair counts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class AlignedAnalysisV1:
    study_intent: str
    reference_arm: str
    arms: tuple[AlignedArmV1, ...]
    contrasts: tuple[AlignedContrastV1, ...]
    aligned_attempts: tuple[AlignedAttemptSetV1, ...]
    task_summaries: tuple[TaskStratifiedSummaryV1, ...]
    schema_version: Literal[1] = 1
    analysis_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported aligned analysis schema version")
        _required_text(self.study_intent, "aligned analysis study intent")
        _required_text(self.reference_arm, "aligned analysis reference arm")
        arm_ids = [item.id for item in self.arms]
        if len(arm_ids) < 2 or len(set(arm_ids)) != len(arm_ids):
            raise ValueError("aligned analysis requires at least two unique arms")
        if self.reference_arm not in arm_ids:
            raise ValueError("aligned analysis reference arm is not declared")
        for contrast in self.contrasts:
            if contrast.reference_arm not in arm_ids or any(
                item not in arm_ids for item in contrast.treatment_arms
            ):
                raise ValueError("aligned contrast references an unknown arm")
        for aligned in self.aligned_attempts:
            if set(aligned.attempt_ids_by_arm) - set(arm_ids):
                raise ValueError("aligned attempt set references an unknown arm")
        computed = self.computed_digest()
        if self.analysis_digest and self.analysis_digest != computed:
            raise ValueError("aligned analysis digest does not match")
        if not self.analysis_digest:
            object.__setattr__(self, "analysis_digest", computed)

    def computed_digest(self) -> str:
        return stable_digest(
            {
                "schema_version": self.schema_version,
                "study_intent": self.study_intent,
                "reference_arm": self.reference_arm,
                "arms": [item.to_dict() for item in self.arms],
                "contrasts": [item.to_dict() for item in self.contrasts],
                "aligned_attempts": [
                    item.to_dict() for item in self.aligned_attempts
                ],
                "task_summaries": [
                    item.to_dict() for item in self.task_summaries
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_intent": self.study_intent,
            "reference_arm": self.reference_arm,
            "arms": [item.to_dict() for item in self.arms],
            "contrasts": [item.to_dict() for item in self.contrasts],
            "aligned_attempts": [
                item.to_dict() for item in self.aligned_attempts
            ],
            "task_summaries": [item.to_dict() for item in self.task_summaries],
            "analysis_digest": self.computed_digest(),
        }


@dataclass(frozen=True)
class SupersededResultV1:
    result_digest: str
    reason: str

    def __post_init__(self) -> None:
        _digest(self.result_digest, "superseded result digest")
        _required_text(self.reason, "supersession reason")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def evidence_drift_check_from_dict(value: Mapping[str, Any]) -> EvidenceDriftCheckV1:
    _reject_unknown(
        value,
        {"status", "expected_digest", "observed_digest", "reason"},
        "evidence drift check",
    )
    status = str(value.get("status") or "")
    if status not in {"matched", "drifted", "unavailable"}:
        raise ValueError("unsupported evidence drift status")
    return EvidenceDriftCheckV1(
        status=status,  # type: ignore[arg-type]
        expected_digest=str(value.get("expected_digest") or ""),
        observed_digest=_optional_text(value.get("observed_digest")),
        reason=_optional_text(value.get("reason")),
    )


def evidence_topology_from_dict(value: Mapping[str, Any]) -> EvidenceTopologyV1:
    _reject_unknown(
        value,
        {
            "schema_version",
            "source_destination",
            "result_destination",
            "source_lock_digest",
            "pre_run_drift",
            "post_run_drift",
            "execution_identity",
            "topology_digest",
        },
        "evidence topology",
    )
    return EvidenceTopologyV1(
        schema_version=int(value.get("schema_version") or 0),  # type: ignore[arg-type]
        source_destination=evidence_destination_from_dict(
            _mapping(value.get("source_destination"), "source destination")
        ),
        result_destination=evidence_destination_from_dict(
            _mapping(value.get("result_destination"), "result destination")
        ),
        source_lock_digest=str(value.get("source_lock_digest") or ""),
        pre_run_drift=evidence_drift_check_from_dict(
            _mapping(value.get("pre_run_drift"), "pre-run drift")
        ),
        post_run_drift=evidence_drift_check_from_dict(
            _mapping(value.get("post_run_drift"), "post-run drift")
        ),
        execution_identity=str(value.get("execution_identity") or ""),
        topology_digest=str(value.get("topology_digest") or ""),
    )


def task_validity_from_dict(value: Mapping[str, Any]) -> TaskValidityV1:
    _reject_unknown(
        value,
        {
            "task_id",
            "status",
            "reasons",
            "discriminating_dimensions",
            "blockers",
        },
        "task validity",
    )
    status = str(value.get("status") or "")
    if status not in {
        "valid",
        "non_discriminating",
        "drifted",
        "invalid",
        "inconclusive",
    }:
        raise ValueError("unsupported task validity status")
    return TaskValidityV1(
        task_id=str(value.get("task_id") or ""),
        status=status,  # type: ignore[arg-type]
        reasons=_text_tuple(value.get("reasons"), "task validity reasons"),
        discriminating_dimensions=_text_tuple(
            value.get("discriminating_dimensions"),
            "task validity discriminating dimensions",
        ),
        blockers=_text_tuple(value.get("blockers"), "task validity blockers"),
    )


def lock_descriptor_from_dict(value: Mapping[str, Any]) -> LockDescriptorV1:
    _reject_unknown(
        value,
        {"id", "label", "digest", "details"},
        "lock descriptor",
    )
    return LockDescriptorV1(
        id=str(value.get("id") or ""),
        label=str(value.get("label") or ""),
        digest=str(value.get("digest") or ""),
        details=dict(_mapping(value.get("details") or {}, "lock details")),
    )


def aligned_analysis_from_dict(value: Mapping[str, Any]) -> AlignedAnalysisV1:
    _reject_unknown(
        value,
        {
            "schema_version",
            "study_intent",
            "reference_arm",
            "arms",
            "contrasts",
            "aligned_attempts",
            "task_summaries",
            "analysis_digest",
        },
        "aligned analysis",
    )
    arms = tuple(_aligned_arm(item) for item in _sequence(value.get("arms"), "arms"))
    contrasts = tuple(
        _aligned_contrast(item)
        for item in _sequence(value.get("contrasts"), "contrasts")
    )
    aligned_attempts = tuple(
        _aligned_attempt_set(item)
        for item in _sequence(value.get("aligned_attempts"), "aligned attempts")
    )
    task_summaries = tuple(
        _task_summary(item)
        for item in _sequence(value.get("task_summaries"), "task summaries")
    )
    return AlignedAnalysisV1(
        schema_version=int(value.get("schema_version") or 0),  # type: ignore[arg-type]
        study_intent=str(value.get("study_intent") or ""),
        reference_arm=str(value.get("reference_arm") or ""),
        arms=arms,
        contrasts=contrasts,
        aligned_attempts=aligned_attempts,
        task_summaries=task_summaries,
        analysis_digest=str(value.get("analysis_digest") or ""),
    )


def superseded_result_from_dict(value: Mapping[str, Any]) -> SupersededResultV1:
    _reject_unknown(value, {"result_digest", "reason"}, "superseded result")
    return SupersededResultV1(
        result_digest=str(value.get("result_digest") or ""),
        reason=str(value.get("reason") or ""),
    )


def _aligned_arm(value: Mapping[str, Any]) -> AlignedArmV1:
    _reject_unknown(value, {"id", "label", "source_revision"}, "aligned arm")
    revision = value.get("source_revision")
    if revision is not None and not isinstance(revision, Mapping):
        raise ValueError("aligned arm source_revision must be an object")
    return AlignedArmV1(
        id=str(value.get("id") or ""),
        label=str(value.get("label") or ""),
        source_revision=dict(revision) if isinstance(revision, Mapping) else None,
    )


def _aligned_contrast(value: Mapping[str, Any]) -> AlignedContrastV1:
    _reject_unknown(
        value,
        {"id", "reference_arm", "treatment_arms", "dimensions"},
        "aligned contrast",
    )
    return AlignedContrastV1(
        id=str(value.get("id") or ""),
        reference_arm=str(value.get("reference_arm") or ""),
        treatment_arms=_text_tuple(
            value.get("treatment_arms"), "contrast treatment arms"
        ),
        dimensions=tuple(
            _aligned_dimension(item)
            for item in _sequence(
                value.get("dimensions"), "contrast dimensions"
            )
        ),
    )


def _aligned_dimension(value: Mapping[str, Any]) -> AlignedDimensionV1:
    _reject_unknown(
        value,
        {"id", "label", "role", "critical"},
        "aligned dimension",
    )
    role = str(value.get("role") or "")
    if role not in {
        "outcome",
        "mechanism",
        "safety_gate",
        "infrastructure",
        "efficiency",
    }:
        raise ValueError("unsupported aligned dimension role")
    critical = value.get("critical")
    if not isinstance(critical, bool):
        raise ValueError("aligned dimension critical must be boolean")
    return AlignedDimensionV1(
        id=str(value.get("id") or ""),
        label=str(value.get("label") or ""),
        role=role,  # type: ignore[arg-type]
        critical=critical,
    )


def _aligned_attempt_set(value: Mapping[str, Any]) -> AlignedAttemptSetV1:
    _reject_unknown(
        value,
        {
            "alignment_id",
            "task_id",
            "task_label",
            "harness",
            "attempt",
            "attempt_ids_by_arm",
        },
        "aligned attempt set",
    )
    raw_attempts = _mapping(
        value.get("attempt_ids_by_arm"), "attempt ids by arm"
    )
    return AlignedAttemptSetV1(
        alignment_id=str(value.get("alignment_id") or ""),
        task_id=str(value.get("task_id") or ""),
        task_label=_optional_text(value.get("task_label")),
        harness=str(value.get("harness") or ""),
        attempt=int(value.get("attempt") or 0),
        attempt_ids_by_arm={
            str(key): str(item) for key, item in raw_attempts.items()
        },
    )


def _task_summary(value: Mapping[str, Any]) -> TaskStratifiedSummaryV1:
    _reject_unknown(
        value,
        {"task_id", "validity", "pair_counts", "blockers"},
        "task summary",
    )
    validity = str(value.get("validity") or "")
    if validity not in {
        "valid",
        "non_discriminating",
        "drifted",
        "invalid",
        "inconclusive",
    }:
        raise ValueError("unsupported task summary validity")
    raw_counts = _mapping(value.get("pair_counts"), "task pair counts")
    return TaskStratifiedSummaryV1(
        task_id=str(value.get("task_id") or ""),
        validity=validity,  # type: ignore[arg-type]
        pair_counts={str(key): int(item) for key, item in raw_counts.items()},
        blockers=_text_tuple(value.get("blockers"), "task summary blockers"),
    )


def _digest(value: str, label: str, *, allow_sha256: bool = False) -> str:
    normalized = value.removeprefix("sha256:") if allow_sha256 else value
    if len(normalized) != _DIGEST_LENGTH or any(char not in _HEX for char in normalized):
        raise ValueError(f"{label} must be an exact sha256 digest")
    return value


def _required_text(value: str, label: str) -> str:
    if not str(value).strip():
        raise ValueError(f"{label} must be non-empty")
    return str(value)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or not value
    ):
        raise ValueError(f"{label} must be a non-empty array")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} entries must be objects")
        result.append(dict(item))
    return tuple(result)


def _text_tuple(value: Any, label: str) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be an array")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{label} entries must be non-empty")
    return result


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} field(s): " + ", ".join(unknown))


def _drop_empty(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item not in (None, (), {}, [])
    }
