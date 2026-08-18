from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from fugue.bench.candidates import stable_digest
from fugue.bench.local_evidence import (
    LocalEvidenceDestinationV1,
    local_evidence_destination_from_dict,
)
from fugue.model_plane import (
    EvidenceDestinationV1,
    evidence_destination_from_dict,
)

EvidenceDestination = EvidenceDestinationV1 | LocalEvidenceDestinationV1

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
AlignedEffectStatus = Literal[
    "improved",
    "regressed",
    "mixed",
    "unchanged",
    "inconclusive",
]
AlignedComputationStatus = Literal["computed", "inconclusive"]
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
                raise ValueError("matched source drift evidence requires equal digests")
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
    source_destination: EvidenceDestination
    result_destination: EvidenceDestination
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
            raise ValueError("non-valid tasks require a safe reason or named blocker")

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
        dimension_ids = [item.id for item in self.dimensions]
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("aligned contrast dimensions must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "reference_arm": self.reference_arm,
            "treatment_arms": list(self.treatment_arms),
            "dimensions": [item.to_dict() for item in self.dimensions],
        }


@dataclass(frozen=True)
class AlignedFactorV1:
    id: str
    off_level: str
    on_level: str

    def __post_init__(self) -> None:
        _required_text(self.id, "aligned factor id")
        _required_text(self.off_level, "aligned factor off level")
        _required_text(self.on_level, "aligned factor on level")
        if self.off_level == self.on_level:
            raise ValueError("aligned factor levels must differ")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AlignedInteractionV1:
    """One declared 2x2 interaction over four real arm identities."""

    id: str
    factors: tuple[AlignedFactorV1, AlignedFactorV1]
    cell_arms: dict[str, str]
    dimensions: tuple[AlignedDimensionV1, ...]

    def __post_init__(self) -> None:
        _required_text(self.id, "aligned interaction id")
        if len({item.id for item in self.factors}) != 2:
            raise ValueError("aligned interaction requires two unique factors")
        if set(self.cell_arms) != {"00", "10", "01", "11"}:
            raise ValueError(
                "aligned interaction cells must be exactly 00, 10, 01, and 11"
            )
        if len(set(self.cell_arms.values())) != 4:
            raise ValueError(
                "aligned interaction cells must reference four unique arms"
            )
        for arm in self.cell_arms.values():
            _required_text(arm, "aligned interaction arm")
        if not self.dimensions:
            raise ValueError("aligned interaction requires declared dimensions")
        dimension_ids = [item.id for item in self.dimensions]
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("aligned interaction dimensions must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "factors": [item.to_dict() for item in self.factors],
            "cell_arms": dict(self.cell_arms),
            "dimensions": [item.to_dict() for item in self.dimensions],
        }


@dataclass(frozen=True)
class AlignedAnalysisDeclarationV1:
    """Preregistered aligned-analysis design, before result rows exist."""

    study_intent: str
    reference_arm: str
    arms: tuple[AlignedArmV1, ...]
    contrasts: tuple[AlignedContrastV1, ...]
    alignment_coordinates: tuple[str, ...]
    interactions: tuple[AlignedInteractionV1, ...] = ()
    schema_version: Literal[1] = 1
    declaration_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported aligned analysis declaration schema version")
        _required_text(self.study_intent, "aligned analysis declaration study intent")
        _required_text(self.reference_arm, "aligned analysis declaration reference arm")
        arm_ids = [item.id for item in self.arms]
        if len(arm_ids) < 2 or len(set(arm_ids)) != len(arm_ids):
            raise ValueError(
                "aligned analysis declaration requires at least two unique arms"
            )
        if self.reference_arm not in arm_ids:
            raise ValueError(
                "aligned analysis declaration reference arm is not declared"
            )
        if not self.alignment_coordinates or len(
            set(self.alignment_coordinates)
        ) != len(self.alignment_coordinates):
            raise ValueError(
                "aligned analysis declaration requires unique alignment coordinates"
            )
        for coordinate in self.alignment_coordinates:
            _required_text(coordinate, "aligned analysis coordinate")
        contrast_ids: set[str] = set()
        for contrast in self.contrasts:
            if contrast.id in contrast_ids:
                raise ValueError(
                    "aligned analysis declaration contrast ids must be unique"
                )
            contrast_ids.add(contrast.id)
            if contrast.reference_arm != self.reference_arm:
                raise ValueError(
                    "aligned analysis declaration contrasts must use the "
                    "declared reference arm"
                )
            if any(item not in arm_ids for item in contrast.treatment_arms):
                raise ValueError(
                    "aligned analysis declaration contrast references an unknown arm"
                )
        if not self.contrasts:
            raise ValueError(
                "aligned analysis declaration requires at least one contrast"
            )
        interaction_ids: set[str] = set()
        for interaction in self.interactions:
            if interaction.id in interaction_ids:
                raise ValueError(
                    "aligned analysis declaration interaction ids must be unique"
                )
            interaction_ids.add(interaction.id)
            if any(item not in arm_ids for item in interaction.cell_arms.values()):
                raise ValueError(
                    "aligned analysis interaction references an unknown arm"
                )
        computed = self.computed_digest()
        if self.declaration_digest and self.declaration_digest != computed:
            raise ValueError("aligned analysis declaration digest does not match")
        if not self.declaration_digest:
            object.__setattr__(self, "declaration_digest", computed)

    def computed_digest(self) -> str:
        return stable_digest(
            {
                "schema_version": self.schema_version,
                "study_intent": self.study_intent,
                "reference_arm": self.reference_arm,
                "arms": [item.to_dict() for item in self.arms],
                "contrasts": [item.to_dict() for item in self.contrasts],
                "alignment_coordinates": list(self.alignment_coordinates),
                "interactions": [item.to_dict() for item in self.interactions],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_intent": self.study_intent,
            "reference_arm": self.reference_arm,
            "arms": [item.to_dict() for item in self.arms],
            "contrasts": [item.to_dict() for item in self.contrasts],
            "alignment_coordinates": list(self.alignment_coordinates),
            "interactions": [item.to_dict() for item in self.interactions],
            "declaration_digest": self.computed_digest(),
        }


@dataclass(frozen=True)
class AlignedAttemptSetV1:
    alignment_id: str
    task_id: str
    attempt: int
    attempt_ids_by_arm: dict[str, str]
    task_label: str | None = None
    harness: str | None = None
    alignment_coordinates: dict[str, str] = field(default_factory=dict)
    dimension_values_by_arm: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _digest(self.alignment_id, "alignment id")
        _required_text(self.task_id, "aligned task id")
        if self.harness is not None:
            _required_text(self.harness, "aligned harness")
        if self.harness is None and not self.alignment_coordinates:
            raise ValueError(
                "aligned attempt set requires a harness or explicit coordinates"
            )
        for key, value in self.alignment_coordinates.items():
            _required_text(key, "aligned attempt coordinate")
            _required_text(value, "aligned attempt coordinate value")
        if self.attempt < 1:
            raise ValueError("aligned attempt index must be positive")
        if len(self.attempt_ids_by_arm) < 2:
            raise ValueError("aligned attempt set requires at least two arms")
        for arm, attempt_id in self.attempt_ids_by_arm.items():
            _required_text(arm, "aligned attempt arm")
            _digest(attempt_id, "aligned attempt id")
        if self.dimension_values_by_arm:
            if set(self.dimension_values_by_arm) != set(self.attempt_ids_by_arm):
                raise ValueError(
                    "aligned dimension values must cover the same arms as "
                    "attempt identities"
                )
            for arm, dimensions in self.dimension_values_by_arm.items():
                _required_text(arm, "aligned dimension-value arm")
                for dimension, value in dimensions.items():
                    _required_text(dimension, "aligned dimension-value dimension")
                    if not isinstance(value, int | float) or isinstance(value, bool):
                        raise ValueError("aligned dimension values must be numeric")

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
class AlignedTaskEffectV1:
    task_id: str
    aligned_sets: int
    reference_mean: float | None
    treatment_mean: float | None
    estimate: float | None
    classification: AlignedEffectStatus

    def __post_init__(self) -> None:
        _required_text(self.task_id, "aligned task effect task_id")
        if self.aligned_sets < 0:
            raise ValueError("aligned task effect set count must be non-negative")
        _finite_optional(self.reference_mean, "aligned reference mean")
        _finite_optional(self.treatment_mean, "aligned treatment mean")
        _finite_optional(self.estimate, "aligned task effect estimate")
        if self.classification == "inconclusive":
            if self.aligned_sets or any(
                value is not None
                for value in (
                    self.reference_mean,
                    self.treatment_mean,
                    self.estimate,
                )
            ):
                raise ValueError(
                    "inconclusive aligned task effects cannot claim estimates"
                )
        elif (
            self.aligned_sets < 1
            or self.reference_mean is None
            or self.treatment_mean is None
            or self.estimate is None
        ):
            raise ValueError("computed aligned task effects require sets and estimates")

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class AlignedContrastResultV1:
    contrast_id: str
    reference_arm: str
    treatment_arm: str
    dimension_id: str
    role: DimensionRole
    critical: bool
    aligned_sets: int
    reference_mean: float | None
    treatment_mean: float | None
    estimate: float | None
    classification: AlignedEffectStatus
    task_effects: tuple[AlignedTaskEffectV1, ...]

    def __post_init__(self) -> None:
        _required_text(self.contrast_id, "aligned contrast result id")
        _required_text(self.reference_arm, "aligned contrast result reference arm")
        _required_text(self.treatment_arm, "aligned contrast result treatment arm")
        _required_text(self.dimension_id, "aligned contrast dimension")
        if self.reference_arm == self.treatment_arm:
            raise ValueError("aligned contrast result requires distinct arms")
        if self.aligned_sets < 0:
            raise ValueError("aligned contrast result set count must be non-negative")
        _finite_optional(self.reference_mean, "aligned reference mean")
        _finite_optional(self.treatment_mean, "aligned treatment mean")
        _finite_optional(self.estimate, "aligned contrast estimate")
        if not self.task_effects:
            raise ValueError("aligned contrast result requires task-stratified effects")
        if self.classification == "inconclusive":
            if self.aligned_sets or any(
                value is not None
                for value in (
                    self.reference_mean,
                    self.treatment_mean,
                    self.estimate,
                )
            ):
                raise ValueError(
                    "inconclusive aligned contrasts cannot claim estimates"
                )
        elif (
            self.aligned_sets < 1
            or self.reference_mean is None
            or self.treatment_mean is None
            or self.estimate is None
        ):
            raise ValueError("computed aligned contrasts require sets and estimates")

    def to_dict(self) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in asdict(self).items()
                if key != "task_effects"
            },
            "task_effects": [item.to_dict() for item in self.task_effects],
        }


@dataclass(frozen=True)
class AlignedTaskInteractionEffectV1:
    task_id: str
    aligned_sets: int
    cell_means: dict[str, float]
    difference_in_differences: float | None
    status: AlignedComputationStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.task_id, "aligned interaction task_id")
        if self.aligned_sets < 0:
            raise ValueError("aligned interaction task set count must be non-negative")
        _interaction_result_values(
            self.status,
            self.aligned_sets,
            self.cell_means,
            self.difference_in_differences,
            self.reason,
            label="aligned task interaction",
        )

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class AlignedInteractionDimensionResultV1:
    dimension_id: str
    role: DimensionRole
    critical: bool
    aligned_sets: int
    cell_means: dict[str, float]
    difference_in_differences: float | None
    status: AlignedComputationStatus
    task_effects: tuple[AlignedTaskInteractionEffectV1, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.dimension_id, "aligned interaction result dimension")
        if self.aligned_sets < 0:
            raise ValueError(
                "aligned interaction result set count must be non-negative"
            )
        _interaction_result_values(
            self.status,
            self.aligned_sets,
            self.cell_means,
            self.difference_in_differences,
            self.reason,
            label="aligned interaction result",
        )
        if not self.task_effects:
            raise ValueError(
                "aligned interaction result requires task-stratified effects"
            )

    def to_dict(self) -> dict[str, Any]:
        value = {
            **{
                key: item for key, item in asdict(self).items() if key != "task_effects"
            },
            "task_effects": [item.to_dict() for item in self.task_effects],
        }
        return _drop_empty(value)


@dataclass(frozen=True)
class AlignedInteractionResultV1:
    interaction_id: str
    factors: tuple[AlignedFactorV1, AlignedFactorV1]
    cell_arms: dict[str, str]
    dimensions: tuple[AlignedInteractionDimensionResultV1, ...]

    def __post_init__(self) -> None:
        _required_text(self.interaction_id, "aligned interaction result id")
        if len({item.id for item in self.factors}) != 2:
            raise ValueError("aligned interaction result requires two unique factors")
        if set(self.cell_arms) != {"00", "10", "01", "11"}:
            raise ValueError(
                "aligned interaction result cells must be 00, 10, 01, and 11"
            )
        if len(set(self.cell_arms.values())) != 4:
            raise ValueError(
                "aligned interaction result cells must reference four arms"
            )
        if not self.dimensions:
            raise ValueError("aligned interaction result requires dimension results")
        dimension_ids = [item.dimension_id for item in self.dimensions]
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ValueError("aligned interaction result dimensions must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "factors": [item.to_dict() for item in self.factors],
            "cell_arms": dict(self.cell_arms),
            "dimensions": [item.to_dict() for item in self.dimensions],
        }


@dataclass(frozen=True)
class AlignedAnalysisV1:
    study_intent: str
    reference_arm: str
    arms: tuple[AlignedArmV1, ...]
    contrasts: tuple[AlignedContrastV1, ...]
    aligned_attempts: tuple[AlignedAttemptSetV1, ...]
    task_summaries: tuple[TaskStratifiedSummaryV1, ...]
    declaration_digest: str | None = None
    alignment_coordinates: tuple[str, ...] = ()
    contrast_results: tuple[AlignedContrastResultV1, ...] = ()
    interactions: tuple[AlignedInteractionV1, ...] = ()
    interaction_results: tuple[AlignedInteractionResultV1, ...] = ()
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
        # Historical V1 artifacts may contain contrast-specific aligned sets.
        # New governed materialization enforces complete grids before creating
        # this contract; the persisted V1 reader remains backward compatible.
        for aligned in self.aligned_attempts:
            if set(aligned.attempt_ids_by_arm) - set(arm_ids):
                raise ValueError("aligned attempt set references an unknown arm")
        if self.declaration_digest is not None:
            _digest(
                self.declaration_digest,
                "aligned analysis declaration digest",
            )
        if len(set(self.alignment_coordinates)) != len(self.alignment_coordinates):
            raise ValueError("aligned analysis coordinates must be unique")
        for coordinate in self.alignment_coordinates:
            _required_text(coordinate, "aligned analysis coordinate")
        _validate_aligned_declaration_binding(self)
        _validate_aligned_contrast_results(self)
        _validate_aligned_interaction_results(self, set(arm_ids))
        computed = self.computed_digest()
        if self.analysis_digest and self.analysis_digest != computed:
            raise ValueError("aligned analysis digest does not match")
        if not self.analysis_digest:
            object.__setattr__(self, "analysis_digest", computed)

    def computed_digest(self) -> str:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "study_intent": self.study_intent,
            "reference_arm": self.reference_arm,
            "arms": [item.to_dict() for item in self.arms],
            "contrasts": [item.to_dict() for item in self.contrasts],
            "aligned_attempts": [item.to_dict() for item in self.aligned_attempts],
            "task_summaries": [item.to_dict() for item in self.task_summaries],
        }
        # Optional governed-result fields are omitted when absent so an
        # existing V1 digest remains valid byte-for-byte.
        if self.declaration_digest is not None:
            value["declaration_digest"] = self.declaration_digest
        if self.alignment_coordinates:
            value["alignment_coordinates"] = list(self.alignment_coordinates)
        if self.contrast_results:
            value["contrast_results"] = [
                item.to_dict() for item in self.contrast_results
            ]
        if self.interactions:
            value["interactions"] = [item.to_dict() for item in self.interactions]
        if self.interaction_results:
            value["interaction_results"] = [
                item.to_dict() for item in self.interaction_results
            ]
        return stable_digest(value)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "study_intent": self.study_intent,
            "reference_arm": self.reference_arm,
            "arms": [item.to_dict() for item in self.arms],
            "contrasts": [item.to_dict() for item in self.contrasts],
            "aligned_attempts": [item.to_dict() for item in self.aligned_attempts],
            "task_summaries": [item.to_dict() for item in self.task_summaries],
            "analysis_digest": self.computed_digest(),
        }
        if self.declaration_digest is not None:
            value["declaration_digest"] = self.declaration_digest
        if self.alignment_coordinates:
            value["alignment_coordinates"] = list(self.alignment_coordinates)
        if self.contrast_results:
            value["contrast_results"] = [
                item.to_dict() for item in self.contrast_results
            ]
        if self.interactions:
            value["interactions"] = [item.to_dict() for item in self.interactions]
        if self.interaction_results:
            value["interaction_results"] = [
                item.to_dict() for item in self.interaction_results
            ]
        return value


def _validate_aligned_declaration_binding(result: AlignedAnalysisV1) -> None:
    if result.declaration_digest is None:
        return
    declaration = AlignedAnalysisDeclarationV1(
        study_intent=result.study_intent,
        reference_arm=result.reference_arm,
        arms=result.arms,
        contrasts=result.contrasts,
        alignment_coordinates=result.alignment_coordinates,
        interactions=result.interactions,
    )
    if declaration.declaration_digest != result.declaration_digest:
        raise ValueError(
            "aligned analysis content disagrees with its declaration digest"
        )


def _validate_aligned_contrast_results(result: AlignedAnalysisV1) -> None:
    declared_contrasts = {item.id: item for item in result.contrasts}
    expected_coordinates = {
        (contrast.id, treatment_arm, dimension.id)
        for contrast in result.contrasts
        for treatment_arm in contrast.treatment_arms
        for dimension in contrast.dimensions
    }
    observed_coordinates = [
        (item.contrast_id, item.treatment_arm, item.dimension_id)
        for item in result.contrast_results
    ]
    if len(set(observed_coordinates)) != len(observed_coordinates):
        raise ValueError("aligned contrast result coordinates must be unique")
    if result.declaration_digest is not None and (
        set(observed_coordinates) != expected_coordinates
    ):
        raise ValueError(
            "aligned contrast results must exactly cover declared coordinates"
        )
    for item in result.contrast_results:
        declared = declared_contrasts.get(item.contrast_id)
        declared_dimension = (
            next(
                (
                    dimension
                    for dimension in declared.dimensions
                    if dimension.id == item.dimension_id
                ),
                None,
            )
            if declared is not None
            else None
        )
        if (
            declared is None
            or item.reference_arm != declared.reference_arm
            or item.treatment_arm not in declared.treatment_arms
            or declared_dimension is None
            or item.role != declared_dimension.role
            or item.critical != declared_dimension.critical
        ):
            raise ValueError("aligned contrast result disagrees with its declaration")


def _validate_aligned_interaction_results(
    result: AlignedAnalysisV1,
    arm_ids: set[str],
) -> None:
    declared_interactions = {item.id: item for item in result.interactions}
    if len(declared_interactions) != len(result.interactions):
        raise ValueError("aligned interaction declaration ids must be unique")
    interaction_ids: set[str] = set()
    for item in result.interaction_results:
        if item.interaction_id in interaction_ids:
            raise ValueError("aligned interaction result ids must be unique")
        interaction_ids.add(item.interaction_id)
        if any(arm not in arm_ids for arm in item.cell_arms.values()):
            raise ValueError("aligned interaction result references an unknown arm")
        declared = declared_interactions.get(item.interaction_id)
        declared_dimensions = (
            {dimension.id: dimension for dimension in declared.dimensions}
            if declared is not None
            else {}
        )
        observed_dimensions = {
            dimension.dimension_id: dimension for dimension in item.dimensions
        }
        if (
            declared is None
            or item.factors != declared.factors
            or item.cell_arms != declared.cell_arms
            or set(observed_dimensions) != set(declared_dimensions)
            or any(
                observed_dimensions[dimension_id].role != declared_dimension.role
                or observed_dimensions[dimension_id].critical
                != declared_dimension.critical
                for dimension_id, declared_dimension in declared_dimensions.items()
            )
        ):
            raise ValueError(
                "aligned interaction result disagrees with its declaration"
            )
    if interaction_ids != set(declared_interactions):
        raise ValueError(
            "aligned interaction results must exactly cover declared interactions"
        )


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
        source_destination=_evidence_destination_from_dict(
            _mapping(value.get("source_destination"), "source destination")
        ),
        result_destination=_evidence_destination_from_dict(
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


def _evidence_destination_from_dict(
    value: Mapping[str, Any],
) -> EvidenceDestination:
    if value.get("kind") == "local":
        return local_evidence_destination_from_dict(value)
    return evidence_destination_from_dict(value)


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
            "declaration_digest",
            "alignment_coordinates",
            "contrast_results",
            "interactions",
            "interaction_results",
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
        declaration_digest=_optional_text(value.get("declaration_digest")),
        alignment_coordinates=_text_tuple(
            value.get("alignment_coordinates"),
            "aligned analysis coordinates",
        ),
        contrast_results=tuple(
            _aligned_contrast_result(item)
            for item in _optional_sequence(
                value.get("contrast_results"),
                "aligned contrast results",
            )
        ),
        interactions=tuple(
            _aligned_interaction(item)
            for item in _optional_sequence(
                value.get("interactions"),
                "aligned interactions",
            )
        ),
        interaction_results=tuple(
            _aligned_interaction_result(item)
            for item in _optional_sequence(
                value.get("interaction_results"),
                "aligned interaction results",
            )
        ),
        analysis_digest=str(value.get("analysis_digest") or ""),
    )


def aligned_analysis_declaration_from_dict(
    value: Mapping[str, Any],
) -> AlignedAnalysisDeclarationV1:
    _reject_unknown(
        value,
        {
            "schema_version",
            "study_intent",
            "reference_arm",
            "arms",
            "contrasts",
            "alignment_coordinates",
            "interactions",
            "declaration_digest",
        },
        "aligned analysis declaration",
    )
    return AlignedAnalysisDeclarationV1(
        schema_version=int(value.get("schema_version") or 0),  # type: ignore[arg-type]
        study_intent=str(value.get("study_intent") or ""),
        reference_arm=str(value.get("reference_arm") or ""),
        arms=tuple(
            _aligned_arm(item)
            for item in _sequence(value.get("arms"), "declaration arms")
        ),
        contrasts=tuple(
            _aligned_contrast(item)
            for item in _sequence(value.get("contrasts"), "declaration contrasts")
        ),
        alignment_coordinates=_text_tuple(
            value.get("alignment_coordinates"),
            "aligned analysis declaration coordinates",
        ),
        interactions=tuple(
            _aligned_interaction(item)
            for item in _optional_sequence(
                value.get("interactions"), "declaration interactions"
            )
        ),
        declaration_digest=str(value.get("declaration_digest") or ""),
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
            for item in _sequence(value.get("dimensions"), "contrast dimensions")
        ),
    )


def _aligned_dimension(value: Mapping[str, Any]) -> AlignedDimensionV1:
    _reject_unknown(
        value,
        {"id", "label", "role", "critical"},
        "aligned dimension",
    )
    role = _dimension_role(value.get("role"))
    critical = value.get("critical")
    if not isinstance(critical, bool):
        raise ValueError("aligned dimension critical must be boolean")
    return AlignedDimensionV1(
        id=str(value.get("id") or ""),
        label=str(value.get("label") or ""),
        role=role,
        critical=critical,
    )


def _aligned_factor(value: Mapping[str, Any]) -> AlignedFactorV1:
    _reject_unknown(
        value,
        {"id", "off_level", "on_level"},
        "aligned factor",
    )
    return AlignedFactorV1(
        id=str(value.get("id") or ""),
        off_level=str(value.get("off_level") or ""),
        on_level=str(value.get("on_level") or ""),
    )


def _aligned_interaction(value: Mapping[str, Any]) -> AlignedInteractionV1:
    _reject_unknown(
        value,
        {"id", "factors", "cell_arms", "dimensions"},
        "aligned interaction",
    )
    raw_factors = _sequence(value.get("factors"), "interaction factors")
    if len(raw_factors) != 2:
        raise ValueError("aligned interaction requires exactly two factors")
    return AlignedInteractionV1(
        id=str(value.get("id") or ""),
        factors=(
            _aligned_factor(raw_factors[0]),
            _aligned_factor(raw_factors[1]),
        ),
        cell_arms={
            str(key): str(item)
            for key, item in _mapping(
                value.get("cell_arms"), "interaction cell arms"
            ).items()
        },
        dimensions=tuple(
            _aligned_dimension(item)
            for item in _sequence(value.get("dimensions"), "interaction dimensions")
        ),
    )


def _aligned_attempt_set(value: Mapping[str, Any]) -> AlignedAttemptSetV1:
    _reject_unknown(
        value,
        {
            "alignment_id",
            "task_id",
            "task_label",
            "harness",
            "alignment_coordinates",
            "attempt",
            "attempt_ids_by_arm",
            "dimension_values_by_arm",
        },
        "aligned attempt set",
    )
    raw_attempts = _mapping(value.get("attempt_ids_by_arm"), "attempt ids by arm")
    return AlignedAttemptSetV1(
        alignment_id=str(value.get("alignment_id") or ""),
        task_id=str(value.get("task_id") or ""),
        task_label=_optional_text(value.get("task_label")),
        harness=_optional_text(value.get("harness")),
        alignment_coordinates={
            str(key): str(item)
            for key, item in _mapping(
                value.get("alignment_coordinates") or {},
                "aligned attempt coordinates",
            ).items()
        },
        attempt=int(value.get("attempt") or 0),
        attempt_ids_by_arm={str(key): str(item) for key, item in raw_attempts.items()},
        dimension_values_by_arm={
            str(arm): {
                str(dimension): float(number)
                for dimension, number in _mapping(
                    dimensions,
                    "aligned dimension values",
                ).items()
            }
            for arm, dimensions in _mapping(
                value.get("dimension_values_by_arm") or {},
                "aligned dimension values by arm",
            ).items()
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


def _aligned_task_effect(value: Mapping[str, Any]) -> AlignedTaskEffectV1:
    _reject_unknown(
        value,
        {
            "task_id",
            "aligned_sets",
            "reference_mean",
            "treatment_mean",
            "estimate",
            "classification",
        },
        "aligned task effect",
    )
    return AlignedTaskEffectV1(
        task_id=str(value.get("task_id") or ""),
        aligned_sets=int(value.get("aligned_sets") or 0),
        reference_mean=_optional_number(value.get("reference_mean")),
        treatment_mean=_optional_number(value.get("treatment_mean")),
        estimate=_optional_number(value.get("estimate")),
        classification=_effect_status(value.get("classification")),
    )


def _aligned_contrast_result(
    value: Mapping[str, Any],
) -> AlignedContrastResultV1:
    _reject_unknown(
        value,
        {
            "contrast_id",
            "reference_arm",
            "treatment_arm",
            "dimension_id",
            "role",
            "critical",
            "aligned_sets",
            "reference_mean",
            "treatment_mean",
            "estimate",
            "classification",
            "task_effects",
        },
        "aligned contrast result",
    )
    role = _dimension_role(value.get("role"))
    critical = value.get("critical")
    if not isinstance(critical, bool):
        raise ValueError("aligned contrast result critical must be boolean")
    return AlignedContrastResultV1(
        contrast_id=str(value.get("contrast_id") or ""),
        reference_arm=str(value.get("reference_arm") or ""),
        treatment_arm=str(value.get("treatment_arm") or ""),
        dimension_id=str(value.get("dimension_id") or ""),
        role=role,
        critical=critical,
        aligned_sets=int(value.get("aligned_sets") or 0),
        reference_mean=_optional_number(value.get("reference_mean")),
        treatment_mean=_optional_number(value.get("treatment_mean")),
        estimate=_optional_number(value.get("estimate")),
        classification=_effect_status(value.get("classification")),
        task_effects=tuple(
            _aligned_task_effect(item)
            for item in _sequence(
                value.get("task_effects"),
                "aligned contrast task effects",
            )
        ),
    )


def _aligned_task_interaction_effect(
    value: Mapping[str, Any],
) -> AlignedTaskInteractionEffectV1:
    _reject_unknown(
        value,
        {
            "task_id",
            "aligned_sets",
            "cell_means",
            "difference_in_differences",
            "status",
            "reason",
        },
        "aligned task interaction effect",
    )
    return AlignedTaskInteractionEffectV1(
        task_id=str(value.get("task_id") or ""),
        aligned_sets=int(value.get("aligned_sets") or 0),
        cell_means={
            str(key): float(item)
            for key, item in _mapping(
                value.get("cell_means") or {},
                "aligned task interaction cell means",
            ).items()
        },
        difference_in_differences=_optional_number(
            value.get("difference_in_differences")
        ),
        status=_computation_status(value.get("status")),
        reason=_optional_text(value.get("reason")),
    )


def _aligned_interaction_dimension_result(
    value: Mapping[str, Any],
) -> AlignedInteractionDimensionResultV1:
    _reject_unknown(
        value,
        {
            "dimension_id",
            "role",
            "critical",
            "aligned_sets",
            "cell_means",
            "difference_in_differences",
            "status",
            "task_effects",
            "reason",
        },
        "aligned interaction dimension result",
    )
    critical = value.get("critical")
    if not isinstance(critical, bool):
        raise ValueError("aligned interaction dimension critical must be boolean")
    return AlignedInteractionDimensionResultV1(
        dimension_id=str(value.get("dimension_id") or ""),
        role=_dimension_role(value.get("role")),
        critical=critical,
        aligned_sets=int(value.get("aligned_sets") or 0),
        cell_means={
            str(key): float(item)
            for key, item in _mapping(
                value.get("cell_means") or {},
                "aligned interaction cell means",
            ).items()
        },
        difference_in_differences=_optional_number(
            value.get("difference_in_differences")
        ),
        status=_computation_status(value.get("status")),
        task_effects=tuple(
            _aligned_task_interaction_effect(item)
            for item in _sequence(
                value.get("task_effects"),
                "aligned interaction task effects",
            )
        ),
        reason=_optional_text(value.get("reason")),
    )


def _aligned_interaction_result(
    value: Mapping[str, Any],
) -> AlignedInteractionResultV1:
    _reject_unknown(
        value,
        {"interaction_id", "factors", "cell_arms", "dimensions"},
        "aligned interaction result",
    )
    raw_factors = _sequence(value.get("factors"), "aligned interaction result factors")
    if len(raw_factors) != 2:
        raise ValueError("aligned interaction result requires exactly two factors")
    return AlignedInteractionResultV1(
        interaction_id=str(value.get("interaction_id") or ""),
        factors=(
            _aligned_factor(raw_factors[0]),
            _aligned_factor(raw_factors[1]),
        ),
        cell_arms={
            str(key): str(item)
            for key, item in _mapping(
                value.get("cell_arms"),
                "aligned interaction result cell arms",
            ).items()
        },
        dimensions=tuple(
            _aligned_interaction_dimension_result(item)
            for item in _sequence(
                value.get("dimensions"),
                "aligned interaction result dimensions",
            )
        ),
    )


def _digest(value: str, label: str, *, allow_sha256: bool = False) -> str:
    normalized = value.removeprefix("sha256:") if allow_sha256 else value
    if len(normalized) != _DIGEST_LENGTH or any(
        char not in _HEX for char in normalized
    ):
        raise ValueError(f"{label} must be an exact sha256 digest")
    return value


def _dimension_role(value: Any) -> DimensionRole:
    role = str(value or "")
    if role not in {
        "outcome",
        "mechanism",
        "safety_gate",
        "infrastructure",
        "efficiency",
    }:
        raise ValueError("unsupported aligned dimension role")
    return role  # type: ignore[return-value]


def _effect_status(value: Any) -> AlignedEffectStatus:
    status = str(value or "")
    if status not in {
        "improved",
        "regressed",
        "mixed",
        "unchanged",
        "inconclusive",
    }:
        raise ValueError("unsupported aligned effect status")
    return status  # type: ignore[return-value]


def _computation_status(value: Any) -> AlignedComputationStatus:
    status = str(value or "")
    if status not in {"computed", "inconclusive"}:
        raise ValueError("unsupported aligned computation status")
    return status  # type: ignore[return-value]


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("aligned numeric result must be finite")
    return float(value)


def _finite_optional(value: float | None, label: str) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _interaction_result_values(
    status: AlignedComputationStatus,
    aligned_sets: int,
    cell_means: Mapping[str, float],
    difference_in_differences: float | None,
    reason: str | None,
    *,
    label: str,
) -> None:
    if status == "computed":
        if aligned_sets < 1:
            raise ValueError(f"{label} requires at least one aligned set")
        if set(cell_means) != {"00", "10", "01", "11"}:
            raise ValueError(f"{label} requires cell means for 00, 10, 01, and 11")
        if difference_in_differences is None:
            raise ValueError(f"{label} requires a difference-in-differences estimate")
        if reason:
            raise ValueError(f"computed {label} cannot have a reason")
    else:
        if aligned_sets or cell_means or difference_in_differences is not None:
            raise ValueError(f"inconclusive {label} cannot claim computed values")
        if not reason:
            raise ValueError(f"inconclusive {label} requires a reason")
    for value in cell_means.values():
        if not math.isfinite(value):
            raise ValueError(f"{label} cell means must be finite")
    _finite_optional(
        difference_in_differences,
        f"{label} difference-in-differences",
    )


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
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} entries must be objects")
        result.append(dict(item))
    return tuple(result)


def _optional_sequence(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    if value in (None, (), []):
        return ()
    return _sequence(value, label)


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
    return {key: item for key, item in value.items() if item not in (None, (), {}, [])}
