from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx
from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.model_plane import (
    EvidenceDestinationV1,
    evidence_destination_from_dict,
)
from fugue.redaction import redact_text
from fugue.research.contracts import (
    RESEARCH_SCHEMA_VERSION,
    AttributionV1,
    JsonValue,
    ResearchError,
    attribution_from_dict,
)

ResearchLogClassification = Literal[
    "lifecycle",
    "observation",
    "decision",
    "evidence",
    "result",
    "budget",
    "limitation",
]
ResearchLogState = Literal[
    "proposed",
    "awaiting_approval",
    "preparing",
    "running",
    "evaluating",
    "completed",
    "failed",
    "cancelled",
    "paused",
]
ResearchRelationshipKind = Literal[
    "derived_from",
    "compares_to",
    "uses_baseline",
    "uses_evidence",
    "supersedes",
]

_CLASSIFICATIONS = {
    "lifecycle",
    "observation",
    "decision",
    "evidence",
    "result",
    "budget",
    "limitation",
}
_STATES = {
    "proposed",
    "awaiting_approval",
    "preparing",
    "running",
    "evaluating",
    "completed",
    "failed",
    "cancelled",
    "paused",
}
_RELATIONSHIPS = {
    "derived_from",
    "compares_to",
    "uses_baseline",
    "uses_evidence",
    "supersedes",
}
# A reconciled 32-cell V3 qualification carries five verified evidence links
# per attempt plus aligned cases and safe judge scores. Keep that canonical
# view in one signed event while retaining a strict bounded-publication limit.
RESEARCH_LOG_MAX_BYTES = 524_288
_PRIVATE_KEYS = {
    "credential",
    "credentials",
    "expected",
    "expected_answer",
    "expected_answers",
    "expected_output",
    "expected_outputs",
    "expected_path",
    "expected_paths",
    "expected_reference",
    "expected_references",
    "expected_value",
    "expected_values",
    "gold",
    "gold_path",
    "gold_paths",
    "hidden_reasoning",
    "private_criteria",
    "prompt",
    "prompt_body",
    "prompt_content",
    "prompt_messages",
    "prompt_text",
    "reasoning_body",
    "reasoning_content",
    "reasoning_text",
    "secret",
    "secrets",
    "trace_body",
}
_PRIVATE_KEY_PREFIXES = ("credential_", "gold_", "private_", "secret_")
_PRIVATE_KEY_SUFFIXES = ("_credential", "_secret")
_PUBLIC_SELECTOR_KEYS = {
    "analysis_id",
    "artifact_name",
    "artifact_version",
    "call_id",
    "commit",
    "dataset_row_id",
    "entity",
    "evaluation_id",
    "operation_id",
    "project",
    "row_id",
    "run_id",
    "trace_id",
}
_WEAVE_HOSTED_KINDS = frozenset(
    {
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_evidence_receipt",
        "dataset",
    }
)
_WEAVE_CALL_KINDS = _WEAVE_HOSTED_KINDS - {"dataset"}
_WEAVE_SLUG_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WEAVE_STUDY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_WEAVE_OBJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,511}$")
_WEAVE_CONTENT_HASH = re.compile(r"^(?:[A-Za-z0-9]{43}|[0-9a-f]{64})$")
_WANDB_REPORT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,509}={0,2}$")


@dataclass(frozen=True)
class ResearchRelationshipV1:
    kind: ResearchRelationshipKind
    target: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchEvidenceRefV1:
    system: str
    kind: str
    ref: str
    uri: str | None = None
    digest: str | None = None
    version: str | None = None
    selector: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: item
            for key, item in asdict(self).items()
            if item not in (None, "", (), [], {})
        }


@dataclass(frozen=True)
class ResearchLogEventV1:
    schema_version: int
    producer_event_id: str
    sequence: int
    timestamp: str
    source: str
    actor: AttributionV1
    research_id: str
    study_id: str | None
    classification: ResearchLogClassification
    state: ResearchLogState
    message: str
    progress: dict[str, JsonValue] = field(default_factory=dict)
    reserved_cost_usd: float | None = None
    observed_cost_usd: float | None = None
    relationships: tuple[ResearchRelationshipV1, ...] = ()
    evidence: tuple[ResearchEvidenceRefV1, ...] = ()
    summary: dict[str, JsonValue] = field(default_factory=dict)
    event_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {
            key: item
            for key, item in value.items()
            if item not in (None, "", (), [], {})
        }


@dataclass(frozen=True)
class WeavePublicationScopeEvidenceV1:
    research_id: str
    study_id: str
    schema_version: Literal[1] = RESEARCH_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeavePublicationTargetEvidenceV1:
    entity: str
    project: str
    study_scope: WeavePublicationScopeEvidenceV1
    destination: EvidenceDestinationV1
    schema_version: Literal[1] = RESEARCH_SCHEMA_VERSION

    @property
    def project_slug(self) -> str:
        return f"{self.entity}/{self.project}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entity": self.entity,
            "project": self.project,
            "study_scope": self.study_scope.to_dict(),
            "destination": self.destination.to_dict(),
        }


@dataclass(frozen=True)
class WeaveHostedEvidenceRefV1:
    attempt_id: str
    kind: Literal[
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_evidence_receipt",
        "dataset",
    ]
    object_id: str
    ref: str
    system: Literal["weave"] = "weave"
    native_agent_call: Literal[False] = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeavePublicationEvidenceV1:
    """Public, digest-bound delivery evidence for one verified Weave receipt."""

    publication_id: str
    receipt_digest: str
    target: WeavePublicationTargetEvidenceV1
    result_digest: str
    qualification_digest: str
    result_file_sha256: str
    local_manifest_digest: str
    local_manifest_file_sha256: str
    hosted_objects: tuple[WeaveHostedEvidenceRefV1, ...]
    publisher_id: str
    publisher_revision: str
    status: Literal["published"]
    published_at: str
    schema_version: Literal[1] = RESEARCH_SCHEMA_VERSION

    @property
    def attempt_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.attempt_id for item in self.hosted_objects}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "receipt_digest": self.receipt_digest,
            "target": self.target.to_dict(),
            "result_digest": self.result_digest,
            "qualification_digest": self.qualification_digest,
            "result_file_sha256": self.result_file_sha256,
            "local_manifest_digest": self.local_manifest_digest,
            "local_manifest_file_sha256": self.local_manifest_file_sha256,
            "hosted_objects": [item.to_dict() for item in self.hosted_objects],
            "publisher_id": self.publisher_id,
            "publisher_revision": self.publisher_revision,
            "status": self.status,
            "published_at": self.published_at,
        }


@dataclass(frozen=True)
class ResearchIndexReportTargetEvidenceV1:
    """The exact W&B project selected for a Research index and Report."""

    project: str
    api_base_url: str
    app_base_url: str
    destination_digest: str
    backend: Literal["wandb"] = "wandb"
    schema_version: Literal[1] = RESEARCH_SCHEMA_VERSION

    @property
    def entity(self) -> str:
        return self.project.split("/", 1)[0]

    @property
    def project_name(self) -> str:
        return self.project.split("/", 1)[1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchIndexReceiptEvidenceV1:
    """A compact projection that reconstructs one Research-index receipt."""

    publication_id: str
    index_digest: str
    index_file_sha256: str
    receipt_digest: str
    receipt_file_sha256: str
    run_url: str
    artifact_url: str
    report_url: str | None
    report_status: Literal["published", "unavailable"]
    publisher_id: str
    publisher_revision: str
    status: Literal["published"]
    published_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def receipt_dict(
        self,
        *,
        research_id: str,
        target: ResearchIndexReportTargetEvidenceV1,
    ) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("receipt_file_sha256")
        return {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "publication_id": self.publication_id,
            "research_id": research_id,
            "index_digest": self.index_digest,
            "index_file_sha256": self.index_file_sha256,
            "target": target.to_dict(),
            **{
                key: value[key]
                for key in (
                    "run_url",
                    "artifact_url",
                    "report_url",
                    "report_status",
                    "publisher_id",
                    "publisher_revision",
                    "status",
                    "published_at",
                    "receipt_digest",
                )
            },
        }


@dataclass(frozen=True)
class ResearchReportReceiptEvidenceV1:
    """A compact projection that reconstructs one reconciled Report receipt."""

    report_publication_id: str
    projection_digest: str
    index_digest: str
    index_file_sha256: str
    index_publication_id: str
    index_publication_receipt_digest: str
    index_publication_receipt_file_sha256: str
    receipt_digest: str
    receipt_file_sha256: str
    renderer_id: str
    renderer_revision: str
    report_id: str
    report_url: str
    report_api: str
    report_api_version: str
    api_stability: Literal["public_preview"]
    readback_projection_digest: str
    rendered_content_digest: str
    readback_status: Literal["reconciled"]
    publisher_id: str
    publisher_revision: str
    access_mode: Literal["project_settings"]
    share_link_action: Literal["not_requested"]
    current_pointer_status: Literal["not_managed"]
    status: Literal["published_and_reconciled"]
    published_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def receipt_dict(
        self,
        *,
        research_id: str,
        target: ResearchIndexReportTargetEvidenceV1,
    ) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("receipt_file_sha256")
        return {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "report_publication_id": self.report_publication_id,
            "research_id": research_id,
            "projection_digest": self.projection_digest,
            "index_digest": self.index_digest,
            "index_file_sha256": self.index_file_sha256,
            "index_publication_id": self.index_publication_id,
            "index_publication_receipt_digest": (
                self.index_publication_receipt_digest
            ),
            "index_publication_receipt_file_sha256": (
                self.index_publication_receipt_file_sha256
            ),
            "target": target.to_dict(),
            **{
                key: value[key]
                for key in (
                    "renderer_id",
                    "renderer_revision",
                    "report_id",
                    "report_url",
                    "report_api",
                    "report_api_version",
                    "api_stability",
                    "readback_projection_digest",
                    "rendered_content_digest",
                    "readback_status",
                    "publisher_id",
                    "publisher_revision",
                    "access_mode",
                    "share_link_action",
                    "current_pointer_status",
                    "status",
                    "published_at",
                    "receipt_digest",
                )
            },
        }


@dataclass(frozen=True)
class ResearchIndexReportStudyMembershipV1:
    """One compact Study binding from the Research index to local evidence."""

    study_id: str
    result_digest: str
    qualification_digest: str
    result_file_sha256: str
    weave_project: str
    weave_publication_id: str
    weave_receipt_digest: str
    attempt_count: int
    attempt_ids_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchIndexReportPublicationEvidenceV1:
    """Public evidence for one index receipt and one reconciled Report receipt."""

    research_id: str
    target: ResearchIndexReportTargetEvidenceV1
    index: ResearchIndexReceiptEvidenceV1
    report: ResearchReportReceiptEvidenceV1
    studies: tuple[ResearchIndexReportStudyMembershipV1, ...]
    schema_version: Literal[1] = RESEARCH_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "research_id": self.research_id,
            "target": self.target.to_dict(),
            "index": self.index.to_dict(),
            "report": self.report.to_dict(),
            "studies": [item.to_dict() for item in self.studies],
        }


@dataclass(frozen=True)
class ExperimentViewPageV1:
    """One bounded page of canonical terminal V3 attempt evidence."""

    schema_version: Literal[1]
    page_set_id: str
    projection_digest: str
    page_index: int
    page_count: int
    attempt_count: int
    paired_case_count: int
    attempts: tuple[dict[str, Any], ...]
    paired_cases: tuple[dict[str, Any], ...]
    page_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "attempts": [dict(item) for item in self.attempts],
            "paired_cases": [dict(item) for item in self.paired_cases],
        }


@dataclass(frozen=True)
class ExperimentViewManifestV1:
    """Digest-bound terminal header for a paged V3 evaluation projection."""

    schema_version: Literal[1]
    page_set_id: str
    projection_digest: str
    page_count: int
    attempt_count: int
    paired_case_count: int
    projection: dict[str, Any]
    manifest_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def experiment_view_page_set_id(
    *,
    projection_digest: str,
    page_count: int,
    attempt_count: int,
    paired_case_count: int,
) -> str:
    return stable_digest(
        {
            "schema_version": 1,
            "projection_digest": projection_digest,
            "page_count": page_count,
            "attempt_count": attempt_count,
            "paired_case_count": paired_case_count,
        }
    )


def experiment_view_page_from_dict(raw: Mapping[str, Any]) -> ExperimentViewPageV1:
    fields = {item.name for item in ExperimentViewPageV1.__dataclass_fields__.values()}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError("experiment view page has unknown fields")
    if raw.get("schema_version") != RESEARCH_SCHEMA_VERSION:
        raise ValueError("unsupported experiment view page schema")
    page_count = _positive_int(raw.get("page_count"), "page_count")
    page_index = _non_negative_int(raw.get("page_index"), "page_index")
    if page_index >= page_count:
        raise ValueError("experiment page index must be smaller than page count")
    attempt_count = _non_negative_int(raw.get("attempt_count"), "attempt_count")
    paired_case_count = _non_negative_int(
        raw.get("paired_case_count"), "paired_case_count"
    )
    from fugue.research.experiment_views import (
        _canonical_paired_case_v3,
        _optional_canonical_attempt_v3,
    )

    attempts: list[dict[str, Any]] = []
    for item in _sequence(raw.get("attempts"), "attempts"):
        attempt = _optional_canonical_attempt_v3(item)
        if attempt is None:
            raise ValueError("experiment page attempt must be an object")
        attempts.append(attempt)
    paired_cases = tuple(
        _canonical_paired_case_v3(item)
        for item in _sequence(raw.get("paired_cases"), "paired_cases")
    )
    if not attempts and not paired_cases:
        raise ValueError("experiment page must contain attempts or paired cases")
    if len(attempts) > attempt_count or len(paired_cases) > paired_case_count:
        raise ValueError("experiment page exceeds its declared result census")
    attempt_by_id = {str(item.get("attempt_id") or ""): item for item in attempts}
    if len(attempt_by_id) != len(attempts) or "" in attempt_by_id:
        raise ValueError("experiment page attempt identities must be unique")
    nested = {
        str(attempt.get("attempt_id") or ""): attempt
        for pair in paired_cases
        for arm in ("baseline", "candidate")
        if isinstance((attempt := pair.get(arm)), Mapping)
    }
    if set(nested) != set(attempt_by_id) or any(
        nested[attempt_id] != attempt for attempt_id, attempt in attempt_by_id.items()
    ):
        raise ValueError(
            "experiment page attempts must match its paired-case attempt evidence"
        )
    projection_digest = _sha256_digest(
        raw.get("projection_digest"), "projection_digest"
    )
    page_set_id = _sha256_digest(raw.get("page_set_id"), "page_set_id")
    expected_set = experiment_view_page_set_id(
        projection_digest=projection_digest,
        page_count=page_count,
        attempt_count=attempt_count,
        paired_case_count=paired_case_count,
    )
    if page_set_id != expected_set:
        raise ValueError("experiment page-set identity does not recompute")
    unsigned = {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "page_set_id": page_set_id,
        "projection_digest": projection_digest,
        "page_index": page_index,
        "page_count": page_count,
        "attempt_count": attempt_count,
        "paired_case_count": paired_case_count,
        "attempts": attempts,
        "paired_cases": list(paired_cases),
    }
    page_digest = _sha256_digest(raw.get("page_digest"), "page_digest")
    if page_digest != stable_digest(unsigned):
        raise ValueError("experiment page digest does not recompute")
    return ExperimentViewPageV1(
        schema_version=RESEARCH_SCHEMA_VERSION,
        page_set_id=page_set_id,
        projection_digest=projection_digest,
        page_index=page_index,
        page_count=page_count,
        attempt_count=attempt_count,
        paired_case_count=paired_case_count,
        attempts=tuple(attempts),
        paired_cases=paired_cases,
        page_digest=page_digest,
    )


def experiment_view_manifest_from_dict(
    raw: Mapping[str, Any],
) -> ExperimentViewManifestV1:
    fields = {
        item.name for item in ExperimentViewManifestV1.__dataclass_fields__.values()
    }
    unknown = set(raw) - fields
    if unknown:
        raise ValueError("experiment view manifest has unknown fields")
    if raw.get("schema_version") != RESEARCH_SCHEMA_VERSION:
        raise ValueError("unsupported experiment view manifest schema")
    page_count = _positive_int(raw.get("page_count"), "page_count")
    attempt_count = _non_negative_int(raw.get("attempt_count"), "attempt_count")
    paired_case_count = _non_negative_int(
        raw.get("paired_case_count"), "paired_case_count"
    )
    projection_digest = _sha256_digest(
        raw.get("projection_digest"), "projection_digest"
    )
    page_set_id = _sha256_digest(raw.get("page_set_id"), "page_set_id")
    expected_set = experiment_view_page_set_id(
        projection_digest=projection_digest,
        page_count=page_count,
        attempt_count=attempt_count,
        paired_case_count=paired_case_count,
    )
    if page_set_id != expected_set:
        raise ValueError("experiment page-set identity does not recompute")
    projection = _json_mapping(raw.get("projection"), "manifest projection")
    if projection.get("schema_version") != 3 or projection.get("kind") != "evaluation":
        raise ValueError("paged experiment manifest requires a V3 evaluation")
    paged_fields = {"attempts", "paired_cases", "canonical_attempts", "aligned_rows"}
    if paged_fields & projection.keys():
        raise ValueError(
            "paged experiment manifest must not repeat materialized attempt fields"
        )
    unsigned = {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "page_set_id": page_set_id,
        "projection_digest": projection_digest,
        "page_count": page_count,
        "attempt_count": attempt_count,
        "paired_case_count": paired_case_count,
        "projection": projection,
    }
    manifest_digest = _sha256_digest(raw.get("manifest_digest"), "manifest_digest")
    if manifest_digest != stable_digest(unsigned):
        raise ValueError("experiment manifest digest does not recompute")
    return ExperimentViewManifestV1(**unsigned, manifest_digest=manifest_digest)


def weave_publication_evidence_from_receipt(
    raw: Mapping[str, Any],
) -> WeavePublicationEvidenceV1:
    """Project one fully verified receipt into bounded public Research evidence."""

    fields = {
        "schema_version",
        "publication_id",
        "target",
        "result_digest",
        "qualification_digest",
        "result_file_sha256",
        "local_manifest_digest",
        "local_manifest_file_sha256",
        "hosted_objects",
        "publisher_id",
        "publisher_revision",
        "status",
        "published_at",
        "receipt_digest",
    }
    unknown = set(raw) - fields
    missing = fields - set(raw)
    if unknown or missing:
        raise ValueError("Weave publication receipt fields are incomplete or unknown")
    receipt_digest = _sha256_digest(raw.get("receipt_digest"), "receipt_digest")
    unsigned = {key: value for key, value in raw.items() if key != "receipt_digest"}
    if stable_digest(unsigned) != receipt_digest:
        raise ValueError("Weave publication receipt digest does not recompute")
    target = _weave_publication_target(raw.get("target"))
    hosted_objects: list[WeaveHostedEvidenceRefV1] = []
    for item in _sequence(raw.get("hosted_objects"), "hosted_objects"):
        value = dict(_mapping(item, "hosted object"))
        object_target = _weave_publication_target(value.pop("target", None))
        if object_target != target:
            raise ValueError("hosted object target disagrees with publication target")
        hosted_objects.append(_weave_hosted_evidence_ref(value, target=target))
    return weave_publication_evidence_from_dict(
        {
            "schema_version": raw.get("schema_version"),
            "publication_id": raw.get("publication_id"),
            "receipt_digest": receipt_digest,
            "target": target.to_dict(),
            "result_digest": raw.get("result_digest"),
            "qualification_digest": raw.get("qualification_digest"),
            "result_file_sha256": raw.get("result_file_sha256"),
            "local_manifest_digest": raw.get("local_manifest_digest"),
            "local_manifest_file_sha256": raw.get("local_manifest_file_sha256"),
            "hosted_objects": [item.to_dict() for item in hosted_objects],
            "publisher_id": raw.get("publisher_id"),
            "publisher_revision": raw.get("publisher_revision"),
            "status": raw.get("status"),
            "published_at": raw.get("published_at"),
        }
    )


def weave_publication_evidence_from_dict(
    raw: Mapping[str, Any],
) -> WeavePublicationEvidenceV1:
    fields = {
        item.name for item in WeavePublicationEvidenceV1.__dataclass_fields__.values()
    }
    unknown = set(raw) - fields
    missing = fields - set(raw)
    if unknown or missing:
        raise ValueError("Weave publication evidence fields are incomplete or unknown")
    if not _is_schema_one(raw.get("schema_version")):
        raise ValueError("unsupported Weave publication evidence schema")
    target = _weave_publication_target(raw.get("target"))
    hosted_objects = tuple(
        _weave_hosted_evidence_ref(item, target=target)
        for item in _sequence(raw.get("hosted_objects"), "hosted_objects")
    )
    if not hosted_objects:
        raise ValueError("Weave publication evidence requires hosted objects")
    if (
        tuple(sorted(hosted_objects, key=lambda item: (item.attempt_id, item.kind)))
        != hosted_objects
    ):
        raise ValueError(
            "Weave publication evidence objects are not canonically sorted"
        )
    observed = {(item.attempt_id, item.kind) for item in hosted_objects}
    if len(observed) != len(hosted_objects):
        raise ValueError("Weave publication evidence contains duplicate objects")
    attempt_ids = tuple(sorted({item.attempt_id for item in hosted_objects}))
    expected = {
        (attempt_id, kind) for attempt_id in attempt_ids for kind in _WEAVE_HOSTED_KINDS
    }
    if observed != expected:
        raise ValueError(
            "Weave publication evidence requires one exact five-object chain "
            "per attempt"
        )
    result_digest = _sha256_digest(raw.get("result_digest"), "result_digest")
    local_manifest_digest = _sha256_digest(
        raw.get("local_manifest_digest"), "local_manifest_digest"
    )
    publication_id = _sha256_digest(raw.get("publication_id"), "publication_id")
    expected_publication_id = stable_digest(
        {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "target": target.to_dict(),
            "result_digest": result_digest,
            "local_manifest_digest": local_manifest_digest,
        }
    )
    if publication_id != expected_publication_id:
        raise ValueError("Weave publication evidence identity does not recompute")
    if raw.get("status") != "published":
        raise ValueError("Weave publication evidence is not published")
    published_at = _timestamp(raw.get("published_at"))
    evidence = WeavePublicationEvidenceV1(
        schema_version=RESEARCH_SCHEMA_VERSION,
        publication_id=publication_id,
        receipt_digest=_sha256_digest(raw.get("receipt_digest"), "receipt_digest"),
        target=target,
        result_digest=result_digest,
        qualification_digest=_sha256_digest(
            raw.get("qualification_digest"), "qualification_digest"
        ),
        result_file_sha256=_sha256_digest(
            raw.get("result_file_sha256"), "result_file_sha256"
        ),
        local_manifest_digest=local_manifest_digest,
        local_manifest_file_sha256=_sha256_digest(
            raw.get("local_manifest_file_sha256"),
            "local_manifest_file_sha256",
        ),
        hosted_objects=hosted_objects,
        publisher_id=_text(raw.get("publisher_id"), "publisher_id", 1000),
        publisher_revision=_text(
            raw.get("publisher_revision"), "publisher_revision", 1000
        ),
        status="published",
        published_at=published_at,
    )
    reconstructed_receipt = evidence.to_dict()
    receipt_digest = reconstructed_receipt.pop("receipt_digest")
    reconstructed_receipt["hosted_objects"] = [
        {**item.to_dict(), "target": target.to_dict()}
        for item in evidence.hosted_objects
    ]
    if stable_digest(reconstructed_receipt) != receipt_digest:
        raise ValueError("Weave publication receipt digest does not recompute")
    return evidence


def _weave_publication_target(raw: Any) -> WeavePublicationTargetEvidenceV1:
    value = dict(_mapping(raw, "Weave publication target"))
    fields = {"schema_version", "entity", "project", "study_scope", "destination"}
    if set(value) != fields:
        raise ValueError("Weave publication target fields are incomplete or unknown")
    if not _is_schema_one(value.get("schema_version")):
        raise ValueError("unsupported Weave publication target schema")
    entity = _text(value.get("entity"), "Weave entity", 128)
    project = _text(value.get("project"), "Weave project", 128)
    if not _WEAVE_SLUG_PART.fullmatch(entity) or not _WEAVE_SLUG_PART.fullmatch(
        project
    ):
        raise ValueError("Weave publication target is invalid")
    scope_value = dict(_mapping(value.get("study_scope"), "Study publication scope"))
    if set(scope_value) != {"schema_version", "research_id", "study_id"}:
        raise ValueError("Study publication scope fields are incomplete or unknown")
    if not _is_schema_one(scope_value.get("schema_version")):
        raise ValueError("unsupported Study publication scope schema")
    research_id = _text(scope_value.get("research_id"), "research_id", 256)
    study_id = _text(scope_value.get("study_id"), "study_id", 256)
    if not _WEAVE_STUDY_ID.fullmatch(research_id) or not _WEAVE_STUDY_ID.fullmatch(
        study_id
    ):
        raise ValueError("Study publication scope is invalid")
    destination = evidence_destination_from_dict(
        _mapping(value.get("destination"), "Weave publication destination")
    )
    if destination.entity != entity or destination.project != project:
        raise ValueError("Weave target and destination disagree")
    target = WeavePublicationTargetEvidenceV1(
        entity=entity,
        project=project,
        study_scope=WeavePublicationScopeEvidenceV1(
            research_id=research_id,
            study_id=study_id,
        ),
        destination=destination,
    )
    if value != target.to_dict():
        raise ValueError("persisted Weave target must use its canonical values")
    return target


def _weave_hosted_evidence_ref(
    raw: Any,
    *,
    target: WeavePublicationTargetEvidenceV1,
) -> WeaveHostedEvidenceRefV1:
    value = dict(_mapping(raw, "hosted evidence object"))
    fields = {
        "attempt_id",
        "kind",
        "object_id",
        "ref",
        "system",
        "native_agent_call",
    }
    if set(value) != fields:
        raise ValueError("hosted evidence object fields are incomplete or unknown")
    attempt_id = _sha256_digest(value.get("attempt_id"), "attempt_id")
    kind = str(value.get("kind") or "")
    if kind not in _WEAVE_HOSTED_KINDS:
        raise ValueError("hosted evidence kind is invalid")
    if value.get("system") != "weave":
        raise ValueError("hosted evidence system must be weave")
    if value.get("native_agent_call") is not False:
        raise ValueError(
            "a published local Agent receipt is not a native Weave Agent call"
        )
    object_id = _text(value.get("object_id"), "hosted object id", 512)
    if not _WEAVE_OBJECT_ID.fullmatch(object_id):
        raise ValueError("hosted evidence object id is invalid")
    if kind == "dataset":
        _immutable_weave_dataset_object_id(object_id)
    object_type = "call" if kind in _WEAVE_CALL_KINDS else "object"
    expected_ref = f"weave:///{target.project_slug}/{object_type}/{object_id}"
    ref = _text(value.get("ref"), "hosted evidence ref", 2000)
    if ref != expected_ref:
        raise ValueError("hosted evidence ref disagrees with its exact Weave target")
    return WeaveHostedEvidenceRefV1(
        attempt_id=attempt_id,
        kind=kind,  # type: ignore[arg-type]
        object_id=object_id,
        ref=ref,
    )


def research_index_report_attempt_ids_digest(attempt_ids: Iterable[str]) -> str:
    """Bind one canonical nonempty attempt census without publishing every ID."""

    values = tuple(sorted(str(item) for item in attempt_ids))
    if not values or len(set(values)) != len(values):
        raise ValueError("Report Study attempts must be nonempty and unique")
    for value in values:
        _sha256_digest(value, "attempt_id")
    return stable_digest(
        {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "attempt_ids": list(values),
        }
    )


def research_index_report_study_membership_from_dict(
    raw: Mapping[str, Any],
) -> ResearchIndexReportStudyMembershipV1:
    fields = {
        item.name
        for item in ResearchIndexReportStudyMembershipV1.__dataclass_fields__.values()
    }
    if set(raw) != fields:
        raise ValueError("Research Report Study membership fields are not exact")
    study_id = _text(raw.get("study_id"), "Study id", 256)
    if not _WEAVE_STUDY_ID.fullmatch(study_id):
        raise ValueError("Research Report Study id is invalid")
    return ResearchIndexReportStudyMembershipV1(
        study_id=study_id,
        result_digest=_sha256_digest(raw.get("result_digest"), "result_digest"),
        qualification_digest=_sha256_digest(
            raw.get("qualification_digest"), "qualification_digest"
        ),
        result_file_sha256=_sha256_digest(
            raw.get("result_file_sha256"), "result_file_sha256"
        ),
        weave_project=_research_project(raw.get("weave_project")),
        weave_publication_id=_sha256_digest(
            raw.get("weave_publication_id"), "weave_publication_id"
        ),
        weave_receipt_digest=_sha256_digest(
            raw.get("weave_receipt_digest"), "weave_receipt_digest"
        ),
        attempt_count=_exact_positive_int(
            raw.get("attempt_count"), "attempt_count"
        ),
        attempt_ids_digest=_sha256_digest(
            raw.get("attempt_ids_digest"), "attempt_ids_digest"
        ),
    )


def research_index_report_publication_evidence_from_dict(
    raw: Mapping[str, Any],
) -> ResearchIndexReportPublicationEvidenceV1:
    fields = {
        item.name
        for item in ResearchIndexReportPublicationEvidenceV1.__dataclass_fields__.values()
    }
    if set(raw) != fields:
        raise ValueError("Research index Report evidence fields are not exact")
    if not _is_schema_one(raw.get("schema_version")):
        raise ValueError("unsupported Research index Report evidence schema")
    research_id = _text(raw.get("research_id"), "research_id", 256)
    if not _WEAVE_STUDY_ID.fullmatch(research_id):
        raise ValueError("Research index Report scope is invalid")
    target = _research_index_report_target(raw.get("target"))
    index = _research_index_receipt_evidence(
        raw.get("index"),
        research_id=research_id,
        target=target,
    )
    report = _research_report_receipt_evidence(
        raw.get("report"),
        research_id=research_id,
        target=target,
    )
    studies = tuple(
        research_index_report_study_membership_from_dict(
            _mapping(item, "Research Report Study membership")
        )
        for item in _sequence(raw.get("studies"), "studies")
    )
    if (
        not studies
        or tuple(sorted(studies, key=lambda item: item.study_id)) != studies
        or len({item.study_id for item in studies}) != len(studies)
    ):
        raise ValueError(
            "Research index Report Study memberships must be nonempty, unique, "
            "and sorted"
        )
    if (
        report.index_digest != index.index_digest
        or report.index_file_sha256 != index.index_file_sha256
        or report.index_publication_id != index.publication_id
        or report.index_publication_receipt_digest != index.receipt_digest
        or report.index_publication_receipt_file_sha256 != index.receipt_file_sha256
    ):
        raise ValueError(
            "Research Report receipt disagrees with the exact Research index receipt"
        )
    return ResearchIndexReportPublicationEvidenceV1(
        research_id=research_id,
        target=target,
        index=index,
        report=report,
        studies=studies,
    )


def _research_index_report_target(
    raw: Any,
) -> ResearchIndexReportTargetEvidenceV1:
    value = dict(_mapping(raw, "Research index Report target"))
    fields = {
        "schema_version",
        "project",
        "api_base_url",
        "app_base_url",
        "backend",
        "destination_digest",
    }
    if set(value) != fields:
        raise ValueError("Research index Report target fields are not exact")
    if not _is_schema_one(value.get("schema_version")):
        raise ValueError("unsupported Research index Report target schema")
    project = _research_project(value.get("project"))
    api_base_url = _research_https_origin(
        value.get("api_base_url"), "W&B API origin"
    )
    app_base_url = _research_https_origin(
        value.get("app_base_url"), "W&B application origin"
    )
    if value.get("backend") != "wandb":
        raise ValueError("Research index Report target backend must be wandb")
    target = ResearchIndexReportTargetEvidenceV1(
        schema_version=RESEARCH_SCHEMA_VERSION,
        project=project,
        api_base_url=api_base_url,
        app_base_url=app_base_url,
        backend="wandb",
        destination_digest=_sha256_digest(
            value.get("destination_digest"), "destination_digest"
        ),
    )
    unsigned = target.to_dict()
    destination_digest = unsigned.pop("destination_digest")
    if stable_digest(unsigned) != destination_digest:
        raise ValueError("Research index Report destination digest does not recompute")
    return target


def _research_index_receipt_evidence(
    raw: Any,
    *,
    research_id: str,
    target: ResearchIndexReportTargetEvidenceV1,
) -> ResearchIndexReceiptEvidenceV1:
    value = dict(_mapping(raw, "Research-index receipt evidence"))
    fields = {
        item.name for item in ResearchIndexReceiptEvidenceV1.__dataclass_fields__.values()
    }
    if set(value) != fields:
        raise ValueError("Research-index receipt evidence fields are not exact")
    report_status = str(value.get("report_status") or "")
    report_url = _optional_receipt_text(
        value.get("report_url"), "index Report URL", 4000
    )
    if report_status == "published":
        if report_url is None:
            raise ValueError("published index Report requires a URL")
        _research_project_url(report_url, target, resource="reports")
    elif report_status == "unavailable":
        if report_url is not None:
            raise ValueError("unavailable index Report cannot have a URL")
    else:
        raise ValueError("Research-index receipt Report status is invalid")
    if value.get("status") != "published":
        raise ValueError("Research-index receipt is not published")
    evidence = ResearchIndexReceiptEvidenceV1(
        publication_id=_sha256_digest(
            value.get("publication_id"), "index publication_id"
        ),
        index_digest=_sha256_digest(value.get("index_digest"), "index_digest"),
        index_file_sha256=_sha256_digest(
            value.get("index_file_sha256"), "index_file_sha256"
        ),
        receipt_digest=_sha256_digest(
            value.get("receipt_digest"), "index receipt_digest"
        ),
        receipt_file_sha256=_sha256_digest(
            value.get("receipt_file_sha256"), "index receipt_file_sha256"
        ),
        run_url=_research_project_url(value.get("run_url"), target, resource="runs"),
        artifact_url=_research_project_url(
            value.get("artifact_url"), target, resource="artifacts"
        ),
        report_url=report_url,
        report_status=report_status,  # type: ignore[arg-type]
        publisher_id=_exact_receipt_text(
            value.get("publisher_id"), "index publisher_id", 1000
        ),
        publisher_revision=_exact_receipt_text(
            value.get("publisher_revision"), "index publisher_revision", 1000
        ),
        status="published",
        published_at=_receipt_timestamp(
            value.get("published_at"), "Research-index publication timestamp"
        ),
    )
    expected_id = stable_digest(
        {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "research_id": research_id,
            "index_digest": evidence.index_digest,
            "index_file_sha256": evidence.index_file_sha256,
            "target": target.to_dict(),
        }
    )
    if evidence.publication_id != expected_id:
        raise ValueError("Research-index publication identity does not recompute")
    reconstructed = evidence.receipt_dict(research_id=research_id, target=target)
    receipt_digest = reconstructed.pop("receipt_digest")
    if stable_digest(reconstructed) != receipt_digest:
        raise ValueError("Research-index receipt digest does not recompute")
    return evidence


def _research_report_receipt_evidence(
    raw: Any,
    *,
    research_id: str,
    target: ResearchIndexReportTargetEvidenceV1,
) -> ResearchReportReceiptEvidenceV1:
    value = dict(_mapping(raw, "Research Report receipt evidence"))
    fields = {
        item.name
        for item in ResearchReportReceiptEvidenceV1.__dataclass_fields__.values()
    }
    if set(value) != fields:
        raise ValueError("Research Report receipt evidence fields are not exact")
    report_id = _safe_receipt_text(
        value.get("report_id"), "Report id", 512, single_line=True
    )
    if not _WANDB_REPORT_ID.fullmatch(report_id):
        raise ValueError("Research Report id is invalid")
    report_url = _research_project_url(
        value.get("report_url"),
        target,
        resource="reports",
        report_id=report_id,
    )
    projection_digest = _sha256_digest(
        value.get("projection_digest"), "Report projection_digest"
    )
    readback_projection_digest = _sha256_digest(
        value.get("readback_projection_digest"),
        "Report readback_projection_digest",
    )
    literals = {
        "api_stability": "public_preview",
        "readback_status": "reconciled",
        "access_mode": "project_settings",
        "share_link_action": "not_requested",
        "current_pointer_status": "not_managed",
        "status": "published_and_reconciled",
    }
    if any(value.get(key) != expected for key, expected in literals.items()):
        raise ValueError("Research Report receipt status fields are invalid")
    if readback_projection_digest != projection_digest:
        raise ValueError("Research Report readback does not bind its projection")
    evidence = ResearchReportReceiptEvidenceV1(
        report_publication_id=_sha256_digest(
            value.get("report_publication_id"), "Report publication_id"
        ),
        projection_digest=projection_digest,
        index_digest=_sha256_digest(value.get("index_digest"), "index_digest"),
        index_file_sha256=_sha256_digest(
            value.get("index_file_sha256"), "index_file_sha256"
        ),
        index_publication_id=_sha256_digest(
            value.get("index_publication_id"), "index_publication_id"
        ),
        index_publication_receipt_digest=_sha256_digest(
            value.get("index_publication_receipt_digest"),
            "index_publication_receipt_digest",
        ),
        index_publication_receipt_file_sha256=_sha256_digest(
            value.get("index_publication_receipt_file_sha256"),
            "index_publication_receipt_file_sha256",
        ),
        receipt_digest=_sha256_digest(
            value.get("receipt_digest"), "Report receipt_digest"
        ),
        receipt_file_sha256=_sha256_digest(
            value.get("receipt_file_sha256"), "Report receipt_file_sha256"
        ),
        renderer_id=_safe_receipt_text(
            value.get("renderer_id"),
            "Report renderer_id",
            128,
            single_line=True,
        ),
        renderer_revision=_safe_receipt_text(
            value.get("renderer_revision"),
            "Report renderer_revision",
            128,
            single_line=True,
        ),
        report_id=report_id,
        report_url=report_url,
        report_api=_safe_receipt_text(
            value.get("report_api"), "Report API", 128, single_line=True
        ),
        report_api_version=_safe_receipt_text(
            value.get("report_api_version"),
            "Report API version",
            64,
            single_line=True,
        ),
        api_stability="public_preview",
        readback_projection_digest=readback_projection_digest,
        rendered_content_digest=_sha256_digest(
            value.get("rendered_content_digest"), "rendered_content_digest"
        ),
        readback_status="reconciled",
        publisher_id=_safe_receipt_text(
            value.get("publisher_id"),
            "Report publisher_id",
            128,
            single_line=True,
        ),
        publisher_revision=_safe_receipt_text(
            value.get("publisher_revision"),
            "Report publisher_revision",
            256,
            single_line=True,
        ),
        access_mode="project_settings",
        share_link_action="not_requested",
        current_pointer_status="not_managed",
        status="published_and_reconciled",
        published_at=_receipt_timestamp(
            value.get("published_at"), "Research Report publication timestamp"
        ),
    )
    expected_id = stable_digest(
        {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "research_id": research_id,
            "projection_digest": evidence.projection_digest,
            "target": target.to_dict(),
        }
    )
    if evidence.report_publication_id != expected_id:
        raise ValueError("Research Report publication identity does not recompute")
    reconstructed = evidence.receipt_dict(research_id=research_id, target=target)
    receipt_digest = reconstructed.pop("receipt_digest")
    if stable_digest(reconstructed) != receipt_digest:
        raise ValueError("Research Report receipt digest does not recompute")
    return evidence


def _research_project(value: Any) -> str:
    project = _exact_receipt_text(value, "W&B project", 257)
    parts = project.split("/")
    if len(parts) != 2 or any(not _WEAVE_SLUG_PART.fullmatch(item) for item in parts):
        raise ValueError("W&B project must be exactly ENTITY/PROJECT")
    return project


def _research_https_origin(value: Any, label: str) -> str:
    url = _exact_receipt_text(value, label, 2000)
    if (
        any(
            character.isspace() or unicodedata.category(character) == "Cc"
            for character in url
        )
        or "\\" in url
    ):
        raise ValueError(f"{label} must be a normalized HTTPS origin")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} must be a normalized HTTPS origin") from exc
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or not hostname
        or parsed.username
        or parsed.password
        or (port is not None and not 1 <= port <= 65535)
        or parsed.path
        or parsed.query
        or parsed.fragment
        or url.endswith("/")
        or redact_text(url) != url
    ):
        raise ValueError(f"{label} must be a normalized HTTPS origin")
    return url


def _research_project_url(
    value: Any,
    target: ResearchIndexReportTargetEvidenceV1,
    *,
    resource: Literal["runs", "artifacts", "reports"],
    report_id: str | None = None,
) -> str:
    url = _exact_receipt_text(value, f"W&B {resource} URL", 4000)
    parsed = urlsplit(url)
    expected = urlsplit(target.app_base_url)
    parts = _canonical_research_url_path(parsed.path)
    if (
        any(character.isspace() for character in url)
        or any(unicodedata.category(character) == "Cc" for character in url)
        or any(character in "\\[]()<>'\"`" for character in url)
        or redact_text(url) != url
        or parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or (parsed.scheme, parsed.netloc) != (expected.scheme, expected.netloc)
        or len(parts) != (6 if resource == "artifacts" else 4)
        or parts[:3] != [target.entity, target.project_name, resource]
        or (resource == "artifacts" and not re.fullmatch(r"v\d+", parts[-1]))
    ):
        raise ValueError(f"W&B {resource} URL disagrees with the target project")
    if report_id is not None:
        accepted = {f"--{report_id}", f"--{report_id.rstrip('=')}"}
        if len(parts) != 4 or not any(parts[3].endswith(item) for item in accepted):
            raise ValueError("W&B Report URL disagrees with the exact Report id")
    return url


def _canonical_research_url_path(path: str) -> list[str]:
    if (
        not path.startswith("/")
        or path.endswith("/")
        or "//" in path
        or "\\" in path
        or "%" in path
    ):
        raise ValueError("W&B resource URL path is not canonical")
    parts = path[1:].split("/")
    for part in parts:
        if (
            not part
            or part in {".", ".."}
            or any(unicodedata.category(character) == "Cc" for character in part)
        ):
            raise ValueError("W&B resource URL path is not canonical")
    return parts


def _immutable_weave_dataset_object_id(value: str) -> None:
    if value.count(":") != 1:
        raise ValueError(
            "hosted Dataset object id must contain one name and content revision"
        )
    name, revision = value.split(":", 1)
    if not name or not _WEAVE_CONTENT_HASH.fullmatch(revision):
        raise ValueError(
            "hosted Dataset object id must use a positive content hash"
        )


def _exact_receipt_text(value: Any, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
    ):
        raise ValueError(f"{label} must contain 1 to {maximum} exact characters")
    return value


def _optional_receipt_text(value: Any, label: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _exact_receipt_text(value, label, maximum)


def _safe_receipt_text(
    value: Any,
    label: str,
    maximum: int,
    *,
    single_line: bool = False,
) -> str:
    text = _exact_receipt_text(value, label, maximum)
    if (
        redact_text(text) != text
        or any(ord(character) < 32 and character not in "\n\t" for character in text)
        or (single_line and ("\n" in text or "\r" in text))
    ):
        raise ValueError(f"{label} is unsafe or invalid")
    lowered = text.casefold()
    if any(item in lowered for item in ("javascript:", "data:", "file:", "<script")):
        raise ValueError(f"{label} contains unsafe content")
    return text


def _receipt_timestamp(value: Any, label: str) -> str:
    text = _exact_receipt_text(value, label, 100)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must use ISO 8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return text


def research_log_event_from_dict(
    raw: Mapping[str, Any], *, require_digest: bool = True
) -> ResearchLogEventV1:
    fields = {item.name for item in ResearchLogEventV1.__dataclass_fields__.values()}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(
            "research log event has unknown fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    schema_version = _positive_int(raw.get("schema_version"), "schema_version")
    if schema_version != RESEARCH_SCHEMA_VERSION:
        raise ValueError("unsupported research log event schema")
    classification = str(raw.get("classification") or "")
    if classification not in _CLASSIFICATIONS:
        raise ValueError("unknown research log classification")
    state = str(raw.get("state") or "")
    if state not in _STATES:
        raise ValueError("unknown research log state")
    relationships = tuple(
        _relationship(item)
        for item in _sequence(raw.get("relationships"), "relationships")
    )
    evidence = tuple(
        _evidence_ref(item) for item in _sequence(raw.get("evidence"), "evidence")
    )
    event = ResearchLogEventV1(
        schema_version=schema_version,
        producer_event_id=_text(
            raw.get("producer_event_id"), "producer_event_id", 1000
        ),
        sequence=_positive_int(raw.get("sequence"), "sequence"),
        timestamp=_timestamp(raw.get("timestamp")),
        source=_text(raw.get("source"), "source", 300),
        actor=attribution_from_dict(_mapping(raw.get("actor"), "actor")),
        research_id=_text(raw.get("research_id"), "research_id", 1000),
        study_id=_optional_text(raw.get("study_id"), "study_id", 1000),
        classification=classification,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        message=_text(raw.get("message"), "message", 4000),
        progress=_json_mapping(raw.get("progress"), "progress"),
        reserved_cost_usd=_cost(raw.get("reserved_cost_usd"), "reserved cost"),
        observed_cost_usd=_cost(raw.get("observed_cost_usd"), "observed cost"),
        relationships=relationships,
        evidence=evidence,
        summary=_json_mapping(raw.get("summary"), "summary"),
        event_digest=str(raw.get("event_digest") or ""),
    )
    experiment_view = event.summary.get("experiment_view")
    experiment_view_page = event.summary.get("experiment_view_page")
    experiment_view_manifest = event.summary.get("experiment_view_manifest")
    weave_publication = event.summary.get("weave_publication")
    research_index_report_publication = event.summary.get(
        "research_index_report_publication"
    )
    experiment_payloads = sum(
        item is not None
        for item in (experiment_view, experiment_view_page, experiment_view_manifest)
    )
    if experiment_payloads > 1:
        raise ValueError("one event cannot contain multiple experiment payloads")
    if experiment_payloads and event.study_id is None:
        raise ValueError("experiment projections require a Study ID")
    if experiment_view is not None:
        if not isinstance(experiment_view, Mapping):
            raise ValueError("summary.experiment_view must be an object")
        from fugue.research.experiment_views import experiment_view_from_dict

        experiment_view_from_dict(experiment_view)
    if experiment_view_page is not None:
        if not isinstance(experiment_view_page, Mapping):
            raise ValueError("summary.experiment_view_page must be an object")
        experiment_view_page_from_dict(experiment_view_page)
        if event.state in {"completed", "failed", "cancelled"}:
            raise ValueError("experiment pages cannot close a Study")
    if experiment_view_manifest is not None:
        if not isinstance(experiment_view_manifest, Mapping):
            raise ValueError("summary.experiment_view_manifest must be an object")
        experiment_view_manifest_from_dict(experiment_view_manifest)
        if event.state not in {"completed", "failed", "cancelled"}:
            raise ValueError("experiment manifests must declare a terminal Study state")
    _validate_weave_publication_event(
        event,
        weave_publication=weave_publication,
        experiment_payloads=experiment_payloads,
    )
    _validate_research_index_report_publication_event(
        event,
        publication_raw=research_index_report_publication,
        experiment_payloads=experiment_payloads,
        weave_publication=weave_publication,
    )
    unsigned = event.to_dict()
    unsigned.pop("event_digest", None)
    if (
        len(json.dumps(unsigned, separators=(",", ":")).encode())
        > RESEARCH_LOG_MAX_BYTES
    ):
        raise ValueError("research log event exceeds the publication size limit")
    digest = stable_digest(unsigned)
    if event.event_digest and event.event_digest != digest:
        raise ValueError("event_digest does not match research log event")
    if require_digest and event.event_digest != digest:
        raise ValueError("event_digest is required")
    return replace(event, event_digest=digest)


def sign_research_log_event(event: ResearchLogEventV1) -> ResearchLogEventV1:
    return research_log_event_from_dict(event.to_dict(), require_digest=False)


def _validate_weave_publication_event(
    event: ResearchLogEventV1,
    *,
    weave_publication: Any,
    experiment_payloads: int,
) -> None:
    if weave_publication is None:
        return
    if experiment_payloads:
        raise ValueError(
            "Weave publication evidence cannot replace an experiment projection"
        )
    if not isinstance(weave_publication, Mapping):
        raise ValueError("summary.weave_publication must be an object")
    publication = weave_publication_evidence_from_dict(weave_publication)
    scope = publication.target.study_scope
    if (
        event.research_id != scope.research_id
        or event.study_id != scope.study_id
        or event.classification != "evidence"
        or event.state != "completed"
    ):
        raise ValueError(
            "Weave publication evidence disagrees with its Research event scope"
        )
    expected_evidence = ResearchEvidenceRefV1(
        system="weave",
        kind="weave_publication_receipt",
        ref=f"weave-publication:{publication.publication_id}",
        digest=publication.receipt_digest,
        version=str(publication.schema_version),
        selector={
            "entity": publication.target.entity,
            "project": publication.target.project,
        },
    )
    if event.evidence != (expected_evidence,):
        raise ValueError(
            "Weave publication event evidence ref disagrees with its receipt"
        )


def _validate_research_index_report_publication_event(
    event: ResearchLogEventV1,
    *,
    publication_raw: Any,
    experiment_payloads: int,
    weave_publication: Any,
) -> None:
    if publication_raw is None:
        return
    if experiment_payloads or weave_publication is not None:
        raise ValueError(
            "Research index Report evidence cannot replace Study evidence"
        )
    if not isinstance(publication_raw, Mapping):
        raise ValueError(
            "summary.research_index_report_publication must be an object"
        )
    publication = research_index_report_publication_evidence_from_dict(
        publication_raw
    )
    if (
        event.research_id != publication.research_id
        or event.study_id is not None
        or event.classification != "evidence"
        or event.state != "completed"
    ):
        raise ValueError(
            "Research index Report evidence disagrees with its Research event scope"
        )
    selector = {
        "entity": publication.target.entity,
        "project": publication.target.project_name,
    }
    expected_evidence = (
        ResearchEvidenceRefV1(
            system="wandb",
            kind="research_index_publication_receipt",
            ref=f"wandb-research-index:{publication.index.publication_id}",
            uri=publication.index.run_url,
            digest=publication.index.receipt_digest,
            version=str(publication.schema_version),
            selector=selector,
        ),
        ResearchEvidenceRefV1(
            system="wandb",
            kind="research_index_report_publication_receipt",
            ref=(
                "wandb-research-report:"
                f"{publication.report.report_publication_id}"
            ),
            uri=publication.report.report_url,
            digest=publication.report.receipt_digest,
            version=str(publication.schema_version),
            selector=selector,
        ),
    )
    if event.evidence != expected_evidence:
        raise ValueError(
            "Research index Report event refs disagree with its exact receipts"
        )


def event_state(value: str) -> ResearchLogState:
    if value in {"queued", "planning", "preparing", "admitting", "launching"}:
        return "preparing"
    if value in {"running", "cancelling"}:
        return "running"
    if value in {"scoring", "analyzing"}:
        return "evaluating"
    if value == "completed":
        return "completed"
    if value == "cancelled":
        return "cancelled"
    if value in {"blocked", "interrupted", "failed"}:
        return "failed"
    return "proposed"


def public_evidence_selector(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Keep only bounded public identities; evidence bodies remain at the source."""

    selector: dict[str, JsonValue] = {}
    for key, item in value.items():
        if key not in _PUBLIC_SELECTOR_KEYS:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            selector[key] = item
        elif (
            isinstance(item, list)
            and len(item) <= 50
            and all(
                isinstance(member, (str, int, float, bool)) or member is None
                for member in item
            )
        ):
            selector[key] = item
    return selector


class ResearchRecordSink(Protocol):
    @property
    def sink_id(self) -> str: ...

    def publish(self, event: ResearchLogEventV1) -> None: ...


class JsonlResearchRecordSink:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    @property
    def sink_id(self) -> str:
        return f"jsonl:{stable_digest(str(self.path))[:20]}"

    def publish(self, event: ResearchLogEventV1) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        with FileLock(lock_path):
            # The append-only log is authoritative. Rebuilding the compact index
            # makes a crash after fsync but before the index rename recoverable
            # without appending the producer event a second time.
            records = self._records_from_log()
            prior = records.get(event.producer_event_id)
            if prior:
                if prior != event.event_digest:
                    raise ResearchError(
                        "publication_conflict",
                        "producer event id was replayed with different content",
                        category="conflict",
                    )
                self._write_index(records)
                return
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            records[event.producer_event_id] = event.event_digest
            self._write_index(records)

    def _records_from_log(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        size = self.path.stat().st_size
        index = self.path.with_suffix(f"{self.path.suffix}.index.json")
        if index.is_file():
            try:
                cached = json.loads(index.read_text(encoding="utf-8"))
                offset = int(cached["offset"])
                raw_records = cached["records"]
                if (
                    cached.get("version") == 1
                    and 0 <= offset <= size
                    and isinstance(raw_records, dict)
                    and all(
                        isinstance(key, str) and isinstance(value, str)
                        for key, value in raw_records.items()
                    )
                ):
                    records = dict(raw_records)
                    if offset == size:
                        return records
                    with self.path.open("rb") as stream:
                        stream.seek(offset)
                        tail = stream.read().decode("utf-8")
                    return self._records_from_lines(
                        tail.splitlines(), records=records, first_line=0
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                # The index is a disposable cache. Rebuild it from the durable
                # append-only log rather than turning cache damage into data loss.
                pass
        return self._records_from_lines(
            self.path.read_text(encoding="utf-8").splitlines(),
            records={},
            first_line=1,
        )

    @staticmethod
    def _records_from_lines(
        lines: Iterable[str], *, records: dict[str, str], first_line: int
    ) -> dict[str, str]:
        for line_number, line in enumerate(lines, start=first_line):
            if not line.strip():
                continue
            try:
                event = research_log_event_from_dict(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"research record JSONL is invalid at line {line_number}"
                ) from exc
            prior = records.get(event.producer_event_id)
            if prior and prior != event.event_digest:
                raise ResearchError(
                    "publication_conflict",
                    "producer event id appears with conflicting content",
                    category="conflict",
                )
            records[event.producer_event_id] = event.event_digest
        return records

    def _write_index(self, records: Mapping[str, str]) -> None:
        index = self.path.with_suffix(f"{self.path.suffix}.index.json")
        temporary = index.with_suffix(f"{index.suffix}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": 1,
                    "offset": self.path.stat().st_size,
                    "records": dict(records),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(index)


class HttpResearchRecordSink:
    def __init__(self, url: str, token: str, *, timeout: float = 10.0) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("research record HTTP sink must use http or https")
        if not token:
            raise ValueError("research record HTTP sink requires an ingest token")

    @property
    def sink_id(self) -> str:
        return f"http:{stable_digest(self.url)[:20]}"

    def publish(self, event: ResearchLogEventV1) -> None:
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=False,
            ) as client:
                response = client.post(
                    self.url,
                    content=json.dumps(
                        event.to_dict(), separators=(",", ":")
                    ).encode(),
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                        "Idempotency-Key": event.producer_event_id,
                    },
                )
                if response.status_code not in {200, 201, 202, 204}:
                    raise RuntimeError(
                        f"research record sink returned HTTP {response.status_code}"
                    )
        except httpx.HTTPError as exc:
            raise RuntimeError("research record sink delivery failed") from exc


class ResearchRecordPublisher:
    def __init__(
        self,
        store: Any,
        sinks: Iterable[ResearchRecordSink],
        *,
        research_ids: Iterable[str] = (),
    ) -> None:
        self.store = store
        self.sinks = tuple(sinks)
        self.research_ids = frozenset(
            str(value).strip() for value in research_ids if str(value).strip()
        )

    @classmethod
    def from_environment(
        cls, store: Any, *, env: Mapping[str, str] | None = None
    ) -> ResearchRecordPublisher:
        values = dict(os.environ if env is None else env)
        sinks: list[ResearchRecordSink] = []
        jsonl = values.get("FUGUE_RESEARCH_RECORD_JSONL", "").strip()
        if jsonl:
            sinks.append(JsonlResearchRecordSink(Path(jsonl)))
        url = values.get("FUGUE_RESEARCH_RECORD_HTTP_URL", "").strip()
        if url:
            token = _secret(values, "FUGUE_RESEARCH_RECORD_TOKEN")
            sinks.append(HttpResearchRecordSink(url, token))
        research_ids = tuple(
            value.strip()
            for value in values.get("FUGUE_RESEARCH_RECORD_RESEARCH_IDS", "").split(",")
            if value.strip()
        )
        return cls(store, sinks, research_ids=research_ids)

    def flush(self, *, limit: int = 100) -> dict[str, int]:
        delivered = 0
        failed = 0
        for sink in self.sinks:
            for event in self.store.pending_research_log_events(
                sink.sink_id,
                limit=limit,
                research_ids=self.research_ids,
            ):
                try:
                    sink.publish(event)
                except Exception as exc:  # publication must not affect a Run
                    self.store.mark_research_log_failed(
                        sink.sink_id, event.sequence, str(exc)
                    )
                    failed += 1
                    continue
                self.store.mark_research_log_delivered(sink.sink_id, event.sequence)
                delivered += 1
        return {"delivered": delivered, "failed": failed}

    def replay(
        self,
        *,
        research_id: str | None = None,
        after: int = 0,
        page_size: int = 1000,
    ) -> dict[str, int]:
        """Republish immutable records without changing the producing operation.

        This is an operator recovery path for a rebuilt projection sink. Event
        identities and digests remain unchanged, so consumers still enforce
        idempotency and conflicting replays still fail.
        """

        delivered = 0
        failed = 0
        if (
            research_id is not None
            and self.research_ids
            and research_id not in self.research_ids
        ):
            raise ValueError("research publication is outside the configured scope")
        cursor = after
        while True:
            events = self.store.research_log_events(
                after=cursor,
                limit=page_size,
            )
            if not events:
                break
            for event in events:
                cursor = event.sequence
                if research_id is not None and event.research_id != research_id:
                    continue
                if self.research_ids and event.research_id not in self.research_ids:
                    continue
                for sink in self.sinks:
                    try:
                        sink.publish(event)
                    except Exception as exc:
                        self.store.mark_research_log_failed(
                            sink.sink_id, event.sequence, str(exc)
                        )
                        failed += 1
                        continue
                    self.store.mark_research_log_delivered(sink.sink_id, event.sequence)
                    delivered += 1
            if len(events) < page_size:
                break
        return {"delivered": delivered, "failed": failed}


def _secret(env: Mapping[str, str], name: str) -> str:
    path = env.get(f"{name}_FILE", "").strip()
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return env.get(name, "").strip()


def _relationship(raw: Any) -> ResearchRelationshipV1:
    value = _mapping(raw, "relationship")
    unknown = set(value) - {"kind", "target"}
    if unknown:
        raise ValueError("research relationship has unknown fields")
    kind = str(value.get("kind") or "")
    if kind not in _RELATIONSHIPS:
        raise ValueError("unknown research relationship kind")
    return ResearchRelationshipV1(
        kind=kind,  # type: ignore[arg-type]
        target=_text(value.get("target"), "relationship target", 2000),
    )


def _evidence_ref(raw: Any) -> ResearchEvidenceRefV1:
    value = _mapping(raw, "evidence reference")
    unknown = set(value) - {
        "system",
        "kind",
        "ref",
        "uri",
        "digest",
        "version",
        "selector",
    }
    if unknown:
        raise ValueError("research evidence reference has unknown fields")
    digest = _optional_text(value.get("digest"), "evidence digest", 64)
    if digest and (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("evidence digest must be sha256")
    uri = _optional_text(value.get("uri"), "evidence uri", 4000)
    if uri and not uri.startswith(("http://", "https://")):
        raise ValueError("evidence uri must use http or https")
    return ResearchEvidenceRefV1(
        system=_text(value.get("system"), "evidence system", 300),
        kind=_text(value.get("kind"), "evidence kind", 300),
        ref=_text(value.get("ref"), "evidence ref", 2000),
        uri=uri,
        digest=digest,
        version=_optional_text(value.get("version"), "evidence version", 300),
        selector=_json_mapping(value.get("selector"), "evidence selector"),
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    return tuple(value)


def _text(value: Any, label: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum:
        raise ValueError(f"{label} must contain 1 to {maximum} characters")
    return text


def _optional_text(value: Any, label: str, maximum: int) -> str | None:
    if value in (None, ""):
        return None
    return _text(value, label, maximum)


def _timestamp(value: Any) -> str:
    text = _text(value, "timestamp", 100)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must use ISO 8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return text


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{label} must be a positive integer")
    return integer


def _exact_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be an exact positive integer")
    return value


def _is_schema_one(value: Any) -> bool:
    return type(value) is int and value == RESEARCH_SCHEMA_VERSION


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return integer


def _sha256_digest(value: Any, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be sha256")
    return digest


def _cost(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be non-negative")
    number = float(value)
    if number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _json_mapping(value: Any, label: str) -> dict[str, JsonValue]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    serialized = json.loads(json.dumps(value))
    if not isinstance(serialized, dict):
        raise ValueError(f"{label} must be an object")
    _reject_private_keys(serialized, label)
    return serialized


def _reject_private_keys(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if (
                normalized in _PRIVATE_KEYS
                or normalized.startswith(_PRIVATE_KEY_PREFIXES)
                or normalized.endswith(_PRIVATE_KEY_SUFFIXES)
            ):
                raise ValueError(f"{label} contains a private field")
            _reject_private_keys(item, label)
    elif isinstance(value, list):
        for item in value:
            _reject_private_keys(item, label)
