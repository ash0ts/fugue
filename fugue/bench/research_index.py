from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from urllib.parse import parse_qsl, urlsplit

from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    ComparisonResultV3,
    comparison_result_from_json,
)
from fugue.bench.files import atomic_write_json
from fugue.bench.local_publication import (
    WeavePublicationReceiptV1,
    weave_publication_receipt_from_dict,
)
from fugue.redaction import redact_text, sensitive_key

RESEARCH_INDEX_SCHEMA_VERSION = 1
RESEARCH_INDEX_PUBLICATION_SCHEMA_VERSION = 1

BehavioralStatus = Literal[
    "invalid", "incomplete", "improved", "regressed", "mixed", "unchanged"
]
ChainStatus = Literal["reconciled", "unresolved", "invalid", "not_applicable"]
EvidenceKind = Literal[
    "evaluation_root",
    "prediction_and_score",
    "prediction",
    "agent_evidence_receipt",
    "dataset",
]
CandidateRole = Literal["baseline", "candidate"]
DecisionStatus = Literal[
    "invalid",
    "blocked",
    "hold",
    "inconclusive",
    "ready_for_signoff",
    "go",
]
TaskValidityStatus = Literal[
    "valid",
    "non_discriminating",
    "drifted",
    "invalid",
    "inconclusive",
]

_KINDS = frozenset(
    {
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_evidence_receipt",
        "dataset",
    }
)
_STATUSES = frozenset(
    {"invalid", "incomplete", "improved", "regressed", "mixed", "unchanged"}
)
_CHAINS = frozenset({"reconciled", "unresolved", "invalid", "not_applicable"})
_GRADES = frozenset({"A", "B", "C", "invalid"})
_BACKENDS = frozenset({"local", "weave"})
_DECISION_STATUSES = frozenset(
    {"invalid", "blocked", "hold", "inconclusive", "ready_for_signoff", "go"}
)
_TASK_VALIDITY_STATUSES = (
    "invalid",
    "drifted",
    "inconclusive",
    "non_discriminating",
    "valid",
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SCOPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OBJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,511}$")


class ResearchIndexError(ValueError):
    """A Research collection could not preserve its evidence contracts."""


@dataclass(frozen=True)
class ResearchIndexSourceV1:
    result_path: Path
    publication_receipt_path: Path


@dataclass(frozen=True)
class ResearchEvidenceRefV1:
    attempt_id: str
    kind: EvidenceKind
    ref: str
    system: Literal["weave"] = "weave"
    native_agent_call: Literal[False] = False

    def __post_init__(self) -> None:
        _digest(self.attempt_id, "attempt id")
        if self.kind not in _KINDS or self.system != "weave":
            raise ValueError("unsupported Research evidence reference")
        if self.native_agent_call is not False:
            raise ValueError("an Agent receipt is not a native Weave Agent Call")
        _safe_ref(self.ref)

    def to_dict(self) -> dict[str, Any]:
        return _json(asdict(self))


@dataclass(frozen=True)
class ResearchCandidateDefinitionV1:
    candidate_id: str
    definition_json: str

    def __post_init__(self) -> None:
        _digest(self.candidate_id, "candidate id")
        try:
            definition = json.loads(self.definition_json)
        except json.JSONDecodeError as exc:
            raise ValueError("candidate definition is not valid JSON") from exc
        if not isinstance(definition, Mapping):
            raise ValueError("candidate definition must be an object")
        canonical = _canonical_json(definition)
        if self.definition_json != canonical:
            raise ValueError("candidate definition JSON is not canonical")
        if stable_digest(definition) != self.candidate_id:
            raise ValueError("candidate definition disagrees with candidate id")

    @classmethod
    def from_definition(
        cls,
        candidate_id: str,
        definition: Mapping[str, Any],
    ) -> ResearchCandidateDefinitionV1:
        return cls(
            candidate_id=candidate_id,
            definition_json=_canonical_json(definition),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "definition": json.loads(self.definition_json),
        }


@dataclass(frozen=True)
class ResearchCandidateAssignmentV1:
    role: CandidateRole
    harness: str
    candidate_id: str

    def __post_init__(self) -> None:
        if self.role not in {"baseline", "candidate"}:
            raise ValueError("unsupported candidate treatment role")
        _scope(self.harness, "harness")
        _digest(self.candidate_id, "candidate id")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchStudyIndexEntryV1:
    research_id: str
    study_id: str
    comparison_id: str
    result_digest: str
    qualification_digest: str
    result_file_sha256: str
    publication_receipt_digest: str
    publication_receipt_file_sha256: str
    project: str
    behavioral_status: BehavioralStatus
    behavioral_recommendation: str
    decision_status: DecisionStatus
    decision_recommendation: str
    task_validity_status: TaskValidityStatus
    rows: int
    evidence_integrity_grade: Literal["A", "B", "C", "invalid"]
    evidence_backend: Literal["local", "weave"]
    local_chain_integrity: ChainStatus
    result_hosted_chain_integrity: ChainStatus
    published_chain_integrity: Literal["reconciled"]
    candidate_ids: tuple[str, ...]
    candidate_definitions: tuple[ResearchCandidateDefinitionV1, ...]
    candidate_assignments: tuple[ResearchCandidateAssignmentV1, ...]
    evidence_refs: tuple[ResearchEvidenceRefV1, ...]
    result_json: str
    publication_receipt_json: str

    def __post_init__(self) -> None:
        _scope(self.research_id, "Research id")
        _scope(self.study_id, "Study id")
        _text(self.comparison_id, "comparison id")
        for label, value in (
            ("result digest", self.result_digest),
            ("qualification digest", self.qualification_digest),
            ("result file digest", self.result_file_sha256),
            ("publication receipt digest", self.publication_receipt_digest),
            ("publication receipt file digest", self.publication_receipt_file_sha256),
        ):
            _digest(value, label)
        _project(self.project)
        if self.behavioral_status not in _STATUSES:
            raise ValueError("unsupported behavioral status")
        _text(self.behavioral_recommendation, "behavioral recommendation")
        if self.decision_status not in _DECISION_STATUSES:
            raise ValueError("unsupported decision status")
        _text(self.decision_recommendation, "decision recommendation")
        if self.task_validity_status not in _TASK_VALIDITY_STATUSES:
            raise ValueError("unsupported task validity status")
        if self.rows < 1:
            raise ValueError("Study rows must be positive")
        if self.evidence_integrity_grade not in _GRADES:
            raise ValueError("unsupported evidence grade")
        if self.evidence_backend not in _BACKENDS:
            raise ValueError("unsupported evidence backend")
        if (
            self.local_chain_integrity not in _CHAINS
            or self.result_hosted_chain_integrity not in _CHAINS
            or self.published_chain_integrity != "reconciled"
        ):
            raise ValueError("unsupported evidence-chain status")
        if (
            not self.candidate_ids
            or tuple(sorted(set(self.candidate_ids))) != self.candidate_ids
        ):
            raise ValueError("candidate ids must be nonempty, unique, and sorted")
        for value in self.candidate_ids:
            _digest(value, "candidate id")
        if (
            not self.candidate_definitions
            or tuple(
                sorted(self.candidate_definitions, key=lambda item: item.candidate_id)
            )
            != self.candidate_definitions
            or tuple(item.candidate_id for item in self.candidate_definitions)
            != self.candidate_ids
        ):
            raise ValueError(
                "candidate definitions must exactly match sorted candidate ids"
            )
        if (
            not self.candidate_assignments
            or tuple(sorted(self.candidate_assignments, key=_assignment_key))
            != self.candidate_assignments
            or len(
                {
                    (item.role, item.harness)
                    for item in self.candidate_assignments
                }
            )
            != len(self.candidate_assignments)
        ):
            raise ValueError(
                "candidate assignments must be nonempty, coordinate-unique, and sorted"
            )
        assigned_ids = {item.candidate_id for item in self.candidate_assignments}
        if assigned_ids != set(self.candidate_ids):
            raise ValueError(
                "candidate assignments must reference exactly the defined candidates"
            )
        if (
            not self.evidence_refs
            or tuple(sorted(self.evidence_refs, key=_ref_key)) != self.evidence_refs
            or len({(item.attempt_id, item.kind) for item in self.evidence_refs})
            != len(self.evidence_refs)
        ):
            raise ValueError("evidence refs must be nonempty, unique, and sorted")
        attempts = {item.attempt_id for item in self.evidence_refs}
        if len(attempts) != self.rows:
            raise ValueError("evidence refs do not match the Study row count")
        for attempt_id in attempts:
            kinds = {
                item.kind for item in self.evidence_refs if item.attempt_id == attempt_id
            }
            if kinds != _KINDS:
                raise ValueError("each Study row requires a complete five-object chain")
        for item in self.evidence_refs:
            object_type = "object" if item.kind == "dataset" else "call"
            prefix = f"weave:///{self.project}/{object_type}/"
            object_id = item.ref.removeprefix(prefix)
            if item.ref == object_id or not _OBJECT_ID.fullmatch(object_id):
                raise ValueError("evidence ref disagrees with the Study project")
        _validate_entry_sources(self)

    def to_dict(self) -> dict[str, Any]:
        value = _json(asdict(self))
        value["candidate_definitions"] = [
            item.to_dict() for item in self.candidate_definitions
        ]
        value["candidate_assignments"] = [
            item.to_dict() for item in self.candidate_assignments
        ]
        value["evidence_refs"] = [item.to_dict() for item in self.evidence_refs]
        return value


@dataclass(frozen=True)
class ResearchIndexV1:
    research_id: str
    title: str
    objective: str
    studies: tuple[ResearchStudyIndexEntryV1, ...]
    study_count: int
    total_rows: int
    schema_version: Literal[1] = RESEARCH_INDEX_SCHEMA_VERSION
    index_digest: str = ""

    def __post_init__(self) -> None:
        _version(self.schema_version, "Research index")
        _scope(self.research_id, "Research id")
        _text(self.title, "title")
        _text(self.objective, "objective")
        if (
            not self.studies
            or tuple(sorted(self.studies, key=lambda item: item.study_id))
            != self.studies
            or len({item.study_id for item in self.studies}) != len(self.studies)
        ):
            raise ValueError("Studies must be nonempty, unique, and sorted")
        if self.study_count != len(self.studies):
            raise ValueError("Study count does not match")
        if self.total_rows != sum(item.rows for item in self.studies):
            raise ValueError("row count does not match")
        if any(item.research_id != self.research_id for item in self.studies):
            raise ValueError("Study Research scope disagrees with the Research index")
        computed = stable_digest(self._unsigned())
        if self.index_digest and self.index_digest != computed:
            raise ValueError("Research index digest does not match")
        if not self.index_digest:
            object.__setattr__(self, "index_digest", computed)

    def _unsigned(self) -> dict[str, Any]:
        return {
            "research_id": self.research_id,
            "title": self.title,
            "objective": self.objective,
            "studies": [item.to_dict() for item in self.studies],
            "study_count": self.study_count,
            "total_rows": self.total_rows,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "index_digest": stable_digest(self._unsigned())}


@dataclass(frozen=True)
class ResearchIndexPublicationTargetV1:
    project: str
    api_base_url: str
    app_base_url: str
    backend: Literal["wandb"] = "wandb"
    schema_version: Literal[1] = RESEARCH_INDEX_PUBLICATION_SCHEMA_VERSION
    destination_digest: str = ""

    def __post_init__(self) -> None:
        _version(self.schema_version, "publication target")
        _project(self.project)
        _safe_origin(self.api_base_url, "W&B API origin")
        _safe_origin(self.app_base_url, "W&B application origin")
        if self.backend != "wandb":
            raise ValueError("unsupported Research-index publication backend")
        computed = stable_digest(self._unsigned())
        if self.destination_digest and self.destination_digest != computed:
            raise ValueError("publication destination digest does not match")
        if not self.destination_digest:
            object.__setattr__(self, "destination_digest", computed)

    def _unsigned(self) -> dict[str, Any]:
        value = _json(asdict(self))
        value.pop("destination_digest")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._unsigned(),
            "destination_digest": stable_digest(self._unsigned()),
        }


@dataclass(frozen=True)
class ResearchIndexPublicationOutcomeV1:
    target: ResearchIndexPublicationTargetV1
    run_url: str
    artifact_url: str
    report_url: str | None
    report_status: Literal["published", "unavailable"]
    publisher_id: str
    publisher_revision: str
    status: Literal["published"] = "published"

    def __post_init__(self) -> None:
        if not isinstance(self.target, ResearchIndexPublicationTargetV1):
            raise ValueError("invalid Research-index publication target")
        _target_url(
            self.run_url,
            self.target,
            "publication Run URL",
            resource="runs",
        )
        _target_url(
            self.artifact_url,
            self.target,
            "publication artifact URL",
            resource="artifacts",
        )
        if self.report_status == "published" and self.report_url is None:
            raise ValueError("published Research Report requires a URL")
        if self.report_status == "unavailable" and self.report_url is not None:
            raise ValueError("unavailable Research Report cannot have a URL")
        if self.report_status not in {"published", "unavailable"}:
            raise ValueError("unsupported Research Report status")
        if self.report_url is not None:
            _target_url(
                self.report_url,
                self.target,
                "publication Report URL",
                resource="reports",
            )
        _text(self.publisher_id, "publisher id")
        _text(self.publisher_revision, "publisher revision")
        if self.status != "published":
            raise ValueError("publication did not complete")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "run_url": self.run_url,
            "artifact_url": self.artifact_url,
            "report_url": self.report_url,
            "report_status": self.report_status,
            "publisher_id": self.publisher_id,
            "publisher_revision": self.publisher_revision,
            "status": self.status,
        }


class ResearchIndexPublisher(Protocol):
    def __call__(
        self,
        index: ResearchIndexV1,
        index_bytes: bytes,
        target: ResearchIndexPublicationTargetV1,
    ) -> ResearchIndexPublicationOutcomeV1: ...


@dataclass(frozen=True)
class ResearchIndexPublicationReceiptV1:
    publication_id: str
    research_id: str
    index_digest: str
    index_file_sha256: str
    target: ResearchIndexPublicationTargetV1
    run_url: str
    artifact_url: str
    report_url: str | None
    report_status: Literal["published", "unavailable"]
    publisher_id: str
    publisher_revision: str
    status: Literal["published"]
    published_at: str
    schema_version: Literal[1] = RESEARCH_INDEX_PUBLICATION_SCHEMA_VERSION
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        _version(self.schema_version, "publication receipt")
        for label, value in (
            ("publication id", self.publication_id),
            ("index digest", self.index_digest),
            ("index file digest", self.index_file_sha256),
        ):
            _digest(value, label)
        _scope(self.research_id, "Research id")
        if not isinstance(self.target, ResearchIndexPublicationTargetV1):
            raise ValueError("invalid Research-index publication target")
        _target_url(
            self.run_url,
            self.target,
            "publication Run URL",
            resource="runs",
        )
        _target_url(
            self.artifact_url,
            self.target,
            "publication artifact URL",
            resource="artifacts",
        )
        if self.report_status == "published" and self.report_url is None:
            raise ValueError("published Research Report requires a URL")
        if self.report_status == "unavailable" and self.report_url is not None:
            raise ValueError("unavailable Research Report cannot have a URL")
        if self.report_status not in {"published", "unavailable"}:
            raise ValueError("unsupported Research Report status")
        if self.report_url is not None:
            _target_url(
                self.report_url,
                self.target,
                "publication Report URL",
                resource="reports",
            )
        _text(self.publisher_id, "publisher id")
        _text(self.publisher_revision, "publisher revision")
        _timestamp(self.published_at)
        if self.status != "published":
            raise ValueError("unsupported publication status")
        expected_id = stable_digest(
            {
                "schema_version": self.schema_version,
                "research_id": self.research_id,
                "index_digest": self.index_digest,
                "index_file_sha256": self.index_file_sha256,
                "target": self.target.to_dict(),
            }
        )
        if self.publication_id != expected_id:
            raise ValueError("publication identity does not match")
        computed = stable_digest(self._unsigned())
        if self.receipt_digest and self.receipt_digest != computed:
            raise ValueError("publication receipt digest does not match")
        if not self.receipt_digest:
            object.__setattr__(self, "receipt_digest", computed)

    def _unsigned(self) -> dict[str, Any]:
        value = _json(asdict(self))
        value["target"] = self.target.to_dict()
        value.pop("receipt_digest")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "receipt_digest": stable_digest(self._unsigned())}


def build_research_index(
    *,
    research_id: str,
    title: str,
    objective: str,
    sources: Sequence[ResearchIndexSourceV1],
    secret_values: Sequence[str] = (),
) -> ResearchIndexV1:
    """Bind exact V3 result and Weave-receipt bytes into one collection."""

    if not sources:
        raise ResearchIndexError("Research index requires at least one source")
    _scope(research_id, "Research id")
    _secret_free(_bytes({"title": title, "objective": objective}), secret_values)
    entries: list[ResearchStudyIndexEntryV1] = []
    snapshots: list[tuple[Path, bytes]] = []
    for source in sources:
        if not isinstance(source, ResearchIndexSourceV1):
            raise ResearchIndexError("source must be ResearchIndexSourceV1")
        result_path = _file(source.result_path, "comparison result")
        receipt_path = _file(source.publication_receipt_path, "publication receipt")
        result_bytes, receipt_bytes = result_path.read_bytes(), receipt_path.read_bytes()
        snapshots.extend(((result_path, result_bytes), (receipt_path, receipt_bytes)))
        _secret_free(result_bytes, secret_values)
        _secret_free(receipt_bytes, secret_values)
        try:
            result = comparison_result_from_json(result_bytes)
            receipt_source = json.loads(receipt_bytes)
            receipt = weave_publication_receipt_from_dict(_map(receipt_source))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ResearchIndexError("source failed canonical validation") from exc
        if not isinstance(result, ComparisonResultV3):
            raise ResearchIndexError("Research indexes require ComparisonResultV3")
        entries.append(
            _entry(
                research_id,
                result,
                result_bytes,
                receipt,
                receipt_bytes,
            )
        )
    if len({item.study_id for item in entries}) != len(entries):
        raise ResearchIndexError("Research index contains duplicate Study ids")
    index = ResearchIndexV1(
        research_id=research_id,
        title=title,
        objective=objective,
        studies=tuple(sorted(entries, key=lambda item: item.study_id)),
        study_count=len(entries),
        total_rows=sum(item.rows for item in entries),
    )
    _secret_free(_bytes(index.to_dict()), secret_values)
    for path, before in snapshots:
        _unchanged(path, before)
    return index


def write_research_index(
    path: Path, index: ResearchIndexV1, *, secret_values: Sequence[str] = ()
) -> Path:
    expected = _bytes(index.to_dict())
    _secret_free(expected, secret_values)
    if path.is_symlink():
        raise ResearchIndexError("Research index cannot be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != expected:
            raise ResearchIndexError("conflicting immutable Research index exists")
        return path
    atomic_write_json(path, index.to_dict())
    if path.read_bytes() != expected or read_research_index(path) != index:
        raise ResearchIndexError("Research index did not round-trip")
    return path


def read_research_index(path: Path) -> ResearchIndexV1:
    raw = _read(path, "Research index")
    _fields(
        raw,
        "schema_version research_id title objective studies study_count total_rows "
        "index_digest",
    )
    return ResearchIndexV1(
        schema_version=_version(raw["schema_version"], "Research index"),
        research_id=str(raw["research_id"]),
        title=str(raw["title"]),
        objective=str(raw["objective"]),
        studies=tuple(_parse_entry(item) for item in _list(raw["studies"])),
        study_count=_int(raw["study_count"]),
        total_rows=_int(raw["total_rows"]),
        index_digest=str(raw["index_digest"]),
    )


def publish_research_index(
    index_path: Path,
    *,
    target: ResearchIndexPublicationTargetV1,
    publisher: ResearchIndexPublisher,
    receipt_path: Path | None = None,
    secret_values: Sequence[str] = (),
    clock: Callable[[], datetime] | None = None,
) -> ResearchIndexPublicationReceiptV1:
    """Publish unchanged index bytes through an injected optional adapter."""

    index_path = _file(index_path, "Research index")
    before = index_path.read_bytes()
    _secret_free(before, secret_values)
    index = read_research_index(index_path)
    if not isinstance(target, ResearchIndexPublicationTargetV1):
        raise ResearchIndexError("invalid Research-index publication target")
    file_digest = hashlib.sha256(before).hexdigest()
    selected_receipt_path = (
        receipt_path
        or index_path.with_name("research-index-publication-receipt.json")
    )
    if selected_receipt_path.is_symlink():
        raise ResearchIndexError("publication receipt cannot be a symlink")
    selected_receipt_path = selected_receipt_path.resolve()
    with FileLock(
        selected_receipt_path.with_name(f".{selected_receipt_path.name}.lock"),
        timeout=120,
    ):
        if selected_receipt_path.exists():
            receipt = read_research_index_publication_receipt(selected_receipt_path)
            _verify_receipt(receipt, index, file_digest, target)
            _unchanged(index_path, before)
            return receipt
        try:
            raw_outcome = publisher(index, before, target)
        finally:
            _unchanged(index_path, before)
        if not isinstance(raw_outcome, ResearchIndexPublicationOutcomeV1):
            raise ResearchIndexError("publisher returned an invalid outcome")
        outcome = raw_outcome
        if outcome.target != target:
            raise ResearchIndexError("publisher returned a different destination")
        _secret_free(_bytes(outcome.to_dict()), secret_values)
        now = (clock or (lambda: datetime.now(UTC)))()
        if now.tzinfo is None:
            raise ValueError("publication clock must be timezone-aware")
        publication_id = stable_digest(
            {
                "schema_version": 1,
                "research_id": index.research_id,
                "index_digest": index.index_digest,
                "index_file_sha256": file_digest,
                "target": target.to_dict(),
            }
        )
        receipt = ResearchIndexPublicationReceiptV1(
            publication_id=publication_id,
            research_id=index.research_id,
            index_digest=index.index_digest,
            index_file_sha256=file_digest,
            target=target,
            run_url=outcome.run_url,
            artifact_url=outcome.artifact_url,
            report_url=outcome.report_url,
            report_status=outcome.report_status,
            publisher_id=outcome.publisher_id,
            publisher_revision=outcome.publisher_revision,
            status="published",
            published_at=now.astimezone(UTC).isoformat(),
        )
        atomic_write_json(selected_receipt_path, receipt.to_dict())
        if read_research_index_publication_receipt(selected_receipt_path) != receipt:
            raise ResearchIndexError("publication receipt did not round-trip")
        _unchanged(index_path, before)
        return receipt


def read_research_index_publication_receipt(
    path: Path,
) -> ResearchIndexPublicationReceiptV1:
    raw = _read(path, "publication receipt")
    _fields(
        raw,
        "schema_version publication_id research_id index_digest index_file_sha256 "
        "target run_url artifact_url report_url report_status publisher_id "
        "publisher_revision status "
        "published_at receipt_digest",
    )
    version = _version(raw.pop("schema_version"), "publication receipt")
    raw["target"] = _parse_publication_target(raw["target"])
    return ResearchIndexPublicationReceiptV1(**raw, schema_version=version)


def _entry(
    research_id: str,
    result: ComparisonResultV3,
    result_bytes: bytes,
    receipt: WeavePublicationReceiptV1,
    receipt_bytes: bytes,
) -> ResearchStudyIndexEntryV1:
    return ResearchStudyIndexEntryV1(
        **_entry_projection(
            research_id,
            result,
            result_bytes,
            receipt,
            receipt_bytes,
        )
    )


def _entry_projection(
    research_id: str,
    result: ComparisonResultV3,
    result_bytes: bytes,
    receipt: WeavePublicationReceiptV1,
    receipt_bytes: bytes,
) -> dict[str, Any]:
    result_json = _json_source(result_bytes, "comparison result")
    receipt_json = _json_source(receipt_bytes, "publication receipt")
    result_sha = hashlib.sha256(result_bytes).hexdigest()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    scope = getattr(receipt.target, "study_scope", None)
    if scope is None or scope.research_id != research_id:
        raise ResearchIndexError("publication receipt has a different Research scope")
    if receipt.result_digest != result.result_digest:
        raise ResearchIndexError("publication receipt result digest disagrees")
    if receipt.qualification_digest != result.qualification_digest:
        raise ResearchIndexError("publication receipt qualification digest disagrees")
    if receipt.result_file_sha256 != result_sha:
        raise ResearchIndexError(
            "publication receipt does not bind exact result file bytes"
        )
    attempts = tuple(
        sorted(
            item.attempt_id
            for pair in result.paired_cases
            for item in (pair.baseline, pair.candidate)
            if item is not None
        )
    )
    if len(attempts) != len(set(attempts)) or len(attempts) != result.rows:
        raise ResearchIndexError("result rows disagree with unique attempts")
    expected = {(attempt, kind) for attempt in attempts for kind in _KINDS}
    observed = {(item.attempt_id, item.kind) for item in receipt.hosted_objects}
    if len(observed) != len(receipt.hosted_objects) or observed != expected:
        raise ResearchIndexError("receipt lacks a complete five-object chain")
    refs = tuple(
        sorted(
            (
                ResearchEvidenceRefV1(
                    attempt_id=item.attempt_id,
                    kind=cast(EvidenceKind, item.kind),
                    ref=item.ref,
                    system=item.system,
                    native_agent_call=item.native_agent_call,
                )
                for item in receipt.hosted_objects
            ),
            key=_ref_key,
        )
    )
    candidate_definitions = tuple(
        ResearchCandidateDefinitionV1.from_definition(candidate_id, definition)
        for candidate_id, definition in sorted(result.candidate_definitions.items())
    )
    assignments: dict[tuple[CandidateRole, str], str] = {}
    for pair in result.paired_cases:
        harness = str(pair.harness)
        _scope(harness, "harness")
        for role, attempt in (
            (cast(CandidateRole, "baseline"), pair.baseline),
            (cast(CandidateRole, "candidate"), pair.candidate),
        ):
            if attempt is None:
                continue
            identity = attempt.identity
            try:
                identity_harness = str(identity["harness"])
                candidate_id = str(identity["candidate"])
            except (KeyError, TypeError) as exc:
                raise ResearchIndexError(
                    "attempt lacks canonical candidate treatment identity"
                ) from exc
            if identity_harness != harness:
                raise ResearchIndexError(
                    "attempt harness disagrees with its paired case"
                )
            coordinate = (role, harness)
            previous = assignments.setdefault(coordinate, candidate_id)
            if previous != candidate_id:
                raise ResearchIndexError(
                    "candidate treatment coordinate maps to multiple candidates"
                )
    candidate_assignments = tuple(
        sorted(
            (
                ResearchCandidateAssignmentV1(
                    role=role,
                    harness=harness,
                    candidate_id=candidate_id,
                )
                for (role, harness), candidate_id in assignments.items()
            ),
            key=_assignment_key,
        )
    )
    return {
        "research_id": research_id,
        "study_id": scope.study_id,
        "comparison_id": result.comparison_id,
        "result_digest": result.result_digest,
        "qualification_digest": result.qualification_digest,
        "result_file_sha256": result_sha,
        "publication_receipt_digest": receipt.receipt_digest,
        "publication_receipt_file_sha256": receipt_sha,
        "project": receipt.target.project_slug,
        "behavioral_status": cast(
            BehavioralStatus, result.behavioral_summary.status
        ),
        "behavioral_recommendation": str(
            result.behavioral_summary.recommendation
        ),
        "decision_status": cast(DecisionStatus, result.decision.status),
        "decision_recommendation": str(result.decision.recommendation),
        "task_validity_status": _task_validity_status(result),
        "rows": result.rows,
        "evidence_integrity_grade": result.decision.evidence_grade,
        "evidence_backend": result.evidence_backend,
        "local_chain_integrity": result.local_chain_integrity,
        "result_hosted_chain_integrity": result.hosted_chain_integrity,
        "published_chain_integrity": "reconciled",
        "candidate_ids": tuple(sorted(result.candidate_definitions)),
        "candidate_definitions": candidate_definitions,
        "candidate_assignments": candidate_assignments,
        "evidence_refs": refs,
        "result_json": result_json,
        "publication_receipt_json": receipt_json,
    }


def _validate_entry_sources(entry: ResearchStudyIndexEntryV1) -> None:
    result_source = _bound_json_source(
        entry.result_json,
        entry.result_file_sha256,
        "comparison result",
    )
    receipt_source = _bound_json_source(
        entry.publication_receipt_json,
        entry.publication_receipt_file_sha256,
        "publication receipt",
    )
    _reject_sensitive_json_values(result_source, "embedded comparison result")
    _reject_sensitive_json_values(receipt_source, "embedded publication receipt")
    try:
        result = comparison_result_from_json(entry.result_json)
        receipt = weave_publication_receipt_from_dict(receipt_source)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("embedded source failed canonical validation") from exc
    if not isinstance(result, ComparisonResultV3):
        raise ValueError("embedded comparison result is not ComparisonResultV3")
    try:
        expected = _entry_projection(
            entry.research_id,
            result,
            entry.result_json.encode("utf-8"),
            receipt,
            entry.publication_receipt_json.encode("utf-8"),
        )
    except (ResearchIndexError, TypeError, ValueError) as exc:
        raise ValueError("embedded canonical sources disagree") from exc
    mismatched = sorted(
        key for key, value in expected.items() if getattr(entry, key) != value
    )
    if mismatched:
        raise ValueError(
            "Research index projection disagrees with canonical sources: "
            + ", ".join(mismatched)
        )


def _task_validity_status(result: ComparisonResultV3) -> TaskValidityStatus:
    statuses = {str(item.status) for item in result.task_validity}
    if not statuses:
        raise ResearchIndexError("comparison result has no task-validity evidence")
    unknown = statuses - set(_TASK_VALIDITY_STATUSES)
    if unknown:
        raise ResearchIndexError("comparison result has unsupported task validity")
    return cast(
        TaskValidityStatus,
        next(status for status in _TASK_VALIDITY_STATUSES if status in statuses),
    )


def _parse_entry(raw: Any) -> ResearchStudyIndexEntryV1:
    value = dict(_map(raw))
    _fields(
        value,
        "research_id study_id comparison_id result_digest qualification_digest "
        "result_file_sha256 publication_receipt_digest "
        "publication_receipt_file_sha256 project behavioral_status "
        "behavioral_recommendation decision_status decision_recommendation "
        "task_validity_status rows evidence_integrity_grade evidence_backend local_chain_integrity "
        "result_hosted_chain_integrity published_chain_integrity candidate_ids "
        "candidate_definitions candidate_assignments evidence_refs result_json "
        "publication_receipt_json",
    )
    value["candidate_ids"] = tuple(str(item) for item in _list(value["candidate_ids"]))
    value["candidate_definitions"] = tuple(
        _parse_candidate_definition(item)
        for item in _list(value["candidate_definitions"])
    )
    value["candidate_assignments"] = tuple(
        _parse_candidate_assignment(item)
        for item in _list(value["candidate_assignments"])
    )
    value["evidence_refs"] = tuple(
        _parse_ref(item) for item in _list(value["evidence_refs"])
    )
    value["rows"] = _int(value["rows"])
    return ResearchStudyIndexEntryV1(**value)


def _parse_candidate_definition(raw: Any) -> ResearchCandidateDefinitionV1:
    value = dict(_map(raw))
    _fields(value, "candidate_id definition")
    definition = _map(value["definition"])
    return ResearchCandidateDefinitionV1.from_definition(
        str(value["candidate_id"]),
        definition,
    )


def _parse_candidate_assignment(raw: Any) -> ResearchCandidateAssignmentV1:
    value = dict(_map(raw))
    _fields(value, "role harness candidate_id")
    return ResearchCandidateAssignmentV1(
        role=cast(CandidateRole, value["role"]),
        harness=str(value["harness"]),
        candidate_id=str(value["candidate_id"]),
    )


def _parse_ref(raw: Any) -> ResearchEvidenceRefV1:
    value = dict(_map(raw))
    _fields(value, "attempt_id kind ref system native_agent_call")
    return ResearchEvidenceRefV1(**value)


def _verify_receipt(
    receipt: ResearchIndexPublicationReceiptV1,
    index: ResearchIndexV1,
    file_digest: str,
    target: ResearchIndexPublicationTargetV1,
) -> None:
    expected = stable_digest(
        {
            "schema_version": 1,
            "research_id": index.research_id,
            "index_digest": index.index_digest,
            "index_file_sha256": file_digest,
            "target": target.to_dict(),
        }
    )
    if (
        receipt.publication_id != expected
        or receipt.index_digest != index.index_digest
        or receipt.index_file_sha256 != file_digest
        or receipt.target != target
    ):
        raise ResearchIndexError("conflicting immutable publication receipt exists")


def _parse_publication_target(raw: Any) -> ResearchIndexPublicationTargetV1:
    value = dict(_map(raw))
    _fields(
        value,
        "schema_version backend project api_base_url app_base_url "
        "destination_digest",
    )
    return ResearchIndexPublicationTargetV1(
        schema_version=_version(value["schema_version"], "publication target"),
        backend=cast(Literal["wandb"], value["backend"]),
        project=str(value["project"]),
        api_base_url=str(value["api_base_url"]),
        app_base_url=str(value["app_base_url"]),
        destination_digest=str(value["destination_digest"]),
    )


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _json_source(payload: bytes, label: str) -> str:
    try:
        text = payload.decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchIndexError(f"{label} bytes are not a UTF-8 JSON object") from exc
    if not isinstance(value, Mapping):
        raise ResearchIndexError(f"{label} bytes are not a UTF-8 JSON object")
    return text


def _bound_json_source(
    value: str,
    digest: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} source must be UTF-8 JSON text")
    try:
        payload = value.encode("utf-8")
        parsed = json.loads(value)
    except (UnicodeEncodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} source must be a UTF-8 JSON object") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} source must be a UTF-8 JSON object")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{label} source bytes disagree with its file digest")
    return cast(Mapping[str, Any], parsed)


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode()


def _file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ResearchIndexError(f"{label} must be a regular file")
    return resolved


def _unchanged(path: Path, before: bytes) -> None:
    if not path.is_file() or path.read_bytes() != before:
        raise ResearchIndexError("source changed while building the Research index")


def _secret_free(payload: bytes, secrets: Sequence[str]) -> None:
    text = payload.decode(errors="replace")
    if any(len(item) >= 8 and item in text for item in secrets):
        raise ResearchIndexError("artifact contains a configured secret value")
    if redact_text(text) != text:
        raise ResearchIndexError("artifact contains secret-shaped content")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return
    _reject_sensitive_json_values(parsed, "artifact")


def _reject_sensitive_json_values(value: Any, label: str, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if sensitive_key(key):
                raise ResearchIndexError(
                    f"{label} contains a value under sensitive key {child_path}"
                )
            _reject_sensitive_json_values(item, label, child_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive_json_values(item, label, f"{path}[{index}]")


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ResearchIndexError(f"{label} is not valid JSON") from exc
    _secret_free(payload, ())
    try:
        return dict(_map(json.loads(payload)))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ResearchIndexError(f"{label} is not valid JSON") from exc


def _map(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("value must be an object")
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("value must be an array")
    return value


def _fields(value: Mapping[str, Any], names: str) -> None:
    expected = set(names.split())
    unknown, missing = set(value) - expected, expected - set(value)
    if unknown or missing:
        raise ValueError(
            "invalid fields: "
            + "; ".join(
                item
                for item in (
                    "unknown " + ", ".join(sorted(unknown)) if unknown else "",
                    "missing " + ", ".join(sorted(missing)) if missing else "",
                )
                if item
            )
        )


def _version(value: Any, label: str) -> Literal[1]:
    if value != 1 or isinstance(value, bool):
        raise ValueError(f"unsupported {label} schema")
    return 1


def _int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("value must be an integer")
    return value


def _digest(value: str, label: str) -> None:
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 digest")


def _scope(value: str, label: str) -> None:
    if not _SCOPE.fullmatch(value):
        raise ValueError(f"invalid {label}")


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty")


def _project(value: str) -> None:
    parts = value.split("/")
    if len(parts) != 2 or any(not _SLUG.fullmatch(item) for item in parts):
        raise ValueError("project must be exactly ENTITY/PROJECT")


def _safe_ref(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "weave"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or redact_text(value) != value
    ):
        raise ValueError("unsafe Weave evidence ref")


def _safe_url(value: str) -> tuple[Any, list[str]]:
    if not isinstance(value, str):
        raise ValueError("unsafe publication URL")
    parsed = urlsplit(value)
    if (
        value != value.strip()
        or any(character.isspace() for character in value)
        or any(unicodedata.category(character) == "Cc" for character in value)
        or any(character in "\\[]()<>'\"`" for character in value)
        or parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(sensitive_key(key) for key, _ in parse_qsl(parsed.query))
        or redact_text(value) != value
    ):
        raise ValueError("unsafe publication URL")
    return parsed, _canonical_url_path(parsed.path)


def _safe_origin(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or any(character.isspace() for character in value)
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError(f"{label} must be a normalized safe HTTPS URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            f"{label} must be a normalized safe HTTPS URL"
        ) from exc
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or not hostname
        or parsed.username
        or parsed.password
        or "\\" in value
        or (port is not None and not 1 <= port <= 65535)
        or parsed.path
        or parsed.query
        or parsed.fragment
        or redact_text(value) != value
    ):
        raise ValueError(f"{label} must be a normalized safe HTTPS URL")


def _target_url(
    value: str,
    target: ResearchIndexPublicationTargetV1,
    label: str,
    *,
    resource: Literal["runs", "artifacts", "reports"],
) -> None:
    parsed, path = _safe_url(value)
    expected = urlsplit(target.app_base_url)
    entity, project = target.project.split("/")
    if (
        (parsed.scheme, parsed.netloc) != (expected.scheme, expected.netloc)
        or len(path) != (6 if resource == "artifacts" else 4)
        or path[:3] != [entity, project, resource]
        or (resource == "artifacts" and not re.fullmatch(r"v\d+", path[-1]))
    ):
        raise ValueError(f"{label} disagrees with the target project path")


def _canonical_url_path(path: str) -> list[str]:
    if (
        not path.startswith("/")
        or path.endswith("/")
        or "//" in path
        or "\\" in path
        or "%" in path
    ):
        raise ValueError("publication URL path is not canonical")
    parts = path[1:].split("/")
    for part in parts:
        if (
            not part
            or part in {".", ".."}
            or any(unicodedata.category(character) == "Cc" for character in part)
        ):
            raise ValueError("publication URL path is not canonical")
    return parts


def _timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid publication timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("publication timestamp must include timezone")


def _ref_key(item: ResearchEvidenceRefV1) -> tuple[str, str]:
    return item.attempt_id, item.kind


def _assignment_key(
    item: ResearchCandidateAssignmentV1,
) -> tuple[str, CandidateRole]:
    return item.harness, item.role
