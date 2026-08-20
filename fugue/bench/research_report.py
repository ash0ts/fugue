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
from urllib.parse import parse_qsl, quote, urlsplit

from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.research_index import (
    ResearchCandidateAssignmentV1,
    ResearchIndexPublicationReceiptV1,
    ResearchIndexPublicationTargetV1,
    ResearchIndexV1,
    read_research_index,
    read_research_index_publication_receipt,
)
from fugue.redaction import redact_text, sensitive_key
from fugue.research.records import (
    ResearchIndexReportPublicationEvidenceV1,
    ResearchIndexReportStudyMembershipV1,
    research_index_report_attempt_ids_digest,
    research_index_report_publication_evidence_from_dict,
    research_index_report_study_membership_from_dict,
)

RESEARCH_INDEX_REPORT_SCHEMA_VERSION = 1
RESEARCH_INDEX_REPORT_RENDERER_ID = "fugue-wandb-research-report"
RESEARCH_INDEX_REPORT_RENDERER_REVISION = "v1"
RESEARCH_INDEX_REPORT_API = "wandb-workspaces.reports.v2"
RESEARCH_INDEX_REPORT_API_VERSION = "0.4.5"
RESEARCH_INDEX_REPORT_API_STABILITY = "public_preview"
RESEARCH_INDEX_REPORT_WIDTH = "readable"
RESEARCH_INDEX_REPORT_WARNING = (
    "API status: Public Preview. This W&B Report is an optional presentation "
    "of the Fugue Research index. Use the local Research index as the authoritative "
    "content record. Use the index-publication receipt as the authoritative "
    "publication record. This Report lists summary fields from immutable Fugue "
    "ComparisonResultV3 artifacts. It does not create or replace first-class "
    "Study records."
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SCOPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OBJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,511}$")
_REPORT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,509}={0,2}$")
_STATUSES = frozenset(
    {"invalid", "incomplete", "improved", "regressed", "mixed", "unchanged"}
)
_GRADES = frozenset({"A", "B", "C", "invalid"})
_BACKENDS = frozenset({"local", "weave"})
_CHAINS = frozenset({"reconciled", "unresolved", "invalid", "not_applicable"})


class ResearchIndexReportError(ValueError):
    """A Research Report projection could not preserve its source contracts."""


@dataclass(frozen=True)
class ResearchIndexReportStudyV1:
    study_id: str
    comparison_id: str
    project: str
    result_digest: str
    qualification_digest: str
    behavioral_status: str
    behavioral_recommendation: str
    decision_status: str
    decision_recommendation: str
    task_validity_status: str
    rows: int
    evidence_integrity_grade: str
    evidence_backend: str
    local_chain_integrity: str
    result_hosted_chain_integrity: str
    published_chain_integrity: str
    candidate_assignments: tuple[ResearchCandidateAssignmentV1, ...]
    evidence_project_url: str
    primary_evidence_url: str
    study_digest: str = ""

    def __post_init__(self) -> None:
        _scope(self.study_id, "Study id")
        _safe_text(self.comparison_id, "comparison id", maximum=512)
        _project(self.project)
        _digest(self.result_digest, "result digest")
        _digest(self.qualification_digest, "qualification digest")
        if self.behavioral_status not in _STATUSES:
            raise ValueError("unsupported behavioral status")
        _safe_text(
            self.behavioral_recommendation,
            "behavioral recommendation",
            maximum=8_000,
        )
        _safe_text(self.decision_status, "decision status", maximum=100)
        _safe_text(
            self.decision_recommendation,
            "decision recommendation",
            maximum=8_000,
        )
        _safe_text(self.task_validity_status, "task validity", maximum=100)
        if (
            not isinstance(self.rows, int)
            or isinstance(self.rows, bool)
            or self.rows < 1
        ):
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
            not self.candidate_assignments
            or tuple(sorted(self.candidate_assignments, key=_assignment_key))
            != self.candidate_assignments
            or len({(item.harness, item.role) for item in self.candidate_assignments})
            != len(self.candidate_assignments)
        ):
            raise ValueError("candidate assignments must be nonempty and sorted")
        _project_url(self.evidence_project_url, self.project, suffix=("weave",))
        _project_url(
            self.primary_evidence_url,
            self.project,
            suffix=("weave", "calls"),
            allow_trailing_object=True,
        )
        computed = stable_digest(self._unsigned())
        if self.study_digest and self.study_digest != computed:
            raise ValueError("Report Study digest does not match")
        if not self.study_digest:
            object.__setattr__(self, "study_digest", computed)

    def _unsigned(self) -> dict[str, Any]:
        value = _json(asdict(self))
        value["candidate_assignments"] = [
            item.to_dict() for item in self.candidate_assignments
        ]
        value.pop("study_digest")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "study_digest": stable_digest(self._unsigned())}


@dataclass(frozen=True)
class ResearchIndexReportProjectionV1:
    research_id: str
    index_digest: str
    index_file_sha256: str
    index_publication_id: str
    index_publication_receipt_digest: str
    index_publication_receipt_file_sha256: str
    target: ResearchIndexPublicationTargetV1
    index_run_url: str
    index_artifact_url: str
    renderer_id: str
    renderer_revision: str
    report_api: str
    report_api_version: str
    api_stability: Literal["public_preview"]
    report_title: str
    report_description: str
    report_width: Literal["readable"]
    warning: str
    studies: tuple[ResearchIndexReportStudyV1, ...]
    study_count: int
    total_rows: int
    schema_version: Literal[1] = RESEARCH_INDEX_REPORT_SCHEMA_VERSION
    projection_digest: str = ""

    def __post_init__(self) -> None:
        _version(self.schema_version, "Research Report projection")
        _scope(self.research_id, "Research id")
        for label, value in (
            ("index digest", self.index_digest),
            ("index file digest", self.index_file_sha256),
            ("index publication id", self.index_publication_id),
            ("index publication receipt digest", self.index_publication_receipt_digest),
            (
                "index publication receipt file digest",
                self.index_publication_receipt_file_sha256,
            ),
        ):
            _digest(value, label)
        if not isinstance(self.target, ResearchIndexPublicationTargetV1):
            raise ValueError("invalid Research Report publication target")
        _target_object_url(self.index_run_url, self.target, resource="runs")
        _target_object_url(self.index_artifact_url, self.target, resource="artifacts")
        _safe_text(self.renderer_id, "renderer id", maximum=128, single_line=True)
        _safe_text(
            self.renderer_revision,
            "renderer revision",
            maximum=128,
            single_line=True,
        )
        _safe_text(self.report_api, "Report API", maximum=128, single_line=True)
        _safe_text(
            self.report_api_version,
            "Report API version",
            maximum=64,
            single_line=True,
        )
        if self.api_stability != "public_preview":
            raise ValueError("Research Reports require the Public Preview API marker")
        _safe_text(self.report_title, "Report title", maximum=128, single_line=True)
        _safe_text(self.report_description, "Report description", maximum=12_000)
        if self.report_width != "readable":
            raise ValueError("unsupported Research Report width")
        if self.warning != RESEARCH_INDEX_REPORT_WARNING:
            raise ValueError("Research Report lacks the mandatory warning")
        if self.warning not in self.report_description:
            raise ValueError("Research Report description lacks the mandatory warning")
        if (
            not self.studies
            or tuple(sorted(self.studies, key=lambda item: item.study_id))
            != self.studies
            or len({item.study_id for item in self.studies}) != len(self.studies)
        ):
            raise ValueError("Report Studies must be nonempty, unique, and sorted")
        if self.study_count != len(self.studies):
            raise ValueError("Report Study count does not match")
        if self.total_rows != sum(item.rows for item in self.studies):
            raise ValueError("Report row count does not match")
        for study in self.studies:
            _url_origin(study.evidence_project_url, self.target.app_base_url)
            _url_origin(study.primary_evidence_url, self.target.app_base_url)
        _digest(self.projection_digest, "Report projection digest")
        title_base = _title_base(self.report_title, self.projection_digest)
        computed = stable_digest(self._unsigned(title_base=title_base))
        if self.projection_digest != computed:
            raise ValueError("Research Report projection digest does not match")

    def _unsigned(self, *, title_base: str) -> dict[str, Any]:
        return {
            "research_id": self.research_id,
            "index_digest": self.index_digest,
            "index_file_sha256": self.index_file_sha256,
            "index_publication_id": self.index_publication_id,
            "index_publication_receipt_digest": (self.index_publication_receipt_digest),
            "index_publication_receipt_file_sha256": (
                self.index_publication_receipt_file_sha256
            ),
            "target": self.target.to_dict(),
            "index_run_url": self.index_run_url,
            "index_artifact_url": self.index_artifact_url,
            "renderer_id": self.renderer_id,
            "renderer_revision": self.renderer_revision,
            "report_api": self.report_api,
            "report_api_version": self.report_api_version,
            "api_stability": self.api_stability,
            "report_title": title_base,
            "report_description": self.report_description,
            "report_width": self.report_width,
            "warning": self.warning,
            "studies": [item.to_dict() for item in self.studies],
            "study_count": self.study_count,
            "total_rows": self.total_rows,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, Any]:
        title_base = _title_base(self.report_title, self.projection_digest)
        value = self._unsigned(title_base=title_base)
        value["report_title"] = self.report_title
        value["projection_digest"] = self.projection_digest
        return value


@dataclass(frozen=True)
class ResearchIndexReportPublicationOutcomeV1:
    target: ResearchIndexPublicationTargetV1
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

    def __post_init__(self) -> None:
        if not isinstance(self.target, ResearchIndexPublicationTargetV1):
            raise ValueError("invalid Research Report publication target")
        _report_id(self.report_id)
        _report_url(self.report_url, self.target, self.report_id)
        _safe_text(self.report_api, "Report API", maximum=128, single_line=True)
        _safe_text(
            self.report_api_version,
            "Report API version",
            maximum=64,
            single_line=True,
        )
        if self.api_stability != "public_preview":
            raise ValueError("unsupported Report API stability")
        _digest(self.readback_projection_digest, "readback projection digest")
        _digest(self.rendered_content_digest, "rendered Report content digest")
        if self.readback_status != "reconciled":
            raise ValueError("Research Report readback did not reconcile")
        _safe_text(self.publisher_id, "publisher id", maximum=128, single_line=True)
        _safe_text(
            self.publisher_revision,
            "publisher revision",
            maximum=256,
            single_line=True,
        )

    def to_dict(self) -> dict[str, Any]:
        value = _json(asdict(self))
        value["target"] = self.target.to_dict()
        return value


class ResearchIndexReportPublisher(Protocol):
    def __call__(
        self,
        projection: ResearchIndexReportProjectionV1,
        *,
        expected_report_id: str | None = None,
        expected_report_url: str | None = None,
    ) -> ResearchIndexReportPublicationOutcomeV1: ...


@dataclass(frozen=True)
class ResearchIndexReportPublicationReceiptV1:
    report_publication_id: str
    research_id: str
    projection_digest: str
    index_digest: str
    index_file_sha256: str
    index_publication_id: str
    index_publication_receipt_digest: str
    index_publication_receipt_file_sha256: str
    target: ResearchIndexPublicationTargetV1
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
    schema_version: Literal[1] = RESEARCH_INDEX_REPORT_SCHEMA_VERSION
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        _version(self.schema_version, "Research Report publication receipt")
        _scope(self.research_id, "Research id")
        for label, value in (
            ("Report publication id", self.report_publication_id),
            ("projection digest", self.projection_digest),
            ("index digest", self.index_digest),
            ("index file digest", self.index_file_sha256),
            ("index publication id", self.index_publication_id),
            ("index publication receipt digest", self.index_publication_receipt_digest),
            (
                "index publication receipt file digest",
                self.index_publication_receipt_file_sha256,
            ),
        ):
            _digest(value, label)
        if not isinstance(self.target, ResearchIndexPublicationTargetV1):
            raise ValueError("invalid Research Report publication target")
        _safe_text(self.renderer_id, "renderer id", maximum=128, single_line=True)
        _safe_text(
            self.renderer_revision,
            "renderer revision",
            maximum=128,
            single_line=True,
        )
        _report_id(self.report_id)
        _report_url(self.report_url, self.target, self.report_id)
        _safe_text(self.report_api, "Report API", maximum=128, single_line=True)
        _safe_text(
            self.report_api_version,
            "Report API version",
            maximum=64,
            single_line=True,
        )
        if self.api_stability != "public_preview":
            raise ValueError("unsupported Report API stability")
        _digest(self.readback_projection_digest, "readback projection digest")
        _digest(self.rendered_content_digest, "rendered Report content digest")
        if (
            self.readback_projection_digest != self.projection_digest
            or self.readback_status != "reconciled"
        ):
            raise ValueError("Research Report readback does not bind the projection")
        _safe_text(self.publisher_id, "publisher id", maximum=128, single_line=True)
        _safe_text(
            self.publisher_revision,
            "publisher revision",
            maximum=256,
            single_line=True,
        )
        if self.access_mode != "project_settings":
            raise ValueError(
                "Research Report access must follow W&B project and Report settings"
            )
        if self.share_link_action != "not_requested":
            raise ValueError("Fugue must not request a W&B Report share link")
        if self.current_pointer_status != "not_managed":
            raise ValueError("Research Report current pointers are not managed")
        if self.status != "published_and_reconciled":
            raise ValueError("unsupported Research Report publication status")
        _timestamp(self.published_at)
        expected_id = stable_digest(
            {
                "schema_version": self.schema_version,
                "research_id": self.research_id,
                "projection_digest": self.projection_digest,
                "target": self.target.to_dict(),
            }
        )
        if self.report_publication_id != expected_id:
            raise ValueError("Research Report publication identity does not match")
        computed = stable_digest(self._unsigned())
        if self.receipt_digest and self.receipt_digest != computed:
            raise ValueError("Research Report receipt digest does not match")
        if not self.receipt_digest:
            object.__setattr__(self, "receipt_digest", computed)

    def _unsigned(self) -> dict[str, Any]:
        value = _json(asdict(self))
        value["target"] = self.target.to_dict()
        value.pop("receipt_digest")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "receipt_digest": stable_digest(self._unsigned())}


def build_research_index_report_projection(
    index_path: Path,
    index_receipt_path: Path,
    *,
    renderer_id: str = RESEARCH_INDEX_REPORT_RENDERER_ID,
    renderer_revision: str = RESEARCH_INDEX_REPORT_RENDERER_REVISION,
    report_api: str = RESEARCH_INDEX_REPORT_API,
    report_api_version: str = RESEARCH_INDEX_REPORT_API_VERSION,
    api_stability: Literal["public_preview"] = RESEARCH_INDEX_REPORT_API_STABILITY,
    report_title: str | None = None,
    report_description: str | None = None,
    report_width: Literal["readable"] = RESEARCH_INDEX_REPORT_WIDTH,
    secret_values: Sequence[str] = (),
) -> ResearchIndexReportProjectionV1:
    """Build an optional Report view from immutable index artifacts."""

    index_path, index_bytes = _source(index_path, "Research index")
    receipt_path, receipt_bytes = _source(
        index_receipt_path,
        "Research-index publication receipt",
    )
    try:
        _secret_free(index_bytes, secret_values)
        _secret_free(receipt_bytes, secret_values)
        index = read_research_index(index_path)
        receipt = read_research_index_publication_receipt(receipt_path)
        index_file_sha = hashlib.sha256(index_bytes).hexdigest()
        receipt_file_sha = hashlib.sha256(receipt_bytes).hexdigest()
        _bind_index_receipt(index, index_file_sha, receipt)
        title_base = _bounded_title(report_title or f"Fugue Research · {index.title}")
        description = report_description or (
            f"{RESEARCH_INDEX_REPORT_WARNING}\n\n{index.objective}"
        )
        common = {
            "research_id": index.research_id,
            "index_digest": index.index_digest,
            "index_file_sha256": index_file_sha,
            "index_publication_id": receipt.publication_id,
            "index_publication_receipt_digest": receipt.receipt_digest,
            "index_publication_receipt_file_sha256": receipt_file_sha,
            "target": receipt.target,
            "index_run_url": receipt.run_url,
            "index_artifact_url": receipt.artifact_url,
            "renderer_id": renderer_id,
            "renderer_revision": renderer_revision,
            "report_api": report_api,
            "report_api_version": report_api_version,
            "api_stability": api_stability,
            "report_description": description,
            "report_width": report_width,
            "warning": RESEARCH_INDEX_REPORT_WARNING,
            "studies": tuple(
                _report_study(item, app_base_url=receipt.target.app_base_url)
                for item in index.studies
            ),
            "study_count": index.study_count,
            "total_rows": index.total_rows,
        }
        unsigned = {
            **common,
            "target": receipt.target.to_dict(),
            "report_title": title_base,
            "studies": [item.to_dict() for item in common["studies"]],
            "schema_version": 1,
        }
        projection_digest = stable_digest(unsigned)
        projection = ResearchIndexReportProjectionV1(
            **common,
            report_title=f"{title_base} · {projection_digest[:12]}",
            projection_digest=projection_digest,
        )
        _secret_free(_bytes(projection.to_dict()), secret_values)
        _unchanged(index_path, index_bytes)
        _unchanged(receipt_path, receipt_bytes)
        return projection
    except ResearchIndexReportError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ResearchIndexReportError(
            "Research Report projection failed canonical validation"
        ) from exc


def publish_research_index_report(
    index_path: Path,
    index_receipt_path: Path,
    publisher: ResearchIndexReportPublisher,
    *,
    receipt_path: Path | None = None,
    secret_values: Sequence[str] = (),
    clock: Callable[[], datetime] | None = None,
) -> ResearchIndexReportPublicationReceiptV1:
    """Publish the exact pure projection and write one immutable receipt."""

    index_path, index_before = _source(index_path, "Research index")
    index_receipt_path, receipt_before = _source(
        index_receipt_path,
        "Research-index publication receipt",
    )
    projection = build_research_index_report_projection(
        index_path,
        index_receipt_path,
        secret_values=secret_values,
    )
    selected_path = receipt_path or index_path.with_name(
        "research-index-report-publication-receipt.json"
    )
    if selected_path.is_symlink():
        raise ResearchIndexReportError("Report publication receipt cannot be a symlink")
    selected_path = selected_path.resolve()
    lock_path = selected_path.with_name(f".{selected_path.name}.lock")
    with FileLock(lock_path, timeout=120):
        if selected_path.exists():
            existing = read_research_index_report_publication_receipt(selected_path)
            _verify_report_receipt(existing, projection)
            try:
                outcome = publisher(
                    projection,
                    expected_report_id=existing.report_id,
                    expected_report_url=existing.report_url,
                )
            finally:
                _unchanged(index_path, index_before)
                _unchanged(index_receipt_path, receipt_before)
            if not isinstance(outcome, ResearchIndexReportPublicationOutcomeV1):
                raise ResearchIndexReportError(
                    "Report publisher returned an invalid readback outcome"
                )
            _verify_existing_report_readback(existing, outcome, projection)
            _unchanged(index_path, index_before)
            _unchanged(index_receipt_path, receipt_before)
            return existing
        try:
            outcome = publisher(projection)
        finally:
            _unchanged(index_path, index_before)
            _unchanged(index_receipt_path, receipt_before)
        if not isinstance(outcome, ResearchIndexReportPublicationOutcomeV1):
            raise ResearchIndexReportError(
                "Report publisher returned an invalid outcome"
            )
        _verify_outcome(outcome, projection)
        _secret_free(_bytes(outcome.to_dict()), secret_values)
        now = (clock or (lambda: datetime.now(UTC)))()
        if now.tzinfo is None:
            raise ResearchIndexReportError(
                "Report publication clock must be timezone-aware"
            )
        report_publication_id = stable_digest(
            {
                "schema_version": 1,
                "research_id": projection.research_id,
                "projection_digest": projection.projection_digest,
                "target": projection.target.to_dict(),
            }
        )
        receipt = ResearchIndexReportPublicationReceiptV1(
            report_publication_id=report_publication_id,
            research_id=projection.research_id,
            projection_digest=projection.projection_digest,
            index_digest=projection.index_digest,
            index_file_sha256=projection.index_file_sha256,
            index_publication_id=projection.index_publication_id,
            index_publication_receipt_digest=(
                projection.index_publication_receipt_digest
            ),
            index_publication_receipt_file_sha256=(
                projection.index_publication_receipt_file_sha256
            ),
            target=projection.target,
            renderer_id=projection.renderer_id,
            renderer_revision=projection.renderer_revision,
            report_id=outcome.report_id,
            report_url=outcome.report_url,
            report_api=outcome.report_api,
            report_api_version=outcome.report_api_version,
            api_stability=outcome.api_stability,
            readback_projection_digest=outcome.readback_projection_digest,
            rendered_content_digest=outcome.rendered_content_digest,
            readback_status=outcome.readback_status,
            publisher_id=outcome.publisher_id,
            publisher_revision=outcome.publisher_revision,
            access_mode="project_settings",
            share_link_action="not_requested",
            current_pointer_status="not_managed",
            status="published_and_reconciled",
            published_at=now.astimezone(UTC).isoformat(),
        )
        _secret_free(_bytes(receipt.to_dict()), secret_values)
        atomic_write_json(selected_path, receipt.to_dict())
        if read_research_index_report_publication_receipt(selected_path) != receipt:
            raise ResearchIndexReportError(
                "Research Report publication receipt did not round-trip"
            )
        _unchanged(index_path, index_before)
        _unchanged(index_receipt_path, receipt_before)
        return receipt


def read_research_index_report_publication_receipt(
    path: Path,
) -> ResearchIndexReportPublicationReceiptV1:
    try:
        return _read_research_index_report_publication_receipt(path)
    except ResearchIndexReportError:
        raise
    except (TypeError, ValueError) as exc:
        raise ResearchIndexReportError(
            "Research Report publication receipt failed canonical validation"
        ) from exc


def build_research_index_report_study_memberships(
    index_path: Path,
    index_receipt_path: Path,
    *,
    secret_values: Sequence[str] = (),
) -> tuple[str, tuple[ResearchIndexReportStudyMembershipV1, ...]]:
    """Read one immutable index and return its compact Research bindings."""

    index_path, index_bytes = _source(index_path, "Research index")
    index_receipt_path, index_receipt_bytes = _source(
        index_receipt_path,
        "Research-index publication receipt",
    )
    try:
        _secret_free(index_bytes, secret_values)
        _secret_free(index_receipt_bytes, secret_values)
        index = read_research_index(index_path)
        index_receipt = read_research_index_publication_receipt(index_receipt_path)
        _bind_index_receipt(
            index,
            hashlib.sha256(index_bytes).hexdigest(),
            index_receipt,
        )
        studies = tuple(_report_membership(item) for item in index.studies)
        _secret_free(
            _bytes([item.to_dict() for item in studies]),
            secret_values,
        )
        _unchanged(index_path, index_bytes)
        _unchanged(index_receipt_path, index_receipt_bytes)
        return index.research_id, studies
    except ResearchIndexReportError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ResearchIndexReportError(
            "Research Report Study memberships failed canonical validation"
        ) from exc


def verify_research_index_report_publication(
    index_path: Path,
    index_receipt_path: Path,
    report_receipt_path: Path,
    *,
    secret_values: Sequence[str] = (),
) -> ResearchIndexReportPublicationReceiptV1:
    """Verify a prior reconciled receipt without reading or rewriting W&B."""

    index_path, index_bytes = _source(index_path, "Research index")
    index_receipt_path, index_receipt_bytes = _source(
        index_receipt_path,
        "Research-index publication receipt",
    )
    report_receipt_path, report_receipt_bytes = _source(
        report_receipt_path,
        "Research Report publication receipt",
    )
    try:
        _secret_free(report_receipt_bytes, secret_values)
        projection = build_research_index_report_projection(
            index_path,
            index_receipt_path,
            secret_values=secret_values,
        )
        receipt = read_research_index_report_publication_receipt(report_receipt_path)
        _verify_report_receipt(receipt, projection)
        _unchanged(index_path, index_bytes)
        _unchanged(index_receipt_path, index_receipt_bytes)
        _unchanged(report_receipt_path, report_receipt_bytes)
        return receipt
    except ResearchIndexReportError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ResearchIndexReportError(
            "Research Report publication receipt failed local verification"
        ) from exc


def build_research_index_report_publication_evidence(
    index_path: Path,
    index_receipt_path: Path,
    report_receipt_path: Path,
    *,
    secret_values: Sequence[str] = (),
) -> ResearchIndexReportPublicationEvidenceV1:
    """Bind exact index and Report receipt bytes into public Research evidence."""

    index_path, index_bytes = _source(index_path, "Research index")
    index_receipt_path, index_receipt_bytes = _source(
        index_receipt_path,
        "Research-index publication receipt",
    )
    report_receipt_path, report_receipt_bytes = _source(
        report_receipt_path,
        "Research Report publication receipt",
    )
    try:
        for payload in (index_bytes, index_receipt_bytes, report_receipt_bytes):
            _secret_free(payload, secret_values)
        index = read_research_index(index_path)
        index_receipt = read_research_index_publication_receipt(index_receipt_path)
        index_file_sha = hashlib.sha256(index_bytes).hexdigest()
        index_receipt_file_sha = hashlib.sha256(index_receipt_bytes).hexdigest()
        report_receipt_file_sha = hashlib.sha256(report_receipt_bytes).hexdigest()
        _bind_index_receipt(index, index_file_sha, index_receipt)
        projection = build_research_index_report_projection(
            index_path,
            index_receipt_path,
            secret_values=secret_values,
        )
        report_receipt = read_research_index_report_publication_receipt(
            report_receipt_path
        )
        _verify_report_receipt(report_receipt, projection)
        index_source = index_receipt.to_dict()
        report_source = report_receipt.to_dict()
        evidence = research_index_report_publication_evidence_from_dict(
            {
                "schema_version": 1,
                "research_id": index.research_id,
                "target": index_receipt.target.to_dict(),
                "index": {
                    "publication_id": index_source["publication_id"],
                    "index_digest": index_source["index_digest"],
                    "index_file_sha256": index_source["index_file_sha256"],
                    "receipt_digest": index_source["receipt_digest"],
                    "receipt_file_sha256": index_receipt_file_sha,
                    "run_url": index_source["run_url"],
                    "artifact_url": index_source["artifact_url"],
                    "report_url": index_source["report_url"],
                    "report_status": index_source["report_status"],
                    "publisher_id": index_source["publisher_id"],
                    "publisher_revision": index_source["publisher_revision"],
                    "status": index_source["status"],
                    "published_at": index_source["published_at"],
                },
                "report": {
                    **{
                        key: report_source[key]
                        for key in (
                            "report_publication_id",
                            "projection_digest",
                            "index_digest",
                            "index_file_sha256",
                            "index_publication_id",
                            "index_publication_receipt_digest",
                            "index_publication_receipt_file_sha256",
                            "receipt_digest",
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
                        )
                    },
                    "receipt_file_sha256": report_receipt_file_sha,
                },
                "studies": [
                    _report_membership(item).to_dict() for item in index.studies
                ],
            }
        )
        _secret_free(_bytes(evidence.to_dict()), secret_values)
        _unchanged(index_path, index_bytes)
        _unchanged(index_receipt_path, index_receipt_bytes)
        _unchanged(report_receipt_path, report_receipt_bytes)
        return evidence
    except ResearchIndexReportError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ResearchIndexReportError(
            "Research index Report evidence failed canonical validation"
        ) from exc


def _read_research_index_report_publication_receipt(
    path: Path,
) -> ResearchIndexReportPublicationReceiptV1:
    raw = _read(path, "Research Report publication receipt")
    _fields(
        raw,
        "schema_version report_publication_id research_id projection_digest "
        "index_digest index_file_sha256 index_publication_id "
        "index_publication_receipt_digest index_publication_receipt_file_sha256 "
        "target renderer_id renderer_revision report_id report_url report_api "
        "report_api_version api_stability readback_projection_digest "
        "rendered_content_digest readback_status publisher_id publisher_revision "
        "status published_at "
        "access_mode share_link_action current_pointer_status receipt_digest",
    )
    return ResearchIndexReportPublicationReceiptV1(
        schema_version=_version(
            raw["schema_version"], "Research Report publication receipt"
        ),
        report_publication_id=str(raw["report_publication_id"]),
        research_id=str(raw["research_id"]),
        projection_digest=str(raw["projection_digest"]),
        index_digest=str(raw["index_digest"]),
        index_file_sha256=str(raw["index_file_sha256"]),
        index_publication_id=str(raw["index_publication_id"]),
        index_publication_receipt_digest=str(raw["index_publication_receipt_digest"]),
        index_publication_receipt_file_sha256=str(
            raw["index_publication_receipt_file_sha256"]
        ),
        target=_parse_target(raw["target"]),
        renderer_id=str(raw["renderer_id"]),
        renderer_revision=str(raw["renderer_revision"]),
        report_id=str(raw["report_id"]),
        report_url=str(raw["report_url"]),
        report_api=str(raw["report_api"]),
        report_api_version=str(raw["report_api_version"]),
        api_stability=cast(Literal["public_preview"], raw["api_stability"]),
        readback_projection_digest=str(raw["readback_projection_digest"]),
        rendered_content_digest=str(raw["rendered_content_digest"]),
        readback_status=cast(Literal["reconciled"], raw["readback_status"]),
        publisher_id=str(raw["publisher_id"]),
        publisher_revision=str(raw["publisher_revision"]),
        access_mode=cast(Literal["project_settings"], raw["access_mode"]),
        share_link_action=cast(Literal["not_requested"], raw["share_link_action"]),
        current_pointer_status=cast(
            Literal["not_managed"], raw["current_pointer_status"]
        ),
        status=cast(Literal["published_and_reconciled"], raw["status"]),
        published_at=str(raw["published_at"]),
        receipt_digest=str(raw["receipt_digest"]),
    )


def _report_study(
    study: Any,
    *,
    app_base_url: str,
) -> ResearchIndexReportStudyV1:
    primary = next(
        (item for item in study.evidence_refs if item.kind == "prediction_and_score"),
        None,
    )
    if primary is None:
        raise ResearchIndexReportError("Study lacks prediction-and-score evidence")
    call_id = primary.ref.rsplit("/", 1)[-1]
    _object_id(call_id, "prediction-and-score Call id")
    return ResearchIndexReportStudyV1(
        study_id=study.study_id,
        comparison_id=study.comparison_id,
        project=study.project,
        result_digest=study.result_digest,
        qualification_digest=study.qualification_digest,
        behavioral_status=study.behavioral_status,
        behavioral_recommendation=study.behavioral_recommendation,
        decision_status=study.decision_status,
        decision_recommendation=study.decision_recommendation,
        task_validity_status=study.task_validity_status,
        rows=study.rows,
        evidence_integrity_grade=study.evidence_integrity_grade,
        evidence_backend=study.evidence_backend,
        local_chain_integrity=study.local_chain_integrity,
        result_hosted_chain_integrity=study.result_hosted_chain_integrity,
        published_chain_integrity=study.published_chain_integrity,
        candidate_assignments=study.candidate_assignments,
        evidence_project_url=f"{app_base_url}/{study.project}/weave",
        primary_evidence_url=(
            f"{app_base_url}/{study.project}/weave/calls/{quote(call_id, safe='')}"
        ),
    )


def _report_membership(study: Any) -> ResearchIndexReportStudyMembershipV1:
    from fugue.bench.local_publication import weave_publication_receipt_from_dict

    try:
        raw_receipt = json.loads(study.publication_receipt_json)
        receipt = weave_publication_receipt_from_dict(_map(raw_receipt))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResearchIndexReportError(
            "Research index Study has an invalid embedded Weave receipt"
        ) from exc
    attempt_ids = tuple(
        sorted({str(item.attempt_id) for item in receipt.hosted_objects})
    )
    scope = receipt.target.study_scope
    if (
        scope is None
        or scope.study_id != study.study_id
        or receipt.result_digest != study.result_digest
        or receipt.qualification_digest != study.qualification_digest
        or receipt.result_file_sha256 != study.result_file_sha256
        or receipt.receipt_digest != study.publication_receipt_digest
        or receipt.target.project_slug != study.project
        or len(attempt_ids) != study.rows
    ):
        raise ResearchIndexReportError(
            "Research index Study membership disagrees with its Weave receipt"
        )
    return research_index_report_study_membership_from_dict(
        {
            "study_id": study.study_id,
            "result_digest": study.result_digest,
            "qualification_digest": study.qualification_digest,
            "result_file_sha256": study.result_file_sha256,
            "weave_project": study.project,
            "weave_publication_id": receipt.publication_id,
            "weave_receipt_digest": receipt.receipt_digest,
            "attempt_count": len(attempt_ids),
            "attempt_ids_digest": research_index_report_attempt_ids_digest(
                attempt_ids
            ),
        }
    )


def _bind_index_receipt(
    index: ResearchIndexV1,
    index_file_sha: str,
    receipt: ResearchIndexPublicationReceiptV1,
) -> None:
    if (
        receipt.research_id != index.research_id
        or receipt.index_digest != index.index_digest
        or receipt.index_file_sha256 != index_file_sha
        or receipt.status != "published"
    ):
        raise ResearchIndexReportError(
            "The index receipt does not match the selected Research index"
        )


def _verify_outcome(
    outcome: ResearchIndexReportPublicationOutcomeV1,
    projection: ResearchIndexReportProjectionV1,
) -> None:
    if (
        outcome.target != projection.target
        or outcome.report_api != projection.report_api
        or outcome.report_api_version != projection.report_api_version
        or outcome.api_stability != projection.api_stability
        or outcome.readback_projection_digest != projection.projection_digest
        or outcome.readback_status != "reconciled"
    ):
        raise ResearchIndexReportError(
            "Report publisher readback disagrees with the exact projection"
        )


def _verify_report_receipt(
    receipt: ResearchIndexReportPublicationReceiptV1,
    projection: ResearchIndexReportProjectionV1,
) -> None:
    expected_id = stable_digest(
        {
            "schema_version": 1,
            "research_id": projection.research_id,
            "projection_digest": projection.projection_digest,
            "target": projection.target.to_dict(),
        }
    )
    if (
        receipt.report_publication_id != expected_id
        or receipt.research_id != projection.research_id
        or receipt.projection_digest != projection.projection_digest
        or receipt.index_digest != projection.index_digest
        or receipt.index_file_sha256 != projection.index_file_sha256
        or receipt.index_publication_id != projection.index_publication_id
        or receipt.index_publication_receipt_digest
        != projection.index_publication_receipt_digest
        or receipt.index_publication_receipt_file_sha256
        != projection.index_publication_receipt_file_sha256
        or receipt.target != projection.target
        or receipt.renderer_id != projection.renderer_id
        or receipt.renderer_revision != projection.renderer_revision
        or receipt.report_api != projection.report_api
        or receipt.report_api_version != projection.report_api_version
        or receipt.api_stability != projection.api_stability
        or receipt.readback_projection_digest != projection.projection_digest
        or receipt.readback_status != "reconciled"
        or receipt.access_mode != "project_settings"
        or receipt.share_link_action != "not_requested"
        or receipt.current_pointer_status != "not_managed"
        or receipt.status != "published_and_reconciled"
    ):
        raise ResearchIndexReportError(
            "conflicting immutable Research Report publication receipt exists"
        )


def _verify_existing_report_readback(
    receipt: ResearchIndexReportPublicationReceiptV1,
    outcome: ResearchIndexReportPublicationOutcomeV1,
    projection: ResearchIndexReportProjectionV1,
) -> None:
    _verify_outcome(outcome, projection)
    if (
        outcome.report_id != receipt.report_id
        or outcome.report_url != receipt.report_url
        or outcome.report_api != receipt.report_api
        or outcome.report_api_version != receipt.report_api_version
        or outcome.api_stability != receipt.api_stability
        or outcome.rendered_content_digest != receipt.rendered_content_digest
        or outcome.publisher_id != receipt.publisher_id
        or outcome.publisher_revision != receipt.publisher_revision
    ):
        raise ResearchIndexReportError(
            "live Research Report readback disagrees with its publication receipt"
        )


def _parse_target(raw: Any) -> ResearchIndexPublicationTargetV1:
    value = dict(_map(raw))
    _fields(
        value,
        "schema_version backend project api_base_url app_base_url destination_digest",
    )
    return ResearchIndexPublicationTargetV1(
        schema_version=_version(value["schema_version"], "publication target"),
        backend=cast(Literal["wandb"], value["backend"]),
        project=str(value["project"]),
        api_base_url=str(value["api_base_url"]),
        app_base_url=str(value["app_base_url"]),
        destination_digest=str(value["destination_digest"]),
    )


def _source(path: Path, label: str) -> tuple[Path, bytes]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ResearchIndexReportError(f"{label} must be a regular file")
    return resolved, resolved.read_bytes()


def _unchanged(path: Path, before: bytes) -> None:
    if not path.is_file() or path.read_bytes() != before:
        raise ResearchIndexReportError(
            "The Research index or index receipt changed during Report publication"
        )


def _read(path: Path, label: str) -> dict[str, Any]:
    _, payload = _source(path, label)
    _secret_free(payload, ())
    try:
        return dict(_map(json.loads(payload)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResearchIndexReportError(f"{label} is not valid JSON") from exc


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode()


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _map(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("value must be an object")
    return value


def _fields(value: Mapping[str, Any], names: str) -> None:
    expected = set(names.split())
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        raise ValueError("invalid fields: " + "; ".join(details))


def _version(value: Any, label: str) -> Literal[1]:
    if value != 1 or isinstance(value, bool):
        raise ValueError(f"unsupported {label} schema")
    return 1


def _digest(value: str, label: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 digest")


def _scope(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SCOPE.fullmatch(value):
        raise ValueError(f"invalid {label}")


def _project(value: str) -> None:
    parts = value.split("/")
    if len(parts) != 2 or any(not _SLUG.fullmatch(item) for item in parts):
        raise ValueError("project must be exactly ENTITY/PROJECT")


def _object_id(value: str, label: str) -> None:
    if not isinstance(value, str) or not _OBJECT_ID.fullmatch(value):
        raise ValueError(f"invalid {label}")


def _report_id(value: str) -> None:
    if not isinstance(value, str) or not _REPORT_ID.fullmatch(value):
        raise ValueError("invalid Report id")


def _safe_text(
    value: str,
    label: str,
    *,
    maximum: int,
    single_line: bool = False,
) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
        or redact_text(value) != value
        or any(ord(char) < 32 and char not in "\n\t" for char in value)
        or (single_line and ("\n" in value or "\r" in value))
    ):
        raise ValueError(f"{label} is unsafe or invalid")
    lowered = value.casefold()
    if any(item in lowered for item in ("javascript:", "data:", "file:", "<script")):
        raise ValueError(f"{label} contains unsafe content")


def _safe_url(value: str) -> tuple[Any, list[str]]:
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
        or redact_text(value) != value
        or any(sensitive_key(key) for key, _ in parse_qsl(parsed.query))
    ):
        raise ValueError("unsafe Research Report URL")
    return parsed, _canonical_url_path(parsed.path)


def _canonical_url_path(path: str) -> list[str]:
    if (
        not path.startswith("/")
        or path.endswith("/")
        or "//" in path
        or "\\" in path
        or "%" in path
    ):
        raise ValueError("Research Report URL path is not canonical")
    parts = path[1:].split("/")
    for part in parts:
        if (
            not part
            or part in {".", ".."}
            or any(unicodedata.category(character) == "Cc" for character in part)
        ):
            raise ValueError("Research Report URL path is not canonical")
    return parts


def _target_object_url(
    value: str,
    target: ResearchIndexPublicationTargetV1,
    *,
    resource: str,
) -> None:
    parsed, parts = _safe_url(value)
    base = urlsplit(target.app_base_url)
    entity, project = target.project.split("/")
    if (
        (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc)
        or len(parts) != (6 if resource == "artifacts" else 4)
        or parts[:3] != [entity, project, resource]
        or (resource == "artifacts" and not re.fullmatch(r"v\d+", parts[-1]))
    ):
        raise ValueError("Research Report source URL disagrees with its target")


def _project_url(
    value: str,
    project: str,
    *,
    suffix: tuple[str, ...],
    allow_trailing_object: bool = False,
) -> None:
    _, parts = _safe_url(value)
    entity, project_id = project.split("/")
    required = [entity, project_id, *suffix]
    if parts[: len(required)] != required:
        raise ValueError("Study evidence URL disagrees with its project")
    trailing = len(parts) - len(required)
    if trailing != (1 if allow_trailing_object else 0):
        raise ValueError("Study evidence URL has an unexpected object path")
    if allow_trailing_object:
        _object_id(parts[-1], "evidence object id")


def _url_origin(value: str, app_base_url: str) -> None:
    parsed = urlsplit(value)
    expected = urlsplit(app_base_url)
    if (parsed.scheme, parsed.netloc) != (expected.scheme, expected.netloc):
        raise ValueError("Study evidence URL has another application origin")


def _report_url(
    value: str,
    target: ResearchIndexPublicationTargetV1,
    report_id: str,
) -> None:
    parsed, parts = _safe_url(value)
    base = urlsplit(target.app_base_url)
    entity, project = target.project.split("/")
    accepted_suffixes = {f"--{report_id}", f"--{report_id.rstrip('=')}"}
    if (
        (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc)
        or len(parts) != 4
        or parts[:3] != [entity, project, "reports"]
        or parts[3] in accepted_suffixes
        or not any(parts[3].endswith(suffix) for suffix in accepted_suffixes)
    ):
        raise ValueError("Research Report URL disagrees with the exact Report")


def _title_base(title: str, projection_digest: str) -> str:
    suffix = f" · {projection_digest[:12]}"
    if not title.endswith(suffix) or title == suffix:
        raise ValueError("Report title is not digest-addressed")
    return title[: -len(suffix)]


def _bounded_title(value: str) -> str:
    _safe_text(value, "Report title", maximum=128, single_line=True)
    normalized = "".join(
        character if character.isalnum() or character in " ._-" else " "
        for character in value
    )
    bounded = " ".join(normalized.split())[:112].rstrip(" ._-")
    if not bounded:
        raise ResearchIndexReportError("Report title has no safe content")
    return bounded


def _timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid Report publication timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("Report publication timestamp must include timezone")


def _assignment_key(item: ResearchCandidateAssignmentV1) -> tuple[str, str]:
    return item.harness, item.role


def _secret_free(payload: bytes, secrets: Sequence[str]) -> None:
    text = payload.decode(errors="replace")
    if any(len(value) >= 8 and value in text for value in secrets):
        raise ResearchIndexReportError(
            "The Report projection or publication receipt contains a secret value"
        )
    if redact_text(text) != text:
        raise ResearchIndexReportError(
            "Research Report artifact contains secret-shaped content"
        )
