from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import ComparisonResultV3
from fugue.model_plane import EvidenceDestinationV1, evidence_destination_from_dict
from fugue.redaction import redact_text

REPORT_SCHEMA_VERSION = 1

BehavioralStatus = Literal[
    "invalid", "incomplete", "improved", "regressed", "mixed", "unchanged"
]
TaskValidityStatus = Literal[
    "valid", "non_discriminating", "drifted", "invalid", "inconclusive"
]
ClaimCategory = Literal[
    "deterministic", "judge", "mechanism", "efficiency", "integrity"
]
ClaimStatus = Literal["supported", "blocked", "advisory", "descriptive", "unavailable"]
ScopeKind = Literal["study", "campaign"]
Visibility = Literal["private", "organization", "public"]
VisualAssetRole = Literal[
    "experiment_matrix",
    "paired_dimension_heatmap",
    "task_difference_plot",
    "behavior_integrity",
    "judge_mechanism_efficiency",
    "provenance_animation",
    "poster",
    "reduced_motion",
    "transcript",
    "source_hashes",
]

_BEHAVIORAL = {"invalid", "incomplete", "improved", "regressed", "mixed", "unchanged"}
_TASK_VALIDITY = {"valid", "non_discriminating", "drifted", "invalid", "inconclusive"}
_CLAIM_CATEGORIES = {"deterministic", "judge", "mechanism", "efficiency", "integrity"}
_CLAIM_STATUSES = {"supported", "blocked", "advisory", "descriptive", "unavailable"}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


class ScientificReportError(ValueError):
    """A strict failure in the public scientific-report chain."""


@dataclass(frozen=True)
class ClaimLedgerEntryV1:
    id: str
    category: ClaimCategory
    status: ClaimStatus
    statement: str
    canonical_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.id, "claim id")
        if self.category not in _CLAIM_CATEGORIES:
            raise ScientificReportError("unsupported claim category")
        if self.status not in _CLAIM_STATUSES:
            raise ScientificReportError("unsupported claim status")
        _public_text(self.statement, "claim statement", 2_000)
        _canonical_texts(self.canonical_refs, "claim references", require=True)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "canonical_refs": list(self.canonical_refs)}


@dataclass(frozen=True)
class VisualAssetV1:
    """One generated, claim-bound visual file in a report bundle."""

    id: str
    role: VisualAssetRole
    group_id: str
    path: str
    media_type: str
    sha256: str
    size_bytes: int
    alt_text: str
    claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.id, "visual asset id")
        _identifier(self.group_id, "visual asset group id")
        if self.role not in {
            "experiment_matrix",
            "paired_dimension_heatmap",
            "task_difference_plot",
            "behavior_integrity",
            "judge_mechanism_efficiency",
            "provenance_animation",
            "poster",
            "reduced_motion",
            "transcript",
            "source_hashes",
        }:
            raise ScientificReportError("unsupported visual asset role")
        _safe_bundle_path(self.path, "visual asset path", prefix="assets/")
        if self.media_type not in {
            "application/json",
            "image/png",
            "image/svg+xml",
            "text/markdown",
            "text/plain",
            "video/mp4",
        }:
            raise ScientificReportError("unsupported visual asset media type")
        _digest(self.sha256, "visual asset digest")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 1:
            raise ScientificReportError("visual asset size must be positive")
        _public_text(self.alt_text, "visual asset alt text", 2_000)
        _canonical_texts(self.claim_ids, "visual asset claim ids", require=True)
        for claim_id in self.claim_ids:
            _identifier(claim_id, "visual asset claim id")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "claim_ids": list(self.claim_ids)}


@dataclass(frozen=True)
class VisualAssetManifestV1:
    """Optional visual provenance bound to one immutable result and claim ledger."""

    schema_version: Literal[1]
    kind: Literal["visual_asset_manifest"]
    source_result_digest: str
    assets: tuple[VisualAssetV1, ...]
    manifest_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "visual_asset_manifest":
            raise ScientificReportError("unsupported visual asset manifest schema")
        _digest(self.source_result_digest, "visual source result digest")
        if not self.assets:
            raise ScientificReportError("visual asset manifest requires files")
        ids = tuple(item.id for item in self.assets)
        paths = tuple(item.path for item in self.assets)
        if ids != tuple(sorted(set(ids))):
            raise ScientificReportError("visual asset ids must be sorted and unique")
        if len(paths) != len(set(paths)):
            raise ScientificReportError("visual asset paths must be unique")
        by_group: dict[str, set[str]] = {}
        for item in self.assets:
            by_group.setdefault(item.group_id, set()).add(item.role)
        for item in self.assets:
            if item.role != "provenance_animation":
                continue
            companions = by_group[item.group_id]
            required = {"reduced_motion", "transcript", "source_hashes"}
            if not required.issubset(companions):
                raise ScientificReportError(
                    "provenance animation requires reduced-motion, transcript, "
                    "and source-hash companions"
                )
        computed = stable_digest(self.unsigned_dict())
        if self.manifest_digest and self.manifest_digest != computed:
            raise ScientificReportError("visual asset manifest digest does not match")
        if not self.manifest_digest:
            object.__setattr__(self, "manifest_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source_result_digest": self.source_result_digest,
            "assets": [item.to_dict() for item in self.assets],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "manifest_digest": self.manifest_digest}


@dataclass(frozen=True)
class ScientificReportV1:
    schema_version: Literal[1]
    kind: Literal["scientific_report"]
    comparison_id: str
    source_result_digest: str
    source_qualification_digest: str
    source_project: str
    result_project: str
    behavioral_status: BehavioralStatus
    task_validity_status: TaskValidityStatus
    evidence_grade: Literal["A", "B", "C", "invalid"]
    pair_count: int
    exact_revisions: tuple[str, ...]
    runtime_digests: tuple[str, ...]
    runtime_lock_digest: str
    baseline_source: ReportSourceIdentityV1
    candidate_source: ReportSourceIdentityV1
    supersedes: tuple[str, ...]
    narrative: dict[str, str]
    named_blockers: tuple[str, ...]
    claim_ledger: tuple[ClaimLedgerEntryV1, ...]
    limitations: tuple[str, ...]
    visual_assets: VisualAssetManifestV1 | None = None
    report_digest: str = ""

    def __post_init__(self) -> None:
        if (
            self.schema_version != REPORT_SCHEMA_VERSION
            or self.kind != "scientific_report"
        ):
            raise ScientificReportError("unsupported scientific report schema")
        _identifier(self.comparison_id, "comparison id")
        _digest(self.source_result_digest, "source result digest")
        _digest(self.source_qualification_digest, "source qualification digest")
        _project(self.source_project, "source project")
        _project(self.result_project, "result project")
        if self.source_project == self.result_project:
            raise ScientificReportError(
                "report publication must not target task inputs"
            )
        if self.behavioral_status not in _BEHAVIORAL:
            raise ScientificReportError("unsupported behavioral status")
        if self.task_validity_status not in _TASK_VALIDITY:
            raise ScientificReportError("unsupported task-validity status")
        if self.evidence_grade not in {"A", "B", "C", "invalid"}:
            raise ScientificReportError("unsupported evidence grade")
        if isinstance(self.pair_count, bool) or self.pair_count < 1:
            raise ScientificReportError("scientific report requires nonzero pairs")
        _canonical_texts(self.exact_revisions, "exact revisions", require=True)
        _canonical_texts(self.runtime_digests, "runtime digests", require=True)
        for digest in self.runtime_digests:
            _digest(digest.removeprefix("sha256:"), "runtime digest")
        _digest(self.runtime_lock_digest, "runtime lock digest")
        if (
            self.baseline_source.role != "baseline"
            or self.candidate_source.role != "candidate"
            or self.baseline_source.identity_digest
            == self.candidate_source.identity_digest
        ):
            raise ScientificReportError("scientific report source identities disagree")
        _canonical_texts(self.supersedes, "superseded result digests")
        for digest in self.supersedes:
            _digest(digest, "superseded result digest")
        if set(self.narrative) != {"what", "why", "how", "finding", "next_action"}:
            raise ScientificReportError(
                "scientific report narrative has the wrong fields"
            )
        for key, value in self.narrative.items():
            _public_text(value, f"narrative {key}", 4_000)
        if not self.claim_ledger:
            raise ScientificReportError("scientific report requires a claim ledger")
        _canonical_ids(self.claim_ledger, "claim")
        _canonical_texts(self.named_blockers, "named deterministic blockers")
        _canonical_texts(self.limitations, "limitations")
        if self.visual_assets is not None:
            if self.visual_assets.source_result_digest != self.source_result_digest:
                raise ScientificReportError(
                    "visual assets do not bind the report source result"
                )
            claim_ids = {item.id for item in self.claim_ledger}
            referenced = {
                claim_id
                for asset in self.visual_assets.assets
                for claim_id in asset.claim_ids
            }
            if not referenced.issubset(claim_ids):
                raise ScientificReportError(
                    "visual assets reference an unknown report claim"
                )
        computed = stable_digest(self.unsigned_dict())
        if self.report_digest and self.report_digest != computed:
            raise ScientificReportError("scientific report digest does not match")
        if not self.report_digest:
            object.__setattr__(self, "report_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "comparison_id": self.comparison_id,
            "source_result_digest": self.source_result_digest,
            "source_qualification_digest": self.source_qualification_digest,
            "source_project": self.source_project,
            "result_project": self.result_project,
            "behavioral_status": self.behavioral_status,
            "task_validity_status": self.task_validity_status,
            "evidence_grade": self.evidence_grade,
            "pair_count": self.pair_count,
            "exact_revisions": list(self.exact_revisions),
            "runtime_digests": list(self.runtime_digests),
            "runtime_lock_digest": self.runtime_lock_digest,
            "baseline_source": self.baseline_source.to_dict(),
            "candidate_source": self.candidate_source.to_dict(),
            "supersedes": list(self.supersedes),
            "narrative": dict(self.narrative),
            "named_blockers": list(self.named_blockers),
            "claim_ledger": [item.to_dict() for item in self.claim_ledger],
            "limitations": list(self.limitations),
        }
        if self.visual_assets is not None:
            value["visual_assets"] = self.visual_assets.to_dict()
        return value

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "report_digest": self.report_digest}


@dataclass(frozen=True)
class ReportLinkV1:
    schema_version: Literal[1]
    id: str
    kind: Literal["report", "artifact", "study", "article", "evidence"]
    label: str
    url: str
    content_sha256: str | None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ScientificReportError("unsupported report link schema")
        _identifier(self.id, "report link id")
        _public_text(self.label, "report link label", 1_000)
        _url(self.url, "report link URL")
        if self.content_sha256 is not None:
            _digest(self.content_sha256, "report link content digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportSourceIdentityV1:
    schema_version: Literal[1]
    role: Literal["baseline", "candidate"]
    kind: str
    id: str
    version_identity: str
    runtime_digest: str
    lock_digest: str | None
    identity_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.role not in {"baseline", "candidate"}:
            raise ScientificReportError("unsupported report source identity")
        for value, label in (
            (self.kind, "source kind"),
            (self.id, "source id"),
            (self.version_identity, "source version"),
        ):
            _public_text(value, label, 1_000)
        _digest(self.runtime_digest, "source runtime digest")
        if self.lock_digest is not None:
            _digest(self.lock_digest, "source lock digest")
        unsigned = self.unsigned_dict()
        computed = stable_digest(unsigned)
        if self.identity_digest and self.identity_digest != computed:
            raise ScientificReportError("report source identity digest does not match")
        if not self.identity_digest:
            object.__setattr__(self, "identity_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "id": self.id,
            "version_identity": self.version_identity,
            "runtime_digest": self.runtime_digest,
            "lock_digest": self.lock_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "kind": self.kind,
            "id": self.id,
            "version_identity": self.version_identity,
            "runtime_digest": self.runtime_digest,
            "lock_digest": self.lock_digest,
            "identity_digest": self.identity_digest,
        }


@dataclass(frozen=True)
class ReportSupersededResultV1:
    result_digest: str
    reason: str

    def __post_init__(self) -> None:
        _digest(self.result_digest, "superseded result digest")
        _public_text(self.reason, "supersession reason", 2_000)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyReportSummaryV1:
    schema_version: Literal[1]
    behavioral_status: BehavioralStatus
    task_validity_status: TaskValidityStatus
    evidence_grade: Literal["A", "B", "C", "invalid"]
    baseline_source: ReportSourceIdentityV1
    candidate_source: ReportSourceIdentityV1
    runtime_lock_digest: str
    pair_count: int
    result_digest: str
    scientific_report_digest: str
    visibility: Visibility
    supersedes: tuple[ReportSupersededResultV1, ...]
    summary_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ScientificReportError("unsupported Study report summary schema")
        if self.behavioral_status not in _BEHAVIORAL:
            raise ScientificReportError("Study behavioral status is invalid")
        if self.task_validity_status not in _TASK_VALIDITY:
            raise ScientificReportError("Study task-validity status is invalid")
        if self.evidence_grade not in {"A", "B", "C", "invalid"}:
            raise ScientificReportError("Study evidence grade is invalid")
        if self.baseline_source.role != "baseline" or self.candidate_source.role != "candidate":
            raise ScientificReportError("Study source roles disagree")
        if self.baseline_source.identity_digest == self.candidate_source.identity_digest:
            raise ScientificReportError("Study sources must differ")
        for value, label in (
            (self.runtime_lock_digest, "Study runtime lock digest"),
            (self.result_digest, "Study result digest"),
            (self.scientific_report_digest, "Study report digest"),
        ):
            _digest(value, label)
        if isinstance(self.pair_count, bool) or self.pair_count < 1:
            raise ScientificReportError("Study report summary requires pairs")
        if self.visibility not in {"private", "organization", "public"}:
            raise ScientificReportError("Study report visibility is invalid")
        superseded = tuple(item.result_digest for item in self.supersedes)
        if superseded != tuple(sorted(set(superseded))) or self.result_digest in superseded:
            raise ScientificReportError("Study superseded results are invalid")
        computed = stable_digest(self.unsigned_dict())
        if self.summary_digest and self.summary_digest != computed:
            raise ScientificReportError("Study report summary digest does not match")
        if not self.summary_digest:
            object.__setattr__(self, "summary_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "behavioral_status": self.behavioral_status,
            "task_validity_status": self.task_validity_status,
            "evidence_grade": self.evidence_grade,
            "baseline_source": self.baseline_source.to_dict(),
            "candidate_source": self.candidate_source.to_dict(),
            "runtime_lock_digest": self.runtime_lock_digest,
            "pair_count": self.pair_count,
            "result_digest": self.result_digest,
            "scientific_report_digest": self.scientific_report_digest,
            "visibility": self.visibility,
            "supersedes": [item.to_dict() for item in self.supersedes],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "summary_digest": self.summary_digest}


@dataclass(frozen=True)
class ReportPublicationReceiptV1:
    schema_version: Literal[1]
    scope_kind: ScopeKind
    scope_id: str
    document_kind: Literal["scientific_report", "campaign_report_index"]
    publication_id: str
    report_id: str
    report_sha256: str
    publication_bundle_digest: str
    artifact_manifest_digest: str
    source_artifact_digests: tuple[str, ...]
    task_source_projects: tuple[str, ...]
    publication_destination: EvidenceDestinationV1
    publication: ReportLinkV1
    related_links: tuple[ReportLinkV1, ...]
    artifact_ref: str
    artifact_version: str
    artifact_digest: str
    publication_run_id: str
    publication_run_kind: Literal["report_only"]
    published_at: str
    publisher_id: str
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.scope_kind not in {"study", "campaign"}:
            raise ScientificReportError("unsupported report publication receipt")
        _identifier(self.scope_id, "report scope id")
        if self.document_kind not in {"scientific_report", "campaign_report_index"}:
            raise ScientificReportError("unsupported report document kind")
        if (self.scope_kind == "study") != (self.document_kind == "scientific_report"):
            raise ScientificReportError("report scope and document kind disagree")
        _digest(self.publication_id, "report publication id")
        _identifier(self.report_id, "report id")
        _digest(self.report_sha256, "report content digest")
        _digest(self.publication_bundle_digest, "report publication bundle digest")
        _digest(self.artifact_manifest_digest, "report artifact manifest digest")
        _canonical_texts(self.source_artifact_digests, "report source digests", require=True)
        for value in self.source_artifact_digests:
            _digest(value, "report source digest")
        _canonical_texts(self.task_source_projects, "task source projects")
        for value in self.task_source_projects:
            _project(value, "task source project")
        if self.publication_project in self.task_source_projects:
            raise ScientificReportError("report publication entered task inputs")
        if (
            self.publication.id != self.report_id
            or self.publication.kind != "report"
            or self.publication.content_sha256 != self.report_sha256
        ):
            raise ScientificReportError("primary report link disagrees with receipt")
        related = tuple((item.id, item.url) for item in self.related_links)
        if related != tuple(sorted(set(related))):
            raise ScientificReportError("related report links must be sorted and unique")
        if not re.fullmatch(r"v[0-9]+", self.artifact_version):
            raise ScientificReportError("report artifact version is invalid")
        expected_ref_prefix = (
            f"wandb-artifact://{self.publication_project}/"
        )
        if (
            not self.artifact_ref.startswith(expected_ref_prefix)
            or not self.artifact_ref.endswith(f":{self.artifact_version}")
        ):
            raise ScientificReportError("report artifact reference is not exact")
        _public_text(self.artifact_digest, "report artifact digest", 500)
        artifact_links = [item for item in self.related_links if item.kind == "artifact"]
        expected_artifact_link_id = (
            f"{self.report_id}:artifact:{self.artifact_version}"
        )
        if len(artifact_links) != 1 or (
            artifact_links[0].id != expected_artifact_link_id
            or artifact_links[0].content_sha256 != self.artifact_manifest_digest
        ):
            raise ScientificReportError("report artifact link disagrees with receipt")
        _identifier(self.publication_run_id, "publication run id")
        if self.publication_run_kind != "report_only":
            raise ScientificReportError("report publication run must be report-only")
        _public_text(self.published_at, "publication timestamp", 100)
        _public_text(self.publisher_id, "publisher id", 200)
        expected_publication_id = stable_digest(self.publication_identity_dict())
        if self.publication_id != expected_publication_id:
            raise ScientificReportError("report publication id does not recompute")
        computed = stable_digest(self.unsigned_dict())
        if self.receipt_digest and self.receipt_digest != computed:
            raise ScientificReportError("publication receipt digest does not match")
        if not self.receipt_digest:
            object.__setattr__(self, "receipt_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope_kind": self.scope_kind,
            "scope_id": self.scope_id,
            "document_kind": self.document_kind,
            "publication_id": self.publication_id,
            "report_id": self.report_id,
            "report_sha256": self.report_sha256,
            "publication_bundle_digest": self.publication_bundle_digest,
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "source_artifact_digests": list(self.source_artifact_digests),
            "task_source_projects": list(self.task_source_projects),
            "publication_destination": self.publication_destination.to_dict(),
            "publication": self.publication.to_dict(),
            "related_links": [item.to_dict() for item in self.related_links],
            "artifact_ref": self.artifact_ref,
            "artifact_version": self.artifact_version,
            "artifact_digest": self.artifact_digest,
            "publication_run_id": self.publication_run_id,
            "publication_run_kind": self.publication_run_kind,
            "published_at": self.published_at,
            "publisher_id": self.publisher_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "receipt_digest": self.receipt_digest}

    @property
    def publication_project(self) -> str:
        return self.publication_destination.project_slug

    @property
    def report_url(self) -> str:
        return self.publication.url

    @property
    def artifact_url(self) -> str:
        return next(item.url for item in self.related_links if item.kind == "artifact")

    def publication_identity_dict(self) -> dict[str, Any]:
        """Return the preparation-bound identity known before remote side effects."""

        return {
            "schema_version": 2,
            "scope_kind": self.scope_kind,
            "scope_id": self.scope_id,
            "document_kind": self.document_kind,
            "document_digest": self.report_id,
            "content_sha256": self.report_sha256,
            "publication_bundle_digest": self.publication_bundle_digest,
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "source_artifact_digests": list(self.source_artifact_digests),
            "publication_destination": self.publication_destination.to_dict(),
            "task_source_projects": list(self.task_source_projects),
            "run_kind": "report_only",
            "excluded_from_task_inputs": True,
            "excluded_from_evaluation_counts": True,
        }


@dataclass(frozen=True)
class StudyReportIndexV1:
    schema_version: Literal[1]
    study_id: str
    result_project: str
    summary: StudyReportSummaryV1
    reports: tuple[ReportPublicationReceiptV1, ...]
    navigation_links: tuple[ReportLinkV1, ...]
    report_count: int
    report_only_run_ids: tuple[str, ...]
    index_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ScientificReportError("unsupported Study report index schema")
        _identifier(self.study_id, "Study id")
        _project(self.result_project, "Study result project")
        if len(self.reports) != 1:
            raise ScientificReportError("Study requires exactly one active report receipt")
        receipt = self.reports[0]
        if (
            receipt.scope_kind != "study"
            or receipt.scope_id != self.study_id
            or receipt.document_kind != "scientific_report"
            or receipt.publication_project != self.result_project
        ):
            raise ScientificReportError("Study report receipt disagrees with index")
        if self.report_count != 1:
            raise ScientificReportError("Study report count does not recompute")
        expected_runs = tuple(sorted(item.publication_run_id for item in self.reports))
        if self.report_only_run_ids != expected_runs or len(set(expected_runs)) != len(expected_runs):
            raise ScientificReportError("Study report-only Runs do not recompute")
        expected_sources = tuple(
            sorted((self.summary.result_digest, self.summary.scientific_report_digest))
        )
        if receipt.source_artifact_digests != expected_sources:
            raise ScientificReportError("Study report sources do not exactly recompute")
        if receipt.report_id != self.summary.scientific_report_digest:
            raise ScientificReportError("Study report identity does not recompute")
        navigation = tuple((item.id, item.url) for item in self.navigation_links)
        if len(self.navigation_links) != 3 or navigation != tuple(sorted(set(navigation))):
            raise ScientificReportError("Study navigation links are invalid")
        if {item.kind for item in self.navigation_links} != {"evidence", "study"}:
            raise ScientificReportError("Study navigation link kinds are invalid")
        computed = stable_digest(self.unsigned_dict())
        if self.index_digest and self.index_digest != computed:
            raise ScientificReportError("Study report index digest does not match")
        if not self.index_digest:
            object.__setattr__(self, "index_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "result_project": self.result_project,
            "summary": self.summary.to_dict(),
            "reports": [item.to_dict() for item in self.reports],
            "navigation_links": [item.to_dict() for item in self.navigation_links],
            "report_count": self.report_count,
            "report_only_run_ids": list(self.report_only_run_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "index_digest": self.index_digest}


@dataclass(frozen=True)
class CampaignStudyMembershipV1:
    study_id: str
    result_project: str

    def __post_init__(self) -> None:
        _identifier(self.study_id, "campaign Study id")
        _project(self.result_project, "campaign Study project")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignMembershipV1:
    schema_version: Literal[1]
    campaign_id: str
    studies: tuple[CampaignStudyMembershipV1, ...]
    membership_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ScientificReportError("unsupported campaign membership schema")
        _identifier(self.campaign_id, "campaign id")
        if not self.studies:
            raise ScientificReportError("campaign membership requires Studies")
        _canonical_ids(self.studies, "campaign membership")
        projects = tuple(item.result_project for item in self.studies)
        if len(set(projects)) != len(projects):
            raise ScientificReportError(
                "campaign membership result projects must be unique"
            )
        computed = stable_digest(self.unsigned_dict())
        if self.membership_digest and self.membership_digest != computed:
            raise ScientificReportError("campaign membership digest does not match")
        if not self.membership_digest:
            object.__setattr__(self, "membership_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "studies": [item.to_dict() for item in self.studies],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "membership_digest": self.membership_digest}


@dataclass(frozen=True)
class CampaignReportIndexV1:
    schema_version: Literal[1]
    campaign_id: str
    publication_destination: EvidenceDestinationV1
    membership: CampaignMembershipV1
    studies: tuple[StudyReportIndexV1, ...]
    study_count: int
    report_count: int
    result_projects: tuple[str, ...]
    report_only_run_ids: tuple[str, ...]
    complete: bool
    no_pooled_ranking: Literal[True] = True
    index_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.membership.campaign_id != self.campaign_id:
            raise ScientificReportError("unsupported campaign report index")
        _identifier(self.campaign_id, "campaign id")
        if not self.no_pooled_ranking:
            raise ScientificReportError("campaign report cannot pool Study outcomes")
        _project(self.publication_project, "campaign publication project")
        _canonical_ids(self.studies, "campaign Study")
        members = {item.study_id: item.result_project for item in self.membership.studies}
        if any(members.get(item.study_id) != item.result_project for item in self.studies):
            raise ScientificReportError("campaign Study disagrees with membership")
        if self.complete != ({item.study_id for item in self.studies} == set(members)):
            raise ScientificReportError("campaign completeness does not recompute")
        all_receipts = tuple(item for study in self.studies for item in study.reports)
        if self.study_count != len(self.studies) or self.report_count != len(all_receipts):
            raise ScientificReportError("campaign counts do not recompute")
        expected_projects = tuple(sorted(item.result_project for item in self.studies))
        if self.result_projects != expected_projects:
            raise ScientificReportError("campaign result projects do not recompute")
        expected_runs = tuple(sorted(item.publication_run_id for item in all_receipts))
        if self.report_only_run_ids != expected_runs or len(set(expected_runs)) != len(expected_runs):
            raise ScientificReportError("campaign report-only Runs do not recompute")
        source_projects = {
            project
            for receipt in all_receipts
            for project in receipt.task_source_projects
        }
        publication_projects = {
            self.publication_project,
            *(item.result_project for item in self.studies),
        }
        overlap = sorted(source_projects & publication_projects)
        if overlap:
            raise ScientificReportError(
                "campaign publication enters task-source project(s): "
                + ", ".join(overlap)
            )
        endpoint_identities = {
            (
                receipt.publication_destination.entity,
                receipt.publication_destination.api_base_url,
                receipt.publication_destination.trace_base_url,
                receipt.publication_destination.app_base_url,
            )
            for receipt in all_receipts
        }
        expected_endpoint = (
            self.publication_destination.entity,
            self.publication_destination.api_base_url,
            self.publication_destination.trace_base_url,
            self.publication_destination.app_base_url,
        )
        if endpoint_identities and endpoint_identities != {expected_endpoint}:
            raise ScientificReportError("campaign publication destinations disagree")
        computed = stable_digest(self.unsigned_dict())
        if self.index_digest and self.index_digest != computed:
            raise ScientificReportError("campaign report index digest does not match")
        if not self.index_digest:
            object.__setattr__(self, "index_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "publication_destination": self.publication_destination.to_dict(),
            "membership": self.membership.to_dict(),
            "studies": [item.to_dict() for item in self.studies],
            "study_count": self.study_count,
            "report_count": self.report_count,
            "result_projects": list(self.result_projects),
            "report_only_run_ids": list(self.report_only_run_ids),
            "complete": self.complete,
            "no_pooled_ranking": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "index_digest": self.index_digest}

    @property
    def publication_project(self) -> str:
        return self.publication_destination.project_slug


@dataclass(frozen=True)
class ArticlePublicationReceiptV1:
    """Append-only article publication evidence bound to a frozen campaign index.

    The article is deliberately downstream of the campaign index.  Attaching its
    URL to a Study or campaign report receipt would create a digest cycle and
    allow editorial publication to mutate scientific facts.
    """

    schema_version: Literal[1]
    kind: Literal["article_publication_receipt"]
    campaign_index_digest: str
    article_url: str
    article_source_sha256: str
    published_at: str
    publisher_id: str
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "article_publication_receipt":
            raise ScientificReportError("unsupported article publication receipt")
        _digest(self.campaign_index_digest, "article campaign index digest")
        _url(self.article_url, "article URL")
        _digest(self.article_source_sha256, "article source digest")
        _public_text(self.published_at, "article publication timestamp", 100)
        _public_text(self.publisher_id, "article publisher id", 200)
        computed = stable_digest(self.unsigned_dict())
        if self.receipt_digest and self.receipt_digest != computed:
            raise ScientificReportError("article publication receipt digest does not match")
        if not self.receipt_digest:
            object.__setattr__(self, "receipt_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "campaign_index_digest": self.campaign_index_digest,
            "article_url": self.article_url,
            "article_source_sha256": self.article_source_sha256,
            "published_at": self.published_at,
            "publisher_id": self.publisher_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "receipt_digest": self.receipt_digest}


def build_scientific_report(
    result: ComparisonResultV3,
    *,
    visual_assets: VisualAssetManifestV1 | None = None,
) -> ScientificReportV1:
    """Project one verified V3 result without copying attempt-level evidence."""

    if not isinstance(result, ComparisonResultV3):
        raise ScientificReportError("scientific reports require ComparisonResultV3")
    behavioral = result.behavioral_summary
    validity = _aggregate_task_validity([item.status for item in result.task_validity])
    integrity_ok = str(result.integrity.get("status") or "") == "reconciled"
    outcome_supported = (
        integrity_ok
        and validity == "valid"
        and result.decision.evidence_grade != "invalid"
        and behavioral.status not in {"invalid", "incomplete"}
    )
    claim = behavioral.supported_claim or (
        f"The candidate was {behavioral.status} across {len(result.paired_cases)} locked pairs."
    )
    task_count = len({pair.task_id for pair in result.paired_cases})
    attempt_count = len(result.paired_cases)
    revisions = _exact_revisions(result)
    named_blockers = _named_deterministic_blockers(result)
    narrative = {
        "what": (
            f"This Study compared {', '.join(revisions)} on "
            f"{task_count} locked tasks and {attempt_count} aligned paired attempts."
        ),
        "why": (
            "The comparison asks whether an exact public Skill revision changes "
            "task-relevant behavior while model, harness, task inputs, scoring, "
            "runtime, and evidence routing remain locked."
        ),
        "how": (
            "Fugue paired baseline and candidate attempts by canonical task, harness, "
            "and attempt identity. The task is the inference unit; repeated attempts "
            "are nested repeatability observations, not additional independent tasks. "
            "Deterministic outcomes remain authoritative while "
            "judge, mechanism, efficiency, and integrity evidence are reported "
            "separately."
        ),
        "finding": _sanitize(claim),
        "next_action": _sanitize(behavioral.next_action or result.decision.next_action),
    }
    claims = tuple(
        sorted(
            (
                ClaimLedgerEntryV1(
                    id="deterministic-outcome",
                    category="deterministic",
                    status="supported" if outcome_supported else "blocked",
                    statement=_sanitize(claim),
                    canonical_refs=(f"result:{result.result_digest}#paired_cases",),
                ),
                ClaimLedgerEntryV1(
                    id="deterministic-blockers",
                    category="deterministic",
                    status="blocked" if named_blockers else "supported",
                    statement=(
                        _blocker_statement(named_blockers)
                        if named_blockers
                        else "No named deterministic blocker was recorded."
                    ),
                    canonical_refs=(
                        f"result:{result.result_digest}#behavioral_summary.critical_blockers",
                        f"result:{result.result_digest}#task_validity",
                    ),
                ),
                ClaimLedgerEntryV1(
                    id="evidence-integrity",
                    category="integrity",
                    status="supported" if integrity_ok else "blocked",
                    statement=(
                        "Required evidence links and result rows reconciled."
                        if integrity_ok
                        else "Required evidence integrity did not reconcile."
                    ),
                    canonical_refs=(f"result:{result.result_digest}#integrity",),
                ),
                _section_claim(result, "judge"),
                _section_claim(result, "mechanism"),
                _section_claim(result, "efficiency"),
            ),
            key=lambda item: item.id,
        )
    )
    limitations = _safe_unique(
        (
            *result.limitations,
            *behavioral.limitations,
            *result.decision.limitations,
            "The task, not an individual attempt, is the inference unit; repeated "
            "attempts characterize repeatability but do not create a population sample.",
            "This repository case study is not pooled with other repositories and "
            "does not support a universal Skill ranking.",
        )
    )
    return ScientificReportV1(
        schema_version=1,
        kind="scientific_report",
        comparison_id=result.comparison_id,
        source_result_digest=result.result_digest,
        source_qualification_digest=result.qualification_digest,
        source_project=result.evidence_topology.source_destination.project_slug,
        result_project=result.evidence_topology.result_destination.project_slug,
        behavioral_status=behavioral.status,
        task_validity_status=validity,
        evidence_grade=result.decision.evidence_grade,
        pair_count=len(result.paired_cases),
        exact_revisions=revisions,
        runtime_digests=tuple(
            sorted({item.digest for item in result.runtime_locks})
        ),
        runtime_lock_digest=stable_digest(
            [
                item.to_dict()
                if hasattr(item, "to_dict")
                else dict(vars(item))
                for item in result.runtime_locks
            ]
        ),
        baseline_source=_report_source_identity(result, role="baseline"),
        candidate_source=_report_source_identity(result, role="candidate"),
        supersedes=tuple(sorted(item.result_digest for item in result.supersedes)),
        narrative=narrative,
        named_blockers=named_blockers,
        claim_ledger=claims,
        limitations=limitations,
        visual_assets=visual_assets,
    )


def build_study_report_index(
    report: ScientificReportV1,
    publication: ReportPublicationReceiptV1,
    *,
    result_url: str,
    study_console_url: str,
    weave_url: str,
    visibility: Visibility = "private",
) -> StudyReportIndexV1:
    accepted_publication = publication_receipt_from_dict(publication.to_dict())
    expected_report_sha256 = hashlib.sha256(
        (json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    expected_sources = tuple(
        sorted((report.source_result_digest, report.report_digest))
    )
    if (
        accepted_publication.scope_kind != "study"
        or accepted_publication.scope_id != report.comparison_id
        or accepted_publication.document_kind != "scientific_report"
        or accepted_publication.report_id != report.report_digest
        or accepted_publication.report_sha256 != expected_report_sha256
        or accepted_publication.source_artifact_digests != expected_sources
        or accepted_publication.task_source_projects != (report.source_project,)
        or accepted_publication.publication_project != report.result_project
    ):
        raise ScientificReportError(
            "Study publication receipt does not exactly bind its scientific report"
        )
    summary = StudyReportSummaryV1(
        schema_version=1,
        behavioral_status=report.behavioral_status,
        task_validity_status=report.task_validity_status,
        evidence_grade=report.evidence_grade,
        baseline_source=report.baseline_source,
        candidate_source=report.candidate_source,
        runtime_lock_digest=report.runtime_lock_digest,
        pair_count=report.pair_count,
        result_digest=report.source_result_digest,
        scientific_report_digest=report.report_digest,
        visibility=visibility,
        supersedes=tuple(
            ReportSupersededResultV1(
                result_digest=digest,
                reason="Superseded by this canonical scientific result.",
            )
            for digest in report.supersedes
        ),
    )
    return StudyReportIndexV1(
        schema_version=1,
        study_id=report.comparison_id,
        result_project=report.result_project,
        summary=summary,
        reports=(accepted_publication,),
        navigation_links=tuple(
            sorted(
                (
                    ReportLinkV1(
                        schema_version=1,
                        id=report.source_result_digest,
                        kind="evidence",
                        label="Canonical result",
                        url=result_url,
                        content_sha256=report.source_result_digest,
                    ),
                    ReportLinkV1(
                        schema_version=1,
                        id=report.comparison_id,
                        kind="study",
                        label="Study Console",
                        url=study_console_url,
                        content_sha256=None,
                    ),
                    ReportLinkV1(
                        schema_version=1,
                        id=f"{report.comparison_id}:weave",
                        kind="evidence",
                        label="Weave evidence",
                        url=weave_url,
                        content_sha256=None,
                    ),
                ),
                key=lambda item: (item.id, item.url),
            )
        ),
        report_count=1,
        report_only_run_ids=(publication.publication_run_id,),
    )


def build_campaign_report_index(
    campaign_id: str,
    publication_destination: EvidenceDestinationV1,
    membership: CampaignMembershipV1,
    studies: Sequence[StudyReportIndexV1],
) -> CampaignReportIndexV1:
    accepted_membership = campaign_membership_from_dict(membership.to_dict())
    if accepted_membership.campaign_id != campaign_id:
        raise ScientificReportError("campaign membership targets another campaign")
    ordered = tuple(sorted(studies, key=lambda item: item.study_id))
    member_projects = {
        item.study_id: item.result_project for item in accepted_membership.studies
    }
    unexpected = sorted(
        item.study_id
        for item in ordered
        if member_projects.get(item.study_id) != item.result_project
    )
    if unexpected:
        raise ScientificReportError(
            "campaign contains unexpected Study identity: " + ", ".join(unexpected)
        )
    nested_receipts = tuple(
        receipt for study in ordered for receipt in study.reports
    )
    return CampaignReportIndexV1(
        schema_version=1,
        campaign_id=campaign_id,
        publication_destination=publication_destination,
        membership=accepted_membership,
        studies=ordered,
        study_count=len(ordered),
        report_count=len(nested_receipts),
        result_projects=tuple(sorted({item.result_project for item in ordered})),
        report_only_run_ids=tuple(
            sorted(item.publication_run_id for item in nested_receipts)
        ),
        complete={item.study_id for item in ordered} == set(member_projects),
    )


def scientific_report_from_dict(raw: Mapping[str, Any]) -> ScientificReportV1:
    _fields(raw, ScientificReportV1, "scientific report", optional={"visual_assets"})
    value = dict(raw)
    value["claim_ledger"] = tuple(
        claim_ledger_entry_from_dict(_mapping(item, "claim"))
        for item in _sequence(value.get("claim_ledger"), "claim ledger")
    )
    value["limitations"] = tuple(_sequence(value.get("limitations"), "limitations"))
    value["exact_revisions"] = tuple(
        _sequence(value.get("exact_revisions"), "exact revisions")
    )
    value["runtime_digests"] = tuple(
        _sequence(value.get("runtime_digests"), "runtime digests")
    )
    value["baseline_source"] = report_source_identity_from_dict(
        _mapping(value.get("baseline_source"), "baseline source")
    )
    value["candidate_source"] = report_source_identity_from_dict(
        _mapping(value.get("candidate_source"), "candidate source")
    )
    value["supersedes"] = tuple(_sequence(value.get("supersedes"), "supersedes"))
    value["named_blockers"] = tuple(
        _sequence(value.get("named_blockers"), "named blockers")
    )
    value["narrative"] = dict(_mapping(value.get("narrative"), "narrative"))
    if value.get("visual_assets") is not None:
        value["visual_assets"] = visual_asset_manifest_from_dict(
            _mapping(value["visual_assets"], "visual asset manifest")
        )
    else:
        value["visual_assets"] = None
    return ScientificReportV1(**value)


def visual_asset_from_dict(raw: Mapping[str, Any]) -> VisualAssetV1:
    _fields(raw, VisualAssetV1, "visual asset")
    value = dict(raw)
    value["claim_ids"] = tuple(
        _sequence(value.get("claim_ids"), "visual asset claim ids")
    )
    return VisualAssetV1(**value)


def visual_asset_manifest_from_dict(
    raw: Mapping[str, Any],
) -> VisualAssetManifestV1:
    _fields(raw, VisualAssetManifestV1, "visual asset manifest")
    value = dict(raw)
    value["assets"] = tuple(
        visual_asset_from_dict(_mapping(item, "visual asset"))
        for item in _sequence(value.get("assets"), "visual assets")
    )
    return VisualAssetManifestV1(**value)


def claim_ledger_entry_from_dict(raw: Mapping[str, Any]) -> ClaimLedgerEntryV1:
    _fields(raw, ClaimLedgerEntryV1, "claim")
    value = dict(raw)
    value["canonical_refs"] = tuple(
        _sequence(value.get("canonical_refs"), "claim refs")
    )
    return ClaimLedgerEntryV1(**value)


def publication_receipt_from_dict(raw: Mapping[str, Any]) -> ReportPublicationReceiptV1:
    _fields(raw, ReportPublicationReceiptV1, "publication receipt")
    value = dict(raw)
    value["source_artifact_digests"] = tuple(
        _sequence(value.get("source_artifact_digests"), "report source digests")
    )
    value["task_source_projects"] = tuple(
        _sequence(value.get("task_source_projects"), "report task source projects")
    )
    try:
        value["publication_destination"] = evidence_destination_from_dict(
            _mapping(value.get("publication_destination"), "publication destination")
        )
    except ValueError as exc:
        raise ScientificReportError(str(exc)) from exc
    value["publication"] = report_link_from_dict(
        _mapping(value.get("publication"), "report publication link")
    )
    value["related_links"] = tuple(
        report_link_from_dict(_mapping(item, "related report link"))
        for item in _sequence(value.get("related_links"), "related report links")
    )
    return ReportPublicationReceiptV1(**value)


def report_link_from_dict(raw: Mapping[str, Any]) -> ReportLinkV1:
    _fields(raw, ReportLinkV1, "report link")
    return ReportLinkV1(**dict(raw))


def report_source_identity_from_dict(
    raw: Mapping[str, Any],
) -> ReportSourceIdentityV1:
    _fields(raw, ReportSourceIdentityV1, "report source identity")
    return ReportSourceIdentityV1(**dict(raw))


def report_superseded_result_from_dict(
    raw: Mapping[str, Any],
) -> ReportSupersededResultV1:
    _fields(raw, ReportSupersededResultV1, "report superseded result")
    return ReportSupersededResultV1(**dict(raw))


def study_report_summary_from_dict(raw: Mapping[str, Any]) -> StudyReportSummaryV1:
    _fields(raw, StudyReportSummaryV1, "Study report summary")
    value = dict(raw)
    value["baseline_source"] = report_source_identity_from_dict(
        _mapping(value.get("baseline_source"), "baseline source")
    )
    value["candidate_source"] = report_source_identity_from_dict(
        _mapping(value.get("candidate_source"), "candidate source")
    )
    value["supersedes"] = tuple(
        report_superseded_result_from_dict(_mapping(item, "superseded result"))
        for item in _sequence(value.get("supersedes"), "superseded results")
    )
    return StudyReportSummaryV1(**value)


def study_report_index_from_dict(raw: Mapping[str, Any]) -> StudyReportIndexV1:
    _fields(raw, StudyReportIndexV1, "Study report index")
    value = dict(raw)
    value["summary"] = study_report_summary_from_dict(
        _mapping(value.get("summary"), "Study summary")
    )
    value["reports"] = tuple(
        publication_receipt_from_dict(_mapping(item, "Study report receipt"))
        for item in _sequence(value.get("reports"), "Study report receipts")
    )
    value["navigation_links"] = tuple(
        report_link_from_dict(_mapping(item, "Study navigation link"))
        for item in _sequence(value.get("navigation_links"), "Study navigation links")
    )
    value["report_only_run_ids"] = tuple(
        _sequence(value.get("report_only_run_ids"), "Study report-only Runs")
    )
    return StudyReportIndexV1(**value)


def campaign_report_index_from_dict(raw: Mapping[str, Any]) -> CampaignReportIndexV1:
    _fields(raw, CampaignReportIndexV1, "campaign report index")
    value = dict(raw)
    value["membership"] = campaign_membership_from_dict(
        _mapping(value.get("membership"), "campaign membership")
    )
    try:
        value["publication_destination"] = evidence_destination_from_dict(
            _mapping(value.get("publication_destination"), "campaign destination")
        )
    except ValueError as exc:
        raise ScientificReportError(str(exc)) from exc
    value["studies"] = tuple(
        study_report_index_from_dict(_mapping(item, "Study index"))
        for item in _sequence(value.get("studies"), "Studies")
    )
    value["result_projects"] = tuple(
        _sequence(value.get("result_projects"), "campaign result projects")
    )
    value["report_only_run_ids"] = tuple(
        _sequence(value.get("report_only_run_ids"), "campaign report-only Runs")
    )
    return CampaignReportIndexV1(**value)


def campaign_membership_from_dict(raw: Mapping[str, Any]) -> CampaignMembershipV1:
    _fields(raw, CampaignMembershipV1, "campaign membership")
    value = dict(raw)
    value["studies"] = tuple(
        CampaignStudyMembershipV1(**dict(_mapping(item, "campaign member")))
        for item in _sequence(value.get("studies"), "campaign members")
    )
    return CampaignMembershipV1(**value)


def article_publication_receipt_from_dict(
    raw: Mapping[str, Any],
) -> ArticlePublicationReceiptV1:
    _fields(raw, ArticlePublicationReceiptV1, "article publication receipt")
    return ArticlePublicationReceiptV1(**dict(raw))


def render_report_markdown(
    value: ScientificReportV1 | CampaignReportIndexV1,
) -> str:
    if isinstance(value, CampaignReportIndexV1):
        studies = "\n".join(
            f"- {item.study_id}: {item.summary.behavioral_status}; task validity "
            f"{item.summary.task_validity_status}; evidence {item.summary.evidence_grade}; "
            f"{item.summary.pair_count} pairs."
            for item in value.studies
        )
        return (
            f"# Fugue campaign: {value.campaign_id}\n\n"
            "Independent case studies; outcomes are not pooled into a ranking.\n\n"
            f"{studies}\n"
        )
    lines = [
        f"# Fugue scientific report: {value.comparison_id}",
        "",
        f"**Finding:** {value.behavioral_status}",
        f"**Task validity:** {value.task_validity_status}",
        f"**Evidence grade:** {value.evidence_grade}",
        f"**Exact revisions:** {', '.join(value.exact_revisions)}",
        f"**Runtime locks:** {', '.join(value.runtime_digests)}",
        "",
        "## What we compared",
        "",
        value.narrative["what"],
        "",
        "## Why we ran it",
        "",
        value.narrative["why"],
        "",
        "## How we ran it",
        "",
        value.narrative["how"],
        "",
        "## Finding",
        "",
        value.narrative["finding"],
        "",
    ]
    if value.named_blockers:
        lines.extend(("## Named deterministic blockers", ""))
        lines.extend(f"- {item}" for item in value.named_blockers)
        lines.append("")
    lines.extend(
        [
        "## Claim ledger",
        "",
        ]
    )
    lines.extend(
        f"- **{item.category} / {item.status}:** {item.statement}"
        for item in value.claim_ledger
    )
    if value.limitations:
        lines.extend(("", "## Limitations", ""))
        lines.extend(f"- {item}" for item in value.limitations)
    if value.visual_assets is not None:
        lines.extend(("", "## Bound visual evidence", ""))
        lines.extend(
            f"- {item.role}: `{item.path}` — {item.alt_text}"
            for item in value.visual_assets.assets
        )
    lines.extend(("", "## Next action", "", value.narrative["next_action"]))
    return "\n".join(lines) + "\n"


def _section_claim(result: ComparisonResultV3, category: str) -> ClaimLedgerEntryV1:
    section = {
        "judge": result.judge_summary,
        "mechanism": result.mechanism_summary,
        "efficiency": result.operational_summary,
    }[category]
    section_status = str(section.get("status") or "").lower()
    available = bool(section) and section_status not in {
        "missing",
        "not_run",
        "not_used",
        "unavailable",
    }
    status: ClaimStatus = (
        "advisory"
        if category == "judge" and available
        else ("descriptive" if available else "unavailable")
    )
    wording = {
        "judge": _judge_statement(section),
        "mechanism": _mechanism_statement(section),
        "efficiency": _efficiency_statement(section),
    }[category]
    if not available:
        wording = f"{category.capitalize()} evidence is unavailable."
    return ClaimLedgerEntryV1(
        id=f"{category}-evidence",
        category=category,  # type: ignore[arg-type]
        status=status,
        statement=wording,
        canonical_refs=(f"result:{result.result_digest}#{category}_summary",),
    )


def _judge_statement(section: Mapping[str, Any]) -> str:
    status = str(section.get("status") or "unavailable")
    claim_status = str(section.get("claim_status") or "unqualified")
    unavailable = int(section.get("unavailable_attempts") or 0)
    means: list[str] = []
    by_variant = section.get("by_variant")
    if isinstance(by_variant, Mapping):
        for variant in ("baseline", "candidate"):
            dimensions = by_variant.get(variant)
            if not isinstance(dimensions, Mapping):
                continue
            for dimension, raw in sorted(dimensions.items()):
                if not isinstance(raw, Mapping) or raw.get("mean") is None:
                    continue
                means.append(f"{variant} {dimension} mean={raw['mean']}")
    detail = "; ".join(means[:8]) or "no comparable judge means"
    return _sanitize(
        f"Judge evidence is advisory: status={status}, qualification={claim_status}, "
        f"unavailable attempts={unavailable}; {detail}. It remains separate from "
        "deterministic outcomes."
    )


def _mechanism_statement(section: Mapping[str, Any]) -> str:
    observations: list[str] = []
    for stage, raw in sorted(section.items()):
        if not isinstance(raw, Mapping):
            continue
        for variant in ("baseline", "candidate"):
            value = raw.get(variant)
            if not isinstance(value, Mapping):
                continue
            observations.append(
                f"{stage}/{variant} observed={int(value.get('observed') or 0)}"
                f" of {int(value.get('applicable') or 0)}"
            )
    detail = "; ".join(observations[:12]) or "aggregate mechanism status recorded"
    return _sanitize(
        f"Mechanism evidence is descriptive: {detail}. Registration, opening, or "
        "invocation cannot create a behavioral pass."
    )


def _efficiency_statement(section: Mapping[str, Any]) -> str:
    cost_status = str(section.get("total_cost_status") or "not_reported")
    observed_cost = section.get("observed_cost_usd")
    agent_cost = section.get("agent_observed_cost_usd")
    latency = section.get("latency_ms")
    usage_rows = int(section.get("usage_rows") or 0)
    return _sanitize(
        "Efficiency evidence is descriptive: "
        f"total cost status={cost_status}, observed cost={observed_cost}, "
        f"Agent observed cost={agent_cost}, aggregate latency_ms={latency}, "
        f"usage rows={usage_rows}. It cannot override task failure or support a "
        "paired efficiency claim when cost is unreconciled."
    )


def _aggregate_task_validity(statuses: Sequence[str]) -> TaskValidityStatus:
    values = set(statuses)
    for status in ("invalid", "drifted", "inconclusive", "non_discriminating"):
        if status in values:
            return status  # type: ignore[return-value]
    if values == {"valid"}:
        return "valid"
    raise ScientificReportError("result task validity is incomplete")


def _exact_revisions(result: ComparisonResultV3) -> tuple[str, ...]:
    values: set[str] = set()
    arms = result.cohort_lineage.get("arms")
    if isinstance(arms, Mapping):
        for arm_id, raw in arms.items():
            if not isinstance(raw, Mapping):
                continue
            revisions = raw.get("source_revisions")
            if isinstance(revisions, Sequence) and not isinstance(
                revisions, (str, bytes)
            ):
                for revision in revisions:
                    if not isinstance(revision, Mapping):
                        continue
                    identity = revision.get("version_identity")
                    if identity:
                        values.add(
                            f"{arm_id}:{revision.get('kind') or 'source'}:"
                            f"{revision.get('id') or 'unknown'}@{identity}"
                        )
            behavior = raw.get("behavior_digest")
            if not revisions and isinstance(behavior, str):
                values.add(f"{arm_id}:behavior@{behavior}")
    if not values:
        raise ScientificReportError("V3 result has no exact arm revisions")
    return tuple(sorted(_sanitize(item) for item in values))


def _report_source_identity(
    result: ComparisonResultV3,
    *,
    role: Literal["baseline", "candidate"],
) -> ReportSourceIdentityV1:
    arms = result.cohort_lineage.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(role), Mapping):
        raise ScientificReportError(f"V3 result lacks its {role} cohort identity")
    arm = arms[role]
    assert isinstance(arm, Mapping)
    revisions = sorted(
        (
            dict(item)
            for item in arm.get("source_revisions") or ()
            if isinstance(item, Mapping)
        ),
        key=lambda item: (
            str(item.get("kind") or ""),
            str(item.get("id") or ""),
            str(item.get("version_identity") or ""),
        ),
    )
    if len(revisions) == 1:
        revision = revisions[0]
        raw_runtime = str(revision.get("runtime_digest") or "").removeprefix(
            "sha256:"
        )
        raw_lock = str(revision.get("lock_digest") or "").removeprefix("sha256:")
        return ReportSourceIdentityV1(
            schema_version=1,
            role=role,
            kind=str(revision.get("kind") or "source"),
            id=str(revision.get("id") or "unknown"),
            version_identity=str(revision.get("version_identity") or ""),
            runtime_digest=raw_runtime,
            lock_digest=raw_lock or None,
        )
    behavior_digest = str(arm.get("behavior_digest") or "")
    _digest(behavior_digest, f"{role} behavior digest")
    normalized = [
        {
            **item,
            "runtime_digest": str(item.get("runtime_digest") or "").removeprefix(
                "sha256:"
            ),
            **(
                {
                    "lock_digest": str(item.get("lock_digest") or "").removeprefix(
                        "sha256:"
                    )
                }
                if item.get("lock_digest")
                else {}
            ),
        }
        for item in revisions
    ]
    return ReportSourceIdentityV1(
        schema_version=1,
        role=role,
        kind="candidate_bundle",
        id=f"{result.comparison_id}-{role}",
        version_identity=(
            f"bundle:{stable_digest(normalized)}"
            if normalized
            else f"behavior:{behavior_digest}"
        ),
        runtime_digest=behavior_digest,
        lock_digest=stable_digest(normalized) if normalized else None,
    )


def _named_deterministic_blockers(
    result: ComparisonResultV3,
) -> tuple[str, ...]:
    blockers: set[str] = {
        _sanitize(item)
        for item in result.behavioral_summary.critical_blockers
        if str(item).strip()
    }
    for validity in result.task_validity:
        for blocker in validity.blockers:
            blockers.add(_sanitize(f"{validity.task_id}: {blocker}"))
    for pair in result.paired_cases:
        for change in pair.dimension_changes:
            if change.critical and change.candidate is False:
                blockers.add(
                    _sanitize(
                        f"{pair.task_id} attempt {pair.attempt}: "
                        f"{change.label} ({change.id}) failed for the candidate"
                    )
                )
    return tuple(sorted(blockers))


def _blocker_statement(blockers: Sequence[str]) -> str:
    prefix = "Named deterministic blockers: "
    selected: list[str] = []
    for blocker in blockers:
        candidate = prefix + "; ".join((*selected, blocker))
        if len(candidate) > 1_800:
            break
        selected.append(blocker)
    remaining = len(blockers) - len(selected)
    suffix = f"; plus {remaining} additional named blocker(s)." if remaining else ""
    return prefix + "; ".join(selected) + suffix


def _safe_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({_sanitize(item) for item in values if str(item).strip()}))


def _sanitize(value: Any) -> str:
    return redact_text(str(value).strip())[:4_000]


def _fields(
    raw: Mapping[str, Any],
    cls: type[Any],
    label: str,
    *,
    optional: set[str] | None = None,
) -> None:
    expected = set(cls.__dataclass_fields__)
    unknown = set(raw) - expected
    missing = expected - set(raw) - (optional or set())
    if unknown:
        raise ScientificReportError(
            f"{label} has unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ScientificReportError(
            f"{label} is missing fields: {', '.join(sorted(missing))}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificReportError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScientificReportError(f"{label} must be an array")
    return value


def _identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ScientificReportError(f"{label} is invalid")


def _safe_bundle_path(value: str, label: str, *, prefix: str = "") -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ScientificReportError(f"{label} is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ScientificReportError(f"{label} is invalid")
    if prefix and not value.startswith(prefix):
        raise ScientificReportError(f"{label} must start with {prefix!r}")


def _digest(value: str, label: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ScientificReportError(f"{label} is invalid")


def _project(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or value.count("/") != 1
        or any(not part or not _ID.fullmatch(part) for part in value.split("/"))
    ):
        raise ScientificReportError(f"{label} is invalid")


def _url(value: str, label: str) -> None:
    parsed = urlsplit(value)
    safe_scheme = parsed.scheme == "https" or (
        parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
    )
    if (
        not safe_scheme
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ScientificReportError(
            f"{label} must be credential-free HTTPS or loopback HTTP"
        )
    sensitive = (
        "token",
        "apikey",
        "secret",
        "password",
        "auth",
        "credential",
        "signature",
    )
    if any(
        name.lower() == "key"
        or any(
            part in name.lower().replace("-", "").replace("_", "") for part in sensitive
        )
        for name, _value in parse_qsl(parsed.query)
    ):
        raise ScientificReportError(f"{label} must not contain credential parameters")


def _public_text(value: str, label: str, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ScientificReportError(f"{label} is invalid")
    if redact_text(value) != value:
        raise ScientificReportError(f"{label} contains sensitive material")


def _canonical_texts(
    values: Sequence[str], label: str, *, require: bool = False
) -> None:
    if require and not values:
        raise ScientificReportError(f"{label} must not be empty")
    if tuple(values) != tuple(sorted(set(values))):
        raise ScientificReportError(f"{label} must be sorted and unique")
    for value in values:
        _public_text(value, label, 4_000)


def _canonical_ids(values: Sequence[Any], label: str) -> None:
    ids = tuple(item.id if hasattr(item, "id") else item.study_id for item in values)
    if ids != tuple(sorted(set(ids))):
        raise ScientificReportError(f"{label} ids must be sorted and unique")
