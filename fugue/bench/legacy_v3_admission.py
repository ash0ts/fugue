from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any, Literal

from fugue.bench.candidates import stable_digest

LEGACY_HOSTED_V3_ADMISSION_RESOURCE = (
    "resources/compatibility/legacy-hosted-v3-admissions-v1.json"
)
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_OBSERVED_BINDINGS = (
    "comparison_id",
    "result_digest",
    "qualification_digest",
    "preview_digest",
    "result_source",
    "source_project",
    "result_project",
    "source_lock_digest",
    "evidence_topology_digest",
    "aligned_analysis_digest",
)


@dataclass(frozen=True)
class LegacyHostedV3ReviewProvenanceV1:
    kind: str
    digest: str
    review_status: Literal["reviewed"]
    preview_artifact_sha256: str
    spec_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "digest": self.digest,
            "review_status": self.review_status,
            "preview_artifact_sha256": self.preview_artifact_sha256,
            "spec_digest": self.spec_digest,
        }


@dataclass(frozen=True)
class LegacyHostedV3AdmissionV1:
    schema_version: Literal[1]
    result_file_sha256: str
    comparison_id: str
    result_digest: str
    qualification_digest: str
    preview_digest: str
    result_source: str
    source_project: str
    result_project: str
    source_lock_digest: str
    evidence_topology_digest: str
    aligned_analysis_digest: str
    reviewed_provenance: LegacyHostedV3ReviewProvenanceV1
    admission_digest: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result_file_sha256": self.result_file_sha256,
            "comparison_id": self.comparison_id,
            "result_digest": self.result_digest,
            "qualification_digest": self.qualification_digest,
            "preview_digest": self.preview_digest,
            "result_source": self.result_source,
            "source_project": self.source_project,
            "result_project": self.result_project,
            "source_lock_digest": self.source_lock_digest,
            "evidence_topology_digest": self.evidence_topology_digest,
            "aligned_analysis_digest": self.aligned_analysis_digest,
            "reviewed_provenance": self.reviewed_provenance.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "admission_digest": self.admission_digest}

    def observed_bindings(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in _OBSERVED_BINDINGS}


@dataclass(frozen=True)
class LegacyHostedV3AdmissionRegistryV1:
    schema_version: Literal[1]
    kind: Literal["legacy-hosted-v3-admission-registry"]
    admissions: tuple[LegacyHostedV3AdmissionV1, ...]
    registry_digest: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "admissions": [item.to_dict() for item in self.admissions],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "registry_digest": self.registry_digest}


def legacy_hosted_v3_admission_registry_from_dict(
    value: Mapping[str, Any],
) -> LegacyHostedV3AdmissionRegistryV1:
    _require_exact_keys(
        value,
        {"schema_version", "kind", "admissions", "registry_digest"},
        "legacy hosted V3 admission registry",
    )
    if value.get("schema_version") != 1:
        raise ValueError("legacy hosted V3 admission registry version is unsupported")
    if value.get("kind") != "legacy-hosted-v3-admission-registry":
        raise ValueError("legacy hosted V3 admission registry kind is invalid")
    raw_admissions = value.get("admissions")
    if not isinstance(raw_admissions, list):
        raise ValueError("legacy hosted V3 admissions must be a list")
    admissions = tuple(_admission_from_dict(item) for item in raw_admissions)
    file_digests = tuple(item.result_file_sha256 for item in admissions)
    if file_digests != tuple(sorted(file_digests)) or len(set(file_digests)) != len(
        file_digests
    ):
        raise ValueError(
            "legacy hosted V3 admissions must have unique sorted result-file digests"
        )
    registry_digest = _digest(value.get("registry_digest"), "registry digest")
    registry = LegacyHostedV3AdmissionRegistryV1(
        schema_version=1,
        kind="legacy-hosted-v3-admission-registry",
        admissions=admissions,
        registry_digest=registry_digest,
    )
    if stable_digest(registry.unsigned_dict()) != registry.registry_digest:
        raise ValueError("legacy hosted V3 admission registry digest does not match")
    return registry


@cache
def load_packaged_legacy_hosted_v3_admission_registry(
) -> LegacyHostedV3AdmissionRegistryV1:
    item = files("fugue").joinpath(LEGACY_HOSTED_V3_ADMISSION_RESOURCE)
    try:
        value = json.loads(item.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load packaged legacy hosted V3 admissions") from exc
    if not isinstance(value, Mapping):
        raise ValueError("legacy hosted V3 admission registry must be a mapping")
    return legacy_hosted_v3_admission_registry_from_dict(value)


def require_legacy_hosted_v3_admission(
    payload: bytes,
    *,
    observed: Mapping[str, str],
    registry: LegacyHostedV3AdmissionRegistryV1 | None = None,
) -> LegacyHostedV3AdmissionV1:
    authority = registry or load_packaged_legacy_hosted_v3_admission_registry()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    admission = next(
        (
            item
            for item in authority.admissions
            if item.result_file_sha256 == payload_sha256
        ),
        None,
    )
    if admission is None:
        raise ValueError(
            "ComparisonResultV3 historical hosted bytes are not in the reviewed "
            "admission registry"
        )
    if set(observed) != set(_OBSERVED_BINDINGS):
        raise ValueError("legacy hosted V3 observed bindings are incomplete")
    expected = admission.observed_bindings()
    if any(str(observed[name]) != expected[name] for name in _OBSERVED_BINDINGS):
        raise ValueError(
            "ComparisonResultV3 historical hosted identity disagrees with its "
            "reviewed admission"
        )
    return admission


def legacy_hosted_v3_view_is_admitted(
    *,
    result_digest: str | None,
    qualification_digest: str | None,
    preview_digest: str | None,
    registry: LegacyHostedV3AdmissionRegistryV1 | None = None,
) -> bool:
    if not result_digest or not qualification_digest or not preview_digest:
        return False
    authority = registry or load_packaged_legacy_hosted_v3_admission_registry()
    return any(
        item.result_digest == result_digest
        and item.qualification_digest == qualification_digest
        and item.preview_digest == preview_digest
        for item in authority.admissions
    )


def _admission_from_dict(value: Any) -> LegacyHostedV3AdmissionV1:
    if not isinstance(value, Mapping):
        raise ValueError("legacy hosted V3 admission must be a mapping")
    expected_keys = {
        "schema_version",
        "result_file_sha256",
        *_OBSERVED_BINDINGS,
        "reviewed_provenance",
        "admission_digest",
    }
    _require_exact_keys(value, expected_keys, "legacy hosted V3 admission")
    if value.get("schema_version") != 1:
        raise ValueError("legacy hosted V3 admission version is unsupported")
    provenance = _provenance_from_dict(value.get("reviewed_provenance"))
    admission = LegacyHostedV3AdmissionV1(
        schema_version=1,
        result_file_sha256=_digest(
            value.get("result_file_sha256"), "result file sha256"
        ),
        comparison_id=_text(value.get("comparison_id"), "comparison id"),
        result_digest=_digest(value.get("result_digest"), "result digest"),
        qualification_digest=_digest(
            value.get("qualification_digest"), "qualification digest"
        ),
        preview_digest=_digest(value.get("preview_digest"), "preview digest"),
        result_source=_text(value.get("result_source"), "result source"),
        source_project=_project(value.get("source_project"), "source project"),
        result_project=_project(value.get("result_project"), "result project"),
        source_lock_digest=_digest(
            value.get("source_lock_digest"), "source lock digest"
        ),
        evidence_topology_digest=_digest(
            value.get("evidence_topology_digest"), "evidence topology digest"
        ),
        aligned_analysis_digest=_digest(
            value.get("aligned_analysis_digest"), "aligned analysis digest"
        ),
        reviewed_provenance=provenance,
        admission_digest=_digest(value.get("admission_digest"), "admission digest"),
    )
    if stable_digest(admission.unsigned_dict()) != admission.admission_digest:
        raise ValueError("legacy hosted V3 admission digest does not match")
    return admission


def _provenance_from_dict(value: Any) -> LegacyHostedV3ReviewProvenanceV1:
    if not isinstance(value, Mapping):
        raise ValueError("legacy hosted V3 review provenance must be a mapping")
    _require_exact_keys(
        value,
        {
            "kind",
            "digest",
            "review_status",
            "preview_artifact_sha256",
            "spec_digest",
        },
        "legacy hosted V3 review provenance",
    )
    if value.get("review_status") != "reviewed":
        raise ValueError("legacy hosted V3 admission must be reviewed")
    return LegacyHostedV3ReviewProvenanceV1(
        kind=_text(value.get("kind"), "review provenance kind"),
        digest=_digest(value.get("digest"), "review provenance digest"),
        review_status="reviewed",
        preview_artifact_sha256=_digest(
            value.get("preview_artifact_sha256"), "preview artifact sha256"
        ),
        spec_digest=_digest(value.get("spec_digest"), "spec digest"),
    )


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate legacy admission key: {key}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has an invalid shape")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"legacy hosted V3 {label} is invalid")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 240:
        raise ValueError(f"legacy hosted V3 {label} is invalid")
    return value


def _project(value: Any, label: str) -> str:
    project = _text(value, label)
    if project.count("/") != 1 or any(not part for part in project.split("/")):
        raise ValueError(f"legacy hosted V3 {label} is invalid")
    return project
