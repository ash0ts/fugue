from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.legacy_v3_admission import (
    legacy_hosted_v3_admission_registry_from_dict,
    legacy_hosted_v3_view_is_admitted,
    load_packaged_legacy_hosted_v3_admission_registry,
    require_legacy_hosted_v3_admission,
)

ROOT = Path(__file__).resolve().parents[1]
FAILURE_LOCK = (
    ROOT
    / "examples/loop-engineering/wandb-evidence-loop/fixtures/"
    "mcp-v10-exact-history-baseline.failure-lock.json"
)


def _test_registry(payload: bytes):
    observed = {
        "comparison_id": "legacy-comparison",
        "result_digest": "1" * 64,
        "qualification_digest": "1" * 64,
        "preview_digest": "2" * 64,
        "result_source": "legacy-run",
        "source_project": "wandb/source",
        "result_project": "wandb/result",
        "source_lock_digest": "3" * 64,
        "evidence_topology_digest": "4" * 64,
        "aligned_analysis_digest": "5" * 64,
    }
    admission = {
        "schema_version": 1,
        "result_file_sha256": hashlib.sha256(payload).hexdigest(),
        **observed,
        "reviewed_provenance": {
            "kind": "test-review-lock",
            "digest": "6" * 64,
            "review_status": "reviewed",
            "preview_artifact_sha256": "7" * 64,
            "spec_digest": "8" * 64,
        },
    }
    admission["admission_digest"] = stable_digest(admission)
    document = {
        "schema_version": 1,
        "kind": "legacy-hosted-v3-admission-registry",
        "admissions": [admission],
    }
    document["registry_digest"] = stable_digest(document)
    return legacy_hosted_v3_admission_registry_from_dict(document), observed


def test_exact_bytes_and_bound_identity_control_legacy_admission() -> None:
    payload = b'{"schema_version":3,"kind":"historical"}'
    registry, observed = _test_registry(payload)

    admitted = require_legacy_hosted_v3_admission(
        payload,
        observed=observed,
        registry=registry,
    )
    assert admitted.result_file_sha256 == hashlib.sha256(payload).hexdigest()
    assert legacy_hosted_v3_view_is_admitted(
        result_digest=observed["result_digest"],
        qualification_digest=observed["qualification_digest"],
        preview_digest=observed["preview_digest"],
        registry=registry,
    )

    with pytest.raises(ValueError, match="not in the reviewed admission registry"):
        require_legacy_hosted_v3_admission(
            payload + b"\n",
            observed=observed,
            registry=registry,
        )
    with pytest.raises(ValueError, match="identity disagrees"):
        require_legacy_hosted_v3_admission(
            payload,
            observed={**observed, "result_source": "forged-run"},
            registry=registry,
        )


def test_packaged_v10_admission_matches_the_reviewed_failure_lock() -> None:
    failure_lock = json.loads(FAILURE_LOCK.read_text(encoding="utf-8"))
    source = failure_lock["source"]
    registry = load_packaged_legacy_hosted_v3_admission_registry()

    assert len(registry.admissions) == 1
    admission = registry.admissions[0]
    assert admission.result_file_sha256 == source["result_artifact_sha256"]
    assert admission.result_digest == source["result_digest"]
    assert admission.qualification_digest == source["qualification_digest"]
    assert admission.comparison_id == source["comparison_id"]
    assert admission.preview_digest == source["preview_digest"]
    assert admission.result_source == source["result_source"]
    assert admission.source_project == source["source_project"]
    assert admission.result_project == source["result_project"]
    assert admission.source_lock_digest == source["source_lock_digest"]
    assert admission.evidence_topology_digest == source["evidence_topology_digest"]
    assert admission.aligned_analysis_digest == source["aligned_analysis_digest"]
    assert admission.reviewed_provenance.digest == failure_lock["lock_sha256"]
    assert admission.reviewed_provenance.preview_artifact_sha256 == source[
        "preview_artifact_sha256"
    ]
    assert admission.reviewed_provenance.spec_digest == source["spec_digest"]


def test_registry_rejects_unsorted_duplicate_or_rehashed_entries() -> None:
    first_payload = b"first"
    first, _ = _test_registry(first_payload)
    document = first.to_dict()
    document["registry_digest"] = "f" * 64
    with pytest.raises(ValueError, match="registry digest does not match"):
        legacy_hosted_v3_admission_registry_from_dict(document)

    duplicate = first.admissions[0].to_dict()
    duplicated = {
        "schema_version": 1,
        "kind": "legacy-hosted-v3-admission-registry",
        "admissions": [duplicate, duplicate],
    }
    duplicated["registry_digest"] = stable_digest(duplicated)
    with pytest.raises(ValueError, match="unique sorted"):
        legacy_hosted_v3_admission_registry_from_dict(duplicated)
