#!/usr/bin/env python3
"""Prepare the descriptive V2 Skill Creator measurement Study.

V2 deliberately reuses the exact V1 source and task-archive preparation, then
binds the independently generated scorer-mutation receipt and compatibility
product contract.  The wrapper never serializes private labels or mutations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_confirmatory import DEFAULT_OUTPUT
from prepare_confirmatory import prepare as prepare_v1
from validate_confirmatory_scorer_v2 import validate as validate_scorer

EXAMPLE = Path(__file__).resolve().parent
SPEC = EXAMPLE / "confirmatory-v2.yaml"
V1_SPEC = EXAMPLE / "confirmatory-v1.yaml"
AMENDMENT = EXAMPLE / "confirmatory-v2-amendment.json"
PRODUCT_CONTRACT = EXAMPLE / "compatibility-product-contract-v2.json"
VALIDATOR = EXAMPLE / "validate_confirmatory_scorer_v2.py"
VALIDATION_RECEIPT = EXAMPLE / "confirmatory-scorer-validation-v2.json"
SCORER = EXAMPLE / "confirmatory_scorer.py"
TASKS = EXAMPLE / "confirmatory-tasks.jsonl"
PRIVATE_LABELS = EXAMPLE / "confirmatory-private-labels.jsonl"
PREREGISTRATION = EXAMPLE / "confirmatory-preregistration.json"
QUALIFICATION_FIXTURES = EXAMPLE / "qualification_fixtures.py"
TASK_FAMILY_LOCK = EXAMPLE / "confirmatory-task-family-lock.json"
SKILL_REVISION_LOCK = EXAMPLE / "skill-revisions.lock.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain one JSON object")
    return value


def _verified_validation_receipt() -> dict[str, Any]:
    recorded = _json(VALIDATION_RECEIPT)
    observed = validate_scorer()
    if recorded != observed:
        raise RuntimeError(
            "checked-in scorer mutation receipt disagrees with fresh validation"
        )
    unsigned = {key: value for key, value in recorded.items() if key != "receipt_digest"}
    if (
        recorded.get("schema_version") != 2
        or recorded.get("status") != "passed"
        or recorded.get("private_inputs_serialized") is not False
        or recorded.get("receipt_digest") != _stable_digest(unsigned)
    ):
        raise RuntimeError("scorer mutation receipt is not qualification-ready")
    return recorded


def _verified_product_contract() -> dict[str, Any]:
    contract = _json(PRODUCT_CONTRACT)
    treatment = contract.get("treatment_scope")
    claim_exclusions = contract.get("claims_excluded")
    if (
        contract.get("schema_version") != 2
        or contract.get("id")
        != "anthropic-skill-creator-compatibility-product-contract-v2"
        or contract.get("status") != "frozen_before_expanded_holdout"
        or not isinstance(treatment, dict)
        or treatment.get("baseline_commit")
        != "a5bcdd7e58cdff48566bf876f0a72a2008dcefbc"
        or treatment.get("candidate_commit")
        != "1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563"
        or not isinstance(claim_exclusions, list)
        or not claim_exclusions
    ):
        raise RuntimeError("compatibility product contract is malformed or drifted")
    return contract


def _verified_amendment() -> dict[str, Any]:
    amendment = _json(AMENDMENT)
    unsigned = {key: value for key, value in amendment.items() if key != "amendment_digest"}
    if amendment.get("amendment_digest") != _stable_digest(unsigned):
        raise RuntimeError("confirmatory V2 amendment digest is invalid")
    prior = amendment.get("prior_study")
    measurement = amendment.get("measurement_revision")
    replacement = amendment.get("replacement_execution")
    frozen = amendment.get("frozen_behavioral_inputs")
    if not all(
        isinstance(value, dict)
        for value in (prior, measurement, replacement, frozen)
    ):
        raise RuntimeError("confirmatory V2 amendment bindings are incomplete")
    actual = {
        "prior_spec": _sha256(V1_SPEC),
        "scorer": _sha256(SCORER),
        "validator": _sha256(VALIDATOR),
        "validation_receipt": _sha256(VALIDATION_RECEIPT),
        "product_contract": _sha256(PRODUCT_CONTRACT),
        "replacement_spec": _sha256(SPEC),
        "source_preparer": _sha256(Path(__file__).resolve()),
        "public_tasks": _sha256(TASKS),
        "private_labels": _sha256(PRIVATE_LABELS),
        "preregistration": _sha256(PREREGISTRATION),
        "qualification_fixtures": _sha256(QUALIFICATION_FIXTURES),
        "task_family_lock": _sha256(TASK_FAMILY_LOCK),
        "skill_revision_lock": _sha256(SKILL_REVISION_LOCK),
    }
    expected = {
        "prior_spec": prior.get("comparison_spec_sha256"),
        "scorer": measurement.get("deterministic_scorer_sha256"),
        "validator": measurement.get("mutation_validator_sha256"),
        "validation_receipt": measurement.get(
            "mutation_validation_receipt_sha256"
        ),
        "product_contract": measurement.get(
            "compatibility_product_contract_sha256"
        ),
        "replacement_spec": replacement.get("comparison_spec_sha256"),
        "source_preparer": replacement.get("source_preparer_sha256"),
        "public_tasks": frozen.get("public_tasks_sha256"),
        "private_labels": frozen.get("private_labels_sha256"),
        "preregistration": frozen.get("repository_preregistration_sha256"),
        "qualification_fixtures": frozen.get(
            "qualification_fixture_generator_sha256"
        ),
        "task_family_lock": frozen.get("host_only_task_family_lock_sha256"),
        "skill_revision_lock": frozen.get("skill_revision_lock_sha256"),
    }
    drifted = sorted(name for name, digest in actual.items() if expected[name] != digest)
    if drifted:
        raise RuntimeError(
            "confirmatory V2 amendment artifact hashes drifted: "
            + ", ".join(drifted)
        )
    receipt = _json(VALIDATION_RECEIPT)
    if measurement.get("mutation_validation_receipt_digest") != receipt.get(
        "receipt_digest"
    ):
        raise RuntimeError("confirmatory V2 amendment receipt digest drifted")
    return amendment


def prepare(anthropic_repo: Path, output: Path) -> dict[str, Any]:
    validation = _verified_validation_receipt()
    product_contract = _verified_product_contract()
    amendment = _verified_amendment()
    base = prepare_v1(anthropic_repo, output)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "id": "anthropic-skill-creator-confirmatory-preparation-v2",
        "study_id": "anthropic-skill-creator-confirmatory-v2",
        "study_class": "measurement_development_descriptive",
        "conference_claim_eligible": False,
        "population_claim_eligible": False,
        "private_inputs_serialized": False,
        "base_preparation_manifest_digest": base["manifest_digest"],
        "comparison_spec_sha256": _sha256(SPEC),
        "amendment": {
            "sha256": _sha256(AMENDMENT),
            "amendment_digest": amendment["amendment_digest"],
        },
        "compatibility_product_contract": {
            "id": product_contract["id"],
            "sha256": _sha256(PRODUCT_CONTRACT),
        },
        "scorer_mutation_validation": {
            "id": validation["id"],
            "validator_sha256": _sha256(VALIDATOR),
            "receipt_sha256": _sha256(VALIDATION_RECEIPT),
            "receipt_digest": validation["receipt_digest"],
        },
        "preparation_inputs": {
            "v1_spec_sha256": _sha256(V1_SPEC),
            "v2_spec_sha256": _sha256(SPEC),
            "v1_preparer_sha256": _sha256(EXAMPLE / "prepare_confirmatory.py"),
            "v2_preparer_sha256": _sha256(Path(__file__).resolve()),
            "upstream_source_preparer_sha256": _sha256(EXAMPLE / "prepare_sources.py"),
            "zero_model_conformance_sha256": _sha256(
                EXAMPLE / "zero_model_conformance.py"
            ),
            "qualification_fixture_generator_sha256": _sha256(
                QUALIFICATION_FIXTURES
            ),
            "host_only_task_family_lock_sha256": _sha256(TASK_FAMILY_LOCK),
            "skill_revision_lock_sha256": _sha256(SKILL_REVISION_LOCK),
        },
    }
    manifest["manifest_digest"] = _stable_digest(manifest)
    (output / "confirmatory-v2-preparation.lock.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anthropic-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.anthropic_repo.resolve(), args.output.resolve())
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
