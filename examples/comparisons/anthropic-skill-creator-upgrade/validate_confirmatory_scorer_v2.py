#!/usr/bin/env python3
"""Mutation-qualify every deterministic Skill Creator scorer dimension.

The original base fixture changed only ``schema_version``.  That proved the
scorer could reject one malformed envelope, but not that its substantive
dimensions detected the defects they claimed to measure.  This trusted,
host-only preparation check starts from every canonical gold artifact and
applies dimension-targeted mutations.  It emits only case identities and
aggregate receipts; private expected values and mutated artifacts are never
serialized.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import runpy
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qualification_fixtures import gold_output, qualification_evidence

EXAMPLE = Path(__file__).resolve().parent
LABELS = EXAMPLE / "confirmatory-private-labels.jsonl"
SCORER = EXAMPLE / "confirmatory_scorer.py"

DIMENSIONS = (
    "artifact_validity",
    "frontmatter_semantics",
    "compatibility_selection",
    "name_help_consistency",
    "behavior_preservation",
    "packaging",
    "instruction_quality",
    "dependency_secret_safety",
    "assigned_script_use",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in LABELS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _scorer() -> Callable[[dict[str, Any], object, dict[str, Any]], dict[str, bool]]:
    return runpy.run_path(SCORER.as_posix())["score"]


def _frontmatter(text: str, key: str, value: str | None) -> str:
    prefix, body = text.split("\n---\n", 1)
    lines = prefix.splitlines()
    updated: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(f"{key}:"):
            replaced = True
            if value is not None:
                updated.append(f"{key}: {value}")
        else:
            updated.append(line)
    if value is not None and not replaced:
        updated.append(f"{key}: {value}")
    return "\n".join(updated) + "\n---\n" + body


def _first_skill_path(result: dict[str, Any], expected: dict[str, Any]) -> str | None:
    declared = expected.get("skill_path")
    if isinstance(declared, str) and declared in result.get("files", {}):
        return declared
    return next(
        (
            path
            for path in result.get("files", {})
            if isinstance(path, str) and path.endswith("/SKILL.md")
        ),
        None,
    )


def _artifact_mutation(result: dict[str, Any], _expected: dict[str, Any]) -> bool:
    result["schema_version"] = 0
    return True


def _frontmatter_mutation(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    path = _first_skill_path(result, expected)
    if path is None:
        if expected.get("disposition") == "rejected":
            result.setdefault("files", {})["skills/invalid/SKILL.md"] = "not frontmatter"
            return True
        return False
    result["files"][path] = _frontmatter(result["files"][path], "unknown-field", "x")
    return True


def _compatibility_mutation(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    contract = expected.get("compatibility")
    if not isinstance(contract, dict) or contract.get("policy") == "not_applicable":
        return False
    path = _first_skill_path(result, expected)
    if path is None:
        return False
    policy = contract.get("policy")
    if policy == "absent":
        value = "invented platform requirement"
    elif policy == "length":
        value = "x"
    elif policy == "exact":
        value = "wrong exact compatibility"
    else:
        value = None
    result["files"][path] = _frontmatter(result["files"][path], "compatibility", value)
    return True


def _name_mutation(result: dict[str, Any], _expected: dict[str, Any]) -> bool:
    result["skill_name"] = "wrong-skill-name"
    return True


def _preservation_mutation(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    exact = expected.get("exact_file_sha256")
    preserved = expected.get("preserved_file_sha256")
    candidates: list[str] = []
    if isinstance(exact, dict):
        candidates.extend(str(path) for path in exact)
    if isinstance(preserved, dict):
        candidates.extend(str(path) for path in preserved)
    path = next((item for item in candidates if item in result.get("files", {})), None)
    if path is not None:
        result["files"][path] += "\n# unintended mutation\n"
        return True
    if expected.get("preserved_body_sha256") or expected.get("preserved_metadata"):
        path = _first_skill_path(result, expected)
        if path is not None:
            result["files"][path] += "\nUnrequested behavioral rewrite.\n"
            return True
    return False


def _packaging_mutation(result: dict[str, Any], _expected: dict[str, Any]) -> bool:
    result.setdefault("files", {})["unexpected.txt"] = "unexpected package content"
    return True


def _instruction_mutation(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    if expected.get("disposition") == "rejected":
        result["maintainer_memo"] = "The request was rejected for an unspecified reason."
        return True
    path = _first_skill_path(result, expected)
    if path is None:
        files = result.get("files", {})
        if files:
            first = next(iter(files))
            files[first] = "def help_text():\n    return 'unrelated output'\n"
            return True
        return False
    result["files"][path] = _frontmatter(result["files"][path], "description", "Too short")
    return True


def _safety_mutation(result: dict[str, Any], _expected: dict[str, Any]) -> bool:
    result["maintainer_memo"] = "Used API_KEY=sk-live-mutation and pip install unsafe-package."
    return True


def _script_mutation(_result: dict[str, Any], expected: dict[str, Any]) -> bool:
    required = expected.get("required_opened_suffixes")
    return isinstance(required, list) and bool(required)


MUTATORS: dict[str, Callable[[dict[str, Any], dict[str, Any]], bool]] = {
    "artifact_validity": _artifact_mutation,
    "frontmatter_semantics": _frontmatter_mutation,
    "compatibility_selection": _compatibility_mutation,
    "name_help_consistency": _name_mutation,
    "behavior_preservation": _preservation_mutation,
    "packaging": _packaging_mutation,
    "instruction_quality": _instruction_mutation,
    "dependency_secret_safety": _safety_mutation,
    "assigned_script_use": _script_mutation,
}


def validate() -> dict[str, Any]:
    score = _scorer()
    positive_failures: list[dict[str, str]] = []
    negative_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows():
        task_id = str(row["id"])
        expected = dict(row["expected"])
        gold = gold_output(task_id, expected)
        evidence = {"expected": expected, **qualification_evidence()}
        positive = score({"id": task_id}, gold, evidence)
        for dimension in DIMENSIONS:
            if positive.get(dimension) is not True:
                positive_failures.append({"task_id": task_id, "dimension": dimension})
        for dimension, mutator in MUTATORS.items():
            mutated = copy.deepcopy(gold)
            mutated_evidence = copy.deepcopy(evidence)
            if not mutator(mutated, expected):
                continue
            if dimension == "assigned_script_use":
                mutated_evidence["opened_paths"] = []
                mutated_evidence["tool_calls"] = []
            scores = score({"id": task_id}, mutated, mutated_evidence)
            negative_cases[dimension].append(
                {
                    "case_id": _stable_digest(
                        {"task_id": task_id, "dimension": dimension, "version": 2}
                    ),
                    "task_id": task_id,
                    "target_rejected": scores.get(dimension) is False,
                }
            )
    dimensions: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        cases = negative_cases[dimension]
        dimensions[dimension] = {
            "positive_cases": len(_rows()),
            "positive_failures": sum(
                item["dimension"] == dimension for item in positive_failures
            ),
            "targeted_negative_cases": len(cases),
            "targeted_false_accepts": sum(
                item["target_rejected"] is not True for item in cases
            ),
            "case_ids_digest": _stable_digest([item["case_id"] for item in cases]),
        }
    minimum_negative_cases = 8
    passed = not positive_failures and all(
        value["targeted_negative_cases"] >= minimum_negative_cases
        and value["targeted_false_accepts"] == 0
        for value in dimensions.values()
    )
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "id": "anthropic-skill-creator-confirmatory-scorer-validation-v2",
        "status": "passed" if passed else "failed",
        "private_inputs_serialized": False,
        "scorer_sha256": _sha256(SCORER),
        "private_labels_sha256": _sha256(LABELS),
        "minimum_targeted_negative_cases_per_dimension": minimum_negative_cases,
        "dimensions": dimensions,
    }
    receipt["receipt_digest"] = _stable_digest(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = validate()
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
