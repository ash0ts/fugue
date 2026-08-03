from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pytest

from fugue.bench.comparison import check_comparison, load_comparison

EXAMPLE = Path("examples/comparisons/vercel-react-best-practices-upgrade")
SPEC = EXAMPLE / "confirmatory-v1.yaml"
TASKS = EXAMPLE / "confirmatory-tasks.jsonl"
PRIVATE = EXAMPLE / "confirmatory-private-labels.jsonl"
LOCK = EXAMPLE / "confirmatory-fixtures.lock.json"
PREREGISTRATION = EXAMPLE / "confirmatory-preregistration.json"
CATALOG = EXAMPLE / "conference_fixture_catalog.py"
SCORER = EXAMPLE / "vercel_confirmatory_scorer.py"
EXPECTED_IDS = (
    "vr-dev-signed-in-nonmember-action",
    "vr-dev-unauthenticated-direct-action",
    "vr-dev-rsc-primitive-derived-array",
    "vr-dev-rsc-object-filtered-array",
    "vr-dev-layout-interleaved-measurement",
    "vr-dev-large-array-max-rangeerror",
    "vr-dev-use-latest-layout-read",
    "vr-dev-window-event-handler-typing",
    "vr-holdout-cross-tenant-delete-action",
    "vr-holdout-admin-or-owner-action",
    "vr-holdout-action-validation-order",
    "vr-holdout-readonly-action-control",
    "vr-holdout-rsc-primitive-sort",
    "vr-holdout-rsc-object-map-clone",
    "vr-holdout-rsc-derived-scalar",
    "vr-holdout-rsc-expensive-derivation-control",
    "vr-holdout-layout-two-write-two-read",
    "vr-holdout-layout-write-only-control",
    "vr-holdout-large-array-min",
    "vr-holdout-large-array-sum-control",
    "vr-holdout-use-latest-child-layout-effect",
    "vr-holdout-use-latest-passive-only-control",
    "vr-holdout-mouseevent-handler",
    "vr-holdout-keyboardevent-handler",
)
DIMENSIONS = {
    "artifact_validity",
    "requested_change",
    "repository_grounding",
    "behavior_preservation",
    "verification",
    "scope_safety",
}


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fixtures() -> tuple[dict, ...]:
    return tuple(runpy.run_path(CATALOG.as_posix())["FIXTURES"])


def _scorer():
    return runpy.run_path(SCORER.as_posix())["score"]


def _expected(fixture: dict) -> dict:
    return {
        "required_file_paths": list(fixture["target_files"]),
        "allowed_file_paths": list(fixture["target_files"]),
        "required_inspected_paths": fixture["required_inspected_paths"],
        "public_test_name": fixture["public_test_name"],
        "verifier": fixture["verifier"],
    }


def _output(fixture: dict, side: str, exit_code: int) -> dict:
    passed = int(exit_code == 0)
    failed = int(exit_code != 0)
    return {
        "schema_version": 1,
        "task_id": fixture["id"],
        "status": "completed",
        "files": {
            path: sources[side] for path, sources in fixture["target_files"].items()
        },
        "inspected_paths": fixture["required_inspected_paths"],
        "verification": [
            {
                "command": "node --test",
                "exit_code": exit_code,
                "stdout": (
                    f"# Subtest: {fixture['public_test_name']}\n"
                    f"# pass {passed}\n# fail {failed}\n"
                ),
            }
        ],
        "summary": "A complete bounded summary of the submitted maintenance change.",
        "limitations": ["Synthetic receipt used only by the scorer unit test."],
    }


def test_confirmatory_spec_freezes_v3_identity_projects_and_192_cells() -> None:
    spec = load_comparison(SPEC, repo_root=Path.cwd())
    tasks = _jsonl(TASKS)

    assert spec.schema_version == 3
    assert spec.id == "vercel-react-best-practices-confirmatory-v1"
    assert spec.baseline.skills == ("vercel-react-best-practices-before",)
    assert spec.candidate.skills == ("vercel-react-best-practices-after",)
    assert spec.execution.research_id == "vercel-react-best-practices-confirmatory-v1"
    assert spec.execution.source_evidence_project == (
        "wandb/fugue-vercel-react-best-practices-source-v1"
    )
    assert spec.execution.evidence_project == (
        "wandb/fugue-vercel-react-best-practices-confirmatory-v1"
    )
    assert spec.execution.model == "anthropic/claude-sonnet-5"
    assert spec.execution.harnesses == ("claude-code",)
    assert spec.execution.attempts == 4
    assert spec.execution.concurrency == 1
    assert spec.execution.evidence_checkpoint_cells == 1
    assert spec.execution.max_cost_usd == 1640
    assert spec.execution.reserve_per_attempt_usd == 8.4
    assert len(tasks) * 2 * spec.execution.attempts == 192


def test_confirmatory_has_exact_frozen_development_and_holdout_task_ids() -> None:
    tasks = _jsonl(TASKS)
    labels = _jsonl(PRIVATE)
    fixtures = _fixtures()

    assert tuple(item["id"] for item in tasks) == EXPECTED_IDS
    assert tuple(item["id"] for item in labels) == EXPECTED_IDS
    assert tuple(item["id"] for item in fixtures) == EXPECTED_IDS
    assert sum(item["partition"] == "discovery" for item in tasks) == 8
    assert sum(item["partition"] == "holdout" for item in tasks) == 16
    assert all("expected" not in task and "gold" not in task for task in tasks)
    assert all("base_output" in label and "gold_output" in label for label in labels)
    assert all("target rule" not in task["input"]["question"].casefold() for task in tasks)
    assert all(len(item["critical_dimensions"]) == 6 for item in tasks)


def test_agent_visible_tests_are_behavioral_and_do_not_read_source_text() -> None:
    for fixture in _fixtures():
        public_test = fixture["public_test_source"]
        assert "node:test" in public_test
        assert "node:assert/strict" in public_test
        assert "node:fs" not in public_test
        assert "readFile" not in public_test
        assert "RegExp" not in public_test
        assert ".match(" not in public_test
        assert ".includes(" not in public_test


def test_private_verifiers_cover_delta_and_control_families() -> None:
    fixtures = _fixtures()
    families = {item["family"] for item in fixtures}
    kinds = {item["verifier"]["kind"] for item in fixtures}

    assert families == {
        "server-action-security",
        "rsc-serialization",
        "dom-batching-control",
        "large-array-control",
        "hook-timing-control",
        "event-signature-control",
    }
    assert kinds == {
        "server_action",
        "rsc_props",
        "dom_batch",
        "dom_write_control",
        "array_extreme",
        "array_sum_control",
        "hook_timing",
        "event_signature",
    }


def test_independent_scorer_rejects_every_base_and_accepts_every_gold() -> None:
    score = _scorer()
    for fixture in _fixtures():
        evidence = {"expected": _expected(fixture)}
        base = score(
            {"id": fixture["id"]}, _output(fixture, "base", 1), evidence
        )
        gold = score(
            {"id": fixture["id"]}, _output(fixture, "gold", 0), evidence
        )
        assert set(base) == DIMENSIONS
        assert set(gold) == DIMENSIONS
        assert not all(base.values()), (fixture["id"], base)
        assert all(gold.values()), (fixture["id"], gold)


def test_canonical_readiness_proves_all_base_fail_gold_pass_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = _scorer()

    def fake_inline_runner(*, source, evidence, reference, profile, limits):
        assert 'if __name__ == "__main__":' in source
        assert profile.id == "python312-sandbox-v1"
        details = score(
            reference["task"],
            reference["output"],
            {**evidence, "expected": reference["expected"]},
        )
        return {
            "score": 1.0 if all(details.values()) else 0.0,
            "reason": "offline qualification fixture",
            "details": details,
        }

    monkeypatch.setattr(
        "fugue.bench.task_authoring.run_inline_scorer", fake_inline_runner
    )
    readiness = check_comparison(
        load_comparison(SPEC, repo_root=Path.cwd()), repo_root=Path.cwd()
    )

    assert readiness.base_failures == 24
    assert readiness.gold_passes == 24
    assert not any("missing base_output" in warning for warning in readiness.warnings)
    assert not any("missing gold_output" in warning for warning in readiness.warnings)


def test_forged_public_receipt_cannot_turn_base_source_into_a_pass() -> None:
    fixture = _fixtures()[0]
    score = _scorer()
    forged = _output(fixture, "base", 0)
    result = score(
        {"id": fixture["id"]},
        forged,
        {"expected": _expected(fixture)},
    )

    assert result["artifact_validity"] is True
    assert result["requested_change"] is False
    assert result["verification"] is False


def test_scope_and_repository_grounding_are_derived_from_submitted_artifacts() -> None:
    fixture = _fixtures()[0]
    score = _scorer()
    unsafe = _output(fixture, "gold", 0)
    unsafe["files"]["package.json"] = '{"dependencies":{"surprise":"latest"}}'
    unsafe["inspected_paths"] = ["README.md"]
    result = score(
        {"id": fixture["id"]},
        unsafe,
        {"expected": _expected(fixture)},
    )

    assert result["requested_change"] is True
    assert result["repository_grounding"] is False
    assert result["scope_safety"] is False


def test_confirmatory_source_lock_and_preregistration_are_canonical() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    supplied = lock.pop("manifest_digest")
    assert supplied == _stable_digest(lock)
    assert {record["path"] for record in lock["sources"]} == {
        "conference_fixture_catalog.py",
        "confirmatory-tasks.jsonl",
        "confirmatory-private-labels.jsonl",
        "vercel_confirmatory_scorer.py",
        "confirmatory-preregistration.json",
        "prepare_confirmatory_fixtures.py",
    }
    for record in lock["sources"]:
        path = EXAMPLE / record["path"]
        assert path.stat().st_size == record["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    preregistration_digest = preregistration.pop("preregistration_digest")
    assert preregistration_digest == _stable_digest(preregistration)
    assert preregistration["execution"]["cells"] == 192
    assert preregistration["tasks"] == {
        "development": 8,
        "holdout": 16,
        "total": 24,
        "families": {
            "dom-batching-control": 3,
            "event-signature-control": 3,
            "hook-timing-control": 3,
            "large-array-control": 3,
            "rsc-serialization": 6,
            "server-action-security": 6,
        },
    }


def test_preparation_builds_deterministic_dependency_free_archives(tmp_path: Path) -> None:
    module = runpy.run_path(
        (EXAMPLE / "prepare_confirmatory_fixtures.py").as_posix()
    )
    fixture = _fixtures()[0]
    public = module["_public_files"](fixture, gold=False)
    package = json.loads(public["package.json"])
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    first_record = module["_build_archive"](fixture, first)
    second_record = module["_build_archive"](fixture, second)

    assert "dependencies" not in package
    assert "devDependencies" not in package
    assert first.read_bytes() == second.read_bytes()
    assert first_record["sha256"] == second_record["sha256"]
    assert first_record["file_count"] == len(public)
    assert all(b"gold" not in value.lower() for value in public.values())


def test_preparation_rejects_infrastructure_errors_as_base_failures() -> None:
    module = runpy.run_path(
        (EXAMPLE / "prepare_confirmatory_fixtures.py").as_posix()
    )
    fixture = _fixtures()[0]
    infrastructure_error = subprocess.CompletedProcess(
        args=["docker", "run"],
        returncode=125,
        stdout="",
        stderr="docker daemon temporarily unavailable",
    )

    with pytest.raises(RuntimeError, match="failing Node test receipt"):
        module["_validate_node_test_receipt"](
            fixture, infrastructure_error, should_pass=False
        )

    base_receipt = subprocess.CompletedProcess(
        args=["node", "--test"],
        returncode=1,
        stdout=(
            "TAP version 13\n"
            f"# Subtest: {fixture['public_test_name']}\n"
            f"# Subtest: host-only verifier for {fixture['id']}\n"
            "# tests 2\n# pass 0\n# fail 2\n"
        ),
        stderr="",
    )
    module["_validate_node_test_receipt"](
        fixture, base_receipt, should_pass=False
    )


def test_confirmatory_scorer_fits_the_locked_inline_runtime_limit() -> None:
    assert len(SCORER.read_bytes()) < 32_768
