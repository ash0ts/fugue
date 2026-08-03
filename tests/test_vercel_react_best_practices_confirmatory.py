from __future__ import annotations

import base64
import copy
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    _run_custom_verifier,
    check_comparison,
    load_comparison,
)

EXAMPLE = Path("examples/comparisons/vercel-react-best-practices-upgrade")
SPEC = EXAMPLE / "confirmatory-v1.yaml"
TASKS = EXAMPLE / "confirmatory-tasks.jsonl"
PRIVATE = EXAMPLE / "confirmatory-private-labels.jsonl"
LOCK = EXAMPLE / "confirmatory-fixtures.lock.json"
PREREGISTRATION = EXAMPLE / "confirmatory-preregistration.json"
CATALOG = EXAMPLE / "conference_fixture_catalog.py"
SCORER = EXAMPLE / "vercel_confirmatory_scorer.py"
HOST_VERIFIER = EXAMPLE / "host_node_verifier.cjs"
PREPARE = EXAMPLE / "prepare_confirmatory_fixtures.py"
NODE_IMAGE = "node:22-bookworm-slim@sha256:53ada149d435c38b14476cb57e4a7da73c15595aba79bd6971b547ceb6d018bf"
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
    return runpy.run_path(PREPARE.as_posix())["_expected_contract"](fixture)


def _host_receipt(passed: bool, task_id: str) -> dict:
    return {
        "schema_version": 2,
        "kind": "post_trial_verifier_receipt",
        "evaluator_id": "vercel-confirmatory",
        "task_id": task_id,
        "attempt_id": "b" * 64,
        "status": "passed" if passed else "failed",
        "failure_kind": None if passed else "public_test_failed",
        "runtime": "node-v22.23.0",
        "command": ["node", "--test", "tests/task.test.mjs"],
        "exit_code": 0 if passed else 1,
        "test_count": 1,
        "pass_count": 1 if passed else 0,
        "fail_count": 0 if passed else 1,
        "output_sha256": "a" * 64,
        "base_archive_sha256": "c" * 64,
        "public_test_sha256": "d" * 64,
        "submitted_artifact_sha256": "e" * 64,
        "final_tree_sha256": "f" * 64,
        "verifier_source_sha256": "1" * 64,
        "runtime_profile_id": "node22-verifier-v1",
        "runtime_profile_digest": "2" * 64,
        "runtime_image": "node@sha256:" + "3" * 64,
        "runtime_platform": "linux/arm64",
        "runtime_image_id": "sha256:" + "4" * 64,
        "runtime_lock_digest": "5" * 64,
        "receipt_digest": "6" * 64,
    }


def _run_pinned_host_verifier(
    tmp_path: Path,
    fixture: dict,
    output: object,
    *,
    mutate_expected=None,
) -> subprocess.CompletedProcess[str]:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker is required for the pinned Node verifier test")
    inspected = subprocess.run(
        [docker, "image", "inspect", NODE_IMAGE],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode != 0:
        pytest.skip("the digest-pinned Node verifier image is unavailable")
    module = runpy.run_path(PREPARE.as_posix())
    expected = module["_expected_contract"](fixture)
    expected["base_archive_base64"] = base64.b64encode(
        module["_archive_bytes"](fixture)
    ).decode("ascii")
    if mutate_expected is not None:
        mutate_expected(expected)
    input_root = tmp_path / f"input-{len(tuple(tmp_path.iterdir()))}"
    input_root.mkdir(mode=0o700)
    input_root.chmod(0o700)
    (input_root / "host_node_verifier.cjs").write_bytes(HOST_VERIFIER.read_bytes())
    (input_root / "input.json").write_text(
        json.dumps(
            {
                "reference": {
                    "task": {"id": fixture["id"]},
                    "output": output,
                    "expected": expected,
                }
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (input_root / "input.json").chmod(0o600)
    return subprocess.run(
        [
            docker,
            "run",
            "--rm",
            "--pull",
            "never",
            "--platform",
            "linux/arm64",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--mount",
            f"type=bind,src={input_root.resolve()},dst=/input,readonly",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            NODE_IMAGE,
            "node",
            "/input/host_node_verifier.cjs",
            "/input/input.json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


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
    console = yaml.safe_load(
        (EXAMPLE / "confirmatory-study-console.yaml").read_text(encoding="utf-8")
    )
    tasks = _jsonl(TASKS)
    deterministic = next(
        item for item in spec.evaluators if item.type == "deterministic"
    )

    assert spec.schema_version == 3
    assert spec.id == "vercel-react-best-practices-confirmatory-v1"
    assert spec.baseline.skills == ("vercel-react-best-practices-before",)
    assert spec.candidate.skills == ("vercel-react-best-practices-after",)
    assert spec.execution.research_id == "vercel-react-best-practices-confirmatory-v1"
    assert console["research"]["id"] == spec.execution.research_id
    assert (
        f"wandb/{console['wandb']['project']}" == spec.execution.evidence_project
    )
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
    assert (
        spec.execution.scheduling_seed
        == "community-skill-upgrade-confirmatory-campaign-v1"
    )
    assert spec.execution.evidence_checkpoint_cells == 2
    assert spec.execution.max_cost_usd == 1640
    assert spec.execution.reserve_per_attempt_usd == 8.4
    assert spec.execution.qualification_inputs == {
        "campaign_manifest_sha256": "examples/comparisons/community-skill-upgrades/conference-campaign-manifest.json",
        "campaign_preregistration_sha256": "examples/comparisons/community-skill-upgrades/conference-preregistration.json",
        "confirmatory_budget_policy_sha256": "examples/comparisons/community-skill-upgrades/confirmatory-budget-policy-v1.json",
        "confirmatory_analysis_profile_sha256": "examples/comparisons/community-skill-upgrades/confirmatory-analysis-profiles.json",
        "repository_preregistration_sha256": "examples/comparisons/vercel-react-best-practices-upgrade/confirmatory-preregistration.json",
    }
    assert len(tasks) * 2 * spec.execution.attempts == 192
    assert deterministic.verifier is not None
    assert deterministic.verifier.type == "node_test"
    assert deterministic.verifier.source == HOST_VERIFIER.as_posix()
    assert deterministic.verifier.runtime == "node22-verifier-v1"
    assert deterministic.verifier.dimension == "verification"


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
    assert all(
        "target rule" not in task["input"]["question"].casefold() for task in tasks
    )
    assert all(len(item["critical_dimensions"]) == 6 for item in tasks)
    private_text = PRIVATE.read_text(encoding="utf-8")
    assert "hidden_test" not in private_text
    assert "host-only verifier" not in private_text
    assert "# tests 2" not in private_text


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
            {"id": fixture["id"]},
            _output(fixture, "base", 1),
            {
                **evidence,
                "host_verifier_receipt": _host_receipt(False, fixture["id"]),
            },
        )
        gold = score(
            {"id": fixture["id"]},
            _output(fixture, "gold", 0),
            {
                **evidence,
                "host_verifier_receipt": _host_receipt(True, fixture["id"]),
            },
        )
        assert set(base) == DIMENSIONS
        assert set(gold) == DIMENSIONS
        assert not all(base.values()), (fixture["id"], base)
        assert all(gold.values()), (fixture["id"], gold)


def test_canonical_readiness_proves_all_base_fail_gold_pass_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = _scorer()

    def fake_verifier(evaluator, *, task, output, **kwargs):
        passed = output["verification"][0]["exit_code"] == 0
        return {
            "score": 1.0 if passed else 0.0,
            "reason": "offline executable qualification fixture",
            "details": _host_receipt(passed, task["id"]),
        }

    def fake_inline_runner(*, source, evidence, reference, profile, **kwargs):
        details = score(
            reference["task"],
            reference["output"],
            {**evidence, "expected": reference["expected"]},
        )
        serialized_input = json.dumps(
            {"evidence": evidence, "reference": reference},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        input_unsigned = {
            "schema_version": 1,
            "status": "bound",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "input_bytes": len(serialized_input),
            "input_sha256": hashlib.sha256(serialized_input).hexdigest(),
            "evidence_digest": stable_digest(evidence),
            "reference_digest": stable_digest(reference),
            "reference_output_digest": stable_digest(reference["output"]),
            "runtime_profile_id": profile.id,
            "runtime_profile_digest": profile.profile_digest,
            "runtime_image": profile.image,
            "runtime_platform": profile.platform,
        }
        runtime_unsigned = {
            "schema_version": 1,
            "status": "verified_absent",
            "container_name_sha256": "7" * 64,
        }
        return {
            "score": 1.0 if all(details.values()) else 0.0,
            "reason": "offline qualification fixture",
            "details": details,
            "fugue_input_receipt": {
                **input_unsigned,
                "receipt_digest": stable_digest(input_unsigned),
            },
            "fugue_runtime_receipt": {
                **runtime_unsigned,
                "receipt_digest": stable_digest(runtime_unsigned),
            },
        }

    monkeypatch.setattr("fugue.bench.comparison._run_custom_verifier", fake_verifier)
    monkeypatch.setattr(
        "fugue.bench.task_authoring.run_inline_scorer",
        fake_inline_runner,
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
        {
            "expected": _expected(fixture),
            "host_verifier_receipt": _host_receipt(False, fixture["id"]),
        },
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
        {
            "expected": _expected(fixture),
            "host_verifier_receipt": _host_receipt(True, fixture["id"]),
        },
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
        "host_node_verifier.cjs",
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


def test_preparation_builds_deterministic_dependency_free_archives(
    tmp_path: Path,
) -> None:
    module = runpy.run_path((EXAMPLE / "prepare_confirmatory_fixtures.py").as_posix())
    fixture = _fixtures()[0]
    public = module["_public_files"](fixture, gold=False)
    package = json.loads(public["package.json"])
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    first_record = module["_build_archive"](fixture, first)
    second_record = module["_build_archive"](fixture, second)
    expected = module["_expected_contract"](fixture)

    assert "dependencies" not in package
    assert "devDependencies" not in package
    assert first.read_bytes() == second.read_bytes()
    assert first_record["sha256"] == second_record["sha256"]
    assert first_record["file_count"] == len(public)
    assert expected["base_archive_sha256"] == first_record["sha256"]
    assert expected["base_archive_size"] == first.stat().st_size
    assert expected["base_archive_file_count"] == len(public)
    assert (
        expected["base_archive_manifest_digest"]
        == first_record["archive_manifest_digest"]
    )
    assert expected["public_test_sha256"] == first_record["public_test_sha256"]
    assert {item["path"] for item in expected["base_archive_files"]} == set(public)
    assert all(b"gold" not in value.lower() for value in public.values())


@pytest.mark.parametrize(
    "relative",
    (
        "../private-labels.jsonl",
        "/tmp/escape.mjs",
        "app\\escape.mjs",
        "app//escape.mjs",
        "README.md",
        "readme.MD",
        "package.json",
        "Tests/Task.Test.MJS",
    ),
)
def test_preparation_rejects_unsafe_or_reserved_target_paths(relative: str) -> None:
    module = runpy.run_path(PREPARE.as_posix())
    fixture = copy.deepcopy(_fixtures()[0])
    fixture["target_files"] = {
        relative: {"base": "export const value = 0;", "gold": "export const value = 1;"}
    }

    with pytest.raises(RuntimeError, match="fixture path|target path collides"):
        module["_public_files"](fixture, gold=False)
    with pytest.raises(RuntimeError, match="fixture path|target path collides"):
        module["_archive_bytes"](fixture)


def test_preparation_rejects_casefold_target_collisions() -> None:
    module = runpy.run_path(PREPARE.as_posix())
    fixture = copy.deepcopy(_fixtures()[0])
    fixture["target_files"] = {
        "app/Team.mjs": {"base": "base", "gold": "gold"},
        "app/team.mjs": {"base": "base", "gold": "gold"},
    }

    with pytest.raises(RuntimeError, match="target path collides"):
        module["_public_files"](fixture, gold=False)


def test_preparation_write_tree_rejects_escape_and_casefold_collisions(
    tmp_path: Path,
) -> None:
    module = runpy.run_path(PREPARE.as_posix())

    with pytest.raises(RuntimeError, match="unsafe confirmatory fixture path"):
        module["_write_tree"](tmp_path / "tree", {"../escape": b"private"})
    with pytest.raises(RuntimeError, match="colliding confirmatory fixture path"):
        module["_write_tree"](
            tmp_path / "tree",
            {"app/Team.mjs": b"one", "app/team.mjs": b"two"},
        )
    assert not (tmp_path / "escape").exists()


def test_runtime_source_lock_requires_exact_unique_source_records(
    tmp_path: Path,
) -> None:
    module = runpy.run_path(PREPARE.as_posix())

    def write_lock(paths: tuple[str, ...]) -> None:
        value = {
            "schema_version": 1,
            "sources": [
                {
                    "path": relative,
                    "sha256": hashlib.sha256((EXAMPLE / relative).read_bytes()).hexdigest(),
                    "size": (EXAMPLE / relative).stat().st_size,
                }
                for relative in paths
            ],
        }
        value["manifest_digest"] = _stable_digest(value)
        lock_path.write_text(json.dumps(value), encoding="utf-8")

    lock_path = tmp_path / "sources.lock.json"
    module["_load_and_verify_source_lock"].__globals__["SOURCE_LOCK"] = lock_path
    exact = tuple(module["_FROZEN_SOURCE_NAMES"])
    write_lock(exact)
    assert module["_load_and_verify_source_lock"]()["manifest_digest"]

    write_lock(exact[:-1])
    with pytest.raises(RuntimeError, match="not exact and unique"):
        module["_load_and_verify_source_lock"]()
    write_lock(exact[:-1] + (exact[0],))
    with pytest.raises(RuntimeError, match="not exact and unique"):
        module["_load_and_verify_source_lock"]()


def test_source_lock_recipe_binds_preparation_and_verifier_implementation() -> None:
    documentation = (EXAMPLE / "CONFIRMATORY.md").read_text(encoding="utf-8")
    for required in (
        "prepare_confirmatory_fixtures.py",
        "host_node_verifier.cjs",
        "confirmatory-fixtures.lock.json",
        "preflight.receipt.json",
    ):
        assert "--extra " in documentation
        assert required in documentation


def test_preparation_rejects_infrastructure_errors_as_base_failures() -> None:
    module = runpy.run_path((EXAMPLE / "prepare_confirmatory_fixtures.py").as_posix())
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
            "# tests 1\n# pass 0\n# fail 1\n"
        ),
        stderr="",
    )
    module["_validate_node_test_receipt"](fixture, base_receipt, should_pass=False)


def test_confirmatory_scorer_fits_the_locked_inline_runtime_limit() -> None:
    assert len(SCORER.read_bytes()) < 32_768
    assert len(HOST_VERIFIER.read_bytes()) < 64_000


def test_host_verifier_denies_private_input_and_child_process_access() -> None:
    source = HOST_VERIFIER.read_text(encoding="utf-8")

    assert '"--permission"' in source
    assert '"--experimental-test-isolation=none"' in source
    assert "--allow-fs-read=${root}" in source
    assert "--allow-fs-read=/input" not in source
    assert "--allow-child-process" not in source
    assert "statSync(dirname(inputPath)).mode & 0o077" in source
    assert 'command: ["node", "--test", PUBLIC_TEST_PATH]' in source
    assert "hidden_test" not in source
    assert "host-only verifier" not in source


def test_pinned_host_verifier_passes_gold_and_reports_base_as_candidate_failure(
    tmp_path: Path,
) -> None:
    fixture = _fixtures()[0]
    gold = _output(fixture, "gold", 0)
    gold["summary"] += " Résumé 🚀"
    base = _output(fixture, "base", 1)

    gold_run = _run_pinned_host_verifier(tmp_path, fixture, gold)
    base_run = _run_pinned_host_verifier(tmp_path, fixture, base)

    assert gold_run.returncode == 0, gold_run.stderr
    assert base_run.returncode == 0, base_run.stderr
    gold_result = json.loads(gold_run.stdout)
    base_result = json.loads(base_run.stdout)
    assert gold_result["score"] == 1
    assert gold_result["reason"] == "public_test_passed_on_frozen_repository"
    assert gold_result["details"]["status"] == "passed"
    assert gold_result["details"]["command"] == [
        "node",
        "--test",
        "tests/task.test.mjs",
    ]
    assert gold_result["details"]["test_count"] == 1
    assert gold_result["details"]["pass_count"] == 1
    assert gold_result["details"]["fail_count"] == 0
    assert gold_result["details"]["submitted_artifact_sha256"] == stable_digest(gold)
    assert base_result["score"] == 0
    assert base_result["reason"] == "candidate_failure:public_test_failed"
    assert base_result["details"]["status"] == "failed"
    assert base_result["details"]["failure_kind"] == "public_test_failed"
    assert base_result["details"]["test_count"] == 1


def test_pinned_host_verifier_fails_closed_without_turning_candidates_into_infra(
    tmp_path: Path,
) -> None:
    fixture = _fixtures()[0]
    gold = _output(fixture, "gold", 0)
    target = next(iter(fixture["target_files"]))

    cases = []
    wrong_task = json.loads(json.dumps(gold))
    wrong_task["task_id"] = "wrong-task"
    cases.append((wrong_task, "output_identity_mismatch"))
    extra_path = json.loads(json.dumps(gold))
    extra_path["files"]["unexpected.mjs"] = "export const unexpected = true;\n"
    cases.append((extra_path, "submitted_files_do_not_match_allowlist"))
    spoof = json.loads(json.dumps(gold))
    spoof["files"][target] = "process.exit(0);\n"
    cases.append((spoof, "submitted_file_can_interfere_with_test_runner"))
    log_flood = json.loads(json.dumps(gold))
    log_flood["files"][target] = (
        "console.log('x'.repeat(3000000));\n" + log_flood["files"][target]
    )
    cases.append((log_flood, "public_test_output_limit_exceeded"))
    invalid_unicode = json.loads(json.dumps(gold))
    invalid_unicode["files"][target] = "export const value = '\ud800';\n"
    cases.append((invalid_unicode, "submitted_file_is_not_bounded_utf8_text"))

    for output, failure_kind in cases:
        completed = _run_pinned_host_verifier(tmp_path, fixture, output)
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["score"] == 0
        assert result["reason"] == f"candidate_failure:{failure_kind}"
        assert result["details"]["failure_kind"] == failure_kind

    private_read = json.loads(json.dumps(gold))
    private_read["files"][target] = (
        "import { readFileSync } from 'node:fs';\n"
        "readFileSync('/input/input.json', 'utf8');\n" + private_read["files"][target]
    )
    private_run = _run_pinned_host_verifier(tmp_path, fixture, private_read)
    assert private_run.returncode == 0, private_run.stderr
    private_result = json.loads(private_run.stdout)
    assert private_result["score"] == 0
    assert private_result["reason"] == "candidate_failure:public_test_failed"
    assert "base_archive_base64" not in private_run.stdout
    assert "ANTHROPIC" not in private_run.stdout

    corrupt_lock = _run_pinned_host_verifier(
        tmp_path,
        fixture,
        gold,
        mutate_expected=lambda expected: expected.__setitem__(
            "base_archive_sha256", "0" * 64
        ),
    )
    assert corrupt_lock.returncode != 0
    assert corrupt_lock.stdout == ""
    assert "frozen archive bytes do not match their lock" in corrupt_lock.stderr


def test_canonical_host_verifier_runner_injects_wraps_and_binds_exact_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker is required for the canonical verifier integration test")
    inspected = subprocess.run(
        [docker, "image", "inspect", NODE_IMAGE],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode != 0:
        pytest.skip("the digest-pinned Node verifier image is unavailable")
    image = json.loads(inspected.stdout)[0]
    fixture = _fixtures()[0]
    task = _jsonl(TASKS)[0]
    expected = _expected(fixture)
    output = _output(fixture, "gold", 0)
    spec = load_comparison(SPEC, repo_root=Path.cwd())
    evaluator = next(
        item for item in spec.evaluators if item.id == "vercel-confirmatory"
    )
    module = runpy.run_path(PREPARE.as_posix())

    profiles = tmp_path / "configs/fugue/task-authoring/profiles.yaml"
    profiles.parent.mkdir(parents=True)
    profiles.write_bytes(
        Path("configs/fugue/task-authoring/profiles.yaml").read_bytes()
    )
    verifier = tmp_path / evaluator.verifier.source
    verifier.parent.mkdir(parents=True)
    verifier.write_bytes(HOST_VERIFIER.read_bytes())
    archive = tmp_path / task["resources"][0]["path"]
    module["_build_archive"](fixture, archive)

    runtime_lock = {
        "image_id": image["Id"],
        "lock_digest": "7" * 64,
    }
    monkeypatch.setattr(
        "fugue.bench.comparison._read_evaluator_runtime_lock",
        lambda profile, **kwargs: runtime_lock,
    )
    attempt_id = "8" * 64
    payload = _run_custom_verifier(
        evaluator,
        task=task,
        output=output,
        expected=expected,
        evidence={"task_id": task["id"], "attempt_id": attempt_id},
        repo_root=tmp_path,
    )
    receipt = payload["details"]

    assert payload["score"] == 1
    assert receipt["schema_version"] == 2
    assert receipt["kind"] == "post_trial_verifier_receipt"
    assert receipt["evaluator_id"] == "vercel-confirmatory"
    assert receipt["task_id"] == task["id"]
    assert receipt["attempt_id"] == attempt_id
    assert receipt["status"] == "passed"
    assert receipt["base_archive_sha256"] == expected["base_archive_sha256"]
    assert receipt["public_test_sha256"] == expected["public_test_sha256"]
    assert receipt["submitted_artifact_sha256"] == stable_digest(output)
    assert receipt["runtime_profile_id"] == "node22-verifier-v1"
    assert receipt["runtime_platform"] == "linux/arm64"
    assert receipt["runtime_image_id"] == image["Id"]
    assert receipt["runtime_cleanup"]["status"] == "verified_absent"
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    assert receipt["receipt_digest"] == stable_digest(unsigned)
