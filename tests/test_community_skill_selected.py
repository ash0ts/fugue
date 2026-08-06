from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from fugue.bench.comparison import (
    _post_trial_verifier_lock,
    _validate_custom_scorer_source,
    load_comparison,
)

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples/comparisons/community-skill-selected-v1"
LANES = (
    "superpowers-writing-plans",
    "anthropic-skill-creator",
    "vercel-react-best-practices",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _files_digest(files: dict[str, str]) -> str:
    return _digest(
        {
            path: hashlib.sha256(content.encode()).hexdigest()
            for path, content in files.items()
        }
    )


def test_manifest_has_three_bounded_unpooled_public_lanes() -> None:
    manifest = _json(EXAMPLE / "campaign-manifest.json")

    assert [lane["id"] for lane in manifest["lanes"]] == list(LANES)
    assert sum(stage["cells"] for stage in manifest["public_schedule"]) == 48
    assert manifest["fixed_conditions"] == {
        "harness": "claude-code",
        "model": "anthropic/claude-sonnet-5",
        "environment": "docker",
        "attempts": 2,
        "global_active_cells_max": 3,
        "study_active_cells_max": 2,
        "wave_size": 4,
    }
    assert "never pooled" in manifest["claim_boundary"]
    assert manifest["judge"]["generation"] == "external_operator_procedure"
    assert manifest["budget"]["public_development_agent_max_usd"] == 120
    assert manifest["budget"]["public_development_judge_max_usd"] == 4.8
    assert manifest["budget"]["public_development_combined_max_usd"] == 124.8
    assert manifest["budget"]["infrastructure_replacement_contingency_max_usd"] == 10
    assert (
        manifest["budget"][
            "maximum_primary_spend_if_all_96_logical_cells_and_four_replacements_run_usd"
        ]
        == 259.6
    )
    assert "optional advisory evaluator" in manifest["judge"][
        "study_evaluator_binding"
    ]


def test_specs_bind_exact_revisions_and_only_public_development_tasks() -> None:
    resource_lock = _json(EXAMPLE / "task-resources.lock.json")
    campaign = _json(EXAMPLE / "campaign-manifest.json")
    for lane in LANES:
        lane_root = EXAMPLE / lane
        spec = load_comparison(lane_root / "comparison.yaml", repo_root=ROOT)
        lock = _json(lane_root / "skill-revisions.lock.json")
        tasks = _rows(lane_root / "tasks.jsonl")

        assert spec.schema_version == 3
        assert spec.changed == ("skills",)
        assert spec.baseline.skills == (lock["baseline"]["id"],)
        assert spec.candidate.skills == (lock["candidate"]["id"],)
        assert spec.execution.model == "anthropic/claude-sonnet-5"
        assert spec.execution.harnesses == ("claude-code",)
        assert spec.execution.environment == {"type": "docker"}
        assert spec.execution.schedule is not None
        assert [stage["id"] for stage in spec.execution.schedule["stages"]] == [
            "checkpoint",
            "complete-development-attempt-0",
            "repeat-development",
        ]
        assert spec.execution.schedule["worker_limit"] == 2
        assert spec.execution.schedule["wave_size"] == 4
        assert spec.execution.schedule["infrastructure_retry_limit"] == 1
        assert spec.execution.schedule["maximum_physical_executions"] == 17
        assert spec.execution.schedule["maximum_in_flight_cost_usd"] == 5.2
        assert spec.execution.max_cost_usd == 44.1
        assert spec.execution.reserve_per_attempt_usd == 2.5
        assert spec.execution.schedule["coordination"] == {
            "group_id": "community-skill-selected-v1",
            "worker_limit": 3,
            "maximum_physical_executions": 100,
            "total_cost_usd": 270,
            "maximum_in_flight_cost_usd": 7.8,
        }
        assert spec.execution.schedule["stages"][0]["trial_indexes"] == [0]
        assert spec.execution.schedule["stages"][0]["pair_complete"] is True
        assert spec.execution.schedule["stages"][1]["trial_indexes"] == [0]
        assert spec.execution.schedule["stages"][2]["trial_indexes"] == [1]
        assert len(tasks) == 4
        assert len({row["id"] for row in tasks}) == 4
        assert all(row["partition"] == "qualification" for row in tasks)
        assert all("expected" not in row for row in tasks)
        locked = {
            row["task_id"]: row for row in resource_lock["lanes"][lane]
        }
        assert all(
            row["resources"]
            == [
                {
                    "path": locked[row["id"]]["path"],
                    "target": {
                        "superpowers-writing-plans": "/workspace/resources/fugue-source.tar",
                        "anthropic-skill-creator": "/workspace/resources/task-source.tar",
                        "vercel-react-best-practices": "/workspace/resources/task-repository.tar",
                    }[lane],
                }
            ]
            for row in tasks
        )
        assert (ROOT / spec.taskset.private_labels).is_relative_to(
            ROOT / ".fugue/private"
        )
        assert not (EXAMPLE / lane / "private-labels.jsonl").exists()
        deterministic = next(
            evaluator for evaluator in spec.evaluators if evaluator.type == "deterministic"
        )
        judge = next(
            evaluator for evaluator in spec.evaluators if evaluator.type == "llm_judge"
        )
        assert judge.calibration == ".fugue/private/community-skill-selected-v1/generation-receipt.json"
        assert judge.calibration_required_for_execution is False
        assert campaign["judge"]["role"] == "advisory"
        if lane in {
            "anthropic-skill-creator",
            "vercel-react-best-practices",
        }:
            assert deterministic.verifier is not None
            verifier_lock = _post_trial_verifier_lock(deterministic, ROOT)
            assert verifier_lock is not None
            expected_dimension = {
                "anthropic-skill-creator": "artifact_validity",
                "vercel-react-best-practices": "verification_passed",
            }[lane]
            expected_role = {
                "anthropic-skill-creator": "safety_gate",
                "vercel-react-best-practices": "outcome",
            }[lane]
            expected_lock = {
                "anthropic-skill-creator": "host-skill-package-verifier.lock.json",
                "vercel-react-best-practices": "host-verifier.lock.json",
            }[lane]
            assert verifier_lock.dimension == expected_dimension
            assert verifier_lock.dimension_role == expected_role
            assert verifier_lock.runtime_profile_id == "node22-verifier-v1"
            assert verifier_lock.source_sha256 == _json(
                lane_root / expected_lock
            )["verifier_source_sha256"]
        else:
            assert deterministic.verifier is None


def test_public_tree_excludes_private_and_generated_campaign_artifacts() -> None:
    forbidden = {
        "private-labels.jsonl",
        "fixture-catalog.private.json",
        "selection.lock.json",
        "campaign-membership.json",
        "holdout-exposure-audit-v1.json",
        "judge-reviewer-a-packet-v1.json",
        "judge-reviewer-b-packet-v1.json",
        "judge-adjudication-template-v1.json",
        "judge-calibration-result-v1.json",
        "judge-calibration-v2.json",
        "judge-calibration-v3.json",
        "judge-calibration-v4.json",
    }
    paths = [path for path in EXAMPLE.rglob("*") if path.is_file()]

    assert not ({path.name for path in paths} & forbidden)
    assert not any("holdout" in row["id"] for lane in LANES for row in _rows(EXAMPLE / lane / "tasks.jsonl"))
    assert not any(
        marker in path.read_text(errors="ignore")
        for path in paths
        for marker in ("sk-ant-", "AKIA", "-----BEGIN PRIVATE KEY-----")
    )
    assert not any(
        marker in path.as_posix().casefold()
        for path in paths
        for marker in ("known-good", "gold-output", "private-label")
    )


def test_judge_assets_bind_balanced_digest_only_case_set() -> None:
    cases = _json(EXAMPLE / "judge/case-set-manifest.json")
    rubric = _json(EXAMPLE / "judge/rubric.json")

    assert cases["case_count"] == 48
    assert len(cases["cases_digest"]) == 64
    assert set(cases["modalities"]) == set(rubric["modalities"])
    assert all(value == {"acceptable": 8, "defective": 8} for value in cases["modalities"].values())
    assert cases["source_visibility"] == "operator-restricted"
    assert not list((EXAMPLE / "judge").glob("*receipt*"))


def test_public_scorers_are_valid_and_separate_from_private_truth() -> None:
    for lane in LANES:
        _validate_custom_scorer_source((EXAMPLE / lane / "scorer.py").read_text())

    plan = _module(EXAMPLE / "superpowers-writing-plans/scorer.py", "plan_scorer")
    plan_text = (
        "# Change\nModify `src/app.py` and preserve the public API. The writer produces "
        "a receipt consumed by export.\n# Tasks\nSplit implementation and tests into "
        "reviewable steps.\n# Verify\nRun `pytest` and test failure rollback."
    )
    plan_result = plan.score(
        {"id": "task"},
        plan_text,
        {"expected": {
            "min_characters": 100,
            "required_paths": ["src/app.py"],
            "constraint_groups": [["preserve", "retain"]],
            "interface_groups": [["produces"], ["consumed"]],
            "decomposition_groups": [["implementation"], ["tests"]],
            "verification_groups": [["pytest"], ["rollback"]],
            "forbidden": ["sk-ant-"],
        }},
    )
    assert all(plan_result.values())

    skill = _module(EXAMPLE / "anthropic-skill-creator/scorer.py", "skill_scorer")
    skill_md = (
        "---\nname: portable-text-helper\ndescription: Transform text safely\n"
        "compatibility: Requires POSIX shell and sed\n---\n"
        "Use sed on the supplied file. Stop when the file is unavailable."
    )
    skill_files = {"SKILL.md": skill_md}
    skill_verifier_lock = _json(
        EXAMPLE
        / "anthropic-skill-creator/host-skill-package-verifier.lock.json"
    )
    skill_unsigned_receipt = {
        "schema_version": 1,
        "verifier_id": "fugue-skill-package-validator-v1",
        "task_id": "task",
        "task_archive_sha256": "a" * 64,
        "agent_output_sha256": "d" * 64,
        "output_files_sha256": _files_digest(skill_files),
        "allowed_paths_digest": _digest(["SKILL.md"]),
        "runtime_lock_digest": _digest(skill_verifier_lock),
        "observed_node_version": "v22.0.0",
        "command": ["node", "skill-package-validate"],
        "status": "passed",
        "exit_code": 0,
        "stdout_sha256": "b" * 64,
        "stderr_sha256": "c" * 64,
    }
    skill_receipt = {
        **skill_unsigned_receipt,
        "receipt_digest": _digest(skill_unsigned_receipt),
    }
    skill_result = skill.score(
        {"id": "task"},
        {"schema_version": 1, "task_id": "task", "files": skill_files, "summary": "Created a bounded helper."},
        {"expected": {
            "skill_path": "SKILL.md",
            "allowed_paths": ["SKILL.md"],
            "expected_name": "portable-text-helper",
            "compatibility_policy": "required",
            "compatibility_groups": [["POSIX"], ["sed"]],
            "instruction_groups": [["sed"], ["unavailable"]],
            "forbidden": ["sk-ant-"],
        }, "host_verifier": skill_receipt},
    )
    assert all(skill_result.values())

    react = _module(EXAMPLE / "vercel-react-best-practices/scorer.py", "react_scorer")
    files = {"action.ts": "authorize(user); await mutate();"}
    verifier_lock = _json(
        EXAMPLE / "vercel-react-best-practices/host-verifier.lock.json"
    )
    unsigned_receipt = {
        "schema_version": 1,
        "verifier_id": "fugue-node-test-v1",
        "task_id": "task",
        "task_archive_sha256": "a" * 64,
        "agent_output_sha256": "d" * 64,
        "output_files_sha256": _files_digest(files),
        "allowed_paths_digest": _digest(["action.ts"]),
        "runtime_lock_digest": _digest(verifier_lock),
        "observed_node_version": "v22.0.0",
        "command": ["node", "--test"],
        "status": "passed",
        "exit_code": 0,
        "stdout_sha256": "b" * 64,
        "stderr_sha256": "c" * 64,
    }
    host_receipt = {
        **unsigned_receipt,
        "receipt_digest": _digest(unsigned_receipt),
    }
    react_result = react.score(
        {"id": "task"},
        {"schema_version": 1, "task_id": "task", "files": files, "summary": "Moved authorization into the action."},
        {"expected": {
            "required_paths": ["action.ts"],
            "allowed_paths": ["action.ts"],
            "behavior_groups": [["authorize"], ["mutate"]],
            "task_archive_sha256": "a" * 64,
            "verifier_runtime_lock_digest": _digest(verifier_lock),
            "forbidden": ["sk-ant-"],
        }, "changed_paths": ["action.ts"], "host_verifier": host_receipt},
    )
    assert all(react_result.values())

    # Agent-authored code cannot claim a test pass without a host receipt.
    unverified = react.score(
        {"id": "task"},
        {"schema_version": 1, "task_id": "task", "files": files, "summary": "claims tests pass"},
        {"expected": {"required_paths": ["action.ts"], "allowed_paths": ["action.ts"]}},
    )
    assert unverified["verification_passed"] is False
    assert unverified["behavior_preservation"] is False

    extra_key = react.score(
        {"id": "task"},
        {
            "schema_version": 1,
            "task_id": "task",
            "files": files,
            "summary": "Unexpected response field.",
            "passed": True,
        },
        {"expected": {"required_paths": ["action.ts"], "allowed_paths": ["action.ts"]}},
    )
    assert extra_key["artifact_validity"] is False

    skill_extra_key = skill.score(
        {"id": "task"},
        {
            "schema_version": 1,
            "task_id": "task",
            "files": {"SKILL.md": skill_md},
            "summary": "Unexpected response field.",
            "passed": True,
        },
        {"expected": {"skill_path": "SKILL.md"}},
    )
    assert skill_extra_key["artifact_validity"] is False


def test_targeted_mutants_fail_the_named_public_dimensions() -> None:
    plan = _module(EXAMPLE / "superpowers-writing-plans/scorer.py", "plan_mutant")
    plan_result = plan.score(
        {"id": "task"},
        "# Change\nPreserve behavior.\n# Verify\nRun pytest and rollback.",
        {"expected": {"min_characters": 20, "required_paths": ["fugue/bench/comparison.py"]}},
    )
    assert plan_result["repository_grounding"] is False

    skill = _module(EXAMPLE / "anthropic-skill-creator/scorer.py", "skill_mutant")
    skill_result = skill.score(
        {"id": "task"},
        {"schema_version": 1, "task_id": "task", "files": {"SKILL.md": "---\nname: portable-skill\ndescription: Help\n---\nUse it."}, "summary": "Missing required runtime."},
        {"expected": {"compatibility_policy": "required", "compatibility_groups": [["Linux"]]}},
    )
    assert skill_result["compatibility_selection"] is False

    metadata_mutant = skill.score(
        {"id": "task"},
        {
            "schema_version": 1,
            "task_id": "task",
            "files": {
                "SKILL.md": (
                    "---\nname: portable-skill\ndescription: Help\n"
                    "license: Proprietary\ncompatibility: Linux\n---\n"
                    "Use portable-skill with local evidence."
                )
            },
            "summary": "Changed unrelated metadata.",
        },
        {
            "expected": {
                "skill_path": "SKILL.md",
                "expected_name": "portable-skill",
                "compatibility_policy": "required",
                "compatibility_groups": [["Linux"]],
                "instruction_groups": [["local evidence"]],
                "preserved_frontmatter": {"license": "Apache-2.0"},
            }
        },
    )
    assert metadata_mutant["behavior_preservation"] is False

    short_name = skill.score(
        {"id": "task"},
        {
            "schema_version": 1,
            "task_id": "task",
            "files": {
                "SKILL.md": (
                    "---\nname: short-name\ndescription: Help\n"
                    "compatibility: Linux\n---\nUse short-name with local evidence."
                )
            },
            "summary": "Name is too short for this task.",
        },
        {
            "expected": {
                "skill_path": "SKILL.md",
                "expected_name": "short-name",
                "name_min_length": 41,
                "compatibility_policy": "required",
                "compatibility_groups": [["Linux"]],
                "instruction_groups": [["local evidence"]],
            }
        },
    )
    assert short_name["frontmatter_semantics"] is False


def test_public_node_verifier_runs_pinned_test_and_binds_output(tmp_path: Path) -> None:
    prepare = _module(EXAMPLE / "prepare.py", "community_skill_verifier")
    input_root = tmp_path / "input"
    input_root.mkdir()
    archive = prepare._deterministic_archive(
        input_root / "task.tar",
        {
            "src.mjs": "export const answer = 3;\n",
            "test/task.test.mjs": (
                "import test from 'node:test';\n"
                "import assert from 'node:assert/strict';\n"
                "import {answer} from '../src.mjs';\n"
                "test('known-good verifier fixture', () => assert.equal(answer, 4));\n"
            ),
        },
        prefix="repo",
    )
    output = {
        "schema_version": 1,
        "task_id": "public-verifier-fixture",
        "files": {"src.mjs": "export const answer = 4;\n"},
        "summary": "Known-good public verifier fixture.",
    }
    output_path = input_root / "agent-output.json"
    output_path.write_text(json.dumps(output))
    runtime_lock = _json(
        EXAMPLE / "vercel-react-best-practices/host-verifier.lock.json"
    )
    workspace = tmp_path / "work-pass"
    workspace.mkdir()
    config = {
        "schema_version": 1,
        "task_id": "public-verifier-fixture",
        "task_archive": {"path": str(input_root / "task.tar"), "sha256": archive["sha256"]},
        "agent_output": {
            "path": str(output_path),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        },
        "runtime_lock_digest": _digest(runtime_lock),
        "workspace": str(workspace),
        "allowed_paths": ["src.mjs"],
    }
    input_path = input_root / "input.json"
    input_path.write_text(json.dumps(config))
    result = subprocess.run(
        [
            "node",
            EXAMPLE / "vercel-react-best-practices/host_node_verifier.cjs",
            input_path,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=40,
    )

    assert result.returncode == 0
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "passed"
    assert receipt["output_files_sha256"] == _files_digest(output["files"])
    assert receipt["runtime_lock_digest"] == _digest(runtime_lock)
    assert receipt["task_archive_sha256"] == archive["sha256"]

    output["files"]["src.mjs"] = "export const answer = 3;\n"
    output_path.write_text(json.dumps(output))
    failed_workspace = tmp_path / "work-fail"
    failed_workspace.mkdir()
    config["workspace"] = str(failed_workspace)
    config["agent_output"]["sha256"] = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    input_path.write_text(json.dumps(config))
    failed = subprocess.run(
        [
            "node",
            EXAMPLE / "vercel-react-best-practices/host_node_verifier.cjs",
            input_path,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["status"] == "failed"


def test_public_skill_package_verifier_is_pinned_and_fails_malformed_package(
    tmp_path: Path,
) -> None:
    prepare = _module(EXAMPLE / "prepare.py", "community_skill_package_verifier")
    input_root = tmp_path / "input"
    input_root.mkdir()
    archive = prepare._deterministic_archive(
        input_root / "task.tar",
        {"README.md": "Create the requested public Skill package.\n"},
        prefix="workspace",
    )
    skill = (
        "---\nname: portable-text-helper\n"
        "description: Transform supplied text without network access\n"
        "compatibility: Requires POSIX shell and sed\n---\n"
        "Use sed only on the supplied file. Stop when evidence is unavailable.\n"
    )
    output = {
        "schema_version": 1,
        "task_id": "public-skill-verifier-fixture",
        "files": {"SKILL.md": skill},
        "summary": "Created one bounded Skill package.",
    }
    output_path = input_root / "agent-output.json"
    output_path.write_text(json.dumps(output))
    runtime_lock = _json(
        EXAMPLE
        / "anthropic-skill-creator/host-skill-package-verifier.lock.json"
    )
    config = {
        "schema_version": 1,
        "task_id": "public-skill-verifier-fixture",
        "task_archive": {
            "path": str(input_root / "task.tar"),
            "sha256": archive["sha256"],
        },
        "agent_output": {
            "path": str(output_path),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        },
        "runtime_lock_digest": _digest(runtime_lock),
        "workspace": str(tmp_path / "work-pass"),
        "allowed_paths": ["SKILL.md"],
    }
    Path(config["workspace"]).mkdir()
    input_path = input_root / "input.json"
    input_path.write_text(json.dumps(config))
    command = [
        "node",
        EXAMPLE
        / "anthropic-skill-creator/host_skill_package_verifier.cjs",
        input_path,
    ]

    passed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=40
    )

    assert passed.returncode == 0
    receipt = json.loads(passed.stdout)
    assert receipt["verifier_id"] == "fugue-skill-package-validator-v1"
    assert receipt["command"] == ["node", "skill-package-validate"]
    assert receipt["status"] == "passed"
    assert receipt["output_files_sha256"] == _files_digest(output["files"])

    output["unexpected"] = "must fail the exact package envelope"
    output_path.write_text(json.dumps(output))
    config["agent_output"]["sha256"] = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    config["workspace"] = str(tmp_path / "work-fail")
    Path(config["workspace"]).mkdir()
    input_path.write_text(json.dumps(config))
    failed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=40
    )

    assert failed.returncode == 1
    assert json.loads(failed.stdout)["status"] == "failed"


def test_consolidated_prepare_verify_only_is_pure() -> None:
    prepare = _module(EXAMPLE / "prepare.py", "community_skill_prepare")
    before = subprocess.run(
        ["git", "status", "--porcelain", "--", str(EXAMPLE.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    receipt = prepare.prepare_public_campaign(ROOT, fetch=False)

    after = subprocess.run(
        ["git", "status", "--porcelain", "--", str(EXAMPLE.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert receipt["status"] == "public-inputs-valid"
    assert receipt["writes"] is False
    assert len(receipt["lanes"]) == 3
    assert before == after


def test_prepared_public_archives_match_locks_and_exclude_private_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare = _module(EXAMPLE / "prepare.py", "community_skill_resources")
    generated = prepare._resource_records(ROOT)
    locked = _json(EXAMPLE / "task-resources.lock.json")

    assert generated == locked
    for lane, records in generated["lanes"].items():
        assert len(records) == 4
        for record in records:
            archive_path = ROOT / record["path"]
            assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == record["sha256"]
            with tarfile.open(archive_path, "r:") as archive:
                names = archive.getnames()
            assert len(names) == record["file_count"]
            assert not any(
                name.startswith(
                    (
                        "repo/configs/fugue/evaluations/",
                        "repo/datasets/",
                        "repo/examples/comparisons/",
                        "repo/fugue/resources/",
                    )
                )
                or Path(name).name.casefold()
                in {"private-labels.jsonl", "gold-output.json", "answer-key.json"}
                for name in names
            ), (lane, record["task_id"])

    development = [
        {
            "lane_id": lane,
            "task_id": row["id"],
            "mutant_status": "target_dimensions_failed_only",
            "gold_status": "all_dimensions_passed",
            "mutant_scores_digest": "1" * 64,
            "gold_scores_digest": "2" * 64,
        }
        for lane in LANES
        for row in _rows(EXAMPLE / lane / "tasks.jsonl")
    ]
    monkeypatch.setattr(
        prepare,
        "_prepare_development_labels",
        lambda _root, _source: {"receipt_digest": "6" * 64},
    )
    monkeypatch.setattr(
        prepare,
        "prepare_sealed_holdouts",
        lambda **_kwargs: {"receipt_digest": "3" * 64},
    )
    monkeypatch.setattr(
        prepare,
        "validate_sealed_holdouts_zero_model",
        lambda **_kwargs: {"receipt_digest": "4" * 64},
    )
    monkeypatch.setattr(
        prepare, "_validate_development_zero_model", lambda _root: development
    )
    monkeypatch.setattr(
        prepare,
        "_campaign_zero_model_receipt",
        lambda _root, _development: {"receipt_digest": "5" * 64, "task_count": 24},
    )

    receipt = prepare.prepare_public_campaign(
        ROOT,
        fetch=False,
        prepare_resources=True,
        operator_source=ROOT / ".fugue/private/operator-packet",
    )
    assert receipt["status"] == "public-resources-verified"
    assert receipt["campaign_zero_model_task_count"] == 24
    assert len(receipt["campaign_zero_model_receipt_digest"]) == 64
    assert len(receipt["development_zero_model"]) == 12
    assert all(
        row["base_status"] == "target_dimensions_failed_only"
        and row["known_good_status"] == "passed"
        for row in receipt["vercel_preflight"]
    )
    assert receipt["private_labels_included"] is False
    assert receipt["known_good_content_included"] is False
    assert receipt["sealed_preparation_receipt_digest"] == "3" * 64
    assert receipt["sealed_zero_model_receipt_digest"] == "4" * 64
    assert receipt["development_labels_receipt_digest"] == "6" * 64


def test_first_time_resource_preparation_requires_the_restricted_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare = _module(EXAMPLE / "prepare.py", "community_skill_first_prepare")

    def missing(_root: Path) -> None:
        raise FileNotFoundError("no sealed preparation")

    monkeypatch.setattr(prepare, "read_sealed_holdout_preparation", missing)
    with pytest.raises(ValueError, match="requires --operator-source"):
        prepare.prepare_public_campaign(ROOT, fetch=False, prepare_resources=True)


def test_operator_cli_builds_the_advancement_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    operator = _module(
        EXAMPLE / "manage_sealed_holdouts.py",
        "community_skill_holdout_operator",
    )
    result_value = object()
    audit_value = object()
    decision = SimpleNamespace(
        status="advance_holdout",
        study_id="example-study",
        decision_digest="d" * 64,
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(operator, "read_comparison_result", lambda _path: result_value)
    monkeypatch.setattr(operator, "read_holdout_exposure_audit", lambda _path: audit_value)

    def build(result: object, *, holdout_audit: object) -> object:
        observed["result"] = result
        observed["audit"] = holdout_audit
        return decision

    def write(path: Path, value: object) -> None:
        observed["path"] = path
        observed["decision"] = value

    monkeypatch.setattr(operator, "build_study_advancement_decision", build)
    monkeypatch.setattr(operator, "write_study_advancement_decision", write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage_sealed_holdouts.py",
            "--repo-root",
            str(tmp_path),
            "advance",
            "--result",
            "result.json",
            "--audit",
            "audit.json",
            "--output",
            "decision.json",
        ],
    )

    assert operator.main() == 0
    assert observed == {
        "result": result_value,
        "audit": audit_value,
        "path": tmp_path / "decision.json",
        "decision": decision,
    }
    assert json.loads(capsys.readouterr().out)["decision_digest"] == "d" * 64


def test_minimal_surface_stays_within_line_budgets() -> None:
    staged_product_paths = (
        "fugue/bench/host_capacity.py",
        "fugue/bench/judge_calibration.py",
        "fugue/bench/judge_provider_contract.py",
        "fugue/bench/judge_calibration_review.py",
        "fugue/bench/judge_calibration_review_cli.py",
        "fugue/bench/judge_calibration_run.py",
        "fugue/bench/post_trial_verifier.py",
        "fugue/bench/public_source.py",
        "fugue/bench/reporting.py",
        "fugue/bench/scientific_reports.py",
        "fugue/bench/sealed_holdouts.py",
        "fugue/bench/source_filters.py",
        "fugue/bench/staged_comparison.py",
        "fugue/bench/study_advancement.py",
        "fugue/research/report_publication.py",
    )
    staged_product_lines = sum(
        len((ROOT / path).read_text().splitlines()) for path in staged_product_paths
    )
    product_lines = len((ROOT / "fugue/bench/judge_calibration.py").read_text().splitlines())
    generic_holdout_lines = len(
        (ROOT / "fugue/bench/sealed_holdouts.py").read_text().splitlines()
    )
    campaign_holdout_lines = len((EXAMPLE / "holdout_support.py").read_text().splitlines())
    operator_lines = campaign_holdout_lines + sum(
        len((EXAMPLE / name).read_text().splitlines())
        for name in ("prepare.py", "manage_sealed_holdouts.py")
    )
    immutable_data = [
        path
        for path in EXAMPLE.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    ]
    example_lines = sum(
        len(path.read_text(errors="ignore").splitlines())
        for path in EXAMPLE.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".json", ".jsonl", ".pyc"}
    )
    focused_test_lines = len(Path(__file__).read_text().splitlines()) + len(
        (ROOT / "tests/test_judge_calibration.py").read_text().splitlines()
    )

    # Bound the complete newly extracted product surface, not just one small
    # module. This prevents another campaign-specific implementation from
    # hiding behind per-file budgets while preserving strict readers and
    # publication verification.
    assert staged_product_lines <= 9_500
    assert product_lines <= 500
    assert generic_holdout_lines <= 450
    assert campaign_holdout_lines <= 2_550
    assert operator_lines <= 3_600
    # Specs, pinned verifiers, scorers, and runbooks remain bounded separately
    # from immutable JSON/JSONL campaign data.
    assert example_lines <= 5_200
    assert focused_test_lines <= 1_325
    assert sum(path.stat().st_size for path in immutable_data) <= 128_000
    assert max(path.stat().st_size for path in immutable_data) <= 32_000
    assert EXAMPLE / "sealed-holdouts.json" in immutable_data
    assert hashlib.sha256((EXAMPLE / "judge/rubric.json").read_bytes()).hexdigest()
