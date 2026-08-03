from __future__ import annotations

import importlib.util
import json
import runpy
from pathlib import Path

from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/anthropic-skill-creator-upgrade")
LOCK = EXAMPLE / "skill-revisions.lock.json"
PRIVATE = EXAMPLE / "private-labels.jsonl"
SCORER = EXAMPLE / "skill_creator_scorer.py"
PREPARE = EXAMPLE / "prepare_sources.py"
DIMENSIONS = {
    "schema_validity",
    "compatibility_preservation",
    "name_rules",
    "packaging",
    "instruction_quality",
    "dependency_secret_safety",
}


def _score():
    return runpy.run_path(SCORER.as_posix())["score"]


def _labels() -> dict[str, dict]:
    return {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in PRIVATE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _prepare_module():
    spec = importlib.util.spec_from_file_location("anthropic_prepare", PREPARE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canary_locks_exact_skill_creator_upgrade_and_four_cells() -> None:
    spec = load_comparison(EXAMPLE / "comparison-v2.yaml", repo_root=Path.cwd())
    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    assert spec.schema_version == 2
    assert spec.id == "anthropic-skill-creator-upgrade-canary-v2"
    assert spec.baseline.skills == ("anthropic-skill-creator-before-compatibility",)
    assert spec.candidate.skills == ("anthropic-skill-creator-compatibility",)
    assert lock["repository"] == "https://github.com/anthropics/skills"
    assert lock["path"] == "skills/skill-creator"
    assert lock["declared_skill_name"] == "skill-creator"
    assert lock["baseline"] == {
        "id": "anthropic-skill-creator-before-compatibility",
        "commit": "a5bcdd7e58cdff48566bf876f0a72a2008dcefbc",
        "bundle_digest": (
            "sha256:cf1484977a45bf2f7ccde2dbf2c71118c159dc642cc2c1521b7aacfbef0e67e3"
        ),
        "file_count": 7,
    }
    assert lock["candidate"] == {
        "id": "anthropic-skill-creator-compatibility",
        "commit": "1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563",
        "bundle_digest": (
            "sha256:aef41868a0a89c6366e13997eabfef2e8554617428fc6c9cb89b68660af1370f"
        ),
        "file_count": 7,
    }
    assert spec.execution.evidence_project == (
        "wandb/fugue-anthropic-skill-creator-upgrade-v1"
    )
    assert spec.execution.study_console_base_url == "http://127.0.0.1:18085"
    assert spec.execution.model == "anthropic/claude-sonnet-5"
    assert spec.execution.harnesses == ("claude-code",)
    assert spec.execution.attempts == 1
    assert spec.execution.max_cost_usd == 34
    assert spec.execution.reserve_per_attempt_usd == 8.4
    assert (
        sum(
            evaluator.reserve_cost_usd
            for evaluator in spec.evaluators
            if evaluator.type == "llm_judge"
        )
        == 0.1
    )

    tasks = [
        json.loads(line)
        for line in (EXAMPLE / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(tasks) == 2
    assert {item["resources"][0]["path"].rsplit("/", 1)[-1] for item in tasks} == {
        "create-skill-workspace.tar",
        "update-skill-workspace.tar",
    }
    for task in tasks:
        question = task["input"]["question"]
        assert "`schema_version` is integer `1`" in question
        assert "maps paths relative to the `skills/` directory" in question
        assert (
            "`validation` has exactly string `command` and boolean `passed`" in question
        )


def test_shared_judge_is_advisory_and_blinded_to_treatment() -> None:
    spec = load_comparison(EXAMPLE / "comparison-v2.yaml", repo_root=Path.cwd())
    judge = next(item for item in spec.evaluators if item.type == "llm_judge")
    rubric = json.loads(
        Path(
            "examples/comparisons/community-skill-upgrades/judge-rubric.json"
        ).read_text(encoding="utf-8")
    )

    assert judge.id == "community-usefulness"
    assert judge.required is False
    assert judge.profile == "anthropic/claude-sonnet-5"
    assert judge.calibration == (
        "examples/comparisons/community-skill-upgrades/judge-calibration-v2.json"
    )
    assert judge.rubric == rubric["rubric"]
    assert judge.dimensions == (
        "useful_actionability",
        "repository_grounding",
        "reviewability",
        "risk_calibration",
    )
    assert judge.evidence == ("inspected_paths", "changed_paths")


def test_private_fixtures_keep_base_fail_and_gold_pass() -> None:
    score = _score()
    for task_id, label in _labels().items():
        base = score(
            {"id": task_id},
            label["base_output"],
            {"expected": label["expected"]},
        )
        gold = score(
            {"id": task_id},
            label["gold_output"],
            {"expected": label["expected"]},
        )

        assert set(gold) == DIMENSIONS
        assert all(gold.values()), (task_id, gold)
        assert not all(base.values())


def test_compatibility_and_preserved_behavior_fail_independently() -> None:
    score = _score()
    labels = _labels()

    create = labels["create-compatible-long-name-skill"]
    without_compatibility = create["gold_output"].replace(
        "compatibility: macOS 13+ or Ubuntu 22.04+, Git 2.40+, and Python 3.12+; works offline\\n",
        "",
    )
    create_scores = score(
        {"id": create["id"]},
        without_compatibility,
        {"expected": create["expected"]},
    )
    assert create_scores["schema_validity"] is True
    assert create_scores["compatibility_preservation"] is False
    assert create_scores["packaging"] is True

    update = labels["add-compatibility-without-behavior-drift"]
    drifted = update["gold_output"].replace(
        "Inventory the supplied local evidence before drawing a conclusion.",
        "Inventory available evidence before drawing a conclusion.",
    )
    update_scores = score(
        {"id": update["id"]},
        drifted,
        {"expected": update["expected"]},
    )
    assert update_scores["schema_validity"] is True
    assert update_scores["compatibility_preservation"] is False
    assert update_scores["name_rules"] is True


def test_long_name_limit_and_unsafe_dependency_fail_closed() -> None:
    score = _score()
    label = _labels()["create-compatible-long-name-skill"]
    too_long = "a" * 65
    renamed = label["gold_output"].replace(
        "cross-platform-release-evidence-verification",
        too_long,
    )
    renamed_expected = dict(label["expected"])
    renamed_expected["skill_name"] = too_long
    renamed_expected["skill_path"] = f"{too_long}/SKILL.md"
    renamed_expected["allowed_paths"] = [
        f"{too_long}/SKILL.md",
        f"{too_long}/references/platform-matrix.md",
    ]
    long_scores = score(
        {"id": label["id"]},
        renamed,
        {"expected": renamed_expected},
    )
    unsafe = label["gold_output"].replace(
        "Offline operation only; all release evidence must already be local.",
        "Run pip install release-checker before reviewing local evidence.",
    )
    unsafe_scores = score(
        {"id": label["id"]},
        unsafe,
        {"expected": label["expected"]},
    )
    documented_prohibition = label["gold_output"].replace(
        "Offline operation only; all release evidence must already be local.",
        "Offline operation only; pip install is forbidden, avoid git clone, and keep all release evidence local. Password: never serialize or persist it.",
    )
    documented_scores = score(
        {"id": label["id"]},
        documented_prohibition,
        {"expected": label["expected"]},
    )

    assert long_scores["name_rules"] is False
    assert unsafe_scores["dependency_secret_safety"] is False
    assert documented_scores["dependency_secret_safety"] is True

    leaked_secret = label["gold_output"].replace(
        "Offline operation only; all release evidence must already be local.",
        "Offline operation only. API key: sk-live-value",
    )
    leaked_scores = score(
        {"id": label["id"]},
        leaked_secret,
        {"expected": label["expected"]},
    )
    assert leaked_scores["dependency_secret_safety"] is False


def test_public_semantics_do_not_require_hidden_headings_or_memo_name() -> None:
    score = _score()
    label = _labels()["create-compatible-long-name-skill"]
    output = (
        label["gold_output"]
        .replace("## Workflow", "## Bounded process")
        .replace("## Validation", "## Verification checks")
        .replace("## Failure handling", "## Incomplete evidence")
        .replace(
            "Created cross-platform-release-evidence-verification as a two-file, offline Skill with explicit platform requirements and bounded evidence handling.",
            "Completed the requested two-file package and its offline validation without adding executable code.",
        )
    )

    scores = score(
        {"id": label["id"]},
        output,
        {"expected": label["expected"]},
    )

    assert scores["name_rules"] is True
    assert scores["instruction_quality"] is True


def test_wrong_task_identity_remains_a_real_schema_failure() -> None:
    score = _score()
    label = _labels()["create-compatible-long-name-skill"]
    output = label["gold_output"].replace(
        '"task_id":"create-compatible-long-name-skill"',
        '"task_id":"cross-platform-release-evidence-verification-skill"',
    )

    scores = score(
        {"id": label["id"]},
        output,
        {"expected": label["expected"]},
    )

    assert scores["schema_validity"] is False


def test_task_fixture_archives_are_deterministic(tmp_path: Path) -> None:
    module = _prepare_module()
    source = EXAMPLE / "fixtures/create-skill-workspace"
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"

    first_record = module._fixture_archive(source, first)
    second_record = module._fixture_archive(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_record["archive_sha256"] == second_record["archive_sha256"]
    assert first_record["paths_digest"] == second_record["paths_digest"]
    assert first_record["file_count"] == 2
