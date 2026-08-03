from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/vercel-react-best-practices-upgrade")
LOCK = EXAMPLE / "skill-revisions.lock.json"
FIXTURE_LOCK = EXAMPLE / "fixture-sources.lock.json"
PRIVATE = EXAMPLE / "private-labels.jsonl"
SCORER = EXAMPLE / "vercel_change_scorer.py"
DIMENSIONS = {
    "artifact_validity",
    "requested_change",
    "repository_grounding",
    "behavior_preservation",
    "verification",
    "scope_safety",
    "skill_mechanism_used",
}


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _labels() -> dict[str, dict]:
    return {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in PRIVATE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _score():
    return runpy.run_path(SCORER.as_posix())["score"]


def test_vercel_canary_locks_exact_revisions_matrix_destination_and_budget() -> None:
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=Path.cwd())
    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    assert spec.schema_version == 2
    assert spec.id == "vercel-react-best-practices-upgrade-canary-v1"
    assert spec.baseline.skills == ("vercel-react-best-practices-before",)
    assert spec.candidate.skills == ("vercel-react-best-practices-after",)
    assert lock["repository"] == "https://github.com/vercel-labs/agent-skills"
    assert lock["path"] == "skills/react-best-practices"
    assert lock["baseline"]["commit"] == (
        "ac6a79af08f6d32c34ee03c829824990f3de0a6d"
    )
    assert lock["candidate"]["commit"] == (
        "20987af2f1bc17857b55e7758af8bed91c364ff5"
    )
    assert spec.execution.evidence_project == (
        "wandb/fugue-vercel-react-best-practices-upgrade-v1"
    )
    assert spec.execution.study_console_base_url == "http://127.0.0.1:18086"
    assert spec.execution.model == "anthropic/claude-sonnet-5"
    assert spec.execution.harnesses == ("claude-code",)
    assert spec.execution.attempts == 1
    assert spec.execution.concurrency == 1
    assert spec.execution.evidence_checkpoint_cells == 1
    assert spec.execution.max_cost_usd == 34
    assert spec.execution.reserve_per_attempt_usd == 8.4

    tasks = [
        json.loads(line)
        for line in (EXAMPLE / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(tasks) == 2
    assert len(tasks) * 2 * spec.execution.attempts == 4
    assert all("expected" not in task for task in tasks)


def test_vercel_skill_source_lock_is_exact_and_materialized() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    supplied = lock.pop("manifest_digest")
    assert supplied == _stable_digest(lock)
    for arm in ("baseline", "candidate"):
        source = {
            "repository": lock["repository"],
            "path": lock["path"],
            "commit": lock[arm]["commit"],
        }
        assert lock[arm]["source_identity_digest"] == (
            "sha256:" + _stable_digest(source)
        )
        assert str(lock[arm]["bundle_digest"]).startswith("sha256:")
        assert len(str(lock[arm]["bundle_digest"])) == 71
        assert lock[arm]["bundle_digest_status"] == "reviewed_and_materialized"


def test_vercel_fixture_source_lock_covers_exact_file_set_and_bytes() -> None:
    lock = json.loads(FIXTURE_LOCK.read_text(encoding="utf-8"))
    supplied = lock.pop("manifest_digest")
    assert supplied == _stable_digest(lock)
    expected_paths = {item["path"] for item in lock["fixtures"]}
    fixtures = EXAMPLE / "fixtures"
    actual_paths = {
        path.relative_to(fixtures).as_posix()
        for path in fixtures.rglob("*")
        if path.is_file()
    }
    assert actual_paths == expected_paths
    for item in lock["fixtures"]:
        path = fixtures / item["path"]
        assert len(path.read_bytes()) == item["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_vercel_fixture_archives_are_deterministic_and_safe(tmp_path: Path) -> None:
    module = runpy.run_path((EXAMPLE / "prepare_fixtures.py").as_posix())
    module["_load_and_verify_source_lock"]()
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    first_record = module["_build_archive"]("server-action-auth", first)
    second_record = module["_build_archive"]("server-action-auth", second)

    assert first.read_bytes() == second.read_bytes()
    assert first_record["sha256"] == second_record["sha256"]
    assert first_record["file_count"] == 5


def test_vercel_private_fixtures_keep_base_fail_gold_pass() -> None:
    score = _score()
    for task_id, label in _labels().items():
        base = score(
            {"id": task_id},
            label["base_output"],
            {"expected": label["expected"], **label["base_evidence"]},
        )
        gold = score(
            {"id": task_id},
            label["gold_output"],
            {"expected": label["expected"], **label["gold_evidence"]},
        )

        assert set(base) == DIMENSIONS
        assert set(gold) == DIMENSIONS
        assert all(gold.values()), (task_id, gold)
        assert not all(base.values()), (task_id, base)
        assert base["skill_mechanism_used"] is False


def test_vercel_scorer_separates_correctness_scope_and_skill_mechanism() -> None:
    score = _score()
    label = _labels()["server-action-authorization"]
    evidence = {"expected": label["expected"], **label["gold_evidence"]}

    wrong = json.loads(json.dumps(label["gold_output"]))
    wrong["facts"]["authorization_inside_action"] = False
    wrong_scores = score({"id": label["id"]}, wrong, evidence)
    assert wrong_scores["requested_change"] is False
    assert wrong_scores["skill_mechanism_used"] is True

    unsafe_evidence = {
        **evidence,
        "changed_paths": [
            *evidence["changed_paths"],
            "/tmp/task-repository/repo/package.json",
        ],
    }
    unsafe_scores = score(
        {"id": label["id"]}, label["gold_output"], unsafe_evidence
    )
    assert unsafe_scores["requested_change"] is True
    assert unsafe_scores["scope_safety"] is False


def test_vercel_judge_is_shared_blinded_and_cannot_replace_deterministic_gate() -> None:
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=Path.cwd())
    judge = next(item for item in spec.evaluators if item.type == "llm_judge")
    deterministic = next(
        item for item in spec.evaluators if item.type == "deterministic"
    )

    assert judge.id == "community-usefulness"
    assert judge.required is False
    assert judge.profile == "anthropic/claude-sonnet-5"
    assert judge.calibration == (
        "examples/comparisons/community-skill-upgrades/judge-calibration.json"
    )
    assert judge.dimensions == (
        "useful_actionability",
        "repository_grounding",
        "reviewability",
        "risk_calibration",
    )
    assert judge.evidence == ("inspected_paths", "changed_paths")
    assert judge.reserve_cost_usd == 0.1
    assert deterministic.required is True
    assert set(deterministic.dimension_roles.values()) == {
        "outcome",
        "safety_gate",
        "mechanism",
    }


def test_vercel_fixtures_encode_real_failing_boundaries_without_private_truth() -> None:
    action = (
        EXAMPLE / "fixtures/server-action-auth/app/actions.mjs"
    ).read_text(encoding="utf-8")
    action_tests = (
        EXAMPLE / "fixtures/server-action-auth/tests/actions.test.mjs"
    ).read_text(encoding="utf-8")
    page = (EXAMPLE / "fixtures/rsc-serialization/app/projects/page.jsx").read_text(
        encoding="utf-8"
    )
    serialization_tests = (
        EXAMPLE / "fixtures/rsc-serialization/tests/serialization.test.mjs"
    ).read_text(encoding="utf-8")

    assert "db.membership" not in action
    assert "signed-in non-member" in action_tests
    assert "projectNames={projectNames}" in page
    assert "server passes one canonical project collection" in serialization_tests
    assert "base_output" not in (EXAMPLE / "tasks.jsonl").read_text(
        encoding="utf-8"
    )
