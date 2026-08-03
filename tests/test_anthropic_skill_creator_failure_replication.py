from __future__ import annotations

import json
import runpy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from fugue.bench.comparison import load_comparison

EXAMPLE = Path("examples/comparisons/anthropic-skill-creator-upgrade")
SPEC = EXAMPLE / "failure-replication.yaml"
SEMANTIC_SPEC = EXAMPLE / "semantic-replication-v2.yaml"
TASKS = EXAMPLE / "failure-replication-tasks.jsonl"
PRIVATE = EXAMPLE / "failure-replication-private-labels.jsonl"
SCORER = EXAMPLE / "failure_replication_scorer.py"
HISTORICAL_SCORER = EXAMPLE / "failure_replication_scorer_v1_archived.py"
REVISION_LOCK = EXAMPLE / "skill-revisions.lock.json"
STUDY_CONSOLE = EXAMPLE / "study-console-failure-replication.yaml"
SEMANTIC_STUDY_CONSOLE = EXAMPLE / "study-console-semantic-replication-v2.yaml"
SEMANTIC_DIMENSIONS = {
    "source_traceability",
    "terminal_success_or_stop_semantics",
    "missing_evidence_status",
}


def _score():
    return runpy.run_path(SCORER.as_posix())["score"]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _task() -> dict:
    return json.loads(TASKS.read_text(encoding="utf-8").strip())


def _private() -> dict:
    return json.loads(PRIVATE.read_text(encoding="utf-8").strip())


def _result(output: str) -> dict:
    return json.loads(output)


def _render(result: dict) -> str:
    return json.dumps(result, separators=(",", ":"), sort_keys=True)


def _replace_skill_body(output: str, old: str, new: str) -> str:
    result = _result(output)
    path = _private()["expected"]["skill_path"]
    result["files"][path] = result["files"][path].replace(old, new)
    return _render(result)


def _output_with_body(body: str) -> str:
    result = _result(_private()["gold_output"])
    path = _private()["expected"]["skill_path"]
    frontmatter, _separator, _old_body = result["files"][path].partition("\n---\n")
    result["files"][path] = f"{frontmatter}\n---\n\n{body.strip()}\n"
    return _render(result)


def test_failure_replication_is_one_task_two_revisions_two_attempts() -> None:
    spec = load_comparison(SPEC, repo_root=Path.cwd())
    tasks = [json.loads(line) for line in TASKS.read_text().splitlines() if line]
    revisions = json.loads(REVISION_LOCK.read_text(encoding="utf-8"))

    # V3 currently models hosted-source drift only. This immutable local task
    # archive therefore stays V2 rather than claiming a false hosted topology.
    assert spec.schema_version == 2
    assert spec.id == "anthropic-skill-creator-instruction-failure-replication-v1"
    assert spec.baseline.skills == ("anthropic-skill-creator-before-compatibility",)
    assert spec.candidate.skills == ("anthropic-skill-creator-compatibility",)
    assert revisions["baseline"]["commit"] == (
        "a5bcdd7e58cdff48566bf876f0a72a2008dcefbc"
    )
    assert revisions["candidate"]["commit"] == (
        "1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563"
    )
    assert spec.execution.attempts == 2
    assert spec.execution.harnesses == ("claude-code",)
    assert spec.execution.model == "anthropic/claude-sonnet-5"
    assert spec.execution.evidence_project == (
        "wandb/fugue-anthropic-skill-creator-failure-replication-v1"
    )
    assert spec.execution.study_console_base_url == "http://127.0.0.1:18087"
    judge = next(item for item in spec.evaluators if item.type == "llm_judge")
    assert judge.required is False
    assert judge.profile == "anthropic/claude-sonnet-5"
    assert tasks[0]["resources"] == [
        {
            "path": ".fugue/comparison-resources/anthropic-skill-creator-upgrade/create-skill-workspace.tar",
            "target": "/workspace/resources/task-source.tar",
        }
    ]
    assert len(tasks) * 2 * len(spec.execution.harnesses) * spec.execution.attempts == 4


def test_failure_replication_study_console_profile_is_dedicated_and_read_only() -> None:
    profile = yaml.safe_load(STUDY_CONSOLE.read_text(encoding="utf-8"))

    assert profile["research"]["id"] == (
        "fugue-anthropic-skill-creator-failure-replication-v1"
    )
    assert profile["wandb"] == {
        "entity": "wandb",
        "project": "fugue-anthropic-skill-creator-failure-replication-v1",
    }
    assert profile["database"] == (
        ".study-console/anthropic-skill-creator-failure-replication.sqlite3"
    )
    assert profile["presentation"] == {
        "default_study_id": (
            "anthropic-skill-creator-instruction-failure-replication-v1"
        ),
        "read_only": True,
    }


def test_semantic_replication_v2_is_independent_but_behaviorally_identical() -> None:
    historical = load_comparison(SPEC, repo_root=Path.cwd())
    semantic = load_comparison(SEMANTIC_SPEC, repo_root=Path.cwd())
    tasks = [json.loads(line) for line in TASKS.read_text().splitlines() if line]

    assert historical.id == (
        "anthropic-skill-creator-instruction-failure-replication-v1"
    )
    assert historical.execution.evidence_project == (
        "wandb/fugue-anthropic-skill-creator-failure-replication-v1"
    )
    assert semantic.schema_version == 2
    assert semantic.id == (
        "anthropic-skill-creator-instruction-semantic-replication-v2"
    )
    assert semantic.execution.evidence_project == (
        "wandb/fugue-anthropic-skill-creator-semantic-replication-v2"
    )
    assert semantic.execution.evidence_destination is not None
    assert semantic.execution.evidence_destination.entity == "wandb"
    assert semantic.execution.evidence_destination.project == (
        "fugue-anthropic-skill-creator-semantic-replication-v2"
    )
    assert semantic.execution.research_id == semantic.id
    assert semantic.execution.study_console_base_url == "http://127.0.0.1:18088"

    # A fresh identity changes only the scorer revision, not the treatment,
    # task, judge, model, harness, runtime, attempt count, or budget.
    assert semantic.question == historical.question
    assert semantic.taskset == historical.taskset
    assert semantic.baseline == historical.baseline
    assert semantic.candidate == historical.candidate
    assert semantic.changed == historical.changed
    historical_deterministic = next(
        item for item in historical.evaluators if item.type == "deterministic"
    )
    semantic_deterministic = next(
        item for item in semantic.evaluators if item.type == "deterministic"
    )
    assert historical_deterministic.scorer == HISTORICAL_SCORER.as_posix()
    assert semantic_deterministic.scorer == SCORER.as_posix()
    historical_contract = asdict(historical_deterministic)
    semantic_contract = asdict(semantic_deterministic)
    historical_contract.pop("scorer")
    semantic_contract.pop("scorer")
    assert semantic_contract == historical_contract
    assert next(
        item for item in semantic.evaluators if item.type == "llm_judge"
    ) == next(item for item in historical.evaluators if item.type == "llm_judge")
    for field in (
        "model",
        "harnesses",
        "attempts",
        "concurrency",
        "max_cost_usd",
        "reserve_per_attempt_usd",
        "approval_required",
        "trace_content",
        "preparation_required",
        "evidence_checkpoint_cells",
        "environment",
    ):
        assert getattr(semantic.execution, field) == getattr(
            historical.execution, field
        )
    planned_cells = (
        len(tasks) * 2 * len(semantic.execution.harnesses) * semantic.execution.attempts
    )
    assert planned_cells == 4


def test_failure_replication_scorer_revisions_are_byte_locked() -> None:
    historical = load_comparison(SPEC, repo_root=Path.cwd())
    semantic = load_comparison(SEMANTIC_SPEC, repo_root=Path.cwd())
    historical_scorer = next(
        item for item in historical.evaluators if item.type == "deterministic"
    )
    semantic_scorer = next(
        item for item in semantic.evaluators if item.type == "deterministic"
    )

    assert historical_scorer.scorer == HISTORICAL_SCORER.as_posix()
    assert semantic_scorer.scorer == SCORER.as_posix()
    assert _sha256(HISTORICAL_SCORER) == (
        "ab0b2c5bee685d4523618dfdd1defe72b6c314c729ede1eabe1d71d5046b9366"
    )
    assert _sha256(SCORER) == (
        "602a5693933119a05baa1d36dec81f73303fa46d19b5d3f6180b64754d2db66f"
    )
    assert HISTORICAL_SCORER.read_bytes() != SCORER.read_bytes()


def test_semantic_replication_v2_study_console_is_dedicated_and_read_only() -> None:
    profile = yaml.safe_load(SEMANTIC_STUDY_CONSOLE.read_text(encoding="utf-8"))

    assert profile["research"]["id"] == (
        "anthropic-skill-creator-instruction-semantic-replication-v2"
    )
    assert profile["wandb"] == {
        "entity": "wandb",
        "project": "fugue-anthropic-skill-creator-semantic-replication-v2",
    }
    assert profile["database"] == (
        ".study-console/anthropic-skill-creator-semantic-replication-v2.sqlite3"
    )
    assert profile["presentation"] == {
        "default_study_id": (
            "anthropic-skill-creator-instruction-semantic-replication-v2"
        ),
        "read_only": True,
    }


def test_semantic_requirements_are_public_and_private_truth_is_structural() -> None:
    question = _task()["input"]["question"]
    expected = _private()["expected"]

    for dimension in sorted(SEMANTIC_DIMENSIONS):
        assert f"`{dimension}`" in question
    assert "meaning rather than any prescribed heading or sentence" in question
    assert set(expected) == {
        "skill_name",
        "skill_path",
        "reference_path",
        "allowed_paths",
        "compatibility_terms",
    }
    assert "instruction_groups" not in expected


def test_qualification_fixture_reproduces_only_public_semantic_failures() -> None:
    score = _score()
    private = _private()
    task = {"id": private["id"]}
    evidence = {"expected": private["expected"]}

    baseline = score(task, private["base_output"], evidence)
    gold = score(task, private["gold_output"], evidence)

    assert baseline["artifact_validity"] is True
    assert baseline["dependency_secret_safety"] is True
    assert not any(baseline[name] for name in SEMANTIC_DIMENSIONS)
    assert all(gold.values())


def test_semantic_scorer_accepts_paraphrases_not_historical_literal_phrases() -> None:
    score = _score()
    private = _private()
    task = {"id": private["id"]}
    evidence = {"expected": private["expected"]}
    paraphrased = (
        private["gold_output"]
        .replace(
            "For every finding or recommendation, record the local file that supports it.",
            "Tie each conclusion to the named evidence artifact on which it rests.",
        )
        .replace(
            "Mark VERIFIED only when every required check succeeds against available evidence.",
            "The outcome can qualify as SUCCESS only if all mandatory checks agree.",
        )
        .replace(
            "Mark FAILED and halt when any required check disagrees with the locked manifest.",
            "Reject the outcome and STOP if a required comparison reports a mismatch.",
        )
        .replace(
            "Mark INCONCLUSIVE when a required file is absent or cannot be read.",
            "An unavailable artifact makes the outcome UNKNOWN.",
        )
        .replace(
            "Missing or unreadable evidence remains INCONCLUSIVE and must never be counted as PASS or zero.",
            "Unavailable evidence stays UNKNOWN and cannot become success or 0.",
        )
    )

    assert "cite the source path" not in paraphrased.lower()
    assert "success or stop condition" not in paraphrased.lower()
    scores = score(task, paraphrased, evidence)
    assert all(scores[name] for name in SEMANTIC_DIMENSIONS), scores


@pytest.mark.parametrize(
    "body",
    (
        """
        # Release verification

        ## Source traceability

        A report assertion is admissible only when the supporting on-disk
        artifact is identified by its local path.

        ## Terminal states

        Success means all required records are readable and support the claim.
        Failure-or-stop means a required check contradicts the claim and blocks
        the release. Inconclusive means a required record is absent or unreadable.

        ## Evidence gaps

        Missing material is classified as inconclusive and must not be reported
        as success or zero.
        """,
        """
        # Release verification

        | Contract | Acceptance condition |
        | --- | --- |
        | Source lineage | Every finding must name the local evidence file that supports it. |
        | Terminal success | All required evidence files exist, are readable, and match the claim. |
        | Failure / stop | A contradiction or required-check mismatch blocks the release. |
        | Inconclusive | Missing, unavailable, or unreadable evidence requires this status. |
        | Gap handling | Unavailable evidence is inconclusive, never PASS and never numeric zero. |
        """,
        """
        # Release verification

        Tie each conclusion back to the named local record that substantiates it;
        a conclusion without that provenance may not be published. A clean verdict
        qualifies only if every mandatory check is readable and agrees. Reject and
        halt when a mandatory check reports a mismatch. The outcome is unknown when
        an artifact cannot be opened. Such unavailable material stays unknown and
        cannot be interpreted as verified or 0.
        """,
        """
        # Release verification

        Each recommendation needs to identify the manifest or log that supports it.
        Mark success when every required check affirmatively demonstrates the claim.
        Mark failure and stop when a required check is invalid or reports an error.
        Mark inconclusive when evidence is missing or unavailable. Missing evidence
        must not be counted as a pass or zero.
        """,
    ),
    ids=("headings", "table", "paraphrase", "compact-prose"),
)
def test_semantic_scorer_accepts_public_contract_forms(body: str) -> None:
    private = _private()
    scores = _score()(
        {"id": private["id"]},
        _output_with_body(body),
        {"expected": private["expected"]},
    )

    assert all(scores[name] for name in SEMANTIC_DIMENSIONS), scores


@pytest.mark.parametrize(
    ("body", "failed_dimensions"),
    (
        (
            """
            # Release verification

            Keywords: claims, findings, evidence, files, paths, cite, success,
            failure, stop, inconclusive, missing, unreadable, never, pass, zero.
            """,
            SEMANTIC_DIMENSIONS,
        ),
        (
            """
            # Release verification

            A report may mention a path next to some findings. The possible labels
            are success, failure-or-stop, and inconclusive. No evidence conditions
            are defined for those labels.
            """,
            {
                "source_traceability",
                "terminal_success_or_stop_semantics",
                "missing_evidence_status",
            },
        ),
        (
            """
            # Release verification

            Every finding must name the local file that supports it. Success means
            all required evidence is readable. Failure means a mismatch. Missing
            evidence may be treated as success with zero failures.
            """,
            {"terminal_success_or_stop_semantics", "missing_evidence_status"},
        ),
        (
            """
            # Release verification

            The output lists claims and paths in separate appendices. A glossary
            defines success, failure, stop, and inconclusive. Missing and unreadable
            are useful search terms; pass and zero are other terms.
            """,
            SEMANTIC_DIMENSIONS,
        ),
    ),
    ids=("keyword-list", "labels-without-rules", "coerced-gap", "glossary"),
)
def test_semantic_scorer_rejects_deceptive_or_incomplete_contracts(
    body: str,
    failed_dimensions: set[str],
) -> None:
    private = _private()
    scores = _score()(
        {"id": private["id"]},
        _output_with_body(body),
        {"expected": private["expected"]},
    )

    assert not any(scores[name] for name in failed_dimensions), scores


def test_semantic_dimensions_fail_independently() -> None:
    score = _score()
    private = _private()
    task = {"id": private["id"]}
    evidence = {"expected": private["expected"]}

    no_trace = _replace_skill_body(
        private["gold_output"],
        "For every finding or recommendation, record the local file that supports it.",
        "Summarize the evidence in a concise report.",
    )
    no_failure_terminal = _replace_skill_body(
        private["gold_output"],
        "Mark FAILED and halt when any required check disagrees with the locked manifest.",
        "Record disagreements for later review.",
    )
    coerced_missing = _replace_skill_body(
        private["gold_output"],
        "Missing or unreadable evidence remains INCONCLUSIVE and must never be counted as PASS or zero.",
        "Missing evidence may be counted as PASS with a value of zero.",
    )

    trace_scores = score(task, no_trace, evidence)
    terminal_scores = score(task, no_failure_terminal, evidence)
    missing_scores = score(task, coerced_missing, evidence)

    assert trace_scores["source_traceability"] is False
    assert trace_scores["terminal_success_or_stop_semantics"] is True
    assert terminal_scores["source_traceability"] is True
    assert terminal_scores["terminal_success_or_stop_semantics"] is False
    assert missing_scores["terminal_success_or_stop_semantics"] is True
    assert missing_scores["missing_evidence_status"] is False


def test_schema_and_dependency_safety_remain_fail_closed() -> None:
    score = _score()
    private = _private()
    task = {"id": private["id"]}
    evidence = {"expected": private["expected"]}
    wrong_task = private["gold_output"].replace(
        '"task_id":"create-traceable-release-evidence-skill-v2"',
        '"task_id":"another-task"',
    )
    unsafe = _replace_skill_body(
        private["gold_output"],
        "Inventory the supplied local files without contacting the network.",
        "Run pip install release-checker before inventorying the local files.",
    )

    assert score(task, wrong_task, evidence)["artifact_validity"] is False
    assert score(task, unsafe, evidence)["dependency_secret_safety"] is False
