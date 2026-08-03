from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import ComparisonResultV3, load_comparison

REPO_ROOT = Path(__file__).parents[1]
CAMPAIGN = REPO_ROOT / "examples/comparisons/community-skill-upgrades"
SPEC_PATH = (
    REPO_ROOT
    / "examples/comparisons/superpowers-writing-plans-upgrade/comparison-v4.yaml"
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPORTER = _load_module(
    "generate_scientific_report",
    CAMPAIGN / "generate_scientific_report.py",
)


class _Record(SimpleNamespace):
    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _link(kind: str, suffix: str) -> _Record:
    return _Record(
        kind=kind,
        status="resolved",
        ref=f"weave:///wandb/fugue-superpowers-writing-plans-upgrade-v1/call/{suffix}",
        url=(
            "https://wandb.ai/wandb/"
            f"fugue-superpowers-writing-plans-upgrade-v1/weave/calls/{suffix}"
        ),
    )


def _attempt(
    *,
    task_id: str,
    arm: str,
    passed: bool,
    observed_judge_cost: bool,
) -> _Record:
    attempt_id = hashlib.sha256(f"{task_id}:{arm}".encode()).hexdigest()
    links = [
        _link("evaluation_root", f"{attempt_id}-evaluation"),
        _link("prediction_and_score", f"{attempt_id}-score"),
        _link("prediction", f"{attempt_id}-prediction"),
        _link("agent_root", f"{attempt_id}-agent"),
        _Record(
            kind="dataset",
            status="resolved",
            ref=(
                "weave:///wandb/fugue-superpowers-writing-plans-upgrade-v1/"
                "object/community-skill-tasks:v1"
            ),
            url=(
                "https://wandb.ai/wandb/"
                "fugue-superpowers-writing-plans-upgrade-v1/weave/objects/"
                "community-skill-tasks/versions/v1"
            ),
        ),
    ]
    return _Record(
        attempt_id=attempt_id,
        passed=passed,
        execution_status="completed",
        evidence_status="reconciled",
        evidence_links=links,
        scores={"plan.repository_grounding": passed},
        score_explanations={
            "plan.repository_grounding": (
                "Grounded in inspected repository paths."
                if passed
                else "Required repository paths were not inspected."
            )
        },
        sanitized_answer_excerpt="A bounded public plan excerpt.",
        tools=("Read", "Write"),
        judge_reviews={
            "community-usefulness": _Record(
                label="strong" if passed else "adequate",
                reason=(
                    "The plan is grounded and reviewable."
                    if passed
                    else "The plan needs more repository grounding."
                ),
                missing_evidence=False,
                cost_status="observed" if observed_judge_cost else "unavailable",
                observed_cost_usd=0.02 if observed_judge_cost else None,
                accounted_reserve_usd=0.1,
            )
        },
        latency_sec=12.0 if arm == "candidate" else 10.0,
        input_tokens=1000.0,
        output_tokens=500.0,
        cost_usd=0.5 if arm == "candidate" else 0.4,
    )


def _result() -> tuple[ComparisonResultV3, Any, dict[str, Any], dict[str, Any]]:
    spec = load_comparison(SPEC_PATH, repo_root=REPO_ROOT)
    manifest = json.loads((CAMPAIGN / "campaign-manifest.json").read_text())
    study = next(item for item in manifest["studies"] if item["id"] == spec.id)
    template = json.loads((CAMPAIGN / "scientific-report-template.json").read_text())
    task_ids = ("credential-rotation-plan-v3", "evidence-destination-plan-v3")
    pairs = []
    for task_id in task_ids:
        baseline = _attempt(
            task_id=task_id,
            arm="baseline",
            passed=False,
            observed_judge_cost=False,
        )
        candidate = _attempt(
            task_id=task_id,
            arm="candidate",
            passed=True,
            observed_judge_cost=True,
        )
        pairs.append(
            _Record(
                pair_id=hashlib.sha256(task_id.encode()).hexdigest(),
                task_id=task_id,
                task_label=task_id.replace("-", " ").title(),
                harness="claude-code",
                attempt=1,
                status="improved",
                dimension_changes=(
                    _Record(
                        value={
                            "id": "plan.repository_grounding",
                            "label": "Repository grounding",
                            "status": "improved",
                            "baseline": False,
                            "candidate": True,
                            "critical": False,
                            "role": "outcome",
                            "baseline_explanation": (
                                "Required repository paths were not inspected."
                            ),
                            "candidate_explanation": (
                                "Grounded in inspected repository paths."
                            ),
                        }
                    ),
                ),
                baseline=baseline,
                candidate=candidate,
            )
        )

    revisions = {
        "baseline": study["baseline_commit"],
        "candidate": study["candidate_commit"],
    }
    runtime_digests = {
        "baseline": "sha256:" + "1" * 64,
        "candidate": "sha256:" + "2" * 64,
    }
    arms = {}
    for arm, candidate in (("baseline", spec.baseline), ("candidate", spec.candidate)):
        arms[arm] = {
            "behavior_digest": stable_digest(candidate.behavior()),
            "source_revisions": [
                {
                    "kind": "skill",
                    "id": candidate.skills[0],
                    "version_identity": f"git:{revisions[arm]}",
                    "runtime_digest": runtime_digests[arm],
                }
            ],
        }
    destination = spec.execution.evidence_destination
    assert destination is not None
    cohort = {
        "schema_version": 1,
        "source_lock_digest": "3" * 64,
        "taskset_digest": _sha(REPO_ROOT / spec.taskset.tasks),
        "private_labels_digest": _sha(REPO_ROOT / spec.taskset.private_labels),
        "arms": arms,
        "execution": {
            "model": spec.execution.model,
            "harnesses": list(spec.execution.harnesses),
            "trace_content": spec.execution.trace_content,
            "environment_digest": stable_digest(spec.execution.environment),
            "source_evidence_project": spec.execution.source_evidence_project,
            "source_evidence_destination": None,
            "result_evidence_project": spec.execution.evidence_project,
            "result_evidence_destination": destination.to_dict(),
        },
        "scorer_digests": {
            item.id: REPORTER._evaluator_digest(item, REPO_ROOT)
            for item in spec.evaluators
        },
    }
    cohort["lineage_digest"] = stable_digest(cohort)

    result = object.__new__(ComparisonResultV3)
    values = {
        "comparison_id": spec.id,
        "evidence_project": spec.execution.evidence_project,
        "evidence_topology": _Record(result_destination=destination),
        "integrity": {"status": "reconciled"},
        "cohort_lineage": cohort,
        "candidate_source_revisions": (
            _Record(
                kind="skill",
                id=spec.candidate.skills[0],
                version_identity=f"git:{revisions['candidate']}",
                runtime_digest=runtime_digests["candidate"],
            ),
        ),
        "paired_cases": tuple(pairs),
        "rows": 4,
        "task_validity": tuple(
            _Record(
                value={
                    "task_id": task_id,
                    "status": "valid",
                    "discriminating_dimensions": ["plan.repository_grounding"],
                }
            )
            for task_id in task_ids
        ),
        "behavioral_summary": _Record(
            status="improved",
            recommendation="Advance this exact upgrade to a repeated confirmation.",
            supported_claim="Candidate improved repository grounding on both tasks.",
            critical_blockers=(),
            next_action="Run a separately approved repeated confirmation.",
            limitations=("This result covers two locked planning tasks.",),
        ),
        "decision": _Record(
            limitations=("This is behavioral evidence, not a package release gate.",),
            value={
                "status": "inconclusive",
                "recommendation": "No package release decision was evaluated.",
                "evidence_grade": "A",
                "gates": [],
                "critical_blockers": [],
                "limitations": [
                    "This is behavioral evidence, not a package release gate."
                ],
                "next_action": "Run a repeated confirmation.",
                "human_signoff_required": False,
            },
        ),
        "limitations": ("The canary has one attempt per arm.",),
        "mechanism_summary": {
            stage: {
                arm: {"observed": 2, "applicable": 2, "unavailable": 0}
                for arm in ("baseline", "candidate")
            }
            for stage in (
                "skill_assigned",
                "skill_registered",
                "skill_invoked",
                "relevant_source_returned",
                "relevant_source_opened",
                "relevant_source_used",
            )
        },
        "operational_summary": {
            "observed_cost_usd": 1.8,
            "cost_rows": 4,
            "accounted_cost_usd": 2.2,
            "accounted_cost_rows": 4,
        },
    }
    for key, value in values.items():
        object.__setattr__(result, key, value)
    return result, spec, study, template


def test_report_is_decision_ready_without_private_labels() -> None:
    result, spec, study, template = _result()

    report = REPORTER.generate_report(
        result=result,
        spec=spec,
        study=study,
        template=template,
        repo_root=REPO_ROOT,
    )

    assert report["status"] == "completed"
    assert report["behavioral_finding"]["status"] == "improved"
    assert report["exact_revisions"] == {
        "baseline": study["baseline_commit"],
        "candidate": study["candidate_commit"],
    }
    assert len(report["deterministic_results"]) == 2
    assert len(report["judge_results"]) == 4
    assert {item["label"] for item in report["judge_results"]} == {
        "adequate",
        "strong",
    }
    assert report["efficiency"]["cost"]["advisory_judge"] == {
        "status": "partial",
        "observed_reviews": 2,
        "expected_reviews": 4,
        "observed_total_usd": 0.04,
        "accounted_reserve_usd": 0.4,
    }
    assert len(report["evidence_links"]) == 20
    serialized = json.dumps(report, sort_keys=True)
    assert "authored_reference" not in serialized
    assert "private_labels" not in serialized
    assert '"expected":' not in serialized
    REPORTER.validate_report(report)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("integrity", "reconciled result integrity"),
        ("project", "project disagrees"),
        ("revision", "Skill revision disagrees"),
        ("evidence", "reconciled attempt evidence"),
        ("terminal", "terminal attempts"),
    ],
)
def test_report_fails_closed_on_mismatched_or_unqualified_input(
    mutation: str,
    message: str,
) -> None:
    result, spec, study, template = _result()
    if mutation == "integrity":
        result.integrity["status"] = "invalid"
    elif mutation == "project":
        object.__setattr__(result, "evidence_project", "wandb/wrong-project")
    elif mutation == "revision":
        result.cohort_lineage["arms"]["candidate"]["source_revisions"][0][
            "version_identity"
        ] = "git:" + "0" * 40
    elif mutation == "evidence":
        result.paired_cases[0].baseline.evidence_status = "missing"
    elif mutation == "terminal":
        result.paired_cases[0].baseline.execution_status = "running"

    with pytest.raises(ValueError, match=message):
        REPORTER.generate_report(
            result=result,
            spec=spec,
            study=study,
            template=template,
            repo_root=REPO_ROOT,
        )


def test_template_and_report_schema_are_strict() -> None:
    result, spec, study, template = _result()
    drifted = copy.deepcopy(template)
    drifted["unexpected"] = True
    with pytest.raises(ValueError, match="template fields drifted"):
        REPORTER.generate_report(
            result=result,
            spec=spec,
            study=study,
            template=drifted,
            repo_root=REPO_ROOT,
        )

    report = REPORTER.generate_report(
        result=result,
        spec=spec,
        study=study,
        template=template,
        repo_root=REPO_ROOT,
    )
    report["private_labels"] = {"secret": True}
    with pytest.raises(ValueError, match="fields disagree"):
        REPORTER.validate_report(report)
