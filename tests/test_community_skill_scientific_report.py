from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    ComparisonResultV2,
    ComparisonResultV3,
    load_comparison,
)

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
    attempt: int = 1,
) -> _Record:
    attempt_id = hashlib.sha256(f"{task_id}:{arm}:{attempt}".encode()).hexdigest()
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
    study = copy.deepcopy(study)
    study["_manifest"] = {
        "id": "community-skill-upgrade-canary-v1",
        "path": "campaign-manifest.json",
        "sha256": _sha(CAMPAIGN / "campaign-manifest.json"),
    }
    study["_study_contract_digest"] = stable_digest(
        {key: value for key, value in study.items() if not key.startswith("_")}
    )
    study["_spec"] = {
        "path": SPEC_PATH.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha(SPEC_PATH),
    }
    template = json.loads(REPORTER.TEMPLATE.read_text())
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
        "schema_version": 3,
        "comparison_id": spec.id,
        "preview_digest": "4" * 64,
        "qualification_digest": "5" * 64,
        "result_digest": "5" * 64,
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


def _result_v2() -> tuple[ComparisonResultV2, Any, dict[str, Any], dict[str, Any]]:
    v3, spec, study, template = _result()
    result = object.__new__(ComparisonResultV2)
    values = {
        "schema_version": 2,
        "comparison_id": v3.comparison_id,
        "preview_digest": v3.preview_digest,
        "qualification_digest": v3.qualification_digest,
        "result_digest": v3.result_digest,
        "evidence_project": v3.evidence_project,
        "evidence_destination": spec.execution.evidence_destination.to_dict(),
        "integrity": {"status": "reconciled"},
        "candidate_source_revisions": v3.candidate_source_revisions,
        "paired_cases": v3.paired_cases,
        "rows": v3.rows,
        "behavioral_summary": _Record(
            status="incomplete",
            recommendation="INCOMPLETE — required behavioral evidence is unavailable.",
            supported_claim=None,
            critical_blockers=("one required evaluation is incomplete",),
            next_action="Repair the missing evaluation before comparison.",
            limitations=("One aligned attempt is incomplete.",),
        ),
        "decision": v3.decision,
        "limitations": ("This source artifact is a canonical V2 result.",),
        "mechanism_summary": v3.mechanism_summary,
        "operational_summary": v3.operational_summary,
    }
    for key, value in values.items():
        object.__setattr__(result, key, value)
    # V2 can carry terminal evidence without a completed optional judge review.
    result.paired_cases[0].baseline.judge_reviews = {}
    result.paired_cases[1].candidate.execution_status = "failed"
    result.paired_cases[1].status = "incomplete"
    return result, spec, study, template


def _generate_report(
    *,
    result: ComparisonResultV2 | ComparisonResultV3,
    spec: Any,
    study: Mapping[str, Any],
    template: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    return REPORTER.generate_report(
        result=result,
        spec=spec,
        study=study,
        template=template,
        repo_root=REPO_ROOT,
        campaign_root=CAMPAIGN,
        source_result_sha256="6" * 64,
        canonical_result_verified=True,
        **kwargs,
    )


def _confirmatory_inputs() -> tuple[
    ComparisonResultV3,
    Any,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    result, spec, study, template = _result()
    object.__setattr__(spec.execution, "attempts", 4)
    task_ids = ("credential-rotation-plan-v3", "evidence-destination-plan-v3")
    pairs = []
    for task_id in task_ids:
        for attempt_number in range(1, 5):
            baseline = _attempt(
                task_id=task_id,
                arm="baseline",
                passed=False,
                observed_judge_cost=True,
                attempt=attempt_number,
            )
            candidate = _attempt(
                task_id=task_id,
                arm="candidate",
                passed=True,
                observed_judge_cost=True,
                attempt=attempt_number,
            )
            pairs.append(
                _Record(
                    pair_id=hashlib.sha256(
                        f"{task_id}:{attempt_number}".encode()
                    ).hexdigest(),
                    task_id=task_id,
                    task_label=task_id.replace("-", " ").title(),
                    harness="claude-code",
                    attempt=attempt_number,
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
                                "baseline_explanation": "Grounding was absent.",
                                "candidate_explanation": "Grounding was present.",
                            }
                        ),
                    ),
                    baseline=baseline,
                    candidate=candidate,
                )
            )
    object.__setattr__(result, "paired_cases", tuple(pairs))
    object.__setattr__(result, "rows", 16)
    result.operational_summary.update(
        {
            "cost_rows": 16,
            "accounted_cost_rows": 16,
            "observed_cost_usd": 7.2,
            "accounted_cost_usd": 8.8,
        }
    )
    for stage in result.mechanism_summary.values():
        for arm in stage.values():
            arm["observed"] = 8
            arm["applicable"] = 8
    study["attempts"] = 4
    study["expected_cells"] = 16
    preregistration_path = (
        REPO_ROOT / "examples/comparisons/superpowers-writing-plans-upgrade/"
        "preregistration-confirmatory-v1.json"
    )
    study["preregistration"] = [
        {
            "role": "study_protocol",
            "path": (
                "../superpowers-writing-plans-upgrade/"
                "preregistration-confirmatory-v1.json"
            ),
            "sha256": _sha(preregistration_path),
            "identity_field": "preregistration_id",
            "identity": "superpowers-writing-plans-confirmatory-v1",
            "applies_to": spec.id,
        }
    ]
    study["trace_audit"] = {
        "minimum_fraction": 0.25,
        "required_behavior_families": {
            "identity": [task_ids[0]],
            "routing": [task_ids[1]],
        },
    }
    study["_study_contract_digest"] = stable_digest(
        {key: value for key, value in study.items() if not key.startswith("_")}
    )
    selected_pairs = []
    for partition, pair in zip(
        ("development", "holdout"),
        (result.paired_cases[0], result.paired_cases[4]),
        strict=True,
    ):
        attempt_ids = sorted((pair.baseline.attempt_id, pair.candidate.attempt_id))
        selected_pairs.append(
            {
                "pair_token": hashlib.sha256(
                    (
                        f"{result.preview_digest}:pair:{pair.task_id}:"
                        f"{pair.harness}:{pair.attempt}"
                    ).encode()
                ).hexdigest(),
                "task_id": pair.task_id,
                "harness": pair.harness,
                "attempt": pair.attempt,
                "partition": partition,
                "artifact_a_attempt_id": attempt_ids[0],
                "artifact_b_attempt_id": attempt_ids[1],
            }
        )
    selection = {
        "schema_version": 1,
        "kind": "blinded_trace_audit_selection",
        "preview_digest": result.preview_digest,
        "selection_frozen_before_execution": True,
        "sampling_fraction": 0.25,
        "population_pairs": len(result.paired_cases),
        "selected_pairs": selected_pairs,
        "selected_attempt_ids": sorted(
            attempt_id
            for pair in selected_pairs
            for attempt_id in (
                pair["artifact_a_attempt_id"],
                pair["artifact_b_attempt_id"],
            )
        ),
        "post_result_additions": {
            "all_discordant_pairs": "required",
            "all_critical_failures": "required",
        },
    }
    selection["selection_digest"] = stable_digest(selection)
    required_ids = sorted(
        attempt.attempt_id
        for pair in result.paired_cases
        for attempt in (pair.baseline, pair.candidate)
    )
    checks = {key: True for key in REPORTER._AUDIT_REQUIRED_CHECKS}
    review: dict[str, Any] = {
        "schema_version": 1,
        "kind": "completed_blinded_trace_audit",
        "campaign_id": study["_manifest"]["id"],
        "study_id": result.comparison_id,
        "status": "completed",
        "preview_digest": result.preview_digest,
        "result_digest": result.result_digest,
        "selection_digest": selection["selection_digest"],
        "selected_attempt_ids": selection["selected_attempt_ids"],
        "required_attempt_ids": required_ids,
        "reviewed_attempts": [
            {
                "attempt_id": attempt_id,
                "reviews": [
                    {
                        "reviewer": reviewer,
                        "disposition": "verified",
                        "checks": checks,
                        "reason": "Every frozen integrity check was verified.",
                    }
                    for reviewer in ("reviewer-a", "reviewer-b")
                ],
                "adjudication": None,
            }
            for attempt_id in required_ids
        ],
        "reviewer_attestations": [],
        "limitations": ["The trace audit validates lineage and mechanism evidence."],
    }
    attested_digest = REPORTER._audit_attested_digest(review)
    review["reviewer_attestations"] = [
        {
            "reviewer": reviewer,
            "reviewed_at": "2026-08-03T18:00:00Z",
            "artifact_digest": attested_digest,
        }
        for reviewer in ("reviewer-a", "reviewer-b")
    ]
    return result, spec, study, template, selection, review


def test_report_is_decision_ready_without_private_labels() -> None:
    result, spec, study, template = _result()

    report = _generate_report(
        result=result,
        spec=spec,
        study=study,
        template=template,
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
    assert report["source_result"]["canonical_reader_verified"] is True
    assert report["trace_audit"]["status"] == ("not_required_for_one_attempt_canary")
    assert REPORTER.CANARY_LIMITATION[0] in report["limitations"]
    serialized = json.dumps(report, sort_keys=True)
    assert "authored_reference" not in serialized
    assert "private_labels" not in serialized
    assert '"expected":' not in serialized
    REPORTER.validate_report(report)


def test_four_attempt_v3_report_binds_preregistration_and_completed_audit() -> None:
    result, spec, study, template, selection, review = _confirmatory_inputs()

    report = _generate_report(
        result=result,
        spec=spec,
        study=study,
        template=template,
        audit_selection=selection,
        audit_review=review,
        audit_selection_sha256="7" * 64,
        audit_review_sha256="8" * 64,
    )

    assert report["study_contract"]["attempts_per_task_arm"] == 4
    assert report["study_contract"]["planned_cells"] == 16
    assert report["preregistration"]["status"] == "bound"
    assert report["trace_audit"]["status"] == "completed"
    assert report["trace_audit"]["required_attempts"] == 16
    assert REPORTER.CANARY_LIMITATION[0] not in report["limitations"]
    assert len(report["deterministic_results"]) == 8
    assert len(report["judge_results"]) == 16
    assert len(report["mechanism_results"]) == 12
    assert len(report["evidence_links"]) == 80
    REPORTER.validate_report(report)


def test_confirmatory_report_rejects_stale_or_unsigned_audit() -> None:
    result, spec, study, template, selection, review = _confirmatory_inputs()
    review["result_digest"] = "9" * 64

    with pytest.raises(ValueError, match="different immutable inputs"):
        _generate_report(
            result=result,
            spec=spec,
            study=study,
            template=template,
            audit_selection=selection,
            audit_review=review,
            audit_selection_sha256="7" * 64,
            audit_review_sha256="8" * 64,
        )


def test_confirmatory_report_requires_completed_audit() -> None:
    result, spec, study, template, _selection, _review = _confirmatory_inputs()

    with pytest.raises(ValueError, match="completed frozen trace audit"):
        _generate_report(
            result=result,
            spec=spec,
            study=study,
            template=template,
        )


def test_v2_report_is_conservative_about_unavailable_v3_contracts() -> None:
    result, spec, study, template = _result_v2()

    report = _generate_report(
        result=result,
        spec=spec,
        study=study,
        template=template,
    )

    assert report["behavioral_finding"] == {
        "source_result_schema_version": 2,
        "task_validity_basis": "not_assessed_in_v2",
        "evidence_topology_basis": "unavailable_in_v2",
        "status": "incomplete",
        "recommendation": "INCOMPLETE — required behavioral evidence is unavailable.",
        "supported_claim": None,
        "critical_blockers": ["one required evaluation is incomplete"],
        "next_action": "Repair the missing evaluation before comparison.",
        "release_decision": result.decision.to_dict(),
    }
    assert {item["status"] for item in report["task_validity"]} == {"not_assessed"}
    assert all(item["reasons"] for item in report["task_validity"])
    assert all(
        arm["presentation_evidence_status"] == "unavailable_in_v2"
        and arm["score_explanations"] == {}
        and arm["sanitized_answer_excerpt"] is None
        for row in report["deterministic_results"]
        for arm in (row["baseline"], row["candidate"])
    )
    unavailable = [
        item for item in report["judge_results"] if item["status"] == "unavailable"
    ]
    assert len(unavailable) == 1
    assert unavailable[0]["label"] is None
    assert report["efficiency"]["cost"]["advisory_judge"]["expected_reviews"] == 4
    assert list(REPORTER.V2_LIMITATIONS) == report["limitations"][3:6]
    REPORTER.validate_report(report)


def test_v2_report_rejects_candidate_revision_mismatch() -> None:
    result, spec, study, template = _result_v2()
    result.candidate_source_revisions[0].version_identity = "git:" + "0" * 40

    with pytest.raises(ValueError, match="candidate Skill revision disagrees"):
        _generate_report(
            result=result,
            spec=spec,
            study=study,
            template=template,
        )


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
        _generate_report(
            result=result,
            spec=spec,
            study=study,
            template=template,
        )


def test_template_and_report_schema_are_strict() -> None:
    result, spec, study, template = _result()
    drifted = copy.deepcopy(template)
    drifted["unexpected"] = True
    with pytest.raises(ValueError, match="template fields drifted"):
        _generate_report(
            result=result,
            spec=spec,
            study=study,
            template=drifted,
        )

    report = _generate_report(
        result=result,
        spec=spec,
        study=study,
        template=template,
    )
    report["private_labels"] = {"secret": True}
    with pytest.raises(ValueError, match="fields disagree"):
        REPORTER.validate_report(report)
