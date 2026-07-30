#!/usr/bin/env python3
"""Create and validate the blinded 48-example maintainer-judge calibration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json

EXAMPLE_COUNT = 48
MIN_RATE = 0.85
JUDGE_ID = "maintainer-actionability"
JUDGE_PROFILE = "wandb/zai-org/GLM-5.2"
RUBRIC = (
    "Judge only whether the maintainer memo turns the response's stated "
    "evidence into a concrete maintain, advance, or block recommendation; "
    "traces that action to the stated evidence; proposes one bounded feasible "
    "next step with a success or stop condition; and calibrates uncertainty to "
    "the inspected cohort. Do not rescore factual correctness, tool mechanics, "
    "revision identity, verbosity, cost, or latency; the deterministic scorer "
    "owns those. Unsupported causal, completeness, or release-wide claims are "
    "critical failures."
)
DIMENSIONS = (
    "maintenance_actionability",
    "evidence_to_action_traceability",
    "bounded_next_step",
    "uncertainty_calibration",
)
EVIDENCE = ("tool_names",)
PASS_THRESHOLD = 0.75

_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "inventory",
        "question": "Summarize the inspected qualification cohort.",
        "pass": (
            "I inspected six locked runs and 24 source conversations. This is "
            "the qualification cohort, not complete project or release coverage."
        ),
        "pass_next_steps": (
            "Keep the release gate blocked; repeat this six-run inventory after "
            "the source lock refresh and stop if any count or identity drifts.",
            "Advance the cohort check, not the package; verify one fresh locked "
            "snapshot and proceed only if the same bounded counts reconcile.",
            "Maintain the current source lock; inspect exactly one replacement "
            "cohort and stop the study if its object versions are not immutable.",
        ),
        "fail_variants": (
            "The project is complete and the release is healthy because all "
            "visible runs look successful.",
            "Six runs and 24 conversations were inspected.",
            "Search every W&B project and keep collecting evidence until nothing "
            "else can possibly be found.",
        ),
        "tools": ["probe_project_tool", "query_wandb_tool"],
        "critical": True,
        "split": "calibration",
    },
    {
        "id": "source-use",
        "question": "Explain a returned-but-not-opened source pattern.",
        "pass": (
            "Four current sources were returned and one was opened. That is an "
            "observed mechanism pattern; the evidence does not establish cause."
        ),
        "pass_next_steps": (
            "Keep the evidence-policy change under review; rerun one locked "
            "trace and stop when every returned current source is either opened "
            "or explicitly declined.",
            "Do not attribute the failure yet; add one source-use assertion and "
            "accept it only when returned, opened, and used counts reconcile.",
            "Investigate the opening step first; test one sanitized failure card "
            "and stop if the same source remains returned but unopened.",
        ),
        "fail_variants": (
            "Retrieval is broken. The server never returned a current source, "
            "so the candidate must be reverted.",
            "Four sources were returned and one was opened, which is interesting.",
            "Replace the entire retrieval stack and rerun every historical trace "
            "without a budget or stopping rule.",
        ),
        "tools": ["query_weave_traces_tool", "resolve_trace_roots_tool"],
        "critical": False,
        "split": "calibration",
    },
    {
        "id": "evaluation",
        "question": "Compare two aligned evaluation revisions.",
        "pass": (
            "The eight aligned pairs contain one regression. The Evaluation "
            "calls and prediction rows reconcile; no causal claim is supported."
        ),
        "pass_next_steps": (
            "Block this candidate pair; inspect the one discordant row and stop "
            "when its prediction and Evaluation lineage either reconcile or fail.",
            "Hold the behavioral claim; rerun only that aligned case and advance "
            "if both attempts preserve the reconciled row count.",
            "Maintain the baseline for now; add one locked replication and stop "
            "if the regression repeats without missing evidence.",
        ),
        "fail_variants": (
            "The new MCP caused regressions everywhere, so it should not ship.",
            "There was one regression among eight aligned pairs.",
            "Launch another hundred unaligned evaluations and choose whichever "
            "aggregate score is highest.",
        ),
        "tools": ["summarize_evaluation_tool", "query_weave_traces_tool"],
        "critical": False,
        "split": "calibration",
    },
    {
        "id": "latency",
        "question": "Investigate the latency anomaly.",
        "pass": (
            "The opened run history shows 4200 ms versus a 920 ms cohort "
            "median. The broad-read pattern is observable, but cause is unproven."
        ),
        "pass_next_steps": (
            "Investigate before release; compare one additional locked run and "
            "stop if broad reads do not co-vary with the measured latency.",
            "Keep the anomaly open; reproduce step 3 once and accept the signal "
            "only if the 4200 ms value and read count both recur.",
            "Do not change the MCP yet; inspect one bounded history window and "
            "stop when latency is measured with complete cost evidence.",
        ),
        "fail_variants": (
            "The slow run proves the 0.4 implementation is 4.6 times slower.",
            "The opened run has 4200 ms latency while the cohort median is 920 ms.",
            "Profile every run indefinitely until the exact root cause is found.",
        ),
        "tools": ["compare_runs_tool", "get_run_history_tool"],
        "critical": False,
        "split": "calibration",
    },
    {
        "id": "cost",
        "question": "Report observed cost coverage.",
        "pass": (
            "Five of six runs report observed cost totaling $0.96. One value is "
            "missing, so total cohort cost and complete coverage are unavailable."
        ),
        "pass_next_steps": (
            "Hold the cost claim; recover the one missing observation and stop "
            "when all six locked runs have explicit values.",
            "Report only the five-run subtotal; rerun the missing-cost case once "
            "and advance when its usage receipt is present.",
            "Keep efficiency unqualified; inspect one bounded billing trace and "
            "stop if it cannot resolve the absent value.",
        ),
        "fail_variants": (
            "All six runs cost $0.96 in total; the missing value can be treated "
            "as zero.",
            "Five runs report $0.96 and one run has no cost.",
            "Query all account billing records without a time bound before "
            "making any maintenance decision.",
        ),
        "tools": ["query_wandb_tool", "get_run_history_tool"],
        "critical": True,
        "split": "calibration",
    },
    {
        "id": "priority",
        "question": "Recommend one maintenance action.",
        "pass": (
            "Prioritize projected reads and structured incomplete-evidence "
            "handling, supported by the latency run, partial case, evaluation "
            "call, and reviewed source-use trace. Validate on a new locked study."
        ),
        "pass_next_steps": (
            "Keep release approval pending; run one four-task confirmation and "
            "stop on any critical honesty or scope regression.",
            "Advance only the combined intervention to holdout; require two "
            "aligned passes and stop if either loses projected reads.",
            "Maintain the current package while testing one pre-frozen cohort; "
            "proceed when latency and evidence completeness stay within policy.",
        ),
        "fail_variants": (
            "Ship 0.4 immediately because its tool names suggest better search.",
            "Projected reads and structured errors should be the priorities.",
            "Rewrite every query tool, deploy it broadly, and decide later how "
            "success should be measured.",
        ),
        "tools": ["diagnose_run_tool", "query_wandb_tool"],
        "critical": False,
        "split": "calibration",
    },
    {
        "id": "partial",
        "question": "Handle a missing candidate evaluation object.",
        "pass": (
            "Agent execution completed, but candidate evaluation is unavailable. "
            "Restore or rerun only the evaluation before comparing outcomes."
        ),
        "pass_next_steps": (
            "Hold the comparison; restore that one Evaluation object and stop "
            "when its prediction row and Agent root both resolve.",
            "Keep Agent execution as mechanism evidence only; rerun the missing "
            "evaluation once and advance if lineage is complete.",
            "Do not assign a zero; repair the single publication gap and stop if "
            "the new row still lacks a prediction-and-score parent.",
        ),
        "fail_variants": (
            "The candidate failed with score zero; rerun all Agent attempts.",
            "Agent execution completed but the Evaluation object is unavailable.",
            "Delete the study and repeat every paid cell until a complete result "
            "appears.",
        ),
        "tools": ["count_weave_traces_tool", "infer_trace_schema_tool"],
        "critical": True,
        "split": "holdout",
    },
    {
        "id": "release",
        "question": "Decide whether the evidence supports a release-wide claim.",
        "pass": (
            "No. The evidence covers the locked non-sensitive cohort only, and "
            "one required measurement is missing. It cannot support a "
            "release-wide quality claim."
        ),
        "pass_next_steps": (
            "Keep the package decision on hold; qualify one independent cohort "
            "and stop if any required measurement remains absent.",
            "Advance only to a bounded confirmation; require every locked "
            "release-note behavior once and stop on a critical regression.",
            "Maintain the narrow cohort claim; obtain a signed package-gate "
            "receipt and stop if its runtime differs from the tested tree.",
        ),
        "fail_variants": (
            "Yes. Every expected row was inspected, so 0.4 is better for all "
            "users and workloads.",
            "The locked cohort has all expected rows except one measurement.",
            "Release globally now and monitor all future workloads without a "
            "predeclared rollback threshold.",
        ),
        "tools": [
            "list_artifact_versions_tool",
            "get_artifact_details_tool",
            "compare_artifact_versions_tool",
        ],
        "critical": True,
        "split": "holdout",
    },
)


def calibration_template() -> list[dict[str, Any]]:
    """Return 24 authored positive and 24 authored negative examples."""
    rows: list[dict[str, Any]] = []
    for scenario in _SCENARIOS:
        for variant in range(3):
            for label in ("pass", "fail"):
                response = (
                    f"{scenario['pass']} {scenario['pass_next_steps'][variant]}"
                    if label == "pass"
                    else scenario["fail_variants"][variant]
                )
                rows.append(
                    {
                        "id": f"{scenario['id']}-{label}-{variant + 1:02d}",
                        "scenario_id": scenario["id"],
                        "split": scenario["split"],
                        "public_task": {"question": scenario["question"]},
                        "response": response,
                        "permitted_evidence": {
                            "tool_names": scenario["tools"],
                        },
                        "authored_reference": {
                            "label": label,
                            "critical_unsupported_completeness": bool(
                                label == "fail" and scenario["critical"]
                            ),
                        },
                        "judge_result": None,
                        "reviews": [],
                        "adjudicated_label": None,
                    }
                )
    return rows


def read_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"calibration line {number} must be an object")
        rows.append(value)
    return rows


def validate_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXAMPLE_COUNT:
        raise ValueError(f"calibration must contain exactly {EXAMPLE_COUNT} examples")
    ids = [str(item.get("id") or "") for item in rows]
    if len(set(ids)) != EXAMPLE_COUNT or any(not value for value in ids):
        raise ValueError("calibration example IDs must be non-empty and unique")
    reference = [
        str((item.get("authored_reference") or {}).get("label") or "") for item in rows
    ]
    if reference.count("pass") != 24 or reference.count("fail") != 24:
        raise ValueError("calibration authored references must be balanced 24/24")
    responses = [str(item.get("response") or "") for item in rows]
    if len(set(responses)) != EXAMPLE_COUNT or any(not item for item in responses):
        raise ValueError("calibration responses must be non-empty and distinct")
    _validate_template_splits(rows, reference)

    completed = True
    final_labels: list[str] = []
    judge_labels: list[str] = []
    evaluated_splits: list[str] = []
    critical_false_passes = 0
    reviewer_ids: set[str] = set()
    disagreement_count = 0
    for item in rows:
        reviews = item.get("reviews")
        judge = item.get("judge_result")
        if (
            not isinstance(reviews, list)
            or len(reviews) != 2
            or not isinstance(judge, Mapping)
        ):
            completed = False
            continue
        reviewers: list[str] = []
        review_labels: list[str] = []
        for review in reviews:
            if not isinstance(review, Mapping):
                raise ValueError(f"{item['id']} has a malformed review")
            reviewer = str(review.get("reviewer") or "")
            label = str(review.get("label") or "")
            if not reviewer or label not in {"pass", "fail"}:
                raise ValueError(f"{item['id']} has an incomplete review")
            reviewers.append(reviewer)
            review_labels.append(label)
            reviewer_ids.add(reviewer)
        if len(set(reviewers)) != 2:
            raise ValueError(f"{item['id']} requires two distinct reviewers")
        adjudicated = item.get("adjudicated_label")
        if review_labels[0] != review_labels[1]:
            disagreement_count += 1
            if adjudicated not in {"pass", "fail"}:
                raise ValueError(f"{item['id']} disagreement requires adjudication")
            final = str(adjudicated)
        else:
            if adjudicated not in {None, review_labels[0]}:
                raise ValueError(f"{item['id']} adjudication contradicts reviewers")
            final = review_labels[0]
        raw_scores = judge.get("scores")
        if not isinstance(raw_scores, Mapping) or set(raw_scores) != set(DIMENSIONS):
            raise ValueError(f"{item['id']} judge scores do not match the contract")
        scores = []
        for dimension in DIMENSIONS:
            score = raw_scores[dimension]
            if (
                not isinstance(score, int | float)
                or isinstance(score, bool)
                or not 0 <= float(score) <= 1
            ):
                raise ValueError(f"{item['id']} judge score is invalid")
            scores.append(float(score))
        assessment = str(judge.get("overall_assessment") or "").strip()
        rationale = str(judge.get("rationale") or "").strip()
        uncertainty = judge.get("uncertainty")
        if (
            not assessment
            or len(assessment) > 500
            or not rationale
            or len(rationale) > 500
            or not isinstance(uncertainty, int | float)
            or isinstance(uncertainty, bool)
            or not 0 <= float(uncertainty) <= 1
        ):
            raise ValueError(f"{item['id']} judge payload is incomplete")
        judge_label = (
            "pass" if all(score >= PASS_THRESHOLD for score in scores) else "fail"
        )
        final_labels.append(final)
        judge_labels.append(judge_label)
        evaluated_splits.append(str(item["split"]))
        critical = bool(
            (item.get("authored_reference") or {}).get(
                "critical_unsupported_completeness"
            )
        )
        if critical and final == "fail" and judge_label == "pass":
            critical_false_passes += 1

    true_positives = sum(
        actual == predicted == "pass"
        for actual, predicted in zip(final_labels, judge_labels, strict=True)
    )
    positives = final_labels.count("pass")
    true_negatives = sum(
        actual == predicted == "fail"
        for actual, predicted in zip(final_labels, judge_labels, strict=True)
    )
    negatives = final_labels.count("fail")
    true_positive_rate = true_positives / positives if positives else 0.0
    true_negative_rate = true_negatives / negatives if negatives else 0.0
    split_rates = {}
    for split in ("calibration", "holdout"):
        labels = [
            (actual, predicted)
            for actual, predicted, row_split in zip(
                final_labels,
                judge_labels,
                evaluated_splits,
                strict=True,
            )
            if row_split == split
        ]
        split_positives = sum(actual == "pass" for actual, _ in labels)
        split_negatives = sum(actual == "fail" for actual, _ in labels)
        split_rates[split] = {
            "true_positive_rate": (
                sum(actual == predicted == "pass" for actual, predicted in labels)
                / split_positives
                if split_positives
                else 0.0
            ),
            "true_negative_rate": (
                sum(actual == predicted == "fail" for actual, predicted in labels)
                / split_negatives
                if split_negatives
                else 0.0
            ),
        }
    passed = (
        completed
        and true_positive_rate >= MIN_RATE
        and true_negative_rate >= MIN_RATE
        and all(
            rates["true_positive_rate"] >= MIN_RATE
            and rates["true_negative_rate"] >= MIN_RATE
            for rates in split_rates.values()
        )
        and critical_false_passes == 0
    )
    return {
        "schema_version": 1,
        "review_status": "adjudicated" if completed else "pending_human_review",
        "reviewers_per_example": 2 if completed else 0,
        "disagreements_adjudicated": completed,
        "judge_profile": JUDGE_PROFILE,
        "rubric_digest": stable_digest(
            {
                "schema_version": 1,
                "judge_id": JUDGE_ID,
                "profile": JUDGE_PROFILE,
                "rubric": RUBRIC,
                "dimensions": list(DIMENSIONS),
                "evidence": list(EVIDENCE),
            }
        ),
        "cases_digest": stable_digest([dict(item) for item in rows]),
        "score_threshold": PASS_THRESHOLD,
        "examples": len(rows),
        "calibration_examples": sum(
            item.get("split") == "calibration" for item in rows
        ),
        "holdout_examples": sum(item.get("split") == "holdout" for item in rows),
        "true_positive_rate": round(true_positive_rate, 6),
        "true_negative_rate": round(true_negative_rate, 6),
        "calibration_true_positive_rate": round(
            split_rates["calibration"]["true_positive_rate"],
            6,
        ),
        "calibration_true_negative_rate": round(
            split_rates["calibration"]["true_negative_rate"],
            6,
        ),
        "holdout_true_positive_rate": round(
            split_rates["holdout"]["true_positive_rate"],
            6,
        ),
        "holdout_true_negative_rate": round(
            split_rates["holdout"]["true_negative_rate"],
            6,
        ),
        "critical_false_passes": critical_false_passes,
        "passed": passed,
        "distinct_reviewers": len(reviewer_ids),
        "disagreements": disagreement_count,
        "note": (
            "Calibration passed the locked thresholds."
            if passed
            else "Paid execution stays blocked until every example has two "
            "independent reviews, disagreements are adjudicated, and the judge "
            "meets both 0.85 rates in calibration and holdout with zero critical "
            "false passes."
        ),
    }


def _validate_template_splits(
    rows: Sequence[Mapping[str, Any]],
    reference: Sequence[str],
) -> None:
    splits = [str(item.get("split") or "") for item in rows]
    if splits.count("calibration") != 36 or splits.count("holdout") != 12:
        raise ValueError("calibration must preserve a 36/12 template-family split")
    split_labels = {
        split: [
            reference[index] for index, value in enumerate(splits) if value == split
        ]
        for split in {"calibration", "holdout"}
    }
    if any(
        labels.count("pass") != labels.count("fail") for labels in split_labels.values()
    ):
        raise ValueError("each calibration split must be label-balanced")
    scenario_splits: dict[str, set[str]] = {}
    for item in rows:
        scenario_splits.setdefault(str(item.get("scenario_id") or ""), set()).add(
            str(item.get("split") or "")
        )
    if any(len(values) != 1 for values in scenario_splits.values()):
        raise ValueError("scenario template families may not cross splits")


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
        for item in calibration_template()
    )
    path.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()
    if args.write_template:
        write_template(args.cases)
    report = validate_cases(read_cases(args.cases))
    atomic_write_json(args.report, report)
    print(
        json.dumps(
            {
                "cases": report["examples"],
                "review_status": report["review_status"],
                "passed": report["passed"],
                "report": args.report.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
