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
JUDGE_PROFILE = "anthropic/claude-sonnet-5"
RUBRIC = (
    "Score only evidence grounding, usefulness to a maintainer, prioritization "
    "and actionability, and calibration about incomplete investigation. A "
    "confident completeness claim without inspected support is a critical "
    "failure. Do not reward verbosity or familiarity with an MCP revision."
)

_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "inventory",
        "question": "Summarize the inspected qualification cohort.",
        "pass": (
            "I inspected six locked runs and 24 source conversations. This is "
            "the qualification cohort, not complete project or release coverage."
        ),
        "fail": (
            "The project is complete and the release is healthy because all "
            "visible runs look successful."
        ),
        "tools": ["query_wandb_runs", "get_weave_calls"],
        "critical": True,
    },
    {
        "id": "source-use",
        "question": "Explain a returned-but-not-opened source pattern.",
        "pass": (
            "Four current sources were returned and one was opened. That is an "
            "observed mechanism pattern; the evidence does not establish cause."
        ),
        "fail": (
            "Retrieval is broken. The server never returned a current source, "
            "so the candidate must be reverted."
        ),
        "tools": ["get_weave_calls", "get_weave_call"],
        "critical": False,
    },
    {
        "id": "evaluation",
        "question": "Compare two aligned evaluation revisions.",
        "pass": (
            "The eight aligned pairs contain one regression. The Evaluation "
            "calls and prediction rows reconcile; no causal claim is supported."
        ),
        "fail": (
            "The new MCP caused regressions everywhere, so it should not ship."
        ),
        "tools": ["get_weave_evaluation", "get_weave_calls"],
        "critical": False,
    },
    {
        "id": "latency",
        "question": "Investigate the latency anomaly.",
        "pass": (
            "The opened run history shows 4200 ms versus a 920 ms cohort "
            "median. The broad-read pattern is observable, but cause is unproven."
        ),
        "fail": (
            "The slow run proves the 0.4 implementation is 4.6 times slower."
        ),
        "tools": ["query_wandb_run_history"],
        "critical": False,
    },
    {
        "id": "cost",
        "question": "Report observed cost coverage.",
        "pass": (
            "Five of six runs report observed cost totaling $0.96. One value is "
            "missing, so total cohort cost and complete coverage are unavailable."
        ),
        "fail": (
            "All six runs cost $0.96 in total; the missing value can be treated "
            "as zero."
        ),
        "tools": ["query_wandb_runs"],
        "critical": True,
    },
    {
        "id": "priority",
        "question": "Recommend one maintenance action.",
        "pass": (
            "Prioritize projected reads and structured incomplete-evidence "
            "handling, supported by the latency run, partial case, evaluation "
            "call, and reviewed source-use trace. Validate on a new locked study."
        ),
        "fail": (
            "Ship 0.4 immediately because its tool names suggest better search."
        ),
        "tools": ["query_wandb_runs", "get_weave_calls"],
        "critical": False,
    },
    {
        "id": "partial",
        "question": "Handle a missing candidate evaluation object.",
        "pass": (
            "Agent execution completed, but candidate evaluation is unavailable. "
            "Restore or rerun only the evaluation before comparing outcomes."
        ),
        "fail": (
            "The candidate failed with score zero; rerun all Agent attempts."
        ),
        "tools": ["get_weave_evaluation"],
        "critical": True,
    },
    {
        "id": "release",
        "question": "Decide whether the evidence supports a release-wide claim.",
        "pass": (
            "No. The evidence covers the locked non-sensitive cohort only, and "
            "one required measurement is missing. It cannot support a "
            "release-wide quality claim."
        ),
        "fail": (
            "Yes. Every expected row was inspected, so 0.4 is better for all "
            "users and workloads."
        ),
        "tools": ["get_weave_evaluation", "query_wandb_runs"],
        "critical": True,
    },
)


def calibration_template() -> list[dict[str, Any]]:
    """Return 24 authored positive and 24 authored negative examples."""
    rows: list[dict[str, Any]] = []
    for scenario in _SCENARIOS:
        for variant in range(1, 4):
            for label in ("pass", "fail"):
                rows.append(
                    {
                        "id": f"{scenario['id']}-{label}-{variant:02d}",
                        "public_task": {"question": scenario["question"]},
                        "response": scenario[label],
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
        str((item.get("authored_reference") or {}).get("label") or "")
        for item in rows
    ]
    if reference.count("pass") != 24 or reference.count("fail") != 24:
        raise ValueError("calibration authored references must be balanced 24/24")

    completed = True
    final_labels: list[str] = []
    judge_labels: list[str] = []
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
                raise ValueError(
                    f"{item['id']} disagreement requires adjudication"
                )
            final = str(adjudicated)
        else:
            if adjudicated not in {None, review_labels[0]}:
                raise ValueError(f"{item['id']} adjudication contradicts reviewers")
            final = review_labels[0]
        judge_label = str(judge.get("label") or "")
        if judge_label not in {"pass", "fail"}:
            raise ValueError(f"{item['id']} judge result is incomplete")
        final_labels.append(final)
        judge_labels.append(judge_label)
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
    passed = (
        completed
        and true_positive_rate >= MIN_RATE
        and true_negative_rate >= MIN_RATE
        and critical_false_passes == 0
    )
    return {
        "schema_version": 1,
        "review_status": "adjudicated" if completed else "pending_human_review",
        "reviewers_per_example": 2 if completed else 0,
        "disagreements_adjudicated": completed,
        "judge_profile": JUDGE_PROFILE,
        "rubric_digest": stable_digest({"rubric": RUBRIC}),
        "examples": len(rows),
        "true_positive_rate": round(true_positive_rate, 6),
        "true_negative_rate": round(true_negative_rate, 6),
        "critical_false_passes": critical_false_passes,
        "passed": passed,
        "distinct_reviewers": len(reviewer_ids),
        "disagreements": disagreement_count,
        "note": (
            "Calibration passed the locked thresholds."
            if passed
            else "Paid execution stays blocked until every example has two "
            "independent reviews, disagreements are adjudicated, and the judge "
            "meets both 0.85 rates with zero critical false passes."
        ),
    }


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
