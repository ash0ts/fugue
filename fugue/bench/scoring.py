from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import median
from typing import Any

from fugue.bench.context import RetrievalHit, RetrievalQuery


def score_retrieval(
    query: RetrievalQuery, hits: list[RetrievalHit]
) -> dict[str, float | int | None]:
    expected = {_normalize_path(path) for path in query.expected_paths}
    raw_ranked = [_normalize_path(hit.path) for hit in hits]
    ranked = list(dict.fromkeys(path for path in raw_ranked if path))
    if not expected:
        return {
            "mrr": None,
            "ndcg_at_10": None,
            "first_relevant_rank": None,
            "empty": int(not hits),
            **{
                f"{metric}_at_{k}": None
                for k in (1, 5, 10, 20)
                for metric in ("recall", "precision")
            },
        }
    relevant_ranks = [
        rank for rank, path in enumerate(ranked, start=1) if path in expected
    ]
    first_rank = min(relevant_ranks) if relevant_ranks else None
    metrics: dict[str, float | int | None] = {
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg_at_10": _ndcg(ranked[:10], expected),
        "first_relevant_rank": first_rank,
        "empty": int(not hits),
        "raw_result_count": len(hits),
        "unique_result_count": len(ranked),
    }
    for k in (1, 5, 10, 20):
        selected = ranked[:k]
        found = sum(1 for path in selected if path in expected)
        metrics[f"recall_at_{k}"] = found / len(expected) if expected else None
        metrics[f"precision_at_{k}"] = found / len(selected) if selected else 0.0
    return metrics


def score_evidence_paths(
    expected_paths: Iterable[str], observed_paths: Iterable[str]
) -> dict[str, float | None]:
    expected = {_normalize_path(path) for path in expected_paths}
    observed = {_normalize_path(path) for path in observed_paths}
    if not expected:
        return {"evidence_recall": None, "evidence_precision": None}
    overlap = expected & observed
    return {
        "evidence_recall": len(overlap) / len(expected),
        "evidence_precision": len(overlap) / len(observed) if observed else 0.0,
    }


def score_fact_recall(
    expected_facts: Iterable[str], observed_text: Iterable[str]
) -> dict[str, float | None]:
    expected = [item.strip().lower() for item in expected_facts if item.strip()]
    corpus = "\n".join(observed_text).lower()
    if not expected:
        return {"fact_recall": None}
    recalled = sum(1 for fact in expected if fact in corpus)
    return {"fact_recall": recalled / len(expected)}


def latency_summary(values: Iterable[float]) -> dict[str, float | None]:
    selected = sorted(float(value) for value in values)
    if not selected:
        return {"p50_ms": None, "p95_ms": None}
    return {
        "p50_ms": median(selected),
        "p95_ms": _percentile(selected, 0.95),
    }


def summarize_metric_rows(
    rows: list[dict[str, Any]], group_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(name) or "unknown") for name in group_keys)
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        passed = sum(1 for row in group if row.get("pass") is True)
        scored = [row for row in group if row.get("pass") is not None]
        summary = {
            name: value for name, value in zip(group_keys, key, strict=True)
        }
        summary.update(
            {
                "total": len(group),
                "scored": len(scored),
                "passed": passed,
                "pass_rate": passed / len(scored) if scored else None,
                "cost_usd": sum(float(row.get("cost_usd") or 0) for row in group),
                "tokens": sum(
                    int(row.get("n_input_tokens") or 0)
                    + int(row.get("n_cache_tokens") or 0)
                    + int(row.get("n_output_tokens") or 0)
                    for row in group
                ),
                "wall_time_sec": sum(float(row.get("wall_time_sec") or 0) for row in group),
                "failures": sum(
                    1
                    for row in group
                    if row.get("pass") is False or row.get("exception_class")
                ),
            }
        )
        for metric in (
            "mrr",
            "ndcg_at_10",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "recall_at_20",
            "precision_at_1",
            "precision_at_5",
            "precision_at_10",
            "precision_at_20",
            "evidence_recall",
            "citation_correctness",
            "query_latency_ms",
            "build_latency_ms",
            "time_to_first_context_ms",
            "fact_recall",
            "judge_correctness",
            "judge_completeness",
            "judge_groundedness",
            "judge_overall",
        ):
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            summary[metric] = sum(values) / len(values) if values else None
        query_latency = latency_summary(
            float(row["query_latency_ms"])
            for row in group
            if row.get("query_latency_ms") is not None
        )
        summary["query_latency_p50_ms"] = query_latency["p50_ms"]
        summary["query_latency_p95_ms"] = query_latency["p95_ms"]
        retrieval_rows = [row for row in group if row.get("record_type") == "retrieval"]
        empty_rows = [row for row in retrieval_rows if row.get("empty") is not None]
        summary["empty_rate"] = (
            sum(float(row.get("empty") or 0) for row in empty_rows) / len(empty_rows)
            if empty_rows
            else None
        )
        summary["error_rate"] = (
            sum(1 for row in retrieval_rows if row.get("exception_class"))
            / len(retrieval_rows)
            if retrieval_rows
            else None
        )
        summary["outcome_quality"] = (
            summary["pass_rate"]
            if summary["pass_rate"] is not None
            else summary["judge_overall"]
        )
        summaries.append(summary)
    return summaries


def pareto_frontier(
    rows: list[dict[str, Any]], *, quality: str, cost: str
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get(quality) is not None and row.get(cost) is not None
    ]
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        dominated = any(
            other is not candidate
            and float(other[quality]) >= float(candidate[quality])
            and float(other[cost]) <= float(candidate[cost])
            and (
                float(other[quality]) > float(candidate[quality])
                or float(other[cost]) < float(candidate[cost])
            )
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def _normalize_path(path: str) -> str:
    return str(path).strip().removeprefix("./").replace("\\", "/")


def _ndcg(ranked: list[str], expected: set[str]) -> float | None:
    if not expected:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, path in enumerate(ranked, start=1)
        if path in expected
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(expected), len(ranked)) + 1)
    )
    return dcg / ideal if ideal else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight
