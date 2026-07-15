from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from fugue.bench.context import RetrievalHit, RetrievalQuery


@dataclass(frozen=True)
class SelectionPolicy:
    metric: str = "pass_rate"
    confidence: float = 0.95
    noninferiority_margin: float = 0.05
    require_complete_grid: bool = True
    bootstrap_samples: int = 2_000
    tie_breakers: tuple[str, ...] = (
        "cost_per_success",
        "median_wall_time_sec",
        "recoverable_error_rate",
    )
    incumbent_candidate_id: str | None = None
    minimum_pass_rate_improvement: float = 0.05
    minimum_cost_improvement: float = 0.15
    minimum_latency_improvement: float = 0.15


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    eligible: bool
    reasons: tuple[str, ...]
    trials: int
    examples: int
    pass_rate: float | None
    cost_per_success: float | None
    median_wall_time_sec: float | None
    recoverable_error_rate: float | None
    delta_to_best: float | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    competitive: bool = False


@dataclass(frozen=True)
class CandidateSelection:
    policy: SelectionPolicy
    best_candidate_id: str | None
    selected_candidate_id: str | None
    incumbent_candidate_id: str | None
    decision: str
    reason: str
    candidates: tuple[CandidateScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_candidate_configuration(
    rows: Iterable[dict[str, Any]],
    policy: SelectionPolicy,
    *,
    seed: str,
) -> CandidateSelection:
    """Select a candidate from normalized trials without model-authored arithmetic."""
    values = [dict(row) for row in rows]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in values:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id:
            grouped.setdefault(candidate_id, []).append(row)
    expected = {
        _trial_coordinate(row)
        for rows_for_candidate in grouped.values()
        for row in rows_for_candidate
        if _trial_coordinate(row) is not None
    }
    candidate_grids = {
        candidate_id: {
            coordinate
            for row in candidate_rows
            if (coordinate := _trial_coordinate(row)) is not None
        }
        for candidate_id, candidate_rows in grouped.items()
    }
    grids_match = len({frozenset(value) for value in candidate_grids.values()}) <= 1
    scores: list[CandidateScore] = []
    eligible_rows: dict[str, list[dict[str, Any]]] = {}
    for candidate_id, candidate_rows in sorted(grouped.items()):
        reasons: list[str] = []
        coordinates = [_trial_coordinate(row) for row in candidate_rows]
        present = {item for item in coordinates if item is not None}
        if any(item is None for item in coordinates):
            reasons.append("missing comparison_example_id or trial_index")
        if len(present) != len(coordinates):
            reasons.append("duplicate candidate/example/trial row")
        if policy.require_complete_grid and present != expected:
            reasons.append("incomplete comparison grid")
        if policy.require_complete_grid and not grids_match:
            reasons.append("candidates do not share one comparison grid")
        if any(row.get("pass") is None for row in candidate_rows):
            reasons.append("missing deterministic outcome")
        eligible = not reasons
        if eligible:
            eligible_rows[candidate_id] = candidate_rows
        successes = sum(row.get("pass") is True for row in candidate_rows)
        costs = [row.get("cost_usd") for row in candidate_rows]
        latencies = [row.get("wall_time_sec") for row in candidate_rows]
        error_rows = [
            row
            for row in candidate_rows
            if float(
                row.get("weave_tool_error_count")
                or row.get("recoverable_tool_errors")
                or 0
            )
            > 0
        ]
        scores.append(
            CandidateScore(
                candidate_id=candidate_id,
                eligible=eligible,
                reasons=tuple(reasons),
                trials=len(candidate_rows),
                examples=len({item[0] for item in present}),
                pass_rate=(successes / len(candidate_rows) if candidate_rows else None),
                cost_per_success=(
                    sum(float(value) for value in costs) / successes
                    if successes and costs and all(value is not None for value in costs)
                    else None
                ),
                median_wall_time_sec=(
                    median(float(value) for value in latencies)
                    if latencies and all(value is not None for value in latencies)
                    else None
                ),
                recoverable_error_rate=(
                    len(error_rows) / len(candidate_rows) if candidate_rows else None
                ),
            )
        )
    eligible_scores = [score for score in scores if score.eligible]
    if not eligible_scores:
        return CandidateSelection(
            policy=policy,
            best_candidate_id=None,
            selected_candidate_id=None,
            incumbent_candidate_id=policy.incumbent_candidate_id,
            decision="blocked",
            reason="no candidate has a complete, uniquely scored comparison grid",
            candidates=tuple(scores),
        )
    best = sorted(
        eligible_scores,
        key=lambda item: (-(item.pass_rate or 0.0), item.candidate_id),
    )[0]
    enriched: list[CandidateScore] = []
    for score in scores:
        if not score.eligible:
            enriched.append(score)
            continue
        low, high = _paired_delta_interval(
            eligible_rows[score.candidate_id],
            eligible_rows[best.candidate_id],
            confidence=policy.confidence,
            samples=policy.bootstrap_samples,
            seed=f"{seed}:{score.candidate_id}:{best.candidate_id}",
        )
        enriched.append(
            CandidateScore(
                **{
                    **asdict(score),
                    "delta_to_best": (score.pass_rate or 0.0)
                    - (best.pass_rate or 0.0),
                    "confidence_low": low,
                    "confidence_high": high,
                    "competitive": low >= -policy.noninferiority_margin,
                }
            )
        )
    competitive = [score for score in enriched if score.competitive]
    selected = sorted(
        competitive,
        key=lambda item: (
            item.cost_per_success is None,
            item.cost_per_success if item.cost_per_success is not None else math.inf,
            item.median_wall_time_sec is None,
            item.median_wall_time_sec
            if item.median_wall_time_sec is not None
            else math.inf,
            item.recoverable_error_rate
            if item.recoverable_error_rate is not None
            else math.inf,
            item.candidate_id,
        ),
    )[0]
    decision, reason = _promotion_decision(
        selected,
        enriched,
        eligible_rows,
        policy,
        seed=seed,
    )
    return CandidateSelection(
        policy=policy,
        best_candidate_id=best.candidate_id,
        selected_candidate_id=selected.candidate_id,
        incumbent_candidate_id=policy.incumbent_candidate_id,
        decision=decision,
        reason=reason,
        candidates=tuple(enriched),
    )


def _promotion_decision(
    selected: CandidateScore,
    scores: list[CandidateScore],
    rows: dict[str, list[dict[str, Any]]],
    policy: SelectionPolicy,
    *,
    seed: str,
) -> tuple[str, str]:
    incumbent_id = policy.incumbent_candidate_id
    if not incumbent_id:
        return "recommend", "no incumbent was supplied; propose the selected candidate for review"
    incumbent = next(
        (item for item in scores if item.candidate_id == incumbent_id and item.eligible),
        None,
    )
    if incumbent is None:
        return "blocked", "the incumbent is absent or ineligible in this comparison scope"
    if incumbent.candidate_id == selected.candidate_id:
        return "no_promotion", "the incumbent remains the selected candidate"
    low, _ = _paired_delta_interval(
        rows[selected.candidate_id],
        rows[incumbent.candidate_id],
        confidence=policy.confidence,
        samples=policy.bootstrap_samples,
        seed=f"{seed}:{selected.candidate_id}:{incumbent.candidate_id}:incumbent",
    )
    if low < -policy.noninferiority_margin:
        return "no_promotion", "the selected candidate is not quality non-inferior to the incumbent"
    pass_improvement = (selected.pass_rate or 0.0) - (incumbent.pass_rate or 0.0)
    cost_improvement = _relative_improvement(
        incumbent.cost_per_success, selected.cost_per_success
    )
    latency_improvement = _relative_improvement(
        incumbent.median_wall_time_sec, selected.median_wall_time_sec
    )
    if (
        pass_improvement >= policy.minimum_pass_rate_improvement
        or cost_improvement >= policy.minimum_cost_improvement
        or latency_improvement >= policy.minimum_latency_improvement
    ):
        return "promote", "quality is non-inferior and an explicit improvement threshold was met"
    return "no_promotion", "quality is non-inferior, but no promotion improvement threshold was met"


def _paired_delta_interval(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    confidence: float,
    samples: int,
    seed: str,
) -> tuple[float, float]:
    candidate = _example_outcomes(candidate_rows)
    baseline = _example_outcomes(baseline_rows)
    examples = sorted(set(candidate) & set(baseline))
    if not examples:
        return 0.0, 0.0
    if candidate == baseline:
        return 0.0, 0.0
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        selected = [rng.choice(examples) for _ in examples]
        deltas.append(
            sum(candidate[item] - baseline[item] for item in selected) / len(selected)
        )
    alpha = (1.0 - confidence) / 2.0
    return _percentile(sorted(deltas), alpha), _percentile(sorted(deltas), 1.0 - alpha)


def _example_outcomes(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        example_id = str(row["comparison_example_id"])
        grouped.setdefault(example_id, []).append(float(row.get("pass") is True))
    return {
        example_id: sum(values) / len(values)
        for example_id, values in grouped.items()
    }


def _trial_coordinate(row: dict[str, Any]) -> tuple[str, int] | None:
    example_id = str(row.get("comparison_example_id") or "").strip()
    try:
        trial_index = int(row.get("trial_index"))
    except (TypeError, ValueError):
        return None
    return (example_id, trial_index) if example_id and trial_index > 0 else None


def _relative_improvement(before: float | None, after: float | None) -> float:
    if before is None or after is None or before <= 0:
        return -math.inf
    return (before - after) / before


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
