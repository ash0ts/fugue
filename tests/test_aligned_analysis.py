from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.ai import (
    AnalysisResult,
    ExperimentAnalyst,
    _materialize_aligned_analysis,
    _scoped_result_project,
    _snapshot,
    _write_analysis,
    get_analysis,
)
from fugue.bench.analysis_contracts import (
    AlignedAnalysisV1,
    AlignedArmV1,
    AlignedAttemptSetV1,
    AlignedContrastV1,
    AlignedDimensionV1,
    TaskStratifiedSummaryV1,
    aligned_analysis_from_dict,
)
from fugue.bench.candidates import attempt_id, attempt_identity, stable_digest
from fugue.bench.catalog import CatalogStatus
from fugue.research.experiment_views import _aligned_comparisons

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PROJECT = "wandb/fugue-harness-experiments-v1"
MEMORY_PROJECT = "wandb/fugue-memory-experiments-v1"


def _row(
    *,
    task: str,
    harness: str,
    variant: str,
    passed: bool,
    cost: float,
    returned: bool = False,
    opened: bool = False,
    used: bool = False,
    project: str,
) -> dict:
    candidate = stable_digest({"task": task, "harness": harness, "variant": variant})
    runtime = stable_digest({"runtime": harness, "variant": variant})
    identity = attempt_identity(
        task_id=task,
        arm=variant,
        harness=harness,
        attempt=1,
        candidate=candidate,
        runtime=runtime,
    )
    canonical_id = attempt_id(**identity)
    return {
        "row_id": stable_digest({"row": canonical_id}),
        "record_type": "trial",
        "run_id": "run-aligned",
        "run_key": f"run-aligned:{task}:{harness}:{variant}",
        "experiment_id": "real-study",
        "task_id": task,
        "task_name": task,
        "harness": harness,
        "variant_id": variant,
        "trial_index": 1,
        "candidate_id": candidate,
        "execution_fingerprint": runtime,
        "attempt_identity": identity,
        "attempt_id": canonical_id,
        "pass": passed,
        "agent_runtime_completed": True,
        "runtime_outcome": "completed",
        "wall_time_sec": 2.0 + cost,
        "tool_calls": 2,
        "cost_usd": cost,
        "relevant_retrieval_returned": returned,
        "relevant_retrieval_opened": opened,
        "relevant_retrieval_used": used,
        "source_evidence_project": "wandb/fugue-source-v1",
        "result_evidence_project": project,
        "trace_project": project,
        "trace_receipt": {"project_slug": project},
        "source": "local",
        "model": "locked/model",
    }


def _harness_rows() -> list[dict]:
    rows: list[dict] = []
    for task in ("task-a", "task-b"):
        for harness in ("claude-code", "hermes", "openclaw", "codex"):
            rows.append(
                _row(
                    task=task,
                    harness=harness,
                    variant="baseline",
                    passed=(task == "task-b" or harness == "hermes"),
                    cost={
                        "claude-code": 0.4,
                        "hermes": 0.3,
                        "openclaw": 0.2,
                        "codex": 0.5,
                    }[harness],
                    project=HARNESS_PROJECT,
                )
            )
    return rows


def _memory_rows() -> list[dict]:
    rows: list[dict] = []
    for task in ("task-a", "task-b"):
        for variant in ("baseline", "rag-dense", "policy-only", "combined"):
            rows.append(
                _row(
                    task=task,
                    harness="claude-code",
                    variant=variant,
                    passed=variant == "combined",
                    cost=0.2,
                    returned=variant in {"rag-dense", "combined"},
                    opened=variant == "combined",
                    used=variant == "combined",
                    project=MEMORY_PROJECT,
                )
            )
    return rows


def test_harness_analysis_materializes_arbitrary_arm_task_sets() -> None:
    declaration = get_analysis(
        "real-harness-task-stratified", REPO_ROOT
    ).aligned_analysis
    assert declaration is not None

    result = _materialize_aligned_analysis(_harness_rows(), declaration)

    assert result is not None
    assert result.declaration_digest == declaration.declaration_digest
    assert result.alignment_coordinates == ("task_name", "trial_index")
    assert len(result.aligned_attempts) == 2
    assert all(
        set(item.attempt_ids_by_arm) == {"claude-code", "hermes", "openclaw", "codex"}
        for item in result.aligned_attempts
    )
    hermes_success = next(
        item
        for item in result.contrast_results
        if item.contrast_id == "hermes-vs-claude-code"
        and item.dimension_id == "task-success"
    )
    assert hermes_success.classification == "improved"
    assert hermes_success.aligned_sets == 2
    assert hermes_success.estimate == 0.5
    assert {
        item.task_id: item.classification for item in hermes_success.task_effects
    } == {"task-a": "improved", "task-b": "unchanged"}
    assert aligned_analysis_from_dict(result.to_dict()) == result


def test_memory_analysis_materializes_declared_two_by_two_interaction() -> None:
    declaration = get_analysis("real-memory-factorial", REPO_ROOT).aligned_analysis
    assert declaration is not None

    result = _materialize_aligned_analysis(_memory_rows(), declaration)

    assert result is not None
    [interaction] = result.interaction_results
    task_success = next(
        item for item in interaction.dimensions if item.dimension_id == "task-success"
    )
    assert interaction.cell_arms == {
        "00": "baseline",
        "10": "rag-dense",
        "01": "policy-only",
        "11": "combined",
    }
    assert task_success.status == "computed"
    assert task_success.aligned_sets == 2
    assert task_success.cell_means == {
        "00": 0.0,
        "10": 0.0,
        "01": 0.0,
        "11": 1.0,
    }
    assert task_success.difference_in_differences == 1.0
    assert {
        item.task_id: item.difference_in_differences
        for item in task_success.task_effects
    } == {"task-a": 1.0, "task-b": 1.0}
    assert result.interactions == declaration.interactions
    assert aligned_analysis_from_dict(result.to_dict()) == result


def test_aligned_analysis_rejects_interaction_result_drift_from_declaration() -> None:
    declaration = get_analysis("real-memory-factorial", REPO_ROOT).aligned_analysis
    assert declaration is not None
    result = _materialize_aligned_analysis(_memory_rows(), declaration)
    assert result is not None
    [interaction] = result.interaction_results

    with pytest.raises(
        ValueError,
        match="interaction result disagrees with its declaration",
    ):
        replace(
            result,
            interaction_results=(
                replace(
                    interaction,
                    cell_arms={
                        "00": "combined",
                        "10": "policy-only",
                        "01": "rag-dense",
                        "11": "baseline",
                    },
                ),
            ),
            analysis_digest="",
        )


@pytest.mark.parametrize("change", ["duplicate", "missing"])
def test_aligned_analysis_rejects_duplicate_or_missing_declared_contrast_results(
    change: str,
) -> None:
    declaration = get_analysis(
        "real-harness-task-stratified", REPO_ROOT
    ).aligned_analysis
    assert declaration is not None
    result = _materialize_aligned_analysis(_harness_rows(), declaration)
    assert result is not None
    contrast_results = list(result.contrast_results)
    if change == "duplicate":
        contrast_results.append(contrast_results[0])
        match = "contrast result coordinates must be unique"
    else:
        contrast_results.pop()
        match = "contrast results must exactly cover declared coordinates"

    with pytest.raises(ValueError, match=match):
        replace(
            result,
            contrast_results=tuple(contrast_results),
            analysis_digest="",
        )


@pytest.mark.parametrize("failure", ["missing", "duplicate", "ambiguous", "tamper"])
def test_governed_aligned_analysis_rejects_invalid_grids(failure: str) -> None:
    declaration = get_analysis("real-memory-factorial", REPO_ROOT).aligned_analysis
    assert declaration is not None
    rows = _memory_rows()
    if failure == "missing":
        rows = [
            row
            for row in rows
            if not (row["task_name"] == "task-a" and row["variant_id"] == "combined")
        ]
        match = "grid is incomplete"
    elif failure == "duplicate":
        rows.append(dict(rows[0]))
        match = "duplicate arm"
    elif failure == "ambiguous":
        rows[0]["harness"] = "rag-dense"
        match = "ambiguously"
    else:
        rows[0]["attempt_id"] = "0" * 64
        match = "attempt id disagrees"

    with pytest.raises(ValueError, match=match):
        _materialize_aligned_analysis(rows, declaration)


def test_existing_v1_contrast_specific_artifact_remains_readable() -> None:
    result = AlignedAnalysisV1(
        study_intent="historical-two-arm-contrast",
        reference_arm="baseline",
        arms=(
            AlignedArmV1(id="baseline", label="Baseline"),
            AlignedArmV1(id="candidate-a", label="Candidate A"),
            AlignedArmV1(id="candidate-b", label="Candidate B"),
        ),
        contrasts=(
            AlignedContrastV1(
                id="candidate-a-vs-baseline",
                reference_arm="baseline",
                treatment_arms=("candidate-a",),
                dimensions=(
                    AlignedDimensionV1(
                        id="task-success",
                        label="Task success",
                        role="outcome",
                        critical=True,
                    ),
                ),
            ),
        ),
        aligned_attempts=(
            AlignedAttemptSetV1(
                alignment_id="a" * 64,
                task_id="task-a",
                harness="claude-code",
                attempt=1,
                attempt_ids_by_arm={
                    "baseline": "b" * 64,
                    "candidate-a": "c" * 64,
                },
            ),
        ),
        task_summaries=(
            TaskStratifiedSummaryV1(
                task_id="task-a",
                validity="valid",
                pair_counts={"improved": 1},
            ),
        ),
    )

    assert aligned_analysis_from_dict(result.to_dict()) == result


def test_analysis_preview_persistence_and_research_use_materialized_result(
    tmp_path: Path,
) -> None:
    spec = replace(
        get_analysis("real-memory-factorial", REPO_ROOT),
        source="local",
    )
    rows = _memory_rows()
    operator = SimpleNamespace(repo_root=tmp_path, env={})
    analyst = ExperimentAnalyst(operator)
    analyst.catalog = SimpleNamespace(
        refresh=lambda: CatalogStatus(
            path="catalog.sqlite",
            experiments=1,
            records=len(rows),
            revision="catalog-v1",
        ),
        records=lambda **_: rows,
    )

    preview = analyst.prepare(spec)

    assert preview.aligned_analysis is not None
    assert preview.aligned_analysis.interaction_results
    result = AnalysisResult(
        spec=spec,
        scope=preview.scope,
        snapshot=preview.snapshot,
        evidence=preview.evidence,
        aggregates=preview.aggregates,
        selection=None,
        report="# Materialized analysis\n",
        report_dir=tmp_path / "reports/analysis/run-1",
        model="locked/model",
        provider="anthropic",
        session_id="session-1",
        input_tokens=1,
        output_tokens=1,
        aligned_analysis=preview.aligned_analysis,
    )
    _write_analysis(result, tmp_path)
    persisted = json.loads((result.report_dir / "analysis.json").read_text())
    assert (
        persisted["aligned_analysis"]["analysis_digest"]
        == preview.aligned_analysis.analysis_digest
    )

    projected = _aligned_comparisons(
        {
            "analysis_results": [
                {
                    "analysis_id": spec.id,
                    "aligned_analysis": preview.aligned_analysis.to_dict(),
                }
            ]
        }
    )
    assert any(
        item.get("comparison_id") == "retrieval-by-evidence-policy:task-success:did"
        and item.get("estimate") == 1.0
        for item in projected
    )


def test_snapshot_digest_binds_safe_row_contents_and_project_identity() -> None:
    rows = _harness_rows()
    original = _snapshot(rows, "catalog-v1")
    changed_outcome = [dict(item) for item in rows]
    changed_outcome[0]["pass"] = not changed_outcome[0]["pass"]
    changed_project = [dict(item) for item in rows]
    changed_project[0]["result_evidence_project"] = "wandb/other-project"
    changed_project[0]["trace_project"] = "wandb/other-project"
    changed_project[0]["trace_receipt"] = {"project_slug": "wandb/other-project"}

    assert _snapshot(changed_outcome, "catalog-v1").row_ids == original.row_ids
    assert _snapshot(changed_outcome, "catalog-v1").digest != original.digest
    assert _snapshot(changed_project, "catalog-v1").digest != original.digest
    assert _snapshot(rows, "unrelated-catalog-revision").digest == original.digest


def test_hybrid_project_must_come_from_one_exact_scoped_destination() -> None:
    rows = _harness_rows()

    assert _scoped_result_project(rows) == HARNESS_PROJECT
    rows[0]["trace_project"] = "wandb/wrong-project"
    with pytest.raises(ValueError, match="one exact result project per row"):
        _scoped_result_project(rows)


def test_governed_hybrid_analysis_never_falls_back_after_weave_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_analysis("real-memory-factorial", REPO_ROOT)
    rows = _memory_rows()
    operator = SimpleNamespace(repo_root=tmp_path, env={})
    analyst = ExperimentAnalyst(operator)
    analyst.catalog = SimpleNamespace(
        refresh=lambda: CatalogStatus(
            path="catalog.sqlite",
            experiments=1,
            records=len(rows),
            revision="catalog-v1",
        ),
        records=lambda **_: rows,
    )
    preview = analyst.prepare(spec)

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("trace backend unavailable")

    monkeypatch.setattr("fugue.bench.ai.fetch_weave_summaries", fail)
    with pytest.raises(
        ValueError,
        match=(
            "governed aligned analysis requires exact Weave enrichment from "
            "wandb/fugue-memory-experiments-v1"
        ),
    ):
        asyncio.run(analyst.execute(preview))
