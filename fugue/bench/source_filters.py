from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def is_non_task_evidence_wandb_run(run: Any) -> bool:
    """Return whether a W&B Run is control-plane metadata, never task evidence."""

    if str(getattr(run, "job_type", "") or "") in {
        "scientific-report",
        "task-source-manifest",
    }:
        return True
    config = getattr(run, "config", None)
    if not isinstance(config, Mapping):
        return False
    fugue = config.get("fugue")
    return isinstance(fugue, Mapping) and (
        fugue.get("run_kind") in {"report_only", "task_source_manifest"}
        or fugue.get("excluded_from_task_inputs") is True
        or fugue.get("excluded_from_evaluation_counts") is True
    )


def task_evidence_wandb_runs(runs: Iterable[Any]) -> list[Any]:
    """Filter control-plane Runs at the shared task-evidence boundary."""

    return [run for run in runs if not is_non_task_evidence_wandb_run(run)]
