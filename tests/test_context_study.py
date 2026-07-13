from __future__ import annotations

import json
from pathlib import Path

from fugue.bench.cli import _preparation_targets, _selected_preset
from fugue.bench.context import list_context_systems
from fugue.bench.library import get_experiment
from fugue.web import _render_payload

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_memory_study_has_truthful_capabilities_and_preset_sizes(
    monkeypatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    experiment = get_experiment("repo-memory-impact", REPO_ROOT)
    systems = {item.id: item for item in list_context_systems(REPO_ROOT)}

    assert experiment.default_preset == "smoke"
    smoke = next(item for item in experiment.presets if item.id == "smoke")
    full = next(item for item in experiment.presets if item.id == "full")
    assert smoke.workload_overrides == {
        "retrieval": {"n_tasks": 3},
        "qa": {"n_tasks": 1},
        "coding": {"n_tasks": 1},
        "continuity": {"n_tasks": 1},
    }
    assert full.workload_overrides["retrieval"] == {"n_tasks": 225}
    assert full.workload_overrides["qa"] == {"n_tasks": 24}
    assert systems["gitnexus"].requires_license_approval is True
    assert systems["project-rag"].version.startswith("git@d5abf98")

    for spec in systems.values():
        if spec.provider.endswith(":CommandContextProvider") and "retrieve" in spec.capabilities:
            assert (spec.config.get("retrieve") or {}).get("command"), spec.id

    targets = _preparation_targets(
        experiment=experiment,
        workloads=experiment.workloads,
        preset=_selected_preset(experiment, "smoke"),
        requested_systems=None,
        manifest_override=None,
        repo_root=REPO_ROOT,
    )
    assert len(targets) == 35


def test_repo_memory_smoke_preview_is_exact_and_side_effect_free(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    runtime = REPO_ROOT / ".fugue" / "runtime" / "web-preview"
    before = sorted(path.as_posix() for path in runtime.rglob("*") if path.is_file())

    payload = _render_payload(
        {
            "experiment_id": "repo-memory-impact",
            "preset": "smoke",
            "model": "openai/gpt-5",
            "run_name": "context-smoke-preview",
        },
        write=False,
    )

    summary = payload["summary"]
    assert {key: summary[key] for key in (
        "cells",
        "task_count",
        "trials_per_cell",
        "variants",
        "harnesses",
        "workloads",
        "systems",
        "cache_ready_cells",
    )} == {
        "cells": 89,
        "task_count": 6,
        "trials_per_cell": 1,
        "variants": 14,
        "harnesses": 4,
        "workloads": 4,
        "systems": 14,
        "cache_ready_cells": 0,
    }
    assert summary["applicable_cells"] + summary["skipped_cells"] == summary["cells"]
    assert summary["estimated_trials"] <= 95
    assert {
        item["workload_id"]: item["task_count"]
        for item in summary["workload_breakdown"]
    } == {"retrieval": 3, "qa": 1, "coding": 1, "continuity": 1}
    after = sorted(path.as_posix() for path in runtime.rglob("*") if path.is_file())
    assert after == before
    serialized = json.dumps(payload)
    assert "OPENAI_API_KEY" not in serialized
    assert "WANDB_API_KEY" not in serialized
