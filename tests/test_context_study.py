from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fugue.bench.cli import _preparation_targets, _selected_preset
from fugue.bench.context import list_context_systems
from fugue.bench.library import get_experiment
from fugue.bench.operator import ExperimentRequest, OperatorService

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
    runtime = REPO_ROOT / ".fugue" / "runtime"
    before = sorted(path.as_posix() for path in runtime.rglob("*") if path.is_file())

    preview = OperatorService(REPO_ROOT).preview(
        ExperimentRequest(
            experiment_id="repo-memory-impact",
            preset="smoke",
            model="openai/gpt-5",
            run_name="context-smoke-preview",
        )
    )

    assert preview.cells == 89
    assert preview.applicable_cells <= preview.cells
    assert preview.estimated_trials <= 95
    assert len(preview.variants) == 14
    assert preview.harnesses == ("claude-code", "codex", "hermes", "openclaw")
    assert preview.workloads == ("coding", "continuity", "qa", "retrieval")
    assert len(preview.systems) == 14
    after = sorted(path.as_posix() for path in runtime.rglob("*") if path.is_file())
    assert after == before
    serialized = json.dumps(asdict(preview))
    assert "OPENAI_API_KEY" not in serialized
    assert "WANDB_API_KEY" not in serialized
