from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fugue.bench.context import ContextRuntime, get_context_system, list_context_systems
from fugue.bench.library import get_experiment
from fugue.bench.operator import (
    ExperimentRequest,
    OperatorService,
    _preparation_targets,
    select_preset,
)
from fugue.bench.workloads import _add_runtime_correlation

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
    assert {
        "codegraph",
        "gitnexus",
        "project-rag",
        "semble",
        "latmd",
        "graphiti",
    }.isdisjoint(smoke.systems)
    assert all(
        systems[system_id].enabled_by_default is False
        for system_id in (
            "codegraph",
            "gitnexus",
            "project-rag",
            "semble",
            "latmd",
            "graphiti",
        )
    )
    retrieval = next(item for item in experiment.workloads if item.id == "retrieval")
    assert "none" in retrieval.systems

    for spec in systems.values():
        if (
            spec.provider.endswith(":CommandContextProvider")
            and "retrieve" in spec.capabilities
        ):
            assert (spec.config.get("retrieve") or {}).get("command"), spec.id

    targets = _preparation_targets(
        experiment=experiment,
        workloads=experiment.workloads,
        preset=select_preset(experiment, "smoke"),
        requested_systems=None,
        manifest_override=None,
        repo_root=REPO_ROOT,
    )
    assert targets
    assert {system_id for system_id, _ in targets} <= set(smoke.systems)


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

    experiment = get_experiment("repo-memory-impact", REPO_ROOT)
    preset = select_preset(experiment, "smoke")
    variants_by_system = {
        variant.context.system_id for variant in experiment.variants if variant.enabled
    }
    expected_cells = 0
    for workload in experiment.workloads:
        systems = set(workload.systems) & set(preset.systems)
        if workload.runner == "harbor":
            task_count = preset.workload_overrides[workload.id]["n_tasks"]
            expected_cells += (
                len(systems & variants_by_system) * len(preset.harnesses) * task_count
            )
        else:
            expected_cells += len(systems)

    assert preview.cells == expected_cells
    assert preview.applicable_cells <= preview.cells
    assert preview.estimated_trials >= preview.applicable_cells
    assert preview.harnesses == ("claude-code", "codex", "hermes", "openclaw")
    assert preview.workloads == ("coding", "continuity", "qa", "retrieval")
    assert set(preview.systems) <= set(preset.systems)
    after = sorted(path.as_posix() for path in runtime.rglob("*") if path.is_file())
    assert after == before
    serialized = json.dumps(asdict(preview))
    assert "OPENAI_API_KEY" not in serialized
    assert "WANDB_API_KEY" not in serialized


def test_explicit_experimental_system_remains_visible_as_not_applicable(
    monkeypatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)

    preview = OperatorService(REPO_ROOT).preview(
        ExperimentRequest(
            experiment_id="repo-memory-impact",
            preset="smoke",
            workloads=("retrieval",),
            systems=("codegraph",),
            model="openai/gpt-5",
        )
    )

    assert preview.cells == 1
    assert preview.applicable_cells == 0
    assert preview.systems == ("codegraph",)


def test_direct_measurements_receive_canonical_identity(tmp_path: Path) -> None:
    spec = get_context_system("rag-bm25", REPO_ROOT)
    runtime = ContextRuntime(
        repo_root=tmp_path,
        cache_root=tmp_path / ".cache",
        env={
            "FUGUE_CANDIDATE_ID": "candidate-a",
            "FUGUE_EXECUTION_FINGERPRINT": "execution-a",
            "FUGUE_IDENTITY_SCHEMA_VERSION": "2",
            "FUGUE_DATASET": "dataset-a",
        },
    )
    rows = [
        {
            "record_type": "retrieval",
            "workload_id": "retrieval",
            "task_name": "task-a",
            "query_id": "probe-a",
            "attempt": attempt,
        }
        for attempt in (1, 2)
    ]

    _add_runtime_correlation(rows, spec, runtime, "run-a")

    assert {row["candidate_id"] for row in rows} == {"candidate-a"}
    assert {row["execution_fingerprint"] for row in rows} == {"execution-a"}
    assert {row["execution_kind"] for row in rows} == {"provider_diagnostic"}
    assert [row["trial_index"] for row in rows] == [1, 2]
    assert len({row["comparison_example_id"] for row in rows}) == 1
