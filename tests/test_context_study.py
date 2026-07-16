from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from fugue.bench.context import ContextRuntime, get_context_system, list_context_systems
from fugue.bench.execution import plan_cells
from fugue.bench.library import get_experiment
from fugue.bench.manifest import load_manifest
from fugue.bench.operator import (
    ExperimentRequest,
    OperatorService,
    _preparation_targets,
    select_preset,
)
from fugue.bench.reproducibility import _planned_prediction_count
from fugue.bench.scoring import (
    build_treatment_selection_lock,
    write_treatment_selection_lock,
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
    research_systems = {
        "codegraph",
        "gitnexus",
        "openwiki",
        "project-rag",
        "semble",
        "latmd",
        "graphiti",
    }
    assert research_systems.isdisjoint(smoke.systems)
    assert research_systems.isdisjoint(full.systems)
    assert all(
        systems[system_id].enabled_by_default is False
        for system_id in (
            "codegraph",
            "gitnexus",
            "openwiki",
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
    assert {target.spec.id for target in targets} <= set(smoke.systems)


def test_hard_memory_v2_lock_is_deterministic_disjoint_and_gold_free() -> None:
    path = REPO_ROOT / "datasets/repo-memory/swe-bench-hard-memory-v2.lock.yaml"
    lock = yaml.safe_load(path.read_text())
    partitions = lock["partitions"]
    rows = [
        row
        for name in ("discovery", "holdout", "gitnexus_ablation")
        for row in partitions[name]
    ]
    ids = [row["id"] for row in rows]

    assert [len(partitions[name]) for name in partitions] == [8, 4, 4]
    assert len(ids) == len(set(ids)) == 16
    assert set(ids).isdisjoint(lock["excluded_prior_live_tasks"])
    assert "expected_paths" not in path.read_text()
    assert all(
        row["selection_sha256"]
        == hashlib.sha256(
            f"{lock['selection_seed']}:{row['id']}".encode()
        ).hexdigest()
        for row in rows
    )
    repositories = {row["repository"]["url"] for row in rows}
    assert all(
        sum(row["repository"]["url"] == repo for row in rows) <= 2
        for repo in repositories
    )

    manifests = {
        "discovery": "swe-bench-hard-discovery-v2.yaml",
        "holdout": "swe-bench-hard-holdout-v2.yaml",
        "gitnexus_ablation": "swe-bench-gitnexus-ablation-v2.yaml",
    }
    for partition, name in manifests.items():
        selected = load_manifest(REPO_ROOT / "datasets/repo-memory" / name)
        assert [task.id for task in selected.tasks] == [
            row["id"] for row in partitions[partition]
        ]


def test_context_preparation_honors_request_task_limit() -> None:
    experiment = get_experiment("repo-memory-impact", REPO_ROOT)
    workload = next(
        item for item in experiment.workloads if item.id == "gitnexus-ablation"
    )
    targets = _preparation_targets(
        experiment=experiment,
        workloads=[workload],
        preset=select_preset(experiment, "gitnexus-ablation"),
        requested_systems=["gitnexus"],
        requested_variants=["gitnexus-vector"],
        requested_n_tasks=1,
        manifest_override=None,
        repo_root=REPO_ROOT,
    )

    assert {target.snapshot.task_id for target in targets} == {
        "sphinx-doc__sphinx-7748"
    }


def test_gitnexus_swe_contract_is_a_distinct_retrieval_required_canary(
    monkeypatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    service = OperatorService(REPO_ROOT)

    canary = service.rendered_jobs(
        ExperimentRequest(
            experiment_id="repo-memory-impact",
            preset="gitnexus-swe-contract",
        ),
        run_id="plan-gitnexus-swe-contract",
        write_configs=False,
    )
    natural = service.rendered_jobs(
        ExperimentRequest(
            experiment_id="repo-memory-impact",
            preset="uptake-diagnostic",
            harnesses=("codex",),
            variants=("uptake-vector",),
            n_tasks=1,
            n_attempts=1,
        ),
        run_id="plan-gitnexus-natural-uptake",
        write_configs=False,
    )

    assert len(canary) == len(natural) == 1
    contract = canary[0]
    unbiased = natural[0]
    assert contract.workload_id == "gitnexus-swe-contract"
    assert contract.task_id == unbiased.task_id == "pydata__xarray-6992"
    assert contract.harness == unbiased.harness == "codex"
    assert contract.prompt_id == "repository-memory-contract"
    assert unbiased.prompt_id is None
    assert (
        contract.resolved_candidate.definition["context"]
        == unbiased.resolved_candidate.definition["context"]
    )
    assert contract.candidate_id != unbiased.candidate_id


def test_pdf_skill_presets_are_controlled_and_study_plans_72_cells(
    tmp_path: Path,
) -> None:
    experiment = get_experiment("skillsbench-pdf-ab", REPO_ROOT)
    smoke = select_preset(experiment, "smoke")
    study = select_preset(experiment, "study")

    assert (smoke.n_attempts, smoke.n_tasks, smoke.n_concurrent) == (1, 1, 4)
    assert (study.n_attempts, study.n_tasks, study.n_concurrent) == (3, 3, 4)
    assert study.scheduling_seed == "skillsbench-pdf-study-v1"
    baseline, treatment = experiment.variants
    baseline_contract = baseline.to_dict()
    treatment_contract = treatment.to_dict()
    for value in (baseline_contract, treatment_contract):
        value.pop("id")
        value.pop("label")
        value.pop("skills")
    assert baseline_contract == treatment_contract
    assert baseline.skills == []
    assert treatment.skills == ["pdf-artifact-workflow"]

    env_file = tmp_path / "env"
    env_file.write_text("WANDB_API_KEY=test\n")
    jobs = OperatorService(REPO_ROOT, env_file=env_file).rendered_jobs(
        ExperimentRequest(
            experiment_id=experiment.id,
            preset="study",
        ),
        run_id="pdf-study-plan",
        write_configs=False,
    )

    assert len(jobs) == 72
    assert {job.trial_index for job in jobs} == {1, 2, 3}
    assert all(job.evaluation_rubrics for job in jobs)
    assert all(experiment.judge_model == "wandb/zai-org/GLM-5.1" for _ in jobs)


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
        if preset.workloads and workload.id not in preset.workloads:
            continue
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


def test_repo_memory_direct_cells_use_direct_result_contract(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    service = OperatorService(REPO_ROOT)
    request = ExperimentRequest(
        experiment_id="repo-memory-impact",
        preset="smoke",
        model="openai/gpt-5",
    )
    jobs = service.rendered_jobs(
        request, run_id="direct-cell-plan", write_configs=False
    )
    direct = [job for job in jobs if job.execution_kind == "provider_diagnostic"]

    assert len(direct) == 8
    assert {cell.n_attempts for cell in direct} == {1}
    assert {job.result_path for job in direct} == {
        REPO_ROOT / ".fugue" / "runtime" / "direct-cell-plan" / "context-results.jsonl"
    }


def test_hard_memory_presets_encode_exact_cohorts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(REPO_ROOT)
    service = OperatorService(REPO_ROOT)
    expected = {
        "context-contract": 48,
        "hard-calibration": 32,
        "hard-discovery": 80,
        "gitnexus-ablation": 96,
    }
    for preset, count in expected.items():
        jobs = service.rendered_jobs(
            ExperimentRequest(
                experiment_id="repo-memory-impact",
                preset=preset,
            ),
            run_id=f"plan-{preset}",
            write_configs=False,
        )
        assert sum(job.n_attempts for job in jobs) == count

    treatments = ("none", "gitnexus-vector", "rag-dense", "rag-hybrid")
    lock = build_treatment_selection_lock(
        source_commit="a" * 40,
        calibration_snapshot_sha256="b" * 64,
        discovery_snapshot_sha256="c" * 64,
        rankings=(
            {"variant_id": "gitnexus-vector"},
            {"variant_id": "rag-dense"},
            {"variant_id": "rag-hybrid"},
            {"variant_id": "agentsmd"},
        ),
        selected_variants=treatments[1:],
    )
    lock_path = write_treatment_selection_lock(tmp_path / "selection.json", lock)
    for preset, count in {
        "hard-holdout": 192,
        "hard-controls": 96,
        "hard-repository-qa": 128,
    }.items():
        jobs = service.rendered_jobs(
            ExperimentRequest(
                experiment_id="repo-memory-impact",
                preset=preset,
                variants=treatments,
                selection_lock=lock_path,
            ),
            run_id=f"plan-{preset}",
            write_configs=False,
        )
        assert sum(job.n_attempts for job in jobs) == count
        assert {job.variant_id for job in jobs} == set(treatments)

    with pytest.raises(ValueError, match="requires --selection-lock"):
        service.rendered_jobs(
            ExperimentRequest(
                experiment_id="repo-memory-impact",
                preset="hard-holdout",
            ),
            run_id="plan-holdout-without-lock",
            write_configs=False,
        )
    with pytest.raises(ValueError, match="disagree"):
        service.rendered_jobs(
            ExperimentRequest(
                experiment_id="repo-memory-impact",
                preset="hard-holdout",
                variants=("none", "agentsmd"),
                selection_lock=lock_path,
            ),
            run_id="plan-holdout-wrong-treatment",
            write_configs=False,
        )


def test_uptake_diagnostic_is_preregistered_and_excluded_from_primary_grid(
    monkeypatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    jobs = OperatorService(REPO_ROOT).rendered_jobs(
        ExperimentRequest(
            experiment_id="repo-memory-impact",
            preset="uptake-diagnostic",
        ),
        run_id="plan-uptake-diagnostic",
        write_configs=False,
    )

    assert len(jobs) == 16
    assert {job.variant_id for job in jobs} == {
        "uptake-none",
        "uptake-vector",
        "uptake-oriented-none",
        "uptake-oriented-vector",
    }
    assert {job.harness for job in jobs} == {"codex"}
    assert {job.task_id for job in jobs} == {
        "pydata__xarray-6992",
        "sphinx-doc__sphinx-7590",
        "django__django-11885",
        "django__django-11400",
    }


def test_direct_study_presets_keep_modes_and_measurement_counts(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    experiment = get_experiment("repo-memory-impact", REPO_ROOT)
    service = OperatorService(REPO_ROOT)
    expected = {
        "retrieval-study": (4, 225, 1),
        "gitnexus-retrieval-study": (2, 225, 1),
        "continuity-study": (6, 6, 3),
    }
    for preset_id, (jobs_count, tasks, attempts) in expected.items():
        preset = select_preset(experiment, preset_id)
        jobs = service.rendered_jobs(
            ExperimentRequest(
                experiment_id=experiment.id,
                preset=preset_id,
            ),
            run_id=f"plan-{preset_id}",
            write_configs=False,
        )
        assert len(jobs) == jobs_count
        assert preset.n_tasks == tasks
        assert preset.n_attempts == attempts

    gitnexus = service.rendered_jobs(
        ExperimentRequest(
            experiment_id=experiment.id,
            preset="gitnexus-retrieval-study",
        ),
        run_id="plan-gitnexus-retrieval",
        write_configs=False,
    )
    assert {job.variant_id for job in gitnexus} == {
        "gitnexus-bm25",
        "gitnexus-vector",
    }
    continuity = service.rendered_jobs(
        ExperimentRequest(
            experiment_id=experiment.id,
            preset="continuity-study",
        ),
        run_id="plan-continuity-reconciliation",
        write_configs=False,
    )
    cells = plan_cells(
        continuity,
        run_id="plan-continuity-reconciliation",
        run_name="continuity reconciliation",
    )
    by_fingerprint = {
        job.resolved_candidate.execution_fingerprint: job for job in continuity
    }
    assert sum(
        _planned_prediction_count(cell, by_fingerprint[cell.execution_fingerprint])
        for cell in cells
    ) == 108
    assert sum(
        _planned_prediction_count(cell, by_fingerprint[cell.execution_fingerprint])
        for cell in cells
        if cell.applicable
    ) == 90


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
