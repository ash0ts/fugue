from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from fugue.bench.job_config import preview_jobs
from fugue.bench.library import get_experiment, get_skill
from fugue.bench.manifest import load_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "datasets" / "skillsbench-pdf.yaml"
SKILLSBENCH_DIGEST = (
    "sha256:145925c10bc09425dc0201772cfa50d9b800010081cf5ad77969554a644d7ae1"
)


def test_skillsbench_pdf_demo_is_a_balanced_side_effect_free_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This assertion covers the authored remote dataset, not machine-local setup state.
    monkeypatch.setattr(
        "fugue.bench.job_config.read_task_runtime_lock",
        lambda *_args, **_kwargs: None,
    )
    manifest = load_manifest(MANIFEST_PATH)
    experiment = get_experiment("skillsbench-pdf-ab", REPO_ROOT)
    skill = get_skill("pdf-artifact-workflow", REPO_ROOT)

    assert manifest.dataset.ref == "benchflow/skillsbench"
    assert manifest.dataset.version == SKILLSBENCH_DIGEST
    assert [task.id for task in manifest.tasks] == [
        "court-form-filling",
        "pdf-excel-diff",
        "paper-anonymizer",
    ]
    assert [harness.name for harness in manifest.harnesses] == [
        "hermes",
        "openclaw",
        "claude-code",
        "codex",
    ]
    assert skill.title == "PDF artifact workflow"
    assert len(skill.sha256) == 64

    run_id = f"test-preview-{uuid4().hex}"
    runtime_dir = REPO_ROOT / ".fugue" / "runtime" / run_id
    assert not runtime_dir.exists()

    rendered = preview_jobs(
        experiment=experiment,
        manifest=manifest,
        manifest_path=MANIFEST_PATH,
        repo_root=REPO_ROOT,
        env={"WANDB_API_KEY": "secret-wandb"},
        run_id=run_id,
    )

    assert len(rendered) == 24
    assert not runtime_dir.exists()
    assert all(not job.config_path.exists() for job in rendered)

    cells = {
        (job.harness, job.variant_id, job.task_id, job.trial_index): job
        for job in rendered
    }
    expected_artifacts = {
        "court-form-filling": ["/root/sc100-filled.pdf"],
        "pdf-excel-diff": ["/root/diff_report.json"],
        "paper-anonymizer": ["/root/redacted"],
    }
    expected_tasks = [
        "court-form-filling",
        "pdf-excel-diff",
        "paper-anonymizer",
    ]

    for harness in ("hermes", "openclaw", "claude-code", "codex"):
        for task_id in expected_tasks:
            for trial_index in (1,):
                baseline_job = cells[(harness, "baseline", task_id, trial_index)]
                skilled_job = cells[
                    (harness, "with-pdf-skill", task_id, trial_index)
                ]
                _assert_balanced_skill_pair(
                    baseline_job.config,
                    skilled_job.config,
                    expected_artifacts[task_id],
                    task_id,
                )
                assert baseline_job.comparison_example_id == (
                    skilled_job.comparison_example_id
                )
                assert baseline_job.candidate_id != skilled_job.candidate_id


def test_pdf_skill_keeps_visual_inspection_bounded_and_provider_neutral() -> None:
    source = (
        REPO_ROOT / "configs/fugue/skills/pdf-artifact-workflow/SKILL.md"
    ).read_text()

    assert "Structure and text are the default inspection path" in source
    assert "documented contract to carry images as structured media" in source
    assert "base64, data URLs, raw bytes" in source
    for bound in ("one page or tight crop at a time", "100 DPI", "1200", "512 KiB"):
        assert bound in source
    assert "at most four previews" in source
    for harness in ("Hermes", "OpenClaw", "Claude", "Codex"):
        assert harness not in source


def _assert_balanced_skill_pair(
    baseline: dict, skilled: dict, expected_artifacts: list[str], task_id: str
) -> None:
    assert baseline["datasets"] == skilled["datasets"]
    assert baseline["datasets"] == [
        {
            "name": "benchflow/skillsbench",
            "ref": SKILLSBENCH_DIGEST,
            "task_names": [f"benchflow/{task_id}"],
            "n_tasks": 1,
        }
    ]
    assert baseline["n_attempts"] == skilled["n_attempts"] == 1
    assert baseline["n_concurrent_trials"] == skilled["n_concurrent_trials"] == 4
    assert baseline["artifacts"] == skilled["artifacts"] == expected_artifacts
    assert baseline["environment"] == skilled["environment"]
    assert baseline.get("extra_instruction_paths", []) == []
    assert skilled.get("extra_instruction_paths", []) == []

    baseline_agent = dict(baseline["agents"][0])
    skilled_agent = dict(skilled["agents"][0])
    assert baseline_agent.pop("skills", []) == []
    assert skilled_agent.pop("skills") == [
        "configs/fugue/skills/pdf-artifact-workflow"
    ]
    assert baseline_agent == skilled_agent

    assert "secret-wandb" not in json.dumps(baseline)
    assert "secret-wandb" not in json.dumps(skilled)
