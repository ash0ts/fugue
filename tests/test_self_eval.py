from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from fugue.bench.ai import get_analysis
from fugue.bench.library import get_experiment, get_prompt, get_skill
from fugue.bench.manifest import load_manifest
from fugue.bench.operator import OperatorService

BASE_COMMIT = "96512017842d68add2546a057f0601de3eaf610e"
ROLES = ("maintainer", "operator")


@pytest.mark.parametrize("role", ROLES)
def test_self_eval_assets_and_experiment(role: str):
    experiment = get_experiment(f"fugue-{role}-self-eval")

    assert experiment.default_preset == "smoke"
    assert experiment.harnesses == ["hermes", "openclaw", "claude-code", "codex"]
    assert [item.id for item in experiment.variants] == [
        "baseline",
        "role-prompt",
        "role-prompt-skill",
        "role-prompt-skill-agentsmd",
    ]
    assert get_prompt(f"fugue-{role}").body
    assert get_skill(f"fugue-{role}").body
    analysis = get_analysis(f"fugue-{role}-selection")
    assert analysis.selection is not None
    assert analysis.filters["tag"] == "phase:holdout"


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("split", ("dev", "holdout"))
def test_self_eval_manifests_are_pinned_and_have_six_tasks(role: str, split: str):
    manifest = load_manifest(Path(f"datasets/fugue-self-eval/{role}-{split}.yaml"))

    assert len(manifest.tasks) == 6
    assert len({task.id for task in manifest.tasks}) == 6
    assert all(task.repo == "ash0ts/fugue" for task in manifest.tasks)
    assert all(task.base_commit == BASE_COMMIT for task in manifest.tasks)
    assert manifest.dataset.path == Path(f"datasets/fugue-self-eval/v1/{role}")


@pytest.mark.parametrize("role", ROLES)
def test_self_eval_smoke_preview_is_48_side_effect_free(role: str):
    service = OperatorService(Path.cwd())
    experiment = service.experiment(f"fugue-{role}-self-eval")
    before = set((Path(".fugue/runtime")).rglob("*")) if Path(".fugue/runtime").exists() else set()

    preview = service.preview(service.request_for_experiment(experiment))

    after = set((Path(".fugue/runtime")).rglob("*")) if Path(".fugue/runtime").exists() else set()
    assert preview.cells == 48
    assert preview.estimated_trials == 48
    assert len(preview.harnesses) == 4
    assert len(preview.variants) == 4
    assert before == after


def test_maintainer_mutations_apply_to_the_current_base_contract():
    paths = sorted(
        Path("datasets/fugue-self-eval/v1/maintainer").glob(
            "*/environment/mutation.patch"
        )
    )

    assert len(paths) == 12
    for path in paths:
        solution_patch = path.parents[1] / "solution/mutation.patch"
        assert solution_patch.read_bytes() == path.read_bytes()
        result = subprocess.run(
            ["git", "apply", "--check", path.as_posix()],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"


@pytest.mark.parametrize("role", ROLES)
def test_self_eval_harbor_tasks_are_complete_and_executable(role: str):
    roots = sorted(Path(f"datasets/fugue-self-eval/v1/{role}").glob("fugue-*"))

    assert len(roots) == 12
    for root in roots:
        for relative in (
            "task.toml",
            "instruction.md",
            "environment/Dockerfile",
            "solution/solve.sh",
            "tests/test.sh",
        ):
            assert (root / relative).is_file(), f"missing {root / relative}"
        assert os.access(root / "solution/solve.sh", os.X_OK)
        assert os.access(root / "tests/test.sh", os.X_OK)
        for script in (root / "solution/solve.sh", root / "tests/test.sh"):
            subprocess.run(["sh", "-n", script.as_posix()], check=True)
        setup = root / "environment/setup.sh"
        if setup.exists():
            subprocess.run(["sh", "-n", setup.as_posix()], check=True)
