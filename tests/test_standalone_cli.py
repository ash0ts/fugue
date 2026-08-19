from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.cli import main

_TEMPLATES = (
    "prompt-change",
    "skill-change",
    "mcp-change",
    "memory-change",
    "harness-change",
)


def _write_study_marker(root: Path, *, template: str = "prompt-change") -> None:
    (root / ".fugue-study.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "fugue_standalone_study",
                "template": template,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("template", _TEMPLATES)
def test_init_forwards_standalone_template(
    template: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fugue.bench import comparison

    captured: dict[str, object] = {}
    destination = tmp_path / template

    def scaffold(
        selected: Path,
        *,
        template: str,
        force: bool,
    ) -> Path:
        captured.update(
            destination=selected,
            template=template,
            force=force,
        )
        return selected / "comparison.yaml"

    monkeypatch.setattr(comparison, "scaffold_comparison", scaffold)

    assert (
        main(
            [
                "init",
                destination.as_posix(),
                "--template",
                template,
                "--force",
            ]
        )
        == 0
    )
    assert captured == {
        "destination": destination,
        "template": template,
        "force": True,
    }


def test_init_defaults_to_prompt_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fugue.bench import comparison

    captured: dict[str, object] = {}

    def scaffold(
        selected: Path,
        *,
        template: str,
        force: bool,
    ) -> Path:
        captured.update(template=template, force=force)
        return selected / "comparison.yaml"

    monkeypatch.setattr(comparison, "scaffold_comparison", scaffold)

    assert main(["init", (tmp_path / "study").as_posix()]) == 0
    assert captured == {"template": "prompt-change", "force": False}


def test_check_infers_generated_study_root_from_exact_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.bench import comparison as comparison_module

    study = tmp_path / "my-study"
    study.mkdir()
    comparison = study / "comparison.yaml"
    comparison.write_text("schema_version: 1\n", encoding="utf-8")
    _write_study_marker(study)
    captured: dict[str, object] = {}
    spec = object()

    def load(path: Path, *, repo_root: Path) -> object:
        captured.update(path=path, load_root=repo_root)
        return spec

    def check(selected: object, *, repo_root: Path) -> SimpleNamespace:
        captured.update(spec=selected, check_root=repo_root)
        return SimpleNamespace(
            status="ready",
            to_dict=lambda: {"status": "ready"},
        )

    monkeypatch.setattr(comparison_module, "load_comparison", load)
    monkeypatch.setattr(comparison_module, "check_comparison", check)
    monkeypatch.chdir(tmp_path)

    assert main(["check", "my-study/comparison.yaml", "--json"]) == 0
    assert captured == {
        "path": comparison.resolve(),
        "load_root": study.resolve(),
        "spec": spec,
        "check_root": study.resolve(),
    }
    assert json.loads(capsys.readouterr().out) == {"status": "ready"}


def test_explicit_repo_root_overrides_generated_study_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fugue.bench import comparison as comparison_module

    study = tmp_path / "my-study"
    study.mkdir()
    (study / "comparison.yaml").write_text(
        "schema_version: 1\n",
        encoding="utf-8",
    )
    _write_study_marker(study)
    captured: dict[str, object] = {}

    def load(path: Path, *, repo_root: Path) -> object:
        captured.update(path=path, root=repo_root)
        return object()

    monkeypatch.setattr(comparison_module, "load_comparison", load)
    monkeypatch.setattr(
        comparison_module,
        "check_comparison",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="ready",
            to_dict=lambda: {"status": "ready"},
        ),
    )
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "check",
                "my-study/comparison.yaml",
                "--repo-root",
                tmp_path.as_posix(),
                "--json",
            ]
        )
        == 0
    )
    assert captured == {
        "path": Path("my-study/comparison.yaml"),
        "root": tmp_path.resolve(),
    }


def test_invalid_study_marker_preserves_current_directory_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fugue.bench import comparison as comparison_module

    study = tmp_path / "my-study"
    study.mkdir()
    (study / "comparison.yaml").write_text(
        "schema_version: 1\n",
        encoding="utf-8",
    )
    _write_study_marker(study, template="unknown-template")
    captured: dict[str, object] = {}

    def load(path: Path, *, repo_root: Path) -> object:
        captured.update(path=path, root=repo_root)
        return object()

    monkeypatch.setattr(comparison_module, "load_comparison", load)
    monkeypatch.setattr(
        comparison_module,
        "check_comparison",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="ready",
            to_dict=lambda: {"status": "ready"},
        ),
    )
    monkeypatch.chdir(tmp_path)

    assert main(["check", "my-study/comparison.yaml", "--json"]) == 0
    assert captured == {
        "path": Path("my-study/comparison.yaml"),
        "root": tmp_path.resolve(),
    }


def test_compare_preview_uses_generated_root_and_root_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.bench import comparison as comparison_module

    study = tmp_path / "my-study"
    study.mkdir()
    comparison = study / "comparison.yaml"
    comparison.write_text("schema_version: 1\n", encoding="utf-8")
    _write_study_marker(study, template="mcp-change")
    captured: dict[str, object] = {}
    spec = object()

    def load(path: Path, *, repo_root: Path) -> object:
        captured.update(path=path, load_root=repo_root)
        return spec

    def preview(
        selected: object,
        *,
        repo_root: Path,
        operator: object,
    ) -> SimpleNamespace:
        captured.update(
            spec=selected,
            preview_root=repo_root,
            operator_root=operator.repo_root,
            env_file=operator.env_file,
        )
        return SimpleNamespace(
            preview_digest="a" * 64,
            matrix={"applicable_cells": 2, "estimated_trials": 2},
            readiness={"estimated_cells": 2},
            to_dict=lambda: {
                "preview_digest": "a" * 64,
                "matrix": {
                    "applicable_cells": 2,
                    "estimated_trials": 2,
                },
                "readiness": {"estimated_cells": 2},
            },
        )

    monkeypatch.setattr(comparison_module, "load_comparison", load)
    monkeypatch.setattr(
        comparison_module,
        "check_comparison",
        lambda *_args, **_kwargs: SimpleNamespace(status="ready"),
    )
    monkeypatch.setattr(comparison_module, "preview_comparison", preview)
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "compare",
                "my-study/comparison.yaml",
                "--preview",
                "--json",
            ]
        )
        == 0
    )
    assert captured == {
        "path": comparison.resolve(),
        "load_root": study.resolve(),
        "spec": spec,
        "preview_root": study.resolve(),
        "operator_root": study.resolve(),
        "env_file": study.resolve() / ".env",
    }
    assert json.loads(capsys.readouterr().out)["approval_eligible"] is True


def test_human_comparison_preview_maps_labels_to_full_candidate_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.bench import comparison as comparison_module

    study = tmp_path / "my-study"
    study.mkdir()
    comparison = study / "comparison.yaml"
    comparison.write_text("schema_version: 1\n", encoding="utf-8")
    _write_study_marker(study)
    spec = SimpleNamespace(execution=SimpleNamespace(evidence_mode="local"))
    baseline_id = "a" * 64
    candidate_id = "b" * 64
    readiness = {
        "status": "ready",
        "question": "Does the treatment help?",
        "evidence_project": None,
        "task_count": 1,
        "actual_changes": ["prompt"],
        "base_failures": 1,
        "gold_passes": 1,
        "judge_evaluators": [],
        "estimated_cells": 2,
        "estimated_cost_usd": 1.0,
        "blockers": [],
        "warnings": [],
    }
    matrix_cells = [
        {
            "variant_id": "baseline",
            "variant_label": "Current prompt",
            "harness": "claude-code",
            "candidate_id": baseline_id,
        },
        {
            "variant_id": "candidate",
            "variant_label": "Proposed prompt",
            "harness": "claude-code",
            "candidate_id": candidate_id,
        },
        # Multiple tasks with the same treatment identity render once.
        {
            "variant_id": "candidate",
            "variant_label": "Proposed prompt",
            "harness": "claude-code",
            "candidate_id": candidate_id,
        },
    ]

    monkeypatch.setattr(
        comparison_module,
        "load_comparison",
        lambda *_args, **_kwargs: spec,
    )
    monkeypatch.setattr(
        comparison_module,
        "check_comparison",
        lambda *_args, **_kwargs: SimpleNamespace(status="ready"),
    )
    monkeypatch.setattr(
        comparison_module,
        "preview_comparison",
        lambda *_args, **_kwargs: SimpleNamespace(
            preview_digest="c" * 64,
            matrix={
                "applicable_cells": 2,
                "estimated_trials": 2,
                "matrix_cells": matrix_cells,
            },
            readiness=readiness,
        ),
    )
    monkeypatch.chdir(tmp_path)

    assert main(["compare", "my-study/comparison.yaml", "--preview"]) == 0

    output = capsys.readouterr().out
    compact = "".join(output.split())
    assert "Treatment identities" in output
    assert "Current prompt (baseline; claude-code)" in output
    assert "Proposed prompt (candidate; claude-code)" in output
    assert baseline_id in compact
    assert candidate_id in compact
    assert compact.count(candidate_id) == 1
