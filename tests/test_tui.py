from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_operator import make_operator_repo
from textual.widgets import Button, Collapsible, ContentSwitcher

from fugue.bench.ai import AssetDraft, ExperimentDraft
from fugue.bench.evaluations import build_evaluation_draft, source_catalog
from fugue.bench.library import experiment_from_data
from fugue.bench.operator import OperatorService
from fugue.tui import (
    CUSTOM_SIZE,
    ConfirmRunScreen,
    FugueApp,
    SaveExperimentScreen,
)


def test_tui_uses_three_step_plan_and_automatic_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    app = FugueApp(service=make_operator_repo(tmp_path), experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=(130, 44)) as pilot:
            await pilot.pause(1)
            assert app.query_one("#workspace").active == "compose"
            assert len(app.query("TabPane")) == 4
            assert app.plan_step == "define-step"
            assert app.plan.preview is not None
            assert app.plan.preview.cells == 1
            assert app.query_one("#setup-table").row_count == 4
            assert app.query_one("#plan-advanced", Collapsible).collapsed
            assert not app.query("#preview")
            assert not app.query("#composer-model")
            assert not app.query("#analysis-filters")

            app.action_next_step()
            assert app.plan_step == "compare-step"
            assert app.query_one("#generate-evaluation", Button)
            app.action_next_step()
            assert app.plan_step == "review-step"
            assert app.query_one("#review-matrix").row_count == 1
            assert "1 cell" in str(app.query_one("#review-summary").render())

            app.action_show_runs()
            assert app.query_one("#workspace").active == "runs"
            app.action_show_results()
            assert app.query_one("#workspace").active == "results"
            app.action_show_setup()
            assert app.query_one("#workspace").active == "setup"

    asyncio.run(exercise())


def test_tui_variants_stay_in_memory_until_explicit_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    app = FugueApp(service=make_operator_repo(tmp_path), experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=(130, 44)) as pilot:
            await pilot.pause()
            app.action_next_step()
            app._duplicate_variant()
            await pilot.pause()

            assert len(app.plan.experiment.variants) == 2
            assert app.plan.dirty
            assert not (
                tmp_path / "configs/fugue/experiments/demo-copy.yaml"
            ).exists()

            await pilot.click("#edit-variant")
            await pilot.pause()
            assert app.screen.query_one("#variant-editor")
            await pilot.click("#save-variant")
            await pilot.pause()

            app.push_screen(
                SaveExperimentScreen(app.plan.experiment),
                app._save_plan,
            )
            await pilot.pause()
            await pilot.click("#confirm-save-experiment")
            await pilot.pause()

            assert (
                tmp_path / "configs/fugue/experiments/demo-copy.yaml"
            ).is_file()
            assert app.experiment_id == "demo-copy"
            assert not app.plan.dirty

    asyncio.run(exercise())


@pytest.mark.parametrize("size", [(80, 24), (100, 32), (130, 44)])
def test_tui_plan_steps_fit_supported_terminal_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    app = FugueApp(service=make_operator_repo(tmp_path), experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            switcher = app.query_one("#plan-steps", ContentSwitcher)
            assert switcher.region.x >= 0
            assert switcher.region.right <= size[0]
            assert switcher.region.bottom <= size[1]
            for step in ("define-step", "compare-step", "review-step"):
                app._show_plan_step(step)
                await pilot.pause()
                assert switcher.current == step

    asyncio.run(exercise())


def test_tui_initial_ai_draft_opens_compare_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    service = make_operator_repo(tmp_path)
    experiment = service.experiment("demo")
    draft = ExperimentDraft(
        experiment=experiment,
        assets=(),
        rationale="Use the checked-in demo configuration.",
        assumptions=(),
        warnings=(),
        diff="",
        preview=service.preview_experiment(experiment),
        model="openai/gpt-5",
        provider="openai",
        session_id="session-1",
        input_tokens=10,
        output_tokens=5,
    )
    app = FugueApp(service=service, experiment_id="demo", initial_draft=draft)

    async def exercise() -> None:
        async with app.run_test(size=(130, 44)) as pilot:
            await pilot.pause()
            assert app.plan_step == "compare-step"
            assert app.plan.experiment == draft.experiment
            assert app.plan.dirty
            assert not (tmp_path / ".fugue").exists()

            scope = SimpleNamespace(
                experiments=("demo",),
                runs=("run-1",),
                rows=1,
                tasks=("task-one",),
                models=("openai/gpt-5",),
                variants=("baseline",),
                sources=("local",),
                warnings=(),
            )
            preview = SimpleNamespace(
                scope=scope,
                spec=SimpleNamespace(id="demo-analysis"),
            )
            app._show_analysis_preview(preview)
            assert app.analysis_preview is preview
            assert "1 experiments" in str(
                app.query_one("#analysis-scope").render()
            )
            assert not app.query_one("#generate-analysis", Button).disabled

            app._show_analysis(
                SimpleNamespace(
                    scope=scope,
                    report="# Demo analysis\n\nOne result [E001].\n",
                    report_dir=tmp_path / "reports/analyses/demo-analysis/run-1",
                )
            )
            assert "1 experiments" in str(
                app.query_one("#analysis-scope").render()
            )

    asyncio.run(exercise())


def test_tui_ai_proposal_requires_explicit_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    service = make_operator_repo(tmp_path)
    experiment = service.experiment("demo")
    draft = ExperimentDraft(
        experiment=replace(experiment, title="Proposed comparison"),
        assets=(),
        rationale="Compare one controlled variant.",
        assumptions=("Use the saved benchmark",),
        warnings=(),
        diff="",
        preview=service.preview_experiment(experiment),
        model="openai/gpt-5",
        provider="openai",
        session_id="session-1",
        input_tokens=10,
        output_tokens=5,
    )
    app = FugueApp(service=service, experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            app._show_proposal(draft)
            assert app.plan.experiment.title == "Demo"
            assert app.plan.proposal is draft
            assert not (tmp_path / ".fugue").exists()

            app._use_proposal()
            assert app.plan.experiment.title == "Proposed comparison"
            assert app.plan.proposal is None
            assert app.plan.dirty
            assert app.plan_step == "compare-step"
            assert not (tmp_path / ".fugue").exists()

    asyncio.run(exercise())


def test_tui_generate_evaluation_reviews_and_saves_all_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    service = make_operator_repo(tmp_path)
    raw = service.experiment("demo").to_dict()
    raw.update(
        {
            "judge_model": "openai/gpt-5-mini",
            "workloads": [{"id": "capabilities", "runner": "harbor"}],
            "evaluation_generation": {
                "size": 8,
                "sources": [
                    {
                        "kind": "seed",
                        "text": "The demo skill uses focused repository search.",
                    }
                ],
            },
        }
    )
    experiment = experiment_from_data(raw)
    strata = ["easy", "boundary", "failure", "integration"]
    cases = [
        {
            "id": f"case-{index + 1:02d}",
            "instruction": f"Explain demo behavior {index + 1}.",
            "family": "skill",
            "source_refs": ["seed:1"],
            "expected": {"facts": ["focused repository search"]},
            "tags": [strata[index % len(strata)]],
        }
        for index in range(8)
    ]
    updated, evaluation = build_evaluation_draft(
        {
            "suite_id": "tui-suite",
            "cases": cases,
            "rubric": {
                "dimensions": [
                    {
                        "id": "task_completion",
                        "criterion": "Complete the requested task.",
                    },
                        {
                            "id": "correctness",
                            "criterion": "Include the grounded fact.",
                        },
                        {
                            "id": "groundedness",
                            "criterion": "Ground the answer in the source.",
                        },
                ]
            },
        },
        experiment,
        generator_model="openai/gpt-5-mini",
        source_catalog=source_catalog(experiment, tmp_path),
        repo_root=tmp_path,
    )
    assets = tuple(
        AssetDraft(
            kind=item.kind,
            id=item.suite_id,
            title=item.path.name,
            body=item.body,
        )
        for item in evaluation.files
    )
    draft = ExperimentDraft(
        experiment=updated,
        assets=assets,
        evaluation=evaluation,
        rationale="Add grounded capability coverage.",
        assumptions=(),
        warnings=(),
        diff="\n".join(f"+++ {item.path}" for item in evaluation.files),
        preview=service.preview_experiment(updated, asset_overlay=evaluation.overlay),
        model="openai/gpt-5-mini",
        provider="openai",
        session_id="session-evaluation",
        input_tokens=100,
        output_tokens=200,
    )
    calls = []

    async def compose(request, **kwargs):
        calls.append((request, kwargs))
        return draft

    monkeypatch.setattr(service, "compose_experiment", compose)
    app = FugueApp(service=service, experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=(130, 44)) as pilot:
            await pilot.pause(1)
            app.action_next_step()
            app._generate_evaluation()
            await pilot.pause(1)

            assert len(calls) == 1
            assert calls[0][1]["base_experiment"].id == "demo"
            assert app.plan.proposal is draft
            assert app.plan_step == "define-step"
            summary = str(app.query_one("#proposal-summary").render())
            assert "Evaluation: 8 cases" in summary
            assert "Provenance: 1 checksum-pinned sources" in summary
            assert "cases.jsonl" in summary
            assert "rubric.yaml" in summary
            assert "manifest.yaml" in summary
            assert not (tmp_path / "configs/fugue/evaluations/tui-suite").exists()

            app._use_proposal()
            await pilot.pause(1)
            assert len(app.plan.assets) == 3
            assert app.plan.dirty
            assert "Save all proposed assets before running" in app._review_blockers

            app._save_plan(("tui-generated", "TUI generated"))
            await pilot.pause(1)

            suite = tmp_path / "configs/fugue/evaluations/tui-suite"
            assert {path.name for path in suite.iterdir()} == {
                "cases.jsonl",
                "rubric.yaml",
                "manifest.yaml",
            }
            assert (tmp_path / "configs/fugue/experiments/tui-generated.yaml").is_file()
            assert app.experiment_id == "tui-generated"
            assert app.plan.assets == ()
            assert not app.plan.dirty
            assert app.plan_step == "review-step"

    asyncio.run(exercise())


def test_full_trace_launch_requires_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    app = FugueApp(service=make_operator_repo(tmp_path), experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause(1)
            app._show_plan_step("review-step")
            app._review_blockers = ()
            app._request_launch()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmRunScreen)

    asyncio.run(exercise())


def test_tui_preset_changes_coverage_without_changing_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    service = make_operator_repo(tmp_path)
    path = tmp_path / "configs/fugue/experiments/demo.yaml"
    path.write_text(
        path.read_text()
        + """
workloads:
  - {id: coding, runner: harbor, manifest: datasets/demo.yaml}
presets:
  smoke: {workloads: [coding], n_tasks: 1, n_attempts: 1}
  full: {workloads: [coding], n_tasks: 5, n_attempts: 2}
default_preset: smoke
"""
    )
    app = FugueApp(service=service, experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            variants = app.plan.request.variants
            app._set_run_size("full")
            assert app.plan.request.preset == "full"
            assert app.plan.request.n_tasks is None
            assert app.plan.request.n_attempts is None
            assert app.plan.request.variants == variants

            app._set_run_size(CUSTOM_SIZE)
            assert app.plan.request.n_tasks == 1
            assert app.plan.request.n_attempts == 1
            assert app.plan.request.variants == variants

    asyncio.run(exercise())


def test_run_shortcut_reviews_before_launching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    service = make_operator_repo(tmp_path)
    launched: list[str] = []
    monkeypatch.setattr(
        service,
        "launch",
        lambda *args, **kwargs: launched.append("run") or None,
    )
    app = FugueApp(service=service, experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            app.action_run()
            assert app.plan_step == "review-step"
            assert launched == []

    asyncio.run(exercise())


def test_memory_smoke_shows_unavailable_cells_without_counting_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    repo_root = Path(__file__).resolve().parents[1]
    service = OperatorService(repo_root, repo_root / ".env")
    app = FugueApp(service=service, experiment_id="repo-memory-impact")

    async def exercise() -> None:
        async with app.run_test(size=(130, 44)) as pilot:
            request = replace(
                app.plan.request,
                preset="smoke",
                workloads=("coding",),
                variants=("none", "rag-bm25"),
                harnesses=("hermes", "openclaw", "claude-code", "codex"),
                n_tasks=1,
                n_attempts=1,
            )
            app.plan = replace(app.plan, request=request, preview=None)
            app._render_plan()
            app._begin_preview()
            await pilot.pause(1)

            assert app.plan.preview is not None
            assert app.plan.preview.cells == 8
            assert app.plan.preview.applicable_cells == 8
            assert app.plan.preview.estimated_trials == 8
            assert set(app.plan.preview.variants) == {"none", "rag-bm25"}
            assert all(cell.applicable for cell in app.plan.preview.matrix_cells)
            codex_rag = next(
                cell
                for cell in app.plan.preview.matrix_cells
                if cell.harness == "codex" and cell.variant_id == "rag-bm25"
            )
            assert codex_rag.context_transport == "portable"

    asyncio.run(exercise())
