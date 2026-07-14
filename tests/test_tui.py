from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_operator import make_operator_repo

from fugue.bench.ai import ExperimentDraft
from fugue.bench.operator import PreviewSummary
from fugue.tui import FugueApp


def test_tui_renders_four_operator_screens_and_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    service = make_operator_repo(tmp_path)
    app = FugueApp(service=service, experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=(130, 44)) as pilot:
            await pilot.pause()
            assert app.query_one("#workspace").active == "compose"
            assert len(app.query("TabPane")) == 4
            assert app.query_one("#setup-table").row_count == 10
            assert app.query_one("#composer-drawer")
            assert app.query_one("#analyst-drawer")
            assert not app.query("#system-list")
            assert not app.query("#run-detached")
            assert not hasattr(app, "_run_cli")

            await pilot.click("#preview")
            await pilot.pause(1)
            assert "1 cells" in str(app.query_one("#matrix-summary").render())

            app.action_show_runs()
            assert app.query_one("#workspace").active == "runs"
            app.action_show_results()
            assert app.query_one("#workspace").active == "results"
            app.action_show_setup()
            assert app.query_one("#workspace").active == "setup"

    asyncio.run(exercise())


def test_tui_edits_variants_and_saves_repo_backed_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    service = make_operator_repo(tmp_path)
    app = FugueApp(service=service, experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=(130, 44)) as pilot:
            await pilot.pause()
            await pilot.click("#edit-variant")
            await pilot.pause()
            assert app.screen.query_one("#variant-editor")
            await pilot.click("#save-variant")
            await pilot.pause()

            await pilot.click("#save-experiment")
            await pilot.pause()
            assert app.screen.query_one("#save-experiment-panel")
            await pilot.click("#confirm-save-experiment")
            await pilot.pause()
            assert (tmp_path / "configs/fugue/experiments/demo-copy.yaml").is_file()
            assert app.experiment_id == "demo-copy"

    asyncio.run(exercise())


def test_tui_compose_columns_do_not_overlap_in_narrow_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUGUE_NO_ANIMATION", "1")
    app = FugueApp(service=make_operator_repo(tmp_path), experiment_id="demo")

    async def exercise() -> None:
        async with app.run_test(size=(84, 34)) as pilot:
            await pilot.pause()
            fields = app.query_one("#compose-fields").region
            summary = app.query_one(".summary-column").region
            assert fields.right <= summary.x
            assert summary.right <= 84

    asyncio.run(exercise())


def test_tui_applies_ai_draft_and_renders_analysis_scope(
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
        preview=PreviewSummary(
            cells=1,
            applicable_cells=1,
            estimated_trials=1,
            harnesses=("codex",),
            variants=("baseline",),
            systems=("none",),
            workloads=("harbor",),
            commands=("harbor run --config preview.json",),
        ),
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
            assert app.applied_draft is draft
            assert "AI draft applied" in str(app.query_one("#matrix-summary").render())

            scope = SimpleNamespace(
                experiments=("demo",),
                runs=("run-1",),
                rows=1,
                tasks=("task-one",),
                models=("openai/gpt-5",),
                variants=("baseline",),
                sources=("local",),
            )
            preview = SimpleNamespace(
                scope=scope,
                spec=SimpleNamespace(id="demo-analysis"),
            )
            app._show_analysis_preview(preview)
            assert app.analysis_preview is preview
            assert "1 experiments" in str(app.query_one("#analysis-scope").render())

            app._show_analysis(
                SimpleNamespace(
                    scope=scope,
                    report="# Demo analysis\n\nOne result [E001].\n",
                    spec=SimpleNamespace(id="demo-analysis"),
                    report_dir=tmp_path / "reports/analyses/demo-analysis/run-1",
                )
            )
            assert "1 experiments" in str(app.query_one("#analysis-scope").render())
            assert app.query_one("#analysis-save-id").value == "demo-analysis"

    asyncio.run(exercise())
