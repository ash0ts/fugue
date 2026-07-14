from __future__ import annotations

import asyncio
import os
import sys
import webbrowser
from collections import defaultdict
from dataclasses import replace
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Footer,
    Input,
    Label,
    RichLog,
    Select,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)

from fugue.bench.context import list_context_systems
from fugue.bench.library import (
    ContextSelection,
    ExperimentSpec,
    get_experiment,
    list_prompts,
    list_skills,
    save_experiment_data,
)
from fugue.bench.operator import ExperimentRequest, OperatorService, RunSummary

HARNESS_LABELS = (
    ("Hermes", "hermes"),
    ("OpenClaw", "openclaw"),
    ("Claude Code", "claude-code"),
    ("Codex", "codex"),
)


class PixelSequencer(Static):
    DEFAULT_CSS = """
    PixelSequencer {
        height: 6;
        padding: 0 2;
        background: #1A1C1F;
        border-bottom: solid #363B44;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.phase = 0
        self.statuses: dict[str, str] = {}

    def on_mount(self) -> None:
        if _animation_enabled():
            self.set_interval(0.18, self._tick)
        self._draw()

    def set_statuses(self, statuses: dict[str, str]) -> None:
        self.statuses = statuses
        self._draw()

    def _tick(self) -> None:
        self.phase = (self.phase + 1) % 18
        self._draw()

    def _draw(self) -> None:
        output = Text()
        for label, harness in HARNESS_LABELS:
            status = self.statuses.get(harness, "idle")
            color = {
                "running": "#00AFC2",
                "passed": "#22C55E",
                "failed": "#EF4444",
                "cancelled": "#F59E0B",
                "interrupted": "#F59E0B",
                "not_applicable": "#6B7280",
                "pending": "#9CA3AF",
            }.get(status, "#6B7280")
            output.append(f"{label.upper():<12}", style="#D1D5DB")
            for index in range(18):
                if index == self.phase and status == "running":
                    output.append("■", style="bold #FFCC33")
                elif index < self.phase and status == "running":
                    output.append("▪", style="#00AFC2")
                elif status in {"passed", "failed", "cancelled", "interrupted"}:
                    output.append("▪", style=color)
                else:
                    output.append("·", style="#454B55")
            output.append(f"  {status.replace('_', ' ')}\n", style=color)
        self.update(output)


class HelpScreen(ModalScreen[None]):
    DEFAULT_CSS = """
    HelpScreen { align: center middle; background: #000000 60%; }
    #help-panel {
        width: 64;
        height: 25;
        padding: 1 2;
        border: solid #FFCC33;
        background: #1A1C1F;
    }
    #close-help { dock: bottom; width: 100%; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-panel"):
            yield Label("FUGUE / KEYBOARD", classes="section-title")
            yield Static(
                "1 Plan        2 Runs       3 Results      4 Setup\n"
                "/ Commands    r Run                       c Cancel\n"
                "e Export      a Agents     w Trace        ? Help\n\n"
                "Fugue operates experiments locally. Weave Agents explains each "
                "conversation, turn, model call, and tool invocation."
            )
            yield Button("Close", id="close-help", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()


class VariantEditorScreen(ModalScreen[str | None]):
    DEFAULT_CSS = """
    VariantEditorScreen { align: center middle; background: #000000 60%; }
    #variant-editor {
        width: 90%;
        max-width: 76;
        height: 90%;
        max-height: 36;
        padding: 1 2;
        border: solid #FFCC33;
        background: #1A1C1F;
    }
    #variant-skills { height: 10; }
    """

    def __init__(self, service: OperatorService, experiment_id: str) -> None:
        super().__init__()
        self.service = service
        self.experiment_id = experiment_id

    def compose(self) -> ComposeResult:
        experiment = get_experiment(self.experiment_id, self.service.repo_root)
        with Vertical(id="variant-editor"):
            yield Label("EDIT VARIANT", classes="section-title")
            yield Select(
                [(item.label, item.id) for item in experiment.variants],
                value=experiment.variants[0].id,
                allow_blank=False,
                id="edit-variant-select",
            )
            yield Label("Prompt", classes="muted")
            yield Select(
                [("Default prompt", ""), *[(item.title, item.id) for item in list_prompts(self.service.repo_root)]],
                value="",
                allow_blank=False,
                id="edit-prompt-select",
            )
            yield Label("Skills", classes="muted")
            yield SelectionList(id="variant-skills")
            yield Label("Context system", classes="muted")
            yield Select(
                [(item.title, item.id) for item in list_context_systems(self.service.repo_root)],
                allow_blank=False,
                id="edit-context-select",
            )
            with Horizontal(classes="button-row"):
                yield Button("Save variant", id="save-variant", variant="primary")
                yield Button("Cancel", id="cancel-variant")

    def on_mount(self) -> None:
        self._load_variant()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "edit-variant-select":
            self._load_variant()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-variant":
            self.dismiss(None)
            return
        if event.button.id != "save-variant":
            return
        try:
            experiment = get_experiment(self.experiment_id, self.service.repo_root)
            variant_id = str(self.query_one("#edit-variant-select", Select).value)
            prompt_value = self.query_one("#edit-prompt-select", Select).value
            context_value = self.query_one("#edit-context-select", Select).value
            skills = list(self.query_one("#variant-skills", SelectionList).selected)
            variants = [
                replace(
                    item,
                    prompt_id=str(prompt_value) or None,
                    skill_ids=skills,
                    context=ContextSelection(system_id=str(context_value)),
                )
                if item.id == variant_id
                else item
                for item in experiment.variants
            ]
            data = experiment.to_dict()
            data["variants"] = [item.to_dict() for item in variants]
            save_experiment_data(experiment.id, data, self.service.repo_root)
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self.dismiss(experiment.id)

    def _load_variant(self) -> None:
        experiment = get_experiment(self.experiment_id, self.service.repo_root)
        selected = str(self.query_one("#edit-variant-select", Select).value)
        variant = next(item for item in experiment.variants if item.id == selected)
        self.query_one("#edit-prompt-select", Select).value = variant.prompt_id or ""
        context = self.query_one("#edit-context-select", Select)
        context.value = variant.context.system_id
        skills = self.query_one("#variant-skills", SelectionList)
        skills.clear_options()
        for item in list_skills(self.service.repo_root):
            skills.add_option((item.title, item.id, item.id in variant.skill_ids))


class SaveExperimentScreen(ModalScreen[str | None]):
    DEFAULT_CSS = """
    SaveExperimentScreen { align: center middle; background: #000000 60%; }
    #save-experiment-panel {
        width: 90%;
        max-width: 68;
        height: 20;
        padding: 1 2;
        border: solid #FFCC33;
        background: #1A1C1F;
    }
    """

    def __init__(
        self,
        service: OperatorService,
        experiment: ExperimentSpec,
        request: ExperimentRequest,
    ) -> None:
        super().__init__()
        self.service = service
        self.experiment = experiment
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="save-experiment-panel"):
            yield Label("SAVE EXPERIMENT", classes="section-title")
            yield Input(
                value=f"{self.experiment.id}-copy",
                placeholder="Experiment id",
                id="save-experiment-id",
            )
            yield Input(
                value=f"{self.experiment.title} copy",
                placeholder="Title",
                id="save-experiment-title",
            )
            yield Static(
                "Saves model, harness, variant, trial, task, concurrency, tag, and trace choices under configs/fugue/experiments.",
                classes="muted",
            )
            with Horizontal(classes="button-row"):
                yield Button("Save", id="confirm-save-experiment", variant="primary")
                yield Button("Cancel", id="cancel-save-experiment")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-save-experiment":
            self.dismiss(None)
            return
        if event.button.id != "confirm-save-experiment":
            return
        item_id = self.query_one("#save-experiment-id", Input).value.strip()
        try:
            data = self.experiment.to_dict()
            selected_variants = set(self.request.variants)
            data.update(
                {
                    "id": item_id,
                    "title": self.query_one("#save-experiment-title", Input).value.strip(),
                    "model": self.request.model or self.experiment.model,
                    "builder_model": self.request.builder_model,
                    "judge_model": self.request.judge_model,
                    "run_name": self.request.run_name,
                    "tags": list(self.request.tags),
                    "harnesses": list(self.request.harnesses),
                    "n_attempts": self.request.n_attempts,
                    "n_tasks": self.request.n_tasks,
                    "n_concurrent": self.request.n_concurrent,
                    "trace_content": self.request.trace_content,
                    "variants": [
                        {**variant.to_dict(), "enabled": variant.id in selected_variants}
                        for variant in self.experiment.variants
                    ],
                }
            )
            save_experiment_data(item_id, data, self.service.repo_root)
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self.dismiss(item_id)


class FugueApp(App[None]):
    TITLE = "Fugue"
    ENABLE_COMMAND_PALETTE = True
    CSS = """
    $background: #171A1F;
    $panel: #1A1C1F;
    $raised: #2B3038;
    $gold: #FFCC33;
    $cyan: #00AFC2;

    Screen { background: $background; color: #FFFFFF; }
    #masthead {
        height: 3;
        padding: 0 2;
        background: $panel;
        content-align: left middle;
        color: $gold;
        text-style: bold;
    }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; }
    ContentSwitcher { height: 1fr; }
    Tabs { background: $panel; color: #D1D5DB; }
    Tab.-active { color: $gold; text-style: bold; }
    .pane { padding: 1 2; height: 1fr; }
    .column { width: 1fr; height: 1fr; padding-right: 2; }
    .summary-column { width: 1fr; min-width: 34; height: 1fr; }
    .section-title { color: $gold; text-style: bold; margin-bottom: 1; }
    .muted { color: #9CA3AF; }
    .warning { color: #F59E0B; }
    Input, Select {
        background: $raised;
        border: tall #454B55;
        margin-bottom: 1;
    }
    Input:focus, Select:focus { border: tall $gold; }
    SelectionList {
        height: 5;
        background: $panel;
        border: solid #363B44;
        margin-bottom: 1;
    }
    DataTable {
        height: 1fr;
        background: $panel;
        border: solid #363B44;
    }
    DataTable:focus { border: solid $cyan; }
    RichLog {
        height: 1fr;
        background: #111317;
        border: solid #363B44;
        color: #D1D5DB;
    }
    Button { margin-right: 1; min-width: 14; }
    Button.-primary { background: $gold; color: #171A1F; }
    Button:focus { text-style: bold; }
    .button-row { height: 3; margin: 1 0; }
    #compose-fields { width: 42; min-width: 36; }
    #matrix-summary { height: 10; background: $panel; padding: 1 2; }
    #command-preview { height: 1fr; }
    #composer-drawer { min-height: 13; margin-bottom: 1; }
    #composer-log { height: 8; }
    #runs-table { height: 10; }
    #cells-table { height: 12; }
    #run-log { height: 1fr; }
    #result-summary { height: 5; padding: 1 2; background: $panel; }
    #analyst-drawer { min-height: 15; margin-bottom: 1; }
    #analysis-report { height: 10; }
    #analysis-scope { min-height: 3; background: $panel; padding: 1 2; }
    #setup-table { height: 16; }
    #setup-log { height: 1fr; }
    Footer { background: $panel; color: #9CA3AF; }
    """
    BINDINGS = [
        Binding("1", "show_compose", "Plan", show=True),
        Binding("2", "show_runs", "Runs", show=True),
        Binding("3", "show_results", "Results", show=True),
        Binding("4", "show_setup", "Setup", show=True),
        Binding("/", "command_palette", "Commands", show=False),
        Binding("?", "help", "Help", show=False),
        Binding("r", "run", "Run", show=False),
        Binding("c", "cancel", "Cancel", show=False),
        Binding("e", "export", "Export", show=False),
        Binding("a", "open_agents", "Agents", show=False),
        Binding("w", "open_trace", "Trace", show=False),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        *,
        service: OperatorService | None = None,
        initial_screen: str = "compose",
        experiment_id: str = "pilot",
        initial_draft: Any = None,
    ) -> None:
        super().__init__()
        self.service = service or OperatorService()
        self.initial_screen = initial_screen
        self.experiment_id = experiment_id
        self.initial_draft = initial_draft
        self.selected_run_id: str | None = None
        self.selected_cell_id: str | None = None
        self.last_preview: ExperimentRequest | None = None
        self.composer_draft: Any = None
        self.applied_draft: Any = None
        self.analysis_preview: Any = None
        self.analysis_result: Any = None
        self._log_target: tuple[str, str | None] | None = None
        self._log_offset = 0

    def compose(self) -> ComposeResult:
        yield Static("FUGUE  /  AGENT EXPERIMENT OPERATOR", id="masthead")
        yield PixelSequencer()
        with TabbedContent(initial=self.initial_screen, id="workspace"):
            with TabPane("Plan", id="compose"):
                yield from self._compose_experiment()
            with TabPane("Runs", id="runs"):
                yield from self._compose_runs()
            with TabPane("Results", id="results"):
                yield from self._compose_results()
            with TabPane("Setup", id="setup"):
                yield from self._compose_setup()
        yield Footer()

    def _compose_experiment(self) -> ComposeResult:
        experiment_options = self.service.experiment_items()
        selected = self.experiment_id if any(value == self.experiment_id for _, value in experiment_options) else experiment_options[0][1]
        with Horizontal(classes="pane"):
            with VerticalScroll(id="compose-fields"):
                yield Label("EXPERIMENT", classes="section-title")
                yield Select(
                    experiment_options,
                    value=selected,
                    allow_blank=False,
                    id="experiment-select",
                )
                yield Select([], allow_blank=True, prompt="Preset", id="preset-select")
                yield Input(placeholder="Model route", id="model-input")
                yield Input(placeholder="Builder model (defaults to target)", id="builder-model-input")
                yield Input(placeholder="Judge model (optional)", id="judge-model-input")
                yield Input(placeholder="Run name", id="run-name-input")
                yield Label("Harnesses", classes="muted")
                yield SelectionList(id="harness-list")
                yield Label("Variants", classes="muted")
                yield SelectionList(id="variant-list")
                yield Label("Workloads", classes="muted")
                yield SelectionList(id="workload-list")
                with Horizontal():
                    yield Input(placeholder="Trials", type="integer", id="attempts-input")
                    yield Input(placeholder="Tasks", type="integer", id="tasks-input")
                    yield Input(placeholder="Concurrency", type="integer", id="concurrency-input")
                yield Input(placeholder="Tags, comma separated", id="tags-input")
                yield Select(
                    [("Full content", "full"), ("Metadata only", "metadata")],
                    value="full",
                    allow_blank=False,
                    id="trace-content-select",
                )
                yield Static(
                    "Full content sends prompts, responses, reasoning, and tool data to Weave.",
                    classes="warning",
                )
            with VerticalScroll(classes="summary-column"):
                with Collapsible(
                    title="Ask Fugue to plan an experiment",
                    collapsed=True,
                    id="composer-drawer",
                ):
                    yield Input(
                        placeholder="Describe the comparison you want to run",
                        id="composer-request",
                    )
                    yield Input(
                        placeholder="Composer model (defaults to active model)",
                        id="composer-model",
                    )
                    with Horizontal(classes="button-row"):
                        yield Button("Draft", id="compose-ai", variant="primary")
                        yield Button("Apply", id="apply-ai-draft")
                    yield Input(
                        placeholder="Experiment id for explicit save",
                        id="composer-save-id",
                    )
                    with Horizontal(classes="button-row"):
                        yield Button("Save draft", id="save-ai-draft")
                        yield Button("Run draft", id="run-ai-draft")
                    yield RichLog(id="composer-log", wrap=True, highlight=True)
                yield Label("MATRIX", classes="section-title")
                with Horizontal(classes="button-row"):
                    yield Button("Preview", id="preview", variant="default")
                    yield Button("Run", id="run-live", variant="primary")
                with Horizontal(classes="button-row"):
                    yield Button("Edit variant", id="edit-variant")
                    yield Button("Save as", id="save-experiment")
                yield Static("Select an experiment to preview its cells.", id="matrix-summary")
                yield Label("COMMANDS", classes="section-title")
                yield RichLog(id="command-preview", wrap=False, highlight=True)

    def _compose_runs(self) -> ComposeResult:
        with Vertical(classes="pane"):
            with Horizontal(classes="button-row"):
                yield Button("Refresh", id="refresh-runs")
                yield Button("All logs", id="all-run-logs")
                yield Button("Cancel", id="cancel-run", variant="warning")
                yield Button("Export", id="export-run")
                yield Button("Open Agents", id="open-agents", variant="primary")
            yield DataTable(id="runs-table", cursor_type="row", zebra_stripes=True)
            yield Label("CELLS", classes="section-title")
            yield DataTable(id="cells-table", cursor_type="row", zebra_stripes=True)
            yield Label("LOG", classes="section-title")
            yield RichLog(id="run-log", wrap=False, highlight=True, max_lines=5_000)

    def _compose_results(self) -> ComposeResult:
        with Vertical(classes="pane"):
            with Horizontal(classes="button-row"):
                yield Button("Refresh", id="refresh-results")
                yield Button("Open Agents", id="results-agents", variant="primary")
            yield Static("No results loaded.", id="result-summary")
            with Collapsible(
                title="Analyze experiments with Fugue",
                collapsed=True,
                id="analyst-drawer",
            ):
                yield Input(
                    placeholder="Ask a comparative question about experiments",
                    id="analysis-question",
                )
                with Horizontal():
                    yield Input(
                        placeholder="Filters: experiment_id=pilot,variant_id=baseline",
                        id="analysis-filters",
                    )
                    yield Select(
                        [("Hybrid", "hybrid"), ("Local", "local")],
                        value="hybrid",
                        allow_blank=False,
                        id="analysis-source",
                    )
                yield Input(
                    placeholder="Analyst model (defaults to active model)",
                    id="analysis-model",
                )
                with Horizontal(classes="button-row"):
                    yield Button("Resolve scope", id="analyze-ai", variant="primary")
                    yield Button("Generate report", id="generate-analysis")
                    yield Input(
                        placeholder="Saved analysis id",
                        id="analysis-save-id",
                    )
                    yield Button("Save analysis", id="save-analysis")
                yield Static("No analysis scope resolved.", id="analysis-scope")
                yield RichLog(id="analysis-report", wrap=True, highlight=True)
            yield DataTable(id="results-table", cursor_type="row", zebra_stripes=True)

    def _compose_setup(self) -> ComposeResult:
        with Vertical(classes="pane"):
            with Horizontal(classes="button-row"):
                yield Button("Run preflight", id="run-preflight", variant="primary")
                yield Button("Start bridge", id="start-bridge")
                yield Button("Open Agents", id="setup-agents")
            yield DataTable(id="setup-table", cursor_type="row")
            yield Label("OUTPUT", classes="section-title")
            yield RichLog(id="setup-log", wrap=True, highlight=True)

    def on_mount(self) -> None:
        self._load_experiment(self.experiment_id)
        if self.initial_draft is not None:
            self.call_after_refresh(self._apply_initial_draft)
        self._refresh_runs()
        self._refresh_results()
        self._refresh_setup()
        self.set_interval(1.0, self._poll_runs)

    def _apply_initial_draft(self) -> None:
        draft = self.initial_draft
        self.initial_draft = None
        if draft is not None:
            self._show_composer_draft(draft)
            self._apply_ai_draft()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "experiment-select" and event.value != Select.NULL:
            self.applied_draft = None
            self._load_experiment(str(event.value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "preview": self._preview,
            "run-live": self._launch,
            "edit-variant": self._edit_variant,
            "save-experiment": self._save_experiment,
            "compose-ai": self._compose_with_ai,
            "apply-ai-draft": self._apply_ai_draft,
            "save-ai-draft": self._save_ai_draft,
            "run-ai-draft": self._run_ai_draft,
            "refresh-runs": self._refresh_runs,
            "all-run-logs": self._show_all_logs,
            "cancel-run": self.action_cancel,
            "export-run": self.action_export,
            "open-agents": self.action_open_agents,
            "refresh-results": self._refresh_results,
            "results-agents": self.action_open_agents,
            "analyze-ai": self._analyze_with_ai,
            "generate-analysis": self._generate_analysis,
            "save-analysis": self._save_analysis,
            "run-preflight": self._run_preflight,
            "start-bridge": self._start_bridge,
            "setup-agents": self.action_open_agents,
        }
        action = actions.get(event.button.id or "")
        if action:
            action()

    def _edit_variant(self) -> None:
        self.push_screen(
            VariantEditorScreen(self.service, self.experiment_id),
            self._experiment_saved,
        )

    def _save_experiment(self) -> None:
        try:
            request = self._request()
            experiment = self.service.experiment(request.experiment_id)
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self.push_screen(
            SaveExperimentScreen(self.service, experiment, request),
            self._experiment_saved,
        )

    def _experiment_saved(self, experiment_id: str | None) -> None:
        if experiment_id is None:
            return
        selector = self.query_one("#experiment-select", Select)
        selector.set_options(self.service.experiment_items())
        selector.value = experiment_id
        self._load_experiment(experiment_id)
        self.notify(f"Saved {experiment_id}")

    def _compose_with_ai(self) -> None:
        request = self.query_one("#composer-request", Input).value.strip()
        if not request:
            self.notify("Describe the experiment you want to plan", severity="warning")
            return
        model = self.query_one("#composer-model", Input).value.strip() or None
        trace_content = str(self.query_one("#trace-content-select", Select).value)
        log = self.query_one("#composer-log", RichLog)
        log.clear()
        log.write("Grounding request in repository experiments and assets...")
        self._compose_ai_worker(request, model, trace_content)

    @work(thread=True, exclusive=True, group="composer")
    def _compose_ai_worker(
        self,
        request: str,
        model: str | None,
        trace_content: str,
    ) -> None:
        try:
            draft = asyncio.run(
                self.service.compose_experiment(
                    request,
                    base_experiment=self.experiment_id,
                    model=model,
                    trace_content=trace_content,
                )
            )
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            self.call_from_thread(
                self.query_one("#composer-log", RichLog).write,
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.call_from_thread(self._show_composer_draft, draft)

    def _show_composer_draft(self, draft: Any) -> None:
        self.composer_draft = draft
        log = self.query_one("#composer-log", RichLog)
        log.clear()
        log.write(f"{draft.experiment.title} [{draft.experiment.id}]")
        log.write(draft.rationale or "No rationale supplied.")
        log.write(
            f"{draft.preview.cells} cells / {draft.preview.estimated_trials} trials / "
            f"{draft.preview.applicable_cells} applicable"
        )
        if draft.assets:
            log.write("Assets: " + ", ".join(f"{item.kind}:{item.id}" for item in draft.assets))
        for warning in draft.warnings:
            log.write(f"WARNING: {warning}")
        if draft.diff:
            log.write(draft.diff)
        self.query_one("#composer-save-id", Input).value = f"{draft.experiment.id}-ai"
        self.notify("Draft validated. Apply, save, or run it explicitly.")

    def _apply_ai_draft(self) -> None:
        if self.composer_draft is None:
            self.notify("Create an AI draft first", severity="warning")
            return
        self.applied_draft = self.composer_draft
        self._apply_experiment(self.composer_draft.experiment)
        self.query_one("#matrix-summary", Static).update(
            f"{self.composer_draft.preview.cells} cells\n"
            f"{self.composer_draft.preview.estimated_trials} estimated trials\n"
            "AI draft applied locally; it is not saved or running."
        )
        self.notify("Draft applied to Plan without writing files")

    def _save_ai_draft(self) -> None:
        if self.composer_draft is None:
            self.notify("Create an AI draft first", severity="warning")
            return
        item_id = self.query_one("#composer-save-id", Input).value.strip()
        if not item_id:
            self.notify("Enter an experiment id", severity="warning")
            return
        from fugue.bench.ai import ExperimentComposer

        try:
            draft = replace(
                self.composer_draft,
                experiment=self._experiment_from_form(self.composer_draft.experiment),
            )
            saved = ExperimentComposer(self.service).save(draft, experiment_id=item_id)
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self._experiment_saved(saved.id)

    def _run_ai_draft(self) -> None:
        if self.composer_draft is None:
            self.notify("Create an AI draft first", severity="warning")
            return
        if self.composer_draft.experiment.trace_content == "full":
            self.notify(
                "Full AI and harness content will be sent to Weave",
                severity="warning",
                timeout=6,
            )
        if self.composer_draft.assets:
            self.notify(
                "Save the experiment and its proposed prompt or skill before running",
                severity="warning",
            )
            return
        try:
            experiment = self._experiment_from_form(self.composer_draft.experiment)
            run = self.service.launch(self._request(), experiment=experiment)
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self.selected_run_id = run.run_id
        self._refresh_runs()
        self.action_show_runs()
        self._show_run(run.run_id)

    def _analyze_with_ai(self) -> None:
        question = self.query_one("#analysis-question", Input).value.strip()
        if not question:
            self.notify("Ask a question about your experiments", severity="warning")
            return
        try:
            filters = _filters(self.query_one("#analysis-filters", Input).value)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        model = self.query_one("#analysis-model", Input).value.strip() or None
        source = str(self.query_one("#analysis-source", Select).value)
        report = self.query_one("#analysis-report", RichLog)
        report.clear()
        report.write("Planning and resolving a reproducible local scope...")
        self._analysis_scope_worker(question, filters, model, source)

    @work(thread=True, exclusive=True, group="analyst")
    def _analysis_scope_worker(
        self,
        question: str,
        filters: dict[str, str],
        model: str | None,
        source: str,
    ) -> None:
        try:
            spec = asyncio.run(
                self.service.plan_analysis(
                    question,
                    filters=filters,
                    model=model,
                    source=source,
                )
            )
            preview = self.service.prepare_analysis(spec)
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            self.call_from_thread(
                self.query_one("#analysis-report", RichLog).write,
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.call_from_thread(self._show_analysis_preview, preview)

    def _show_analysis_preview(self, preview: Any) -> None:
        self.analysis_preview = preview
        self.analysis_result = None
        scope = preview.scope
        self.query_one("#analysis-scope", Static).update(
            f"{len(scope.experiments)} experiments / "
            f"{len(scope.runs)} runs / {scope.rows} records / "
            f"{len(scope.tasks)} tasks\n"
            f"Models: {', '.join(scope.models)}  Sources: {', '.join(scope.sources)}"
        )
        report = self.query_one("#analysis-report", RichLog)
        report.clear()
        report.write("Scope resolved locally. Review it, then generate the report explicitly.")
        self.query_one("#analysis-save-id", Input).value = preview.spec.id
        self.notify("Analysis scope ready; no Weave query or report call has run yet")

    def _generate_analysis(self) -> None:
        if self.analysis_preview is None:
            self.notify("Resolve an analysis scope first", severity="warning")
            return
        self.query_one("#analysis-report", RichLog).write(
            "Enriching the confirmed scope and generating an evidence-backed report..."
        )
        self._analysis_report_worker(self.analysis_preview)

    @work(thread=True, exclusive=True, group="analyst")
    def _analysis_report_worker(self, preview: Any) -> None:
        try:
            result = asyncio.run(self.service.execute_analysis(preview))
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            self.call_from_thread(
                self.query_one("#analysis-report", RichLog).write,
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.call_from_thread(self._show_analysis, result)

    def _show_analysis(self, result: Any) -> None:
        self.analysis_result = result
        self.query_one("#analysis-scope", Static).update(
            f"{len(result.scope.experiments)} experiments / "
            f"{len(result.scope.runs)} runs / {result.scope.rows} records / "
            f"{len(result.scope.tasks)} tasks\n"
            f"Models: {', '.join(result.scope.models)}  "
            f"Sources: {', '.join(result.scope.sources)}"
        )
        log = self.query_one("#analysis-report", RichLog)
        log.clear()
        log.write(result.report)
        self.query_one("#analysis-save-id", Input).value = result.spec.id
        self.notify(f"Analysis saved to {result.report_dir}")

    def _save_analysis(self) -> None:
        if self.analysis_result is None:
            self.notify("Run an analysis first", severity="warning")
            return
        item_id = self.query_one("#analysis-save-id", Input).value.strip()
        if not item_id:
            self.notify("Enter an analysis id", severity="warning")
            return
        from fugue.bench.ai import save_analysis

        try:
            path = save_analysis(
                replace(self.analysis_result.spec, id=item_id),
                self.service.repo_root,
            )
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self.notify(f"Saved analysis definition to {path}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "runs-table":
            self.selected_run_id = str(event.row_key.value)
            self.selected_cell_id = None
            self._show_run(self.selected_run_id)
        elif event.data_table.id == "cells-table" and self.selected_run_id:
            self.selected_cell_id = str(event.row_key.value)
            self._show_run(self.selected_run_id)

    def action_show_compose(self) -> None:
        self.query_one("#workspace", TabbedContent).active = "compose"

    def action_show_runs(self) -> None:
        self.query_one("#workspace", TabbedContent).active = "runs"

    def action_show_results(self) -> None:
        self.query_one("#workspace", TabbedContent).active = "results"

    def action_show_setup(self) -> None:
        self.query_one("#workspace", TabbedContent).active = "setup"

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_run(self) -> None:
        if self.query_one("#workspace", TabbedContent).active == "compose":
            self._launch()

    def action_cancel(self) -> None:
        if not self.selected_run_id:
            self.notify("Select a running experiment first", severity="warning")
            return
        run = self.service.supervisor.cancel(self.selected_run_id)
        self.notify(f"{run.run_id}: {run.status}")
        self._refresh_runs()

    def action_export(self) -> None:
        if not self.selected_run_id:
            self.notify("Select a run to export", severity="warning")
            return
        self._export_worker(self.selected_run_id)

    @work(thread=True, exclusive=True, group="export")
    def _export_worker(self, run_id: str) -> None:
        try:
            summary = self.service.export_run(run_id, fetch_weave=True)
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            return
        self.call_from_thread(
            self.notify,
            f"Exported {summary.rows} rows to {summary.path}",
        )

    def action_open_agents(self) -> None:
        links = (
            self.service.run_links(self.selected_run_id)
            if self.selected_run_id
            else self.service.deep_links()
        )
        webbrowser.open(links.agents)
        self.notify("Opened Weave Agents")

    def action_open_trace(self) -> None:
        if self.selected_run_id:
            references = self.service.run_trace_refs(
                self.selected_run_id,
                cell_id=self.selected_cell_id,
            )
            conversation_id = next(
                (
                    value
                    for reference in references
                    for value in reference.conversation_ids
                ),
                None,
            )
            if conversation_id:
                self.copy_to_clipboard(conversation_id)
                self.notify(f"Copied conversation {conversation_id}")
        self.action_open_agents()

    def _load_experiment(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        experiment = self.service.experiment(experiment_id)
        self._apply_experiment(experiment)

    def _apply_experiment(self, experiment: ExperimentSpec) -> None:
        self.query_one("#model-input", Input).value = experiment.model or ""
        self.query_one("#builder-model-input", Input).value = experiment.builder_model or ""
        self.query_one("#judge-model-input", Input).value = experiment.judge_model or ""
        self.query_one("#run-name-input", Input).value = experiment.run_name or ""
        self.query_one("#attempts-input", Input).value = _number(experiment.n_attempts)
        self.query_one("#tasks-input", Input).value = _number(experiment.n_tasks)
        self.query_one("#concurrency-input", Input).value = _number(experiment.n_concurrent)
        self.query_one("#tags-input", Input).value = ",".join(experiment.tags)
        self.query_one("#trace-content-select", Select).value = experiment.trace_content
        self._replace_selection(
            "harness-list",
            [(label, value, value in experiment.harnesses) for label, value in HARNESS_LABELS],
        )
        self._replace_selection(
            "variant-list",
            [(variant.label, variant.id, variant.enabled) for variant in experiment.variants],
        )
        self._replace_selection(
            "workload-list",
            [(item.id, item.id, True) for item in experiment.workloads]
            or [("Manifest tasks", "harbor", False)],
        )
        presets = [(item.id, item.id) for item in experiment.presets]
        preset_select = self.query_one("#preset-select", Select)
        preset_select.set_options(presets)
        if presets:
            preset_select.value = experiment.default_preset or presets[0][1]
        else:
            preset_select.clear()
        self._render_variant_reference(experiment)

    def _experiment_from_form(self, experiment: ExperimentSpec) -> ExperimentSpec:
        from fugue.bench.library import experiment_from_data

        request = self._request()
        selected_variants = set(request.variants)
        data = experiment.to_dict()
        data.update(
            {
                "model": request.model or experiment.model,
                "builder_model": request.builder_model,
                "judge_model": request.judge_model,
                "run_name": request.run_name,
                "tags": list(request.tags),
                "harnesses": list(request.harnesses),
                "n_attempts": request.n_attempts,
                "n_tasks": request.n_tasks,
                "n_concurrent": request.n_concurrent,
                "trace_content": request.trace_content,
                "variants": [
                    {**variant.to_dict(), "enabled": variant.id in selected_variants}
                    for variant in experiment.variants
                ],
            }
        )
        return experiment_from_data(data, item_id=experiment.id)

    def _render_variant_reference(self, experiment: ExperimentSpec) -> None:
        log = self.query_one("#command-preview", RichLog)
        log.clear()
        log.write("Variants")
        for variant in experiment.variants:
            prompt = variant.prompt_id or "default prompt"
            skills = ", ".join(variant.skill_ids) or "no skills"
            log.write(
                f"  {variant.id:<20} {prompt:<22} {skills:<28} {variant.context.system_id}"
            )

    def _replace_selection(
        self, widget_id: str, options: list[tuple[str, str, bool]]
    ) -> None:
        widget = self.query_one(f"#{widget_id}", SelectionList)
        widget.clear_options()
        for label, value, selected in options:
            widget.add_option((label, value, selected))

    def _request(self) -> ExperimentRequest:
        return ExperimentRequest(
            experiment_id=str(self.query_one("#experiment-select", Select).value),
            preset=_select_value(self.query_one("#preset-select", Select)),
            workloads=tuple(self.query_one("#workload-list", SelectionList).selected),
            harnesses=tuple(self.query_one("#harness-list", SelectionList).selected),
            variants=tuple(self.query_one("#variant-list", SelectionList).selected),
            model=self.query_one("#model-input", Input).value or None,
            builder_model=self.query_one("#builder-model-input", Input).value or None,
            judge_model=self.query_one("#judge-model-input", Input).value or None,
            n_attempts=_positive_input(self.query_one("#attempts-input", Input)),
            n_tasks=_positive_input(self.query_one("#tasks-input", Input)),
            n_concurrent=_positive_input(self.query_one("#concurrency-input", Input)),
            run_name=self.query_one("#run-name-input", Input).value or None,
            tags=tuple(_csv(self.query_one("#tags-input", Input).value)),
            trace_content=str(self.query_one("#trace-content-select", Select).value),
        )

    def _preview(self) -> None:
        try:
            request = self._request()
            experiment = (
                self._experiment_from_form(self.applied_draft.experiment)
                if self.applied_draft is not None
                else None
            )
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self._preview_worker(request, experiment)

    @work(thread=True, exclusive=True, group="preview")
    def _preview_worker(
        self,
        request: ExperimentRequest,
        experiment: ExperimentSpec | None = None,
    ) -> None:
        try:
            preview = (
                self.service.preview_experiment(experiment, request=request)
                if experiment is not None
                else self.service.preview(request)
            )
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            return
        self.call_from_thread(self._apply_preview, request, preview)

    def _apply_preview(self, request: ExperimentRequest, preview: Any) -> None:
        self.last_preview = request
        summary = (
            f"{preview.cells} cells\n"
            f"{preview.estimated_trials} estimated trials\n"
            f"{len(preview.harnesses)} harnesses / {len(preview.variants)} variants\n"
            f"{', '.join(preview.workloads)}\n\n"
            f"{preview.applicable_cells} ready / "
            f"{preview.cells - preview.applicable_cells} not applicable"
        )
        self.query_one("#matrix-summary", Static).update(summary)
        log = self.query_one("#command-preview", RichLog)
        log.clear()
        for command in preview.commands:
            log.write(f"$ {command}")

    def _launch(self) -> None:
        request = self._request()
        if request.trace_content == "full":
            self.notify(
                "Full prompts, responses, reasoning, and tool data will be sent to Weave",
                severity="warning",
                timeout=6,
            )
        try:
            if self.applied_draft is not None and self.applied_draft.assets:
                raise ValueError(
                    "save the experiment and its proposed prompt or skill before running"
                )
            run = (
                self.service.launch(
                    request,
                    experiment=self._experiment_from_form(
                        self.applied_draft.experiment
                    ),
                )
                if self.applied_draft is not None
                else self.service.launch(request)
            )
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self.selected_run_id = run.run_id
        self._refresh_runs()
        self.action_show_runs()
        self._show_run(run.run_id)

    def _refresh_runs(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Run", "Experiment", "Status", "Pass", "Fail", "Waiting")
        runs = self.service.runs()
        for run in runs:
            table.add_row(
                run.run_id,
                run.experiment_id,
                run.status,
                str(run.passed),
                str(run.failed),
                str(run.pending),
                key=run.run_id,
            )
        if runs and self.selected_run_id is None:
            self.selected_run_id = runs[0].run_id
            self._show_run(runs[0].run_id)
        self._update_sequencer(runs)

    def _poll_runs(self) -> None:
        self._refresh_runs()
        if self.selected_run_id:
            self._show_run(self.selected_run_id)

    def _show_run(self, run_id: str) -> None:
        run = self.service.run_summary(run_id)
        cells = self.query_one("#cells-table", DataTable)
        cells.clear(columns=True)
        cells.add_columns("Harness", "Variant", "Context", "Task", "Status", "Time")
        for cell in run.cells:
            cells.add_row(
                cell.harness,
                cell.variant_id,
                cell.context_system_id,
                cell.task_id,
                cell.status.replace("_", " "),
                f"{cell.wall_time_sec:.1f}s" if cell.wall_time_sec is not None else "-",
                key=cell.cell_id,
            )
        log = self.query_one("#run-log", RichLog)
        target = (run_id, self.selected_cell_id)
        if target != self._log_target:
            self._log_target = target
            self._log_offset = 0
            log.clear()
        text, self._log_offset = self.service.supervisor.read_log_chunk(
            run_id,
            cell_id=self.selected_cell_id,
            offset=self._log_offset,
        )
        if text:
            log.write(text)
        elif not log.lines:
            label = self.selected_cell_id or "combined run"
            log.write(f"Waiting for {label} output...")

    def _show_all_logs(self) -> None:
        self.selected_cell_id = None
        if self.selected_run_id:
            self._show_run(self.selected_run_id)

    def _update_sequencer(self, runs: list[RunSummary]) -> None:
        statuses: dict[str, str] = {}
        selected = next((run for run in runs if run.run_id == self.selected_run_id), None)
        if selected:
            grouped: dict[str, list[str]] = defaultdict(list)
            for cell in selected.cells:
                grouped[cell.harness].append(cell.status)
            for harness, values in grouped.items():
                statuses[harness] = _aggregate_status(values)
        self.query_one(PixelSequencer).set_statuses(statuses)

    def _refresh_results(self) -> None:
        result = self.service.results()
        rate = f"{result.pass_rate:.1%}" if result.pass_rate is not None else "N/A"
        reward = (
            f"{result.average_reward:.3f} reward"
            if result.average_reward is not None
            else "N/A reward"
        )
        latency = (
            f"{result.average_wall_time_sec:.1f}s avg"
            if result.average_wall_time_sec is not None
            else "N/A latency"
        )
        self.query_one("#result-summary", Static).update(
            f"{result.total} trials   {rate} pass rate   {reward}   {latency}\n"
            f"${result.cost_usd:.4f}   "
            f"{result.input_tokens:,} in / {result.output_tokens:,} out tokens   "
            f"{result.turns} turns   {result.tool_calls} tools   "
            f"{sum(len(item.conversation_ids) for item in result.agent_traces)} conversations"
        )
        table = self.query_one("#results-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "Harness",
            "Experiment",
            "Variant",
            "Context",
            "Model",
            "Trials",
            "Pass rate",
            "Reward",
            "Time",
            "Cost",
            "Tokens",
            "Tools",
            "Failures",
            "Conversations",
        )
        groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in result.rows:
            key = (
                str(row.get("harness") or "unknown"),
                str(row.get("experiment_id") or "unknown"),
                str(row.get("variant_id") or "baseline"),
                str(row.get("context_system_id") or "none"),
                str(row.get("model") or "unknown"),
            )
            groups[key].append(row)
        for (harness, experiment, variant, context, model), rows in sorted(groups.items()):
            scored = [row for row in rows if row.get("pass") is not None]
            passed = sum(row.get("pass") is True for row in rows)
            pass_rate = f"{passed / len(scored):.1%}" if scored else "N/A"
            rewards = [
                value
                for row in rows
                if (value := _optional_float(row.get("reward"))) is not None
            ]
            times = [
                value
                for row in rows
                if (value := _optional_float(row.get("wall_time_sec"))) is not None
            ]
            cost = sum(_optional_float(row.get("cost_usd")) or 0 for row in rows)
            tokens = sum(
                int(row.get("n_input_tokens") or 0) + int(row.get("n_output_tokens") or 0)
                for row in rows
            )
            conversations = {
                value
                for row in rows
                for value in (row.get("weave_conversation_ids") or [])
            }
            table.add_row(
                harness,
                experiment,
                variant,
                context,
                model,
                str(len(rows)),
                pass_rate,
                f"{sum(rewards) / len(rewards):.3f}" if rewards else "N/A",
                f"{sum(times) / len(times):.1f}s" if times else "N/A",
                f"${cost:.4f}",
                f"{tokens:,}",
                str(sum(int(row.get("weave_tool_call_count") or 0) for row in rows)),
                str(sum(bool(row.get("exception_class")) or row.get("pass") is False for row in rows)),
                str(len(conversations)),
            )

    def _refresh_setup(self) -> None:
        try:
            status = self.service.status(self._request())
        except Exception as exc:
            self.query_one("#setup-log", RichLog).write(str(exc))
            return
        table = self.query_one("#setup-table", DataTable)
        table.clear(columns=True)
        table.add_columns("System", "State", "Detail")
        rows = [
            (
                f"{route.role.title()} model",
                route.key_present,
                f"{route.model} / {route.key_env}",
            )
            for route in status.routes
        ]
        rows.extend(
            (
                ("Weave", status.trace_key_present, status.trace_project),
                ("Docker", status.docker_present, "Container runtime"),
                ("Harbor", status.harbor_present, "Agent evaluation runner"),
                ("Bridge", status.bridge_ready, "127.0.0.1:4000"),
                (
                    "Context",
                    True,
                    f"{len(status.selected_context_systems)} selected / "
                    f"{status.context_system_count} available / "
                    f"{status.context_cache_entries} cached",
                ),
            )
        )
        for name, ready, detail in rows:
            table.add_row(name, "ready" if ready else "missing", detail)
        table.add_row(
            "Trace content",
            status.trace_content,
            "Full capture has no automatic PII scrubbing" if status.trace_content == "full" else "Structure and usage only",
        )

    def _run_preflight(self) -> None:
        request = self._request()
        self.query_one("#setup-log", RichLog).write("Running observational preflight...")
        self._preflight_worker(request)

    @work(thread=True, exclusive=True, group="setup")
    def _preflight_worker(self, request: ExperimentRequest) -> None:
        try:
            checks = self.service.preflight(request, live=True)
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            return
        self.call_from_thread(self._show_preflight, checks)

    def _show_preflight(self, checks: tuple[Any, ...]) -> None:
        log = self.query_one("#setup-log", RichLog)
        log.clear()
        for check in checks:
            log.write(f"{'ready' if check.ok else 'missing':<8} {check.name}: {check.detail}")
        self.notify(
            "Preflight passed" if all(check.ok for check in checks) else "Preflight found missing setup",
            severity="information" if all(check.ok for check in checks) else "warning",
        )
        self._refresh_setup()

    def _start_bridge(self) -> None:
        request = self._request()
        self.query_one("#setup-log", RichLog).write("Starting the local model bridge...")
        self._bridge_worker(request)

    @work(thread=True, exclusive=True, group="setup")
    def _bridge_worker(self, request: ExperimentRequest) -> None:
        try:
            files = self.service.start_bridge(request)
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            return
        self.call_from_thread(self._show_bridge_started, files.runtime_dir)

    def _show_bridge_started(self, runtime_dir: Any) -> None:
        self.query_one("#setup-log", RichLog).write(f"Bridge running from {runtime_dir}")
        self.notify("Bridge started")
        self._refresh_setup()


def run_tui(
    *,
    initial_screen: str = "compose",
    experiment_id: str = "pilot",
    service: OperatorService | None = None,
    initial_draft: Any = None,
) -> None:
    FugueApp(
        service=service,
        initial_screen=initial_screen,
        experiment_id=experiment_id,
        initial_draft=initial_draft,
    ).run()


def _animation_enabled() -> bool:
    return not (
        os.environ.get("FUGUE_NO_ANIMATION") == "1"
        or "NO_COLOR" in os.environ
        or not sys.stdout.isatty()
    )


def _number(value: int | None) -> str:
    return "" if value is None else str(value)


def _positive_input(widget: Input) -> int | None:
    if not widget.value.strip():
        return None
    value = int(widget.value)
    if value < 1:
        raise ValueError(f"{widget.placeholder or 'value'} must be positive")
    return value


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _filters(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _csv(value):
        if "=" not in item:
            raise ValueError(f"analysis filter must be FIELD=VALUE: {item}")
        key, selected = item.split("=", 1)
        if not key.strip() or not selected.strip():
            raise ValueError(f"analysis filter must be FIELD=VALUE: {item}")
        result[key.strip()] = selected.strip()
    return result


def _select_value(widget: Select) -> str | None:
    return None if widget.value == Select.NULL else str(widget.value)


def _aggregate_status(values: list[str]) -> str:
    for status in ("running", "failed", "interrupted", "cancelled", "pending"):
        if status in values:
            return status
    if values and all(value == "not_applicable" for value in values):
        return "not_applicable"
    if values and all(value == "passed" for value in values):
        return "passed"
    return values[0] if values else "idle"


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
