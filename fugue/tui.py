from __future__ import annotations

import asyncio
import os
import sys
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
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
    ContentSwitcher,
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
from fugue.bench.evaluations import evaluation_asset_path
from fugue.bench.library import (
    ContextSelection,
    ExperimentSpec,
    FeatureVariant,
    list_prompts,
    list_skills,
    validate_id,
)
from fugue.bench.operator import (
    ExperimentRequest,
    OperatorService,
    OperatorStatus,
    PreviewSummary,
    RunSummary,
)
from fugue.bench.sources import list_skill_source_ids

HARNESS_LABELS = (
    ("Hermes", "hermes"),
    ("OpenClaw", "openclaw"),
    ("Claude Code", "claude-code"),
    ("Codex", "codex"),
)
HARNESS_NAMES = dict((value, label) for label, value in HARNESS_LABELS)
CUSTOM_SIZE = "__custom__"
TERMINAL_RUN_STATES = {"passed", "failed", "cancelled", "interrupted"}
ACTIVE_RUN_STATES = {"starting", "running"}


@dataclass(frozen=True)
class PlanState:
    base_experiment_id: str
    experiment: ExperimentSpec
    request: ExperimentRequest
    assets: tuple[Any, ...] = ()
    proposal: Any = None
    preview: PreviewSummary | None = None
    dirty: bool = False


class PixelSequencer(Static):
    DEFAULT_CSS = """
    PixelSequencer {
        height: 6;
        padding: 0 2;
        background: #1A1C1F;
        border-bottom: solid #363B44;
    }
    PixelSequencer.compact { height: 2; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.phase = 0
        self.statuses: dict[str, str] = {}
        self.compact = True

    def on_mount(self) -> None:
        if _animation_enabled():
            self.set_interval(0.18, self._tick)
        self._draw()

    def set_statuses(self, statuses: dict[str, str]) -> None:
        self.statuses = statuses
        self._draw()

    def set_compact(self, compact: bool) -> None:
        self.compact = compact
        self.set_class(compact, "compact")
        self._draw()

    def _tick(self) -> None:
        self.phase = (self.phase + 1) % 18
        self._draw()

    def _draw(self) -> None:
        output = Text()
        if self.compact:
            for index, (label, harness) in enumerate(HARNESS_LABELS):
                status = self.statuses.get(harness, "idle")
                output.append("  " if index else "")
                output.append(label.upper(), style="#D1D5DB")
                output.append(" " + _status_glyph(status), style=_status_color(status))
            self.update(output)
            return
        for label, harness in HARNESS_LABELS:
            status = self.statuses.get(harness, "idle")
            color = _status_color(status)
            output.append(f"{label.upper():<12}", style="#D1D5DB")
            for index in range(18):
                if index == self.phase and status == "running":
                    output.append("■", style="bold #FFCC33")
                elif index < self.phase and status == "running":
                    output.append("▪", style="#00AFC2")
                elif status in TERMINAL_RUN_STATES:
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
                "n Next        p Previous   r Review / run\n"
                "/ Commands    c Cancel     e Export\n"
                "a Agents      w Trace      ? Help\n\n"
                "Plan defines the comparison. Runs operates it. Weave Agents "
                "explains the conversations, model calls, and tool use."
            )
            yield Button("Close", id="close-help", variant="primary")

    def on_button_pressed(self, _event: Button.Pressed) -> None:
        self.dismiss()


class VariantEditorScreen(ModalScreen[FeatureVariant | None]):
    DEFAULT_CSS = """
    VariantEditorScreen { align: center middle; background: #000000 60%; }
    #variant-editor {
        width: 90%;
        max-width: 76;
        height: 90%;
        max-height: 38;
        padding: 1 2;
        border: solid #FFCC33;
        background: #1A1C1F;
    }
    #variant-skills { height: 9; }
    """

    def __init__(
        self,
        service: OperatorService,
        variant: FeatureVariant,
        *,
        existing_ids: set[str],
        new: bool,
    ) -> None:
        super().__init__()
        self.service = service
        self.variant = variant
        self.existing_ids = existing_ids
        self.new = new

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="variant-editor"):
            yield Label(
                "ADD VARIANT" if self.new else "EDIT VARIANT",
                classes="section-title",
            )
            yield Label("Name", classes="muted")
            yield Input(
                value=self.variant.id,
                placeholder="variant-id",
                id="edit-variant-id",
                disabled=not self.new,
            )
            yield Input(
                value=self.variant.label,
                placeholder="Display label",
                id="edit-variant-label",
            )
            yield Label("Prompt", classes="muted")
            yield Select(
                [
                    ("Default prompt", ""),
                    *[
                        (item.title, item.id)
                        for item in list_prompts(self.service.repo_root)
                    ],
                ],
                value=self.variant.prompt_id or "",
                allow_blank=False,
                id="edit-prompt-select",
            )
            yield Label("Skills", classes="muted")
            yield SelectionList(
                *[
                    (
                        item.title,
                        item.id,
                        item.id in self.variant.selected_skill_ids,
                    )
                    for item in list_skills(self.service.repo_root)
                ],
                *[
                    (
                        f"{item_id} (remote)",
                        item_id,
                        item_id in self.variant.selected_skill_ids,
                    )
                    for item_id in list_skill_source_ids(self.service.repo_root)
                ],
                id="variant-skills",
            )
            yield Label("Context", classes="muted")
            yield Select(
                [
                    (item.title, item.id)
                    for item in list_context_systems(self.service.repo_root)
                ],
                value=self.variant.context.system_id,
                allow_blank=False,
                id="edit-context-select",
            )
            with Collapsible(title="Advanced agent configuration", collapsed=True):
                yield Static(_variant_advanced_summary(self.variant), classes="muted")
            with Horizontal(classes="button-row"):
                yield Button("Use variant", id="save-variant", variant="primary")
                yield Button("Cancel", id="cancel-variant")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-variant":
            self.dismiss(None)
            return
        if event.button.id != "save-variant":
            return
        try:
            variant_id = validate_id(
                self.query_one("#edit-variant-id", Input).value,
                kind="variant id",
            )
            if self.new and variant_id in self.existing_ids:
                raise ValueError(f"variant already exists: {variant_id}")
            label = self.query_one("#edit-variant-label", Input).value.strip()
            if not label:
                raise ValueError("variant label is required")
            prompt_value = self.query_one("#edit-prompt-select", Select).value
            context_value = str(
                self.query_one("#edit-context-select", Select).value
            )
            context = self.variant.context
            if context.system_id != context_value:
                context = ContextSelection(system_id=context_value)
            updated = replace(
                self.variant,
                id=variant_id,
                label=label,
                prompt_id=str(prompt_value) or None,
                skills=list(
                    self.query_one("#variant-skills", SelectionList).selected
                ),
                skill_ids=[],
                context=context,
                enabled=True,
            )
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self.dismiss(updated)


class SaveExperimentScreen(ModalScreen[tuple[str, str] | None]):
    DEFAULT_CSS = """
    SaveExperimentScreen { align: center middle; background: #000000 60%; }
    #save-experiment-panel {
        width: 90%;
        max-width: 68;
        height: 18;
        padding: 1 2;
        border: solid #FFCC33;
        background: #1A1C1F;
    }
    """

    def __init__(self, experiment: ExperimentSpec) -> None:
        super().__init__()
        self.experiment = experiment

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
                "Writes the accepted comparison to configs/fugue/experiments.",
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
        title = self.query_one("#save-experiment-title", Input).value.strip()
        try:
            validate_id(item_id, kind="experiment id")
            if not title:
                raise ValueError("experiment title is required")
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self.dismiss((item_id, title))


class ConfirmRunScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmRunScreen { align: center middle; background: #000000 60%; }
    #confirm-run-panel {
        width: 90%;
        max-width: 68;
        height: 15;
        padding: 1 2;
        border: solid #F59E0B;
        background: #1A1C1F;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-run-panel"):
            yield Label("FULL TRACE CONTENT", classes="warning")
            yield Static(
                "Prompts, responses, reasoning, tool arguments, and tool results "
                "will be sent to the configured Weave project. Harness plugins do "
                "not provide automatic PII scrubbing."
            )
            with Horizontal(classes="button-row"):
                yield Button("Run experiment", id="confirm-run", variant="primary")
                yield Button("Cancel", id="cancel-run-confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-run")


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
    TabbedContent { height: 1fr; margin-bottom: 1; }
    TabPane { padding: 0; }
    Tabs { background: $panel; color: #D1D5DB; }
    Tab.-active { color: $gold; text-style: bold; }
    .pane { padding: 1 2; height: 1fr; }
    .plan-pane { padding: 0 2 1 2; height: 1fr; }
    .plan-step { height: 1fr; padding: 1 0; }
    .section-title { color: $gold; text-style: bold; margin-bottom: 1; }
    .muted { color: #9CA3AF; }
    .warning { color: #F59E0B; }
    .success { color: #22C55E; }
    .hidden { display: none; }
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
    #harness-list, #workload-list { height: 6; }
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
    Button { margin-right: 1; min-width: 12; }
    Button.-primary { background: $gold; color: #171A1F; }
    Button:focus { text-style: bold; }
    .button-row { height: 3; margin: 1 0; }
    #plan-progress { height: 2; padding: 0 1; color: #D1D5DB; }
    #define-summary, #proposal-summary, #compare-sentence, #preview-status,
    #review-summary, #review-warnings, #analysis-scope, #result-summary {
        background: $panel;
        padding: 1 2;
        margin-bottom: 1;
    }
    #proposal-panel { height: auto; border: solid #00AFC2; padding: 1; }
    #variant-table { height: 9; }
    #review-matrix { height: 12; }
    #execution-details { min-height: 4; }
    #execution-log { height: 8; }
    #runs-table { height: 10; }
    #cells-table { height: 12; }
    #run-log { height: 1fr; }
    #result-summary { height: 5; }
    #analysis-report { height: 10; }
    #setup-table { height: 10; }
    #setup-log { height: 8; }
    Footer { background: $panel; color: #9CA3AF; }
    """
    BINDINGS = [
        Binding("1", "show_compose", "Plan", show=True),
        Binding("2", "show_runs", "Runs", show=True),
        Binding("3", "show_results", "Results", show=True),
        Binding("4", "show_setup", "Setup", show=True),
        Binding("n", "next_step", "Next", show=False),
        Binding("p", "previous_step", "Previous", show=False),
        Binding("/", "command_palette", "Commands", show=False),
        Binding("?", "help", "Help", show=False),
        Binding("r", "run", "Review / run", show=False),
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
        options = self.service.experiment_items()
        available = {value for _, value in options}
        selected_id = experiment_id if experiment_id in available else options[0][1]
        if initial_draft is not None:
            experiment = initial_draft.experiment
            self.plan = PlanState(
                base_experiment_id=selected_id,
                experiment=experiment,
                request=self.service.request_for_experiment(experiment),
                assets=tuple(initial_draft.assets),
                preview=initial_draft.preview,
                dirty=True,
            )
            self.plan_step = "compare-step"
        else:
            experiment = self.service.experiment(selected_id)
            self.plan = PlanState(
                base_experiment_id=selected_id,
                experiment=experiment,
                request=self.service.request_for_experiment(experiment),
            )
            self.plan_step = "define-step"
        self.initial_screen = initial_screen
        self.experiment_id = experiment.id
        self.selected_variant_id = experiment.variants[0].id
        self.selected_run_id: str | None = None
        self.selected_cell_id: str | None = None
        self.analysis_preview: Any = None
        self.analysis_result: Any = None
        self._syncing = False
        self._sync_generation = 0
        self._preview_generation = 0
        self._preview_timer: Any = None
        self._review_blockers: tuple[str, ...] = ()
        self._log_target: tuple[str, str | None] | None = None
        self._log_offset = 0

    def compose(self) -> ComposeResult:
        yield Static("FUGUE  /  AGENT EXPERIMENT OPERATOR", id="masthead")
        yield PixelSequencer()
        with TabbedContent(initial=self.initial_screen, id="workspace"):
            with TabPane("Plan", id="compose"):
                yield from self._compose_plan()
            with TabPane("Runs", id="runs"):
                yield from self._compose_runs()
            with TabPane("Results", id="results"):
                yield from self._compose_results()
            with TabPane("Setup", id="setup"):
                yield from self._compose_setup()
        yield Footer()

    def _compose_plan(self) -> ComposeResult:
        yield Static(id="plan-progress")
        with ContentSwitcher(initial=self.plan_step, id="plan-steps"):
            with VerticalScroll(id="define-step", classes="plan-step plan-pane"):
                yield Label("WHAT DO YOU WANT TO LEARN?", classes="section-title")
                yield Input(
                    placeholder="What do you want to compare?",
                    id="composer-request",
                )
                with Horizontal(classes="button-row"):
                    yield Button("Create plan", id="compose-ai", variant="primary")
                yield Label("OR LOAD A SAVED EXPERIMENT", classes="section-title")
                yield Select(
                    self.service.experiment_items(),
                    value=self.plan.base_experiment_id,
                    allow_blank=False,
                    id="experiment-select",
                )
                yield Static(id="define-summary")
                with Vertical(id="proposal-panel", classes="hidden"):
                    yield Static(id="proposal-summary")
                    with Horizontal(classes="button-row"):
                        yield Button("Use proposal", id="use-proposal", variant="primary")
                        yield Button("Refine", id="refine-proposal")
                        yield Button("Discard", id="discard-proposal")
                with Horizontal(classes="button-row"):
                    yield Button("Continue", id="define-next", variant="primary")
            with VerticalScroll(id="compare-step", classes="plan-step plan-pane"):
                yield Label("BUILD THE COMPARISON", classes="section-title")
                yield Static(id="compare-sentence")
                yield Label("Variants", classes="muted")
                yield DataTable(
                    id="variant-table",
                    cursor_type="row",
                    zebra_stripes=True,
                )
                with Horizontal(classes="button-row"):
                    yield Button("Add", id="add-variant")
                    yield Button("Duplicate", id="duplicate-variant")
                    yield Button("Edit", id="edit-variant")
                    yield Button("Remove", id="remove-variant")
                    yield Button("Enable / disable", id="toggle-variant")
                yield Label("Harnesses", classes="muted")
                yield SelectionList(id="harness-list")
                yield Label("Evaluation coverage", classes="muted")
                yield SelectionList(id="workload-list")
                yield Static(id="workload-default", classes="muted")
                yield Label("Run size", classes="muted")
                yield Select(
                    [("Custom", CUSTOM_SIZE)],
                    value=CUSTOM_SIZE,
                    allow_blank=False,
                    id="run-size-select",
                )
                with Horizontal(id="custom-size-row", classes="hidden"):
                    yield Input(
                        placeholder="Task limit",
                        type="integer",
                        id="tasks-input",
                    )
                    yield Input(
                        placeholder="Trials per cell",
                        type="integer",
                        id="attempts-input",
                    )
                with Collapsible(title="Advanced", collapsed=True, id="plan-advanced"):
                    yield Input(placeholder="Target model", id="model-input")
                    yield Input(
                        placeholder="Builder model (inherits target)",
                        id="builder-model-input",
                    )
                    yield Input(
                        placeholder="Judge model (optional)",
                        id="judge-model-input",
                    )
                    yield Input(placeholder="Run name", id="run-name-input")
                    yield Input(
                        placeholder="Concurrency",
                        type="integer",
                        id="concurrency-input",
                    )
                    yield Input(placeholder="Tags, comma separated", id="tags-input")
                    yield Select(
                        [("Full content", "full"), ("Metadata only", "metadata")],
                        value="full",
                        allow_blank=False,
                        id="trace-content-select",
                    )
                    yield Static(
                        "Other Harbor settings inherit from the checked-in experiment.",
                        classes="muted",
                    )
                yield Static("Preparing preview...", id="preview-status")
                with Horizontal(classes="button-row"):
                    yield Button("Back", id="compare-back")
                    yield Button("Generate evaluation", id="generate-evaluation")
                    yield Button("Review", id="compare-next", variant="primary")
            with VerticalScroll(id="review-step", classes="plan-step plan-pane"):
                yield Label("REVIEW THE RUN", classes="section-title")
                yield Static(id="review-summary")
                yield DataTable(id="review-matrix", zebra_stripes=True)
                yield Static(id="review-warnings")
                with Collapsible(
                    title="Execution details",
                    collapsed=True,
                    id="execution-details",
                ):
                    yield RichLog(id="execution-log", wrap=False, highlight=True)
                with Horizontal(classes="button-row"):
                    yield Button("Back", id="review-back")
                    yield Button("Save experiment", id="save-experiment")
                    yield Button("Open Setup", id="review-setup")
                    yield Button(
                        "Run experiment",
                        id="run-live",
                        variant="primary",
                        disabled=True,
                    )

    def _compose_runs(self) -> ComposeResult:
        with Vertical(classes="pane"):
            with Horizontal(classes="button-row"):
                yield Button("Combined log", id="all-run-logs")
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
                yield Button("Open Agents", id="results-agents", variant="primary")
            yield Static("No results loaded.", id="result-summary")
            yield Label("ASK FUGUE ABOUT THESE RESULTS", classes="section-title")
            yield Input(
                placeholder="Which variant worked best, and why?",
                id="analysis-question",
            )
            with Collapsible(title="Analysis settings", collapsed=True):
                yield Select(
                    [("Hybrid local + Weave", "hybrid"), ("Local only", "local")],
                    value="hybrid",
                    allow_blank=False,
                    id="analysis-source",
                )
                yield Input(
                    placeholder="Analyst model (inherits active model)",
                    id="analysis-model",
                )
            with Horizontal(classes="button-row"):
                yield Button("Resolve scope", id="analyze-ai", variant="primary")
                yield Button(
                    "Generate report",
                    id="generate-analysis",
                    disabled=True,
                )
            yield Static("No analysis scope resolved.", id="analysis-scope")
            yield RichLog(id="analysis-report", wrap=True, highlight=True)
            yield DataTable(id="results-table", cursor_type="row", zebra_stripes=True)

    def _compose_setup(self) -> ComposeResult:
        with Vertical(classes="pane"):
            with Horizontal(classes="button-row"):
                yield Button("Check setup", id="run-preflight", variant="primary")
                yield Button("Start bridge", id="start-bridge")
                yield Button("Prepare context", id="prepare-context")
                yield Button("Open Agents", id="setup-agents")
            yield DataTable(id="setup-table", cursor_type="row")
            with Collapsible(title="Details", collapsed=True):
                yield Static(id="setup-details")
            with Collapsible(title="Operation output", collapsed=True):
                yield RichLog(id="setup-log", wrap=True, highlight=True)

    def on_mount(self) -> None:
        self._render_plan()
        self._refresh_runs()
        self._refresh_results()
        self._refresh_setup()
        self.set_interval(1.0, self._poll_runs)
        self._queue_preview()

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._syncing:
            return
        if event.select.id == "experiment-select" and event.value != Select.NULL:
            if str(event.value) != self.plan.base_experiment_id:
                self._load_experiment(str(event.value))
        elif event.select.id == "run-size-select":
            self._set_run_size(str(event.value))
        elif event.select.id == "trace-content-select":
            self._update_request(trace_content=str(event.value))

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        if self._syncing:
            return
        if event.selection_list.id == "harness-list":
            self._update_request(
                harnesses=tuple(event.selection_list.selected),
                dirty=True,
            )
        elif event.selection_list.id == "workload-list":
            self._update_request(
                workloads=tuple(event.selection_list.selected),
                dirty=True,
            )

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._syncing:
            return
        values: dict[str, Any] = {}
        if event.input.id == "model-input":
            values["model"] = event.value.strip() or None
        elif event.input.id == "builder-model-input":
            values["builder_model"] = event.value.strip() or None
        elif event.input.id == "judge-model-input":
            values["judge_model"] = event.value.strip() or None
        elif event.input.id == "run-name-input":
            values["run_name"] = event.value.strip() or None
        elif event.input.id == "concurrency-input":
            values["n_concurrent"] = _optional_positive(event.value, "Concurrency")
        elif event.input.id == "tags-input":
            values["tags"] = tuple(_csv(event.value))
        elif event.input.id == "tasks-input":
            values["n_tasks"] = _optional_positive(event.value, "Task limit")
        elif event.input.id == "attempts-input":
            values["n_attempts"] = _optional_positive(
                event.value, "Trials per cell"
            )
        if values:
            self._update_request(**values, dirty=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "compose-ai": self._compose_with_ai,
            "use-proposal": self._use_proposal,
            "refine-proposal": self._compose_with_ai,
            "discard-proposal": self._discard_proposal,
            "define-next": lambda: self._show_plan_step("compare-step"),
            "compare-back": lambda: self._show_plan_step("define-step"),
            "compare-next": lambda: self._show_plan_step("review-step"),
            "generate-evaluation": self._generate_evaluation,
            "review-back": lambda: self._show_plan_step("compare-step"),
            "add-variant": self._add_variant,
            "duplicate-variant": self._duplicate_variant,
            "edit-variant": self._edit_variant,
            "remove-variant": self._remove_variant,
            "toggle-variant": self._toggle_variant,
            "save-experiment": self._save_experiment,
            "run-live": self._request_launch,
            "review-setup": self.action_show_setup,
            "all-run-logs": self._show_all_logs,
            "cancel-run": self.action_cancel,
            "export-run": self.action_export,
            "open-agents": self.action_open_agents,
            "results-agents": self.action_open_agents,
            "analyze-ai": self._analyze_with_ai,
            "generate-analysis": self._generate_analysis,
            "run-preflight": self._run_preflight,
            "start-bridge": self._start_bridge,
            "prepare-context": self._prepare_context,
            "setup-agents": self.action_open_agents,
        }
        action = actions.get(event.button.id or "")
        if action:
            try:
                action()
            except ValueError as exc:
                self.notify(str(exc), severity="error")

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        if event.data_table.id == "variant-table":
            self.selected_variant_id = str(event.row_key.value)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "variant-table":
            self.selected_variant_id = str(event.row_key.value)
            self._edit_variant()
        elif event.data_table.id == "runs-table":
            self.selected_run_id = str(event.row_key.value)
            self.selected_cell_id = None
            self._show_run(self.selected_run_id)
        elif event.data_table.id == "cells-table" and self.selected_run_id:
            self.selected_cell_id = str(event.row_key.value)
            self._show_run(self.selected_run_id)

    def _load_experiment(self, experiment_id: str) -> None:
        experiment = self.service.experiment(experiment_id)
        self.experiment_id = experiment.id
        self.selected_variant_id = experiment.variants[0].id
        self.plan = PlanState(
            base_experiment_id=experiment.id,
            experiment=experiment,
            request=self.service.request_for_experiment(experiment),
        )
        self._render_plan()
        self._queue_preview()
        self._refresh_setup()

    def _render_plan(self) -> None:
        self._syncing = True
        self._sync_generation += 1
        generation = self._sync_generation
        try:
            self._render_progress()
            selector = self.query_one("#experiment-select", Select)
            if self.plan.base_experiment_id in {
                value for _, value in self.service.experiment_items()
            }:
                selector.value = self.plan.base_experiment_id
            self._render_define_summary()
            self._render_proposal()
            self._render_variants()
            self._replace_selection(
                "harness-list",
                [
                    (label, value, value in self.plan.request.harnesses)
                    for label, value in HARNESS_LABELS
                ],
            )
            workload_options = [
                (
                    _workload_label(item.id, item.runner),
                    item.id,
                    item.id in self.plan.request.workloads,
                )
                for item in self.plan.experiment.workloads
            ]
            self._replace_selection("workload-list", workload_options)
            self.query_one("#workload-list").display = bool(workload_options)
            default_workload = self.query_one("#workload-default", Static)
            default_workload.display = not workload_options
            default_workload.update(
                f"Benchmark tasks from {self.plan.experiment.manifest.as_posix()}"
            )
            size = self._run_size()
            size_select = self.query_one("#run-size-select", Select)
            size_select.set_options(
                [
                    (_preset_label(item.id), item.id)
                    for item in self.plan.experiment.presets
                ]
                + [("Custom", CUSTOM_SIZE)]
            )
            size_select.value = size
            self.query_one("#custom-size-row").display = size == CUSTOM_SIZE
            request = self.plan.request
            experiment = self.plan.experiment
            self.query_one("#tasks-input", Input).value = _number(
                request.n_tasks or experiment.n_tasks
            )
            self.query_one("#attempts-input", Input).value = _number(
                request.n_attempts or experiment.n_attempts
            )
            self.query_one("#model-input", Input).value = (
                request.model or experiment.model or ""
            )
            self.query_one("#builder-model-input", Input).value = (
                request.builder_model or experiment.builder_model or ""
            )
            self.query_one("#judge-model-input", Input).value = (
                request.judge_model or experiment.judge_model or ""
            )
            self.query_one("#run-name-input", Input).value = (
                request.run_name or experiment.run_name or ""
            )
            self.query_one("#concurrency-input", Input).value = _number(
                request.n_concurrent or experiment.n_concurrent
            )
            self.query_one("#tags-input", Input).value = ",".join(request.tags)
            self.query_one("#trace-content-select", Select).value = (
                request.trace_content or experiment.trace_content
            )
            self._render_compare_sentence()
            if self.plan.preview:
                self._render_preview(self.plan.preview)
        finally:
            self.set_timer(
                0.05,
                lambda: self._finish_plan_sync(generation),
            )

    def _finish_plan_sync(self, generation: int) -> None:
        if generation == self._sync_generation:
            self._syncing = False

    def _render_progress(self) -> None:
        output = Text()
        for step, label in (
            ("define-step", "DEFINE"),
            ("compare-step", "COMPARE"),
            ("review-step", "REVIEW"),
        ):
            if output:
                output.append("  >  ", style="#6B7280")
            output.append(
                label,
                style="bold #FFCC33" if self.plan_step == step else "#D1D5DB",
            )
        if self.plan.dirty:
            output.append("  * unsaved", style="#F59E0B")
        self.query_one("#plan-progress", Static).update(output)

    def _render_define_summary(self) -> None:
        experiment = self.plan.experiment
        request = self.plan.request
        coverage = (
            f"{len(experiment.workloads)} evaluation groups"
            if experiment.workloads
            else experiment.manifest.as_posix()
        )
        model = request.model or experiment.model or "FUGUE_MODEL"
        project = self.service.deep_links().weave
        description = experiment.description or "Saved Fugue experiment"
        self.query_one("#define-summary", Static).update(
            f"{experiment.title}\n{description}\n\n"
            f"Benchmark: {coverage}\nModel: {model}\nWeave: {project}"
        )

    def _render_proposal(self) -> None:
        panel = self.query_one("#proposal-panel")
        draft = self.plan.proposal
        panel.display = draft is not None
        if draft is None:
            return
        assumptions = "; ".join(draft.assumptions) or "None"
        warnings = "; ".join(draft.warnings) or "None"
        self.query_one("#proposal-summary", Static).update(
            f"{draft.experiment.title}\n{draft.rationale}\n\n"
            f"Comparison: {len(draft.preview.variants)} variants across "
            f"{len(draft.preview.harnesses)} harnesses\n"
            f"Scale: {draft.preview.cells} cells / "
            f"{draft.preview.estimated_trials} trials\n"
            f"Assumptions: {assumptions}\nWarnings: {warnings}"
            + _evaluation_proposal_summary(draft)
            + f"\n\nProposed changes:\n{draft.diff or 'No file changes'}"
        )

    def _render_variants(self) -> None:
        table = self.query_one("#variant-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Use", "Variant", "Prompt", "Skills", "Context", "Agent")
        selected = set(self.plan.request.variants)
        for variant in self.plan.experiment.variants:
            table.add_row(
                "yes" if variant.id in selected else "no",
                variant.label,
                variant.prompt_id or "default",
                ", ".join(variant.selected_skill_ids) or "none",
                variant.context.system_id,
                "custom" if _has_agent_config(variant) else "default",
                key=variant.id,
            )

    def _render_compare_sentence(self) -> None:
        request = self.plan.request
        size = _preset_label(request.preset) if self._run_size() != CUSTOM_SIZE else "Custom"
        self.query_one("#compare-sentence", Static).update(
            f"Run {_count(len(request.variants), 'variant')} across "
            f"{_count(len(request.harnesses), 'harness')} on "
            f"{_count(len(request.workloads) or 1, 'evaluation group')} "
            f"at {size} scale."
        )

    def _replace_selection(
        self, widget_id: str, options: list[tuple[str, str, bool]]
    ) -> None:
        widget = self.query_one(f"#{widget_id}", SelectionList)
        widget.clear_options()
        for option in options:
            widget.add_option(option)

    def _set_run_size(self, value: str) -> None:
        if value == CUSTOM_SIZE:
            request = replace(
                self.plan.request,
                n_tasks=self.plan.experiment.n_tasks or 1,
                n_attempts=self.plan.experiment.n_attempts or 1,
            )
        else:
            preset = next(
                item for item in self.plan.experiment.presets if item.id == value
            )
            request = replace(
                self.plan.request,
                preset=value,
                workloads=tuple(
                    preset.workloads
                    or [item.id for item in self.plan.experiment.workloads]
                ),
                harnesses=tuple(preset.harnesses or self.plan.experiment.harnesses),
                n_tasks=None,
                n_attempts=None,
                n_concurrent=None,
            )
        self.plan = replace(self.plan, request=request, dirty=True, preview=None)
        self._render_plan()
        self._queue_preview()

    def _run_size(self) -> str:
        if self.plan.request.n_tasks is not None or self.plan.request.n_attempts is not None:
            return CUSTOM_SIZE
        preset_ids = {item.id for item in self.plan.experiment.presets}
        return (
            self.plan.request.preset
            if self.plan.request.preset in preset_ids
            else CUSTOM_SIZE
        )

    def _update_request(self, dirty: bool = True, **values: Any) -> None:
        request = replace(self.plan.request, **values)
        self.plan = replace(
            self.plan,
            request=request,
            dirty=self.plan.dirty or dirty,
            preview=None,
        )
        self._render_progress()
        self._render_compare_sentence()
        self._queue_preview()

    def _compose_with_ai(self) -> None:
        request = self.query_one("#composer-request", Input).value.strip()
        if not request:
            self.notify("Describe the comparison you want to run", severity="warning")
            return
        self.query_one("#define-summary", Static).update(
            "Grounding your request in saved experiments, prompts, skills, and context systems..."
        )
        self._compose_ai_worker(request)

    def _generate_evaluation(self) -> None:
        self.query_one("#preview-status", Static).update(
            "Generating a grounded evaluation draft for review..."
        )
        self._generate_evaluation_worker(self.plan.experiment)

    @work(thread=True, exclusive=True, group="composer")
    def _generate_evaluation_worker(self, experiment: ExperimentSpec) -> None:
        request = (
            "Generate or complete the evaluation assets for this experiment. "
            "Preserve its comparison, variants, models, and run settings exactly. "
            "Use the configured sources, fill only missing coverage, produce the "
            "configured case count, and return the assets for explicit review."
        )
        try:
            draft = asyncio.run(
                self.service.compose_experiment(
                    request,
                    base_experiment=experiment,
                    trace_content=(
                        self.plan.request.trace_content or experiment.trace_content
                    ),
                )
            )
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            self.call_from_thread(self._queue_preview)
            return
        self.call_from_thread(self._show_evaluation_proposal, draft)

    def _show_evaluation_proposal(self, draft: Any) -> None:
        self._show_proposal(draft)
        self._show_plan_step("define-step")

    @work(thread=True, exclusive=True, group="composer")
    def _compose_ai_worker(self, request: str) -> None:
        try:
            draft = asyncio.run(
                self.service.compose_experiment(
                    request,
                    base_experiment=self.plan.base_experiment_id,
                    trace_content=(
                        self.plan.request.trace_content
                        or self.plan.experiment.trace_content
                    ),
                )
            )
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            self.call_from_thread(self._render_define_summary)
            return
        self.call_from_thread(self._show_proposal, draft)

    def _show_proposal(self, draft: Any) -> None:
        self.plan = replace(self.plan, proposal=draft)
        self._render_define_summary()
        self._render_proposal()
        self.notify("Proposal ready. Use it, refine the request, or discard it.")

    def _use_proposal(self) -> None:
        draft = self.plan.proposal
        if draft is None:
            self.notify("Create a proposal first", severity="warning")
            return
        self.plan = PlanState(
            base_experiment_id=self.plan.base_experiment_id,
            experiment=draft.experiment,
            request=self.service.request_for_experiment(draft.experiment),
            assets=tuple(draft.assets),
            preview=draft.preview,
            dirty=True,
        )
        self.experiment_id = draft.experiment.id
        self.selected_variant_id = draft.experiment.variants[0].id
        self._show_plan_step("compare-step")
        self._render_plan()
        self._queue_preview()
        self.notify("Proposal applied locally. Nothing has been saved or run.")

    def _discard_proposal(self) -> None:
        self.plan = replace(self.plan, proposal=None)
        self._render_proposal()
        self.notify("Proposal discarded")

    def _selected_variant(self) -> FeatureVariant:
        return next(
            (
                item
                for item in self.plan.experiment.variants
                if item.id == self.selected_variant_id
            ),
            self.plan.experiment.variants[0],
        )

    def _add_variant(self) -> None:
        existing = {item.id for item in self.plan.experiment.variants}
        item_id = _unique_id("variant", existing)
        variant = FeatureVariant(id=item_id, label="New variant")
        self.push_screen(
            VariantEditorScreen(
                self.service,
                variant,
                existing_ids=existing,
                new=True,
            ),
            self._variant_added,
        )

    def _variant_added(self, variant: FeatureVariant | None) -> None:
        if variant is None:
            return
        experiment = replace(
            self.plan.experiment,
            variants=[*self.plan.experiment.variants, variant],
        )
        request = replace(
            self.plan.request,
            variants=(*self.plan.request.variants, variant.id),
        )
        self.selected_variant_id = variant.id
        self.plan = replace(
            self.plan,
            experiment=experiment,
            request=request,
            dirty=True,
            preview=None,
        )
        self._render_plan()
        self._queue_preview()

    def _duplicate_variant(self) -> None:
        source = self._selected_variant()
        existing = {item.id for item in self.plan.experiment.variants}
        item_id = _unique_id(f"{source.id}-copy", existing)
        duplicate = replace(source, id=item_id, label=f"{source.label} copy", enabled=True)
        self._variant_added(duplicate)

    def _edit_variant(self) -> None:
        variant = self._selected_variant()
        self.push_screen(
            VariantEditorScreen(
                self.service,
                variant,
                existing_ids={item.id for item in self.plan.experiment.variants},
                new=False,
            ),
            self._variant_edited,
        )

    def _variant_edited(self, variant: FeatureVariant | None) -> None:
        if variant is None:
            return
        experiment = replace(
            self.plan.experiment,
            variants=[
                variant if item.id == variant.id else item
                for item in self.plan.experiment.variants
            ],
        )
        self.plan = replace(
            self.plan,
            experiment=experiment,
            dirty=True,
            preview=None,
        )
        self._render_plan()
        self._queue_preview()

    def _remove_variant(self) -> None:
        variants = self.plan.experiment.variants
        if len(variants) == 1:
            self.notify("An experiment needs at least one variant", severity="warning")
            return
        selected_id = self._selected_variant().id
        remaining = [item for item in variants if item.id != selected_id]
        enabled = tuple(
            item for item in self.plan.request.variants if item != selected_id
        )
        if not enabled:
            enabled = (remaining[0].id,)
        self.selected_variant_id = remaining[0].id
        self.plan = replace(
            self.plan,
            experiment=replace(self.plan.experiment, variants=remaining),
            request=replace(self.plan.request, variants=enabled),
            dirty=True,
            preview=None,
        )
        self._render_plan()
        self._queue_preview()

    def _toggle_variant(self) -> None:
        selected_id = self._selected_variant().id
        enabled = list(self.plan.request.variants)
        if selected_id in enabled:
            if len(enabled) == 1:
                self.notify("At least one variant must be enabled", severity="warning")
                return
            enabled.remove(selected_id)
        else:
            enabled.append(selected_id)
        self.plan = replace(
            self.plan,
            request=replace(self.plan.request, variants=tuple(enabled)),
            dirty=True,
            preview=None,
        )
        self._render_variants()
        self._render_compare_sentence()
        self._render_progress()
        self._queue_preview()

    def _save_experiment(self) -> None:
        self.push_screen(SaveExperimentScreen(self.plan.experiment), self._save_plan)

    def _save_plan(self, values: tuple[str, str] | None) -> None:
        if values is None:
            return
        item_id, title = values
        try:
            saved = self.service.save_working_experiment(
                self.plan.experiment,
                self.plan.request,
                experiment_id=item_id,
                title=title,
                assets=self.plan.assets,
            )
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        selector = self.query_one("#experiment-select", Select)
        selector.set_options(self.service.experiment_items())
        self._load_experiment(saved.id)
        self._show_plan_step("review-step")
        self.notify(f"Saved {saved.id}")

    def _queue_preview(self) -> None:
        if not self.is_mounted:
            return
        if self._preview_timer is not None:
            self._preview_timer.stop()
        self._preview_timer = self.set_timer(0.3, self._begin_preview)

    def _begin_preview(self) -> None:
        if self._preview_timer is not None:
            self._preview_timer.stop()
            self._preview_timer = None
        request = self.plan.request
        if not request.harnesses or not request.variants:
            self.query_one("#preview-status", Static).update(
                "Select at least one harness and one variant."
            )
            return
        self._preview_generation += 1
        generation = self._preview_generation
        self.query_one("#preview-status", Static).update("Updating exact matrix...")
        self._preview_worker(
            generation,
            request,
            self.plan.experiment,
            self.plan.assets,
        )

    @work(thread=True, exclusive=True, group="preview")
    def _preview_worker(
        self,
        generation: int,
        request: ExperimentRequest,
        experiment: ExperimentSpec,
        assets: tuple[Any, ...],
    ) -> None:
        try:
            preview = self.service.preview_experiment(
                experiment,
                request=request,
                asset_overlay=_asset_overlay(assets),
            )
            status = self.service.status(request, experiment=experiment)
        except Exception as exc:
            self.call_from_thread(self._preview_failed, generation, str(exc))
            return
        self.call_from_thread(self._apply_preview, generation, preview, status)

    def _preview_failed(self, generation: int, message: str) -> None:
        if generation != self._preview_generation:
            return
        self.query_one("#preview-status", Static).update(f"Preview failed: {message}")
        self.query_one("#run-live", Button).disabled = True

    def _apply_preview(
        self,
        generation: int,
        preview: PreviewSummary,
        status: OperatorStatus,
    ) -> None:
        if generation != self._preview_generation:
            return
        self.plan = replace(self.plan, preview=preview)
        self._review_blockers = self._blockers(preview, status)
        self._render_preview(preview)

    def _render_preview(self, preview: PreviewSummary) -> None:
        unavailable = preview.cells - preview.applicable_cells
        self.query_one("#preview-status", Static).update(
            f"{preview.cells} cells / {preview.estimated_trials} trials / "
            f"{preview.applicable_cells} ready"
            + (f" / {unavailable} unavailable" if unavailable else "")
        )
        tasks = {
            (item.workload_id, item.task_id)
            for item in preview.matrix_cells
            if item.task_id
        }
        experiment = self.plan.experiment
        model = self.plan.request.model or experiment.model or "FUGUE_MODEL"
        contexts = ", ".join(
            sorted(
                {
                    f"{item.context_system_id} ({item.context_transport})"
                    for item in preview.matrix_cells
                    if item.context_system_id != "none"
                }
            )
        ) or "none"
        self.query_one("#review-summary", Static).update(
            f"{experiment.title}\n{experiment.description}\n\n"
            f"Benchmark: {', '.join(_workload_label(item, item) for item in preview.workloads)}\n"
            f"Model: {model}\nWeave: {self.service.deep_links().weave}\n"
            f"Context treatments: {contexts}\n"
            f"{_count(preview.cells, 'cell')} / {_count(len(tasks), 'task')} / "
            f"{_count(preview.estimated_trials, 'trial')}"
        )
        self._render_review_matrix(preview)
        warnings = list(self._review_blockers)
        warnings.extend(
            sorted(
                {
                    f"{item.harness}/{item.variant_id}: {item.reason}"
                    for item in preview.matrix_cells
                    if not item.applicable and item.reason
                }
            )
        )
        self.query_one("#review-warnings", Static).update(
            "Ready to run."
            if not warnings
            else "Needs attention:\n- " + "\n- ".join(warnings)
        )
        log = self.query_one("#execution-log", RichLog)
        log.clear()
        for command in preview.commands:
            log.write(f"$ {command}")
        self.query_one("#run-live", Button).disabled = bool(self._review_blockers)
        if self.plan_step == "review-step":
            self._show_preview_sequencer(preview)

    def _render_review_matrix(self, preview: PreviewSummary) -> None:
        table = self.query_one("#review-matrix", DataTable)
        table.clear(columns=True)
        cells = preview.matrix_cells
        variants = list(dict.fromkeys(item.variant_id for item in cells))
        labels = {
            item.variant_id: item.variant_label or item.variant_id for item in cells
        }
        table.add_columns("Harness", *[labels.get(item, item) for item in variants])
        harnesses = list(dict.fromkeys(item.harness for item in cells))
        for harness in harnesses:
            values = []
            for variant in variants:
                selected = [
                    item
                    for item in cells
                    if item.harness == harness and item.variant_id == variant
                ]
                ready = sum(item.trial_count for item in selected if item.applicable)
                unavailable = sum(not item.applicable for item in selected)
                if ready and unavailable:
                    values.append(f"{ready} ready / {unavailable} N/A")
                elif ready:
                    values.append(f"{ready} trials")
                elif unavailable:
                    values.append("N/A")
                else:
                    values.append("-")
            table.add_row(HARNESS_NAMES.get(harness, harness.title()), *values)

    def _blockers(
        self, preview: PreviewSummary, status: OperatorStatus
    ) -> tuple[str, ...]:
        blockers = []
        if not status.model_key_present:
            blockers.append(f"Model credentials are missing: {status.model_key_env}")
        if not status.trace_key_present:
            blockers.append("Weave tracing requires WANDB_API_KEY")
        generated_scoring = any(
            any(not scorer.startswith("builtin:") for scorer in workload.scorers)
            for workload in self.plan.experiment.workloads
        )
        explicit_judge = (
            self.plan.request.judge_model or self.plan.experiment.judge_model
        )
        if generated_scoring and not explicit_judge:
            blockers.append("Generated evaluation rubrics require an explicit judge model")
        judge_route = next(
            (route for route in status.routes if route.role == "judge"),
            None,
        )
        if generated_scoring and judge_route is not None and not judge_route.key_present:
            blockers.append(
                f"Judge credentials are missing: {judge_route.key_env}"
            )
        harnesses = tuple(
            dict.fromkeys(
                item.harness
                for item in preview.matrix_cells
                if item.harness not in {"direct", "sequence"}
            )
        )
        if harnesses and not status.docker_present:
            blockers.append("Docker is not available")
        if harnesses and not status.harbor_present:
            blockers.append("Harbor is not available")
        if (
            _bridge_required(status.model_provider, harnesses)
            and not status.bridge_ready
        ):
            blockers.append("The selected harness/model combination requires the local bridge")
        missing_context = sorted(
            {
                item.context_system_id
                for item in preview.matrix_cells
                if item.applicable
                and item.harness not in {"direct", "sequence"}
                and item.context_system_id != "none"
                and not item.context_cache_ready
            }
        )
        if missing_context:
            blockers.append(
                "Prepare context for: " + ", ".join(missing_context)
            )
        if preview.applicable_cells == 0:
            blockers.append("No matrix cells are applicable")
        if self.plan.assets:
            blockers.append("Save all proposed assets before running")
        return tuple(blockers)

    def _show_preview_sequencer(self, preview: PreviewSummary) -> None:
        statuses: dict[str, str] = {}
        for harness in {item.harness for item in preview.matrix_cells}:
            if harness not in HARNESS_NAMES:
                continue
            cells = [item for item in preview.matrix_cells if item.harness == harness]
            statuses[harness] = (
                "pending" if any(item.applicable for item in cells) else "not_applicable"
            )
        sequencer = self.query_one(PixelSequencer)
        sequencer.set_compact(False)
        sequencer.set_statuses(statuses)

    def _show_plan_step(self, step: str) -> None:
        self.plan_step = step
        self.query_one("#plan-steps", ContentSwitcher).current = step
        self._render_progress()
        if step == "review-step":
            if self.plan.preview:
                self._render_preview(self.plan.preview)
            else:
                self._begin_preview()
        else:
            self.query_one(PixelSequencer).set_compact(True)

    def _request_launch(self) -> None:
        if self.plan_step != "review-step":
            self._show_plan_step("review-step")
            return
        if self._review_blockers:
            self.notify(self._review_blockers[0], severity="warning")
            return
        trace_content = (
            self.plan.request.trace_content or self.plan.experiment.trace_content
        )
        if trace_content == "full":
            self.push_screen(ConfirmRunScreen(), self._launch_if_confirmed)
        else:
            self._launch_if_confirmed(True)

    def _launch_if_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        try:
            run = self.service.launch(
                self.plan.request,
                experiment=self.plan.experiment,
            )
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self.selected_run_id = run.run_id
        self._refresh_runs()
        self.action_show_runs()
        self._show_run(run.run_id)

    def action_show_compose(self) -> None:
        self.query_one("#workspace", TabbedContent).active = "compose"
        if self.plan_step == "review-step" and self.plan.preview:
            self._show_preview_sequencer(self.plan.preview)
        else:
            self.query_one(PixelSequencer).set_compact(True)

    def action_show_runs(self) -> None:
        self.query_one("#workspace", TabbedContent).active = "runs"
        self.query_one(PixelSequencer).set_compact(False)

    def action_show_results(self) -> None:
        self.query_one("#workspace", TabbedContent).active = "results"
        self.query_one(PixelSequencer).set_compact(True)

    def action_show_setup(self) -> None:
        self.query_one("#workspace", TabbedContent).active = "setup"
        self.query_one(PixelSequencer).set_compact(True)
        self._refresh_setup()

    def action_next_step(self) -> None:
        if self.query_one("#workspace", TabbedContent).active != "compose":
            return
        next_step = {
            "define-step": "compare-step",
            "compare-step": "review-step",
        }.get(self.plan_step)
        if next_step:
            self._show_plan_step(next_step)

    def action_previous_step(self) -> None:
        if self.query_one("#workspace", TabbedContent).active != "compose":
            return
        previous = {
            "compare-step": "define-step",
            "review-step": "compare-step",
        }.get(self.plan_step)
        if previous:
            self._show_plan_step(previous)

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_run(self) -> None:
        if self.query_one("#workspace", TabbedContent).active == "compose":
            self._request_launch()

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
        cells.add_columns(
            "Harness", "Variant", "Context", "Transport", "Task", "Status", "Time"
        )
        for cell in run.cells:
            cells.add_row(
                cell.harness,
                cell.variant_id,
                cell.context_system_id,
                cell.context_transport,
                cell.task_id,
                cell.status.replace("_", " "),
                f"{cell.wall_time_sec:.1f}s" if cell.wall_time_sec is not None else "-",
                key=cell.cell_id,
            )
        self._update_run_actions(run)
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

    def _update_run_actions(self, run: RunSummary) -> None:
        active = run.status in ACTIVE_RUN_STATES
        self.query_one("#cancel-run", Button).display = active
        self.query_one("#export-run", Button).display = run.status in TERMINAL_RUN_STATES

    def _show_all_logs(self) -> None:
        self.selected_cell_id = None
        if self.selected_run_id:
            self._show_run(self.selected_run_id)

    def _update_sequencer(self, runs: list[RunSummary]) -> None:
        statuses: dict[str, str] = {}
        selected = next(
            (run for run in runs if run.run_id == self.selected_run_id), None
        )
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
            f"{result.context_registered}/{result.context_assigned} registered   "
            f"{result.context_invoked}/{result.context_assigned} context used   "
            f"{result.runtime_mismatched} runtime mismatches   "
            f"{result.attributed_errors} attributed errors   "
            f"{result.linked_traces} linked traces   "
            f"{result.usage_unavailable} usage N/A"
        )
        table = self.query_one("#results-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "Harness",
            "Experiment",
            "Variant",
            "Context",
            "Transport",
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
        groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for row in result.rows:
            key = (
                str(row.get("harness") or "unknown"),
                str(row.get("experiment_id") or "unknown"),
                str(row.get("variant_id") or "baseline"),
                str(row.get("context_system_id") or "none"),
                str(row.get("context_transport") or "portable"),
                str(row.get("model") or "unknown"),
            )
            groups[key].append(row)
        for (harness, experiment, variant, context, transport, model), rows in sorted(
            groups.items()
        ):
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
                int(row.get("n_input_tokens") or 0)
                + int(row.get("n_output_tokens") or 0)
                for row in rows
            )
            conversations = {
                value
                for row in rows
                for value in (row.get("weave_conversation_ids") or [])
            }
            assigned = sum(bool(row.get("context_assigned")) for row in rows)
            invoked = sum(bool(row.get("context_invoked")) for row in rows)
            linked = sum(row.get("trace_link_status") == "linked" for row in rows)
            usage_unavailable = all(
                row.get("weave_usage_status") == "unavailable" for row in rows
            )
            table.add_row(
                harness,
                experiment,
                variant,
                f"{context} ({invoked}/{assigned} used)" if assigned else context,
                transport,
                model,
                str(len(rows)),
                pass_rate,
                f"{sum(rewards) / len(rewards):.3f}" if rewards else "N/A",
                f"{sum(times) / len(times):.1f}s" if times else "N/A",
                f"${cost:.4f}",
                "N/A" if usage_unavailable else f"{tokens:,}",
                str(sum(int(row.get("weave_tool_call_count") or 0) for row in rows)),
                str(
                    sum(
                        bool(row.get("exception_class"))
                        or row.get("pass") is False
                        for row in rows
                    )
                ),
                f"{len(conversations)} / {linked} linked",
            )

    def _analyze_with_ai(self) -> None:
        question = self.query_one("#analysis-question", Input).value.strip()
        if not question:
            self.notify("Ask a question about your experiments", severity="warning")
            return
        model = self.query_one("#analysis-model", Input).value.strip() or None
        source = str(self.query_one("#analysis-source", Select).value)
        report = self.query_one("#analysis-report", RichLog)
        report.clear()
        report.write("Resolving a reproducible local scope...")
        self._analysis_scope_worker(question, model, source)

    @work(thread=True, exclusive=True, group="analyst")
    def _analysis_scope_worker(
        self,
        question: str,
        model: str | None,
        source: str,
    ) -> None:
        try:
            spec = asyncio.run(
                self.service.plan_analysis(
                    question,
                    filters={},
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
        warnings = "; ".join(scope.warnings) if scope.warnings else "None"
        self.query_one("#analysis-scope", Static).update(
            f"{len(scope.experiments)} experiments / "
            f"{len(scope.runs)} runs / {scope.rows} records / "
            f"{len(scope.tasks)} tasks\n"
            f"Models: {', '.join(scope.models)}\nWarnings: {warnings}"
        )
        report = self.query_one("#analysis-report", RichLog)
        report.clear()
        report.write(
            "Scope resolved locally. Review it before generating the report."
        )
        self.query_one("#generate-analysis", Button).disabled = False
        self.notify("Scope ready; no Weave query or report call has run yet")

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
        self.analysis_preview = None
        self.query_one("#analysis-scope", Static).update(
            f"{len(result.scope.experiments)} experiments / "
            f"{len(result.scope.runs)} runs / {result.scope.rows} records / "
            f"{len(result.scope.tasks)} tasks\n"
            f"Report: {result.report_dir}"
        )
        log = self.query_one("#analysis-report", RichLog)
        log.clear()
        log.write(result.report)
        self.query_one("#generate-analysis", Button).disabled = True
        self.notify(f"Analysis saved to {result.report_dir}")

    def _refresh_setup(self) -> None:
        try:
            status = self.service.status(
                self.plan.request,
                experiment=self.plan.experiment,
            )
        except Exception as exc:
            self.query_one("#setup-log", RichLog).write(str(exc))
            return
        table = self.query_one("#setup-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Area", "State", "Summary")
        model_ready = all(
            route.key_present
            for route in status.routes
            if route.role in {"target", "builder", "judge"}
        )
        runtime_ready = status.docker_present and status.harbor_present
        selected_context = [
            item for item in status.selected_context_systems if item != "none"
        ]
        rows = (
            (
                "Model provider",
                model_ready,
                f"{status.model_provider} / {status.model}",
            ),
            (
                "W&B / Weave",
                status.trace_key_present,
                status.trace_project,
            ),
            (
                "Local runtime",
                runtime_ready,
                f"Docker {'ready' if status.docker_present else 'missing'}, "
                f"Harbor {'ready' if status.harbor_present else 'missing'}",
            ),
            (
                "Required context",
                not selected_context or status.context_cache_entries > 0,
                (
                    ", ".join(selected_context)
                    if selected_context
                    else "No prepared context required"
                ),
            ),
        )
        for name, ready, detail in rows:
            table.add_row(name, "ready" if ready else "needs attention", detail)
        self.query_one("#start-bridge", Button).display = (
            _bridge_required(status.model_provider, self.plan.request.harnesses)
            and not status.bridge_ready
        )
        self.query_one("#prepare-context", Button).display = bool(selected_context)
        route_details = "\n".join(
            f"{route.role}: {route.model} via {route.key_env} "
            f"({'present' if route.key_present else 'missing'})"
            for route in status.routes
        )
        self.query_one("#setup-details", Static).update(
            f"{route_details}\n"
            f"Bridge: {'ready' if status.bridge_ready else 'offline'} at 127.0.0.1:4000\n"
            f"Trace content: {status.trace_content}\n"
            f"Context cache entries: {status.context_cache_entries}\n"
            f"Weave: {status.links.weave}"
        )

    def _run_preflight(self) -> None:
        self.query_one("#setup-log", RichLog).write(
            "Running observational preflight..."
        )
        self._preflight_worker(self.plan.request, self.plan.experiment)

    @work(thread=True, exclusive=True, group="setup")
    def _preflight_worker(
        self,
        request: ExperimentRequest,
        experiment: ExperimentSpec,
    ) -> None:
        try:
            checks = self.service.preflight(
                request,
                live=True,
                experiment=experiment,
            )
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            return
        self.call_from_thread(self._show_preflight, checks)

    def _show_preflight(self, checks: tuple[Any, ...]) -> None:
        log = self.query_one("#setup-log", RichLog)
        log.clear()
        for check in checks:
            log.write(
                f"{'ready' if check.ok else 'missing':<8} "
                f"{check.name}: {check.detail}"
            )
        ready = all(check.ok for check in checks)
        self.notify(
            "Preflight passed" if ready else "Preflight found missing setup",
            severity="information" if ready else "warning",
        )
        self._refresh_setup()

    def _start_bridge(self) -> None:
        self.query_one("#setup-log", RichLog).write(
            "Starting the local model bridge..."
        )
        self._bridge_worker(self.plan.request, self.plan.experiment)

    @work(thread=True, exclusive=True, group="setup")
    def _bridge_worker(
        self,
        request: ExperimentRequest,
        experiment: ExperimentSpec,
    ) -> None:
        try:
            files = self.service.start_bridge(request, experiment=experiment)
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            return
        self.call_from_thread(self._show_bridge_started, files.runtime_dir)

    def _show_bridge_started(self, runtime_dir: Any) -> None:
        self.query_one("#setup-log", RichLog).write(
            f"Bridge running from {runtime_dir}"
        )
        self.notify("Bridge started")
        self._refresh_setup()

    def _prepare_context(self) -> None:
        self.query_one("#setup-log", RichLog).write(
            "Preparing selected context systems..."
        )
        self._prepare_context_worker(self.plan.request, self.plan.experiment)

    @work(thread=True, exclusive=True, group="setup")
    def _prepare_context_worker(
        self,
        request: ExperimentRequest,
        experiment: ExperimentSpec,
    ) -> None:
        try:
            records = self.service.prepare_context(request, experiment=experiment)
        except Exception as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            return
        self.call_from_thread(self._show_context_prepared, records)

    def _show_context_prepared(self, records: tuple[Any, ...]) -> None:
        log = self.query_one("#setup-log", RichLog)
        log.clear()
        for item in records:
            log.write(
                f"{item.status:<8} {item.system_id}/{item.task_id}: {item.detail}"
            )
        self.notify(f"Context preparation finished for {len(records)} targets")
        self._queue_preview()
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


def _asset_overlay(assets: tuple[Any, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for asset in assets:
        if asset.kind == "prompt":
            path = Path("configs/fugue/prompts") / f"{asset.id}.md"
        elif asset.kind == "skill":
            path = Path("configs/fugue/skills") / asset.id / "SKILL.md"
        else:
            path = evaluation_asset_path(asset.kind, asset.id)
        values[path.as_posix()] = asset.body
    return values


def _evaluation_proposal_summary(draft: Any) -> str:
    evaluation = getattr(draft, "evaluation", None)
    if evaluation is None:
        return ""
    dimensions = [
        str(item.get("id")) for item in evaluation.rubric.get("dimensions") or []
    ]
    source_hashes = (
        evaluation.rubric.get("generation", {}).get("source_hashes", {})
    )
    coverage = ", ".join(
        f"{key}={value}" for key, value in sorted(evaluation.coverage.items())
    )
    return (
        f"\nEvaluation: {len(evaluation.cases)} cases ({coverage})"
        f"\nDimensions: {', '.join(dimensions)}"
        f"\nProvenance: {len(source_hashes)} checksum-pinned sources"
    )


def _animation_enabled() -> bool:
    return not (
        os.environ.get("FUGUE_NO_ANIMATION") == "1"
        or "NO_COLOR" in os.environ
        or not sys.stdout.isatty()
    )


def _number(value: int | None) -> str:
    return "" if value is None else str(value)


def _optional_positive(value: str, label: str) -> int | None:
    if not value.strip():
        return None
    selected = int(value)
    if selected < 1:
        raise ValueError(f"{label} must be positive")
    return selected


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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


def _status_color(status: str) -> str:
    return {
        "running": "#00AFC2",
        "passed": "#22C55E",
        "failed": "#EF4444",
        "cancelled": "#F59E0B",
        "interrupted": "#F59E0B",
        "not_applicable": "#6B7280",
        "pending": "#9CA3AF",
    }.get(status, "#6B7280")


def _status_glyph(status: str) -> str:
    return {
        "running": "▶",
        "passed": "■",
        "failed": "×",
        "cancelled": "!",
        "interrupted": "!",
        "not_applicable": "-",
        "pending": "·",
    }.get(status, "·")


def _preset_label(value: str | None) -> str:
    return str(value or "Custom").replace("-", " ").replace("_", " ").title()


def _count(value: int, noun: str) -> str:
    return f"{value} {noun if value == 1 else noun + 's'}"


def _workload_label(item_id: str, runner: str) -> str:
    labels = {
        "retrieval": "Retrieval",
        "qa": "Repository QA",
        "coding": "Coding",
        "continuity": "Continuity",
        "harbor": "Benchmark tasks",
    }
    return labels.get(item_id, labels.get(runner, item_id.replace("-", " ").title()))


def _has_agent_config(variant: FeatureVariant) -> bool:
    return any(
        (
            variant.agent_kwargs,
            variant.agent_env,
            variant.mcp_servers,
            variant.environment,
            variant.verifier,
            variant.retry,
            variant.artifacts,
        )
    )


def _variant_advanced_summary(variant: FeatureVariant) -> str:
    if not _has_agent_config(variant):
        return "Uses experiment and Harbor defaults."
    names = [
        label
        for label, value in (
            ("agent kwargs", variant.agent_kwargs),
            ("agent env", variant.agent_env),
            ("MCP servers", variant.mcp_servers),
            ("environment", variant.environment),
            ("verifier", variant.verifier),
            ("retry", variant.retry),
            ("artifacts", variant.artifacts),
        )
        if value
    ]
    return "Preserves custom " + ", ".join(names) + ". Edit YAML for raw values."


def _unique_id(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    index = 2
    while f"{base}-{index}" in existing:
        index += 1
    return f"{base}-{index}"


def _bridge_required(provider: str, harnesses: tuple[str, ...]) -> bool:
    native = {
        "hermes": {"wandb", "openai"},
        "openclaw": {"wandb", "openai"},
        "claude-code": {"anthropic"},
        "codex": {"openai"},
    }
    return any(provider not in native.get(harness, set()) for harness in harnesses)
