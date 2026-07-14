# Fugue

Fugue plans, runs, and analyzes controlled agent experiments across Hermes,
OpenClaw, Claude Code, and Codex. Harbor executes each experiment cell; W&B
Weave records agent conversations and traces; Fugue keeps the comparison
matrix, local run state, and normalized outcomes coherent.

```mermaid
flowchart LR
    USER["Natural language or saved experiment"] --> PLAN["Fugue Plan"]
    PLAN --> OP["OperatorService"]
    OP --> MATRIX["Harness x variant x task x trial"]
    MATRIX --> HARBOR["Harbor jobs"]
    HARBOR --> LOCAL["Durable local run state"]
    HARBOR --> WEAVE["Weave agent traces"]
    LOCAL --> SCOPE["Deterministic analysis scope"]
    WEAVE --> ENRICH["Narrow trace enrichment"]
    SCOPE --> REPORT["Evidence-backed report"]
    ENRICH --> REPORT
```

## Quick Start

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
```

Install `.[context]` as well when running local RAG or persistent-memory
providers.

At minimum, configure W&B for tracing and the selected model provider:

```bash
WANDB_API_KEY=
WANDB_ENTITY=wandb
WANDB_PROJECT=fugue-experiments

OPENAI_API_KEY=
ANTHROPIC_API_KEY=

FUGUE_MODEL=wandb/zai-org/GLM-5.2
LITELLM_MASTER_KEY=sk-fugue-local
```

`WANDB_API_KEY` is always required for Weave tracing. It also pays for model
calls when the selected route starts with `wandb/`. OpenAI and Anthropic keys
are needed only when their provider route is selected.

Run bare `fugue` in a terminal to open the Rich command center:

```bash
fugue
```

It shows the active model and Weave project, operational readiness, recent
runs, and a harness sequencer. The full-screen workspace remains available as:

```bash
fugue tui
fugue tui --screen results
```

## Command Model

Fugue has six explicit commands plus the bare command center:

```text
fugue
fugue plan
fugue run
fugue runs
fugue analyze
fugue setup
fugue tui
```

```mermaid
flowchart TB
    HOME["fugue command center"]
    HOME --> P["plan: design an experiment"]
    HOME --> R["run: preview or execute"]
    HOME --> RS["runs: inspect, export, open"]
    HOME --> A["analyze: resolve scope, then report"]
    HOME --> S["setup: check, bridge, context"]
    HOME --> T["tui: full-screen workspace"]
```

Commands accept `--json` where structured automation is useful. JSON mode does
not emit Rich decoration or interactive prompts.

## Setup

Show setup for an experiment:

```bash
fugue setup --experiment pilot
```

Run observational checks, start the explicit local bridge, or prepare selected
context systems:

```bash
fugue setup --experiment pilot --check

fugue setup \
  --experiment pilot \
  --model wandb/zai-org/GLM-5.2 \
  --start-bridge

fugue setup \
  --experiment repo-memory-impact \
  --preset smoke \
  --workloads coding \
  --systems none,rag-bm25 \
  --prepare-context
```

Preflight never starts containers or writes bridge files. `--start-bridge` and
`--prepare-context` are explicit mutations.

Model precedence is:

```text
CLI override > experiment/harness configuration > environment > Fugue default
```

Target, builder, judge, composer, and analyst routes are resolved separately.

## Plan Experiments

Saved experiments live under `configs/fugue/experiments/`. Prompts and skills
live under `configs/fugue/prompts/` and `configs/fugue/skills/`.

Plan from natural language:

```bash
fugue plan \
  "Compare BM25 with no context across every harness for one coding task" \
  --from repo-memory-impact
```

Fugue grounds the request in checked-in manifests, prompts, skills, context
systems, presets, and model routes. It then validates the generated experiment
and renders a side-effect-free matrix preview.

```mermaid
sequenceDiagram
    participant U as User
    participant C as Experiment planner
    participant O as OperatorService
    participant T as Rich or Textual
    U->>C: Describe comparison
    C->>O: Request repository catalog
    C->>O: Submit ExperimentSpec
    O->>O: Strict validation and preview
    O-->>T: Draft, diff, cells, trials, warnings
    T-->>U: Continue in TUI, save, run, or discard
    U->>T: Explicit approval
```

Scripted save and launch remain explicit:

```bash
fugue plan "Create a smaller PDF skill comparison" \
  --from skillsbench-pdf-ab \
  --save pdf-skill-smoke

fugue plan "Run the checked-in configuration unchanged" \
  --from pilot \
  --run \
  --yes
```

Generated prompt or skill assets must be saved before the draft can run.

## Run Experiments

Preview is side-effect free: it does not write runtime state, generated
JobConfigs, downloads, indexes, or experiment files.

```bash
fugue run pilot --preview
```

Start a durable run and wait while Rich renders the live cell matrix:

```bash
fugue run pilot
```

Return immediately while the same managed run continues in its process group:

```bash
fugue run pilot --detach
```

```mermaid
flowchart TD
    REQUEST["ExperimentRequest"] --> PREP["Prepare datasets and context"]
    PREP --> CONFIG["One JobConfig per planned cell"]
    CONFIG --> STATE[".fugue/runtime/run-id"]
    STATE --> WORKERS["Bounded concurrent workers"]
    WORKERS --> H1["Hermes"]
    WORKERS --> H2["OpenClaw"]
    WORKERS --> H3["Claude Code"]
    WORKERS --> H4["Codex"]
    H1 --> CELLS["cells.jsonl and per-cell logs"]
    H2 --> CELLS
    H3 --> CELLS
    H4 --> CELLS
    CELLS --> FINAL["passed / failed / not_applicable"]
```

Every run receives an immutable generated run ID. The requested run name is a
human grouping label only. One failed cell does not stop sibling cells.

## Inspect Runs

```bash
fugue runs
fugue runs RUN_ID
fugue runs RUN_ID --logs --follow
fugue runs RUN_ID --logs --cell CELL_ID
fugue runs RUN_ID --cancel
```

Export normalized JSONL and optionally enrich/publish through Weave:

```bash
fugue runs RUN_ID \
  --export \
  --out reports/run.jsonl \
  --fetch-weave \
  --to-weave
```

Open the stable W&B destinations:

```bash
fugue runs RUN_ID --open agents
fugue runs RUN_ID --open trace --cell CELL_ID
fugue runs RUN_ID --open project
```

Fugue opens an exact trace only when a verified URL exists. Otherwise it opens
Weave Agents and prints the conversation ID rather than inventing a URL.

## Analyze Results

Ask a comparative question:

```bash
fugue analyze \
  "Which context system improved coding outcomes without excessive latency?" \
  --filter experiment_id=repo-memory-impact
```

Analysis has an explicit confirmation boundary:

```mermaid
flowchart LR
    Q["Natural-language question"] --> SPEC["AnalysisSpec"]
    SPEC --> LOCAL["Local catalog filter"]
    LOCAL --> SNAP["Immutable row snapshot"]
    SNAP --> PREVIEW["Scope and deterministic aggregates"]
    PREVIEW --> CONFIRM{"Confirm report?"}
    CONFIRM -->|No| STOP["No Weave query or report write"]
    CONFIRM -->|Yes| WEAVE["Narrow Weave enrichment"]
    WEAVE --> MODEL["Evidence-bound interpretation"]
    MODEL --> REPORT["report.md + scope + evidence"]
```

For automation, `--yes` confirms report generation:

```bash
fugue analyze "Compare PDF skill lift by harness" \
  --filter experiment_id=skillsbench-pdf-ab \
  --save pdf-skill-lift \
  --yes

fugue analyze --list
fugue analyze --saved pdf-skill-lift --yes
```

Official arithmetic is deterministic Python over the immutable snapshot. The
model interprets aggregates and must cite registered evidence IDs. Hybrid mode
starts from local outcomes and requests only matching Weave run keys and
conversation IDs.

## Context Systems

Context providers implement preparation, binding, retrieval, and optional
ingestion without changing the experiment runner. Cached indexes live under
`.fugue/cache/context/v2`; per-run state lives under `.fugue/runtime/`.

```mermaid
flowchart LR
    TASK["Dataset + task + repository commit"] --> KEY["Content-addressed key"]
    KEY --> LOCK["Process-safe build lock"]
    LOCK --> PROVIDER["Context provider prepare"]
    PROVIDER --> CACHE["Atomic cache publication"]
    CACHE --> BIND["Instructions / MCP / mounts / sidecars"]
    BIND --> CELL["Harbor cell"]
```

Default studies use only systems with runnable prerequisites. CodeGraph,
GitNexus, Project-RAG, Semble, and lat.md remain explicit research adapters
until their pinned Harbor runtimes pass integration tests. Unsupported cells
are recorded as `not_applicable`, never as failed trials.

## Weave Agent Model

Harness identities are stable across experiments:

```text
hermes-agent
openclaw
claude-code
codex
```

```mermaid
flowchart TD
    AGENT["Stable harness agent"] --> CONV["Trial conversation"]
    CONV --> TURN["invoke_agent turn"]
    TURN --> CHAT["chat spans"]
    TURN --> TOOL["execute_tool spans"]
    TURN --> SUB["sub-agent spans"]
    CONV --> ATTR["fugue.run / experiment / variant / context / task / trial"]
```

Native harness integrations own model and tool spans. Fugue supplies stable
conversation identity and flat filterable attributes without duplicating
instrumentation. Full trace content is the default and may include prompts,
responses, reasoning, tool arguments, and tool results. Use metadata mode only
for integrations that can guarantee suppression.

## Included Demos

### PDF Skill A/B

```bash
fugue setup --experiment skillsbench-pdf-ab --check
fugue setup --experiment skillsbench-pdf-ab --start-bridge
fugue run skillsbench-pdf-ab --preview
fugue run skillsbench-pdf-ab --detach
```

This compares a Fugue-authored PDF workflow skill with a no-skill baseline
across four harnesses and three SkillsBench tasks. It is not an official
SkillsBench leaderboard reproduction.

### Context A/B Smoke

```bash
fugue setup \
  --experiment repo-memory-impact \
  --preset smoke \
  --workloads coding \
  --systems none,rag-bm25 \
  --prepare-context

fugue run repo-memory-impact \
  --preset smoke \
  --workloads coding \
  --systems none,rag-bm25 \
  --harnesses hermes,openclaw,claude-code,codex \
  -k 1 -n 2 -l 1 \
  --preview
```

Remove `--preview` to launch the eight-cell comparison.

## Development

```bash
python -m compileall fugue
python -m ruff check .
python -m pytest
```

Generated state belongs under `.fugue/`, `jobs/`, or `reports/`. Saved
experiments, prompts, skills, analyses, and context-system definitions belong
under `configs/fugue/`.
