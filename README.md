# fugue

**Run multiple agent harnesses on the same tasks, trace every run in W&B
Weave, and compare the results note for note.**

Fugue is a thin, opinionated layer on top of
[Harbor](https://github.com/laude-institute/harbor) for sandboxed trial
execution and [W&B Weave](https://wandb.ai/site/weave) for traces and
evaluation. W&B is always the trace plane. The model plane is provider
neutral: runs can bill through W&B Inference, OpenAI, or Anthropic by changing
one model string.

```text
wandb/zai-org/GLM-5.2
openai/gpt-5
anthropic/claude-sonnet-4-5
```

## System At A Glance

Fugue turns a saved experiment into isolated Harbor jobs, keeps context
artifacts specific to each task, and joins local outcomes with Weave traces for
comparison.

```mermaid
flowchart LR
    subgraph authoring["Authoring"]
        UI["Textual terminal operator"]
        CLI["Rich headless CLI"]
        CONFIG["Experiments, prompts, skills, and manifests"]
    end

    subgraph orchestration["Fugue orchestration"]
        PLAN["Validate and plan cells"]
        CONTEXT["Prepare task-specific context"]
        RENDER["Render Harbor JobConfigs"]
        RUN["Run cells with bounded concurrency"]
    end

    subgraph execution["Harbor execution"]
        HERMES["Hermes"]
        OPENCLAW["OpenClaw"]
        CLAUDE["Claude Code"]
        CODEX["Codex CLI"]
    end

    subgraph outputs["Results"]
        LOCAL["Harbor jobs, artifacts, and cell state"]
        EXPORT["Normalized export"]
        WEAVE["W&B Weave traces and evaluations"]
        COMPARE["Terminal summaries"]
    end

    UI --> PLAN
    CLI --> PLAN
    CONFIG --> PLAN
    PLAN --> CONTEXT --> RENDER --> RUN
    RUN --> HERMES
    RUN --> OPENCLAW
    RUN --> CLAUDE
    RUN --> CODEX
    HERMES --> LOCAL
    OPENCLAW --> LOCAL
    CLAUDE --> LOCAL
    CODEX --> LOCAL
    LOCAL --> EXPORT --> COMPARE
    EXPORT --> WEAVE
    HERMES --> WEAVE
    OPENCLAW --> WEAVE
    CLAUDE --> WEAVE
    CODEX --> WEAVE
```

## Harnesses

| harness | adapter | model routing | Weave plugin |
|---|---|---|---|
| Hermes | `fugue.agents:FugueHermes` | OpenAI-compatible chat, direct or bridged | local `hermes-otel` checkout |
| OpenClaw | `fugue.agents:FugueOpenClaw` | OpenAI-compatible chat, direct or bridged | `weave-openclaw` |
| Claude Code | `fugue.agents:FugueClaudeCode` | native Anthropic Messages for `anthropic/...`, bridge otherwise | `weave-claude-code` |
| Codex CLI | `fugue.agents:FugueCodex` | native OpenAI Responses for `openai/...`, bridge otherwise | `weave-codex` |
| Letta Code | `fugue.agents:FugueLetta` | native OpenAI/Anthropic, W&B through the bridge | exported trial metadata + MemFS artifact |

The local LiteLLM bridge is generated under `.fugue/bridge/` by
`fugue bridge up`. It binds to `127.0.0.1:4000`; Harbor task containers reach
it at `http://host.docker.internal:4000`. It exposes separate target, builder,
and judge aliases and runs a pinned LiteLLM image.

### Model And Trace Routing

The model prefix selects the billing provider. Target, builder, and judge are
resolved as separate roles, while all operational traces still go to the same
configured W&B project.

```mermaid
flowchart LR
    HARNESS["Harness or context builder"] --> ROLE["Target, builder, or judge role"]
    ROLE --> ROUTER{"Model prefix"}
    ROUTER -->|"wandb/..."| WB["W&B Inference"]
    ROUTER -->|"openai/..."| OAI["OpenAI native API or local bridge"]
    ROUTER -->|"anthropic/..."| ANT["Anthropic native API or local bridge"]

    HARNESS --> TRACE["Weave instrumentation"]
    TRACE --> PROJECT["WANDB_ENTITY / WANDB_PROJECT"]

    ROUTER --> CREDENTIALS["Validate credentials for the selected provider"]
    ENV["Provider API keys"] --> CREDENTIALS
    WANDBKEY["WANDB_API_KEY"] --> TRACE
```

## Quick Start

```bash
cp .env.example .env
uv venv && uv pip install -e ".[dev,context]"

fugue status
fugue bridge up --model wandb/zai-org/GLM-5.2
fugue                         # full-screen terminal operator

# The same workflow remains scriptable.
fugue run --experiment pilot --dry-run
fugue run --experiment pilot --detach
fugue runs list
```

Requirements: Docker Desktop, Harbor (`uv tool install harbor`), `jq`, and for
Hermes a local `hermes-otel` checkout (`HERMES_OTEL_CHECKOUT`, default
`~/Documents/GitHub/hermes-otel`).

## Experiment Runner

An experiment describes the comparison, while Fugue expands it into the
smallest independently executable cells. Harbor repeats each cell for the
requested number of trials.

```mermaid
flowchart LR
    EXP["Saved experiment"] --> MERGE["Resolve manifest, preset, and CLI overrides"]
    MANIFEST["Dataset and task manifest"] --> MERGE
    OVERRIDES["Model, harness, system, workload, and limit overrides"] --> MERGE
    MERGE --> MATRIX["Expand the experiment matrix"]
    MATRIX --> CELL["PlannedCell: harness + variant + task"]
    CELL --> CONFIG["One task-specific Harbor JobConfig"]
    CONFIG --> TRIALS["Harbor runs N trials for the cell"]
    TRIALS --> RESULT["Trial results and artifacts"]
```

```bash
fugue prompts list
fugue skills list
fugue experiments list

fugue run --experiment pilot --manifest datasets/pilot.yaml \
  --model openai/gpt-5 \
  --harnesses hermes,openclaw \
  --variants baseline,prompt-skill \
  --run-name gpt5-prompt-skill-sweep \
  --tags pilot,gpt5 \
  -k 1 -l 3
fugue export --jobs jobs/pilot --out reports/pilot.jsonl --to-weave
```

## Skill A/B Demo

The checked-in `skillsbench-pdf-ab` experiment measures one controlled change:
the `baseline` variant receives no skill, while `with-pdf-skill` receives
Fugue's original `pdf-artifact-workflow` skill. Both variants otherwise use the
same model, tasks, harness settings, prompt, context system, and trial count.

The demo uses three PDF-heavy tasks from
[SkillsBench v1.1](https://www.skillsbench.ai/blogs/skillsbench-1-1): form
filling, PDF-to-spreadsheet comparison, and document anonymization. Across four
harnesses and two attempts, the complete run contains 48 trials:

```text
4 harnesses x 2 variants x 3 tasks x 2 trials = 48 trials
```

Check model and trace credentials, then start the local bridge:

```bash
fugue preflight --model wandb/zai-org/GLM-5.2 --no-bridge-up
fugue bridge up --model wandb/zai-org/GLM-5.2
```

In another terminal, render and inspect the 24 task-specific Harbor JobConfigs before
launching the matrix:

```bash
fugue run --experiment skillsbench-pdf-ab --dry-run
fugue run --experiment skillsbench-pdf-ab \
  --run-name skillsbench-pdf-ab-v1
```

Export the joined results to JSONL and Weave, or inspect each trial and its
collected output artifacts in Harbor's local viewer:

```bash
fugue export --jobs jobs/skillsbench-pdf-ab \
  --out reports/skillsbench-pdf-ab.jsonl \
  --fetch-weave \
  --to-weave
harbor view jobs
```

Compare pass rate, reward, cost, tokens, wall time, and failures by harness and
variant. For individual wins and regressions, open the Weave traces and check
whether the harness found and followed the skill before producing its artifact.

This is a Fugue-authored skill experiment running on public SkillsBench tasks.
It does not copy SkillsBench's bundled skills and is not an official
SkillsBench leaderboard reproduction.

Model precedence is:

1. CLI `--model`
2. Harness `model`
3. Experiment `model`
4. Manifest `model`
5. `FUGUE_MODEL`
6. `wandb/zai-org/GLM-5.2`

Builder and judge routes resolve independently from CLI flags, experiment
fields, and `FUGUE_BUILDER_MODEL` / `FUGUE_JUDGE_MODEL`. Shell variables take
precedence over `.env`; blank dotenv entries do not erase exported credentials.

Saved experiments live under `configs/fugue/experiments/`, prompts under
`configs/fugue/prompts/`, and Harbor skills under `configs/fugue/skills/`.
Each experiment defines feature variants: named bundles of prompt, skills,
context system, and advanced Harbor agent settings. Fugue renders one Harbor
JobConfig per harness, variant, and task; Harbor owns the configured number of
trials inside that cell. This prevents an index prepared for one task from
being mounted into another task. `fugue export` joins Harbor `result.json`,
`agent/fugue-meta.json`, context telemetry, and optional Weave span summaries.

Every live execution receives an immutable generated `run_id`; `--run-name` is
only a label. Cell transitions are appended to
`.fugue/runtime/<run_id>/cells.jsonl` as `pending`, `running`, `passed`,
`failed`, or `not_applicable`. Cells run with bounded concurrency, and one
failed Harbor command does not stop the remaining experiment.

### Cell Lifecycle

Every planned cell remains visible, including unsupported combinations. This
makes missing coverage distinguishable from an actual failed evaluation.

```mermaid
stateDiagram-v2
    [*] --> Pending
    Pending --> Running: prerequisites satisfied
    Pending --> NotApplicable: unsupported or unavailable
    Running --> Passed: Harbor command succeeds
    Running --> Failed: Harbor command fails
    Pending --> Cancelled: operator cancels run
    Running --> Cancelled: operator cancels run
    Pending --> Interrupted: process disappears
    Running --> Interrupted: process disappears
    Passed --> [*]
    Failed --> [*]
    Cancelled --> [*]
    Interrupted --> [*]
    NotApplicable --> [*]

    note right of Failed
        The error is persisted.
        Independent cells continue.
    end note
```

## Context-System Evaluation

`repo-memory-impact` compares context systems without forcing them through one
fake interface. Harbor workloads measure whether a system helps Hermes,
OpenClaw, Claude Code, and Codex complete repository tasks. Direct retrieval
workloads score systems that expose ranked hits. Provider diagnostic sequence
workloads measure ingestion, recall, latency, and storage growth separately
from harness trials. Episodes remain ordered within a cohort while independent
cohorts run concurrently. Metrics that a system
cannot produce remain `N/A`; they are not converted into failures or zeroes.

### Evaluation Lanes

Context systems keep their native capabilities. Fugue compares equivalent
measurements without pretending that every system offers ranked retrieval or
agent-managed continuity.

```mermaid
flowchart LR
    STUDY["Experiment preset"] --> HARBOR["Harbor task lane"]
    STUDY --> RETRIEVAL["Direct retrieval lane"]
    STUDY --> SEQUENCE["Provider diagnostic sequence lane"]

    HARBOR --> OUTCOME["Pass rate, reward, cost, tokens, and failures"]
    RETRIEVAL --> RANKED["MRR, NDCG, recall, precision, and query latency"]
    SEQUENCE --> MEMORY["Ingestion, recall over time, latency, and storage"]

    OUTCOME --> ROWS["Typed normalized records"]
    RANKED --> ROWS
    MEMORY --> ROWS
    NA["Unsupported metrics"] -->|"reported as N/A"| ROWS
```

Context definitions live in `configs/fugue/context-systems/`. Prepared indexes
are content-addressed under `.fugue/cache/context/v2/`; cache keys include the
dataset, task, repository, commit, provider/version/config, builder model, and
embedding model. OS-backed locks coordinate builders, and completed manifests
and the shared index are published through atomic replacement.
The built-in controlled baselines use the same chunker with BM25, dense
`BAAI/bge-small-en-v1.5` retrieval, or reciprocal-rank hybrid retrieval.
Fugue-owned MCP retrieval runs in the pinned image defined by
`Dockerfile.context`, attached to each Harbor cell as a Compose sidecar with
only that task's prepared index mounted. It is never installed into the agent
container. Third-party MCP systems without a declared, pinned runtime are
reported as `not_applicable` instead of failing after a trial starts.

### Context Preparation And Isolation

The cache is reusable, but its identity includes the task repository and exact
model routes. Each Harbor cell receives only the context artifact prepared for
its own task.

```mermaid
flowchart TB
    INPUT["Dataset, task, repository, and commit"] --> KEY["Content-addressed cache key"]
    SYSTEM["Context system, version, and config"] --> KEY
    MODELS["Builder and embedding routes"] --> KEY
    KEY --> LOCK["Acquire OS-backed build lock"]
    LOCK --> CHECK{"Valid v2 cache entry?"}
    CHECK -->|"yes"| READY["Reuse prepared artifact"]
    CHECK -->|"no"| BUILD["Build index or context artifact"]
    BUILD --> SAMPLE["Measure build time and process resources"]
    SAMPLE --> PUBLISH["Atomically publish manifest and shared index"]
    PUBLISH --> READY
    READY --> BIND["Bind task-specific mount, instructions, MCP, and artifacts"]
    BIND --> JOB["Harbor cell for this task only"]
```

### One Context-Aware Trial

```mermaid
sequenceDiagram
    participant F as Fugue
    participant H as Harbor
    participant C as Context runtime
    participant A as Agent harness
    participant M as Model provider
    participant W as W&B Weave

    F->>F: Resolve routes and planned cell
    F->>H: Start task-specific JobConfig
    H->>C: Mount prepared context artifact
    H->>A: Start isolated agent trial
    A->>C: Request repository evidence
    C-->>A: Return bounded ranked evidence
    C-->>H: Write normalized retrieval telemetry
    A->>M: Call target model
    M-->>A: Return model response
    A-->>W: Trace agent and model operations
    A-->>H: Write result and artifacts
    H-->>F: Return trial status and local artifacts
    F-->>W: Trace orchestration and scoring operations
```

Start with the smoke preset:

```bash
fugue context list
fugue preflight --experiment repo-memory-impact --preset smoke --no-live
fugue bridge up --model wandb/zai-org/GLM-5.2

# Build only the context artifacts needed by the selected preset.
fugue context prepare --experiment repo-memory-impact --preset smoke \
  --systems none,agentsmd,rag-bm25

# Preview first; this never builds indexes or downloads benchmark data.
fugue run --experiment repo-memory-impact --preset smoke --dry-run \
  --systems none,agentsmd,rag-bm25
fugue run --experiment repo-memory-impact --preset smoke \
  --systems none,agentsmd,rag-bm25 \
  --run-name repo-memory-smoke-v1

fugue export --jobs jobs/repo-memory-impact .fugue/runtime \
  --out reports/repo-memory-smoke-v1.jsonl \
  --judge-model openai/gpt-5-mini --fetch-weave --to-weave
```

With every optional dependency ready, the complete smoke definition expands to
89 cells and 95 evaluations: three
ranked retrieval cases for each controlled RAG system, one repository-QA and
one coding task across eligible context systems and all four harnesses, and
one short continuity sequence per longitudinal system. The TUI and operator preview
show this breakdown before launch. Missing prerequisites and unsupported cells
remain visible as `not applicable` and are excluded from estimated trial count.

The QA lane uses a deterministic 24-repository selection from the
MIT-licensed SWE-QA-Pro benchmark because it includes exact repository commits.
Fugue downloads the pinned, checksum-verified source and atomically materializes
local Harbor tasks under `.fugue/cache/datasets/`; questions and reference
answers are not copied into this repository. Harbor verifies output format.
`fugue export --judge-model ...` separately scores correctness, completeness,
and groundedness so format completion is never presented as answer accuracy.

Fugue-owned sidecars record one normalized event for each logical retrieval.
Eligible third-party stdio MCP servers are wrapped by `fugue.mcp_proxy`, a
transparent JSON-RPC relay that distinguishes proxy and upstream events. It
records tool name, bounded/redacted arguments, response size, errors, and
latency while leaving upstream schemas and responses unchanged. Full responses
stay in local Harbor artifacts. Weave receives normalized scores, metadata,
and bounded evidence paths rather than raw repository content.

The `none` and `markdown-log` baselines expose explicit retrieval behavior, so
they receive real recall measurements rather than disappearing from retrieval
comparisons. File-level MRR, NDCG, recall, and precision deduplicate chunks by
canonical path and stay within `[0, 1]`; raw chunk counts remain available.
Preparation reports build latency, process-tree CPU and peak memory, index
size, and cache hits. Builder token/cost fields remain `N/A` unless measured
directly.

Letta Code is available as the pinned, opt-in Harbor adapter
`fugue.agents:FugueLetta`. It is deliberately not modeled as a context system:
Letta owns the agent loop, conversation, and MemFS. Its local backend is
isolated under each Harbor trial's agent logs and exported alongside the study
as a separate stateful-harness result. The default context matrix remains
Hermes, OpenClaw, Claude Code, and Codex so portable context-system results are
not conflated with a different agent architecture.

Third-party systems remain opt-in local dependencies. `fugue context
preflight` names missing commands, Python extras, environment variables, and
license gates. GitNexus is excluded from presets until
`FUGUE_LICENSE_APPROVED_GITNEXUS=true` is set because its PolyForm
Noncommercial license requires explicit approval. The `full` preset also
requires a separate `FUGUE_JUDGE_MODEL` and materializes remote benchmark data
only during explicit preparation or a live run, never during preview. A live
`fugue run` prepares missing task-specific indexes automatically; `fugue
context prepare` remains useful for warming caches and measuring build cost
separately. The lat.md adapter is experimental and should only be used after
its opt-in runtime integration check passes.

Live preparation, retrieval, ingestion, trial, and scoring operations trace to
Weave during execution. Direct workload runners write normalized rows locally;
`fugue export --to-weave` is the sole evaluation publisher. A ledger under
`.fugue/runtime/publications/` prevents duplicate publication. Use
`--republish` only when duplicate publication is intentional.

### Results And Publication

Live operations are traced as they happen. Normalized evaluation rows have one
publisher: `fugue export`. The local ledger prevents an accidental second
publication of the same row to the same project.

```mermaid
flowchart LR
    HARBOR["Harbor result.json and artifacts"] --> JOIN["fugue export"]
    META["Fugue metadata and cell state"] --> JOIN
    EVENTS["Context telemetry and direct workload rows"] --> JOIN
    SPANS["Optional fetched Weave spans"] --> JOIN

    JOIN --> NORMALIZE["Normalize typed result rows"]
    NORMALIZE --> FILES["JSONL or Parquet report"]
    NORMALIZE --> LEDGER{"Already published to this project?"}
    LEDGER -->|"no"| EVAL["Weave EvaluationLogger"]
    LEDGER -->|"yes"| SKIP["Skip unless --republish"]

    LIVE["Preparation, retrieval, ingestion, trial, and scoring"] --> TRACES["Live Weave operation traces"]
```

W&B traces default to the **Fugue Experiments** project at
`wandb/fugue-experiments`; override `WANDB_ENTITY`,
`WANDB_PROJECT`, or `WEAVE_PROJECT` only when you intentionally want a
different trace project. Use `--run-name` and `--tags` to separate experiments
inside the same project.

## AI Experiment Copilot And Analyst

Fugue has two grounded operator agents. The composer turns a natural-language
experiment idea into a real `ExperimentSpec`; the analyst finds a reproducible
result cohort and explains deterministic Fugue metrics. Neither agent receives
shell access, environment values, or an unrestricted filesystem tool.

```mermaid
flowchart LR
    USER["Natural-language experiment request"] --> COMPOSER["Fugue experiment composer"]
    CATALOG["Repo experiments, manifests, prompts, skills, context systems"] --> COMPOSER
    COMPOSER --> DRAFT["Untrusted structured draft"]
    DRAFT --> VALIDATE["ExperimentSpec parsing and reference validation"]
    VALIDATE --> PREVIEW["Side-effect-free Harbor matrix preview"]
    PREVIEW --> REVIEW{"Explicit operator action"}
    REVIEW -->|"Apply"| FORM["Unsaved Compose state"]
    REVIEW -->|"Save"| REPO["configs/fugue/experiments"]
    REVIEW -->|"Run"| SNAPSHOT["Immutable runtime experiment snapshot"]
    SNAPSHOT --> HARBOR["Harbor cells"]
```

Draft from the CLI. Without `--save` or `--run`, this only prints the validated
proposal and exact matrix:

```bash
fugue compose "Compare BM25 and no context across all harnesses for one task"
fugue compose "Create a PDF skill A/B based on skillsbench-pdf-ab" --save pdf-v2
fugue compose "Run a one-task smoke test" --run
```

`FUGUE_COMPOSER_MODEL` selects the composer model. It falls back to the active
experiment model and then `FUGUE_MODEL`. In the TUI, expand **Ask Fugue to
compose an experiment** on Compose. `Apply` changes the current form without
writing files; `Save draft` and `Run draft` are separate explicit actions.

The analyst uses a rebuildable local catalog to bucket records by benchmark,
workload, intervention, experiment, variant, harness, context system, provider,
model, status, and time. Hybrid analysis starts from normalized local rows and
enriches the selected cohort from Weave when credentials and connectivity are
available.

```mermaid
flowchart TD
    CONFIGS["Saved experiment definitions and content hashes"] --> CATALOG["SQLite experiment catalog"]
    RUNTIME["Durable run state and Harbor trial results"] --> CATALOG
    REPORTS["Normalized JSONL exports and local artifacts"] --> CATALOG
    WEAVE["Selected Weave calls, costs, conversations, and traces"] --> CATALOG
    QUESTION["Natural-language analysis question"] --> SCOPE["Deterministic filters and grouping plan"]
    CATALOG --> SCOPE
    SCOPE --> SNAPSHOT["Immutable row-id snapshot"]
    SNAPSHOT --> METRICS["Deterministic pass, reward, latency, token, cost, failure, tool, and retrieval metrics"]
    METRICS --> ANALYST["Evidence-constrained interpretation"]
    ARTIFACTS["Bounded redacted artifact excerpts"] --> ANALYST
    ANALYST --> OUTPUTS["report.md, analysis.json, scope.json, evidence.jsonl"]
    OUTPUTS --> LINKS["Local evidence and Weave conversation links"]
```

```bash
fugue catalog refresh --source hybrid
fugue catalog facets
fugue analyze "Which context system improved coding pass rate without excessive latency?"
fugue analyze "Compare the PDF skill lift by harness" \
  --filter experiment_id=skillsbench-pdf-ab --save pdf-skill-lift
fugue analyses list
fugue analyses run pdf-skill-lift
```

Saved analysis definitions live in `configs/fugue/analyses`. Each analysis run
writes an immutable evidence bundle under `reports/analyses/<analysis-id>/`.
Every generated finding must cite a concrete evidence id. Fugue computes the
numbers; the analyst only interprets them. Mixed workload/model cohorts are
stratified and called out instead of being presented as one lift estimate.

Composer and analyst sessions are traced as `fugue-experiment-composer` and
`fugue-analysis-agent` in Weave Agents. Full trace mode includes requests,
responses, reports, and tool evidence; metadata mode omits those bodies.

## Terminal Operator

Bare `fugue` opens a keyboard-first Textual workspace. The TUI and headless
commands call the same presentation-neutral operator services; neither owns
experiment execution. Secrets are represented only as present or missing.

```text
  FUGUE / AGENT EXPERIMENT OPERATOR
  HERMES       ··················  pending
  OPENCLAW     ······■···········  running
  CLAUDE CODE  ··················  pending
  CODEX        ··················  pending

  [1 Compose] [2 Runs] [3 Results] [4 Setup]
```

- **Compose** loads or saves an experiment, edits variant prompt/skill/context
  choices, selects target/builder/judge models, previews exact cells, and starts
  attached or detached work.
- **Runs** shows durable runs, a harness-by-variant cell matrix, combined or
  per-cell logs, cancellation, export, and Weave Agents actions.
- **Results** summarizes local Harbor trial outcomes by harness, variant, and
  context system. Detailed conversations and traces stay in Weave.
- **Setup** checks selected model keys, W&B project, Docker, Harbor, bridge, and
  full-versus-metadata trace policy without displaying secret values.

```mermaid
flowchart LR
    SETUP["1. Setup routes and tools"] --> COMPOSE["2. Compose experiment matrix"]
    COMPOSE --> PREVIEW["3. Preview exact cells and commands"]
    PREVIEW --> LAUNCH["4. Launch or detach"]
    LAUNCH --> RUNS["5. Follow cells and logs"]
    RUNS --> EXPORT["6. Export and enrich results"]
    EXPORT --> RESULTS["7. Compare local summaries"]
    RESULTS --> AGENTS["8. Inspect behavior in Weave Agents"]
    RUNS --> HARBOR["Inspect artifacts with harbor view jobs"]
```

Useful keys are `1` through `4` for screens, `/` for command search, `?` for
help, `r` to run, `d` to detach, `c` to cancel, `e` to export, `a` for Weave
Agents, and `w` for the selected conversation. Set `FUGUE_NO_ANIMATION=1` or
`NO_COLOR=1` for a static terminal.

Direct screen navigation and headless operation remain available:

```bash
fugue tui --screen setup
fugue status --experiment repo-memory-impact
fugue runs list
fugue runs show RUN_ID
fugue runs logs RUN_ID --follow
fugue runs logs RUN_ID --cell CELL_ID --follow
fugue runs cancel RUN_ID
fugue runs export RUN_ID --fetch-weave --to-weave
fugue runs open RUN_ID --target agents
```

### Detached Execution

The operator launches every live run in its own process group. The child owns
the run even when the TUI exits; a new TUI discovers it from atomic state and
reattaches. Cancellation terminates the process group and marks unfinished
cells explicitly instead of leaving them in `running` forever.

```mermaid
sequenceDiagram
    participant T as Terminal operator
    participant S as Run supervisor
    participant C as Fugue child process group
    participant H as Harbor cells
    participant D as .fugue/runtime/RUN_ID

    T->>S: Launch experiment
    S->>D: Atomically write run.json
    S->>C: Start detached process group
    C->>D: Append events and cell states
    C->>H: Execute bounded concurrent cells
    H-->>D: Stream combined and per-cell logs
    T--xT: Exit or detach
    T->>D: Reopen and recover run
    T->>S: Cancel when requested
    S->>C: SIGTERM process group
    S->>D: Mark unfinished cells cancelled
```

### Weave Agents

Fugue uses stable agents (`hermes-agent`, `openclaw`, `claude-code`, and
`codex`) and filterable `fugue.*` attributes. A stateless Harbor trial is one
conversation with one agent turn; a continuity-capable harness can reuse a
cohort conversation key across episodes. Provider-only diagnostic sequences
remain separate from harness conversations. Native harness plugins remain
authoritative for turn, LLM, and tool spans, so Fugue does not add duplicate
wrappers.

This follows Weave's documented
[agent data model](https://docs.wandb.ai/weave/guides/tracking/trace-agents)
and uses the supported
[harness integrations](https://docs.wandb.ai/weave/guides/tracking/trace-agent-integrations).

```mermaid
flowchart TB
    AGENT["Stable harness agent"] --> CONV["Conversation: one trial or continuity cohort"]
    CONV --> TURN1["invoke_agent turn"]
    CONV --> TURN2["later continuity turn"]
    TURN1 --> LLM1["chat: model call"]
    TURN1 --> LLM2["chat: model call"]
    LLM1 --> TOOL1["execute_tool"]
    LLM2 --> TOOL2["execute_tool"]

    ATTRS["fugue.run_id, experiment, workload, harness, variant, context, task, trial, model, prompt, skills, tags"] --> TURN1
```

`trace_content` defaults to `full`, which can send prompts, responses,
reasoning, tool arguments, and tool results to Weave. Harness plugins do not
provide automatic PII scrubbing. Select `metadata` only for integrations that
can guarantee content suppression; preflight reports unsupported selections.
Export joins traces by conversation identity and `fugue.run_key`, then remains
the sole publisher of normalized evaluation rows.

## Environment

```bash
WANDB_API_KEY=          # Weave tracing; also model billing for wandb/...
WANDB_BASE_URL=https://api.wandb.ai
WANDB_ENTITY=wandb      # default trace entity
WANDB_PROJECT=fugue-experiments
FUGUE_RUN_NAME=         # optional; defaults to fugue-<UTC timestamp>
FUGUE_TAGS=             # optional comma-separated tags

OPENAI_API_KEY=        # model billing for openai/...
ANTHROPIC_API_KEY=     # model billing for anthropic/...

FUGUE_MODEL=wandb/zai-org/GLM-5.2
LITELLM_MASTER_KEY=sk-fugue-local

# Context-system evaluation.
FUGUE_BUILDER_MODEL=     # optional; defaults to the target model
FUGUE_JUDGE_MODEL=       # required by the full preset
LAT_LLM_KEY=             # only for lat.md semantic search
# FUGUE_ENABLE_EXPERIMENTAL_LATMD=true
FUGUE_GRAPHITI_URI=      # local Neo4j-compatible Graphiti endpoint
# FUGUE_LICENSE_APPROVED_GITNEXUS=true
```

Optional base URL overrides:

```bash
WANDB_INFERENCE_BASE_URL=https://api.inference.wandb.ai/v1
OPENAI_BASE_URL=https://api.openai.com/v1
ANTHROPIC_BASE_URL=https://api.anthropic.com
```

## Layout

```text
fugue/
├── fugue/
│   ├── agents/          # Harbor adapters and Weave plugin wiring
│   ├── bench/           # context providers, workload runners, render/export CLI
│   ├── bridge.py        # generated LiteLLM bridge config
│   ├── context_server.py # normalized context-search MCP server
│   ├── mcp_proxy.py     # transparent MCP telemetry relay
│   ├── model_plane.py   # provider routing
│   └── tui.py           # Textual terminal operator
├── Dockerfile.context   # pinned Fugue context MCP sidecar
├── datasets/pilot.yaml
├── configs/fugue/       # saved prompts, skills, and experiments
├── scripts/
├── tasks/
├── jobs/                # gitignored Harbor jobs and artifacts
├── reports/             # gitignored exports
└── .fugue/              # gitignored context cache, runtime, bridge, and JobConfigs
```

Inspect completed Harbor jobs and artifacts locally with:

```bash
harbor view jobs
```
