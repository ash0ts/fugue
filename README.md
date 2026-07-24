# Fugue

**Fugue is the governed experiment layer for agent loop engineering and
autoresearch.**

An outer research loop can inspect traces, form a hypothesis, and decide what
to try next. Fugue turns that proposal into a controlled experiment: it fixes
the comparison, locks the inputs, enforces human approval and budget limits,
runs isolated Agent cells through Harbor, reconciles each cell with native W&B
Weave evidence, and returns a result whose scope is explicit.

Fugue does not replace the research Agent, the execution sandbox, or the
observability system. It is the laboratory between them.

```mermaid
flowchart TB
    EVIDENCE["Production evidence<br/>Weave calls, evaluations, artifacts"]
    RESEARCHER["Researcher or outer loop<br/>Aria, Senpai, or a custom Agent"]
    FUGUE["Fugue<br/>design, locks, admission, evaluation"]
    HUMAN["Human operator<br/>approval and spend authority"]
    EXECUTION["Harbor + native Agent harness<br/>one isolated environment per cell"]
    WEAVE["W&B Weave<br/>conversations, traces, evaluations"]
    RESULT["Fugue sourced Result<br/>observation, interpretation, limitations"]
    CONSOLE["Optional Study Console<br/>read-only research projection"]

    EVIDENCE --> RESEARCHER
    RESEARCHER -->|"question, hypothesis, proposed comparison"| FUGUE
    HUMAN -->|"approve exact preview digest and cap"| FUGUE
    FUGUE -->|"locked task × treatment × attempt cells"| EXECUTION
    EXECUTION --> WEAVE --> RESULT
    FUGUE -->|"public-safe lifecycle events"| CONSOLE
```

## Why Fugue exists

Agent behavior belongs to the whole loop around the model. The harness decides
how tools are exposed and called. Retrieval changes what evidence is available.
Instructions change whether the Agent opens and uses that evidence. Runtime,
prompt, task selection, retry policy, and evaluation can all change the result.

That is what makes loop engineering useful—and what makes casual comparisons
easy to get wrong.

Without a governed experiment layer, an autoresearch loop can accidentally:

- change several dimensions while claiming to test one;
- compare different tasks, runtimes, prompts, or attempts across treatments;
- prepare dependencies or retrieval indexes inside a live trial;
- retry failures until one arm looks better;
- count infrastructure failures as task failures;
- lose the route, conversation, root trace, or exact source revision;
- write a confident conclusion from incomplete or irreconcilable evidence.

Fugue makes these failure modes explicit. It declares what is **held fixed**,
what is **changed**, and what is **measured**; resolves the full matrix before
execution; and keeps operational health, Agent execution, task success,
authored evaluation, evidence integrity, latency, usage, and cost as separate
outcomes.

The goal is not to manufacture certainty. The goal is to produce a bounded,
inspectable claim that can support the next research decision.

## Where Fugue sits in an autoresearch loop

The outer loop remains the researcher. Fugue remains the laboratory.

| Layer | Owns | Deliberately does not own |
| --- | --- | --- |
| Researcher or outer loop | Evidence review, hypotheses, intervention ideas, research priority, interpretation, the next question | Execution authority, self-approval, hidden changes to an accepted design |
| Fugue | Experiment validation, immutable identity, matrix expansion, preparation, approval checks, admission, scheduling, recovery, evaluation, evidence reconciliation | Hypothesis generation, winner selection, writable application code, raw trace storage |
| Harbor | One isolated environment for each admitted Agent cell | Experiment design, scoring, research memory |
| W&B Weave | Agent conversations, calls, evaluations, annotations, prediction-level evidence | Admission policy, runtime isolation, experiment locking |
| Study Console or another sink | A public-safe view of why the Study exists, its design, live progress, result, and evidence links | Execution, approval, retries, copied trace bodies |

This division is important. An Agent can say, “these production traces suggest
the support loop is over-sharing data; compare three safer loop designs.” Fugue
can validate and preview that comparison. A human approves the exact six cells
and cost ceiling. Fugue runs them and returns the task-aligned evidence. The
Agent can then interpret the result and propose another Study—but it cannot
quietly approve or launch that follow-up itself.

```mermaid
flowchart TB
    OBSERVE["1. Observe<br/>review traces and failures"]
    HYPOTHESIZE["2. Hypothesize<br/>name one explanation"]
    DESIGN["3. Propose<br/>fixed, varied, measured"]
    PREVIEW["4. Fugue preview<br/>exact cells and cost"]
    APPROVE{"5. Human approval?"}
    RUN["6. Fugue run<br/>lock, admit, isolate"]
    EVALUATE["7. Evaluate<br/>task, behavior, evidence"]
    INTERPRET["8. Interpret<br/>result and limitations"]
    NEXT["9. Choose the next justified question"]

    OBSERVE --> HYPOTHESIZE --> DESIGN --> PREVIEW --> APPROVE
    APPROVE -->|yes| RUN --> EVALUATE --> INTERPRET --> NEXT
    APPROVE -->|no| DESIGN
```

## What Fugue guarantees

- **One declared matrix.** Every task, treatment, harness, and attempt is known
  before execution.
- **Pure preview.** Preview estimates applicability, cells, calls, and reserved
  cost without preparing assets, calling a model, or creating runtime state.
- **Locked inputs.** Source revisions, task suites, candidates, prompts,
  runtimes, context, skills, routes, and private evaluation assets are
  committed before the first cell runs.
- **Exact approval.** Paid Agent, judge, or interactor calls require approval
  bound to one preview digest, cell limit, and spend ceiling.
- **Isolated execution.** Each cell runs through Harbor without access to the
  Docker socket, host paths, dependency installation, downloads, builds, or
  service startup.
- **Durable recovery.** Operation IDs, leases, and immutable run identities
  make control-plane retries safe. An already-launched trial is reconciled,
  never silently launched again.
- **Reconciled evidence.** Every coordinate becomes terminal, explicitly not
  applicable, or cancelled. Agent cells must resolve to one native
  conversation, one `invoke_agent` root, route/runtime receipts, and one
  normalized prediction row.
- **Bounded conclusions.** Results retain their task, treatment, attempt,
  evidence, uncertainty, exclusions, and limitations. Fugue does not emit a
  universal model or harness ranking.

## Architecture

Fugue has one execution path. The CLI, Python SDK, REST API, and MCP interface
all converge on the same campaign and operator services; none introduces a
second runner.

```mermaid
flowchart TB
    subgraph CLIENTS["Human and Agent interfaces"]
        direction LR
        CLI["Human<br/>CLI and TUI"]
        AGENT_API["External Agent<br/>Python, REST/SSE, or MCP"]
    end

    subgraph CONTROL["Governed control plane"]
        direction LR
        RESEARCH["Research service<br/>Research, Study, Result"]
        CAMPAIGN["CampaignService<br/>policy, preview, preparation, admission"]
        APPROVAL["Operator approval ledger<br/>exact digest, cells, cost"]
        OPERATOR["OperatorService<br/>canonical plan and run lifecycle"]
        RESEARCH --> CAMPAIGN --> OPERATOR
        APPROVAL --> CAMPAIGN
    end

    subgraph EXECUTION["Private execution plane"]
        direction LR
        LOCKS["Content-addressed locks<br/>taskset, candidate, runtime, evaluation"]
        WORKER["Durable worker<br/>leases and recovery"]
        CELL["Harbor + native Agent harness<br/>one isolated environment per cell"]
        LOCKS --> WORKER --> CELL
    end

    subgraph EVIDENCE["Evidence systems"]
        direction LR
        WEAVE["W&B Weave<br/>calls, conversations, evaluations"]
        ROW["PredictionRowV1<br/>normalized terminal outcome"]
        STORE["Append-only Study store + outbox<br/>revisions, cursors, public-safe events"]
        UI["Optional Study Console"]
        WEAVE --> ROW --> STORE --> UI
    end

    CLI --> OPERATOR
    AGENT_API --> RESEARCH
    OPERATOR --> LOCKS
    CELL --> WEAVE
    CELL --> ROW
```

The canonical artifact path is equally important:

```mermaid
flowchart TB
    SPEC["ExperimentSpec<br/>declared comparison"]
    PLAN["ResolvedRunPlan<br/>pure exact matrix"]
    CANDIDATE["ResolvedCandidate<br/>behavioral identity"]
    CELL["PlannedCell<br/>task × treatment × attempt"]
    SNAPSHOT["RunSnapshotV1<br/>immutable execution receipt"]
    HARBOR["Rendered Harbor job"]
    WEAVE["Native Weave evidence"]
    ROW["PredictionRowV1<br/>normalized outcome"]
    ANALYSIS["Evaluation and analysis"]

    SPEC --> PLAN
    PLAN --> CANDIDATE
    PLAN --> CELL
    CANDIDATE --> SNAPSHOT
    CELL --> SNAPSHOT
    SNAPSHOT --> HARBOR
    HARBOR --> WEAVE
    HARBOR --> ROW
    WEAVE --> ROW
    ROW --> ANALYSIS
```

Candidate identity contains behavior-changing inputs such as the harness,
model route, prompt, reviewed skills, context interface, integrations, and
Agent configuration. Runtime, Harbor, tracing, and scheduling live in a
separate execution fingerprint. Dataset, task, and attempt remain explicit
comparison coordinates. Fugue never reconstructs identity from labels or UI
state.

## When to use Fugue

Use Fugue when you need to:

- compare harnesses or loop implementations on aligned tasks;
- test a prompt, tool, repository search, memory, interaction, or policy change
  while holding the intended controls fixed;
- turn reviewed production-trace failures into locked discovery and holdout
  tasks;
- evaluate deterministic task success alongside tool, trace, artifact, diff,
  or blind-judge criteria;
- let Aria, Senpai, or another external research Agent propose and monitor
  experiments while a human retains approval and spend authority;
- preserve a durable chain from a research question to a controlled Study,
  admitted Run, evaluation, and sourced Result.

Use something simpler for a one-off prompt check, ordinary trace inspection, or
a disposable local benchmark where identity, isolation, restart recovery,
approval, and evidence provenance do not matter.

Fugue does not host your application, edit its writable checkout, invent the
next hypothesis, approve its own spend, copy raw Weave traces, or turn a small
study into a general capability claim.

## Choose an interface

Use the operator CLI or TUI when a human is directly designing and running a
saved experiment:

```text
ExperimentSpec → preview → prepare → run → normalized evidence → analysis
```

Use the Research service when Aria or another outer loop needs a governed
laboratory:

```text
Research → controlled Study → admitted Run → evaluation → sourced Result
```

The Research service is bearer-authenticated and available through Python,
REST, and MCP. Pure preview remains free of preparation and model calls. A
private worker is the only service allowed to operate Harbor, and paid work
requires a separate operator approval bound to the exact preview and cost cap.
Weave remains the prediction-level evidence system. Optional consoles consume a
generic, public-safe research-record projection rather than copied trace bodies
or private evaluation data.

Fugue 0.1.2 supports Hermes, OpenClaw, Claude Code, and Codex as stable harness
identities. Direct provider diagnostics remain separate from Agent cells and do
not synthesize conversations or Agent roots.

Start with:

- [`docs/research.md`](docs/research.md) for the Python, REST, and MCP model;
- [`docs/research-container.md`](docs/research-container.md) for the isolated
  control/worker deployment;
- [`docs/campaigns.md`](docs/campaigns.md) for the governed execution boundary;
- the
  [`autoresearch-loop` example](examples/research/autoresearch-loop/README.md)
  for a Result-to-child-Study handoff;
- the
  [`retrieval-to-action` example](examples/research/retrieval-to-action-canary/README.md)
  for a complete external-Agent approval and run flow.

## Install

[`uv`](https://docs.astral.sh/uv/) is the recommended environment manager.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
```

Keep credentials outside the checkout and pass their existing path directly to
`--env-file`. Fugue reads the file but never copies it into runtime artifacts,
snapshots, jobs, or Git. `.env.example` lists the supported names:

```dotenv
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_PROJECT=fugue-experiments

OPENAI_API_KEY=
ANTHROPIC_API_KEY=
FUGUE_MODEL=openai/gpt-5
```

Model selection precedence is CLI override, experiment configuration,
environment, then Fugue’s default. Builder and judge models are independent
roles and must be selected explicitly when their features are used.

```mermaid
flowchart TB
    HOME["fugue command center"]
    HOME --> P["plan: design an experiment"]
    HOME --> R["run: preview or execute"]
    HOME --> RS["runs: inspect, log, cancel, export, package"]
    HOME --> A["analyze: resolve scope, then report"]
    HOME --> S["setup: check, bridge, skills, context"]
    HOME --> T["tui: full-screen workspace"]
```

## Preview and run

Preview never downloads sources, prepares context, calls a model, writes job
configuration, or mutates runtime state:

```bash
fugue run pilot --preview
```

Check dependencies and prepare only what the selected experiment requires:

```bash
fugue setup --experiment pilot --check
fugue setup --experiment pilot --skills
fugue setup --experiment repo-memory-impact --prepare \
  --env-file /path/to/existing/.env
```

`--check` is observational. `--prepare` is the plan-resolved state-changing
boundary: it content-locks remote workload data, context indexes, harness
runtimes, and unique task images for the selected architecture. Repeated setup
reuses verified locks. `--prepare-context` remains a narrower adapter-only
action. Preview and active trials never install, download, build, pull, start a
service, or use the Docker socket. Graphiti has a separate local Neo4j lifecycle:

```bash
fugue setup --experiment repo-memory-impact --systems graphiti \
  --start-services --env-file /path/to/existing/.env
fugue setup --experiment repo-memory-impact --systems graphiti \
  --service-status --env-file /path/to/existing/.env
fugue setup --experiment repo-memory-impact --systems graphiti \
  --stop-services --env-file /path/to/existing/.env
```

When Graphiti credentials are absent, Fugue generates an ignored mode-0600
credential file and resolves host and Harbor endpoints internally. Stopping the
service preserves its named data volume; 0.1.2 has no destructive purge action.

Remote skills are fetched for review but not executed. Approve one exact
reviewed digest before it can enter a run:

```bash
fugue setup --approve-skill hallmark=sha256:REVIEWED_DIGEST \
  --acknowledge-risk network-access
```

Start a run and wait, or return while the same durable worker continues:

```bash
fugue run pilot
fugue run pilot --detach
```

Before the first cell starts, `OperatorService` resolves the full plan,
persists the experiment snapshot, prepares context, renders jobs, plans cells,
and atomically writes `.fugue/runtime/RUN_ID/input-lock.json`. A failure before
that commit leaves the run failed in its `starting` phase and executes no cell.

```mermaid
flowchart TD
    REQUEST["ExperimentRequest"] --> RESOLVE["Resolve and validate exact plan"]
    RESOLVE --> SNAPSHOT["Persist experiment snapshot"]
    SNAPSHOT --> CONTEXT["Prepare required context"]
    CONTEXT --> RENDER["Render jobs and planned cells"]
    RENDER --> LOCK["Atomically write input-lock.json"]
    LOCK --> RUNNING["Transition run to running"]
    RUNNING --> WORKERS["Start bounded workers"]
    RESOLVE -->|Failure| FAILED["Failed in starting; zero cells"]
    SNAPSHOT -->|Failure| FAILED
    CONTEXT -->|Failure| FAILED
    RENDER -->|Failure| FAILED
    LOCK -->|Failure| FAILED
```

## Inspect runs

Run operations use nested actions:

```bash
fugue runs
fugue runs RUN_ID
fugue runs RUN_ID logs
fugue runs RUN_ID logs --follow
fugue runs RUN_ID cancel
fugue runs RUN_ID export --out reports/run.jsonl --fetch-weave
fugue runs RUN_ID open agents
fugue runs RUN_ID open evaluation
fugue runs RUN_ID open evaluation --cell CELL_ID
```

Each run groups cells by behavioral candidate and shows deterministic benchmark
passes and failures separately from execution failures, unscored cells,
pending cells, and not-applicable cells. Completeness follows the execution
lifecycle; packageability follows the benchmark outcome. The terminal displays
a unique candidate prefix; JSON and snapshots retain the full SHA-256
identifier.

Live runs publish one Weave evaluation per candidate and workload. Fugue keeps
the returned evaluation URLs in the run manifest and attaches each verified
agent root to its prediction with Weave's GenAI span reference. Open the
evaluation to compare candidates, then select a prediction to navigate into the
linked agent conversation and trace.

```mermaid
flowchart LR
    PLAN["Planned cells"] --> SCOPE["Dataset + workload + examples + scorers"]
    SCOPE --> DATASET["Shared Weave Dataset"]
    SCOPE --> EVAL["Evaluation definition"]
    PLAN --> CANDIDATES["Resolved candidate models"]
    CANDIDATES --> PREDICT["Open prediction before Agent execution"]
    EVAL --> PREDICT
    PREDICT --> AGENT["invoke_agent / chat / execute_tool"]
    AGENT --> LINK["Verified genai_span_ref"]
    LINK --> SCORES["Outcome, latency, usage, and errors"]
    SCORES --> SUMMARY["Evaluation summary and stable URL"]
    PLAN --> DIRECT["Direct provider diagnostics"]
    DIRECT --> POSTHOC["Post-hoc scored rows; Agent link N/A"]
```

```mermaid
sequenceDiagram
    participant F as Fugue worker
    participant W as Weave EvaluationLogger
    participant H as Harbor cell
    participant A as Native harness integration
    F->>W: Open prediction for shared example
    W-->>F: Exact predict_and_score call ID
    F->>H: Execute with canonical correlation attributes
    H->>A: Run one Agent trial
    A->>W: One invoke_agent root with nested chat and tools
    H-->>F: Harbor result and artifacts
    F->>W: Resolve exact root and attach genai_span_ref
    F->>W: Finalize scores and prediction output
    F->>W: Log evaluation summary
```

This follows Weave's documented [EvaluationLogger](https://docs.wandb.ai/weave/guides/evaluation/evaluation_logger),
[Agent data model](https://docs.wandb.ai/weave/guides/tracking/trace-agents),
[supported Agent integrations](https://docs.wandb.ai/weave/guides/tracking/trace-agent-integrations),
and [Agent activity views](https://docs.wandb.ai/weave/guides/tracking/view-agent-activity).
Only agent-backed predictions receive conversation deep links. Retrieval and
continuity diagnostics remain ordinary Weave operations and report Agent
linking as `not_applicable`; Fugue never constructs undocumented URLs or fake
Agent conversations.

Candidate identity contains only behavior-affecting inputs: harness, provider
and model route, prompt digest, reviewed skill digests, context definition and
delivery, typed integrations, and advanced agent configuration. Experiment
names, variant IDs and labels, preset names, run names, judge/scorer state, and
trial ordinals do not affect it. Runtime, Harbor, concurrency, and tracing
policy instead affect a separate execution fingerprint.

Fugue preserves each harness's native model protocol: Hermes and OpenClaw use
Chat Completions, Claude Code uses Messages, and Codex uses Responses. A
provider is called directly when it exposes that protocol; otherwise a local
LiteLLM bridge translates the wire format without changing candidate identity.
Bridged runs fail preflight unless the running container uses the pinned image
digest and exact locked configuration. The snapshot, per-trial metadata, and
Agent root attributes record the expected protocol, direct-or-bridge endpoint
class, and upstream host so exported evidence can reconcile the route.

Model routing and tool delivery remain independent. Each Codex native-MCP cell
receives a new, isolated `CODEX_HOME` containing only its resolved allowlisted
servers; MCP payloads never pass through the model bridge. Fugue never reads or
mutates the user's global Codex credentials, skills, configuration, or MCP
definitions, and it never downgrades native MCP to portable instructions. The
locked runtime contains Codex 0.143.0 and weave-codex 0.1.1; trial startup
verifies the exact MCP inventory before the model turn.

## Experiment contract

Saved experiments live in `configs/fugue/experiments/`. The public YAML schema
is strict: use `skills`, use `context.delivery`, and select typed integrations.
Raw MCP server configuration is an internal rendering detail.

```yaml
id: search-comparison
title: Repository search comparison
manifest: datasets/pilot.yaml
model: openai/gpt-5
harnesses: [codex]

integrations:
  - id: shared-observer

variants:
  - id: baseline
    label: Baseline
    context: {system_id: none, delivery: portable}

  - id: treatment
    label: Reviewed search treatment
    prompt_id: search-instructions
    skills: [hallmark]
    context: {system_id: agentsmd, delivery: portable}
    integrations:
      - id: repository-search
        config: {top_k: 10}
```

Experiment integrations apply to every variant. Variant integrations are
additions. Duplicate IDs in the effective list are rejected; there is no
inherit/replace/null tri-state. To vary configuration, declare the integration
only on the variants that need it.

Context definitions declare their supported deliveries. `portable` never
injects native MCP, while `native_mcp` preserves the provider interface.
Selecting an unsupported delivery makes the cell `not_applicable` before
binding. Research adapters remain outside default presets until their pinned
Harbor runtimes pass live integration tests.

The experimental managed adapters are opt-in:

| System | Delivery | Explicit requirement |
| --- | --- | --- |
| GitNexus 1.6.3 | native MCP | `FUGUE_LICENSE_APPROVED_GITNEXUS=true`; noncommercial research only |
| CodeGraph 0.9.0 | native MCP | Prepared pinned platform bundle |
| Semble 0.5.1 | native MCP | Prepared local model and parser assets |
| Project RAG `d5abf98…` | native MCP | Prepared Rust runtime and isolated LanceDB state |
| Graphiti 0.29.2 | portable or native MCP | Managed Neo4j 5.26 service or explicit compatible endpoint |
| OpenWiki 0.1.1 | portable | Isolated builder workspace and Fugue model bridge |
| lat.md 0.11.0 | native MCP | `LAT_LLM_KEY` and `FUGUE_ENABLE_EXPERIMENTAL_LATMD=true` |

GitNexus, CodeGraph, Semble, Project RAG, and lat.md receive the same read-only
task snapshot at `/workspace/repository` and a separate writable state area.
The Fugue gateway preserves upstream MCP schemas and cancellation while adding
cell correlation to tool results. Without a lat.md key, semantic lat.md cells
are deterministically `not_applicable` and are not a release failure.

GitNexus has two strict candidates. Their mode and model assets participate in
candidate identity and context cache keys:

```yaml
context:
  system_id: gitnexus
  delivery: native_mcp
  config: {retrieval_mode: bm25}
```

```yaml
context:
  system_id: gitnexus
  delivery: native_mcp
  config:
    retrieval_mode: hybrid_vector
    embedding_model: Snowflake/snowflake-arctic-embed-xs
    embedding_revision: d8c86521100d3556476a063fc2342036d45c106f
    embedding_dimensions: 384
    vector_required: true
```

Hybrid setup bundles and verifies the 384-dimensional model and CPU ONNX
runtime, builds the index with networking disabled, and runs a semantic probe.
It fails closed for an absent index, zero embeddings, wrong dimensions, failed
vector query, or GitNexus's 50,000-node vector limit. It never reports BM25
fallback as a vector result.

```mermaid
flowchart LR
    TASK["Dataset + task + repository commit"] --> KEY["Content-addressed key"]
    KEY --> LOCK["Process-safe build lock"]
    LOCK --> PREPARE["Context provider prepare"]
    PREPARE --> CACHE["Atomic cache publication"]
    CACHE --> DELIVERY{"Declared delivery"}
    DELIVERY -->|portable| PORTABLE["Instructions, artifacts, or mounts"]
    DELIVERY -->|native_mcp| MCP["Preserved upstream MCP interface"]
    PORTABLE --> CELL["Harbor cell"]
    MCP --> CELL
    DELIVERY -->|unsupported| NA["not_applicable before binding"]
```

## Plan in Rich or Textual

Run bare `fugue` for the Rich command center, or open the full workspace:

```bash
fugue
fugue tui
```

Textual keeps one in-memory plan:

```mermaid
flowchart LR
    DEFINE["Define: intent or saved experiment"]
    COMPARE["Compare: variants, coverage, proposals"]
    REVIEW["Review: exact matrix and launch authority"]
    RUNS["Runs: cells, candidates, and logs"]
    DEFINE --> COMPARE --> REVIEW --> RUNS
    COMPARE -->|Accept proposal| MEMORY["Update PlanState only"]
    MEMORY --> COMPARE
    REVIEW -->|Missing setup| SETUP["Setup: credentials, skills, context"]
    SETUP --> REVIEW
```

- Define selects intent or a saved experiment.
- Compare shows variants, evaluation coverage, and generated-evaluation
  proposals.
- Review owns the exact matrix and launch authority.

Proposals update the plan only after acceptance and still require an explicit
save before execution. Multi-file evaluation saves validate and stage every
asset, write the experiment last as the commit marker, and remove newly created
orphan assets if saving fails. Agent presets live under Advanced and start a
new plan from their declared base experiment; a dirty plan requires a
replacement-diff confirmation.

Natural-language planning is explicit and produces an untrusted draft that is
parsed and previewed before it can be saved:

```mermaid
sequenceDiagram
    participant U as User
    participant C as Experiment planner
    participant O as OperatorService
    participant V as Rich or Textual
    U->>C: Describe a controlled comparison
    C->>O: Read the repository catalog
    C->>O: Propose a strict ExperimentSpec
    O->>O: Validate and preview without side effects
    O-->>V: Draft, diff, cells, trials, and blockers
    V-->>U: Review, save, run, or discard
    U->>V: Explicit approval
```

```bash
fugue plan \
  "Compare BM25 with no context across every harness for one coding task" \
  --from repo-memory-impact

fugue plan "Create a smaller PDF skill comparison" \
  --from skillsbench-pdf-ab \
  --save pdf-skill-smoke
```

## Results and analysis

Local export is normalized JSONL. Comparison example identity contains only
dataset, workload, and task; trial index is a separate cell coordinate.
Deterministic outcomes, rubric scores, and judge errors remain separate—Fugue
does not invent a composite score or convert a judge outage into a Harbor
failure. Unmeasured token usage remains unavailable rather than becoming zero.

`RunSnapshotV1` records source and resolved experiment digests, capability
decisions, logical predictions, runtime locks, planned cells, asset identities,
cohort identity, treatment-selection digest, evaluation-lock digest, and
publication schema before execution. This is the first public contract, so
earlier development snapshots are intentionally unreadable. A normalized
prediction is distinct from its raw retrieval or episode
measurements. Summaries therefore report planned and executed cells, logical
predictions, measurements, Agent and direct predictions, conversations, links,
canaries, and remediation cohorts separately.

Answer-bearing localization data lives in a mode-0600 host-only
`EvaluationAssetLockV1`. Fugue never mounts its raw paths in a task or Agent
container and never puts them in rendered jobs, snapshots, Agent metadata, or
trace inputs. Host-side scoring may publish derived recall and MRR together
with the lock digest.

W&B publication is idempotent by project, prediction ID, scorer version, and
revision. An explicit republish creates a new active revision with a reason and
`supersedes` link; it never merges evidence by a display label. Dashboard views
should filter to the active revision and facet by source commit, snapshot
digest, cohort, execution kind, harness, context system, and skill treatment.

### Public Experiment Atlas

The static Experiment Atlas is a reviewed evidence publication layer, not a
browser frontend for the operator. `tools/experiment_atlas.py` reduces
canonical normalized JSONL through an explicit allowlist, verifies the source,
dataset, coordinates, and digest against immutable run snapshots, recomputes
metrics, and writes versioned `PublicExperimentV1` snapshots plus an ordered
`ExperimentIndexV1`. Editorial YAML explains the question and decision; it
cannot supply or override results, links, or provenance.

Public snapshots may contain task IDs, fixed routes, harnesses, treatments,
outcomes, efficiency measurements, provenance, and verified Weave links. They
reject prompts, outputs, reasoning, tool content, gold data, exceptions, local
paths, environment values, secrets, unknown fields, and unapproved URL
domains. GitHub Pages builds only the committed snapshots. It receives no W&B
credential and makes no browser API call; Weave links are user-initiated and
explicitly labeled as requiring sign-in.

```bash
uv run python tools/experiment_atlas.py \
  --editorial-dir atlas/editorial \
  --rows EXPERIMENT_ID=reports/RUN_ID.jsonl \
  --run-summary EXPERIMENT_ID=.fugue/runtime/RUN_ID/run.json \
  --snapshot EXPERIMENT_ID=.fugue/runtime/RUN_ID/input-lock.json \
  --output atlas/public/data
npm --prefix atlas ci
npm --prefix atlas run build
python tools/check_atlas_build.py atlas/dist
```

The atlas orders complete experiments by evidence tier and then explicit
decision value. Active and blocked studies remain visible but unranked.
One-attempt grids show raw numerators and denominators without confidence
claims; only replicated holdout evidence receives the deterministic paired
bootstrap used by Fugue's selection policy. Quality, cost, latency, retrieval,
errors, and observability remain separate—there is no composite score.

The no-context frontier campaign uses one fixed model per immutable run:

```bash
fugue run swe-frontier-harness --preset canary \
  --model wandb/zai-org/GLM-5.2 --preview
fugue run swe-frontier-harness --preset discovery \
  --model wandb/zai-org/GLM-5.2 --preview
fugue run swe-frontier-harness --preset frontier-ceiling \
  --model anthropic/claude-fable-5 --preview
```

After each canary export, admit the full cohort through the ignored campaign
ledger. The ledger prices unmeasured cells at the canary's maximum measured
cell cost, applies a 1.5× margin to every planned cell, and refuses a cohort
that would cross the cumulative cap:

```bash
uv run python tools/frontier_campaign.py admit \
  --ledger reports/frontier-budget.json \
  --canary-rows reports/GLM_CANARY.jsonl \
  --cohort-id glm-5.2 --model wandb/zai-org/GLM-5.2 \
  --canary-predictions 4 --cohort-predictions 32 \
  --cap-usd 2000 --safety-margin 1.5
```

Its locked hard discovery cases require production and test changes, two to
eight touched files, repository diversity, and either issue-path lexical
mismatch or cross-directory production work. Model cohorts share comparison
example identities while model and harness changes remain distinct behavioral
candidates.

The hard-memory study is encoded as separate immutable cohorts. Discovery uses
a fixed Latin-square harness assignment; holdout and control treatments must be
selected from discovery results instead of being silently baked into the
experiment:

```bash
fugue run repo-memory-impact --preset context-contract --preview
fugue run repo-memory-impact --preset hard-calibration --preview
fugue run repo-memory-impact --preset hard-discovery --preview
fugue analyze --saved repo-memory-discovery-selection --yes
fugue run repo-memory-impact --preset hard-holdout \
  --selection-lock REPORT_DIR/treatment-selection-lock.json --preview
fugue run repo-memory-impact --preset gitnexus-ablation --preview
fugue run repo-memory-impact --preset gitnexus-swe-contract --preview
fugue run repo-memory-impact --preset retrieval-study --preview
fugue run repo-memory-impact --preset gitnexus-retrieval-study --preview
fugue run repo-memory-impact --preset continuity-study --preview
```

The planned prediction counts are 48 context-contract, 32 calibration, 80
discovery, 192 holdout, 96 easy-control, 128 repository-QA, 96 independent
GitNexus ablation, and 72 PDF-skill Agent predictions: 744 primary cells in
total. Direct cohorts contain 900 general
retrieval measurements, 450 GitNexus BM25/vector measurements, and 108
continuity sequence attempts. The 225-probe retrieval source is materialized
and locked by `setup --prepare`; a trial refuses to download it.

Discovery ranks variants against the same task, harness, and trial baseline.
Official SWE resolution is primary, followed by localization recall@10, MRR,
recoverable-error rate, measured cost, and stable variant ID. The versioned
`TreatmentSelectionLockV1` captures the complete ranking and chosen three;
holdout, controls, and repository-QA reject manual treatments that disagree
with it. A separate 16-cell uptake diagnostic uses qualification-only tasks and
never contributes to efficacy denominators.

`gitnexus-swe-contract` is a one-cell Codex qualification canary. It explicitly
requires a semantic lookup before editing so it can prove native MCP, vector
telemetry, the official verifier, and the Agent deep link together. It is not
part of the unbiased GitNexus ablation and must not enter an efficacy estimate.

SWE-bench verifier dependencies and grading metadata are prepared into the
locked task image. Qualification runs the official verifier offline twice: the
base checkout must fail and the pinned gold patch must pass. A verifier that
tries to resolve packages or metadata during a trial invalidates the task
image instead of producing a benchmark result.

Analysis first resolves and displays an immutable local scope. `--yes` is the
explicit boundary for model interpretation and report writing:

```mermaid
flowchart LR
    Q["Comparative question"] --> SPEC["AnalysisSpec"]
    SPEC --> LOCAL["Local catalog filter"]
    LOCAL --> SNAP["Immutable row snapshot"]
    SNAP --> PREVIEW["Scope and deterministic aggregates"]
    PREVIEW --> CONFIRM{"Confirm interpretation?"}
    CONFIRM -->|No| STOP["No remote query or report write"]
    CONFIRM -->|Yes| WEAVE["Narrow Weave enrichment"]
    WEAVE --> MODEL["Evidence-bound interpretation"]
    MODEL --> REPORT["Report, scope, and evidence"]
```

```bash
fugue analyze \
  "Which context improved coding outcomes without excessive latency?" \
  --filter experiment_id=repo-memory-impact

fugue analyze --saved fugue-maintainer-selection --yes
```

### Weave Agent hierarchy

Harness identities are stable across experiments: `hermes-agent`, `openclaw`,
`claude-code`, and `codex`. Each Harbor Agent trial is one conversation with
one root turn. The harness integration owns model and tool spans; Fugue adds
flat correlation attributes and verifies the observed root before linking it
to an evaluation prediction.

```mermaid
flowchart TD
    AGENT["Stable harness Agent"] --> CONV["One trial conversation"]
    CONV --> TURN["One invoke_agent root turn"]
    TURN --> CHAT["Nested chat spans"]
    CHAT --> TOOL["Nested execute_tool spans"]
    TURN --> ATTR["run, candidate, example, trial, evaluation call"]
    ATTR --> LINK["Verified evaluation genai_span_ref"]
    DIRECT["Provider diagnostic"] --> OPS["Ordinary Weave operation calls"]
    OPS --> NOLINK["Agent linking not_applicable"]
```

## Advanced: generated evaluations

Generation is a separate explicit action; ordinary preview cannot read MCP,
call a model, write assets, or merge hidden files from disk. A generation
request names its exact suite, workload, size, and typed sources:

```yaml
judge_model: openai/gpt-5-mini
evaluation_generation:
  suite_id: repository-search-v1
  workload_id: capabilities
  size: 8
  sources:
    - {kind: seed, text: "Evaluate reviewed repository search behavior."}
    - {kind: file, path: README.md}

workloads:
  - id: capabilities
    runner: harbor
    scorers:
      - {type: builtin, id: harbor-outcome}
      - type: rubric
        path: configs/fugue/evaluations/repository-search-v1/rubric.yaml
```

Rubric scorers require an explicit judge model. Generation must produce the
configured case count and required strata. Existing suites require an explicit
regenerate/overwrite confirmation, and workload collisions fail instead of
silently renaming or modifying another workload.

## Advanced: candidate packaging and serving

Packaging requires a clean tracked Fugue source checkout and a clean production
workspace. It copies only an explicit Git-tracked runtime allowlist and rejects
submodules, escaping symlinks, credential-bearing remotes, secrets, dirty
source, integrations, and input-lock drift.

```bash
fugue runs RUN_ID package CANDIDATE_PREFIX \
  --workspace /path/to/clean/production-checkout \
  --image example/fugue-service:candidate \
  --yes
```

All applicable cells for the candidate must be terminal, and at least one
deterministic benchmark outcome must pass. Terminal unscored cells may provide
coverage but never satisfy that pass requirement. Benchmark or execution
failures require both `--allow-failed` and confirmation. Judge and rubric
results never replace the deterministic outcome. Another candidate may fail
the overall run without blocking a complete candidate.

Serving is an optional Python 3.13 feature, outside the operator path:

```bash
uv sync --extra dev --extra serve --python 3.13
python -m fugue.serve
```

The stateless text-only gateway implements the documented Open Responses,
Chat Completions, and AG-UI subset. Defaults are one executing request and an
eight-request queue. Configure `FUGUE_SERVE_MAX_CONCURRENCY` and
`FUGUE_SERVE_QUEUE_DEPTH` to change them. Admission overflow returns 429 with
`Retry-After`. Requests are limited to 1 MiB, 128 messages, and 256 KiB of text.
Cancellation terminates the isolated Harbor process group and removes request
state. `/readyz` performs a bounded Docker daemon or configured remote probe.

Stored conversations, server-side tools, media, compaction, WebSockets,
registry push, deployment management, and autoscaling are not part of 0.1.2.

## Experimental: self-evaluation

`fugue-maintainer-v1` and `fugue-operator-v1` are separate suites pinned to
Fugue commit `96512017842d68add2546a057f0601de3eaf610e`. Their tasks, mutations,
fixtures, and verifiers remain unchanged. Fugue 0.1.2 ships no promoted agent
preset and makes no efficacy claim; a release-current benchmark requires a
separately reviewed v2.

## Experimental: curator

The curator has two wrappers over one shared policy:

- `fugue-curator-dry-run.md` is manual and read-only. It cannot edit or create
  a pull request, and no-op reports do not create issues.
- `fugue-curator.md` is manual or scheduled only when the repository variable
  `FUGUE_CURATOR_ENABLED` is `true`. It may create at most one draft PR.

Curator output is restricted to skill-source declarations, context-system
definitions, and controlled experiments. It cannot change code, tests,
workflows, dependencies, datasets, presets, README, or vendored skill content.
Skill proposals still require human review through `fugue setup --skills`.

## Development

```bash
uv lock --check
uv run ruff check .
PYTHONWARNINGS=error uv run python -m compileall -q fugue tests
PYTHONWARNINGS=error uv run pytest
uv build
```

Core and context suites support Python 3.12 and 3.13. Serving and protocol
compatibility run on Python 3.13. See `docs/campaigns.md` for the governed
campaign contract, `docs/task-authoring.md` for flexible tasks, scenarios,
judges, and meta-analysis, `docs/research.md` for the packaged outer-loop SDK,
and `docs/extension-guide.md` for context and
integration definitions, `docs/releases/0.1.md` for the base release, and
`docs/releases/0.1.1.md` for the original evidence hardening, and
`docs/releases/0.1.2.md` for governed Studies, trace-grounded task authoring,
external-Agent packaging, and research-record projection.

Fugue is licensed under the Apache License 2.0. See `LICENSE`.
