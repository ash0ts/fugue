# Fugue

Fugue is a local-first operator for controlled agent experiments. It resolves
an experiment into comparable candidates, renders Harbor jobs, executes the
exact matrix, records native W&B Weave traces, and exports reproducible results.
Fugue 0.1 supports Hermes, OpenClaw, Claude Code, and Codex on Python 3.12+.

The core workflow is deliberately small:

1. Define or load an experiment.
2. Preview the exact candidate × task × trial matrix.
3. Prepare reviewed skills and declared context explicitly.
4. Run through the durable operator transaction.
5. Inspect candidates and export normalized JSONL.

Generated evaluations, self-evaluation, automated curation, and candidate
serving are advanced or experimental extensions. None runs implicitly.

```mermaid
flowchart LR
    USER["Saved experiment or planning intent"] --> RESOLVE["Resolve candidates once"]
    RESOLVE --> OP["OperatorService transaction"]
    OP --> MATRIX["Candidate x example x trial"]
    MATRIX --> HARBOR["Harbor agent cells"]
    MATRIX --> DIRECT["Provider diagnostics"]
    HARBOR --> LOCAL["Durable local state"]
    DIRECT --> LOCAL
    HARBOR --> WEAVE["Weave Agent conversations"]
    DIRECT --> OPS["Ordinary Weave operation calls"]
    LOCAL --> EXPORT["Normalized export and analysis"]
    WEAVE --> EXPORT
    OPS --> EXPORT
```

## Install

[`uv`](https://docs.astral.sh/uv/) is the recommended environment manager.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
cp .env.example .env
```

Configure W&B for tracing and the credentials required by the selected model
route:

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
fugue setup --experiment repo-memory-impact --prepare-context
```

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

All applicable cells for the candidate must be terminal and deterministically
scored, and at least one benchmark outcome must pass. Benchmark or execution
failures require both `--allow-failed` and confirmation; unscored cells block
packaging. Judge and rubric results never replace the deterministic outcome.
Another candidate may fail the overall run without blocking a complete
candidate.

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
registry push, deployment management, and autoscaling are not part of 0.1.

## Experimental: self-evaluation

`fugue-maintainer-v1` and `fugue-operator-v1` are separate suites pinned to
Fugue commit `96512017842d68add2546a057f0601de3eaf610e`. Their tasks, mutations,
fixtures, and verifiers remain unchanged. Fugue 0.1 ships no promoted agent
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
compatibility run on Python 3.13. See `docs/extension-guide.md` for context and
integration definitions, and `docs/releases/0.1.md` for release scope and
manual gates.

Fugue is licensed under the Apache License 2.0. See `LICENSE`.
