---
name: fugue-dev
description: Use when modifying the Fugue repository, especially provider routing, context-system plugins and caching, Harbor JobConfig rendering, experiments/prompts/skills, workload runners, the Textual terminal operator, Weave agent observability/export, or tests. Helps preserve Fugue's provider-neutral experiment contracts and avoid memory-specific or legacy compatibility surfaces.
---

# Fugue Development

## Core Model

Use the product vocabulary consistently:

- `Experiment`: a saved reusable run definition in `configs/fugue/experiments/*.yaml`.
- `Prompt`: a portable prompt file in `configs/fugue/prompts/*.md`.
- `Skill`: a Harbor skill in `configs/fugue/skills/<id>/SKILL.md`.
- `FeatureVariant`: a named comparison bundle of prompt, skills, one `ContextSelection`, and advanced Harbor config.
- `ContextSystem`: a repo-backed plugin definition with explicit prepare, bind, retrieve, ingest, and sequence capabilities.
- `Workload`: a `harbor`, `retrieval`, or `sequence` evaluation lane selected through an experiment preset.
- `Run ID`: an immutable generated execution identity with durable state under `.fugue/runtime/<run-id>/`.
- `Run name`: the W&B/Weave grouping name for one execution.

Do not add user-facing `profile`, `instruction`, `condition`, `variant_key`, or `memory_variant` compatibility paths. This is a net-new repo; prefer one clean contract.

## Architecture Rules

- W&B/Weave is always the trace plane. Model calls route through `wandb/...`, `openai/...`, or `anthropic/...`.
- W&B Inference calls must carry `OpenAI-Project` for the resolved trace project. Use `provider_request_headers` for direct HTTP, `provider_client_env` for OpenAI-compatible SDKs (`OPENAI_PROJECT` and `OPENAI_PROJECT_ID` are both required across versions), and the same header in generated LiteLLM config.
- Weave agent spans must use the Agents OTLP endpoint with `project_id` and Basic `api:<WANDB_API_KEY>` headers. Build those headers only in the trial process environment with `weave_agents_otel_headers`; never serialize them into Harbor configs, plugin YAML, logs, or metadata.
- Keep provider selection in `fugue/model_plane.py`; keep experiment library logic in `fugue/bench/library.py`; keep Harbor JobConfig rendering in `fugue/bench/job_config.py`.
- Saved authoring state belongs under `configs/fugue/`. Generated runtime state belongs under ignored `.fugue/runtime/` or job/report/artifact directories.
- Preview APIs must be side-effect free: no config writes, downloads, index builds, saved experiment writes, or generated context instructions.
- Render/run paths may write generated JobConfigs and runtime files.
- Context preparation is content-addressed under `.fugue/cache/context`; include repository commit, provider/version/config, builder model, and embedding model in cache keys. Publish atomically under a lock.
- Only advertise a capability a provider implements. Unsupported cells are `not_applicable`/`N/A`, not failures or zero scores.
- Preserve each native context interface. Wrap stdio MCP with `fugue.mcp_proxy` for bounded, redacted telemetry instead of replacing upstream tools.
- Codex MCP tools use Responses namespaces. Until a bridge advertises and passes namespace-tool compatibility, render bridged Codex MCP cells as `not_applicable`; never substitute a static mount while retaining an MCP-backed treatment label.
- Never serialize raw API keys. Use env var names and presence booleans only.
- Keep operator behavior in `fugue.bench.operator`; both Rich commands and Textual consume those presentation-neutral services.
- Keep experiment selection, overrides, and job planning in `OperatorService`; the CLI only translates arguments and executes the returned plan.
- Keep the public operator surface to bare `fugue` plus `plan`, `run`, `runs`, `analyze`, `setup`, and `tui`. Do not expose catalog, asset CRUD, bridge, export, preflight, or context internals as separate top-level commands.
- Rich is the lightweight command center and Textual is the full-screen workspace. Share typed requests and results, never Rich renderables or CLI subprocess calls.
- Keep AI transport normalization in `fugue.assistant`, experiment/result indexing in `fugue.bench.catalog`, and grounded composer/analyst behavior in `fugue.bench.ai`. CLI and Textual must consume these services instead of embedding prompts or model calls.
- Composer output is an untrusted draft. Parse it through `ExperimentSpec`, validate every prompt/skill/context/manifest reference, and run side-effect-free preview before exposing Apply, Save, or Run.
- Analyst arithmetic is deterministic Python over an immutable catalog snapshot. The model may interpret aggregates and inspect bounded evidence, but it must not calculate official metrics or cite evidence ids that do not exist.
- Pair variants through the deterministic `comparison_example_id`, which identifies the shared task/query/episode and excludes `trial_index`. Trial ordinal is a separate prediction coordinate. Never pair by Harbor directory names or list position.
- `.fugue/cache/catalog/v2` is rebuildable local generated state. Hybrid analysis starts from a local cohort and queries Weave only for its run keys and conversation IDs. Saved analysis definitions belong in `configs/fugue/analyses`; generated reports and evidence belong in `reports/analyses`.
- Assistant tools may read only bounded, redacted Fugue metadata and result artifacts. Never expose arbitrary filesystem, shell, environment, or Python execution to an assistant model.
- Keep `fugue.tui` presentation-only. Long operations run in workers or detached process groups and communicate through durable state/events; never block Textual's event loop.
- Keep TUI planning intent-first: Define chooses natural language or a saved experiment, Compare edits variants/harnesses/coverage/size, and Review owns the exact matrix and launch authority. Low-frequency model roles, concurrency, tags, and tracing stay under Advanced.
- Use one in-memory plan state for the working `ExperimentSpec`, `ExperimentRequest`, draft assets, dirty state, and latest `PreviewSummary`. Widgets render that state; do not add parallel applied-draft or form-reconstruction paths.
- Automatic Plan previews must remain debounced, side-effect free, and stale-result safe. `PreviewSummary.matrix_cells` is the shared Rich/Textual source for trial counts, applicability, and cache readiness.
- Variant edits remain in memory until explicit save or run. An AI proposal is untrusted until accepted, and generated prompt or skill assets must be saved before execution.
- The `r` shortcut navigates to Review before launch. Full-content tracing requires confirmation at launch, not a persistent warning competing with primary configuration.
- Textual has one durable launch mode. Do not add cosmetic attached/detached controls; use `fugue run --detach` for the explicit headless option.
- Run state is append-friendly and recoverable: atomic `run.json`, `events.jsonl`, `cells.jsonl`, combined logs, and per-cell logs. A cell failure must not stop independent cells.
- The browser frontend has been removed. Do not add FastAPI, static web assets, or HTTP job abstractions back into the operator path.
- Preflight is observational. Starting the bridge and preparing context are explicit `fugue setup` actions.
- Experiment YAML is strict. Do not add compatibility aliases for removed fields.
- JSONL is the only normalized local export format.
- CodeGraph, GitNexus, Project-RAG, Semble, and lat.md are opt-in research adapters until their pinned Harbor MCP runtimes pass integration tests. Preserve their definitions without placing them in default presets.

## Weave Agent Contract

- Use stable harness agent names: `hermes-agent`, `openclaw`, `claude-code`, and `codex`. Experiments, variants, and trials are attributes, never agent identities.
- Set `gen_ai.agent.name`, deterministic `gen_ai.conversation.id`, and flat `fugue.*` attributes for run, experiment, workload, harness, variant, context, task, trial, model, prompt, skills, tags, and run key.
- Native harness integrations own `invoke_agent`, `chat`, and `execute_tool` spans. Never add wrapper spans that duplicate native turns, model calls, or tools.
- Store deterministic Fugue conversation identity and native session IDs in trial metadata. Export joins Agents spans by conversation ID and `fugue.run_key`, not by per-trial agent names.
- `trace_content` defaults to `full`. Metadata mode must fail preflight or render `not_applicable` when an integration cannot guarantee content suppression.
- Trace the operator agents with stable names `fugue-experiment-composer` and `fugue-analysis-agent`. Validation repairs retain one session identity. Do not persist a second write-only assistant session store.
- Live traces and evaluation predictions are emitted during execution. Export owns normalized JSONL plus idempotent verification/backfill; it must skip rows already finalized in publication ledger v3.
- Render one Harbor JobConfig per `harness x variant x task x trial`, with Harbor `n_attempts=1`. Keep `trial_index`, `comparison_example_id`, `candidate_id`, and run keys deterministic and present in config, trial metadata, cells, exports, and trace attributes.
- Partition publication by exact shared example and scorer scope, then candidate. Dataset inputs contain only invariant benchmark/example identity. Candidate fields belong in a uniquely named model object; Evaluation attributes remain scope-only so candidates reuse one Evaluation definition.
- Open `EvaluationLogger.log_prediction()` before each Harbor cell and finish it after resolving the authoritative native root. Inject the exact `weave.eval.*` link attributes, attach `genai_span_ref`, call `log_summary()` without a custom nested summary, and write a v3 marker only after successful finalization. Do not publish administrative cell or preparation rows.
- Keep `planned_conversation_id` and `observed_conversation_id` separate. Evaluation output and deep links use only an observed native root that matches run key, task, stable agent name, and prediction call id.
- Report context assignment, availability, invocation, query/result counts, latency, and errors separately. Never infer context use from prompt text or static mount names.
- Prefer measured child `chat` usage. Use root aggregate usage only when child usage is absent; never add both. Preserve missing usage as `None` with an unavailable status rather than zero.
- Query Calls through `WF_TRACE_SERVER_URL` or `https://trace.wandb.ai`, use the current Calls filter schema and NDJSON response shape, and surface transport errors. Raw resource attributes are diagnostic fallback only; normalized Agents rows should expose flat `fugue.*` span attributes.
- Analysis must stop after `AnalysisPreview` until the user confirms report generation. Local scope resolution cannot query Weave, call the report model, or write a report.

## Metadata And Tags

Trial metadata and exported rows should make comparison easy:

- Include `experiment_id`, `preset_id`, `workload_id`, `run_name`, `variant_id`, `context_system_id`, `context_version`, `context_config_hash`, cache keys, prompt/skill ids and hashes, agent config hash, harness, model role/provider/model, trace project, and local artifact paths.
- Tags should include `fugue`, experiment, preset, workload, variant, context system, prompt, skill, harness, provider, model, and run name where applicable.
- Results should group by experiment, workload, context system, variant, prompt, skill, harness, and provider. Keep outcome, retrieval, evidence, efficiency, and utilization metrics separate; do not invent a composite score.

## Change Workflow

1. Read the relevant tests and local module before editing. Use `rg` first.
2. If changing schema, update `library.py`, `job_config.py`, operator/CLI callers, metadata/export, and tests together.
3. If changing terminal behavior, keep Plan, Runs, Results, and Setup coherent; add a Textual Pilot test for the workflow.
4. If changing provider behavior, validate routing, required env vars, generated bridge config, and adapter expectations.
5. Keep edits scoped. Avoid broad refactors unless they reduce real duplication or remove a stale abstraction.

## Validation

Run the light checks before handing off:

```bash
python -m compileall fugue
python -m ruff check .
python -m pytest
```

For terminal changes, run Textual headlessly and smoke Rich output:

```bash
FUGUE_NO_ANIMATION=1 python -m pytest tests/test_tui.py
fugue setup
fugue runs
```

Check that Plan preview does not write `.fugue/runtime`, live runs persist durable state, Results shows local summaries, and Weave actions never expose credentials or invent unverified trace URLs.
