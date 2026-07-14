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
- Keep provider selection in `fugue/model_plane.py`; keep experiment library logic in `fugue/bench/library.py`; keep Harbor JobConfig rendering in `fugue/bench/job_config.py`.
- Saved authoring state belongs under `configs/fugue/`. Generated runtime state belongs under ignored `.fugue/runtime/` or job/report/artifact directories.
- Preview APIs must be side-effect free: no config writes, downloads, index builds, saved experiment writes, or generated context instructions.
- Render/run paths may write generated JobConfigs and runtime files.
- Context preparation is content-addressed under `.fugue/cache/context`; include repository commit, provider/version/config, builder model, and embedding model in cache keys. Publish atomically under a lock.
- Only advertise a capability a provider implements. Unsupported cells are `not_applicable`/`N/A`, not failures or zero scores.
- Preserve each native context interface. Wrap stdio MCP with `fugue.mcp_proxy` for bounded, redacted telemetry instead of replacing upstream tools.
- Never serialize raw API keys. Use env var names and presence booleans only.
- Keep operator behavior in `fugue.bench.operator`; both Rich commands and Textual consume those presentation-neutral services.
- Keep AI transport normalization in `fugue.assistant`, experiment/result indexing in `fugue.bench.catalog`, and grounded composer/analyst behavior in `fugue.bench.ai`. CLI and Textual must consume these services instead of embedding prompts or model calls.
- Composer output is an untrusted draft. Parse it through `ExperimentSpec`, validate every prompt/skill/context/manifest reference, and run side-effect-free preview before exposing Apply, Save, or Run.
- Analyst arithmetic is deterministic Python over an immutable catalog snapshot. The model may interpret aggregates and inspect bounded evidence, but it must not calculate official metrics or cite evidence ids that do not exist.
- `.fugue/cache/catalog` is rebuildable generated state. Saved analysis definitions belong in `configs/fugue/analyses`; generated reports and evidence belong in `reports/analyses`.
- Assistant tools may read only bounded, redacted Fugue metadata and result artifacts. Never expose arbitrary filesystem, shell, environment, or Python execution to an assistant model.
- Keep `fugue.tui` presentation-only. Long operations run in workers or detached process groups and communicate through durable state/events; never block Textual's event loop.
- Run state is append-friendly and recoverable: atomic `run.json`, `events.jsonl`, `cells.jsonl`, combined logs, and per-cell logs. A cell failure must not stop independent cells.
- The browser frontend has been removed. Do not add FastAPI, static web assets, or HTTP job abstractions back into the operator path.

## Weave Agent Contract

- Use stable harness agent names: `hermes-agent`, `openclaw`, `claude-code`, and `codex`. Experiments, variants, and trials are attributes, never agent identities.
- Set `gen_ai.agent.name`, deterministic `gen_ai.conversation.id`, and flat `fugue.*` attributes for run, experiment, workload, harness, variant, context, task, trial, model, prompt, skills, tags, and run key.
- Native harness integrations own `invoke_agent`, `chat`, and `execute_tool` spans. Never add wrapper spans that duplicate native turns, model calls, or tools.
- Store deterministic Fugue conversation identity and native session IDs in trial metadata. Export joins Agents spans by conversation ID and `fugue.run_key`, not by per-trial agent names.
- `trace_content` defaults to `full`. Metadata mode must fail preflight or render `not_applicable` when an integration cannot guarantee content suppression.
- Trace the operator agents with stable names `fugue-experiment-composer` and `fugue-analysis-agent`. Each natural-language request is one conversation turn with nested assistant model and typed-tool spans.
- Live traces are emitted during execution. `fugue export` remains the only normalized evaluation publisher and must honor the publication ledger.

## Metadata And Tags

Trial metadata and exported rows should make comparison easy:

- Include `experiment_id`, `preset_id`, `workload_id`, `run_name`, `variant_id`, `context_system_id`, `context_version`, `context_config_hash`, cache keys, prompt/skill ids and hashes, agent config hash, harness, model role/provider/model, trace project, and local artifact paths.
- Tags should include `fugue`, experiment, preset, workload, variant, context system, prompt, skill, harness, provider, model, and run name where applicable.
- Results should group by experiment, workload, context system, variant, prompt, skill, harness, and provider. Keep outcome, retrieval, evidence, efficiency, and utilization metrics separate; do not invent a composite score.

## Change Workflow

1. Read the relevant tests and local module before editing. Use `rg` first.
2. If changing schema, update `library.py`, `job_config.py`, operator/CLI callers, metadata/export, and tests together.
3. If changing terminal behavior, keep Compose, Runs, Results, and Setup coherent; add a Textual Pilot test for the workflow.
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
fugue status
fugue runs list
```

Check that Compose preview does not write `.fugue/runtime`, live runs persist durable state, Results shows local summaries, and Weave actions never expose credentials or invent unverified trace URLs.
