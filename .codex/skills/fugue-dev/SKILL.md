---
name: fugue-dev
description: Use when modifying the Fugue repository, especially provider routing, Harbor JobConfig rendering, saved experiments/prompts/skills, feature variants, the FastAPI static operator UI, Weave metadata/export, or tests. Helps keep Fugue focused as a simple experiment runner and avoid reintroducing legacy profile/instruction/condition compatibility surfaces.
---

# Fugue Development

## Core Model

Use the product vocabulary consistently:

- `Experiment`: a saved reusable run definition in `configs/fugue/experiments/*.yaml`.
- `Prompt`: a portable prompt file in `configs/fugue/prompts/*.md`.
- `Skill`: a Harbor skill in `configs/fugue/skills/<id>/SKILL.md`.
- `FeatureVariant`: a named comparison bundle of prompt, skills, optional memory, and advanced Harbor config.
- `Run name`: the W&B/Weave grouping name for one execution.

Do not add user-facing `profile`, `instruction`, `condition`, `variant_key`, or `memory_variant` compatibility paths. This is a net-new repo; prefer one clean contract.

## Architecture Rules

- W&B/Weave is always the trace plane. Model calls route through `wandb/...`, `openai/...`, or `anthropic/...`.
- Keep provider selection in `fugue/model_plane.py`; keep experiment library logic in `fugue/bench/library.py`; keep Harbor JobConfig rendering in `fugue/bench/job_config.py`.
- Saved authoring state belongs under `configs/fugue/`. Generated runtime state belongs under ignored `.fugue/runtime/` or job/report/artifact directories.
- Preview APIs must be side-effect free: no config writes, no saved experiment writes, no generated memory instruction writes.
- Render/run paths may write generated JobConfigs and runtime files.
- Never serialize raw API keys. Use env var names and presence booleans only.
- Operator UI stays no-build: FastAPI plus static `index.html`, `app.css`, and `app.js`. Do not add React, Vite, Tailwind, or a frontend package manager.

## Metadata And Tags

Trial metadata and exported rows should make comparison easy:

- Include `experiment_id`, `run_name`, `variant_id`, `prompt_id`, `skill_ids`, `feature_memory`, `agent_config_hash`, `harness`, `model_provider`, `model`, `trace_project`, and local trial paths.
- Tags should include `fugue`, `experiment-id:<id>`, `variant:<id>`, `memory:<id>`, `prompt:<id>`, `skill:<id>`, `harness:<id>`, `provider:<id>`, `model:<id>`, and `run:<name>` where applicable.
- Results should group by experiment, variant, prompt, skill, harness, provider, and feature memory.

## Change Workflow

1. Read the relevant tests and local module before editing. Use `rg` first.
2. If changing schema, update `library.py`, `job_config.py`, CLI/web callers, metadata/export, and tests together.
3. If changing UI behavior, update API shape and static JS in the same pass; keep Run, Compare, and Setup as the only primary tabs.
4. If changing provider behavior, validate routing, required env vars, generated bridge config, and adapter expectations.
5. Keep edits scoped. Avoid broad refactors unless they reduce real duplication or remove a stale abstraction.

## Validation

Run the light checks before handing off:

```bash
python -m compileall fugue
python -m ruff check .
python -m pytest
```

For UI changes, also start the local app and smoke the current browser path:

```bash
fugue web --host 127.0.0.1 --port 8765
```

Check that Run can preview without writing `.fugue/runtime`, Render writes configs, and Compare still shows Weave links without leaking secrets.
