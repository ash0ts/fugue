---
name: fugue-maintainer
description: Repair and extend Fugue while preserving its experiment, Harbor, and Weave contracts.
---

# Fugue Maintainer

## Workflow

1. Read the failing test, implementation, and adjacent contract before editing.
2. Reproduce the narrow failure and identify which layer owns it.
3. Keep experiment parsing in `library.py`, planning in `operator.py`, Harbor
   rendering in `job_config.py`, provider routing in `model_plane.py`, and
   deterministic analysis in `ai.py` and `scoring.py`.
4. Make the smallest coherent repair and add a regression test at the owning layer.
5. Run focused tests, Ruff, compileall, and the relevant broader suite.

## Invariants

- Preview never writes runtime files, downloads data, or prepares context.
- Unsupported cells are `not_applicable`, not failures or substitutes.
- Model calls are provider-neutral while W&B Weave remains the trace plane.
- Raw credentials never enter configs, logs, telemetry, or normalized rows.
- Native harness integrations own model and tool spans.
- Missing usage, cost, or retrieval measurements remain `N/A`.
- Independent Harbor cells continue after a sibling failure.
- Public behavior flows through `OperatorService`; presentation layers do not
  invoke each other.

Do not solve a local failure by adding a compatibility branch or weakening a
deterministic verifier.
