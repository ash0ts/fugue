---
name: fugue-operator
description: Plan, run, inspect, export, and analyze Fugue experiments safely through the public operator interface.
---

# Fugue Operator

## Public Workflow

1. Use `fugue setup --experiment ID` to inspect readiness and `--check` for live,
   observational validation.
2. Use `fugue run ID --preview --json` to verify the exact matrix without writes.
3. Launch only after confirming model, variants, harnesses, task coverage, trials,
   and trace policy.
4. Use `fugue runs`, `fugue runs RUN_ID`, and cell logs to diagnose durable work.
5. Export a managed run with `fugue runs RUN_ID --export --out PATH`.
6. Resolve an analysis scope before allowing Weave enrichment or report generation.

## Interpretation

- Compare only rows sharing benchmark, workload, task coverage, and model.
- Treat `not_applicable` as an eligibility result, not a failed trial.
- Keep pass rate, reward, latency, cost, tokens, failures, and context utilization
  separate. Do not invent a composite score.
- Preserve missing measurements as unavailable.
- Use W&B Weave for trace details and local JSONL for normalized outcomes.

Never place API keys in a command, generated experiment, or result artifact.
