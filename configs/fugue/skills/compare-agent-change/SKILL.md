---
name: compare-agent-change
description: Turn a concrete Agent failure or proposed system change into a governed Fugue baseline-versus-candidate comparison. Use when an Agent needs to author public tasks and private labels, import a normal MCP server or Agent Skill, validate evaluation readiness, preview aligned attempts and cost, request external approval, inspect results, or propose a bounded next test.
---

# Compare an Agent change

Treat Fugue as the experiment compiler and laboratory. Keep hypothesis
generation and implementation outside Fugue.

## Workflow

1. State one observed failure with immutable evidence references. Separate the
   observation from possible explanations.
2. Define the taskset. Put Agent-visible inputs in public JSONL and expected
   values in a separate private JSONL file. Include discovery and holdout
   partitions deliberately. Use `fugue taskset schema` for the strict
   contracts or `fugue taskset import-weave` for one immutable public Weave
   Dataset revision.
3. Define the baseline and candidate with exactly one declared behavioral
   change when possible. Import normal components before referencing them:

   ```bash
   uv run fugue mcp import --config mcp.json --server SERVER --as ID
   uv run fugue mcp inspect ID
   uv run fugue mcp lock ID --acknowledge-package-code

   uv run fugue skills import PATH_OR_PINNED_GIT_SOURCE
   uv run fugue skills inspect ID
   uv run fugue skills lock ID
   ```

4. Define a deterministic evaluator first. A custom scorer must implement
   `score(task, output, evidence)` and declare the exact dimensions it returns;
   Fugue runs it in the pinned scorer sandbox. Add a blind judge only for
   qualities exact checks cannot measure, and only with reviewed calibration.
5. Run `uv run fugue check COMPARISON.yaml`. Stop on `blocked` or
   `no_comparison_justified`. Resolve `needs_review` before release-quality
   evidence.
6. Run `uv run fugue compare COMPARISON.yaml --preview`. Explain the exact
   task, candidate, harness, attempt, cost, and evidence bounds.
7. Ask an operator to approve that exact preview digest. Never approve spend,
   loosen evidence requirements, or substitute a new preview.
8. Start only the accepted preview. Never retry a failed attempt silently or
   create a follow-up automatically.
9. Read `result.json`, aligned traces, and evaluations. Keep these separate:
   deterministic task outcome, blind-judge dimensions, mechanism evidence,
   infrastructure health, and evidence completeness.
10. Record observation, interpretation, limitation, and next test separately.
    Do not turn a null result into equivalence or a small cohort into a
    universal Agent, harness, MCP, model, or Skill ranking.

## Readiness rules

- Require non-empty tasks, at least one attempt row, stable identities, and an
  exact candidate diff.
- Require intended base failures and known-good passes before spend.
- Keep private values out of prompts, trial files, traces, snapshots, and
  public bundles.
- Treat assignment, registration, invocation, evidence opening, evidence use,
  and task pass as different states.
- Treat unavailable required scoring or missing trace reconciliation as an
  incomplete experiment, not an Agent failure.
- Prefer fixing the scorer, collecting harder failures, or shrinking the
  question when Fugue says the proposed comparison is not ready.
