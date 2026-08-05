# W&B MCP 0.4 maintainer qualification

This directory contains Fugue's source-isolated comparison of the exact W&B
MCP `main` baseline with the exact current 0.4 staging candidate.

The question is deliberately behavioral:

> Does the 0.4 candidate help a Claude Code maintainer answer factual, bounded
> W&B/Weave questions more reliably than `main`?

Package release, managed-service, Serverless, and Helm qualification are
separate decisions.

## Current result

The canonical completed Study is
`mcp-main-vs-0-4-tool-surface-confirmation-v10`.

| Field | Locked value |
|---|---|
| Baseline | `wandb-mcp-main` at `53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0` |
| Candidate | `wandb-mcp-0-4-current` at `5c6cc1c9a1079296daf6613ea6d12daebdd8bcba` |
| Source evidence | `wandb/fugue-mcp-release-source-v2` |
| Result evidence | `wandb/fugue-mcp-release-qualification-v1` |
| Runtime | Local Docker through Harbor |
| Matrix | 4 tasks × 2 revisions × 2 attempts = 16 cells |
| Model and harness | `anthropic/claude-sonnet-5` with Claude Code |
| Observed cost | `$7.331252` |
| Evidence | 16/16 attempts reconciled; grade A link/privacy integrity |
| Behavioral verdict | **REGRESSED** |
| Release decision | **HOLD** |

The result is useful rather than uniformly positive:

- Evaluation direct-child reconciliation improved from `0/2` to `2/2`.
- Bounded inventory behavior improved from `0/2` to `2/2`.
- Exact history did not improve and one candidate attempt lost factual
  correctness.
- Failure triage remained unbounded in both revisions.
- The candidate passed every required deterministic dimension in only `1/8`
  attempts and retained seven critical failures.
- One aligned outcome pair regressed; seven were unchanged; none produced a
  promotable outcome improvement.

Evidence grade A means that attempt identities, Weave links, project routing,
privacy checks, and exported rows reconcile. It is not a task grade.

The blind maintainer judge is advisory and pending human calibration. Its
0–1 scores rate memo actionability, evidence-to-action traceability, a bounded
next step, and claim-uncertainty calibration. They are not correctness
probabilities or release confidence and cannot rescue a deterministic failure.

The result digest is
`e062f5b392a36d9ebd97adc3ab58b6e253cdd9dd943381342d51d76303bbcf38`.
The package remains on HOLD. Do not run a larger paid confirmation until the
history regression and named critical blockers are repaired under a new
candidate and Study identity.

## Why these tasks match 0.4

The task set is derived from the exact candidate's
[0.4 release notes](https://github.com/wandb/wandb-mcp-server/blob/5c6cc1c9a1079296daf6613ea6d12daebdd8bcba/docs/releases/v0.4.0.md).
It exercises the changes most likely to affect a normal maintainer:

| Task | Release behavior under test |
|---|---|
| `run-inventory-projection` | SDK-first structured queries, exact counts, projected fields, explicit coverage |
| `filtered-failure-triage` | selective filtering, cursor continuation, bounded collection reasoning |
| `evaluation-summary-accuracy` | direct `Evaluation.predict_and_score` reconciliation |
| `exact-history-target` | bounded history, custom axis, exact target verification, honest structured failures |

The Study separately records actual query scope, reported project identity,
tools used, release-mechanism use, latency, cost, structured errors, and source
coverage. Mechanism evidence does not count as task correctness.

The release-note ledger also classifies deployment, security, raw-GraphQL,
read-only, telemetry, admission, timeout, and wheel/CI behaviors as separate
infrastructure gates or explicit non-applicability. Those gates were not run
by this local behavioral Study and remain unavailable rather than being
silently treated as passes. Exact report-name/title behavior remains
unqualified.

## Canonical inputs

The V10 result binds these files:

- `tool-surface-confirmation-local-v10.yaml`
- `tool-surface-confirmation-tasks-v8.jsonl`
- `tool-surface-confirmation-private-v8.jsonl`
- `tool_surface_scorer_v7.py`
- `current-candidate.json`
- `release-notes.current.lock.json`
- `mcp.json`

Public task prompts and host-only expected facts are separate. The private
bundle must never enter Agent inputs, traces, published Study events, or result
excerpts.

V4 and V6 inputs remain only because they produced canonical, superseded audit
results. Aborted V5/V7/V8/V9 specs and the never-run V4 confirmation are not
part of the supported surface.

## Evidence topology

The source project contains the locked read-only task cohort:

- six W&B Runs with projected configuration, summary, history, and latency
  evidence;
- 24 native conversation roots and 48 tool children;
- a versioned eight-row Weave Dataset;
- two Evaluation objects with 16 prediction rows and two summary children.

Agent traces, Evaluations, predictions, decisions, and Study events are written
only to the result project. Source-lock checks run before, during, and after
execution. Any source drift, source-project write, cross-project query, missing
authoritative Call, leaked private fact, duplicate attempt, or run-scoped
Harbor orphan invalidates the result.

Every valid attempt must resolve five distinct authoritative Weave links:

1. Evaluation root
2. `Evaluation.predict_and_score`
3. prediction
4. native Agent root
5. Dataset

OTel trace/span IDs remain diagnostic metadata and are never substituted for
Weave Call IDs.

## Inspect the completed Study

With the local Research store available:

```bash
uv run fugue result \
  mcp-main-vs-0-4-tool-surface-confirmation-v10 \
  --json
```

In Study Console, open:

```text
http://127.0.0.1:18080/?research_id=fugue-mcp-release-qualification-v1&study_id=mcp-main-vs-0-4-tool-surface-confirmation-v10
```

Start with the behavioral verdict, then inspect the one regressed
`exact-history-target` pair. Open each baseline/candidate attempt, read its
deterministic dimensions, treat the judge as advisory memo review, and follow
the prediction-and-score link into the complete Weave evidence chain.

## Create a new qualification

Do not reuse the V10 identity or approval. After changing the candidate,
taskset, scorer, runtime, or policy, copy the canonical spec to a new Study ID
and run the governed path:

```bash
SPEC=examples/comparisons/wandb-mcp-maintenance/NEW-STUDY.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json

env -u OPENAI_API_KEY uv run fugue compare "$SPEC" \
  --prepare --env-file "$ENV_FILE" --json

env -u OPENAI_API_KEY uv run fugue compare "$SPEC" \
  --preview --env-file "$ENV_FILE" --json
```

Approve the exact preview digest with an explicit cell/cost cap, then execute
that approval with `--fetch-weave`. Never copy a prior approval across a
candidate, source, result-project, runtime, scorer, task, or policy change.

## Validation

The merge tree must pass:

```bash
uv run ruff check fugue tests
uv run python -W error -m compileall -q fugue
uv run vulture fugue vulture_whitelist.py --min-confidence 80
uv run vulture fugue vulture_whitelist.py --min-confidence 60
uv run deptry fugue
uv run pytest
uv build
```

CI also runs the repository's exact Python 3.12/3.13 core, context, dev,
research, serving, image, managed-adapter, and security partitions.
