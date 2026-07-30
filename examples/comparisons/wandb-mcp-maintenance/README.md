# W&B MCP 0.4 maintainer qualification

This directory contains the canonical, source-isolated Fugue qualification for
the W&B MCP Python package:

> Does the exact 0.4 staging revision improve a maintainer's ability to
> reconcile W&B/Weave evidence and assess project health compared with the
> exact `main` revision?

The current release decision is **HOLD** until the natural-maintainer studies,
package gates, and human sign-off all pass. A local behavioral finding does not
certify W&B Serverless, managed-service, or Helm readiness.

## Locked design

| Role | Locked value |
|---|---|
| Source evidence | `wandb/fugue-mcp-release-source-v1` |
| Result evidence | `wandb/fugue-mcp-release-qualification-v1` |
| Baseline | `wandb-mcp-main` at `53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0` |
| Provisional candidate | [`wandb-mcp-0-4-staging` at PR #126 head `3dd4447ef0054d4707aafc515e3f2ddfb11b17bd`](https://github.com/wandb/wandb-mcp-server/pull/126) |
| Execution | local Docker through Harbor |
| Canary | 2 tasks × 2 arms × 2 attempts = 8 cells, maximum $10 |
| Confirmation | 4 tasks × 2 arms × 2 attempts = 16 cells, maximum $20 |

The checked-in candidate is the provisional reviewed fix-PR head, not yet the
final `staging/0.4.0` head. It is suitable for offline contract preparation
only. After PR #126 merges, lock the resulting staging head and regenerate
every candidate, runtime, scorer, preview, and approval lock before running a
behavioral study.

The source project contains immutable task evidence only. Agent traces,
Evaluations, result rows, and release decisions go to the result project.
Publication to the source project, source drift, or a query outside the locked
source scope fails qualification.

## Prepare and freeze source evidence

Install the trusted preparation environment:

```bash
uv sync --python 3.13 --frozen --extra dev --extra research-worker
```

Seed or validate the dedicated source project idempotently. V3 preparation has
no single-project writer: source and result must remain the exact distinct
projects below. The output path and its adjacent
`evidence.lock.json.progress.json` recovery receipt are operator-local and must
not be committed:

```bash
uv run python \
  examples/comparisons/wandb-mcp-maintenance/prepare_hosted_project.py \
  --source-project wandb/fugue-mcp-release-source-v1 \
  --result-project wandb/fugue-mcp-release-qualification-v1 \
  --env-file /Users/ashah/Documents/common_tools/.env \
  --output .fugue/qualification/mcp-release-source-v1/evidence.lock.json
```

The command never serializes credentials or invokes a model provider. Before
any hosted write it takes two read-only inventories and rejects a changing
snapshot, extra seeded Runs or Calls, duplicates, or content drift. It verifies
the six Runs using exhaustive history, summary, and the downloaded exact
artifact payload; it verifies the Dataset rows, 24 conversation roots and 48
tool children, two immutable Evaluation objects, and each Evaluation's eight
prediction children plus one summary child.

The local progress receipt is written before each hosted mutation. On retry,
Fugue first inventories the remote source. A visible in-flight object is
reconciled and reused; an unresolved write outcome stops preparation instead
of retrying and risking duplicate evidence. Deleting only the final evidence
lock therefore reconstructs it without creating hosted objects. Deleting or
editing both recovery artifacts is not a supported recovery procedure.

W&B Runs and Weave Calls are mutable remote records rather than immutable
object versions, so every lock reuse re-reads and hashes their selected
terminal content. Dataset and Evaluation refs are exact content-addressed
versions. The pinned Weave SDK does not expose a reliable public API for
enumerating every historical object version; preparation validates the exact
`qualification-v1` object versions plus all relevant Calls and fails on any
ambiguity. It does not rewrite private labels to accommodate drift.

The checked-in `evidence.lock.json` belongs to the earlier single-project
study. It is retained only as historical audit input and is not the current
source lock.

## Import and lock exact MCP revisions

`mcp.json` declares the exact repositories and revisions. Import and lock them
from the trusted operator boundary:

```bash
uv run fugue mcp import \
  --config examples/comparisons/wandb-mcp-maintenance/mcp.json \
  --server wandb-main \
  --as wandb-mcp-main

uv run fugue mcp import \
  --config examples/comparisons/wandb-mcp-maintenance/mcp.json \
  --server wandb-0-4-staging \
  --as wandb-mcp-0-4-staging

uv run fugue mcp lock wandb-mcp-main \
  --acknowledge-package-code \
  --platform linux/amd64

uv run fugue mcp lock wandb-mcp-0-4-staging \
  --acknowledge-package-code \
  --platform linux/amd64
```

Preparation resolves package code and captures initialized tool manifests.
Agent cells receive locked assets and may not clone, install, or mutate either
candidate.

## Prove the mechanism before spending

Run the zero-model source-conformance check against the two exact Evaluation
roots:

```bash
uv run python \
  examples/comparisons/wandb-mcp-maintenance/verify_hosted_source.py \
  --lock .fugue/qualification/mcp-release-source-v1/evidence.lock.json \
  --env-file /Users/ashah/Documents/common_tools/.env \
  --output .fugue/qualification/mcp-release-source-conformance.json
```

The baseline must reproduce 18 direct children: 16 prediction-and-score calls
plus two summary calls. The candidate must report exactly the 16 direct
`Evaluation.predict_and_score` children.

Then qualify both locked MCP runtimes without an Agent model:

```bash
uv run python \
  examples/comparisons/wandb-mcp-maintenance/qualify_locked_revisions.py \
  --source-lock .fugue/qualification/mcp-release-source-v1/evidence.lock.json \
  --env-file /Users/ashah/Documents/common_tools/.env \
  --output .fugue/qualification/mcp-release-mechanism-receipt.json
```

Package release gates are separate and must produce
`.fugue/qualification/mcp-python-package-release-gates.json`. That receipt
binds the exact final staging tree, fresh wheels on Python 3.11 and 3.12,
W&B-latest compatibility, CI, dependency and source-security checks, local
conformance, and the human-maintainer actionability review.

## Preview the natural-maintainer studies

The eight-cell canary is the only first paid stage:

```bash
SPEC=examples/comparisons/wandb-mcp-maintenance/natural-maintainer-canary-local-v3.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json

env -u OPENAI_API_KEY uv run fugue compare "$SPEC" \
  --preview --env-file "$ENV_FILE" --json
```

Previewing does not authorize spend. An operator must approve that exact
preview digest with a maximum of eight cells and $10 before executing it.

Only if the canary is valid, non-regressing, reconciled, and useful may an
operator generate and separately approve the 16-cell confirmation:

```bash
SPEC=examples/comparisons/wandb-mcp-maintenance/natural-maintainer-confirmation-local-v3.yaml

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json

env -u OPENAI_API_KEY uv run fugue compare "$SPEC" \
  --preview --env-file "$ENV_FILE" --json
```

The confirmation requires a new approval capped at 16 cells and $20. Neither
command in this document runs a model or grants approval.

## What the tasks and scorer measure

Each attempt produces exactly one machine-readable JSON answer plus a concise
maintainer memo. Public prompts are stored separately from host-only expected
facts.

The deterministic scorer reports independent dimensions:

- factual task outcome;
- actual locked-project scope;
- bounded evidence collection;
- honest treatment of incomplete evidence;
- observed release-mechanism use.

Mechanism use cannot convert a factually wrong answer into a pass. Actual MCP
calls and queried scopes are authoritative; an incorrect project name written
in the Agent answer is shown separately. Human review judges whether the memo
is actionable for a maintainer and remains distinct from deterministic scores.

Fugue writes one stable attempt identity across the comparison design, Harbor
cell, Claude conversation, MCP calls, Weave Evaluation chain, Study event, and
result. A valid attempt resolves its Dataset, Evaluation root,
prediction-and-score call, prediction call, and native Agent root in the
result project. OTel identifiers remain diagnostic metadata and are never used
as Weave links.

## Reading the result

Study Console should be opened on the result project. Start with the behavioral
finding and package-release decision, then inspect each paired task:

1. Compare baseline and candidate facts, scope, boundedness, honesty, tools,
   cost, and latency.
2. Open the primary prediction-and-score evidence action.
3. Follow the Evaluation, prediction, Agent-root, and Dataset relationships.
4. Confirm the actual queried project is the locked source project.
5. Reconcile terminal cells, result rows, Evaluation rows, and native Agent
   roots before accepting the finding.

A `GO` decision additionally requires the immutable package-gate receipt and a
release-owner signature over the final result digest. A valid `HOLD` or
non-discriminating result remains a useful outcome and must not be rewritten
to manufacture a release win.

## Historical assets

The older `discovery*.yaml`, `primary*.yaml`,
`wandb-replication*.yaml`, and checked-in `evidence.lock.json` are retained for
audit compatibility. They are unregistered and are not current package-release
evidence. The Research comparison registry exposes only the canonical V3
canary and confirmation from this directory.
