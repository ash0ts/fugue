# W&B MCP 0.4 maintainer qualification

This directory contains the canonical, source-isolated Fugue qualification for
the W&B MCP Python package:

> Does the exact 0.4 staging revision improve a maintainer's ability to
> reconcile W&B/Weave evidence and assess project health compared with the
> exact `main` revision?

The current release posture is **HOLD** until the natural-maintainer studies,
package gates, and human sign-off all pass. No V3 canary or confirmation result
has been approved or executed by this branch, so there is no current V3
behavioral conclusion. A later local behavioral finding will not certify W&B
Serverless, managed-service, or Helm readiness.

## Locked design

| Role | Locked value |
|---|---|
| Source evidence | `wandb/fugue-mcp-release-source-v2` |
| Result evidence | `wandb/fugue-mcp-release-qualification-v1` |
| Baseline | `wandb-mcp-main` at `53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0` |
| Candidate | [`wandb-mcp-0-4-staging` at `29cc1b5b5cf4061afa1faa712021fa1b68ad0bf7`](https://github.com/wandb/wandb-mcp-server/commit/29cc1b5b5cf4061afa1faa712021fa1b68ad0bf7) |
| Execution | local Docker through Harbor |
| Canary | 2 tasks × 2 arms × 2 attempts = 8 cells, maximum $10 |
| Confirmation | 4 tasks × 2 arms × 2 attempts = 16 cells, maximum $20 |

The checked-in candidate is the `staging/0.4.0` commit selected for this
qualification and contains the reconciliation and STDIO protocol-safety fixes.
The package gate must independently prove that it is still the final reviewed
staging head. Any later staging change invalidates the candidate runtime,
mechanism, preparation, preview, and approval locks. The source evidence lock
may be reused only after its normal drift verification still passes.

The source project contains a locked, read-only task-evidence cohort. Agent
traces, Evaluations, result rows, and release decisions go to the result
project.
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
  --source-project wandb/fugue-mcp-release-source-v2 \
  --result-project wandb/fugue-mcp-release-qualification-v1 \
  --env-file /Users/ashah/Documents/common_tools/.env \
  --output .fugue/qualification/mcp-release-source-v2/evidence.lock.json
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
versions. Preparation exhaustively inventories all 13 expected Weave object
and operation versions plus all 124 relevant Calls, and rejects any missing,
duplicate, unexpected, or changed identity. It does not rewrite private labels
to accommodate drift.

The checked-in `evidence.lock.json` belongs to the earlier single-project
study. It is retained only as historical audit input and is not the current
source lock.

## Import and lock exact MCP revisions

`mcp.json` declares the exact repositories and revisions. Import and lock them
from the trusted operator boundary:

```bash
ENV_FILE=/Users/ashah/Documents/common_tools/.env

uv run fugue mcp import \
  --config examples/comparisons/wandb-mcp-maintenance/mcp.json \
  --server wandb-main \
  --as wandb-mcp-main

uv run fugue mcp import \
  --config examples/comparisons/wandb-mcp-maintenance/mcp.json \
  --server wandb-0-4-staging \
  --as wandb-mcp-0-4-staging

uv run --env-file "$ENV_FILE" fugue mcp lock wandb-mcp-main \
  --acknowledge-package-code \
  --platform linux/amd64

uv run --env-file "$ENV_FILE" fugue mcp lock wandb-mcp-0-4-staging \
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
  --evidence-lock .fugue/qualification/mcp-release-source-v2/evidence.lock.json \
  --env-file /Users/ashah/Documents/common_tools/.env \
  --output .fugue/qualification/mcp-release-source-conformance.json
```

This first receipt proves only that the locked source cohort contains 18 direct
children: 16 prediction-and-score calls plus two summary calls. It does not
exercise either MCP revision.

Then qualify both locked MCP runtimes without an Agent model:

```bash
uv run python \
  examples/comparisons/wandb-mcp-maintenance/qualify_locked_revisions.py \
  --evidence-lock .fugue/qualification/mcp-release-source-v2/evidence.lock.json \
  --env-file /Users/ashah/Documents/common_tools/.env \
  --output .fugue/qualification/mcp-release-mechanism-receipt.json
```

The second receipt must show that main reports all 18 direct children while
the repaired staging runtime reports exactly the 16
`Evaluation.predict_and_score` children.

Package release gates are separate and must produce
`.fugue/qualification/mcp-python-package-release-gates.json`. That receipt
binds the exact final staging tree, fresh wheels on Python 3.11 and 3.12,
W&B-latest compatibility, CI, dependency and source-security checks, local
conformance, and the declared package infrastructure gates. Human actionability
review is a separate post-result `DecisionAttestationV1`; it is never inferred
from this infrastructure receipt.

## Preview the natural-maintainer studies

The eight-cell canary is the only first paid stage:

```bash
SPEC=examples/comparisons/wandb-mcp-maintenance/natural-maintainer-canary-local-v3.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json

env -u OPENAI_API_KEY uv run fugue compare "$SPEC" \
  --prepare --env-file "$ENV_FILE" --json

env -u OPENAI_API_KEY uv run fugue compare "$SPEC" \
  --preview --env-file "$ENV_FILE" --json
```

`--prepare` is the trusted no-spend boundary that freezes inputs and builds the
exact local task and Agent images. It does not run a cell or create approval.
Previewing also does not authorize spend. An operator must approve that exact
preview digest with a maximum of eight cells and $10 before executing it with
`--fetch-weave`; without fetched evidence links the V3 result cannot qualify.

The confirmation is intentionally absent from the Research comparison
registry. Only after the canonical canary Result is valid, non-regressing,
reconciled, and owner-reviewed may the release workstream bind that exact
qualification digest, register the confirmation, and generate its separate
preview:

```bash
SPEC=examples/comparisons/wandb-mcp-maintenance/natural-maintainer-confirmation-local-v3.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json

env -u OPENAI_API_KEY uv run fugue compare "$SPEC" \
  --prepare --env-file "$ENV_FILE" --json

env -u OPENAI_API_KEY uv run fugue compare "$SPEC" \
  --preview --env-file "$ENV_FILE" --json
```

Until that prerequisite is recorded, the command above is specification
inspection only and the Study must not be launched. The confirmation requires
a new approval capped at 16 cells and $20. The confirmation `check`, `prepare`,
and `preview` commands do not run a model or grant approval.

After a valid, non-regressing canary has completed and a maintainer has reviewed
it as useful, authorize the exact result for the declared confirmation:

```bash
uv run fugue result mcp-main-vs-0-4-natural-maintainer-canary-v3 \
  --authorize-followup \
  examples/comparisons/wandb-mcp-maintenance/natural-maintainer-confirmation-local-v3.yaml \
  --reviewed-by RELEASE_OWNER
```

This writes the confirmation spec's canonical prerequisite result and
attestation. It does not register, approve, or execute the confirmation.

## What the tasks and scorer measure

Each attempt produces exactly one machine-readable JSON answer plus a concise
maintainer memo. Public prompts are stored separately from host-only expected
facts.

The deterministic scorer reports independent dimensions:

- factual task outcome;
- actual locked-project scope;
- project identity reported in the Agent answer;
- bounded evidence collection;
- honest treatment of incomplete evidence;
- observed release-mechanism use.

Mechanism use cannot convert a factually wrong answer into a pass. Actual MCP
calls and queried scopes are authoritative; an incorrect project name written
in the Agent answer is shown separately. Human review judges whether the memo
is actionable for a maintainer and remains distinct from deterministic scores.

Fugue writes one stable attempt identity across the comparison design, Harbor
cell, Claude conversation, MCP calls, Weave Evaluation chain, Study event, and
result. A valid attempt resolves the Dataset in the locked source project and
the Evaluation root, prediction-and-score call, prediction call, and native
Agent root in the result project, with a verified cross-project relationship.
OTel identifiers remain diagnostic metadata and are never used as Weave links.

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

Evidence grade `A` means that required lineage, links, and privacy checks
reconcile. It does not mean the tasks passed, the candidate improved, or the
package is ready to release.

A `GO` decision additionally requires the immutable package-gate receipt and a
release-owner actionability signature over a `ready_for_signoff` result:

```bash
uv run fugue result mcp-main-vs-0-4-natural-maintainer-confirmation-v3 \
  --signoff-by RELEASE_OWNER
```

The command signs the exact qualification digest; it does not merge or publish
the package. A valid `HOLD` or
non-discriminating result remains a useful outcome and must not be rewritten
to manufacture a release win.

## Historical assets

The older `discovery*.yaml`, `primary*.yaml`,
`wandb-replication*.yaml`, and checked-in `evidence.lock.json` are retained for
audit compatibility. They are unregistered and are not current package-release
evidence. The Research comparison registry exposes only the canonical V3
canary from this directory; confirmation remains unregistered until its
prerequisite result and owner review exist.

The source-v1 invalidation record describes a failed preparation and has no
ComparisonResult digest. It must not be used as a V3 `supersedes` entry. A
historical Study result may be marked superseded only after its exact immutable
result digest is recovered.
