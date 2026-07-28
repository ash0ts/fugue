# W&B MCP release-maintenance comparison

This is the advanced Fugue technical-preview example. It asks a real
maintenance question:

> Does the exact W&B MCP 0.4 candidate improve how an Agent investigates W&B
> evidence compared with the exact 0.3.7 baseline?

It is a comparison of two complete Agent candidates, not a special “MCP eval”
mode. The model, task, prompt base, runtime, and harness are fixed. Only the
locked MCP integration revision changes.

No result is checked in and no paid run has been performed under this example.
The current judge calibration is deliberately marked pending. `fugue check`
must block paid execution until a real 48-example, double-reviewed calibration
passes.

## Evidence project

The tasks target the coherent demo project:

`ashah-weights-biases/loop-engineering-demo`

Before running, that project must contain the immutable demo evidence described
in the task labels: six W&B Runs, 24 source conversations, eight evaluated
Agent attempts, a Weave Dataset, and a Weave Evaluation. Use the existing
loop-engineering demo seeder; do not edit labels to match an incomplete
workspace.

Set the local evidence endpoint and key without writing either value into an
MCP declaration:

```bash
export WANDB_BASE_URL=https://api.wandb.test
export WANDB_API_KEY
```

Cloud Agent and judge inference remain separately routed and billed by Fugue’s
normal inference configuration.

## Import ordinary MCP configurations

The included `mcp.json` is a normal `mcpServers` document. Each selected server
is pinned to one full Git commit:

- 0.3.7 baseline:
  `80252b3aa23ae3c1fdde089ce2b7dfb106dafb38`
- 0.4 candidate:
  `a2bae7271323ac43262ffb73454b0aff01ddc808`

Import and inspect only those two declarations:

```bash
uv run fugue mcp import \
  --config examples/comparisons/wandb-mcp-maintenance/mcp.json \
  --server wandb-0-3-7 \
  --as wandb-mcp-0-3-7

uv run fugue mcp import \
  --config examples/comparisons/wandb-mcp-maintenance/mcp.json \
  --server wandb-0-4 \
  --as wandb-mcp-0-4

uv run fugue mcp inspect wandb-mcp-0-3-7
uv run fugue mcp inspect wandb-mcp-0-4
```

Locking explicitly acknowledges that preparation executes package code. Fugue
checks out the exact revision, builds a wheel, initializes the MCP server,
captures its tool schemas, and materializes a read-only runtime:

```bash
uv run fugue mcp lock wandb-mcp-0-3-7 \
  --acknowledge-package-code
uv run fugue mcp lock wandb-mcp-0-4 \
  --acknowledge-package-code
```

The resulting Harbor attempts use the prepared runtime. They do not run `uvx`,
clone Git repositories, install packages, or edit a user-global Agent
configuration.

## Calibrate the blind judge

Deterministic facts are the primary gate. The blind judge separately measures:

- evidence grounding;
- usefulness to a maintainer;
- prioritization and actionability;
- calibration about incomplete investigation.

The judge sees the public task, final response, permitted tool names, and public
rubric. It does not see the MCP revision, baseline/candidate identity, harness,
deterministic result, private expected values, receipts, or internal IDs.

Replace `judge-calibration-cases.jsonl` with 48 balanced reviewed examples.
Every example needs two reviewers; disagreements must be adjudicated. Then
write the measured confusion matrix and locked rubric digest to
`judge-calibration.json`. Required thresholds are:

- at least 0.85 true-positive rate;
- at least 0.85 true-negative rate;
- zero false passes on critical unsupported-completeness examples.

Until that is true, this is the expected result:

```bash
uv run fugue check \
  examples/comparisons/wandb-mcp-maintenance/discovery.yaml
# Status: blocked — calibration is not adjudicated
```

This is intentional. A plausible rubric is not the same thing as a calibrated
evaluator.

## Run the study in stages

Once the imported integrations and judge are qualified:

```bash
# Four discovery tasks × two revisions × two harnesses × one attempt = 16
uv run fugue compare \
  examples/comparisons/wandb-mcp-maintenance/discovery.yaml \
  --preview
```

Use discordant discovery traces to write a maintainer note. Do not change the
holdout tasks or scorer from discovery observations.

The primary confirmation is:

```bash
# Eight holdouts × two revisions × Codex × two attempts = 32
uv run fugue compare \
  examples/comparisons/wandb-mcp-maintenance/primary.yaml \
  --preview
```

Approve only the exact returned digest in a trusted terminal:

```bash
uv run fugue approve PREVIEW_DIGEST --max-usd 80 --max-cells 32
uv run fugue compare \
  examples/comparisons/wandb-mcp-maintenance/primary.yaml \
  --run --approval APPROVAL_DIGEST
```

Only after interpreting the Codex primary should the separate Claude Code
replication be previewed. Never pool away a harness reversal:

```bash
uv run fugue compare \
  examples/comparisons/wandb-mcp-maintenance/claude-replication.yaml \
  --preview
```

## Read the result

Fugue reports four layers separately:

1. MCP preparation and infrastructure conformance.
2. Deterministic task pass.
3. Blind-judge quality dimensions.
4. Tool behavior, trace links, latency, tokens, and available observed cost.

It does not average them into one quality number. A missing required judge
returns CI exit code `3`; it never turns a completed Agent attempt into a task
failure. A null difference is not evidence that the two revisions are
equivalent.
