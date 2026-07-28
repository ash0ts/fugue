# W&B MCP release-maintenance qualification

This is the real-evidence Fugue qualification workflow:

> Does the exact W&B MCP 0.4 revision improve how an Agent investigates
> bounded W&B and Weave maintenance evidence compared with exact 0.3.7?

It compares two complete Agent candidates. The task, model, harness, runtime,
attempt policy, and evidence snapshot are fixed inside each comparison. Only
the locked MCP integration revision changes.

The active path uses:

- W&B Serverless Sandboxes through Harbor's `wandb` environment;
- W&B API and W&B Inference credentials;
- Anthropic credentials for Claude Code and the blind judge;
- `anthropic/claude-sonnet-5` for Claude discovery and primary;
- `wandb/deepseek-ai/DeepSeek-V4-Flash` for OpenClaw discovery and
  replication.

It does not require CoreWeave or OpenAI. Local Harbor is the behavioral parity
baseline; W&B Serverless is the remote execution policy.

## Real hosted evidence

The immutable lock in `evidence.lock.json` points to the dedicated,
non-sensitive hosted project:

`wandb/fugue-mcp-release-qualification-v1`

The checked-in lock contains:

- six genuine W&B Runs with configuration, three-step histories, summaries,
  and versioned evidence artifacts;
- 24 standardized Weave Agent conversations and 48 tool spans;
- one versioned eight-row Weave Dataset;
- two versioned Weave Evaluations with eight aligned rows each;
- 16 Evaluation prediction rows;
- one observed latency anomaly, one missing-cost case, and one deliberately
  incomplete-evaluation case.

These deterministic objects are seeded prior evidence for investigating the
MCP behavior. They are not the result of the MCP comparison and are not
customer data.

Recreate or validate the snapshot idempotently:

```bash
uv sync --python 3.13 --frozen --extra dev --extra research-worker

uv run python \
  examples/comparisons/wandb-mcp-maintenance/prepare_hosted_project.py \
  --project wandb/fugue-mcp-release-qualification-v1 \
  --env-file /Users/ashah/Documents/common_tools/.env \
  --output \
  examples/comparisons/wandb-mcp-maintenance/evidence.lock.json
```

The preparation command loads only the required W&B value into its child
process and never writes a credential into the lock. An existing lock must
validate exactly before it is reused. Drift in counts, versions, seed identity,
or content is an error; do not edit private labels to match drifted data.

## Import and lock exact MCP revisions

The ordinary `mcp.json` declaration pins:

- 0.3.7: `80252b3aa23ae3c1fdde089ce2b7dfb106dafb38`
- 0.4: `a2bae7271323ac43262ffb73454b0aff01ddc808`

Load credentials into the trusted operator shell without printing them. The
declaration references environment variable names only:

```bash
set -a
source /Users/ashah/Documents/common_tools/.env
set +a
export WANDB_BASE_URL=https://api.wandb.ai

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

uv run fugue mcp lock wandb-mcp-0-3-7 \
  --acknowledge-package-code
uv run fugue mcp lock wandb-mcp-0-4 \
  --acknowledge-package-code
```

The lock operation prepares the exact target-platform MCP runtime and captures
its initialized tool manifest. Agent attempts do not clone repositories or
install MCP package code.

## Calibrate the blind judge

`judge-calibration-cases.jsonl` contains 48 authored, balanced examples: 24
passes and 24 failures, including critical unsupported-completeness failures.
Authored references are not human reviews.

Each example requires two distinct human reviewers. Disagreements require
adjudication. Add the blinded judge result, then regenerate the report:

```bash
uv run python \
  examples/comparisons/wandb-mcp-maintenance/validate_judge_calibration.py \
  --cases \
  examples/comparisons/wandb-mcp-maintenance/judge-calibration-cases.jsonl \
  --report \
  examples/comparisons/wandb-mcp-maintenance/judge-calibration.json
```

Paid execution remains blocked until the report has:

- 48 reviewed examples and two reviewers per example;
- adjudication for every disagreement;
- true-positive and true-negative rates of at least 0.85;
- zero false passes on critical unsupported-completeness cases.

## Exact 80-cell study

The full staged study is:

| Stage | Agent route | Tasks × revisions × attempts | Cells |
|---|---|---:|---:|
| Claude discovery | Claude Code + Anthropic Sonnet 5 | 4 × 2 × 1 | 8 |
| W&B discovery | OpenClaw + W&B DeepSeek V4 Flash | 4 × 2 × 1 | 8 |
| Claude primary | Claude Code + Anthropic Sonnet 5 | 8 × 2 × 2 | 32 |
| W&B replication | OpenClaw + W&B DeepSeek V4 Flash | 8 × 2 × 2 | 32 |
| Total | two direct provider routes | | 80 |

Preview and approve each immutable digest separately:

```bash
uv run fugue compare \
  examples/comparisons/wandb-mcp-maintenance/discovery.yaml \
  --preview

uv run fugue compare \
  examples/comparisons/wandb-mcp-maintenance/discovery-wandb.yaml \
  --preview

uv run fugue compare \
  examples/comparisons/wandb-mcp-maintenance/primary.yaml \
  --preview

uv run fugue compare \
  examples/comparisons/wandb-mcp-maintenance/wandb-replication.yaml \
  --preview
```

Use each preview's exact digest with `fugue approve`, including an explicit
cell and dollar cap, before `fugue compare --run --approval ...`. Do not pool
away a reversal between Claude and the W&B Inference Agent.

The full study remains blocked until judge calibration, both MCP runtime locks,
W&B Serverless organization access, and the public runtime-image lock pass.
The current credential can prepare W&B/Weave evidence, but the organization
must separately enable Serverless Sandboxes.

## Evidence and claims

Fugue reports four layers separately:

1. MCP initialization and infrastructure conformance.
2. Deterministic task outcomes.
3. Calibrated blind-judge dimensions.
4. Mechanism evidence: selected tools, projected and broad reads, sources
   returned/opened/used, structured errors, latency, tokens, observed cost,
   traces, and Evaluation reconciliation.

The study may support a whole-release recommendation only within the locked
models, Agents, tasks, attempts, runtimes, and dates. It may not attribute an
outcome to an individual MCP feature. A null or harness-reversing result is
valid but not presentation-ready, and missing required evidence is exit code
`3`, never a zero score.
