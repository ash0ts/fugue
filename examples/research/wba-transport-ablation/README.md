# WBA transport compatibility suite

This example is an offline compatibility and conformance fixture for the
`wba-responses` harness. It is not a live experiment runbook.

PR #17 originally proposed a three-profile transport ablation. The completed
V1 cohort could not answer that question because its output contract was
undisclosed and its stream diagnostics were asymmetric. V2 repairs those
contracts, but the tested LiteLLM proxy images do not satisfy the required
Responses streaming-event contract.

No V2 canary or primary cohort is part of this PR. Do not approve, launch, or
partially execute either preset.

## What this PR qualifies

- a visible JSON Schema with private expected values addressed by JSON pointer;
- base-fail/reference-pass validation for every materialized task;
- shared turn and tool-call validation across all three profiles;
- shared Responses-event validation across both Responses profiles;
- separate Agent-call and compaction retry, timeout, and fallback accounting;
- queryable model-call evidence and corrected normalized exports;
- fail-closed proxy support based on a reproducible offline conformance test.

The registered V2 matrix remains as a reproducible fixture for a separately
scoped future decision. Its `responses-proxy` coordinate is currently blocked
by preflight, and this PR has no live-execution acceptance gate.

## Current result

The only defensible result is:

> V1 qualified the 48-cell execution path but not the transport comparison.
> V2 correctly rejects the tested proxy before approval or spend.

Read the reviewed [`v1-results.md`](./v1-results.md) for the bounded
interpretation. The machine-readable
[`v1-shareable-results.json`](./v1-shareable-results.json) contains only
allowlisted aggregate evidence. The immutable post-hoc diagnostic remains in
[`v1-exploratory-audit.json`](./v1-exploratory-audit.json).

## Run the offline gates

Run the focused contract, scorer, evidence, and publication checks:

```bash
uv run --frozen pytest -q \
  tests/test_wba_transport.py \
  tests/test_model_plane.py \
  tests/test_task_runtime.py \
  tests/test_export.py \
  tests/test_experiment_atlas.py
```

Reproduce proxy qualification against an immutable image:

```bash
uv run --frozen python scripts/check_wba_proxy_conformance.py \
  ghcr.io/berriai/litellm@sha256:IMAGE_DIGEST
```

The command must fail for a non-conformant image and write no support claim.
An image is supportable only when every required Responses event carries a
strictly increasing sequence number, output-item ordering is valid, and the
fragmented tool call reconciles exactly once.

## Publication boundary

The shareable artifacts contain no raw Agent content, trace content, prompts,
private expected values, local paths, secrets, or observed-cost claim. No Atlas
record is created. This independently authored harness does not import,
execute, or establish a result about `wandb/core`.
