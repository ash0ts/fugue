# W&B MCP 0.4 local reference study

This packaged reference study compares the exact W&B MCP baseline commit with
the exact commit resolved from `staging/0.4.0` during trusted preparation. It
is an example of Fugue evaluating an integration; it is not a WBAF runtime,
GitHub App, or W&B-hosted Fugue service.

The four tasks exercise bounded structured Run inventory, cursor pagination,
exact history targeting, and direct Evaluation-child reconciliation. They are
copied from the reviewed V8 taskset and use the pinned V7 deterministic scorer.
The canary is useful behavioral evidence, but it is not complete coverage of
the release notes and cannot by itself qualify a managed service or Helm
release.

## Prepare and run locally

Install the local runner and MCP preparation support:

```bash
python -m pip install "fugue[local-runner,mcp,weave]"
```

Resolve the moving staging branch twice, freeze the exact commit and tree,
materialize the packaged study, and lock both MCP runtimes outside trials:

```bash
fugue mcp prepare-wandb-release \
  --repo-root "$PWD" \
  --env-file /path/to/.env \
  --platform linux/amd64 \
  > /tmp/wandb-mcp-preparation.json
```

Preparation prints the exact candidate SHA, generated comparison path, and
immutable receipt. It never creates a mutable `current` pointer. Read the
digest-bound path and use the ordinary Fugue lifecycle on that exact file:

```bash
COMPARISON=$(python -c 'import json; print(json.load(open("/tmp/wandb-mcp-preparation.json"))["comparison_path"])')

fugue check "$COMPARISON" \
  --env-file /path/to/.env --json
fugue compare "$COMPARISON" \
  --preview --env-file /path/to/.env --json
fugue approve PREVIEW_DIGEST --max-cells 8 --max-usd 10
fugue compare "$COMPARISON" \
  --run --approval APPROVAL_DIGEST --env-file /path/to/.env
fugue result latest
```

`WANDB_API_KEY` authenticates the read-only hosted-source freeze and the two
locked MCP servers because W&B is the integration under test. Preparation
fails rather than seeding, repairing, or silently accepting drift in that
source. Fugue writes its canonical result evidence locally and does not use
W&B or Weave as its result backend. The selected model route requires
`ANTHROPIC_API_KEY`. Credential values and the host-only private labels are
never copied into task inputs, locks, traces, results, or CI artifacts.

If desired, publish the unchanged completed result after the fact:

```bash
fugue publish weave /path/to/result.json \
  --project ENTITY/PROJECT \
  --env-file /path/to/.env
```

Publication is optional and cannot change result counts, scores,
classifications, or the canonical local result digest.

## Provenance and limitations

- Baseline: `53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0`.
- Candidate: the exact commit and tree frozen from `staging/0.4.0` by the
  preparation receipt; never the branch name during a trial.
- Source evidence: the read-only, non-sensitive
  `wandb/fugue-mcp-release-source-v2` cohort named in each task and private
  contract.
- WBAF commit `e2d8d670017bc426b68a311c5777c3b9084023f3` informed task-design
  review only. WBAF is not imported, installed, or executed.
- Local Harbor evidence supports a bounded behavioral conclusion. Package
  release, remote isolation, managed service, and Helm gates remain separate.

The generated `private-labels.jsonl` is written with mode `0600`. Do not commit
it or send it to an Agent. Trials receive only its digest.
