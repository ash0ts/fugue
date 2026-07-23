# Prompt-injection action-gate study

This example lets an external Agent use Fugue as a governed outer research
loop. It compares no extra defense, a one-line warning, and an enforced
pre-tool action gate across Codex and Claude Code:

```text
6 tasks × 3 treatments × 2 harnesses × 2 attempts = 72 cells
```

Four tasks contain realistic indirect instructions in repository, trace, CI,
or support data. Two benign controls require legitimate sensitive actions so a
defense cannot pass by refusing everything. Credentials are synthetic, sinks
are local, and prepared Harbor task images run without external network access.

## Start the research stack

```bash
uv run --frozen fugue research bootstrap \
  --repo-root . \
  --env-file /Users/ashah/Documents/common_tools/.env

uv run --frozen fugue setup \
  --experiment prompt-injection-action-gate-v1 \
  --preset study \
  --prepare \
  --env-file /Users/ashah/Documents/common_tools/.env

FUGUE_TRACE_SOURCES_FILE="$PWD/examples/research/prompt-injection-action-gate/trace-sources.compose.yaml" \
FUGUE_TRACE_DATA_DIR="$PWD/configs/fugue/research/fixtures" \
docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml up --build fugue-control fugue-worker
```

The control API is available at `http://127.0.0.1:8787`. Use the token in
`.fugue/secrets/research_api_key` only as a secret HTTP bearer token.

## Approval boundary

After Aria returns an eligible exact Study preview, approve only its digest in
a trusted terminal:

```bash
docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml run --rm --no-deps fugue-operator \
  approve PREVIEW_DIGEST \
  --max-cells 72 \
  --max-usd 500 \
  --approved-by "$USER"
```

Aria may then submit the unchanged preview without receiving the approval
receipt. Fugue resolves the exact operator approval server-side. Use
`GET /v1/research-studies/{id}/events:watch` with a cursor and at most a
30-second wait; terminal replay begins at cursor zero. Do not create an Atlas
record or expose private expected facts.

## Completed reference run

The exact-head Research is `aria-action-gate-loop-demo-v2`; its controlled
Study is `aria-action-gate-loop-demo-v2.aria-action-gate-001-7734e37566b7`
and its admitted Run is `20260721T165054-b3901c8ced`. It reconciled all 72
serial cells on source `8e83e300e4ba71965a09722ab7a60ec4c635bd8e`
with snapshot
`5346a09a141b271afeba41263d14806dc632240c0a97cbbb0af9a1c1dd25cd58`.

The official deterministic classifications were:

| treatment | safe and useful | compromised | incorrect | safe but refused |
| --- | ---: | ---: | ---: | ---: |
| baseline | 17 / 24 | 4 / 24 | 3 / 24 | 0 / 24 |
| warning only | 17 / 24 | 4 / 24 | 3 / 24 | 0 / 24 |
| action gate | 20 / 24 | 0 / 24 | 4 / 24 | 0 / 24 |

The native gate blocked seven attempted hostile actions and allowed eight
authorized sensitive actions. All 24 benign controls passed. Total accounted
cost was `$7.526006`; evidence eligibility passed with 72 unique prediction,
conversation, trace, and Agent-root identities and no infrastructure or
evidence-contract failures.

Treat the result as replication-worthy rather than conclusive. The
task-cluster bootstrap interval for the official safe-and-useful difference
crosses zero, and the `poisoned-ci-log` prompt did not define the exact label
required by its verifier. All twelve Agents diagnosed runner clock skew; a
transparent post-hoc semantic sensitivity therefore yields `24/24` for the
action gate and `20/24` for each comparator, but that is not the primary score.

Inspect the preserved run locally:

```bash
uv run --frozen fugue runs 20260721T165054-b3901c8ced --json
uv run --frozen fugue runs 20260721T165054-b3901c8ced open agents --print --json
uv run --frozen fugue tui
```

The Research contains the sourced Result and one unapproved 72-cell replication
Study preview. Do not approve that child until the CI-label scorer is repaired
and a new immutable preview has been reviewed.
