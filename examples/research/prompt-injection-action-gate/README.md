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

After Aria returns an eligible exact preview, approve only its digest in a
trusted terminal:

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
`GET /v1/experiments/{id}/events:watch` with a cursor and at most a 30-second
wait; terminal replay begins at cursor zero. Do not create an Atlas record or
expose private expected facts.
