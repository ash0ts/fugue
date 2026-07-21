# Prompt-injection loop study

This example lets an external Agent use Fugue as a governed outer research
loop. It compares three defenses across Codex and Claude Code on three offline
tasks:

```text
3 tasks × 3 loop treatments × 2 harnesses × 1 attempt = 18 cells
```

The tasks contain only a synthetic sentinel, a harmless local capture sink,
and immutable offline fixtures. Harbor runs without external network access.
The Agent can preview and start an operator-approved plan, but cannot approve
spend or change the registered matrix.

## Start the research stack

```bash
uv run --frozen fugue research bootstrap \
  --repo-root . \
  --env-file /Users/ashah/Documents/common_tools/.env

uv run --frozen fugue setup \
  --experiment prompt-injection-loop-v1 \
  --preset study \
  --prepare \
  --env-file /Users/ashah/Documents/common_tools/.env

FUGUE_TRACE_SOURCES_FILE="$PWD/examples/research/prompt-injection-loop/trace-sources.compose.yaml" \
FUGUE_TRACE_DATA_DIR="$PWD/configs/fugue/research/fixtures" \
docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml up --build fugue-control fugue-worker
```

The control API is available at `http://127.0.0.1:8787`. Use the token in
`.fugue/secrets/research_api_key` only as a secret HTTP bearer token.

## Approval boundary

After the external Agent returns an eligible exact preview, approve only its
digest in a trusted terminal:

```bash
docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml run --rm --no-deps fugue-operator \
  approve PREVIEW_DIGEST \
  --max-cells 18 \
  --max-usd 500 \
  --approved-by "$USER"
```

The external Agent may then submit the unchanged preview without receiving the
approval receipt. Fugue resolves the prior exact operator approval server-side.
An unapproved or drifted preview still fails closed.

Use `GET /v1/experiments/{id}/events:watch` with an event cursor and at most a
30-second wait. Terminal replay begins at cursor zero. Do not create an Atlas
record or expose expected task facts.
