# Queryable retrieval-to-action qualification

This example lets an external Agent preview, run, monitor, and later query one
small loop-engineering experiment through Fugue. It compares repository search
OFF/ON with standard versus inspect-and-verify instructions across Codex and
Claude Code:

```text
1 task × 2 harnesses × 4 treatments × 1 attempt = 8 cells
```

The run is serial, has no retries, and requires a separate operator approval
bound to the exact preview digest. Repair pass count is an observation; the
qualification gate is trustworthy execution and evidence.

This is the smallest execution qualification, not the full research story. To
demonstrate a fresh Agent reading trace evidence, recording a parent Result, and
handing one lineage-bound child preview to another Agent, continue with the
[`autoresearch-loop` example](../autoresearch-loop/README.md).

## Start Fugue

From a clean Fugue checkout:

```bash
uv run --frozen fugue research bootstrap \
  --repo-root . \
  --env-file /Users/ashah/Documents/common_tools/.env

docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml up --build -d

curl --fail http://127.0.0.1:8787/readyz
```

Bootstrap reads only the allowlisted `WANDB_API_KEY` value. It does not source
the dotenv file or copy unrelated credentials.

## Connect Codex as the external Agent

```bash
FUGUE_DEMO_DIR="$(mktemp -d)"
mkdir -p "$FUGUE_DEMO_DIR/.agents/skills"

docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml run --rm --no-deps \
  --user "$(id -u):$(id -g)" \
  -v "$FUGUE_DEMO_DIR/.agents/skills:/export" \
  fugue-operator skill export \
  --destination /export/optimize-agent-with-fugue

export FUGUE_RESEARCH_TOKEN
FUGUE_RESEARCH_TOKEN="$(<.fugue/secrets/research_api_key)"

codex mcp add fugue \
  --url http://127.0.0.1:8787/mcp/ \
  --bearer-token-env-var FUGUE_RESEARCH_TOKEN

cp examples/research/retrieval-to-action-canary/agent-prompt.md \
  "$FUGUE_DEMO_DIR/agent-prompt.md"

codex -s read-only -C "$FUGUE_DEMO_DIR" \
  "$(cat "$FUGUE_DEMO_DIR/agent-prompt.md")"
```

If `codex mcp get fugue` shows an older endpoint, remove that entry with
`codex mcp remove fugue` before adding it again.

The Agent stops after preview and prints the exact digest and estimate. Keep
that Codex task open.

## Approve and run

In a second trusted terminal, copy the preview digest into
`FUGUE_PREVIEW_DIGEST`, then run:

```bash
export FUGUE_PREVIEW_DIGEST=REPLACE_WITH_PREVIEW_DIGEST

docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml run --rm --no-deps \
  fugue-operator approve "$FUGUE_PREVIEW_DIGEST" \
  --max-usd 200 \
  --max-cells 8 \
  --approved-by "$USER"
```

Paste the returned `approval_digest` into the original Codex task. The Agent
submits the unchanged preview, follows durable events, and records a sourced
Result only after the experiment is terminal.

## Query the evidence from another Agent

Use the Study and experiment identifiers returned above:

```bash
codex -s read-only -C "$FUGUE_DEMO_DIR" \
  '$optimize-agent-with-fugue Read Study STUDY_ID and experiment EXPERIMENT_ID through Fugue. Inspect the exact outcome evidence, compare the four treatments on aligned cells, and report observations, limitations, and the next justified experiment without declaring a universal winner.'
```

Stop the services without deleting the durable Study volume:

```bash
docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml down
```
