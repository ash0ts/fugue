# Queryable WBA transport ablation

This example lets an external research Agent preview, run, monitor, and later
query a controlled transport experiment through Fugue. The Agent loop, model,
tool, prompt, task resources, runtime, and sampling stay fixed. Only the route
from the loop to the W&B Inference Chat endpoint changes:

```text
responses-proxy   OpenAI Responses client -> Fugue proxy -> Chat Completions
responses-inline  LiteLLM Responses conversion in the Agent process
chat-inline       LiteLLM Chat Completions reference
```

The live qualification is:

```text
1 locked task × 3 transport profiles × 1 attempt = 3 cells
```

If its execution evidence is eligible, the unchanged primary design is:

```text
8 locked tasks × 3 transport profiles × 2 attempts = 48 cells
```

Both runs are serial, have no cell retries, and require separate human
approvals bound to their exact preview digests. Task passes are observations.
Preparation, isolation, identity, route receipts, accounting, and reconciled
evidence determine whether the research result is usable.

This is an independently authored compatibility study. It does not import,
execute, or make claims about `wandb/core`. A later Core-integrated
qualification remains a separate proposed experiment.

## Start the governed laboratory

From a clean Fugue checkout:

```bash
uv run --frozen fugue research bootstrap \
  --repo-root . \
  --env-file /Users/ashah/Documents/common_tools/.env

docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml up --build -d

curl --fail http://127.0.0.1:8787/readyz
```

Bootstrap reads only allowlisted credentials. It never sources the dotenv file
or copies unrelated values.

## Connect an external Agent

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

cp examples/research/wba-transport-ablation/agent-prompt.md \
  "$FUGUE_DEMO_DIR/agent-prompt.md"

codex -s read-only -C "$FUGUE_DEMO_DIR" \
  "$(cat "$FUGUE_DEMO_DIR/agent-prompt.md")"
```

The Agent creates or recovers the Study, reads the safe catalog, previews the
exact three-cell qualification, and stops. Keep that Agent task open.

## Approve the qualification

In a separate trusted terminal, copy the returned preview digest:

```bash
export FUGUE_CANARY_PREVIEW_DIGEST=REPLACE_WITH_CANARY_PREVIEW_DIGEST

docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml run --rm --no-deps \
  fugue-operator approve "$FUGUE_CANARY_PREVIEW_DIGEST" \
  --max-usd 200 \
  --max-cells 3 \
  --approved-by "$USER"
```

Paste the returned `approval_digest` into the original Agent task. The Agent
starts only the unchanged preview, follows its resumable event cursor, and
records a bounded Result with exact evidence references. If the outcome is
evidence-eligible, it previews the unchanged 48-cell primary and stops again.

## Approve the primary

Copy the second preview digest only after inspecting its parent evidence and
estimate:

```bash
export FUGUE_PRIMARY_PREVIEW_DIGEST=REPLACE_WITH_PRIMARY_PREVIEW_DIGEST

docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml run --rm --no-deps \
  fugue-operator approve "$FUGUE_PRIMARY_PREVIEW_DIGEST" \
  --max-usd 1800 \
  --max-cells 48 \
  --approved-by "$USER"
```

Paste that `approval_digest` into the same Agent task. The Agent starts the
unchanged primary, watches it to terminal, and records the sourced analysis.
The two approvals remain within the campaign's cumulative $2,000 ceiling.

## Query the finished Study from a fresh Agent

Use the Study and experiment identifiers returned by the first Agent:

```bash
cp examples/research/wba-transport-ablation/query-prompt.md \
  "$FUGUE_DEMO_DIR/query-prompt.md"

export STUDY_ID=REPLACE_WITH_STUDY_ID
export EXPERIMENT_ID=REPLACE_WITH_PRIMARY_EXPERIMENT_ID
QUERY_PROMPT="$(awk \
  -v study="$STUDY_ID" \
  -v experiment="$EXPERIMENT_ID" \
  '{gsub(/\$STUDY_ID/, study); gsub(/\$EXPERIMENT_ID/, experiment); print}' \
  "$FUGUE_DEMO_DIR/query-prompt.md")"
codex -s read-only -C "$FUGUE_DEMO_DIR" "$QUERY_PROMPT"
```

The fresh Agent reads immutable Study references and route receipts. It may
propose a later Core-integrated qualification, but it cannot approve or launch
paid work itself.

Stop services without deleting the durable Study volume:

```bash
docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml down
```
