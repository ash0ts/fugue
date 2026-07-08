#!/usr/bin/env bash
# Preflight: verify everything the smoke/pilot runs depend on, without
# starting any agent. Safe to re-run; total cost is a couple of cents.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"                    # repo root

PASS=0
FAIL=0
ok()   { echo "  [ok]   $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "== fugue preflight =="

# 1. Env
if [ -f "$ROOT/.env" ]; then
    set -a; source "$ROOT/.env"; set +a
    ok "loaded $ROOT/.env"
else
    bad "missing $ROOT/.env (copy .env.example)"
fi
[ -n "${WANDB_API_KEY:-}" ] && ok "WANDB_API_KEY set" || bad "WANDB_API_KEY empty"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-fugue-local}"

# Weave trace routing: every harness plugin ships spans to entity/project.
if [ -n "${WANDB_ENTITY:-}" ] && [ -n "${WANDB_PROJECT:-}" ]; then
    ok "Weave project: ${WEAVE_PROJECT:-$WANDB_ENTITY/$WANDB_PROJECT}"
else
    bad "WANDB_ENTITY/WANDB_PROJECT unset (needed for Weave tracing)"
fi

BASE_URL="${WANDB_INFERENCE_BASE_URL:-https://api.inference.wandb.ai/v1}"
MODEL_ID="${FUGUE_MODEL:-zai-org/GLM-5.2}"

# Local hermes-otel checkout (uploaded into Hermes containers).
HERMES_OTEL="${HERMES_OTEL_CHECKOUT:-$HOME/Documents/GitHub/hermes-otel}"
if [ -f "$HERMES_OTEL/plugin.yaml" ]; then
    ok "hermes-otel checkout at $HERMES_OTEL"
else
    bad "hermes-otel checkout missing plugin.yaml at $HERMES_OTEL"
fi

# 2. Docker
if docker info >/dev/null 2>&1; then ok "docker daemon reachable"; else bad "docker daemon not reachable"; fi

# 3. W&B Inference direct (OpenAI protocol)
if curl -sS -m 20 "$BASE_URL/models" -H "Authorization: Bearer $WANDB_API_KEY" | grep -q "\"$MODEL_ID\""; then
    ok "W&B Inference lists $MODEL_ID"
else
    bad "W&B Inference /models missing $MODEL_ID"
fi
TOOLRES=$(curl -sS -m 40 "$BASE_URL/chat/completions" \
    -H "Authorization: Bearer $WANDB_API_KEY" -H "content-type: application/json" \
    -d "{\"model\":\"$MODEL_ID\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"call the ping tool\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"ping\",\"description\":\"ping\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}" \
    | jq -r '.choices[0].finish_reason // .error.message' 2>/dev/null)
if [ "$TOOLRES" = "tool_calls" ]; then ok "chat completions + tool calling"; else bad "tool-calling check returned: $TOOLRES"; fi

# 4. Anthropic bridge (for Claude Code)
docker compose -f "$ROOT/proxy/docker-compose.yaml" up -d >/dev/null 2>&1
for _ in $(seq 1 30); do
    LIVE=$(curl -sS -m 3 http://127.0.0.1:4000/health/liveliness 2>/dev/null || true)
    [ "$LIVE" = '"I\x27m alive!"' ] || [ "$LIVE" = '"I'"'"'m alive!"' ] && break
    sleep 2
done
if [ -n "${LIVE:-}" ]; then ok "anthropic bridge container is up"; else bad "anthropic bridge did not become healthy"; fi
STOP=$(curl -sS -m 60 http://127.0.0.1:4000/v1/messages \
    -H "x-api-key: $LITELLM_MASTER_KEY" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" \
    -d "{\"model\":\"$MODEL_ID\",\"max_tokens\":1200,\"messages\":[{\"role\":\"user\",\"content\":\"use the ping tool\"}],\"tools\":[{\"name\":\"ping\",\"description\":\"ping\",\"input_schema\":{\"type\":\"object\",\"properties\":{}}}]}" \
    | jq -r '.stop_reason // .error.message' 2>/dev/null)
if [ "$STOP" = "tool_use" ]; then ok "bridge /v1/messages + tool use (claude code path)"; else bad "bridge tool-use check returned: $STOP"; fi
RSTAT=$(curl -sS -m 60 http://127.0.0.1:4000/v1/responses \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "content-type: application/json" \
    -d "{\"model\":\"$MODEL_ID\",\"input\":\"call the ping tool\",\"max_output_tokens\":1200,\"tools\":[{\"type\":\"function\",\"name\":\"ping\",\"description\":\"ping\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}]}" \
    | jq -r '.status // .error.message' 2>/dev/null)
if [ "$RSTAT" = "completed" ]; then ok "bridge /v1/responses + tools (codex path)"; else bad "bridge responses check returned: $RSTAT"; fi

# 5. Harbor + custom agents importable
HARBOR_PY="$(dirname "$(readlink -f "$(command -v harbor)")")/python"
if PYTHONPATH="$ROOT" "$HARBOR_PY" -c "import fugue.agents" 2>/dev/null; then
    ok "fugue agents import under harbor's python"
else
    bad "cannot import fugue.agents (PYTHONPATH=$ROOT)"
fi

echo "== $PASS ok, $FAIL failed =="
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
