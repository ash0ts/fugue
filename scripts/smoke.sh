#!/usr/bin/env bash
# Smoke run: 1 tiny task per harness against W&B Inference, all inside Harbor
# containers. Usage:
#   scripts/smoke.sh                 # all four harnesses
#   scripts/smoke.sh codex hermes    # subset
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"                     # repo root

[ -f "$ROOT/.env" ] && { set -a; source "$ROOT/.env"; set +a; }
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-fugue-local}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Weave tracing: agents fail fast without entity/project routing.
: "${WANDB_ENTITY:?set WANDB_ENTITY in .env (Weave tracing)}"
: "${WANDB_PROJECT:?set WANDB_PROJECT in .env (Weave tracing)}"
export WEAVE_PROJECT="${WEAVE_PROJECT:-$WANDB_ENTITY/$WANDB_PROJECT}"
echo "Weave project: $WEAVE_PROJECT"

MODEL="wandb/${FUGUE_MODEL:-zai-org/GLM-5.2}"
TASK="$ROOT/tasks/smoke/bridge-check"
JOBS="$ROOT/jobs/smoke"

declare -A IMPORTS=(
    [hermes]="fugue.agents.wandb_inference:WandbHermes"
    [openclaw]="fugue.agents.wandb_inference:WandbOpenClaw"
    [claude-code]="fugue.agents.wandb_inference:WandbClaudeCode"
    [codex]="fugue.agents.wandb_inference:WandbCodex"
)

# Keep the bridge up for claude-code.
docker compose -f "$ROOT/proxy/docker-compose.yaml" up -d >/dev/null

HARNESSES=("$@")
[ ${#HARNESSES[@]} -eq 0 ] && HARNESSES=(codex hermes openclaw claude-code)

SUMMARY=()
for h in "${HARNESSES[@]}"; do
    import="${IMPORTS[$h]:-}"
    if [ -z "$import" ]; then
        echo "unknown harness: $h (choices: ${!IMPORTS[*]})"; exit 2
    fi
    echo
    echo "===== smoke: $h ====="
    rm -rf "$JOBS/smoke-$h"   # fresh run; otherwise harbor resumes the old job
    harbor run \
        -p "$TASK" \
        -a "$import" \
        -m "$MODEL" \
        --job-name "smoke-$h" \
        -o "$JOBS" \
        -n 1
    result="$JOBS/smoke-$h/result.json"
    if [ -f "$result" ]; then
        mean=$(jq -r '.stats.evals | to_entries[0].value.metrics[0].mean // "n/a"' "$result" 2>/dev/null)
        errs=$(jq -r '.stats.n_errored_trials // "?"' "$result" 2>/dev/null)
        SUMMARY+=("$h: mean_reward=${mean:-n/a} errored_trials=${errs:-?}")
    else
        SUMMARY+=("$h: no result.json (see $JOBS/smoke-$h)")
    fi
done

echo
echo "== smoke summary =="
printf '%s\n' "${SUMMARY[@]}"
