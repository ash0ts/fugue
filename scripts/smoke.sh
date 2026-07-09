#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

[ -f "$ROOT/.env" ] && { set -a; source "$ROOT/.env"; set +a; }
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-fugue-local}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${FUGUE_MODEL:-wandb/zai-org/GLM-5.2}"
HARNESSES=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            HARNESSES+=("$1")
            shift
            ;;
    esac
done

: "${WANDB_ENTITY:?set WANDB_ENTITY in .env (Weave tracing)}"
: "${WANDB_PROJECT:?set WANDB_PROJECT in .env (Weave tracing)}"
export WEAVE_PROJECT="${WEAVE_PROJECT:-$WANDB_ENTITY/$WANDB_PROJECT}"
export FUGUE_MODEL="$MODEL"

TASK="$ROOT/tasks/smoke/bridge-check"
JOBS="$ROOT/jobs/smoke"

declare -A IMPORTS=(
    [hermes]="fugue.agents:FugueHermes"
    [openclaw]="fugue.agents:FugueOpenClaw"
    [claude-code]="fugue.agents:FugueClaudeCode"
    [codex]="fugue.agents:FugueCodex"
)

python -m fugue.bench.cli bridge up --repo-root "$ROOT" --model "$MODEL"

[ ${#HARNESSES[@]} -eq 0 ] && HARNESSES=(codex hermes openclaw claude-code)

SUMMARY=()
for h in "${HARNESSES[@]}"; do
    import="${IMPORTS[$h]:-}"
    if [ -z "$import" ]; then
        echo "unknown harness: $h (choices: ${!IMPORTS[*]})"
        exit 2
    fi
    echo
    echo "===== smoke: $h / $MODEL ====="
    rm -rf "$JOBS/smoke-$h"
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
