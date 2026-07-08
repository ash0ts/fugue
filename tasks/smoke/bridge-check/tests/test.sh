#!/usr/bin/env bash
# Deterministic verifier: reward 1 iff answer.txt contains exactly 7431.
set -u

REWARD_PATH="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_PATH")"

if [ -f /app/answer.txt ] && [ "$(tr -d '[:space:]' < /app/answer.txt)" = "7431" ]; then
    echo 1 > "$REWARD_PATH"
else
    echo 0 > "$REWARD_PATH"
fi
