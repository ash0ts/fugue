#!/bin/sh
set -eu
answer=/logs/artifacts/fugue-answer.txt
mkdir -p /logs/verifier
reward=0.0
if [ -f "$answer" ] && [ "$(tr -d '\r\n' < "$answer")" = 'src/relay/amber_lantern.py' ]; then
  reward=1.0
fi
printf '{"reward": %s, "path_resolution": %s}\n' "$reward" "$reward" > /logs/verifier/reward.json
