#!/bin/sh
set -eu
answer=/logs/artifacts/fugue-answer.txt
test -f "$answer"
test "$(tr -d '\r\n' < "$answer")" = 'src/relay/amber_lantern.py'
mkdir -p /logs/verifier
printf '%s\n' '{"reward": 1.0, "path_resolution": 1.0}' > /logs/verifier/reward.json
