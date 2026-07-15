#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/operator-answer.json <<'EOF'
{"target":"agents","url":"https://wandb.ai/wandb/fugue-experiments/weave/agents"}
EOF
