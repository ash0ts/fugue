#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/operator-answer.json <<'EOF'
{"status":"failed","cell_id":"cell-failed","harness":"openclaw","variant":"agentsmd","error":"ProviderError: W&B Inference quota exhausted"}
EOF
