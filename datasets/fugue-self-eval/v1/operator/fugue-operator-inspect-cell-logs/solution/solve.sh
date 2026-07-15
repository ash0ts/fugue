#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/operator-answer.json <<'EOF'
{"cell_id":"cell-failed","failure_owner":"provider","cause":"W&B Inference quota exhausted"}
EOF
