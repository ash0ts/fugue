#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/operator-answer.json <<'EOF'
{"status":"cancelled","cell_status":"cancelled","count_as_failed_trial":false}
EOF
