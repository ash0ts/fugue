#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/operator-answer.json <<'EOF'
{"row_ids":["r1","r2","r3","r4"],"excluded_candidate_ids":["candidate-c"]}
EOF
