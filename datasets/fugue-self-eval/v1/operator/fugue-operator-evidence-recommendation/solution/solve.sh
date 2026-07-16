#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/operator-answer.json <<'EOF'
{"recommended_candidate":"candidate-b","candidate_a_pass_rate":0.5,"candidate_b_pass_rate":1,"evidence_row_ids":["r1","r2","r3","r4"]}
EOF
