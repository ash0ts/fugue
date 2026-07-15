#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/operator-answer.json <<'EOF'
{"experiment_id":"operator-fixture","cells":8,"applicable_cells":8,"trials":8,"harnesses":["claude-code","codex","hermes","openclaw"],"variants":["agentsmd","baseline"]}
EOF
