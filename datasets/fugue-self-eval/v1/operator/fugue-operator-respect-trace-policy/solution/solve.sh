#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/operator-answer.json <<'EOF'
{"blocked_harness":"claude-code","action":"use full tracing or exclude claude-code"}
EOF
