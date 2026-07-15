#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/operator-answer.json <<'EOF'
{"cells":8,"applicable_cells":6,"not_applicable_cells":2,"unavailable_harness":"claude-code"}
EOF
