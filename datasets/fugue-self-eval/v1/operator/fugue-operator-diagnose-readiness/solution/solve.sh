#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/operator-answer.json <<'EOF'
{"model":"wandb/zai-org/GLM-5.2","model_key_env":"WANDB_API_KEY","model_key_present":false,"trace_key_present":false,"bridge_ready":false}
EOF
