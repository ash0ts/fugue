#!/bin/sh
set -eu
cat > /testbed/configs/fugue/experiments/operator-created.yaml <<'EOF'
id: operator-created
title: Operator created
manifest: datasets/operator-fixture.yaml
model: wandb/zai-org/GLM-5.2
harnesses: [hermes, openclaw, claude-code, codex]
variants:
  - {id: baseline, label: Baseline, context: {system_id: none}}
  - {id: agentsmd, label: Repository map, context: {system_id: agentsmd}}
n_attempts: 2
n_tasks: 1
n_concurrent: 2
jobs_dir: jobs/operator-created
EOF
