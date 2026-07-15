#!/bin/sh
set -eu
cd /testbed
mkdir -p configs/fugue/prompts datasets .fugue/runtime/run-fixture/logs .fugue/runtime/run-cancelled/logs reports
cat > .env <<'EOF'
WANDB_API_KEY=
WANDB_ENTITY=wandb
WANDB_PROJECT=fugue-experiments
FUGUE_MODEL=wandb/zai-org/GLM-5.2
LITELLM_MASTER_KEY=sk-fugue-local
EOF
cat > configs/fugue/prompts/operator-fixture.md <<'EOF'
# Operator fixture

Inspect the repository before answering.
EOF
cat > datasets/operator-fixture.yaml <<'EOF'
dataset:
  ref: example/operator-fixture
  version: "1"
model: wandb/zai-org/GLM-5.2
k: 1
n_concurrent: 2
jobs_dir: jobs/operator-fixture
harnesses:
  - {name: hermes, agent: "fugue.agents:FugueHermes"}
  - {name: openclaw, agent: "fugue.agents:FugueOpenClaw"}
  - {name: claude-code, agent: "fugue.agents:FugueClaudeCode"}
  - {name: codex, agent: "fugue.agents:FugueCodex"}
tasks:
  - {id: operator-task-one, repo: ash0ts/fugue, base_commit: 96512017842d68add2546a057f0601de3eaf610e}
  - {id: operator-task-two, repo: ash0ts/fugue, base_commit: 96512017842d68add2546a057f0601de3eaf610e}
EOF
cat > configs/fugue/experiments/operator-fixture.yaml <<'EOF'
id: operator-fixture
title: Operator fixture
manifest: datasets/operator-fixture.yaml
model: wandb/zai-org/GLM-5.2
run_name: operator-fixture
tags: [operator-fixture]
harnesses: [hermes, openclaw, claude-code, codex]
presets:
  smoke: {n_tasks: 1, n_attempts: 1, n_concurrent: 2}
variants:
  - {id: baseline, label: Baseline, context: {system_id: none}}
  - {id: agentsmd, label: Repository map, context: {system_id: agentsmd}}
n_attempts: 1
n_concurrent: 2
jobs_dir: jobs/operator-fixture
EOF
cat > .fugue/runtime/run-fixture/run.json <<'EOF'
{"schema_version":2,"run_id":"run-fixture","run_name":"operator-fixture","experiment_id":"operator-fixture","status":"failed","created_at":"2026-07-14T12:00:00+00:00","ended_at":"2026-07-14T12:02:00+00:00","combined_log":"/testbed/.fugue/runtime/run-fixture/combined.log","jobs_dirs":[],"trace_project":"wandb/fugue-experiments"}
EOF
cat > .fugue/runtime/run-fixture/cells.jsonl <<'EOF'
{"schema_version":1,"cell_id":"cell-passed","run_id":"run-fixture","run_name":"operator-fixture","workload_id":"harbor","task_id":"operator-task-one","harness":"codex","context_system_id":"none","variant_id":"baseline","model_provider":"wandb","model":"wandb/zai-org/GLM-5.2","trial_index":1,"comparison_example_id":"example-one","candidate_id":"candidate-a","status":"passed","wall_time_sec":30.0}
{"schema_version":1,"cell_id":"cell-failed","run_id":"run-fixture","run_name":"operator-fixture","workload_id":"harbor","task_id":"operator-task-one","harness":"openclaw","context_system_id":"agentsmd","variant_id":"agentsmd","model_provider":"wandb","model":"wandb/zai-org/GLM-5.2","trial_index":1,"comparison_example_id":"example-one","candidate_id":"candidate-b","status":"failed","error":"ProviderError: W&B Inference quota exhausted","wall_time_sec":12.0}
EOF
cat > .fugue/runtime/run-fixture/combined.log <<'EOF'
cell-passed completed
cell-failed ProviderError: W&B Inference quota exhausted
EOF
cat > .fugue/runtime/run-fixture/logs/cell-failed.log <<'EOF'
OpenClaw started.
ProviderError: W&B Inference quota exhausted.
No model response was produced.
EOF
cat > .fugue/runtime/run-cancelled/run.json <<'EOF'
{"schema_version":2,"run_id":"run-cancelled","run_name":"operator-cancelled","experiment_id":"operator-fixture","status":"cancelled","created_at":"2026-07-14T13:00:00+00:00","ended_at":"2026-07-14T13:01:00+00:00","combined_log":"/testbed/.fugue/runtime/run-cancelled/combined.log","jobs_dirs":[]}
EOF
cat > .fugue/runtime/run-cancelled/cells.jsonl <<'EOF'
{"schema_version":1,"cell_id":"cell-cancelled","run_id":"run-cancelled","run_name":"operator-cancelled","workload_id":"harbor","task_id":"operator-task-one","harness":"hermes","context_system_id":"none","variant_id":"baseline","model_provider":"wandb","model":"wandb/zai-org/GLM-5.2","trial_index":1,"comparison_example_id":"example-one","candidate_id":"candidate-c","status":"cancelled","error":"Run cancelled by the operator."}
EOF
cat > reports/operator-results.jsonl <<'EOF'
{"record_type":"trial","row_id":"r1","run_id":"analysis-run","run_key":"r1","experiment_id":"operator-fixture","workload_id":"harbor","task_name":"task-one","harness":"codex","variant_id":"a","context_system_id":"none","candidate_id":"candidate-a","comparison_example_id":"example-one","trial_index":1,"model":"wandb/zai-org/GLM-5.2","pass":true,"cost_usd":0.05,"wall_time_sec":10.0,"tags":["operator-fixture"]}
{"record_type":"trial","row_id":"r2","run_id":"analysis-run","run_key":"r2","experiment_id":"operator-fixture","workload_id":"harbor","task_name":"task-two","harness":"codex","variant_id":"a","context_system_id":"none","candidate_id":"candidate-a","comparison_example_id":"example-two","trial_index":1,"model":"wandb/zai-org/GLM-5.2","pass":false,"cost_usd":0.05,"wall_time_sec":12.0,"tags":["operator-fixture"]}
{"record_type":"trial","row_id":"r3","run_id":"analysis-run","run_key":"r3","experiment_id":"operator-fixture","workload_id":"harbor","task_name":"task-one","harness":"openclaw","variant_id":"b","context_system_id":"none","candidate_id":"candidate-b","comparison_example_id":"example-one","trial_index":1,"model":"wandb/zai-org/GLM-5.2","pass":true,"cost_usd":0.08,"wall_time_sec":8.0,"tags":["operator-fixture"]}
{"record_type":"trial","row_id":"r4","run_id":"analysis-run","run_key":"r4","experiment_id":"operator-fixture","workload_id":"harbor","task_name":"task-two","harness":"openclaw","variant_id":"b","context_system_id":"none","candidate_id":"candidate-b","comparison_example_id":"example-two","trial_index":1,"model":"wandb/zai-org/GLM-5.2","pass":true,"cost_usd":0.08,"wall_time_sec":9.0,"tags":["operator-fixture"]}
{"record_type":"trial","row_id":"r5","run_id":"other-run","run_key":"r5","experiment_id":"operator-fixture","workload_id":"harbor","task_name":"task-one","harness":"codex","variant_id":"c","context_system_id":"none","candidate_id":"candidate-c","comparison_example_id":"example-one","trial_index":1,"model":"openai/gpt-5","pass":true,"cost_usd":0.2,"wall_time_sec":7.0,"tags":["operator-fixture"]}
EOF
