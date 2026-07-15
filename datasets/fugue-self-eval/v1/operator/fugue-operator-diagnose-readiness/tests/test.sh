#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
path = Path("/logs/artifacts/operator-answer.json")
value = json.loads(path.read_text())
assert value == {"model":"wandb/zai-org/GLM-5.2","model_key_env":"WANDB_API_KEY","model_key_present":False,"trace_key_present":False,"bridge_ready":False}
PY
