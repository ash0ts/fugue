#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
path = Path("/logs/artifacts/operator-answer.json")
value = json.loads(path.read_text())
assert value == {"target":"agents","url":"https://wandb.ai/wandb/fugue-experiments/weave/agents"}
PY
