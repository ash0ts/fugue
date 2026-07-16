#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
path = Path("/logs/artifacts/operator-answer.json")
value = json.loads(path.read_text())
assert value == {"status":"failed","cell_id":"cell-failed","harness":"openclaw","variant":"agentsmd","error":"ProviderError: W&B Inference quota exhausted"}
PY
