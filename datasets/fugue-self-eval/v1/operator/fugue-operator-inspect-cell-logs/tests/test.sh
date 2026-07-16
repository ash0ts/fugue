#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
path = Path("/logs/artifacts/operator-answer.json")
value = json.loads(path.read_text())
assert value == {"cell_id":"cell-failed","failure_owner":"provider","cause":"W&B Inference quota exhausted"}
PY
