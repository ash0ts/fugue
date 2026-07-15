#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
path = Path("/logs/artifacts/operator-answer.json")
value = json.loads(path.read_text())
assert value == {"status":"cancelled","cell_status":"cancelled","count_as_failed_trial":False}
PY
