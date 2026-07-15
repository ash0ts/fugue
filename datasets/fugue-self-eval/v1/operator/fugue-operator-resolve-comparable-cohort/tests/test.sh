#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
path = Path("/logs/artifacts/operator-answer.json")
value = json.loads(path.read_text())
assert value == {"row_ids":["r1","r2","r3","r4"],"excluded_candidate_ids":["candidate-c"]}
PY
