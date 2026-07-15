#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
path = Path("/logs/artifacts/operator-answer.json")
value = json.loads(path.read_text())
assert value == {"recommended_candidate":"candidate-b","candidate_a_pass_rate":0.5,"candidate_b_pass_rate":1,"evidence_row_ids":["r1","r2","r3","r4"]}
PY
