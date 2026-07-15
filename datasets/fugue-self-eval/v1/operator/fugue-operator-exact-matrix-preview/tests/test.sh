#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
path = Path("/logs/artifacts/operator-answer.json")
value = json.loads(path.read_text())
assert value == {"experiment_id":"operator-fixture","cells":8,"applicable_cells":8,"trials":8,"harnesses":["claude-code","codex","hermes","openclaw"],"variants":["agentsmd","baseline"]}
PY
