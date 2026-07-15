#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
path = Path("/logs/artifacts/operator-answer.json")
value = json.loads(path.read_text())
assert value == {"cells":8,"applicable_cells":6,"not_applicable_cells":2,"unavailable_harness":"claude-code"}
PY
