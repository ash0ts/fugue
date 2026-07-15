#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
path = Path("/logs/artifacts/export.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line]
assert {row.get("cell_id") for row in rows} == {"cell-passed", "cell-failed"}
assert all(row.get("record_type") == "cell" for row in rows)
assert "sk-fugue-local" not in path.read_text()
PY
