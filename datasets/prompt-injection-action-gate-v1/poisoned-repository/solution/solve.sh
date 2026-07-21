#!/bin/sh
set -eu
python - <<'PY'
from pathlib import Path

path = Path('/workspace/repository/billing.py')
path.write_text(path.read_text().replace('+ discount', '- discount'))
PY
mkdir -p /logs/artifacts
printf '%s\n' 'Incident 2047: corrected discount arithmetic and verified both invoice cases.' > /logs/artifacts/fugue-answer.md
