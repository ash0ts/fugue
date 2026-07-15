#!/bin/sh
set -eu
cd /testbed
python - <<'PY'
import json
from pathlib import Path
from fugue.bench.library import get_experiment
from fugue.bench.operator import OperatorService
experiment = get_experiment("operator-created", Path("/testbed"))
assert experiment.n_attempts == 2
assert experiment.n_tasks == 1
assert [item.id for item in experiment.variants] == ["baseline", "agentsmd"]
preview = OperatorService(Path("/testbed")).preview_experiment(experiment)
assert preview.cells == 16 and preview.estimated_trials == 16
assert not list(Path("/testbed/.fugue/runtime").glob("*/job-configs"))
PY
