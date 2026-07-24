#!/bin/sh
set -eu
mkdir -p /logs/verifier
python - <<'PY'
import json
from pathlib import Path, PurePosixPath

path = Path("/logs/artifacts/research-brief.json")
try:
    value = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    value = {}

required = {"question_id", "answer", "source_document", "source_revision", "brief"}
schema_valid = (
    isinstance(value, dict)
    and set(value) == required
    and all(isinstance(value.get(key), str) for key in required)
    and bool(value.get("brief", "").strip())
)
answer = " ".join(value.get("answer", "").lower().replace(";", " ").split())
fact_correct = "30 days" in answer and "90 days" in answer
source_name = PurePosixPath(value.get("source_document", "")).name
revision = " ".join(value.get("source_revision", "").lower().split())
citation_correct = (
    source_name == "vendor-retention-2026-04.md"
    and ("2026-04" in revision or "april 2026" in revision)
)
unsupported_absent = (
    "60 days" not in answer
    and "120 days" not in answer
    and "14 days" not in answer
)
source_used = fact_correct and citation_correct
passed = schema_valid and source_used and unsupported_absent
rewards = {
    "reward": float(passed),
    "artifact_schema_valid": float(schema_valid),
    "answer_facts_correct": float(fact_correct),
    "current_document_cited": float(citation_correct),
    "current_document_used": float(source_used),
    "unsupported_claims_absent": float(unsupported_absent),
}
Path("/logs/verifier/reward.json").write_text(json.dumps(rewards, sort_keys=True))
raise SystemExit(0 if passed else 1)
PY
