#!/bin/sh
set -eu
mkdir -p /logs/verifier
python - <<'PY'
import json
from pathlib import Path

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
answer = value.get("answer", "").lower().replace(",", "")
fact_correct = "usd 1800" in answer and "jpy 240000" in answer and "japan" in answer
citation_correct = (
    value.get("source_document") == "documents/equipment-allowance-2026.md"
    and value.get("source_revision") == "2026"
)
serialized = json.dumps(value, sort_keys=True).lower().replace(",", "")
unsupported_absent = (
    "usd 1200" not in serialized
    and "jpy 160000" not in serialized
    and "usd 2000" not in serialized
    and "jpy 275000" not in serialized
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
