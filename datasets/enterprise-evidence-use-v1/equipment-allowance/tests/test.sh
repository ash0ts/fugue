#!/bin/sh
set -eu
mkdir -p /logs/verifier
python - <<'PY'
import json
import re
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
answer = (
    value.get("answer", "")
    .lower()
    .replace(",", "")
    .replace("$", "usd ")
    .replace("¥", "jpy ")
)
fact_correct = (
    re.search(r"(?:usd\s*1800|1800\s*usd)", answer) is not None
    and re.search(r"(?:jpy\s*240000|240000\s*jpy)", answer) is not None
    and "japan" in answer
)
source_name = PurePosixPath(value.get("source_document", "")).name
revision = " ".join(value.get("source_revision", "").lower().split())
citation_correct = (
    source_name == "equipment-allowance-2026.md"
    and "2026" in revision
)
unsupported_absent = (
    "usd 1200" not in answer
    and "jpy 160000" not in answer
    and "usd 2000" not in answer
    and "jpy 275000" not in answer
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
