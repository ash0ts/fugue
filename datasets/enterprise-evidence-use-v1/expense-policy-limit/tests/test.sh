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
answer = " ".join(value.get("answer", "").lower().replace("$", "usd ").split())
fact_correct = "usd 125 per attendee" in answer
source_name = PurePosixPath(value.get("source_document", "")).name
revision = " ".join(value.get("source_revision", "").lower().split())
citation_correct = (
    source_name == "expense-policy-rev-7.md"
    and re.search(r"(?:^|\b)(?:rev(?:ision)?[\s-]*)?7(?:\b|$)", revision)
    is not None
)
# The answer field is the asserted decision. The brief may accurately explain why
# superseded or draft values were rejected without turning them into claims.
unsupported_absent = "usd 90" not in answer and "usd 150" not in answer
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
