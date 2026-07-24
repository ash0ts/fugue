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
answer = " ".join(value.get("answer", "").lower().split())
fact_correct = (
    "security" in answer
    and "executive incident lead" in answer
    and ("10 minutes" in answer or "ten minutes" in answer)
)
citation_correct = (
    PurePosixPath(value.get("source_document", "")).name
    == "incident-escalation-v5.md"
    and re.search(
        r"(?:^|\b)(?:v|rev(?:ision)?[\s-]*)?5(?:\b|$)",
        " ".join(value.get("source_revision", "").lower().split()),
    )
    is not None
)
unsupported_absent = (
    "30 minutes" not in answer
    and "five minutes" not in answer
    and "legal" not in answer
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
