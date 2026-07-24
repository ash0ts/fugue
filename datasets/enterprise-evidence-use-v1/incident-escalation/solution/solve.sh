#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/research-brief.json <<'JSON'
{"question_id":"incident-escalation","answer":"Page Security and the Executive Incident Lead within 10 minutes","source_document":"documents/incident-escalation-v5.md","source_revision":"v5","brief":"A Severity 1 declaration requires paging Security and the Executive Incident Lead within 10 minutes."}
JSON
