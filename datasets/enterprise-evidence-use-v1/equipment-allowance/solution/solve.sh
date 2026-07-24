#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/research-brief.json <<'JSON'
{"question_id":"equipment-allowance","answer":"USD 1,800 annually; Japan uses JPY 240,000","source_document":"documents/equipment-allowance-2026.md","source_revision":"2026","brief":"The annual allowance is USD 1,800, while employees in Japan use the JPY 240,000 regional cap."}
JSON
