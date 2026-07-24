#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/research-brief.json <<'JSON'
{"question_id":"expense-policy-limit","answer":"USD 125 per attendee","source_document":"documents/expense-policy-rev-7.md","source_revision":"rev-7","brief":"The current client-meal reimbursement limit is USD 125 per attendee, including tax and gratuity."}
JSON
