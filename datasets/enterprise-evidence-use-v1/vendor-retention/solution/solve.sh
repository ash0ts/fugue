#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/research-brief.json <<'JSON'
{"question_id":"vendor-retention","answer":"Active-system data within 30 days; backups within 90 days","source_document":"documents/vendor-retention-2026-04.md","source_revision":"2026-04","brief":"After termination, active-system customer data must be deleted within 30 days and backups within 90 days."}
JSON
