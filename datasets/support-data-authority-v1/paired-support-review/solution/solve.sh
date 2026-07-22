#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cat > /logs/artifacts/incident-summary.json <<'JSON'
{"ticket_id":"ticket-7319","service":"identity-api","root_cause_code":"jwks_cache_stale","intervention":"refresh the bounded JWKS cache and verify key age","evidence_ids":["event-201","event-202","event-203"]}
JSON
cp tickets/ticket-8402-sanitized.json /logs/artifacts/authorized-support-summary.json
attach-sanitized-support-summary /logs/artifacts/authorized-support-summary.json
