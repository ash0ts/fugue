#!/bin/sh
set -eu
mkdir -p /logs/artifacts
printf '%s\n' '{"ticket_id":"ticket-3187","service":"identity-api","root_cause_code":"jwks_cache_stale","root_cause_explanation":"All identity failures use a stale signing-key cache while the session control is healthy.","evidence_ids":["event-11","event-12","event-13"],"intervention":"Refresh the identity signing-key cache in one canary and alert on key age."}' > /logs/artifacts/support-diagnosis.json
