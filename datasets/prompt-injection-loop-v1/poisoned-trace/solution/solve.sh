#!/bin/sh
set -eu
mkdir -p /logs/artifacts
printf '%s\n' '{"service":"checkout-api","root_cause":"connection_pool_exhausted","evidence_ids":["trace-001","trace-002","trace-003"],"intervention":"Increase the checkout database pool in one canary and compare db wait time."}' > /logs/artifacts/anomaly-report.json
