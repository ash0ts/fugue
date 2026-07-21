#!/bin/sh
set -eu
mkdir -p /logs/artifacts
printf '%s\n' '{"service":"checkout-api","root_cause_code":"connection_pool_exhausted","root_cause_explanation":"Checkout requests spend most of their latency waiting for a database connection while the catalog control remains healthy.","evidence_ids":["trace-001","trace-002","trace-003"],"intervention":"Increase the checkout database pool in one canary and compare db wait time."}' > /logs/artifacts/anomaly-report.json
