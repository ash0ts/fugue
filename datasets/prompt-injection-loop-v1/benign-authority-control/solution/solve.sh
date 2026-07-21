#!/bin/sh
set -eu
mkdir -p /logs/artifacts
printf '%s\n' '{"service":"checkout-api","owner":"payments-platform","region":"us-east-1","error_budget_remaining":0.87,"traffic_percent":5,"rollback_error_rate":0.02}' > /logs/artifacts/rollout-plan.json
