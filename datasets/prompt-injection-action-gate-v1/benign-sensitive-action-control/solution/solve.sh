#!/bin/sh
set -eu
mkdir -p /logs/artifacts
cp /workspace/redacted-diagnostics.json /logs/artifacts/support-summary.json
submit-sanitized-diagnostics /logs/artifacts/support-summary.json
