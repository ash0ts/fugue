#!/bin/sh
set -eu
cd /testbed
mkdir -p /logs/artifacts
fugue runs run-fixture --export --out /logs/artifacts/export.jsonl --json
