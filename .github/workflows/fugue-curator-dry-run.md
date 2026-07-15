---
name: Fugue Curator Dry Run
description: Manual read-only discovery and deterministic candidate evaluation
on:
  workflow_dispatch:
permissions:
  contents: read
  issues: read
  pull-requests: read
  actions: read
tracker-id: fugue-curator-dry-run
engine:
  id: copilot
  copilot-sdk: true
imports:
  - uses: shared/fugue-curator-core.md
    with:
      mode: dry_run
safe-outputs:
  noop:
    report-as-issue: false
timeout-minutes: 90
max-turns: 160
---

# Read-only Fugue curator dry run

Follow the imported policy exactly. Do not edit files. End with `noop`.
