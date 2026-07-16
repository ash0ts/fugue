---
name: Fugue Curator
description: Manual or explicitly enabled scheduled draft proposals for one high-confidence Fugue candidate
on:
  schedule:
    - cron: "0 14 * * 1"
  workflow_dispatch:
if: github.event_name == 'workflow_dispatch' || vars.FUGUE_CURATOR_ENABLED == 'true'
permissions:
  contents: read
  issues: read
  pull-requests: read
  actions: read
tracker-id: fugue-curator
engine:
  id: copilot
  copilot-sdk: true
imports:
  - uses: shared/fugue-curator-core.md
    with:
      mode: live
tools:
  edit:
safe-outputs:
  create-pull-request:
    title-prefix: "[fugue-curator] "
    draft: true
    max: 1
    base-branch: main
    allowed-base-branches: [main]
    allowed-branches: ["fugue-curator/*"]
    fallback-as-issue: false
    if-no-changes: warn
    max-patch-files: 12
    max-patch-size: 1024
    protected-files: blocked
    allowed-files:
      - "configs/fugue/skill-sources/**"
      - "configs/fugue/context-systems/**"
      - "configs/fugue/experiments/**"
  noop:
    report-as-issue: false
timeout-minutes: 90
max-turns: 160
---

# Live Fugue curator

Follow the imported policy exactly. End with one draft pull request or `noop`.
