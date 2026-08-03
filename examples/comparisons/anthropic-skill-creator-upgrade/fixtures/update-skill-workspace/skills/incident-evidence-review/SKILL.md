---
name: incident-evidence-review
description: Review a local incident evidence bundle and produce a bounded, cited operations recommendation. Use when an operator asks whether available incident evidence supports mitigation, escalation, or further collection.
---

# Incident Evidence Review

## Workflow

1. Inventory the supplied local evidence before drawing a conclusion.
2. Separate observations from hypotheses and cite the source path for each observation.
3. State missing evidence explicitly; never convert an unavailable measurement into zero.
4. Recommend one bounded next action with a success or stop condition.

## Validation

Confirm every factual claim has a local citation and that the recommendation does not exceed the inspected evidence.

## Failure handling

If a required file is absent or unreadable, report the gap and stop before making a release-wide or causal claim.
