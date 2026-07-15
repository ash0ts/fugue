---
name: fugue-dev
description: Use when modifying Fugue experiments, candidate resolution, context or integration bindings, Harbor rendering, run lifecycle, evaluation publication, packaging, serving, curation, or their tests. Preserves Fugue identity, reproducibility, and evidence contracts.
---

# Fugue Development

Preserve these invariants across schema, execution, presentation, and tests.

## Identity and planning

- Resolve one immutable candidate and reuse it everywhere. Version its identity.
- Candidate identity contains behavior only: harness, model route, prompt,
  reviewed skills, context interface/delivery/configuration, typed integrations,
  and advanced agent settings. Presentation, scoring, run, preset, and trial
  state never affect it.
- Operational runtime, Harbor, gateway, tracing, and scheduling details belong
  only to the execution fingerprint.
- Require authored context delivery. Resolve typed capabilities for the exact
  workload, context, delivery, harness, and provider route before binding.

## Execution and evidence

- `OperatorService` owns resolve, immutable secret-free snapshot, preparation,
  render/plan, atomic input lock, running transition, and cell execution in that
  order. Nothing executes before the lock is durable.
- Setup is the only stateful preparation boundary. Preview and active trials do
  not install packages, download runtimes, start services, use the Docker
  socket, or mutate the production checkout.
- Record assigned and confirmed skill/context registration separately. A
  required treatment does not execute when registration failed. Invocation
  evidence stays explicit and may remain unavailable; assignment is not use.
- Normalize every logical outcome into one versioned prediction row. Keep raw
  retrieval and episode measurements separate. Reconcile every planned
  coordinate to terminal, not applicable, or explicitly cancelled.
- Publish idempotently by project, prediction identity, scorer version, and
  revision. Explicit revisions name what they supersede and why.
- Agent cells have one native conversation and one matching `invoke_agent`
  root. Direct diagnostics never synthesize Agent identity, conversations, or
  roots. Preserve unavailable usage instead of zero.

## Public contracts and release safety

- Keep experiment YAML strict: `skills`, required delivery, additive typed
  integrations, typed scorers, and no raw public MCP configuration. Generated
  evaluation is explicit and preview remains side-effect free.
- Package a candidate only when every applicable cell is terminal and at least
  one deterministic outcome passed. Terminal unscored cells are permitted but
  cannot satisfy the pass requirement; failures require explicit override.
- Package only clean tracked inputs with an explicit runtime allowlist. Reject
  drift, unsafe links or remotes, submodules, secrets, and unsupported serving
  contracts.
- Keep serving optional, isolated, and bounded. Admission, request size,
  readiness, cancellation, and exactly-once execution remain testable.
- Curator output stays inside its immutable declaration allowlist. It cannot
  modify code or tests, and it cannot bypass reviewed skill-source setup.

When a contract changes, update its parser, resolver, operator consumer,
snapshot/result representation, presentation, checked-in configurations, and
focused failure tests together.
