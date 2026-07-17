---
name: fugue-dev
description: Use when modifying Fugue experiments, candidate resolution, context or integration bindings, Harbor rendering, run lifecycle, evaluation publication, packaging, serving, curation, or their tests. Preserves Fugue identity, reproducibility, and evidence contracts.
---

# Fugue Development

Preserve these invariants across code, configuration, presentation, and tests.

## Contracts and planning

- Resolve one immutable candidate and reuse it everywhere. Version its identity.
- Candidate identity contains behavior only: harness, model route, prompt,
  reviewed skills, context interface/delivery/configuration, typed integrations,
  and advanced agent settings. Presentation, scoring, run, preset, and trial
  state never affect it.
- Operational runtime, Harbor, gateway, tracing, and scheduling details belong
  only to the execution fingerprint.
- Keep one canonical V1 for each Fugue-owned persisted artifact. Reject unknown
  versions and incomplete identity; do not add prerelease compatibility paths.
- Require authored context delivery. Resolve typed capabilities for the exact
  workload, context, delivery, harness, and provider route before binding.
- Compile preview, setup, and execution from the same pure resolved plan. A
  materialized plan must preserve its coordinates and identities.

## Execution and evidence

- `OperatorService` owns resolve, immutable secret-free snapshot, preparation,
  render/plan, atomic input lock, running transition, and cell execution in that
  order. Nothing executes before the lock is durable.
- Setup is the only stateful preparation boundary. Setup may build and download
  locked assets; preview and active trials may not install, download, build,
  pull, start services, use the Docker socket, or mutate the checkout. Require
  architecture-qualified runtime locks.
- Dataset verifiers use a pinned offline profile prepared into the task image.
  Validate base failure and gold success before a paid cohort; a verifier may
  not resolve packages or benchmark metadata during a trial.
- Keep answer-bearing evaluation data in a private host-only lock. Containers,
  Agent inputs, job configuration, snapshots, and traces receive only its
  digest; derived metrics may be published, but raw gold paths may not.
- Vector treatments fail closed when vector indexing or retrieval is absent.
  BM25 and vector modes are different candidates and cache entries; a vector
  label may never conceal lexical fallback.
- Record assigned and confirmed skill/context registration separately. A
  required treatment does not execute when registration failed. Invocation
  evidence stays explicit and may remain unavailable; assignment is not use.
- Normalize every logical outcome into one versioned prediction row. Keep raw
  retrieval and episode measurements separate. Export through one ordered
  pipeline and reconcile every planned coordinate to terminal, not applicable,
  or explicitly cancelled.
- Freeze discovery rankings in a versioned treatment-selection lock.
  Confirmatory cohorts reject treatments that disagree with that lock.
- Publish idempotently by project, prediction identity, scorer version, and
  revision. Explicit revisions name what they supersede and why.
- Build public reports only from reviewed, allowlisted canonical evidence
  snapshots. Never publish raw traces, Agent content, gold data, or secrets.
- Agent cells have one native conversation and one matching `invoke_agent`
  root. Direct diagnostics never synthesize Agent identity, conversations, or
  roots. Preserve unavailable usage instead of zero.
- Derive each harness's native wire protocol from the resolved model route and
  record its direct-or-bridge receipt. Bridged runs require the pinned image
  and exact locked configuration; model routing and MCP transport stay separate.
- Codex native MCP, reviewed skills, and Agent/evaluation links require live
  proof from the exact release head and runtime locks. Unit tests or evidence
  from an earlier head cannot satisfy that release gate.

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

Change a contract only with its parser, resolver, operator consumer,
representation, presentation, checked-in configuration, and focused failure
tests in the same patch.
