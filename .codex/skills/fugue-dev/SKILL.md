---
name: fugue-dev
description: Use when modifying Fugue experiments, provider routes, candidate resolution, context or integration bindings, managed runtimes, Harbor rendering, run lifecycle, Weave evidence, analysis, packaging, serving, curation, or their tests. Preserves Fugue's identity, reproducibility, isolation, and evaluation contracts.
---

# Fugue Development

Preserve these invariants across code, configuration, presentation, and tests.

## Canonical architecture

- Keep one path from strict `ExperimentSpec` authoring through a pure
  `ResolvedRunPlan`, immutable `ResolvedCandidate`, `RenderedJob` and
  `PlannedCell`, `RunSnapshotV1`, and normalized `PredictionRowV1`. Reuse each
  representation; do not rebuild identity, coordinates, or result meaning in
  a caller.
- Keep schema parsing in the experiment library, behavioral identity in the
  candidate resolver, planning and lifecycle in `OperatorService`, Harbor
  translation in the job renderer, and result normalization/publication in
  export. Rich and Textual consume those services rather than owning variants
  of the contracts.
- Resolve one immutable candidate and reuse it everywhere. Version its
  identity. Candidate identity contains behavior only: harness and version,
  model route, prompt digest, reviewed skills, context interface, delivery and
  configuration, typed integrations, and advanced agent settings.
- Presentation, scoring, evaluation, run, preset, cohort, label, and trial
  state never affect candidate identity. Runtime, Harbor, gateway, tracing,
  and scheduling details belong only to the execution fingerprint.
- Comparison example identity contains dataset, workload, and logical task.
  Trial index is a separate coordinate; never pair rows by labels, paths, list
  position, or evaluation display names.
- Keep one canonical V1 for each Fugue-owned persisted artifact. Reject
  unknown versions and incomplete canonical identity; do not add prerelease
  compatibility paths or reconstruct identity from presentation fields.

## Experiments and extensions

- Keep experiment YAML strict: `skills`, required `context.delivery`, additive
  typed integrations, typed scorers, and typed immutable `repository` task
  sources. Reject raw public MCP configuration and removed compatibility
  fields.
- Resolve typed capabilities for the exact workload, context system, delivery,
  harness, and provider route before binding. Preserve portable and native MCP
  as distinct interfaces; unsupported coordinates become `not_applicable`
  with an exact reason instead of silently changing treatment.
- Treat reviewed skills, context systems, and integrations as different
  concepts. Common and variant integrations add together; duplicate effective
  IDs fail. Context and integration declarations render internal MCP details.
- Keep research adapters experimental and outside default presets until their
  pinned runtimes pass the support matrix. A support label must reflect tested
  preparation, binding, registration, invocation, and evidence behavior.
- Keep generated evaluation explicit and exact-size. Preview may not call a
  model, inspect live context, merge hidden files, prepare assets, or write.

## Preparation and execution

- Compile preview, setup, and execution from the same pure resolved plan. A
  materialized plan must preserve its coordinates and identities.
- Keep preview and `setup --check` observational. `setup --prepare` is the only
  plan-resolved preparation boundary: it may fetch or build locked assets,
  context indexes, harness runtimes, and task images. Managed services have
  separate explicit start, status, and stop actions.
- Active trials verify prepared locks and may not install, download, build,
  pull, start services, access the Docker socket, or mutate the checkout.
  Require architecture-qualified runtime locks, a read-only task repository,
  and isolated writable adapter state.
- `OperatorService.execute_run` owns the snapshot-before-execution transaction:
  resolve the exact plan, persist the resolved experiment, materialize and
  verify jobs and cells, write host-only evaluation assets, atomically commit
  the secret-free input lock, transition to running, then start cells. Nothing
  executes before the lock is durable.
- Dataset verifiers use a pinned offline profile prepared into the task image.
  Validate base failure and gold success before a paid cohort; a verifier may
  not resolve packages or benchmark metadata during a trial.
- Keep answer-bearing evaluation data in a private host-only lock. Containers,
  Agent inputs, job configuration, snapshots, and traces receive only its
  digest; derived metrics may be published, but raw gold paths may not.
- Vector treatments fail closed when vector indexing or retrieval is absent.
  BM25 and vector modes are different candidates and cache entries; a vector
  label may never conceal lexical fallback.

## Harnesses, tools, and evidence

- Preserve the supported Agent harnesses and stable identities: Hermes,
  OpenClaw, Claude Code, and Codex. Harness execution returns structured
  benchmark, fatal, recoverable, provider, registration, and observability
  states; do not infer outcomes from log strings.
- Keep model routing and tool transport orthogonal. Native MCP preserves the
  upstream interface through Fugue's correlated gateway. Codex gets a new
  per-cell `CODEX_HOME` containing only its resolved route and allowlisted MCP
  servers; never inherit global Codex state or send MCP configuration through
  the model bridge.
- Record assigned and confirmed skill/context registration separately. A
  required treatment does not execute when registration failed. Invocation
  evidence stays explicit and may remain unavailable; assignment is not use.
- Agent cells open the Weave evaluation prediction before execution, then bind
  exactly one observed native conversation and `invoke_agent` root matching
  the evaluation call and canonical cell identity. Native integrations own
  nested chat and tool spans; do not create duplicate wrapper roots.
- Direct diagnostics remain ordinary Weave operations. Direct diagnostics
  never synthesize Agent identity, conversations, roots, or deep links.
  Preserve unavailable usage instead of zero.
- Normalize every logical Agent outcome into one versioned prediction row.
  Keep raw retrieval and episode measurements separate. Export through one
  ordered pipeline and reconcile every planned coordinate to terminal,
  `not_applicable`, or explicitly cancelled.
- Publish idempotently by project, prediction identity, scorer version, and
  revision. Explicit revisions name what they supersede and why; never merge
  evidence by display name.
- Build public reports only from reviewed, allowlisted canonical evidence
  snapshots. Never publish raw traces, Agent content, gold data, or secrets.
- Derive each harness's native wire protocol from the resolved model route and
  record its direct-or-bridge receipt. Bridged runs require the pinned image
  and exact locked configuration; model routing and MCP transport stay separate.
- Freeze discovery rankings in a versioned treatment-selection lock.
  Confirmatory cohorts reject treatments that disagree with that lock.
- Final-head live proof must use the exact code and runtime locks being
  qualified. Unit tests or evidence from an earlier head cannot satisfy Codex
  native MCP, reviewed-skill, vector, or Agent/evaluation-link release gates.

## Release safety

- Package a candidate only when every applicable cell is terminal and at least
  one deterministic outcome passed. Terminal unscored cells are permitted but
  cannot satisfy the pass requirement; failures require explicit override.
- Package only clean tracked inputs with an explicit runtime allowlist. Reject
  drift, unsafe links or remotes, submodules, secrets, and unsupported serving
  contracts.
- Keep serving optional, isolated, stateless, and bounded. Admission, request
  size, readiness, cancellation, unavailable usage, and exactly-once execution
  remain testable.
- Curator output stays inside its immutable declaration allowlist. It cannot
  modify code, tests, dependencies, presets, datasets, or workflows, and it
  cannot bypass reviewed skill-source setup.

When changing a contract, update its parser, resolver, operator consumer,
serialized representation, Rich/Textual presentation, checked-in
configuration, and focused failure tests together. Run compileall, Ruff, the
relevant focused suites, and the full suite before handoff.
