---
name: fugue-dev
description: Use when modifying Fugue experiments, candidate resolution, Harbor rendering, context or integration bindings, operator lifecycle, evaluation, packaging, serving, curation, or their tests. Preserves Fugue 0.1 identity, reproducibility, and safety contracts.
---

# Fugue Development

Preserve these invariants across schema, implementation, UI, and tests.

## Candidate identity

- Resolve one canonical, immutable candidate representation and reuse it for
  rendering, snapshots, results, presets, export, and packaging.
- Version candidate identity. Hash only behavior-affecting harness, model route,
  prompt, reviewed skills, context delivery/configuration, typed integrations,
  and advanced agent settings.
- Keep experiment and variant names, labels, run names, preset names, scoring or
  judge configuration, UI state, and trial index out of candidate identity.
- Put runtime, Harbor, concurrency, and instrumentation policy in a separate
  execution fingerprint.
- Identify comparison examples by dataset, workload, and task. Trial index is a
  separate cell coordinate.

## Experiment and evaluation contracts

- Experiment YAML is strict. Use `skills`; do not add legacy aliases.
- Integrations are typed and additive across experiment and variant scopes.
  Reject duplicate effective IDs. Never expose raw public MCP configuration.
- Context definitions declare supported deliveries and serving deliveries.
  Pass delivery into binding; portable delivery must not inject native MCP, and
  native MCP must preserve the upstream interface.
- Scorer selections are typed. Rubric scoring requires an explicit judge model.
- Evaluation generation is explicit, exact about suite/workload/size, and
  side-effect free during preview. Keep deterministic outcomes, judge scores,
  and judge errors separate.

## Run lifecycle

- `OperatorService` owns one run transaction: resolve, snapshot the experiment,
  prepare context, render and plan, atomically write the immutable secret-free
  input lock, transition to running, then execute cells.
- A failure before the running transition records a failed starting run and
  executes no cell. CLI and TUI delegate to the operator rather than duplicating
  orchestration.
- Group results by candidate. Display a unique short prefix, retain full IDs in
  JSON and snapshots, and reject ambiguous input prefixes.
- A candidate is packageable only when all planned applicable cells are
  terminal and at least one passed. Failed cells require explicit override and
  confirmation; unrelated run failure does not block a complete candidate.

## Packaging and serving

- Package only from clean production and Fugue checkouts, using a tracked
  runtime allowlist. Reject lock drift, dirty source, submodules, escaping
  symlinks, credential-bearing remotes, secrets, and unsupported integrations.
- Package context only when its selected delivery declares tested serving
  support. Do not silently convert delivery during packaging.
- Keep serving optional and outside the operator path. Isolate each execution,
  bound request size and admission, terminate the process group on cancellation,
  and remove request state.
- Tracing is best effort but candidate execution is exactly once. Preserve
  unavailable usage instead of synthesizing zero.

## Curator boundary

- Curator proposals may change only declared skill sources, context systems,
  and controlled experiments. They may not change code, tests, workflows,
  dependencies, datasets, presets, README, or vendored skill content.
- A skill proposal adds a pinned source and experiment. Human review through
  the reviewed skill-source setup remains mandatory; automation cannot vendor
  or approve unreviewed content.

## Validation

When changing a contract, update its parser, resolver, operator consumers,
presentation, checked-in configurations, and focused tests together. Verify
preview side-effect freedom, snapshot-before-cell ordering, identity boundaries,
and failure behavior—not only the happy path.
