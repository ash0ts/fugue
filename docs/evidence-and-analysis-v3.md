# Evidence and aligned analysis V3

Fugue writes new controlled comparisons as `ComparisonResultV3` and projects
them as `ExperimentViewV3`. Existing V1 and V2 artifacts remain readable and
are never rewritten in place.

V3 separates four questions that older result summaries could blur:

1. Did the candidate change the task outcome?
2. Did the assigned mechanism run?
3. Is the evidence chain complete and private-safe?
4. Does a separate release policy permit promotion?

## Source and result topology

`EvidenceTopologyV1` binds an immutable source destination and a distinct
result destination. Source locks are checked before and after execution.
Changing either destination, the source lock, a task, scorer, candidate,
runtime, or decision policy changes the preview digest and invalidates prior
approval.

Every attempt carries one stable identity through the planned cell, native
Agent conversation, Evaluation, prediction-and-score Call, prediction, Study
event, and exported result. W&B and Weave Call IDs remain distinct from OTel
trace and span IDs.

Evidence is reconciled only when all five required relationships resolve:

- Evaluation root;
- `Evaluation.predict_and_score`;
- prediction Call;
- native Agent root;
- Dataset version.

API identity resolution is authoritative. A malformed or signed-out browser
link is a navigation warning, not a substitute for API verification.

## Aligned analysis

`AlignedAnalysisV1` supports one reference arm, arbitrary treatment arms,
declared contrasts, aligned attempts, and task-stratified summaries. A
two-candidate comparison is one specialization; Skill×MCP, memory, and harness
studies use the same contract.

Dimensions have one declared role:

- `outcome`
- `mechanism`
- `safety_gate`
- `infrastructure`
- `efficiency`

Mechanism use cannot create an improvement. A pure outcome regression is
`regressed`; identical critical failures are `unchanged` with named blockers;
`mixed` requires both outcome improvements and regressions.

`TaskValidityV1` is independent of link integrity and reports `valid`,
`non_discriminating`, `drifted`, `invalid`, or `inconclusive`.

## Safe presentation

V3 result projections may include sanitized answer excerpts, deterministic
score explanations, actual queried scope, and the project identity reported by
the answer. Host-private expected values are never copied into the result or
Study event.

Study Console receives canonical paired cases and evidence links. It does not
reconstruct pairs from display labels or assume fixed arm names.

## Release decisions

`DecisionSummaryV1` is distinct from `BehavioralSummaryV1`. Local Harbor can
support a bounded behavioral finding, but it cannot certify a remote backend or
sign a package release. A governed release decision remains `hold` or
`ready_for_signoff` until every declared gate is satisfied and the immutable
result digest has the required human attestation.
