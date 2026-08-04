# Community Skill upgrade campaign

## Current executable canaries

The current execution authority is
`canary-execution-policy-v2.json`: three fresh four-cell Studies, one per
repository. They use new Study identities because every earlier canary is
audit history produced before the final artifact, cancellation, scorer, and
trace-integrity repairs. Each Study has its own exact preview and approval with
a $34 ceiling. The 12 new Agent cells therefore have a $102 ceiling; the
already completed judge diagnostic accounts for the remaining $8 of the
original $110 campaign allocation and must not be rerun.

The same-family Sonnet judge is advisory. Its frozen 48-case synthetic result
passed the authored-reference thresholds and its V2 sanitizer compatibility
receipt is valid, but human review remains pending. Deterministic task and
safety gates are authoritative, and no result may claim a human-qualified
judge outcome.

The active specs are:

- `../superpowers-writing-plans-upgrade/comparison-v5.yaml`
- `../anthropic-skill-creator-upgrade/comparison-v2.yaml`
- `../vercel-react-best-practices-upgrade/comparison-v2.yaml`

Each run stops after its first cell unless project routing, the five-link Weave
chain, privacy, cost accounting, and Harbor cleanup reconcile.

## Deferred confirmatory campaign V1

The original four-cell canaries are exploratory audit history. They are not
pooled with the confirmatory campaign and they do not support a general Skill
upgrade claim. The frozen confirmatory design is in
`conference-preregistration.json`; the four operator stages are enumerated in
`conference-campaign-manifest.json`.

The deferred Superpowers confirmatory design is V5. V3 is retained as invalid audit
history after a bounded Agent timeout was misclassified as infrastructure
failure. V4 is also invalid audit history: its checkpoint exposed silent
primary-artifact truncation and an over-broad path classifier before the run
was stopped. V5 preregisters a complete 192-cell restart with the same behavior inputs,
an amended measurement contract, and a fresh project, source lock, preview,
and approval. V1-V4 rows are excluded from every effect estimate, and the V5
report must include the preregistered conservative sensitivity for holdout
tasks exposed by earlier attempts.

Each repository Study contains 24 tasks (8 scorer-development tasks and 16
untouched holdouts), two exact Skill revisions, and four attempts per task and
arm: 192 cells per Study and 576 Agent cells in total. The inference unit is a
task. Repeated attempts estimate within-task variability and are not counted as
independently sampled tasks.

### Confirmatory cost and approval boundary

The current descriptive measurement-development limits are frozen in
`confirmatory-budget-policy-v2.json` for exact Superpowers V6, Anthropic V2,
and Vercel V2 specs. They are intentionally much larger than
the original canary budget: each Study reserves 192 × ($8.40 Agent + $0.10
advisory judge) = **$1,632**. The independent hard caps are $1,700 for
Superpowers and $1,640 each for Anthropic and Vercel, for 576 cells, $4,896 of
estimated reserve, and at most $4,980 across three separately approved runs.

The checked-in policy is not approval, and no confirmatory execution is
currently authorized. Each Study would require a fresh receipt for
its exact preview and cap. The V1 budget policy, historical
`campaign-manifest.json`, and
`budget-ledger.json` describe the original four-cell canaries and their $110
ceiling only; neither those files nor any approval derived from them authorizes
one of these 192-cell confirmatory Studies. The current judge diagnostic still
awaits independent human review and remains advisory; its exact V2 artifact is
bound without contributing new cells or spend to this policy.

The governed comparison contract is intentionally two-arm. V1 estimates the
exact candidate-minus-baseline revision effect. It does not estimate absolute
benefit relative to no Skill, and the report must retain that limitation. A
generic three-arm `fugue run` is not a substitute because it does not have the
same comparison approval and result contracts.

The blinded Sonnet judge is descriptive and secondary. It shares a model family
with the Agent and remains `pending_human_review`; deterministic host-side
verification is the primary endpoint. A judge-qualified claim requires the
two-reviewer protocol in the preregistration.

Before execution, prepare every immutable source/task archive and independently
prove each executable verifier's base-fail/gold-pass contract. Then, for each
Study, run `check`, `compare --prepare`, and `compare --preview` twice. The two
preview digests must match. Freeze the blinded trace sample from that exact
preview before approving it:

```bash
uv run python \
  examples/comparisons/community-skill-upgrades/freeze_trace_audit.py \
  /tmp/STUDY.preview.json \
  --fraction 0.25 \
  --preregistration /path/to/STUDY.preregistration.json \
  --output /tmp/STUDY.trace-audit-selection.json
```

Approval always binds the exact final preview and its finite worst-case ceiling,
even when campaign budget is not a limiting factor:

```bash
PREVIEW=/tmp/STUDY.preview.json
APPROVAL=/tmp/STUDY.approval.json
PREVIEW_DIGEST=$(jq -er .preview_digest "$PREVIEW")
MAX_CELLS=$(jq -er .readiness.estimated_cells "$PREVIEW")
MAX_USD=$(jq -er .readiness.estimated_cost_usd "$PREVIEW")
SELECTION_DIGEST=$(jq -er .selection_digest /tmp/STUDY.trace-audit-selection.json)

uv run fugue approve "$PREVIEW_DIGEST" \
  --max-cells "$MAX_CELLS" \
  --max-usd "$MAX_USD" \
  --bind "trace_audit_selection=$SELECTION_DIGEST" \
  --approved-by operator \
  --operation-id "approve-STUDY-${PREVIEW_DIGEST}" \
  --expires-in 86400 > "$APPROVAL"
```

Regenerate the preview immediately before execution and reject any digest
change. The approval must bind the exact deterministic trace-audit selection;
the confirmatory analyzer rejects an unbound or substituted sample. Run only
with the one-use approval receipt, exact `.env` path, and no OpenAI credential.
The first aligned pair is an automatic fail-closed evidence checkpoint. A
failed checkpoint cancels the remaining cells.

After execution, reload the canonical result and attempt rows, run
`analyze_confirmatory.py`, and manually review the study-specific frozen paired
trace sample (25% for Superpowers; 10% for Anthropic and Vercel) plus every
discordant or critical pair. A report is incomplete until the five Weave
relationships, task/runtime locks, host verifier, privacy receipt, and Harbor
cleanup are checked for every audited attempt.

This directory binds three independent, task-specific Skill upgrade Studies.
It does not create an alternate runner or a campaign-wide spend approval: each
lane remains an ordinary Fugue comparison with its own preview, finite cap,
approval, execution identity, evidence project, and result.

The checked-in calibration is intentionally **pending human review**. Judge
output therefore remains advisory and cannot support a judge-qualified claim.
The bounded synthetic-gold calibration is nevertheless a mandatory execution
gate: no Agent cell or paid judge call may start until its exact approved result
passes the locked thresholds. It does not replace two independent reviews and
adjudication.

The gate freezes both representations of its private case cohort: a canonical
parsed-content digest and the repository-relative case artifact path plus its
exact byte SHA-256. It also binds the paid runner bytes and the transmitted
JSON response schema. Those bindings, the rubric, model route, request bounds,
and budget are part of the preview digest. Moving or changing any bound input
therefore invalidates the preview and every approval derived from it.

The four original Superpowers holdout rows are retired because a stopped paid
run exposed their outcomes before the rubric repair. The frozen cohort now uses
an unseen provider-neutral task-importer planning scenario with two acceptable
and two defective examples, including one critical false-pass case. Outcomes
from the retired rows are audit history only and cannot support calibration.

`calibration-prior-runs.json` is the immutable conservative accounting ledger
for the seven earlier paid or potentially paid runs. It records 79 attempted
requests, exact archived artifact hashes where evidence survived, explicit
limitations where receipts did not survive, and a total reserve of $7.094273.
Its byte SHA-256 and canonical JSON digest are bound into the campaign manifest,
calibration report, preview, and approval identity. The next-run ceiling is
derived as `$8.000000 - $7.094273 = $0.905727`; it is not a mutable scalar.
The legacy V1 result field named `prior_failed_requests` remains readable and
projects the prior-run count of seven until those artifacts migrate.

## Offline validation

```bash
uv run python \
  examples/comparisons/community-skill-upgrades/validate_campaign.py
```

The command validates the historical canary fixtures: strict schemas, exact
revisions and Study destinations, the 36/12 balanced calibration split, the
non-approved $110 canary ceiling, and the claim-free scientific report
template. It does not authorize or price the confirmatory cohort. The focused
confirmatory contract test independently reconciles
`confirmatory-budget-policy-v2.json` with all three active specs while retaining
the V1 policy as immutable audit history. Validation
also hashes the exact calibration case bytes and rejects drift before emitting
judge input or making a provider request. It exits successfully when the
fixture set is internally valid while reporting `pending_human_review`; the
three previews remain approvable, but execution requires their new exact
approval receipts.

To produce a judge-only input file that omits authored references and reviews:

```bash
uv run python \
  examples/comparisons/community-skill-upgrades/validate_campaign.py \
  --emit-judge-input /tmp/community-skill-judge-input.jsonl
```

After actual judge outputs and two independent human reviews are recorded in
the case file, regenerate the calibration report explicitly:

```bash
uv run python \
  examples/comparisons/community-skill-upgrades/validate_campaign.py \
  --write-report \
  examples/comparisons/community-skill-upgrades/judge-calibration.json
```

That report passes only when overall, calibration-split, and holdout-split
balanced accuracy are each >= 0.85, with zero critical false passes. TPR and
TNR remain visible as diagnostic metrics rather than independent gates. Each
lane then needs a new immutable preview and its own
operator approval. The historical budget ledger is a canary ceiling
configuration, not approval and not a confirmatory spending authority.
For this acceptable-versus-defective calibration, `adequate` is the minimum
acceptable qualitative label: its limitations stay visible, and deterministic
task and safety gates remain authoritative. `weak` and `unusable` are defective;
`strong` and `exceptional` exceed the minimum rather than defining it.

The mandatory advisory-judge gate runs the locked Sonnet judge over all 48
synthetic authored-reference cases:

```bash
uv run python \
  examples/comparisons/community-skill-upgrades/run_synthetic_calibration.py \
  --dry-run
```

The dry run makes no provider request and writes no result. After the separate
calibration-stage approval, first emit its immutable preview:

```bash
uv run python \
  examples/comparisons/community-skill-upgrades/run_synthetic_calibration.py \
  --preview > /tmp/community-skill-calibration-preview.json

uv run fugue approve \
  "$(jq -r .preview_digest /tmp/community-skill-calibration-preview.json)" \
  --max-cells 48 --max-usd 0.905727 --approved-by operator \
  > /tmp/community-skill-calibration-approval.json
```

Then execute the exact approved diagnostic:

```bash
uv run python \
  examples/comparisons/community-skill-upgrades/run_synthetic_calibration.py \
  --env-file /Users/ashah/Documents/common_tools/.env \
  --output .fugue/runtime/community-skill-upgrades/judge-calibration.result.json \
  --approval "$(jq -r .approval_digest \
    /tmp/community-skill-calibration-approval.json)" \
  --confirm-paid-synthetic-calibration
```

It performs up to 48 sequential, non-retrying requests and incrementally writes a
separate `synthetic_gold_diagnostic`. Anthropic is constrained to the exact
JSON shape. Because Anthropic structured outputs do not enforce string-length
constraints, the prompt requests 500-character explanations and the transport
plus local validator independently enforce a 16,000-character response
envelope. Judge reasoning is explicitly disabled so the bounded output budget
is reserved for the schema-constrained result; a `max_tokens` stop remains a
named fail-closed protocol error. Public Study explanations remain redacted and
capped at 500. A malformed or failed response produces a body-free failure
receipt without overwriting completed rows. It never edits
`judge-calibration.json`, invents reviewers, or satisfies the human gate. Its
replacement-run ceiling is $0.905727. Together with the frozen ledger's
conservatively accounted $7.094273 reserve for seven failed or stopped runs,
including the latest approved thirty-request run that stopped when its thresholds
became unreachable, the campaign's calibration allocation remains exactly $8.
After every durable row, the runner upper-bounds overall, calibration-split,
and holdout-split balanced accuracy from the frozen remaining class counts;
when any locked threshold is mathematically unreachable, it writes a
digest-bound termination receipt and stops before another request.
The provider response does not currently
return authoritative dollar cost; the artifact therefore keeps observed cost
unavailable and accounts the locked reserve instead of fabricating spend. The
three canaries may proceed only when this exact result is digest-valid, bound
to the approved preview, and passes all three balanced-accuracy gates with zero
critical false passes. Validation reloads the frozen cases, checks every case
identity, repository, split, score schema, and label, and recomputes all
metrics from the host-only authored references; self-reported summary metrics
are not trusted. The execution boundary must also resolve the approval digest
to the exact preview with limits of 48 requests and $0.905727 before accepting
the result. Their deterministic gates remain authoritative while human judge
calibration is pending.

## Reporting

One-attempt canaries retain the historical `scientific-report-template.json`.
Repeated confirmations use `scientific-report-template-v2.json` only after
canonical rows are terminal, reconciled, and their frozen trace audit is
completed by two reviewers. Reports keep deterministic outcomes, blind judge
labels, Skill-use mechanism evidence, efficiency, evidence links, and
limitations separate. They may
not pool scores into a cross-repository ranking or claim repeatability from the
one-attempt canary. Canonical V2 and V3 results are supported. V2 reports mark
task validity, source topology, aligned analysis, score explanations, and
sanitized excerpts as not assessed or unavailable; they never reconstruct
those later contracts from labels or display fields. V3 retains the stronger
lineage, topology, and task-validity checks.

Generate each report independently from that Study's canonical V3 result and
the exact checked-in spec, for example:

```bash
uv run python \
  examples/comparisons/community-skill-upgrades/generate_scientific_report.py \
  --result .fugue/results/comparisons/PREVIEW_DIGEST/result.json \
  --spec examples/comparisons/superpowers-writing-plans-upgrade/confirmatory-v6.yaml \
  --confirmatory-analysis /tmp/STUDY.confirmatory-analysis.json \
  --audit-selection /tmp/STUDY.trace-audit-selection.json \
  --audit-review /tmp/STUDY.trace-audit-completed.json \
  --output .fugue/results/comparisons/PREVIEW_DIGEST/scientific-report.json
```

Repeat with the Anthropic and Vercel result/spec pair. The generator refuses V1
results, nonterminal or unreconciled attempts, unresolved Weave chains, or any
identity mismatch that the source result version can prove. It emits
the canonical result, Study/spec, preregistration, selection, and signed audit
digests;
deterministic dimensions, advisory anchored judge labels and reasons (or an
explicit unavailable row for a missing optional V2 review), explicit
observed-versus-reserved cost status, aggregate Skill
assignment/registration/invocation evidence, latency and token coverage, and
all five resolved Weave links per attempt. For V2, the exact candidate revision
is checked against result metadata while the baseline revision is reported from
the checked-in campaign contract with a mandatory limitation because V2 has no
cohort lineage. It reads private-label bytes only through the canonical result
verification boundary and never serializes private labels or authored
references. Generate one report per Study; the tool intentionally has no
multi-repository pooling mode.
