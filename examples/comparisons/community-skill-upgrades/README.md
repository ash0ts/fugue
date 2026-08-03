# Community Skill upgrade campaign

This directory binds three independent, task-specific Skill upgrade canaries
under one offline-validated campaign ceiling. It does not create an alternate
runner: each lane remains an ordinary Fugue comparison with its own preview,
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

## Offline validation

```bash
uv run python \
  examples/comparisons/community-skill-upgrades/validate_campaign.py
```

The command validates strict schemas, exact revisions and Study destinations,
the 36/12 balanced calibration split, the non-approved $110 ceiling, and the
claim-free scientific report template. It also hashes the exact calibration
case bytes and rejects drift before emitting judge input or making a provider
request. It exits successfully when the fixture set is internally valid while
reporting `pending_human_review`; the three previews remain approvable, but
execution stays gated on the synthetic result.

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

That report still passes only at TPR and TNR >= 0.85 in both splits, with zero
critical false passes. Each lane then needs a new immutable preview and its own
operator approval. The budget ledger is a ceiling configuration, not approval.

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
  --max-cells 48 --max-usd 4.953631 --approved-by operator \
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

It performs 48 sequential, non-retrying requests and incrementally writes a
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
replacement-run ceiling is $4.953631. Together with the conservatively
accounted $3.046369 reserve for three failed calibration runs, including the 16
completed requests abandoned with the third run, the campaign's calibration
allocation remains exactly $8. The provider response does not currently
return authoritative dollar cost; the artifact therefore keeps observed cost
unavailable and accounts the locked reserve instead of fabricating spend. The
three canaries may proceed only when this exact result is digest-valid, bound
to the approved preview, and passes overall plus held-out TPR/TNR with zero
critical false passes. Validation reloads the frozen cases, checks every case
identity, repository, split, score schema, and label, and recomputes all
metrics from the host-only authored references; self-reported summary metrics
are not trusted. The execution boundary must also resolve the approval digest
to the exact preview with limits of 48 requests and $4.953631 before accepting
the result. Their deterministic gates remain authoritative while human judge
calibration is pending.

## Reporting

Each Study copies `scientific-report-template.json` only after canonical rows
are terminal and reconciled. Reports must keep deterministic outcomes, blind
judge labels, Skill-use mechanism evidence, and efficiency separate. They may
not pool scores into a cross-repository ranking or claim repeatability from the
one-attempt canary.

Generate each report independently from that Study's canonical V3 result and
the exact checked-in spec, for example:

```bash
uv run python \
  examples/comparisons/community-skill-upgrades/generate_scientific_report.py \
  --result .fugue/results/superpowers-writing-plans-upgrade-canary-v4/result.json \
  --spec examples/comparisons/superpowers-writing-plans-upgrade/comparison-v4.yaml \
  --output .fugue/results/superpowers-writing-plans-upgrade-canary-v4/scientific-report.json
```

Repeat with the Anthropic and Vercel result/spec pair. The generator refuses
V1/V2 results, nonterminal or unreconciled attempts, unresolved Weave chains,
or any project, taskset, scorer, behavior, runtime-policy, or exact Skill
revision mismatch. It emits deterministic dimensions, advisory anchored judge
labels and reasons, explicit observed-versus-reserved cost status, aggregate
Skill assignment/registration/invocation evidence, latency and token coverage,
and all five resolved Weave links per attempt. It reads private-label bytes only
to verify their locked SHA-256 and never serializes private labels or authored
references. Generate one report per Study; the tool intentionally has no
multi-repository pooling mode.
