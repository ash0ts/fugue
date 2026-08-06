# Advisory judge calibration

The 48 operator-restricted cases are digest-bound and balanced without publishing private truth. Legacy model outputs are incompatible; see `legacy-model-output-status.json`.

```bash
ROOT=examples/comparisons/community-skill-selected-v1/judge
PRIVATE=.fugue/private/community-skill-selected-v1
uv run python "$ROOT/materialize_cases.py" --manifest "$ROOT/case-set-manifest.json" --source /OPERATOR/PATH/judge-calibration-cases.jsonl
uv run python "$ROOT/run_calibration.py" preview --manifest "$ROOT/case-set-manifest.json" --cases "$PRIVATE/cases.jsonl" --rubric "$ROOT/rubric.json" --packet "$PRIVATE/review-packet.json" --preview-out "$PRIVATE/generation-preview.json"
DIGEST=$(jq -r .preview_digest "$PRIVATE/generation-preview.json")
uv run fugue approve "$DIGEST" --max-cells 48 --max-usd 8 --approved-by operator
env -u OPENAI_API_KEY uv run python "$ROOT/run_calibration.py" run --manifest "$ROOT/case-set-manifest.json" --cases "$PRIVATE/cases.jsonl" --rubric "$ROOT/rubric.json" --packet "$PRIVATE/review-packet.json" --preview-out "$PRIVATE/generation-preview.json" --approval APPROVAL_DIGEST --env-file /Users/ashah/Documents/common_tools/.env
```

The runner is serial, never retries, binds its own source digest, removes any `OPENAI_API_KEY` loaded from the environment, and refuses ambiguous replay. Two real blinded human reviews plus full disagreement adjudication remain mandatory. Finalization uses evaluated Agent family `anthropic` and exact judge profile `anthropic/claude-sonnet-5`.

After model generation completes, create two independent editable forms. The forms contain the public task, sanitized artifact, permitted evidence, and blank `decision` fields; they never contain authored truth or treatment identity. Give each reviewer only their own form and the checked-in rubric.

```bash
REVIEW="$ROOT/review_calibration.py"
COMMON="--repo-root . --manifest $ROOT/case-set-manifest.json --cases $PRIVATE/cases.jsonl --rubric $ROOT/rubric.json --packet $PRIVATE/review-packet.json"

uv run python "$REVIEW" review-templates $COMMON \
  --reviewer-a-out "$PRIVATE/reviewer-a-template.json" \
  --reviewer-b-out "$PRIVATE/reviewer-b-template.json"
```

Each human edits only `acceptable`, `critical_defect`, and `reason` beneath every case's `decision`. `acceptable` and `critical_defect` are booleans; `critical_defect=true` is valid only with `acceptable=false`. Use stable, pseudonymous 64-hex identity digests from the operator's reviewer registry; never put names, email addresses, credentials, or private truth in a form.

```bash
uv run python "$REVIEW" submit-review $COMMON \
  --review-template "$PRIVATE/reviewer-a-template.json" \
  --reviewer-identity-digest "$REVIEWER_A_DIGEST" \
  --submission-out "$PRIVATE/reviewer-a-submission.json"

uv run python "$REVIEW" submit-review $COMMON \
  --review-template "$PRIVATE/reviewer-b-template.json" \
  --reviewer-identity-digest "$REVIEWER_B_DIGEST" \
  --submission-out "$PRIVATE/reviewer-b-submission.json"
```

Generate a form containing every and only the two reviewers' disagreements. An empty form is correct when they agree on all 48 cases. The adjudicator edits only its blank `decision` fields, then signs the result.

```bash
REVIEWS="--first-submission $PRIVATE/reviewer-a-submission.json --second-submission $PRIVATE/reviewer-b-submission.json"

uv run python "$REVIEW" adjudication-template $COMMON $REVIEWS \
  --adjudication-template-out "$PRIVATE/adjudication-template.json"

uv run python "$REVIEW" submit-adjudication $COMMON $REVIEWS \
  --adjudication-template "$PRIVATE/adjudication-template.json" \
  --adjudicator-identity-digest "$ADJUDICATOR_DIGEST" \
  --adjudication-out "$PRIVATE/adjudication.json"
```

Finally, independently rebuild the model-output receipt, require it to equal the approved generation receipt, bind both signed submissions and the signed adjudication, and write the strict receipt used by Study previews:

```bash
uv run python "$REVIEW" finalize $COMMON $REVIEWS \
  --model-outputs "$PRIVATE/model-outputs.jsonl" \
  --generation-receipt "$PRIVATE/model-output-receipt.json" \
  --adjudication "$PRIVATE/adjudication.json" \
  --evaluated-agent-model-family anthropic \
  --final-receipt-out "$PRIVATE/judge-calibration-receipt.json"
```

Finalization prints overall and per-modality TPR/TNR, critical false passes, status, and the immutable receipt digest. `PASSED` means TPR/TNR are at least 0.85 overall and within every modality with zero critical false passes. The judge remains advisory because it shares the Agent model family; a passing calibration does not override deterministic task or safety failures.

Each Study binds an optional $0.10 advisory judge to the exact rubric, private receipt, profile, and lane modality. Fugue requires 48 adjudicated cases, overall and per-modality TPR/TNR ≥0.85, and zero critical false passes.

Approval is intentionally two-step: complete the approved calibration first, then generate and approve the exact Study previews. Missing, legacy, failed, or changed calibration blocks Agent admission. A judge-service failure after admission remains advisory/unavailable and does not change deterministic task outcomes or safety gates.
