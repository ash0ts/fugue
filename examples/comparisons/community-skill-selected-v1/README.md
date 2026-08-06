# Public Skill revision case studies

This directory contains the public, reproducible half of three independent Skill-upgrade studies. Each lane fixes Claude Code, Sonnet 5, local Docker, task prompts, revision identities, deterministic scoring code, and evidence destinations. Results must be interpreted per repository; the campaign does not pool lanes or rank Skills globally.

Before preparation, an operator must verify that every declared source, result,
and campaign-index project under the `wandb` entity is private. Fugue verifies
the exact destination identity and rejects cross-project writes; project access
policy remains an administrator-controlled W&B prerequisite rather than an
Agent-visible setting.

Only four development tasks per lane are authored here. Every task binds one deterministically generated, digest-locked public archive; archives are built under ignored `.fugue/comparison-resources` and are not duplicated in Git. Private labels, sealed holdouts, reviewer submissions, adjudication content, model outputs, approvals, and execution receipts are operator-owned inputs under `.fugue/` and intentionally absent from Git. The checked-in specs fail closed until an operator prepares those private paths.

Validate the public contracts without fetching or writing:

```bash
uv run python examples/comparisons/community-skill-selected-v1/prepare.py
```

Prepare and verify all 24 selected development and sealed tasks in one trusted boundary. The restricted packet supplies reviewed known-good artifacts and the preregistered reserve pool; preparation builds one isolated targeted mutant per task and publishes only digests:

Before first preparation, a host reviewer audits every historical project in `sealed-holdouts.json`. The query requests only task identities and public input/resource fields, normalizes them to fingerprints, and rejects output, score, feedback, label, and cost fields. Its private receipt covers all projects without truncation and stores only digests, match status, and a reviewer-identity digest:

```bash
env -u OPENAI_API_KEY uv run python \
  examples/comparisons/community-skill-selected-v1/manage_sealed_holdouts.py \
  audit-history \
  --env-file /Users/ashah/Documents/common_tools/.env \
  --operator-source /PATH/TO/RESTRICTED/SEALED-PACKET \
  --reviewed-by REVIEWER_ID
```

The reviewer must inspect the reported exact-ID and input-fingerprint matches.
An exposed selected task requires its declared same-family reserve; any exposed
reserve blocks preparation. The following command then re-derives every
selected/reserve fingerprint and refuses a missing, tampered, partial, or
outcome-bearing historical receipt:

```text
RESTRICTED-PACKET/LANE/
  development-private-labels.jsonl  # four locked development labels
  tasks.jsonl                       # four selected + four reserve holdouts
  private-labels.jsonl              # matching selected + reserve truth
  resources/...                     # digest-locked holdout archives
```

Repeat that layout for each of the three lane IDs. Files are copied to ignored
mode-`0600` state only after their public digest locks match.

```bash
uv run python examples/comparisons/community-skill-selected-v1/prepare.py \
  --prepare-resources \
  --operator-source /PATH/TO/RESTRICTED/SEALED-PACKET
```

Fetch, inspect, and lock all six exact public Skill revisions through Fugue's
canonical import boundary:

```bash
uv run python examples/comparisons/community-skill-selected-v1/prepare.py \
  --fetch-skills --prepare-resources \
  --operator-source /PATH/TO/RESTRICTED/SEALED-PACKET
```

`fugue compare --prepare` then publishes one excluded `task-source-manifest` Run/artifact to the lane's declared source project. Its immutable version and digest are bound into preview, approval, and pre/checkpoint/post drift checks; deletion, drift, or result-project substitution stops execution.

## Generic post-trial verifier boundary

The Anthropic and Vercel comparisons declare Fugue's generic host-verifier
adapter on their deterministic evaluators. Each verifier is configured only
from the reviewed spec, never from Agent output, and performs these steps after
the Agent exits:

1. Materialize the approved public task archive by its locked SHA-256.
2. Overlay only the Agent's returned `files`, rejecting links, traversal, and
   undeclared paths.
3. Run either the strict Skill-package envelope validator or the pinned
   `node --test` suite in an exact no-network, read-only OCI runtime.
4. Add its signed, host-owned receipt to scorer evidence as `host_verifier`.

Each receipt binds `task_id`, public archive SHA-256, canonical Agent-files
SHA-256, runtime-lock digest, exact verifier identity and command, exit status,
and stdout/stderr digests. The public scorer independently recomputes the
artifact and receipt digests and fails its bound dimension closed when the
receipt is missing or inconsistent. A trial may not supply this evidence field
itself. For Vercel, `verification_passed` is an authoritative outcome; the
separate `behavior_preservation` safety gate additionally binds the declared
preserved-file contract. For Anthropic, the host validates package structure,
while task-specific compatibility and instruction truth remain private-label
backed scorer outcomes.

Judge generation and human review are separate operator procedures. The
campaign deliberately uses a two-step approval flow: approve and complete the
48-request calibration first, obtain two independent blinded reviews, resolve
every disagreement, and write the passing receipt. Only then generate the
three exact Study previews and their finite stage approvals. Pre-calibration
`fugue check` is blocked by design. Changing one byte of the receipt changes
the evaluator, preview, approved-input, and stage-subset digests, invalidating
every prior Study approval.

Each lane's judge remains optional and advisory at evaluation time. A later
judge-service failure is recorded as unavailable and cannot turn an Agent
artifact into a task failure. The strict shared receipt/rubric is nevertheless
mandatory before any campaign Agent cell is admitted. Deterministic outcomes
and safety gates remain authoritative.

## Sealed holdouts

Advancement requires observed assignment, registration, and opening for both
the baseline and candidate Skill bundles. When a task declares relevant Skill
files, every applicable arm must open them. Native invocation remains reported
mechanism evidence rather than a correctness proxy.

`sealed-holdouts.json` publishes only the 12 preregistered task IDs, behavior
families, same-family reserve IDs, and content digests. Prompts, labels,
qualification fixtures, and task resources stay in the operator packet passed
to the single preparation command above and are materialized with mode `0600`
beneath ignored `.fugue/private` state.

The command independently proves the targeted base/mutant fails and the
known-good artifact passes for all 12 tasks. Immediately before any holdout
preview, query every declared historical W&B/Weave project using only task
identity fields:

```bash
env -u OPENAI_API_KEY uv run python \
  examples/comparisons/community-skill-selected-v1/manage_sealed_holdouts.py \
  audit --env-file /Users/ashah/Documents/common_tools/.env
```

The resulting `HoldoutExposureAuditV1` expires after one hour, binds the private
historical receipt, prepared pool fingerprints, complete project coverage, and
the fresh safe query projection, and declares
`outcome_data_consulted=false`. A matched task fails closed unless its exact
preregistered same-family reserve is supplied. Matching uses exact task IDs and
prompt/input fingerprints, so renaming an exposed prompt does not bypass the
gate. Reserve content must still be prepared and digest-locked before preview.

Create the immutable advancement decision from the canonical development result
(and include the fresh audit when advancing):

```bash
uv run python \
  examples/comparisons/community-skill-selected-v1/manage_sealed_holdouts.py \
  advance --result /PATH/TO/result.json \
  --audit .fugue/private/community-skill-selected-v1/LANE/holdout-exposure-audit.json \
  --output /PATH/TO/ADVANCEMENT-DECISION.json
```

For an `advance_holdout` decision, build the private 16-cell comparison while
that audit is fresh, then preview and approve the generated spec as a new
immutable cohort:

```bash
uv run python \
  examples/comparisons/community-skill-selected-v1/manage_sealed_holdouts.py \
  holdout --lane LANE \
  --decision /PATH/TO/ADVANCEMENT-DECISION.json \
  --audit .fugue/private/community-skill-selected-v1/LANE/holdout-exposure-audit.json
```

The generated authorization binds the development decision, exposure audit,
private suite and zero-model receipts. The campaign allocation receipt permits
exactly one follow-up branch per lane and caps the campaign at 96 logical cells
and four predeclared infrastructure replacements.

When a valid development result is non-discriminating, do not open holdouts.
Build a separate candidate-versus-no-Skill diagnostic from its immutable
advancement decision:

```bash
uv run python \
  examples/comparisons/community-skill-selected-v1/manage_sealed_holdouts.py \
  diagnostic --lane LANE --decision /PATH/TO/ADVANCEMENT-DECISION.json
```

That private spec contains exactly two checkpoint tasks, two arms, and one
attempt (four logical cells). Its lock states
`allocation_action=replaces_holdout` and admits zero holdout cells, so it
cannot silently expand campaign spend.
