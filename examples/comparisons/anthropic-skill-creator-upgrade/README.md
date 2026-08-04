# Anthropic `skill-creator` compatibility upgrade canary

This Study compares two exact revisions of
[`anthropics/skills`](https://github.com/anthropics/skills) at
`skills/skill-creator`:

- Baseline: `a5bcdd7e58cdff48566bf876f0a72a2008dcefbc`
- Candidate: `1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563`
- Upstream diff: <https://github.com/anthropics/skills/compare/a5bcdd7e58cdff48566bf876f0a72a2008dcefbc...1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563>

The candidate adds compatibility-frontmatter guidance and reconciles the Skill
name limit across authoring, generation, validation, and help. The two locked
tasks exercise those surfaces directly: one creates a 44-character,
platform-bounded Skill; the other adds compatibility metadata without changing
an existing Skill's instructions or package.

Each public task declares the exact JSON value types, task identity, path base,
file set, and validation-command shape expected by the deterministic scorer.
The scorer checks requested semantics without requiring undocumented section
titles or repeating the Skill name in a memo, and distinguishes an instruction
that forbids an install command from one that tells the user to run it.

The deterministic scorer owns correctness and safety. The shared blinded
Sonnet judge reports community usefulness using anchored labels, but remains
advisory because it shares a model family with the Agent and its synthetic/gold
calibration is not a substitute for independent human review. One attempt per
arm is a canary, not a general ranking of Anthropic Skills or Claude.

## Prepare exact sources

Import and review the exact bundles, whose expected Fugue digests are checked
into `skill-revisions.lock.json`:

```bash
uv run fugue skills import \
  'git+https://github.com/anthropics/skills@a5bcdd7e58cdff48566bf876f0a72a2008dcefbc#path=skills/skill-creator' \
  --as anthropic-skill-creator-before-compatibility
uv run fugue skills inspect anthropic-skill-creator-before-compatibility
uv run fugue skills lock anthropic-skill-creator-before-compatibility

uv run fugue skills import \
  'git+https://github.com/anthropics/skills@1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563#path=skills/skill-creator' \
  --as anthropic-skill-creator-compatibility
uv run fugue skills inspect anthropic-skill-creator-compatibility
uv run fugue skills lock anthropic-skill-creator-compatibility
```

Use a local clone that already contains both commits to prepare immutable
source and task archives. The script never fetches:

```bash
uv run python \
  examples/comparisons/anthropic-skill-creator-upgrade/prepare_sources.py \
  --anthropic-repo /PATH/TO/anthropics-skills
```

Preparation records exact commits, subtree IDs, paths, archive hashes, the
reviewed Fugue bundle digests, and deterministic task-fixture hashes. Trials
receive only those prepared, read-only inputs and cannot clone or install.

## Governed canary

V2 is the fresh post-integrity-repair Study. V1 remains immutable audit
history and is not reused or reinterpreted.

```bash
SPEC=examples/comparisons/anthropic-skill-creator-upgrade/comparison-v2.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --prepare \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --preview \
  --env-file "$ENV_FILE" --json
```

Approve only that exact preview with four cells and a `$34` ceiling. Fugue
reserves `$8.40` for each Agent attempt and `$0.10` for each judge call. The
single governed run executes one aligned baseline/candidate pair as its
automatic two-cell checkpoint; it continues only after the W&B project, Agent
and Evaluation links, privacy scan, cost, and Harbor cleanup reconcile.

The result destination is
`wandb/fugue-anthropic-skill-creator-upgrade-v1`. Start the generic read-only
Study Console with `study-console.yaml` on port `18085`; do not reuse another
campaign's database or project profile.

## Historical instruction-failure replication (V1)

`failure-replication.yaml` identifies the completed four-cell V1 Study: one
public task, the same two exact Skill revisions, and two attempts per arm. Its
result (`0e40aaaeaf1105f1efc48800f1047151c280975623ec22d4ac3f8fbfd0d1f991`)
is diagnostic-only. Its deterministic scorer split ordinary Markdown headings,
definitions, and table rows into separate lexical clauses, incorrectly marking
two semantically valid artifacts as failures. A read-only audit of all four
frozen artifacts found the required traceability, PASS / FAIL-STOP /
INCONCLUSIVE semantics, and missing-evidence handling in every response.
Separate blinded same-family Sonnet judge calls produced derived `strong`
labels for all four responses. Those calls are advisory and uncalibrated; they
are not independent model-family validation and cannot override deterministic
gates.

The historical result and approval are retained as immutable audit evidence;
neither may be reused or reinterpreted with the repaired scorer. The checked-in
V1 inspection spec resolves the byte-for-byte archived scorer at
`failure_replication_scorer_v1_archived.py` (SHA-256
`ab0b2c5bee685d4523618dfdd1defe72b6c314c729ede1eabe1d71d5046b9366`).
It is not a boundary for a new preview or run. Use
`study-console-failure-replication.yaml` on port `18087` only to inspect that
diagnostic Study in
`wandb/fugue-anthropic-skill-creator-failure-replication-v1`.

## Independent semantic replication (V2)

`semantic-replication-v2.yaml` is a fresh four-cell Study with a new comparison,
research, result-project, and approval identity. It preserves the V1 task,
private labels, exact candidates, Sonnet 5 route, Claude Code harness, two
attempts per arm, Harbor runtime, and `$34` ceiling. Its repaired deterministic
scorer evaluates source traceability, terminal success-or-stop semantics, and
honest missing-evidence handling by meaning rather than hidden literal phrases;
the blinded Sonnet review remains advisory.

No V1 preparation, preview, approval, or result was valid for V2. V2 was
prepared, approved, and completed under its own immutable identity. The
commands below document the governed flow that produced it; do not reuse its
preview, approval, or Study identity for new trials. A further replication must
use a new comparison identity and fresh approval after the exact Skill bundles
and local Harbor runtime above are prepared:

```bash
SPEC=examples/comparisons/anthropic-skill-creator-upgrade/semantic-replication-v2.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env
PREVIEW=/tmp/anthropic-skill-creator-semantic-v2-preview.json
APPROVAL=/tmp/anthropic-skill-creator-semantic-v2-approval.json

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --prepare \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --preview \
  --env-file "$ENV_FILE" --json > "$PREVIEW"

PREVIEW_DIGEST=$(jq -r .preview_digest "$PREVIEW")
uv run fugue approve "$PREVIEW_DIGEST" \
  --max-cells 4 --max-usd 34 --approved-by operator > "$APPROVAL"
APPROVAL_DIGEST=$(jq -r .approval_digest "$APPROVAL")

env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --run \
  --approval "$APPROVAL_DIGEST" --fetch-weave \
  --env-file "$ENV_FILE"
```

Use `study-console-semantic-replication-v2.yaml` for its dedicated read-only
Study Console on port `18088`. The profile reads only
`wandb/fugue-anthropic-skill-creator-semantic-replication-v2`, defaults to
`anthropic-skill-creator-instruction-semantic-replication-v2`, and keeps state
in `.study-console/anthropic-skill-creator-semantic-replication-v2.sqlite3`.

The repaired scorer accepts those public semantic forms while continuing to
reject keyword lists, undefined status labels, and missing-evidence coercion.
Its completed V2 result digest is
`1b32ae2f1970dca377770029641cd3b2458aa24351d705e3715d92932bf6d94c`.
The four rows formed two aligned pairs and found no behavioral difference on
this one locked task. That narrow `unchanged` result does not retroactively
repair V1, qualify a general Skill Creator upgrade claim, or provide a valid
loop-engineering failure card.

## Descriptive measurement-development Study (V2)

`confirmatory-v1.yaml` and its generated previews remain immutable preparation
history; no V1 Agent cell ran and no behavioral result exists. Its base fixture
changed only `schema_version`, so it did not prove that the substantive scorer
dimensions rejected their intended defects. `confirmatory-v2.yaml` is the fresh
Study identity. It preserves the exact tasks, private truth, scorer, Skill
revisions, model, harness, limits, scheduling, and 192-cell matrix while binding
dimension-targeted mutation qualification and an explicit compatibility product
contract.

This is measurement-development evidence for the exact authored benchmark, not
a population or conference-qualified claim. It contains eight development
tasks and sixteen untouched holdouts, two arms, and four attempts per task:
**192 governed cells**. Attempts are within-task replication; the task is the
inference unit. Candidate-minus-baseline effects, safety regressions, mechanism
evidence, infrastructure, and the advisory judge are reported separately.

The Agent-visible task identities, prompts, and mounted archives do not name a
frontmatter field or disclose boundary answers. The answer-bearing family
mapping and expected values live only in `confirmatory-task-family-lock.json`
and `confirmatory-private-labels.jsonl`; neither is mounted into a trial. Task
archives contain no validator. `confirmatory_scorer.py` independently parses
the returned frontmatter, applies the canonical schema and boundary rules,
checks preservation hashes and allowed paths, and ignores self-reported pass
booleans.

Before a V2 preview, preparation must run the exact upstream zero-model matrix for
frontmatter absence, empty and bounded scalar values, over-bound and non-string
values, unknown fields, name lengths 40/41/64/65, and initializer help. It also
builds one deterministic, read-only archive per task from checked-in inputs:

```bash
uv run python \
  examples/comparisons/anthropic-skill-creator-upgrade/prepare_confirmatory_v2.py \
  --anthropic-repo /PATH/TO/anthropics-skills

uv run python \
  examples/comparisons/community-skill-upgrades/prepare_local_source_lock.py \
  examples/comparisons/anthropic-skill-creator-upgrade/confirmatory-v2.yaml \
  --output .fugue/qualification/community-skill-confirmatory/anthropic-v2/source.lock.json
```

The V2 spec declares every preparation receipt, validator, fixture generator,
source preparer, task-family lock, Skill revision lock, analysis input, and
scientific-report input as an approval-bound `qualification_input`. The source
lock collector resolves those declarations directly and fails before preview if
any file is absent or drifted; no README-only supplemental file list is
authoritative.

The V2 source topology declares
`wandb/fugue-anthropic-skill-creator-source-v1` for the public task/source-lock
Dataset and `wandb/fugue-anthropic-skill-creator-confirmatory-v2` for results.
Cells use the digest-locked mounted archives as their actual inputs; they do
not query the hosted source project. Result publication must never write into
the source project. `prepare_confirmatory_v2.py` recomputes the host-only
mutation suite, requires the checked-in receipt to match byte-for-byte, verifies
the product contract, and binds both into a V2 preparation lock. Neither the
receipt nor the lock serializes private expected values or mutated artifacts.

Use the normal governed flow with a fresh preview and approval. The exact
finite ceiling is `$1640` for 192 Agent calls plus advisory judge reserve; it
is an approval identity, not a spending target:

```bash
SPEC=examples/comparisons/anthropic-skill-creator-upgrade/confirmatory-v2.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --prepare \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --preview \
  --env-file "$ENV_FILE" --json
```

The single governed run executes one aligned baseline/candidate pair as its
automatic two-cell integrity checkpoint. It continues only after both cells'
exact result project, five Weave links, candidate/runtime identity,
private-label scan, cost, and Harbor cleanup reconcile. The same-family Sonnet
judge remains descriptive until two independent reviewers adjudicate the frozen
calibration; it cannot satisfy or override a deterministic gate. The read-only
Study Console profile is `study-console-confirmatory-v2.yaml` on port `18105`.
Its title and objective keep the non-conference descriptive claim boundary
visible alongside every result.
