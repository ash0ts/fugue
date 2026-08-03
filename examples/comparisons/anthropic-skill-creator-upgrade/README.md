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

```bash
SPEC=examples/comparisons/anthropic-skill-creator-upgrade/comparison.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --prepare \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --preview \
  --env-file "$ENV_FILE" --json
```

Approve only that exact preview with four cells and a `$34` ceiling. Fugue
reserves `$8.40` for each Agent attempt and `$0.10` for each judge call. Run
the checkpoint cell first; continue only after the W&B project, Agent and
Evaluation links, privacy scan, cost, and Harbor cleanup reconcile.

The result destination is
`wandb/fugue-anthropic-skill-creator-upgrade-v1`. Start the generic read-only
Study Console with `study-console.yaml` on port `18085`; do not reuse another
campaign's database or project profile.
