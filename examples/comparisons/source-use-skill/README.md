# Source-verification Skill comparison

This is the live flagship design for Fugue's technical preview. It asks one
bounded question: does adding the reviewed `verify-current-source` Agent Skill
change outcomes on a failure-enriched source-use cohort?

The eight holdout tasks contain current, superseded, and draft documents.
Expected facts and the authoritative filename remain in the private labels.
The Agent receives only the public task and locked document bundle.

```bash
uv run fugue check \
  examples/comparisons/source-use-skill/comparison.yaml

uv run fugue compare \
  examples/comparisons/source-use-skill/comparison.yaml \
  --preview
```

The preview resolves exactly:

`8 tasks × 2 candidates × Codex × 2 attempts = 32 attempts`

Live execution requires approval of the exact preview digest. No live cohort is
included in this branch. The four discovery cases in `discovery-cases.md` are
separate from the frozen holdout and must be used before any paid qualification.

Mechanism reporting keeps these states distinct:

`Skill assigned → registered → invoked → current source opened → current source used → task passed`

An unavailable mechanism signal remains unavailable; assignment is never
reported as invocation.
