# Vercel React Best Practices Skill upgrade canary

This lane compares two exact revisions of the public
[`vercel-labs/agent-skills`](https://github.com/vercel-labs/agent-skills)
`skills/react-best-practices` Skill. It does not rank the repository or claim
that one revision is universally better.

- Baseline: `ac6a79af08f6d32c34ee03c829824990f3de0a6d`
- Candidate: `20987af2f1bc17857b55e7758af8bed91c364ff5`
- Upstream diff: <https://github.com/vercel-labs/agent-skills/compare/ac6a79af08f6d32c34ee03c829824990f3de0a6d...20987af2f1bc17857b55e7758af8bed91c364ff5>
- Matrix: two tasks × two revisions × one Claude Code attempt = four cells
- Evidence: `wandb/fugue-vercel-react-best-practices-upgrade-v1`
- Harbor budget: $34, including a $0.10 judge reservation per cell

The tasks are small but executable Next.js-style maintenance cases. One
requires authentication and authorization inside a Server Action before a
mutation. The other removes redundant React Server Component serialization
while preserving accessible output. The checked-in fixtures intentionally fail
their focused Node tests before repair. Each public task declares an exact JSON
answer schema containing the complete final changed files and the complete
`node --test` receipt. The deterministic scorer derives behavior from those
file contents and test names; it does not trust self-reported correctness
booleans. Private expected source properties never enter Agent inputs.

Skill registration and invocation remain Fugue mechanism evidence. They are
not reimplemented as a filename heuristic inside the task-correctness scorer.

The checked-in Skill lock binds repository, path, and exact commits. Bundle
digests remain `null` until Fugue performs its explicit reviewed import; the
repository does not fabricate a content digest without obtaining the upstream
Skill. Import and lock both revisions before previewing:

```bash
uv run fugue skills import \
  'git+https://github.com/vercel-labs/agent-skills@ac6a79af08f6d32c34ee03c829824990f3de0a6d#path=skills/react-best-practices' \
  --as vercel-react-best-practices-before
uv run fugue skills inspect vercel-react-best-practices-before
uv run fugue skills lock vercel-react-best-practices-before

uv run fugue skills import \
  'git+https://github.com/vercel-labs/agent-skills@20987af2f1bc17857b55e7758af8bed91c364ff5#path=skills/react-best-practices' \
  --as vercel-react-best-practices-after
uv run fugue skills inspect vercel-react-best-practices-after
uv run fugue skills lock vercel-react-best-practices-after

uv run python \
  examples/comparisons/vercel-react-best-practices-upgrade/prepare_fixtures.py
```

`prepare_fixtures.py` performs no network access. It verifies every fixture
file against `fixture-sources.lock.json`, produces normalized deterministic
archives under `.fugue/`, and records their hashes. Trials receive the prepared
archives and may not clone, install, or download.

The shared 48-example synthetic/gold judge calibration is a separate governed
preview. Its human-review fields remain honestly pending rather than inventing
reviewers. Until an adjudicated report passes the 0.85 TPR/TNR and
zero-critical-false-pass gates, the same-model qualitative judge is advisory
and may be unavailable. It cannot override a deterministic security or
correctness failure.

V2 is the fresh post-integrity-repair Study. V1 remains immutable audit
history and is not reused or reinterpreted. Use the normal governed flow after
calibration and preparation:

```bash
SPEC=examples/comparisons/vercel-react-best-practices-upgrade/comparison-v2.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --prepare \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --preview \
  --env-file "$ENV_FILE" --json
```

Approve that exact preview with four cells and $34 only after the checkpoint
inputs are reviewed. Start the read-only Study Console with
`study-console.yaml` on port `18086`. A single-attempt result is a canary and
must display that variance limitation.
