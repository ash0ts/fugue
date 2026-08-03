# Superpowers `writing-plans` upgrade canary

This is a real, bounded upgrade decision for the `writing-plans` Skill from
[`obra/superpowers`](https://github.com/obra/superpowers). Fugue locks only the
selected Skill directory, not unrelated repository changes.

- Baseline: `de4672b171213a6ff6960228d8b95c46ea0b09f4`
- Candidate: `8e1262a3bae92b640d87fa81c51c53b65e490590`
- Path: `skills/writing-plans`
- Upstream diff: <https://github.com/obra/superpowers/compare/de4672b171213a6ff6960228d8b95c46ea0b09f4...8e1262a3bae92b640d87fa81c51c53b65e490590>

The candidate adds task right-sizing, a Global Constraints section, and
explicit producer/consumer interfaces. The canary tests whether those changes
transfer to two historical Fugue maintenance briefs: a small credential-
rotation repair and a cross-layer evidence-destination identity change.

`superpowers-writing-plans-fugue-canary-v1` and V2 remain failed audit history.
V1 exposed a stable-alias/runtime-name mismatch before Claude ran. V2 exposed
two more product defects: Fugue continued after observed spend made its
approval ceiling impossible, and the private scorer required literal
candidate-authored headings. V2's seven reviewable plans are diagnostic only;
they do not support an upgrade claim.

V4 uses complete historical repository trees, candidate-neutral plan checks,
and a four-cell canary: `2 tasks × 2 exact Skill revisions × 1 attempt`. It uses
Claude Code, Anthropic Sonnet 5, and local Docker/Harbor. The result project is
`wandb/fugue-superpowers-writing-plans-upgrade-v1`. A positive result supports only
the two locked briefs; it is not a universal ranking of Superpowers or Claude.

Import and lock the exact reviewed bundles before previewing:

```bash
uv run fugue skills import \
  'git+https://github.com/obra/superpowers@de4672b171213a6ff6960228d8b95c46ea0b09f4#path=skills/writing-plans' \
  --as superpowers-writing-plans-before-contracts
uv run fugue skills inspect superpowers-writing-plans-before-contracts
uv run fugue skills lock superpowers-writing-plans-before-contracts

uv run fugue skills import \
  'git+https://github.com/obra/superpowers@8e1262a3bae92b640d87fa81c51c53b65e490590#path=skills/writing-plans' \
  --as superpowers-writing-plans-contracts
uv run fugue skills inspect superpowers-writing-plans-contracts
uv run fugue skills lock superpowers-writing-plans-contracts

uv run python \
  examples/comparisons/superpowers-writing-plans-upgrade/prepare_snapshots_v3.py
```

The preparation script uses local Git only. It exports each complete historical
Fugue tree into `.fugue/`, records the commit, tree, archive digest, path-list
digest, and file count, and lets preparation copy the immutable archives into
the task images. Trials do not clone, download, or infer later source state.

Then use the normal governed flow:

```bash
SPEC=examples/comparisons/superpowers-writing-plans-upgrade/comparison-v4.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

env -u OPENAI_API_KEY uv run fugue check "$SPEC" --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --prepare \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --preview \
  --env-file "$ENV_FILE" --json
```

Every paid execution requires approval of that exact preview. Runtime accounting
now stops queued cells when cost is missing, observed spend crosses the ceiling,
or observed spend plus the locked remaining reservations cannot fit. One atomic
cell can still exceed its reservation, so the preview must use a realistic
per-cell reserve. The deterministic scorer owns correctness. Skill registration
and invocation remain separate mechanism evidence and cannot manufacture an
improvement.

For a dedicated read-only Study Console, start the sibling Study Console
checkout on port `18084` with `study-console.yaml`. The experiment backlink and
Research record sink must target that same instance; do not reuse an MCP-only
Console profile.
