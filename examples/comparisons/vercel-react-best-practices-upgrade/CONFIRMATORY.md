# Vercel React Best Practices confirmatory study

This governed study compares the exact public Skill revisions
`ac6a79af08f6d32c34ee03c829824990f3de0a6d` and
`20987af2f1bc17857b55e7758af8bed91c364ff5`. The upstream diff adds Server
Action authentication/authorization guidance and RSC prop-deduplication
guidance. The study therefore treats those two families as intended deltas and
pre-registers four established-rule families as non-regression controls.

The matrix is 24 tasks × two revisions × four attempts = 192 Harbor cells. The
eight development tasks are reported separately. The sixteen untouched
holdouts supply the primary task-clustered paired estimate; four attempts
measure execution variance but are not counted as independent tasks.

All fixtures are dependency-free and prepared outside trials. Agent-visible
tests describe symptoms through executable behavior rather than source regexes.
The gold sources and verifier contracts remain in the trusted preparation and
host-only evaluation boundary. Preparation runs every base and gold repository,
requires base failure and gold success, independently scores both sources, and
emits immutable archive and preflight receipts. Trials may not install, clone,
download, or access the Docker socket.

The deterministic scorer checks the submitted final sources and executable-test
receipt; it never treats a self-reported pass as task correctness. The Sonnet 5
judge is blinded and advisory because its separate two-reviewer calibration is
not yet complete. Judge labels cannot override deterministic security, scope,
behavior-preservation, privacy, or evidence failures.

Preparation and preview are separate:

```bash
SPEC=examples/comparisons/vercel-react-best-practices-upgrade/confirmatory-v1.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

python3 examples/comparisons/vercel-react-best-practices-upgrade/prepare_confirmatory_fixtures.py
uv run python \
  examples/comparisons/community-skill-upgrades/prepare_local_source_lock.py \
  "$SPEC" \
  --extra .fugue/comparison-resources/vercel-react-best-practices-confirmatory-v1/fixtures.lock.json \
  --extra .fugue/comparison-resources/vercel-react-best-practices-confirmatory-v1/preflight.receipt.json \
  --extra examples/comparisons/vercel-react-best-practices-upgrade/conference_fixture_catalog.py \
  --extra examples/comparisons/vercel-react-best-practices-upgrade/confirmatory-fixtures.lock.json \
  --extra examples/comparisons/vercel-react-best-practices-upgrade/prepare_confirmatory_fixtures.py \
  --extra examples/comparisons/vercel-react-best-practices-upgrade/host_node_verifier.cjs \
  --output .fugue/qualification/community-skill-confirmatory/vercel/source.lock.json
env -u OPENAI_API_KEY uv run fugue check "$SPEC" --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --prepare --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --preview --env-file "$ENV_FILE" --json
```

The exact preview requires a new 192-cell approval. The single governed run
executes one aligned baseline/candidate pair as its automatic two-cell
checkpoint before it can continue. Continue only when project routing, the
five-link Weave chain, Dataset, privacy, accounted cost, and run-scoped Harbor
cleanup reconcile. Do not adapt the taskset, scorer, candidate, or stopping
rule after inspecting outcomes.
