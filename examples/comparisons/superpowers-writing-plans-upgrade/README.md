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

V5 is the fresh post-integrity-repair Study. It uses complete historical
repository trees, candidate-neutral plan checks,
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
SPEC=examples/comparisons/superpowers-writing-plans-upgrade/comparison-v5.yaml
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

## Preregistered confirmatory Study

> **Current rerun identity — V5.** V1 and V2 remain invalid infrastructure
> audit history. V3 run `20260803T173311-30fd2fd2f4` produced seven terminal
> evidence rows, then Fugue misclassified a bounded `AgentTimeoutError` as an
> infrastructure failure and cancelled 185 cells. V3 has no canonical result
> and contributes no behavioral rows. V4 run
> `20260803T202659-d4afb1438c` then exposed silent 16,000-character artifact
> truncation, unsafe path classification, and cancellation scoring at its
> first pair; it was stopped before the remaining 190 cells. V4 also has no
> behavioral result. `confirmatory-v5.yaml` restarts all 192 cells with the
> versioned full-artifact and V3 scorer contracts under a fresh Study, project,
> source lock, preview, and approval.

`confirmatory-v1.yaml` froze the design without reinterpreting the four-cell
canaries. `preregistration-confirmatory-v5-amendment.json` hashes that exact V1
preregistration plus the V4 amendment, records the V4 integrity failure before
any valid pair existed, and versions the deterministic scorer and exact primary
artifact contract. The hypotheses, taskset, holdout membership, treatments,
model, harness, limits, budget, scheduling seed, and primary decision rules
remain frozen. V1–V4 rows may not be pooled, used as priors, or selectively
resumed. A mandatory descriptive
sensitivity excludes the one V3-exposed holdout task and reports whether the
V1 conclusion changes. The design contains eight scorer-development briefs and sixteen untouched holdouts
across target, indirect-transfer, negative-transfer, ambiguous-evidence,
safety, and repository-shape strata:

```text
24 tasks × 2 exact Skill revisions × 4 repeated attempts = 192 cells
```

The logical task—not each repeated attempt—is the inferential unit. The
preregistered task-cluster analysis, multiplicity correction, missing-data
policy, trace-audit sample, decision rules, and claim limits are recorded in
`preregistration-confirmatory-v1.json` before the original preview. Development
tasks may validate the scorer but are excluded from the primary holdout effect.

The deterministic V3 scorer uses host-only repository oracles. It verifies
exact modification paths and inspected symbols, cross-component producer and
consumer edges, cohesive work units, scenario-level assertions, and safety. It
does not reward candidate-authored `Global Constraints` or `Interfaces`
headings and rejects unsupported self-reports. The documentation-only holdout
marks an invented code interface as negative transfer rather than requiring
interface ceremony.

The same-family LLM judge remains optional and advisory. Its synthetic
calibration is not reviewer-qualified, so it cannot satisfy the confirmatory
endpoint or override deterministic correctness and safety failures. A qualified
judge revision would require reviewed calibration, a new scorer/spec digest,
and a new preview. The frozen human trace audit is descriptive mechanism and
validity evidence; it cannot override deterministic safety failures.

Trusted preparation uses a local Git object only:

Run these commands only after the V5 implementation, preparation logic, and
analysis code are committed on the exact clean head that will be previewed.
Any later change to either script invalidates the source lock, preview, and
approval and requires this sequence again.

```bash
uv run python \
  examples/comparisons/superpowers-writing-plans-upgrade/prepare_confirmatory_sources.py

uv run python \
  examples/comparisons/community-skill-upgrades/prepare_local_source_lock.py \
  examples/comparisons/superpowers-writing-plans-upgrade/confirmatory-v5.yaml \
  --extra examples/comparisons/superpowers-writing-plans-upgrade/preregistration-confirmatory-v1.json \
  --extra examples/comparisons/superpowers-writing-plans-upgrade/preregistration-confirmatory-v5-amendment.json \
  --extra examples/comparisons/community-skill-upgrades/analyze_confirmatory.py \
  --extra examples/comparisons/superpowers-writing-plans-upgrade/prepare_confirmatory_sources.py \
  --extra .fugue/comparison-resources/superpowers-writing-plans-conference-v1/preparation.receipt.json \
  --output .fugue/qualification/community-skill-confirmatory/superpowers-v5/source.lock.json
```

The first command verifies the exact source commit and tree, validates every
private path and symbol oracle against that tree, and exports the tracked
implementation and public-test tree after removing every dataset verifier,
reference solution, private label, recorded answer, evaluation definition, and
demo credential fixture. Its receipt records the exact excluded paths and both
visible/excluded path digests alongside the archive, taskset, private-label,
scorer, preregistration, and Skill-lock digests. The second command binds every
prepared local input, including both exact Skill bundles, the source-archive
generator, and the confirmatory analyzer that computes the mandatory
leave-exposed-task-out sensitivity, into the generic V5 local source lock used
for pre-run, checkpoint, and post-run drift checks. The
archived commit predates this taskset, so the mounted source cannot contain its
private oracle. Trials may unpack the read-only archive but may not clone,
install, download, or build.

The source topology is public metadata in
`wandb/fugue-superpowers-writing-plans-source-v1`: versioned task IDs, public
brief/resource digests, source tree/archive digests, and exact Skill commits.
It contains no expected values and remains read-only during execution. All
Agent traces, Evaluations, results, and decisions go to
`wandb/fugue-superpowers-writing-plans-confirmatory-v5`. Mounted archives remain
the actual locked task inputs; cells do not query the hosted source catalogue.

Use the governed flow with the new identity:

```bash
SPEC=examples/comparisons/superpowers-writing-plans-upgrade/confirmatory-v5.yaml
ENV_FILE=/Users/ashah/Documents/common_tools/.env

env -u OPENAI_API_KEY uv run fugue check "$SPEC" \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --prepare \
  --env-file "$ENV_FILE" --json
env -u OPENAI_API_KEY uv run fugue compare "$SPEC" --preview \
  --env-file "$ENV_FILE" --json
```

The finite $1,700 ceiling is an admission guard, not a spending target. It
covers the previously observed worst-case per-cell reservation while preserving
an exact approval identity. The first aligned two-cell pair is the evidence
checkpoint; the other 190 cells cannot start until both cells' project,
Agent/Evaluation/Dataset links, privacy scan, spend, and run-scoped Harbor
cleanup reconcile. Use the dedicated read-only
`study-console-confirmatory-v5.yaml` profile on port `18103`.
