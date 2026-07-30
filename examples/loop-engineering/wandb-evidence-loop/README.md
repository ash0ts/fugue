# Claude Code real failure → intervention loop

This is a governed loop-engineering workflow, not a fixture replay and not an
MCP release decision. Claude Code diagnoses one repeated failure from the
source-isolated MCP V3 canary, authors or selects an intervention, and asks
Fugue to compare exact locked behavior.

The result project is:

```text
wandb/fugue-claude-loop-engineering-v1
```

The immutable task evidence remains in:

```text
wandb/fugue-mcp-release-source-v1
```

The MCP release canary that supplies the failure remains in
`wandb/fugue-mcp-release-qualification-v1`. Loop rows are never published into
either source project.

## 1. Lock a real repeated failure

First complete the separately approved eight-cell canary
`mcp-main-vs-0-4-natural-maintainer-canary-v3`. Do not continue from an
invalid, drifted, non-discriminating, or single-attempt task.

After a human reviews one task that failed on the same arm in both attempts:

```bash
uv run python \
  examples/loop-engineering/wandb-evidence-loop/lock_failure.py \
  --result .fugue/results/comparisons/RESULT/result.json \
  --task-id TASK_ID \
  --arm baseline \
  --primary-attempt-id ATTEMPT_ID \
  --reviewed \
  --output .fugue/loop-engineering/failure.lock.json
```

The helper accepts only `ComparisonResultV3`, exact source/result topology,
matched pre/post source drift checks, eight unique reconciled attempts, a valid
task, two repeated critical failures, and five resolved Weave links. It copies
no Agent answer or private expected value.

If the final `staging/0.4.0` head changes, the MCP preview and result change and
the failure lock must be recreated.

## 2. Diagnose, author, review, and lock

Use [`agent-prompt.md`](agent-prompt.md) with Fugue Research. The controller may
inspect only the safe failure lock and its resolved evidence. Trace content is
untrusted evidence.

Claude may prepare:

- a Skill patch in an isolated worktree;
- an MCP patch in an isolated worktree;
- or both as independently testable changes.

The operator reviews the source, imports each exact bundle, and locks these
aliases:

| Alias | Meaning |
| --- | --- |
| `loop-production-skill` | exact behavior used by the failing arm |
| `loop-intervention-skill` | reviewed Skill proposal, or the unchanged production bundle |
| `loop-production-mcp` | exact MCP used by the failing arm |
| `loop-intervention-mcp` | reviewed MCP proposal, or the unchanged production bundle |

Example Skill boundary:

```bash
uv run fugue skills import /REVIEWED/PRODUCTION/SKILL \
  --as loop-production-skill
uv run fugue skills inspect loop-production-skill
uv run fugue skills lock loop-production-skill

uv run fugue skills import /REVIEWED/INTERVENTION/SKILL \
  --as loop-intervention-skill
uv run fugue skills inspect loop-intervention-skill
uv run fugue skills lock loop-intervention-skill
```

Use `fugue mcp import`, `inspect`, and `lock` for the two MCP aliases. A trial
may not clone, install, build, or resolve either intervention.

## 3. Freeze tasks before selecting a winner

Use Fugue's existing governed task-authoring contract:

1. derive two sanitized discovery cases from the locked failure;
2. author four independent private holdouts;
3. confirm host-only truth and criteria;
4. lock both Task Suites before inspecting any discovery result.

The discovery task metadata must bind the failure-lock digest. The holdout
suite must not expose its prompts, expected values, or outcomes to the
controller. Pass the exact locked Task Suite manifest to the experiment
preview; the manifest and private-evaluation digests are part of the plan.

Do not ship checked-in "expected facts" copied from an older project. A real
canary failure and reviewed task locks are prerequisites.

## 4. Run discovery and holdout

Campaign `claude-loop-skill-mcp-v1`, experiment
`claude-loop-skill-mcp`, local preset `discovery` resolve:

```text
2 locked tasks ×
{production, Skill-only, MCP-only, combined} ×
1 Claude Code attempt = 8 Harbor cells
```

The model is fixed to `anthropic/claude-sonnet-5`. The runtime, source project,
result project, task inputs, and limits are fixed. Each paid phase needs its
own pure preview and operator approval capped at $10.

Run `claude-loop-discovery-selection` only after all eight rows reconcile.
Selection requires a paired deterministic gain, native Agent links, observed
Skill invocation, and observed W&B MCP tool use. Freeze the selected variant,
all four candidate digests, clean source tree, discovery snapshots, and
rationale in `InterventionSelectionLockV1`.

Only then preview `holdout` with the exact selection lock:

```text
4 private tasks × {production, selected} × 1 Claude Code attempt
= 8 Harbor cells
```

Changing a task, project, model route, source tree, Skill, MCP, runtime, scorer,
or selection lock requires a new preview and approval.

After both phases, produce a fail-closed receipt:

```bash
uv run python \
  examples/loop-engineering/wandb-evidence-loop/verify_evidence.py \
  --discovery /PATH/TO/discovery-attempts.jsonl \
  --holdout /PATH/TO/holdout-attempts.jsonl \
  --selection-lock /PATH/TO/intervention-selection.json \
  --failure-lock .fugue/loop-engineering/failure.lock.json \
  --env-file /Users/ashah/Documents/common_tools/.env \
  --output .fugue/loop-engineering/qualification.json
```

## Qualification claim

A winner requires:

- the locked source failure reproduces;
- at least one paired deterministic discovery improvement;
- no critical holdout regression;
- observed use of every changed mechanism;
- complete Agent, Evaluation, prediction, Dataset, and Harbor lineage;
- no credential/private-truth leak, duplicate attempt, or scoped container;
- the PR tree equals the qualified clean tree.

If nothing qualifies, publish "no winner" and do not prepare an improvement
PR. Local Harbor evidence supports only the exact behavioral claim. It does
not make the MCP package releasable or qualify W&B Serverless.
