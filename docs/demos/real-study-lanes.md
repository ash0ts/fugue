# Real Fugue Study lanes

These are three independent controlled Studies. They share Fugue's canonical
campaign lifecycle, local Docker/Harbor execution, immutable preparation, and
human approval boundary, but they do not share W&B/Weave projects or claims.

| Study | W&B/Weave project | Exact matrix | Fixed route |
| --- | --- | --- | --- |
| Claude loop engineering | `wandb/fugue-claude-loop-engineering-v1` | discovery: 2 tasks × 4 Skill/MCP arms = 8; holdout: 4 tasks × production/selected = 8 | `anthropic/claude-sonnet-5`, Claude Code |
| Harness behavior | `wandb/fugue-harness-experiments-v1` | 2 locked tasks × Hermes/OpenClaw/Claude Code/Codex = 8 | `wandb/zai-org/GLM-5.2` |
| Repository memory | `wandb/fugue-memory-experiments-v1` | 2 locked tasks × baseline/rag-dense/policy/combined = 8 | `anthropic/claude-sonnet-5`, Claude Code |

None of these Studies writes to a legacy shared demo project. They do not use
Serverless, CoreWeave, Modal, WBAF, or OpenAI. No result from one project is a
row in another Study.

## The governed execution path

Use Research and the registered campaign/experiment IDs below:

```text
catalog
→ pure Study preview
→ durable approval request
→ trusted operator approval of the exact preview digest and spend cap
→ immutable runtime, route, task, context, Skill, and integration preparation
→ campaign admission
→ local Harbor execution
→ reconciled outcome
```

The Research Agent may create a preview and request approval. It cannot issue
approval. The trusted operator uses `fugue research approve` for the exact
preview digest. The worker prepares and locks the accepted plan before campaign
admission; missing or drifted preparation fails before launch. A revised
project, model route, source tree, taskset, candidate, context artifact, or
runtime requires a new preview and approval.

Operator preflight, context preparation, and runtime preparation all bind the
experiment's declared evidence destination. Sequential lane work such as
harness → memory → harness must explicitly reactivate the harness destination;
legacy `.env` project values never select a lane.

The configs deliberately contain no alternate direct runner. Use:

- `claude-loop-skill-mcp-v1` / `claude-loop-skill-mcp`
- `real-harness-study-v1` / `real-harness-study`
- `real-memory-study-v1` / `real-memory-study`

## Claude loop engineering

Follow
[`examples/loop-engineering/wandb-evidence-loop/README.md`](../../examples/loop-engineering/wandb-evidence-loop/README.md)
to lock a reviewed real failure and materialize separate discovery and private
holdout tasksets.

The local `discovery` preset uses operator-imported, reviewed locks to
compare:

- production Skill + production MCP;
- proposed Skill + production MCP;
- production Skill + proposed MCP;
- proposed Skill + proposed MCP.

It is exactly eight cells: two tasks, four arms, one Claude Code attempt. After
the terminal discovery is exported, run the registered
`claude-loop-discovery-selection` analysis and freeze the intervention lock.
Only then preview the local `holdout` preset. The selection lock reduces the
holdout to production versus the selected arm on four independent tasks:
exactly eight more cells.

The source must be a repeated valid failure from
`mcp-main-vs-0-4-natural-maintainer-canary-v3`. The old wrong-project Study
and placeholder trace cannot be used. Both discovery and private holdout Task
Suites are locked before discovery results are inspected.
The selection lock binds the reviewed failure-lock digest, both Suite digests,
the complete four-arm candidate map, discovery snapshots, and the ordered
failure-lock → Suite-freeze → discovery → selection chronology.
The discovery campaign proposal carries those prefreeze inputs as governed
proposal metadata, and its declared discovery Suite must match the Task Suite
actually rendered into every row.

Discovery and holdout require separate approvals. A useful result requires
baseline reproduction, a paired deterministic gain, mechanism evidence for
the changed Skill or MCP, no critical holdout regression, reconciled native
Agent and Evaluation evidence, and the qualified source tree.

## Harness behavior

Campaign `real-harness-study-v1`, experiment `real-harness-study`, preset
`canary` use two immutable SWE-bench Verified cases:

- `sympy__sympy-13031`
- `astropy__astropy-13033`

Each case runs once through Hermes, OpenClaw, Claude Code, and Codex. The route
is fixed to W&B Inference `wandb/zai-org/GLM-5.2`; changing the model is outside
the campaign allowlist and its prepared route lock. Task commits, verifier
runtime, context (`none`), attempts, and local Docker policy are also fixed.
The separately approved preview is capped at $10.

Interpret this as task-specific harness behavior. Compare completion, verifier
outcome, tool trajectory, cost, latency, and evidence integrity by aligned
task with `real-harness-task-stratified`. Study Console presents the harness as
the varied factor while keeping runtime completion separate from task success.
Do not publish a universal harness ranking from two cases.

## Repository-memory intervention

Campaign `real-memory-study-v1`, experiment `real-memory-study`, preset
`canary` use two immutable SWE-bench Verified cases:

- `sympy__sympy-18199`
- `sphinx-doc__sphinx-9461`

The four arms are:

- baseline: no extra retrieval or policy;
- rag-dense: locked dense retrieval;
- policy-only: the registered inspect-and-verify prompt;
- combined: dense retrieval plus the prompt.

Claude Code, `anthropic/claude-sonnet-5`, task inputs, attempts, and local
Docker policy remain fixed. Fugue preparation builds and locks `rag-dense`
before admission. Dense embedding/index preparation and its offline semantic
probe must succeed. There is no BM25 substitution: a missing, corrupt, or
unqueryable dense artifact blocks the Study.
The separately approved preview is capped at $10.

Interpret outcome evidence separately from retrieval mechanism evidence.
Assignment to `rag-dense` is not proof of use; inspect the context calls and
opened source paths before claiming that memory affected an answer. The
`real-memory-factorial` analysis and Study view expose the exact
dense-retrieval × evidence-policy levels and report returned, opened, and used
evidence separately. Dense mode never computes or falls back to BM25; lexical
retrieval exists only in the separately named BM25 and hybrid systems.

## Before approving any preview

Confirm:

- the displayed project is the lane's exact project above;
- the resolved plan has exactly eight cells;
- the model and harness set match the campaign allowlist;
- every task commit and dataset digest is locked;
- local Docker is the only execution backend;
- runtime and route locks are ready;
- `rag-dense` is prepared for both memory-bearing candidates;
- private task truth and credentials are absent from Agent-visible artifacts;
- the approval cap covers only the displayed eight cells.

This repository change defines and tests the lanes only. It does not authorize
or execute paid cells.
