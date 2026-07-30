# Fugue 4A — Telemetry Is Not Evaluation: Designing a Loop That Can Learn

> **Fugue: Evals for the Agentic Software Factory · Part 4A**  
> A standalone architecture field note for loop-engineering and autoresearch
> teams. **Status:** concept. **Reading time:** about 10 minutes.

This article defines both loops, every authority, and the immutable state
transitions locally—you do not need to know Claude Code, Aria, Fugue, Weave,
or Study Console before reading.

The failure that produced it was a loop diagram with no authority boundary.
The same researcher could observe a weak result, rewrite the comparison,
approve more spend, retry the failed cells, and interpret the new outcome.
Every individual action was useful. The loop as a whole could grade its own
homework.

The claim this essay defends:

> A loop becomes engineering only when observations can produce controlled,
> externally evaluated, reversible changes.

If an unconstrained self-improvement loop reliably preserved evidence,
budget, diversity, and honest nulls under pressure, this claim would be
false and the architecture below would be ceremony. Our design assumes the
opposite: proposal, approval, execution, and evaluation should remain
separate authorities even when agents help with all four.

## Scope and terms

The **inner loop** is one Agent attempting one task. The **outer loop**
turns observations across attempts into a bounded hypothesis and a
reversible engineering decision. **Telemetry** records what happened.
**Evaluation** applies a declared procedure to decide what that evidence
supports. **Authority** determines who may propose, approve, execute, retry,
or accept.

The architecture does not make autonomous improvement safe by itself. It
makes authority, evidence gaps, failed proposals, and rollback points
inspectable.

## Two loops, different responsibilities

The inner loop is the Agent doing one task:

```text
observe task → reason → use tool → inspect result → edit or answer → stop
```

The outer loop changes the system that runs the inner loop:

```text
observe evidence → hypothesize → preregister → compare → evaluate → decide
```

Conflate them and you get recursion. An Agent that can change its tools or
evaluation during a task optimizes the measurement instead of the software.
An outer-loop researcher that edits a holdout after seeing the result turns
discovery into evidence without admitting the change.

```mermaid
flowchart TD
    subgraph Inner["Inner Agent loop"]
      IT["Task"] --> IA["Agent action"]
      IA --> IR["Tool/environment response"]
      IR --> IA
      IA --> IO["Attempt output"]
    end
    subgraph Outer["Outer engineering loop"]
      OE["Evidence"] --> OH["Hypothesis"]
      OH --> OP["Locked Study proposal"]
      OP --> OX["External execution and evaluation"]
      OX --> OD["Bounded decision"]
      OD --> OE
    end
    IO --> OE
    OD -. "changes a future candidate" .-> IT
```

The loops communicate through artifacts, not shared mutable intention. An
attempt output becomes evidence only after identity and reconciliation. A
decision changes a future candidate through a new reviewed artifact.

Lilian Weng’s survey of harness engineering describes exactly this
propose–evaluate–accept pattern and the risks of self-improving loops,
including evaluator weakness and homogenization
([Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/)). [@weng-harness]
The survey motivates the pattern; Fugue’s authority boundaries are our
engineering response. Dudycz’s OODA analysis supplies the operational test
we keep returning to: telemetry is observation—it has not yet oriented the
decision maker or authorized action. [@dudycz-ooda]

## The governed cycle

Our outer loop is:

```text
observe → hypothesize → preregister → preview → approve
→ execute → evaluate → interpret
```

Each transition creates or consumes a durable object.

| Stage       | Object                                       | Authority                |
| ----------- | -------------------------------------------- | ------------------------ |
| Observe     | Evidence references and Research note        | Researcher or loop engineer |
| Hypothesize | Bounded question and expected mechanism      | Researcher or loop engineer |
| Preregister | Registered comparison spec and Study brief   | Repository review        |
| Preview     | Pure expanded plan and digest                | Fugue comparison service |
| Approve     | Digest-bound cell/cost grant                 | Human operator           |
| Execute     | Immutable attempts and lifecycle events      | Local Harbor             |
| Evaluate    | Deterministic checks and calibrated judgment | External scorers         |
| Interpret   | Sourced Result and proposed follow-up        | Human maintainer         |

The cycle is deliberately not closed by an automatic “accept” arrow. A
release or code change still crosses the repository’s review and deployment
process, and a proposed follow-up stops at preview until a human approves
it.

Hamel’s error-analysis workflow argues for the same ordering from the eval
side: inspect concrete failures and define criteria with domain experts
before automating anything [@hamel-field-guide]. Automated graders then stay
what the auto-evals playbook says they should be—aids to triage and
measurement, never an authority that approves its own optimization target.
[@auto-evals]

## The division of labor

The flagship uses five roles with distinct jobs.

### Aria reads and explains

Aria is an optional read-only presentation shell. It can read safe Research
context, registered comparison metadata, Study state, and evidence-linked
Results, then explain what the immutable records support.

It cannot access private labels, request or issue approval, start execution,
change accepted candidates or tasks, alter execution policy, retry attempts,
select an intervention, or launch a follow-up. Removing Aria from the
flagship must not change candidate or execution semantics.

### Fugue locks and reconciles

Fugue owns candidate resolution, task and evidence identities, pure preview,
approval checking, admission, canonical execution, result export, and
evidence reconciliation. It does not decide which product question matters
or declare a winner.

### A human grants authority

The human sees the exact candidates, fixed controls, cell matrix, judge,
mechanism measures, runtime policy, cost cap, and preview digest. Approval
authorizes that object until a deadline. It does not authorize “whatever the
loop needs next.”

### Local Harbor executes

Harbor renders and executes the prepared native Agent job through local
Docker. The runtime cannot install candidate code during the attempt. Every
cell must publish a run-scoped policy receipt, preserve the private-label
boundary, clean up, and prove zero remaining run-scoped containers.

This supports the declared local execution policy. It does not certify
Serverless isolation; any future remote runtime remains a separate
qualification.

### Weave and Study Console make state inspectable

Weave stores native Agent conversations, nested tool spans, predictions, and
Evaluations. Study Console projects the canonical Study events a human needs
to understand design, approval, progress, results, and safe evidence links.

Study Console is not another database of truth. If it shows a different
approval digest or lifecycle state, the projection is wrong.

```mermaid
flowchart LR
    A["Optional Aria<br/>read-only projection"] --> F["Fugue<br/>lock/admit/reconcile"]
    H["Human<br/>approve exact digest"] --> F
    F --> S["Local Harbor<br/>execute prepared cells"]
    S --> W["Weave<br/>native evidence"]
    F --> E["Study event store"]
    W --> F
    E --> C["Study Console<br/>safe projection"]
    W --> C
    C --> A
```

## Why telemetry alone cannot drive the loop

Telemetry is broad and opportunistic. It can reveal that latency rose, a
tool was rarely selected, or a class of errors appeared. It is excellent for
finding questions. It usually lacks the controls to answer them.

A latency spike in traces may come from a changed model, provider load,
larger tasks, more tool use, or a runtime regression. A decline in one
tool’s usage may mean its description got worse, another tool got better, or
the task mix changed. A high judge score may reflect better answers or a new
rubric.

So the loop treats telemetry as discovery evidence. A researcher can say:

> In the locked failure, reviewed traces show that the current source was
> returned but not opened. I propose comparing the four registered Skill/MCP
> arms on the frozen discovery tasks.

It may not say:

> The candidate fixed the issue.

The second sentence requires the Study.

## Immutable Study state

A controlled Study progresses monotonically:

```mermaid
stateDiagram-v2
    [*] --> Draft
    Draft --> Previewed: pure resolution
    Previewed --> ApprovalRequested
    ApprovalRequested --> Approved: human digest grant
    Approved --> Admitted: caps reserved
    Admitted --> Running
    Running --> Reconciling
    Reconciling --> Completed
    Reconciling --> Incomplete
    Running --> Cancelled
    Completed --> [*]
    Incomplete --> [*]
    Cancelled --> [*]
```

No transition returns to Draft. A changed plan creates another Study. A
cancelled or interrupted run is finalized into terminal evidence before a
successor is proposed.

Events are append-only. A correction does not rewrite an earlier finding; it
adds a Result with a `supersedes` reference. Research notes preserve failed
ideas and nulls so the loop does not keep rediscovering them.

Append-only state also makes recovery tractable. The execution queue leases
work; if the control process restarts after admission, it recovers the
existing operation and run identity instead of launching a duplicate set of
Agent cells.

## One service behind every interface

An optional Aria shell reaches only the safe read surface. An operator may
use the CLI. Study Console uses typed HTTP and an event stream. Python clients
support orchestration. All of them consume the same comparison and Research
services, which prevents a presentation path from acquiring an accidental
capability the human preview does not show.

The registered-comparison allowlist binds a stable ID to a
repository-relative path and exact spec digest, so an MCP caller cannot
submit an arbitrary filesystem path. The public preview omits private-label
content and digests that would reveal it, while keeping the accepted
comparison identity.

Approval issuance and execution start are absent from the read-only shell. A
trusted operator command is the authority:

```bash
uv run fugue research approve PREVIEW_DIGEST \
  --max-usd 20 \
  --max-cells 8
```

The admission transaction checks the exact reservation against those caps. A
stale preview, expired grant, or increased cost fails before work is
recorded.

## Learning without Goodharting

A loop that optimizes repeatedly against one evaluator will find its blind
spots. The dramatic version is reward hacking. The common versions are
softer: proposals drift stylistically toward what the judge rewards, tasks
narrow toward behaviors that already score well, every agent converges on
the harness with the most legible traces, negative cases get reclassified as
infrastructure, retries hide instability, and the holdout grows familiar
through repeated trace review.

Our brakes are structural rather than exhortative. The candidate cannot
change its own grader, and the Agent never receives expected answers. Judge
calibration blocks the cohort on critical false passes. Discovery and
holdout stay separate, so tuning creates a new Study identity instead of
quietly improving the old one. Negative controls—easy cases and known
failures—expose scorer drift. Independent ledgers keep a mechanism gain from
concealing a task regression. Human approval keeps adaptive spend and scope
visible. And append-only memory makes nulls and rejected proposals durable,
so the loop cannot forget evidence that resists its preferred story.

None of this solves Goodhart’s law. It makes the pressure observable enough
to govern. In OODA terms: the loop may observe and orient continuously, but
deciding and acting stay behind authority the optimizer cannot reach.
[@dudycz-ooda]

## Reversibility is part of the hypothesis

An engineering loop should state not only what success looks like but what
can be undone.

For a code candidate, the rollback point is the parent tree and any
migration boundary. For an MCP release, it is the exact baseline integration
lock. For a runtime, it is the previous immutable image digest. For a Study,
rollback does not delete evidence; it stops future admission and records the
terminal reason. For a published Result, correction means a new record with
`supersedes`, never an edit that makes the earlier decision disappear.

Rollback goes in the proposal itself:

```yaml
rollback:
  trigger:
    - critical_evidence_honesty_regression
    - duplicate_execution
    - credential_or_private_label_exposure
    - unreconciled_harbor_policy_or_cleanup
  code_boundary: "<parent tree>"
  candidate_boundary: "<baseline fingerprint>"
  runtime_boundary: "<prior image digest>"
  evidence_action: "preserve and mark ineligible"
```

This makes a hypothesis operational. “Try the new release” becomes “admit
these cells; stop on these conditions; preserve these facts; return to this
known boundary.” Rollback can still be hard when data schemas or external
systems change—naming the boundary early reveals that risk before approval.

## Failure drills before autonomy

Happy-path tests are insufficient for a durable loop. We rehearse failures
that target authority and recovery:

- submit the same start request twice and verify exactly one execution;
- restart the worker after admission and recover the same run identity;
- expire approval between preview and start;
- change one task byte after approval and verify digest rejection;
- make the runtime image unavailable;
- interrupt an Agent after its Weave prediction opens;
- withhold a required Evaluation record;
- fail local Harbor cleanup and verify the result remains ineligible;
- propose a follow-up from an incomplete parent and verify it stops before
  approval.

A drill succeeds when state remains explainable, not when the system somehow
produces a score. Recovery must preserve the original coordinates and
evidence; a compensating retry with a new attempt identity may be allowed in
another Study, but it cannot overwrite the interrupted attempt.

The drills also clarify who owns remediation. Fugue can refuse
interpretation and reconcile known state. The local runtime operator may need
to clean up an orphan. The human may need to issue a new approval. Aria can
describe the blocker from the safe projection. None of them may quietly
impersonate another authority to make the loop appear self-healing.

## Diversity is an engineering asset

Self-improvement loops converge because the evaluator prefers familiar
outputs. The proposal that looks like previous winners is easier to approve,
and the harness that emits the most legible traces receives more investment.
That can eliminate useful alternatives before the task distribution changes.

We preserve diversity by reporting harnesses separately, retaining rejected
and null proposals, and requiring a stated reason when a candidate family is
removed. Diversity is not itself a score—expensive weak candidates should
not run forever—but it is an option value that stays visible in the Research
record.

Where two native harnesses reverse, the loop should ask what interaction
caused the difference rather than collapse to the pooled winner. Where two
task-authoring approaches disagree, the holdout can preserve both strata.
Where all candidates saturate a regression suite, keep it as a gate and
build a separate capability suite.

## The artifact: one outer-loop handoff

A practical handoff is a small, structured record:

```yaml
observation:
  evidence:
    - "weave:///PROJECT/call/CALL_ID"
    - "wandb:///ENTITY/PROJECT/run/RUN_ID"
  pattern: "Partial Evaluation evidence preceded unsupported completeness."
  limitations: "Two reviewed traces; discovery only."

hypothesis:
  question: "Does a reviewed Skill/MCP intervention improve source use?"
  expected_mechanism:
    - "current source opened before a supported claim"
    - "bounded MCP reads"

proposal:
  registered_comparison: "claude-loop-skill-mcp-v1"
  expected_spec_digest: "<sha256>"
  requested_cells: 8
  requested_max_usd: "<reviewed cap>"

authority:
  preview_digest: "<generated by Fugue>"
  approval: null

stop_condition:
  - "critical evidence-honesty regression"
  - "runtime or evidence identity mismatch"
  - "incomplete required reconciliation"
```

Claude Code or a human researcher may create the observation and candidate
source. Fugue supplies the preview digest. Only the operator fills approval.
Aria, if connected, may read the resulting safe projection but does not write
this handoff. No free-form Agent instruction can override the stop
conditions.

## Accept, reject, or learn?

An outer loop needs more vocabulary than “winner.”

- **Accept for release** means all go gates pass and the bounded result
  supports the specific maintenance decision.
- **Reject** means a predeclared critical regression or policy violation
  makes the candidate unsuitable in scope.
- **No decision** means evidence is incomplete, the result reverses in a way
  the claim cannot absorb, or uncertainty remains too high.
- **Learn and follow up** means a null or mechanism observation motivates a
  new preregistered question.

The fourth outcome is why nulls must stay durable. A treatment that improves
retrieval without task outcomes can generate a focused source-use study. A
harness reversal can generate a protocol compatibility investigation.
Neither justifies rewriting the current result.

The implemented Research boundary behind this design is reviewable in the
public Fugue stack: proposal and inspection are separated from approval,
execution, and result acceptance. [@fugue-research]

## Try this in 15 minutes

Draw your current improvement loop and put a name beside proposer, approver,
executor, evaluator, retry authority, and rollback owner. If one identity
owns all six, choose one irreversible transition and move its authority
outside the loop.

Now trace one failed proposal. If the null, rejection, or retry is not
durable, the loop has no memory of evidence that resists its preferred
story.

## When an outer loop is unnecessary or insufficient

A known defect with a clear owner should enter the normal engineering queue,
not wait for an autoresearch loop. The governed outer loop earns its cost
when evidence must produce a new bounded hypothesis. It is insufficient when
the proposer can approve its own work, the evaluator changes after seeing
the holdout, or rollback exists only as a diagram.

## What this does not show

Authority separation does not make the human correct. A human can approve a
bad design or ignore a limitation. Append-only events can faithfully
preserve poor evidence. External judges can share the candidate’s biases.

Local Harbor execution does not prove a runtime contains the intended assets;
the runtime lock and probes do that. A local receipt also does not prove
Serverless isolation. Weave traces do not prove source use; the evidence
relation and evaluation do that. Study Console does not create evidence by
rendering a card.

Nor does this architecture establish that an autonomous outer loop improves
software. It describes a bounded loop in which proposals can be evaluated
without silently changing authority. The real flagship run remains
preregistered.

Governance also does not need to make every action slow. Read-only
observation, pure preview, and deterministic reconciliation can stay
automatic. The human boundary is reserved for changed scope, spend, release
authority, and claims—the acts whose consequences the system should not
infer from convenience.

## Next: make the loop visible end to end

An architecture diagram can hide every inconvenient detail: placeholder
evidence, replayed attempts, a button that does not bind to the digest, a
“remote” cell that never leaves the laptop.

In **Fugue 4B**, we turn the governed cycle into a demo runbook. Claude Code
diagnoses a reviewed failure and authors reviewed Skill/MCP candidates;
Fugue crosses a visible human approval, runs 8-cell discovery and 8-cell
holdout through local Harbor, opens aligned evidence, and qualifies or
rejects the source tree. Aria remains an optional read-only shell. If any
link is missing, the demo says so.

## References

Complete source metadata and numbered citations are generated from this
article package’s `sources.json`.
