# Governed research campaigns

Fugue's campaign layer is the stable in-process boundary for an outer research
loop. It lets that loop discover approved components, propose an experiment,
prepare and admit an immutable plan, launch it through `OperatorService`, and
receive a reconciled outcome without acquiring authority to write code or
configuration.

The registered-experiment lifecycle is:

```text
catalog → proposal → plan receipt → prepared plan → admission → run → outcome
```

When the campaign policy enables governed task authoring, the same service
extends that lifecycle without adding another executor:

```text
catalog → task draft → task preview → task lock → proposal → plan → admission
        → run → evaluation → meta-analysis
```

See [Governed task authoring](task-authoring.md) for the strict task, scenario,
interaction, criteria, scoring, and analysis contracts.

Campaign authority is a strict, source-controlled `ResearchCampaignSpecV1` in
`configs/fugue/campaigns`. The policy allowlists registered components and sets
stage, cell, attempt, concurrency, trace, evidence, and spend limits. Once a
campaign has admitted a plan, its persisted policy digest cannot change.

The Python entry point is `fugue.bench.campaigns.CampaignService`:

```python
catalog = service.catalog("my-campaign")
plan = service.preview(proposal)  # pure: creates no runtime state
prepared = service.prepare(plan, "prepare-operation")
admission = service.admit(prepared, "admit-operation")
status = service.launch(admission, "launch-operation")
outcome = service.finalize(status.runs[0]["run_id"], "finalize-operation")
```

Authored suites add four methods to the same boundary:

```python
preview = service.preview_task_suite("my-campaign", catalog.catalog_digest, draft)
suite = service.lock_task_suite(preview, "lock-suite-operation")

# proposal.task_suite_digest = suite.suite_digest; then use preview/prepare/
# admit/launch/finalize exactly as above.
evaluation = service.score_task_suite(
    run_id, suite.suite_digest, scoring_revision, "score-operation"
)
analysis = service.analyze_task_study(
    run_id,
    "task-study-v1",
    "analyze-operation",
    evaluation_digest=evaluation.evaluation_digest,
)
```

Every mutating call requires a caller-supplied operation ID. Repeating an ID
with identical input returns the original artifact or run; reusing it with
different input fails. This makes transport retries safe without retrying an
Agent cell. A new execution always needs a new proposal and attempt identity.

The outer loop decides what registered experiment to propose next and records
that rationale with the proposal. Fugue decides whether the proposal is
allowed and whether its eventual evidence is eligible. Performance remains an
observation: stage progression requires complete locks, terminal rows, exact
Agent conversation/root reconciliation, valid route receipts, and complete
cost accounting—not a particular pass count.

Outcome packets contain only policy-scoped structured rows and immutable trace
identities. They never copy raw conversations, commands, environment values,
expected paths, or gold data. Campaigns cannot publish to the public Atlas.

`fugue.research` packages this lifecycle as the higher-level
`Research → controlled Study → admitted Run → evaluation → sourced Result`
interface for outer loops. Its Python,
HTTP/SSE, and MCP surfaces all wrap `CampaignService`; none calls operator
internals or introduces another executor. See `docs/research.md`.
