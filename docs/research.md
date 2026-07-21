# Fugue Research SDK

The research SDK packages Fugue as a governed laboratory for an external
research loop. The public model is deliberately small:

```text
Study → Experiment → Run → Result
```

A **Study** is durable research memory. An **Experiment** is one controlled,
hypothesis-bearing comparison. A **Run** is an admitted execution through the
existing Fugue campaign lifecycle. A **Result** is an immutable statement with
explicit scope, uncertainty, exclusions, and evidence references.

The outer loop still decides which question matters and what to try next.
Fugue validates that choice against a source-controlled campaign, locks its
inputs, enforces cost and concurrency policy, runs the canonical operator, and
reconciles the evidence. It does not declare a winner or invent a finding.

## Python

Install the transport dependencies when needed:

```bash
pip install 'fugue[research]'
```

The local client can start a recoverable background worker. `preview()` is
pure. A human operator must approve its exact digest outside the Agent-facing
client before `start()` can queue work.

```python
import os

from fugue.research import FugueResearchClient

with FugueResearchClient.local(repo_root, env_file=credentials_file) as fugue:
    study = fugue.studies.create(
        study_id="harness-sensitive-agents",
        campaign_id="agent-research",
        title="Harness-sensitive agent behavior",
        question="Which agent-loop components change outcomes?",
        idempotency_key="create-harness-study",
    )

    preview = study.experiments.preview(
        question="Does repository search help?",
        hypothesis="Search helps when Agents inspect and verify retrieved code.",
        task_suite=task_suite_draft,
        design={
            "stage_id": "discovery",
            "experiment_id": "retrieval-to-action",
            "model": "glm-5.2",
            "fixed_dimensions": ["model", "tasks", "runtime"],
            "varied_dimensions": ["search", "instructions", "harness"],
            "measured_dimensions": ["repair pass", "evidence use"],
            "harnesses": ["codex", "claude-code"],
            "n_attempts": 2,
            "n_concurrent": 1,
        },
    )

    print(preview.preview_digest, preview.estimated_cost_usd)
    # A human runs `fugue research approve ...` in a separate operator shell.
    experiment = study.experiments.start(
        preview,
        approval_digest=os.environ["FUGUE_APPROVAL_DIGEST"],
        idempotency_key="retrieval-study-001",
    )
    for event in experiment.watch():
        print(event.state, event.message)

    outcome = experiment.result()
    study.record(
        "The experiment completed with reconciled run evidence.",
        runs=outcome["run_refs"],
        expected_revision=study.revision,
        idempotency_key="record-retrieval-outcome",
    )
```

Notes are append-only and do not rewrite the Study brief. Promoted findings
must be recorded as `StudyResultV1` values with exact evidence references.
Corrections add a new result with `supersedes`; historical values remain
available.

## Recovery and identity

The Study store is SQLite in WAL mode under `.fugue/research.db`. Study events,
revisions, and experiment events are append-only. Mutations use caller-supplied
idempotency keys and Study updates can require an `expected_revision`.

The execution queue uses leases. Validation, locking, planning, and preparation
can resume after a process restart. Once Fugue has launched a run, recovery uses
the campaign operation ledger and the existing run identity. It never silently
launches the Agent cells a second time. Cancelled or interrupted runs are
finalized into terminal evidence before another run can be proposed.

Evidence remains in its authoritative system. Study records contain
content-addressed references to Fugue outcomes, normalized rows, analyses,
Weave traces, Agent conversations, and versioned resources rather than copies
of private trace data.

## HTTP and MCP

Run the typed HTTP API, authenticated Streamable HTTP MCP endpoint, and
resumable SSE event stream:

```bash
FUGUE_RESEARCH_API_KEY=... fugue research serve --repo-root .
```

REST is versioned under `/v1`; MCP is mounted at `/mcp/`. Both require the same
bearer token. Reconnect to
`GET /v1/experiments/{id}/events` with `Last-Event-ID` to resume from an event
cursor. Request bodies are bounded.

Run the MCP adapter over the same client and database:

```bash
fugue research mcp --repo-root .
```

MCP exposes only high-level Study and Experiment operations plus bounded Study
context, experiment status, and outcome resources. It does not expose the
operator, raw environment, shell commands, credentials, or experimental MCP
Tasks.

Use the `advance_research_cycle` MCP prompt for autoresearch-style handoffs. One
cycle may reconcile a terminal parent, record a sourced Result, and produce one
lineage-bound child preview. It always stops before approving or starting that
child, which keeps adaptive research useful without turning it into an unbounded
spend loop.

Approvals are deliberately absent from REST and MCP. After reviewing a preview,
the operator approves its exact digest and a hard spend cap from a trusted shell:

```bash
fugue research approve PREVIEW_DIGEST --max-usd 200 --max-cells 8
```

The worker checks the exact campaign reservation against this cap inside the
admission transaction. A stale preview, changed plan, expired approval, or cost
above the cap fails before an admission is recorded.

For the isolated control/worker deployment and portable Agent Skill, see
[`research-container.md`](research-container.md).
