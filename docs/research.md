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

The local client starts a recoverable background worker. `preview()` is pure;
`start()` is the explicit spend boundary and accepts only the exact signed
preview.

```python
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

    experiment = study.experiments.start(
        preview,
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

Run the typed HTTP API and resumable SSE event stream:

```bash
FUGUE_RESEARCH_API_KEY=... fugue research serve --repo-root .
```

The API is versioned under `/v1`. Reconnect to
`GET /v1/experiments/{id}/events` with `Last-Event-ID` to resume from an event
cursor. Authentication uses a bearer token when `FUGUE_RESEARCH_API_KEY` is
set, and request bodies are bounded.

Run the MCP adapter over the same client and database:

```bash
fugue research mcp --repo-root .
```

MCP exposes only high-level Study and Experiment operations plus bounded Study
context, experiment status, and outcome resources. It does not expose the
operator, raw environment, shell commands, credentials, or experimental MCP
Tasks.
