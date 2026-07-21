# One governed autoresearch cycle

This example shows the north-star Fugue workflow for an external research Agent:

```text
bounded traces → observation → hypothesis → exact preview → human approval
              → Harbor run → sourced Result → one child preview → stop
```

The Agent owns the research judgment. Fugue owns policy, immutable inputs, cost
admission, isolated execution, evidence reconciliation, and durable lineage. The
cycle is intentionally bounded: it may create one next preview, but it cannot approve
or start that child.

## Operator setup

Register a Weave trace source in an operator-controlled YAML file. The Agent sees
only the source ID, allowed fields and filters, redaction policy, and digest.

```yaml
version: 1
sources:
  - id: production-agent
    adapter: weave
    project: entity/project
    allowed_fields:
      [status, operation, errors, tools, latency, tokens, cost, conversation]
    allowed_filters: [run_id, status, harness, model]
```

Point Compose at that file before starting Fugue:

```bash
export FUGUE_TRACE_SOURCES_FILE=/absolute/path/to/trace-sources.yaml

uv run --frozen fugue research bootstrap \
  --repo-root . \
  --env-file /path/to/operator-credentials.env

docker compose --env-file .fugue/compose.env \
  -f compose.research.yaml up --build -d
```

Export and configure the portable skill as shown in the
[`retrieval-to-action-canary` runbook](../retrieval-to-action-canary/README.md).
Then copy this example's `agent-prompt.md` into the Agent workspace and run it with a
read-only Agent.

## What the audience should see

The first Agent reads a bounded trace cohort and states one observation separately
from its hypothesis. It previews the eight-cell retrieval-to-action qualification and
stops with an exact digest and cost. A trusted operator approves that digest. After
the run, the Agent records a limited Result and previews one child canary carrying
the exact parent experiment, parent outcome, and decision rationale.

A fresh Agent can then continue from durable state rather than chat history:

```bash
codex -s read-only -C "$FUGUE_DEMO_DIR" \
  '$optimize-agent-with-fugue Advance exactly one bounded research cycle for Study agent-optimization-demo. Read the Study and experiment lineage from Fugue, reconcile any terminal parent before proposing work, and stop after one eligible child preview or an explicit reason not to run another experiment.'
```

The compelling part is not autonomous spending. It is that a different Agent can
recover the question, evidence, result, and branch rationale, then make the next
research decision inside the same controls.
