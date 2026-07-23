# Research record projection

Fugue keeps execution authority and the durable source record. A console is a
projection:

```text
Research → controlled Study → admitted Run → evaluation → sourced Result
```

Every persisted lifecycle change appends a `ResearchLogEventV1` to the same
SQLite transaction. Delivery happens later through a `ResearchRecordSink`, so a
missing visualization cannot change a Run, retry a cell, or alter a Result.

Configure either sink:

```bash
export FUGUE_RESEARCH_RECORD_JSONL=/absolute/path/research-events.jsonl

export FUGUE_RESEARCH_RECORD_HTTP_URL=http://127.0.0.1:3000/api/research-log-events
export FUGUE_RESEARCH_RECORD_TOKEN_FILE=/absolute/path/ingest-token
```

The HTTP adapter sends the producer event ID as `Idempotency-Key`. Receivers
must return the prior result for an identical replay and reject a replay whose
event digest differs.

Pure preview is still side-effect free. An Agent makes a preview visible by
calling `fugue_study_request_approval`; that durable approval request is the
first publication event containing the preview reference and reserved cost.
Operator approval remains outside the Agent interface.

Publication payloads contain lifecycle state, aggregate progress, separately
labeled reserved and observed cost, typed relationships, and immutable evidence
references. They never contain task prompts, trace bodies, credentials, private
criteria, expected values, gold paths, or hidden reasoning. The underlying
Weave, W&B Run, artifact, analysis, and source systems remain authoritative.

Inspect delivery health without changing it:

```bash
curl -H "Authorization: Bearer $FUGUE_RESEARCH_TOKEN" \
  http://127.0.0.1:8787/v1/research-publications
```
