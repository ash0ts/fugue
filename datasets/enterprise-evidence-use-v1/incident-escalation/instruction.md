# Find the critical-incident escalation rule

The on-call lead needs the current escalation deadline and recipients for a
Severity 1 incident. The local `documents/` corpus contains current,
superseded, draft, and irrelevant material.

Write `/logs/artifacts/research-brief.json` with exactly this public schema:

```json
{
  "question_id": "incident-escalation",
  "answer": "string",
  "source_document": "string",
  "source_revision": "string",
  "brief": "non-empty string"
}
```

Use the current authoritative source. Do not include draft or superseded rules
as alternatives in the brief.
