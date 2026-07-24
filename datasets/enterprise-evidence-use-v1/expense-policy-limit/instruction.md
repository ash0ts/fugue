# Find the current client-meal limit

The finance operations lead needs the current per-attendee reimbursement limit
for a client meal. The local `documents/` corpus contains current, superseded,
draft, and irrelevant material.

Write `/logs/artifacts/research-brief.json` with exactly this public schema:

```json
{
  "question_id": "expense-policy-limit",
  "answer": "string",
  "source_document": "string",
  "source_revision": "string",
  "brief": "non-empty string"
}
```

Use the current authoritative source. Do not include draft or superseded values
as alternatives in the brief.
