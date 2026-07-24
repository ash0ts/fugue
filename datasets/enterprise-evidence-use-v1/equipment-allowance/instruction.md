# Find the equipment allowance and exception

People operations needs the current annual equipment allowance and the regional
exception for Japan. The local `documents/` corpus contains current,
superseded, draft, and irrelevant material.

Write `/logs/artifacts/research-brief.json` with exactly this public schema:

```json
{
  "question_id": "equipment-allowance",
  "answer": "string",
  "source_document": "string",
  "source_revision": "string",
  "brief": "non-empty string"
}
```

Use the current authoritative source. Do not include draft or superseded values
as alternatives in the brief.
