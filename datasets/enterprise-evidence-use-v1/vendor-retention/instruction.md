# Find the current vendor retention requirement

The procurement lead needs the current deletion timeline after a vendor
contract ends. The local `documents/` corpus contains current, superseded,
draft, and irrelevant material.

Write `/logs/artifacts/research-brief.json` with exactly this public schema:

```json
{
  "question_id": "vendor-retention",
  "answer": "string",
  "source_document": "string",
  "source_revision": "string",
  "brief": "non-empty string"
}
```

Use the current authoritative source. Do not include draft or superseded
timelines as alternatives in the brief.
