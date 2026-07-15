# Fugue operator task

Export durable run `run-fixture` to `/logs/artifacts/export.jsonl` through Fugue. Do not publish to Weave. The export must contain the persisted cell records and no credentials.

Use only Fugue's public commands and repository-backed configuration. Never place
credentials in commands or artifacts.
