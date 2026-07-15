# Fugue operator task

Inspect durable run `run-fixture`. Write `/logs/artifacts/operator-answer.json` with its final `status`, the failed `cell_id`, `harness`, `variant`, and exact `error`.

Use only Fugue's public commands and repository-backed configuration. Never place
credentials in commands or artifacts.
