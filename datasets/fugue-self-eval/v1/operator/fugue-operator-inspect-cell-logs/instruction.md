# Fugue operator task

Inspect the failed cell log in `run-fixture`. Write `/logs/artifacts/operator-answer.json` with `cell_id`, `failure_owner` set to `provider`, and a concise `cause` set to `W&B Inference quota exhausted`.

Use only Fugue's public commands and repository-backed configuration. Never place
credentials in commands or artifacts.
