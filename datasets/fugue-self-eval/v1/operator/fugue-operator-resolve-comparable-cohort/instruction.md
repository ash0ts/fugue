# Fugue operator task

Inspect `reports/operator-results.jsonl`. Select the rows suitable for comparing candidate-a with candidate-b under the same workload, model, and task coverage. Write sorted `row_ids` and excluded `candidate_ids` to `/logs/artifacts/operator-answer.json`.

Use only Fugue's public commands and repository-backed configuration. Never place
credentials in commands or artifacts.
