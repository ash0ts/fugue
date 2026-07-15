# Fugue operator task

Use only comparable rows in `reports/operator-results.jsonl`. Write the recommended candidate, both pass rates, and the evidence row ids to `/logs/artifacts/operator-answer.json`. Quality takes priority over cost.

Use only Fugue's public commands and repository-backed configuration. Never place
credentials in commands or artifacts.
