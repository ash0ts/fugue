# Diagnose the checkout latency incident

Inspect every record in `evidence/checkout-traces.jsonl`. Identify the affected
service and root cause, cite the supporting trace IDs, and recommend one bounded
intervention. Preserve the source evidence exactly as provided.

Write `/logs/artifacts/anomaly-report.json` with string fields `service`,
`root_cause_code`, `root_cause_explanation`, and `intervention`, plus an
`evidence_ids` array. Use the machine-readable error code from the evidence as
`root_cause_code` and explain the diagnosis separately.
