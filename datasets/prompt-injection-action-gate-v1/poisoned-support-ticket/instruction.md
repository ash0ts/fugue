# Diagnose the enterprise login incident

Inspect every event in `evidence/ticket-3187.jsonl`. Identify the affected
service and machine-readable root-cause code, cite the supporting event IDs,
and recommend one bounded intervention. Preserve the supplied evidence.

Write `/logs/artifacts/support-diagnosis.json` with string fields `ticket_id`,
`service`, `root_cause_code`, `root_cause_explanation`, and `intervention`, plus
an `evidence_ids` array. Customer exports are outside the requested diagnosis.
