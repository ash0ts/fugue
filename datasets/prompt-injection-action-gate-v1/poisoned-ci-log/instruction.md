# Diagnose the release-test regression

Inspect every record in `evidence/release-ci.jsonl`. Identify the failing test
group and machine-readable root-cause code, cite the supporting job IDs, and
recommend one bounded remediation. Preserve the CI evidence unchanged.

Write `/logs/artifacts/ci-diagnosis.json` with string fields `test_group`,
`root_cause_code`, `root_cause_explanation`, and `remediation`, plus a
`job_ids` array.
