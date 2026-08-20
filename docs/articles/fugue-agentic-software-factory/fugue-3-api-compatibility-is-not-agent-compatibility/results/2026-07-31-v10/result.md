# MCP V10 reviewed result

This appendix records the public-safe outcome of
`mcp-main-vs-0-4-tool-surface-confirmation-v10`. It is not a reconstruction of
private task rows. The authoritative public projection is bound by result
digest `e062f5b392a36d9ebd97adc3ab58b6e253cdd9dd943381342d51d76303bbcf38`.

- Baseline: W&B MCP `main` at `53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0`.
- Candidate: W&B MCP 0.4 at `5c6cc1c9a1079296daf6613ea6d12daebdd8bcba`.
- Matrix: four tasks, two revisions, two attempts; 16 of 16 cells terminal.
- Observed cost: `$7.331252`.
- Evidence integrity: 16 attempt records and 80 of 80 required links; no
  credential leak, private-label leak, duplicate attempt, or orphaned runtime.
- Deterministic outcome: candidate passed 1 of 8 required rows; seven critical
  failures.
- Aligned pairs: zero improved, one regressed, seven unchanged.
- Mechanism: Evaluation reconciliation and bounded inventory reads improved;
  exact-history behavior did not improve and one answer became factually worse.
- Judge: advisory because human calibration remained incomplete.
- Decision: behavioral verdict `REGRESSED`; release decision `HOLD`.

The detailed Atlas record is summary-only because the exact public V3 task-row
artifact was not available in this checkout. No row was inferred from aggregate
counts. The bounded claim is about this exact matrix, not an individual 0.4
feature or all W&B MCP usage.
