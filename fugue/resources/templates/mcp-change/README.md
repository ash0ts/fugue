# MCP-change study

This local study compares two locked configurations of a tiny read-only MCP
server. The server uses only Python's standard library, is copied into every
task image during preparation, and cannot access the network. The matrix has
eight logical cells.

Copy `.env.example` to `.env`, add `ANTHROPIC_API_KEY`, then run:

```bash
fugue doctor
fugue check comparison.yaml --env-file .env
fugue compare comparison.yaml --prepare --env-file .env
fugue compare comparison.yaml --preview --env-file .env --json > preview.json
fugue approve PREVIEW_DIGEST --max-cells 8 --max-usd 10
fugue compare comparison.yaml --run --approval APPROVAL_DIGEST --env-file .env
fugue result latest
```

Tool registration and invocation are mechanism evidence. The deterministic
answers remain the outcome. No W&B project or credential is required.
