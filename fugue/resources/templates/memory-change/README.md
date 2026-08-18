# Memory-change study

This local study compares no prepared context with Fugue's deterministic
`agentsmd` repository-map context. Both arms use the same Claude Code harness;
this is a context intervention, not a harness ranking. The matrix contains
eight logical cells.

Create a private credential file, add `ANTHROPIC_API_KEY`, then run:

```bash
install -m 600 .env.example .env
```

```bash
fugue doctor --require local-runner --model anthropic/claude-sonnet-5 --env-file .env
fugue check comparison.yaml --env-file .env
fugue compare comparison.yaml --prepare --env-file .env
fugue compare comparison.yaml --preview --env-file .env --json > preview.json
fugue approve PREVIEW_DIGEST --max-cells 8 --max-usd 10
fugue compare comparison.yaml --run --approval APPROVAL_DIGEST --env-file .env
fugue result latest
```

Context registration and use are mechanism evidence; exact task answers remain
the outcome. No W&B project or credential is required.
