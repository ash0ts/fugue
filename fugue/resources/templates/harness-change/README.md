# Harness-configuration-change study

This local study holds the Claude Code harness fixed and compares only its
locked `max_turns` configuration. It does **not** rank Claude Code against
another harness. The matrix contains eight logical cells.

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

Observed completion and tool use are mechanism/operational evidence. Exact
task answers remain the outcome. No W&B project or credential is required.
