# Skill-change study

This local study compares Claude Code with and without one reviewed Skill. The
Skill is a real versioned candidate asset; its assignment and observed use are
mechanism evidence, while deterministic task correctness remains the outcome.
The matrix contains eight logical cells.

Create `.env` with mode `0600`:

```bash
install -m 600 .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env`. Then run:

```bash
fugue doctor --require local-runner --model anthropic/claude-sonnet-5 --env-file .env
fugue check comparison.yaml --env-file .env
fugue compare comparison.yaml --prepare --env-file .env
fugue compare comparison.yaml --preview --env-file .env --json > preview.json
fugue approve PREVIEW_DIGEST --max-cells 8 --max-usd 10
fugue compare comparison.yaml --run --approval APPROVAL_DIGEST --env-file .env
fugue result latest
```

Private labels remain host-only. No W&B project or credential is required.
