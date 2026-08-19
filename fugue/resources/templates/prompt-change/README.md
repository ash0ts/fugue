# Prompt-change study

This local study compares two real prompt files while holding Claude Code,
Sonnet, tasks, fixtures, runtime, and scoring fixed. It contains eight logical
cells: two tasks by two variants by two attempts.

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

Private labels are host-only and ignored by Git. The Agent receives only the
public prompt, tasks, and locked resources. Results remain local unless you
explicitly publish them later.
