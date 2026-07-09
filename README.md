# fugue

**Run multiple agent harnesses on the same tasks, trace every run in W&B
Weave, and compare the results note for note.**

Fugue is a thin, opinionated layer on top of
[Harbor](https://github.com/laude-institute/harbor) for sandboxed trial
execution and [W&B Weave](https://wandb.ai/site/weave) for traces and
evaluation. W&B is always the trace plane. The model plane is provider
neutral: runs can bill through W&B Inference, OpenAI, or Anthropic by changing
one model string.

```text
wandb/zai-org/GLM-5.2
openai/gpt-5
anthropic/claude-sonnet-4-5
```

## Harnesses

| harness | adapter | model routing | Weave plugin |
|---|---|---|---|
| Hermes | `fugue.agents:FugueHermes` | OpenAI-compatible chat, direct or bridged | local `hermes-otel` checkout |
| OpenClaw | `fugue.agents:FugueOpenClaw` | OpenAI-compatible chat, direct or bridged | `weave-openclaw` |
| Claude Code | `fugue.agents:FugueClaudeCode` | native Anthropic Messages for `anthropic/...`, bridge otherwise | `weave-claude-code` |
| Codex CLI | `fugue.agents:FugueCodex` | native OpenAI Responses for `openai/...`, bridge otherwise | `weave-codex` |

The local LiteLLM bridge is generated under `.fugue/bridge/` by
`fugue bridge up`. It binds to `127.0.0.1:4000`; Harbor task containers reach
it at `http://host.docker.internal:4000`.

## Quick Start

```bash
cp .env.example .env
uv venv && uv pip install -e ".[dev]"

fugue preflight --model wandb/zai-org/GLM-5.2
fugue bridge up --model openai/gpt-5

scripts/smoke.sh --model wandb/zai-org/GLM-5.2
scripts/smoke.sh --model openai/gpt-5 codex hermes
scripts/smoke.sh --model anthropic/claude-sonnet-4-5 claude-code
```

Requirements: Docker Desktop, Harbor (`uv tool install harbor`), `jq`, and for
Hermes a local `hermes-otel` checkout (`HERMES_OTEL_CHECKOUT`, default
`~/Documents/GitHub/hermes-otel`).

## RepoMemBench Runner

```bash
fugue prepare --manifest datasets/pilot.yaml
fugue run --manifest datasets/pilot.yaml \
  --model openai/gpt-5 \
  --harnesses hermes,openclaw \
  --conditions none,openwiki \
  -k 1 -l 3
fugue export --jobs jobs/pilot --out reports/pilot.jsonl --to-weave
```

Model precedence is:

1. CLI `--model`
2. Manifest `model`
3. `FUGUE_MODEL`
4. `wandb/zai-org/GLM-5.2`

`fugue prepare` builds memory artifacts under `artifacts/memory/` and writes
`artifacts/lock.json`. `fugue run` invokes Harbor once per harness × condition
with task filters from the manifest. `fugue export` joins Harbor
`result.json`, `agent/fugue-meta.json`, and optional Weave span summaries.

## Operator UI

Install the web extra and start the local operator console:

```bash
uv pip install -e ".[web]"
fugue web --host 127.0.0.1 --port 8765
```

The UI is a W&B-style operator console with tabs for Overview, Setup, Run
matrix, Jobs, and Results. It shows key presence, selected provider/model,
bridge health, manifest shape, a matrix stave of harnesses × conditions, live
job logs, result summaries, and links into W&B/Weave. It never returns raw API
keys; status only reports whether each key is present.

## Environment

```bash
WANDB_API_KEY=          # Weave tracing; also model billing for wandb/...
WANDB_ENTITY=
WANDB_PROJECT=

OPENAI_API_KEY=        # model billing for openai/...
ANTHROPIC_API_KEY=     # model billing for anthropic/...

FUGUE_MODEL=wandb/zai-org/GLM-5.2
LITELLM_MASTER_KEY=sk-fugue-local
```

Optional base URL overrides:

```bash
WANDB_INFERENCE_BASE_URL=https://api.inference.wandb.ai/v1
OPENAI_BASE_URL=https://api.openai.com/v1
ANTHROPIC_BASE_URL=https://api.anthropic.com
```

## Layout

```text
fugue/
├── fugue/
│   ├── agents/          # Harbor adapters and Weave plugin wiring
│   ├── bench/           # prepare/run/export CLI
│   ├── bridge.py        # generated LiteLLM bridge config
│   ├── model_plane.py   # provider routing
│   └── web.py           # local operator UI
├── datasets/pilot.yaml
├── scripts/
├── tasks/
├── artifacts/           # gitignored prepared memory artifacts
├── jobs/                # gitignored Harbor and web jobs
├── reports/             # gitignored exports
└── .fugue/              # gitignored runtime bridge config
```
