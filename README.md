# fugue

**Run any set of agent harnesses on the same tasks — one model plane, one
trace plane — and compare them note for note.**

A *fugue* states one subject and lets several voices develop it in turn;
that is exactly what this framework does: one task suite, many agent
harnesses, every voice fully traced. (And a *fugue state* is memory loss —
fitting, since the flagship study measures what repository memory is worth
to a coding agent.)

Fugue is a thin, opinionated layer on top of
[Harbor](https://github.com/laude-institute/harbor) (sandboxed, parallel
trial execution) and [W&B Weave](https://wandb.ai/site/weave)
(observability + evaluation):

- **One model plane.** Every harness calls the *same* model through
  W&B Inference — differences between harnesses can't hide in different
  providers, and everything bills to one key.
- **One trace plane.** Each harness's Weave plugin is installed inside the
  task container, so every trial produces a full trace (LLM calls, tool
  calls, tokens) in a single Weave project, stamped with a join key back to
  the Harbor trial.
- **Hermetic trials.** Harbor builds a fresh container per trial, installs
  the harness from scratch, runs the task, and scores it with the task's
  verifier. Nothing installs on the host.

Four harnesses are wired today; adding one means writing a single adapter
class (see [Extending](#extending-adding-a-harness)):

| harness | adapter | model wiring | Weave plugin |
|---|---|---|---|
| Hermes | `fugue.agents.wandb_inference:WandbHermes` | direct (custom `wandb` provider) | local [hermes-otel](https://github.com/briancaffey/hermes-otel) checkout |
| OpenClaw | `fugue.agents.wandb_inference:WandbOpenClaw` | direct (`openai` provider + baseUrl) | `weave-openclaw` (npm) |
| Claude Code | `fugue.agents.wandb_inference:WandbClaudeCode` | LiteLLM bridge (`/v1/messages`) | `weave-claude-code` (npm) |
| Codex CLI | `fugue.agents.wandb_inference:WandbCodex` | LiteLLM bridge (`/v1/responses`) | `weave-codex` (npm) |

## Flagship study: is a repo wiki worth it? (RepoMemBench)

The default experiment this repo exists to run: **does giving a coding
agent a "wiki" of the repository make it better, and does the answer depend
on the harness?** Memory conditions:

- **OpenWiki** — generated wiki committed alongside the repo
- **DeepWiki MCP** — wiki served as MCP tools
- **semantic search** — embedding index over the repo
- **AGENTS.md** — single curated context file
- **none** — control ("fugue state")

Each (repo × question × condition) is a Harbor task dir; the condition is
stamped on every trace via `FUGUE_CONDITION`, so the whole harness ×
condition grid can be sliced in one Weave project: pass rate, cost,
latency, wiki-tool reads, failure taxonomy.

The task matrix + fixture repos are being built (see [Status](#status));
the smoke suite below runs the full pipeline end to end today.

## Quick start

```bash
cp .env.example .env          # fill in WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT
scripts/preflight.sh          # validates endpoint, bridge, tool-calling, imports

# 1 trivial task per harness (~2 min each, fractions of a cent):
scripts/smoke.sh                        # all four
scripts/smoke.sh codex claude-code      # subset

# ad-hoc harbor runs against any task dir:
export PYTHONPATH=$(pwd)
export FUGUE_CONDITION=openwiki         # experimental condition (default: baseline)
harbor run -p tasks/smoke/bridge-check \
  -a fugue.agents.wandb_inference:WandbHermes \
  -m wandb/zai-org/GLM-5.2 -o jobs/dev -n 5
```

Requirements: Docker (Desktop), [harbor](https://github.com/laude-institute/harbor)
(`uv tool install harbor`), `jq`, and for the Hermes adapter a local
[hermes-otel](https://github.com/briancaffey/hermes-otel) checkout
(`HERMES_OTEL_CHECKOUT`, default `~/Documents/GitHub/hermes-otel`).

## Model plane: one endpoint, one key

W&B Inference (`https://api.inference.wandb.ai/v1`) is OpenAI
chat-completions only. It serves no Anthropic `/v1/messages` and no OpenAI
`/v1/responses` (both 404). Harnesses reach it two ways:

```
hermes    ──(chat completions, custom "wandb" provider)──────────┐
openclaw  ──(chat completions, openai provider + baseUrl)────────┤──> api.inference.wandb.ai/v1
claude    ──(anthropic /v1/messages)──> litellm bridge ──────────┤        (bills to WANDB_API_KEY)
codex     ──(openai /v1/responses)────> (same bridge) ───────────┘
```

The bridge is one LiteLLM container (`proxy/docker-compose.yaml`) bound to
`127.0.0.1:4000`; task containers reach it at `host.docker.internal:4000`.
It translates both non-chat protocols into chat completions. The provider
prefix in `proxy/litellm.config.yaml` must stay `nebius/*` — see the comment
there for the tested failure modes of the alternatives.

All four adapters take the same model string: `-m wandb/zai-org/GLM-5.2`
(default when `-m` is omitted; override with `FUGUE_MODEL`).

## Trace plane: one Weave project, two span stores

All traces land in `WANDB_ENTITY/WANDB_PROJECT`, but on two different
backend surfaces:

| harness | span store | join key |
|---|---|---|
| Hermes | **Calls** (`/otel/v1/traces`, query `calls/stream_query`) | `otel_span.resource.attributes.fugue.run_key` |
| OpenClaw | **Agents** (`/agents/otel/v1/traces`, query `agents/spans/query`) | `agent_name` = run key |
| Claude Code | **Agents** | `agent_name` = run key |
| Codex CLI | **Agents** | `agent_name` = run key (patched; upstream hardcodes `codex`) |

The run key is the Harbor trial dir name (e.g. `bridge-check__d5JgHT9`),
also recorded host-side in each trial's `agent/fugue-meta.json` together
with the harness, model, condition, timestamps, and harness session ids.

Hard-won wiring notes (all verified empirically; see the adapter docstrings
in `fugue/agents/wandb_inference.py` for detail):

- **Hermes**: plugin enablement is persisted as a `plugins:` block *inside*
  `~/.hermes/config.yaml` — any later config overwrite silently disables the
  plugin (no error, no debug.log, no spans). The adapter bakes the block
  into its generated config and captures `hermes plugins list` plus the
  plugin debug log as trial artifacts.
- **OpenClaw**: plugin services only start in **gateway mode** (`--local`
  never initializes the exporter), so the adapter runs a loopback gateway
  per trial. OpenClaw's managed npm overrides force `@opentelemetry/core@2.x`
  while published weave SDKs (<=0.16.x) use the OTel 1.x trace stack and
  crash at plugin load; the adapter overrides the plugin's `weave` dep with
  `vendor/weave-node-sdk.tgz`, built from the OTel-2.x migration branch
  ([wandb/weave#7541](https://github.com/wandb/weave/pull/7541)). Known
  plugin limitation: gateway chat spans carry zero token counts/messages;
  tokens are still captured in the Harbor trajectory from the CLI envelope.
- **Claude Code**: the adapter patches the installed npm package's
  marketplace.json (`source: github -> ./`) so install works without
  git/ssh, and its transcript-path validation to accept Harbor's
  `CLAUDE_CONFIG_DIR` (outside `$HOME`); credentials and `agent_name` are
  persisted via `weave-claude-code config set` because the non-interactive
  installer doesn't. Adaptive thinking is disabled (GLM thinking blocks
  fail Anthropic validation on round-trip through the bridge).
- **Codex**: the README'd `bypass_hook_trust` config key is a no-op for
  `codex exec` (0.143.0); the working switch is the
  `--dangerously-bypass-hook-trust` CLI flag. The adapter patches emit.js
  so `WEAVE_CODEX_AGENT_NAME` (set to the run key) overrides the hardcoded
  agent name.

## Extending: adding a harness

An adapter is one class subclassing the harness's Harbor agent plus
`_TrialMetaMixin`, responsible for three things:

1. **Model routing** — point the harness at W&B Inference (directly if it
   speaks OpenAI chat completions, else via the bridge).
2. **Weave plugin install** — install/configure the harness's Weave plugin
   inside the container at `run()` time.
3. **Join key** — stamp the Harbor run key on the traces (resource
   attribute or `agent_name`) and extract harness session ids into
   `fugue-meta.json`.

The four existing adapters cover the protocol/plugin patterns you're likely
to meet; copy the nearest one.

## Layout

```
fugue/
├── fugue/                # python package
│   └── agents/           # Harbor agent adapters (model + trace wiring)
├── proxy/                # LiteLLM anthropic+responses bridge (docker compose)
├── vendor/
│   └── weave-node-sdk.tgz    # weave JS SDK built from the OTel-2.x branch
├── tasks/
│   └── smoke/bridge-check/   # tiny end-to-end task (read file -> write answer)
├── scripts/
│   ├── preflight.sh      # cheap validation of the whole model plane
│   └── smoke.sh          # 1-task-per-harness smoke runs
└── jobs/                 # harbor outputs (gitignored)
```

## Status

- Smoke suite (2026-07-08, GLM-5.2): all four harnesses reward 1.0 with
  traces verified in Weave (run keys joined on both span stores). A
  post-rename re-run reconfirmed Hermes + Codex end to end (`fugue.*`
  resource attributes / `agent_name` run keys); OpenClaw and Claude Code
  were cut short mid-suite by W&B Inference quota exhaustion, not by the
  framework.
- Next: fixture repos + memory-condition task generator for the RepoMemBench
  study; analysis notebooks over the Weave export.
- The vendored weave tgz goes away once
  [wandb/weave#7541](https://github.com/wandb/weave/pull/7541) ships to npm.
