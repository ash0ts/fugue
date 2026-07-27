# Enterprise evidence-use Study

This example turns a reviewed Weave failure cohort into a governed factorial
experiment. It is synthetic and source-neutral: no customer document or company
data is included.

## Research question

Search returned the current document, but the Agent answered from an older
source. Does added repository search help on its own, does requiring source
inspection help on its own, or do they work together?

The experiment holds the model, task corpus, tools, base instructions, runtime,
and sampling fixed. It varies:

- repository search: off or on;
- source inspection: standard workflow or must inspect and cite;
- harness: Codex or Claude Code, as a robustness factor.

The canary is eight attempts. The primary is 64 attempts:

```text
canary  = 1 task × 4 treatments × 2 harnesses × 1 attempt
primary = 4 tasks × 4 treatments × 2 harnesses × 2 attempts
```

## Safe local checks

These commands do not call a model:

```bash
uv run --frozen fugue run enterprise-evidence-use-v1 \
  --preset canary --preview --json

uv run --frozen fugue run enterprise-evidence-use-v1 \
  --preset primary --preview --json

uv run --frozen --extra dev --extra research pytest -q \
  tests/test_enterprise_evidence_use.py
```

Every task has a public artifact schema and a private deterministic verifier.
Preparation must prove the empty workspace fails and the locked gold artifact
passes before a live attempt is admitted.

## Governed Agent flow

An external Agent should follow
[agent-prompt.md](agent-prompt.md). It selects exactly four reviewed Weave
calls, records an observation rather than a diagnosis, derives the registered
recipe, and previews the eight-attempt canary. It then stops.

Paid work requires a separate operator approval bound to the exact preview:

```bash
fugue research approve PREVIEW_DIGEST \
  --max-usd 45 \
  --max-cells 8 \
  --approved-by "$USER"
```

The primary requires a different preview and a separate approval. Do not reuse
the canary approval, retry an attempt under the same Study identity, or promote
a treatment from the canary.

## What to inspect

- Study Console explains the question, factor design, deterministic pass rule,
  arm totals, and evidence-use funnel.
- W&B Weave is authoritative for each Agent conversation, prediction-and-score
  call, Evaluation, Dataset, and review annotation.
- Fugue contains the immutable preview, task lock, runtime lock, admission,
  normalized rows, analysis snapshot, and evidence reconciliation.

The Study supports a bounded implementation decision. It does not prove that
one harness or retrieval strategy is universally better.

## Run the cohesive local demo

The demo writes every local evidence object to one project:
`ashah-weights-biases/loop-engineering-demo`. That includes six optimizer W&B
Runs, 24 source conversations, Fugue Agent attempts, prediction-and-score
calls, the Dataset, the Evaluation, and scores. GLM-5.2 inference remains
separately billed through `wandb/fugue-experiments`.

```bash
cd /Users/ashah/Documents/GitHub/core-aria-fugue-study-loop

export FUGUE_DEMO_REPO_ROOT=/Users/ashah/Documents/GitHub/fugue-split-inference-evidence-routing
export STUDY_CONSOLE_REPO_ROOT=/Users/ashah/Documents/GitHub/study-console-unified-autoresearch
DEMO=services/wb_agent/examples/loop-engineering-demo/demo.sh

$DEMO --profile enterprise-evidence up
```

In another terminal:

```bash
$DEMO --profile enterprise-evidence status --open
$DEMO --profile enterprise-evidence console
```

After the exact preview is approved and the local qualification completes, the
demo should show:

1. the four exact reviewed Weave Calls and their safe review status;
2. the locked search × source-inspection design before execution;
3. eight reconciled attempts with deterministic scores;
4. links from each attempt to its authoritative Weave evidence;
5. the honest conclusion that the 8/8 canary was non-discriminating.

Restart Fugue control and Study Console once while replaying. Publication
should resume from the durable outbox without duplicating the Study or any
attempt. Study Console availability must not alter the stored outcome.

Stop the local stack with:

```bash
$DEMO --profile enterprise-evidence down
```

The eight-attempt qualification has its own exact-digest approval and a $45
cap. A primary is intentionally not part of this runbook and always requires a
different preview and separate operator approval.
