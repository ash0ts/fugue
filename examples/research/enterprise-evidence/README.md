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
  --max-usd 200 \
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
