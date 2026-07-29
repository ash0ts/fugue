# Fugue: Evals for the Agentic Software Factory

> A nine-part field guide to measuring, changing, and governing agent
> behavior.

Coding agents made plausible patches cheap. They did not make it cheap to
decide whether those patches belong in a living software system. These
standalone field notes document the experiments, locks, approvals, evidence,
and cleanup boundaries we added while building Fugue.

Each installment defines its own vocabulary, opens with a concrete failure,
states a claim you could prove wrong, includes a copyable artifact, and says
plainly what its evidence does not show. Reading earlier installments adds
context but is never required.

## Publication sequence

The [series manifest](series.json) is authoritative. Published entries are
released and indexable. Working drafts are linked, mutable, visibly labeled,
and excluded from search indexing. A draft preregistration becomes immutable
only when its article and preview acceptance fields exist.

1. [Fugue 0A — Passing Tests Is Not the Same as Shipping Software](fugue-0a-passing-tests-is-not-shipping/article.md)
   — **published**
2. [Fugue 0B — The Eval Problem Is Not a Scoring Problem](fugue-0b-eval-problem-is-not-scoring/article.md) — working draft
3. [Fugue 1 — From Vibes to Studies](fugue-1-from-vibes-to-studies/article.md) — working draft
4. [Fugue 2A — The Model Is Not the Agent](fugue-2a-model-is-not-the-agent/article.md) — draft preregistration
5. [Fugue 2B — Memory Is Not Context](fugue-2b-memory-is-not-context/article.md) — draft preregistration
6. [Fugue 3 — API Compatibility Is Not Agent Compatibility](fugue-3-api-compatibility-is-not-agent-compatibility/article.md) — draft preregistration
7. [Fugue 4A — Telemetry Is Not Evaluation](fugue-4a-telemetry-is-not-evaluation/article.md) — working draft
8. [Fugue 4B — From a Trace to a Release Decision](fugue-4b-trace-to-release-decision/article.md) — draft preregistration
9. [Fugue Extra — Building the Evaluator with the Evaluated](fugue-extra-building-the-evaluator-with-the-evaluated/article.md) — working draft

The canonical public index is:

```text
https://ash0ts.github.io/fugue/articles/
```

## Evidence and result contract

Every empirical appendix names the exact source tree, candidate, taskset,
evidence snapshot, runtime, planned cells, terminal cells, exclusions, and
canonical Weave or Study references. Results stay separated into:

1. infrastructure and protocol conformance;
2. deterministic task outcomes;
3. authored or calibrated-judge evaluation;
4. mechanism and evidence integrity.

Missing evidence is never converted to a zero. Preregistered article text and
base films are content-hashed and append-only; later evidence is added under
a dated result file and a separate result-coda film.

## Runtime boundary

The flagship workflow uses W&B Serverless Sandboxes, W&B Inference,
Anthropic, W&B Runs, and Weave Calls, Datasets, and Evaluations. It contains
credential names only, never values. Local Harbor is a parity baseline.
OpenAI and direct CoreWeave operation are not part of the runtime
instructions.

## Editorial influences

The series borrows the operational distinctions and myth testing of Oskar
Dudycz, the taxonomy and limitation discipline of Lilian Weng’s research
surveys, and the trace-first, copyable practitioner workflows of Hamel
Husain. The voice, examples, claims, and visual language remain Fugue’s own.
Background research sharpens questions; only locked Fugue evidence can
answer Fugue-specific ones.
