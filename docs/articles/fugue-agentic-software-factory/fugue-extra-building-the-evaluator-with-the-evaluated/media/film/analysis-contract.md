# Analysis contract — Building the Evaluator with the Evaluated

## Audience and question

- **Audience:** Agent-first engineering teams and maintainers.
- **Single question:** What does Agent-assisted evaluator development prove—and what does it not prove?
- **Intended takeaway:** Auditable co-development improves the process; external boundaries establish evaluator trust.
- **Out of scope:** Agent productivity and PR volume do not validate the evaluator.

## Article relationship

- **Synthesizes:** the cited article sections through Rejected work is durable input
- **Bridges to:** A copyable co-development ledger
- **What to watch:** Inspect the public PR stack, the defects and cleanup decisions it records, and the external boundaries that prevent the evaluator from grading its own homework.

## Evidence boundary

The cited article section is authoritative for every scene. Numerical values
are serialized into the claim ledger and shown with their denominator, unit,
or evidence status. Illustrative examples remain illustrative. A planned or
pending study contains no implied result.

## Scene contract

| Scene | Read time | Takeaway | Evidence | Visual relationship |
| --- | ---: | --- | --- | --- |
| public-stack | 10s | Seven merged PRs record the current integration layers. | article.md#the-stack-as-research-evidence | pr stack |
| identity-case | 13s | A label stopped standing in for candidate identity. | article.md#case-study-1-identity-became-code | pipeline |
| qualification-defect | 14s | Qualification found a runtime defect before the claim shipped. | article.md#case-study-2-exact-tree-preparation-found-a-real-runtime-defect | state machine |
| cleanup-case | 13s | Static findings became an adjudication queue, not a delete command. | article.md#case-study-3-garbage-collection-needed-judgment | state machine |
| division-of-labor | 13s | Agents implement and inspect; humans set claims and approval. | article.md#separation-despite-shared-agents | swimlane |
| recursion-boundary | 12s | The evaluated system cannot rewrite its holdout or approve itself. | article.md#separation-despite-shared-agents | boundary |
| evidence-boundary | 13s | Public commits and checks are auditable; local anecdotes remain bounded. | article.md#what-we-can-quantify | ledgers |
| series-context | 12s | Co-development can improve the factory without validating its evaluator. | article.md#rejected-work-is-durable-input | series map |

## Semantic system

- Blue: locked evidence or fixed identity.
- Green: verified observation or terminal success.
- Amber: planned, required, missing, or approval-bound state.
- Coral: failure, forbidden transition, or unsupported interpretation.
- Violet: judgment, interpretation, or decision.

Motion preserves an identity, expands a matrix, reveals a denominator,
transitions state, or reconciles evidence. The final series map consumes less
than 30% of runtime; the article-local case remains primary.

Essential visual labels use a 28 px floor, supporting labels use a 24 px
floor, and compact identifiers may use 18 px. Essential content remains above
the film's 100 px bottom control-safe area.
