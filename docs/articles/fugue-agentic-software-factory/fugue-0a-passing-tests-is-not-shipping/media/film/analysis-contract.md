# Analysis contract — Passing Tests Is Not Shipping

## Audience and question

- **Audience:** Staff engineers and maintainers.
- **Single question:** What does a green task verifier leave unanswered about a patch?
- **Intended takeaway:** A passing patch can still make the next change harder.
- **Out of scope:** This does not argue that tests are weak or that every patch needs a study.

## Article relationship

- **Synthesizes:** the cited article sections through Garbage collection belongs in “done”
- **Bridges to:** Try this in 15 minutes
- **What to watch:** Track one passing patch as local success turns into repository ownership and removal cost.

## Evidence boundary

The cited article section is authoritative for every scene. Numerical values
are serialized into the claim ledger and shown with their denominator, unit,
or evidence status. Illustrative examples remain illustrative. A planned or
pending study contains no implied result.

## Scene contract

| Scene | Read time | Takeaway | Evidence | Visual relationship |
| --- | ---: | --- | --- | --- |
| passing-patch | 10s | The verifier sees the outcome, not the trajectory. | article.md#the-patch-and-the-trajectory | patch |
| ownership-cost | 13s | Every surviving surface creates another ownership obligation. | article.md#a-review-burden-is-an-output | trajectory |
| quality-ladder | 13s | Generated, correct, mergeable, and maintainable are not synonyms. | article.md#the-patch-and-the-trajectory | stack |
| review-output | 12s | Review burden is part of the patch’s output. | article.md#a-review-burden-is-an-output | ledgers |
| acceptance-gates | 12s | Different gates answer different acceptance questions. | article.md#necessary-tests-insufficient-evidence | pipeline |
| ci-boundary | 12s | Known defects belong in CI; disagreement belongs in review. | article.md#necessary-tests-insufficient-evidence | boundary |
| series-context | 12s | Maintainability frames both the question and the final decision. | article.md#garbage-collection-belongs-in-done | series map |

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
