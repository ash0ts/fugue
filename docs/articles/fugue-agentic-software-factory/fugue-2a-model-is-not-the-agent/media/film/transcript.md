# Transcript — The Model Is Not the Agent

**Series:** Fugue: Evals for the Agentic Software Factory

**Part:** FUGUE 2A

**Audience:** Agent researchers and platform engineers

**Duration:** 98 seconds

**Status:** DRAFT DESIGN / NO RESULT YET

## Article placement

This article-level synthesis appears after **Valid results, including nulls**
and continues into **Try this in 15 minutes**. Its scenes cite
only sections that appear before the film.

## Question

What exactly is the treatment when the same model runs through different harnesses?

## Intended takeaway

Report locked model–harness–environment candidates, not a model leaderboard.

## What the film does not claim

The preregistration contains no observed winner or causal harness result.

## 01 — One model can produce several behavioral candidates.

**Time:** 00:00–00:10

**Visual relationship:** candidate assembly

**On-screen support:** Harness, environment, tools, and stopping policy change observable behavior. The model name alone is not the reporting unit.

**Evidence status:** preregistered design

**Source:** `article.md#the-candidate-lattice`

**Displayed values:** none

## 02 — Each harness candidate receives the same locked task coordinates.

**Time:** 00:10–00:23

**Visual relationship:** matrix

**On-screen support:** Task identity and attempt index align the comparison. A missing coordinate remains missing.

**Evidence status:** preregistered design

**Source:** `article.md#assignment-ordering-and-attempts`

**Displayed values:** TASK 1 / A1; TASK 1 / A2; TASK 2 / A1; TASK 2 / A2

## 03 — Counterbalance execution order before observing outcomes.

**Time:** 00:23–00:37

**Visual relationship:** timeline

**On-screen support:** Half the blocks begin with A; half begin with B. Warm caches and temporal drift cannot become the treatment by accident.

**Evidence status:** preregistered design

**Source:** `article.md#assignment-ordering-and-attempts`

**Displayed values:** BLOCK 1; BLOCK 2

## 04 — Estimate paired candidate differences under locked conditions.

**Time:** 00:37–00:50

**Visual relationship:** boundary

**On-screen support:** The reporting unit is model–harness–environment. The design does not estimate a universal model ranking.

**Evidence status:** preregistered design

**Source:** `article.md#the-bounded-question`

**Displayed values:** none

## 05 — A reversal is a result, not a reason to pool.

**Time:** 00:50–01:02

**Visual relationship:** interaction plot

**On-screen support:** Candidate A can lead on one task family while B leads on another. Separate reporting preserves the interaction.

**Evidence status:** illustrative example

**Source:** `article.md#reversals-and-the-no-pooling-rule`

**Displayed values:** 0.75; 0.35

## 06 — Protocol differences can prevent a pure harness claim.

**Time:** 01:02–01:14

**Visual relationship:** dependency stack

**On-screen support:** If adapters or tool semantics differ, report the full candidates. Do not subtract away behavior-changing preparation.

**Evidence status:** preregistered design

**Source:** `article.md#threats-we-will-check-before-interpretation`

**Displayed values:** none

## 07 — No difference and harness reversal are both publishable outcomes.

**Time:** 01:14–01:26

**Visual relationship:** state machine

**On-screen support:** The design is useful even when there is no winner. Incomplete infrastructure can still make the study uninterpretable.

**Evidence status:** preregistered design

**Source:** `article.md#valid-results-including-nulls`

**Displayed values:** none

## 08 — Harness identity belongs inside experimental design.

**Time:** 01:26–01:38

**Visual relationship:** series map

**On-screen support:** It travels with the candidate through execution and evidence. The final claim remains bounded to that locked system.

**Evidence status:** preregistered design

**Source:** `article.md#valid-results-including-nulls`

**Displayed values:** none
