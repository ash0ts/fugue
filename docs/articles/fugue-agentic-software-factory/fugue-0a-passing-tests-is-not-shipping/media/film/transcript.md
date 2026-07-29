# Transcript — Passing Tests Is Not Shipping

**Series:** Fugue: Evals for the Agentic Software Factory

**Part:** FUGUE 0A

**Audience:** Staff engineers and maintainers

**Duration:** 84 seconds

**Status:** PUBLISHED / ILLUSTRATIVE CASE

## Article placement

This article-level synthesis appears after **Garbage collection belongs in “done”**
and continues into **Try this in 15 minutes**. Its scenes cite
only sections that appear before the film.

## Question

What does a green task verifier leave unanswered about a patch?

## Intended takeaway

A passing patch can still make the next change harder.

## What the film does not claim

This does not argue that tests are weak or that every patch needs a study.

## 01 — The verifier sees the outcome, not the trajectory.

**Time:** 00:00–00:10

**Visual relationship:** patch

**On-screen support:** The bounded task passes. Three repository costs remain outside that verdict.

**Evidence status:** illustrative example

**Source:** `article.md#the-patch-and-the-trajectory`

**Displayed values:** none

## 02 — Every surviving surface creates another ownership obligation.

**Time:** 00:10–00:23

**Visual relationship:** trajectory

**On-screen support:** The patch is cheap to generate. Understanding, correcting, owning, and removing it are not.

**Evidence status:** illustrative example

**Source:** `article.md#a-review-burden-is-an-output`

**Displayed values:** 1; 1.2; 1.4; 1.6; 1.8; 2; 3.4; 5.2; 7.4

## 03 — Generated, correct, mergeable, and maintainable are not synonyms.

**Time:** 00:23–00:36

**Visual relationship:** stack

**On-screen support:** Each rung needs additional evidence. A lower rung cannot stand in for a higher one.

**Evidence status:** cited background

**Source:** `article.md#the-patch-and-the-trajectory`

**Displayed values:** none

## 04 — Review burden is part of the patch’s output.

**Time:** 00:36–00:48

**Visual relationship:** ledgers

**On-screen support:** Two candidates may solve the same tasks. The one requiring less repair leaves more capacity for the next change.

**Evidence status:** cited background

**Source:** `article.md#a-review-burden-is-an-output`

**Displayed values:** none

## 05 — Different gates answer different acceptance questions.

**Time:** 00:48–01:00

**Visual relationship:** pipeline

**On-screen support:** Compiler and tests reject known deterministic defects. Review and cleanup examine ownership and trajectory.

**Evidence status:** cited background

**Source:** `article.md#necessary-tests-insufficient-evidence`

**Displayed values:** none

## 06 — Known defects belong in CI; disagreement belongs in review.

**Time:** 01:00–01:12

**Visual relationship:** boundary

**On-screen support:** Do not experimentalize a compiler error. Do not pretend a green check settled architectural ownership.

**Evidence status:** cited background

**Source:** `article.md#necessary-tests-insufficient-evidence`

**Displayed values:** none

## 07 — Maintainability frames both the question and the final decision.

**Time:** 01:12–01:24

**Visual relationship:** series map

**On-screen support:** The workflow begins with a repository problem. It ends only when evidence supports a bounded acceptance decision.

**Evidence status:** cited background

**Source:** `article.md#garbage-collection-belongs-in-done`

**Displayed values:** none
