# Transcript — Memory Is Not Context

**Series:** Fugue: Evals for the Agentic Software Factory

**Part:** FUGUE 2B

**Audience:** Context, retrieval, MCP, and developer-tool teams

**Duration:** 104 seconds

**Status:** DRAFT DESIGN / NO RESULT YET

## Article placement

This article-level synthesis appears after **Privacy and leakage**
and continues into **Try this in 15 minutes**. Its scenes cite
only sections that appear before the film.

## Question

What can a retrieval trace honestly establish about evidence use?

## Intended takeaway

Returned, opened, used, and successful are separate events with separate denominators.

## What the film does not claim

Mechanism evidence does not by itself prove that memory caused task success.

## 01 — Most returned evidence never becomes used evidence.

**Time:** 00:00–00:10

**Visual relationship:** mechanism funnel

**On-screen support:** 12 sources returned; 3 opened; 1 used. Across four attempts, 0 tasks passed.

**Evidence status:** illustrative example

**Source:** `article.md#primary-and-secondary-outcomes`

**Displayed values:** 12; 3; 1; 0; 4

## 02 — The Agent opened the evidence but did not satisfy the declared use relation.

**Time:** 00:10–00:25

**Visual relationship:** paired row

**On-screen support:** Assigned 1/1; returned 1/1; opened 1/1; used 0/1. The official task still passed.

**Evidence status:** illustrative example

**Source:** `article.md#what-used-can-honestly-mean`

**Displayed values:** task-07 / harness-H / attempt-2 / arm-D; 1/1; 0/1

## 03 — The row supports an audit statement, not a causal story.

**Time:** 00:25–00:40

**Visual relationship:** boundary

**On-screen support:** Supported: the locked source was returned and opened. Unsupported: memory caused the pass.

**Evidence status:** preregistered design

**Source:** `article.md#what-used-can-honestly-mean`

**Displayed values:** none

## 04 — Separate memory from the evidence-use policy.

**Time:** 00:40–00:54

**Visual relationship:** matrix

**On-screen support:** Two interventions produce four legal arms. Their interaction is estimated rather than assumed.

**Evidence status:** preregistered design

**Source:** `article.md#the-22-intervention`

**Displayed values:** none

## 05 — A vector failure cannot silently become vector success.

**Time:** 00:54–01:07

**Visual relationship:** state machine

**On-screen support:** Fallback is a structured failure or a distinct candidate. The explicit BM25 arm keeps its own identity.

**Evidence status:** preregistered design

**Source:** `article.md#failure-and-fallback-semantics`

**Displayed values:** BM25 ARM; VECTOR FAILED → SILENT BM25 → VECTOR SUCCESS

## 06 — More retrieved text can displace instructions or task evidence.

**Time:** 01:07–01:20

**Visual relationship:** budget bar

**On-screen support:** Delivery policy belongs in candidate identity. Usage and latency are secondary outcomes, not free capacity.

**Evidence status:** preregistered design

**Source:** `article.md#context-budget-and-displacement`

**Displayed values:** 30; 10

## 07 — Localization can improve without completion improving.

**Time:** 01:20–01:32

**Visual relationship:** ledgers

**On-screen support:** Policy may help without memory helping. Cost can rise without any evidence uptake.

**Evidence status:** preregistered design

**Source:** `article.md#analysis-and-useful-nulls`

**Displayed values:** none

## 08 — Mechanism evidence explains where a treatment reached—not why it won.

**Time:** 01:32–01:44

**Visual relationship:** series map

**On-screen support:** The funnel strengthens evidence integrity. The controlled comparison still determines the outcome claim.

**Evidence status:** preregistered design

**Source:** `article.md#privacy-and-leakage`

**Displayed values:** none
