# What the first WBA transport run actually tells us

We ran the same WBA-style Agent loop, GLM-5.2 model, tools, tasks, runtime,
and sampling settings through three transport paths. All 48 planned cells ran
to completion and produced the requested Markdown and JSON artifacts. The
official task result was still `0 / 48`.

That zero is real under the V1 scorer, but it is not a usable comparison of
the transports. The task instructions did not tell the Agent which top-level
JSON shape the scorer expected, and the answer check required literal phrases
rather than the underlying facts. The result is best shared as an evaluation
design and protocol-conformance finding—not as a transport benchmark.

## The controlled comparison

The primary cohort contained eight offline analysis tasks across four
scenarios, three transport profiles, and two attempts:

| Profile | Agent-facing path |
| --- | --- |
| `responses-proxy` | OpenAI Responses client → Fugue LiteLLM proxy → provider Chat Completions |
| `responses-inline` | LiteLLM Responses conversion inside the Agent process |
| `chat-inline` | LiteLLM Chat Completions inside the Agent process |

The model, endpoint, task resources, system prompt, shell tool, loop limits,
runtime, and sampling settings were fixed. Only the transport profile varied.

## What completed successfully

- `48 / 48` planned rows reached a terminal state.
- All 48 prediction, conversation, trace, and root identities were unique and
  reconciled.
- All 48 rows had linked Agent evidence, equivalent locked runtimes, and zero
  provider errors.
- Every cell produced both requested files; every Markdown answer was non-empty
  and every JSON artifact parsed.

This qualifies the V1 execution and evidence path. It does not qualify the V1
task scorer or establish that the profiles behaved equivalently.

## Why every official score was zero

The V1 task score was the conjunction of two checks:

1. every expected term had to appear literally in the Markdown answer;
2. every expected top-level JSON key had to contain the exact expected value.

The two components never passed in the same cell:

| V1 check | Passed |
| --- | ---: |
| Requested artifact pair present | 48 / 48 |
| Literal Markdown answer check | 34 / 48 |
| Scorer-compatible top-level JSON contract | 2 / 48 |
| Both checks, the official task score | 0 / 48 |

The output patterns make the failure mode clearer. Agents generally produced
richer, nested artifacts with explanations and evidence instead of guessing a
minimal flat object that had never been disclosed. A post-hoc containment
check found the expected scalar text somewhere in all 48 serialized JSON
artifacts. That is useful diagnostic evidence, but it is not a semantic score:
it does not prove that a value appeared at the right path or played the right
role.

The literal answer failures were task-shaped, not profile-shaped. Five tasks
passed that component in all six cells. The evaluation-plan task passed it in
`0 / 6`, the regression task in `1 / 6`, and the intervention task in `3 / 6`.
Equivalent number wording and formatting were enough to fail. By profile, the
same component was `12 / 16` for Chat inline and `11 / 16` for each Responses
path. There is no meaningful profile separation in those counts.

## Operational observations

These figures are descriptive only. V1 used asymmetric stream diagnostics,
exception-class retry behavior, and no queryable Weave model-call children.
The token counts came from the local runner rather than a Weave usage join,
and observed cost was unavailable.

| Profile | Median wall time | Runner-reported tokens | Tool calls | Compactions | Retries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chat inline | 104.8 s | 490,182 | 134 | 40 | 0 |
| Responses inline | 98.3 s | 314,355 | 151 | 49 | 0 |
| Responses proxy | 89.9 s | 506,517 | 162 | 58 | 13 |

The mean and median latency orderings differed, and aligned cells did not move
consistently in one direction. With two attempts per task and the V1 telemetry
limitations, these numbers do not support a speed, token-efficiency, cost, or
reliability claim.

One protocol signal does matter. The proxy arm recorded 30,006 stream
anomalies in V1, dominated by missing sequence numbers. Because V1 did not
apply the same validator to every applicable arm, that count could not serve
as a fair comparison. The V2 qualification therefore moved protocol
conformance ahead of live execution and tested the proxy independently.

Both immutable proxy images tested for V2 failed the same small conformance
fixture: each emitted 14 Responses events, 12 without a required sequence
number and one with non-monotonic ordering. Fugue consequently marks
`responses-proxy` unsupported and blocks the cohort before approval or spend.
This is the current actionable result.

## Bounded conclusion

The V1 run demonstrated that Fugue could execute, isolate, and reconcile the
full 48-cell matrix. It also exposed two flaws that prevent a scientific
transport conclusion: an undisclosed, brittle output contract and asymmetric
wire diagnostics.

V2 fixes the first problem with a visible JSON Schema and private values at
declared JSON pointers. It fixes the second with shared turn validation,
shared Responses-event validation, call-kind-specific retry and compaction
policies, and separate protocol, task, and operational results.

There is not yet a V2 transport result, and no live V2 cohort is planned in
this PR. Any future comparison should be a separate decision made only after a
proxy image passes conformance. The defensible statement is:

> The first cohort qualified the execution path and invalidated its own scorer.
> The repaired study correctly stops because the proxy is not wire-conformant.

It does **not** show transport equivalence, a winner, a cost advantage, or any
result about `wandb/core`.

## Provenance and publication boundary

- V1 source commit: `3a4dcb9b9980e522f7db9dcc555d60f21396a095`
- Primary run: `20260722T150531-73a8b45a27`
- Outcome digest: `b11b834a5c3644d7c2df46025fc70d608f042cd6720d4af73892a82c6aa00ff8`
- Snapshot digest: `03a16ae99e70d6a8146843953f0a53b8ef858c82b8899e33bd16306ad27ea321`
- Export SHA-256: `0e4185d31edad87fb81f3fd9cdd2424ac9c2bbb28e6d88f0e372cf387e2f46d1`
- V2 qualification head: `35a2aa86e8e27348148a12d9d31e55235abacf05`

The companion
[`v1-shareable-results.json`](./v1-shareable-results.json) contains only
reviewed aggregate evidence. Neither artifact includes raw Agent content,
trace content, prompts, private expected values, local paths, secrets, or an
observed-cost claim. The immutable post-hoc audit remains separate in
[`v1-exploratory-audit.json`](./v1-exploratory-audit.json). No Atlas record was
created.
