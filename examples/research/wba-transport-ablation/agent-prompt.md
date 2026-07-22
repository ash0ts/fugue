Use $optimize-agent-with-fugue and the configured Fugue MCP server.

Create or recover Study `wba-transport-ablation-demo` in campaign
`wba-transport-ablation-v2` with this question: “When the Agent loop and final
model endpoint stay fixed, do Responses conversion topology or wire format
change task outcomes or protocol integrity?” Read the bounded Study context and
safe catalog before proposing anything.

Preview exactly this registered qualification experiment:

- proposal id: `wba-transport-qualification-v2-001`
- stage: `qualification`
- registered experiment: `wba-transport-ablation-v2`
- model: `wandb/zai-org/GLM-5.2`
- workload: `transport`
- harness: `wba-responses`
- context system: `none`
- variants: `responses-proxy`, `responses-inline`, `chat-inline`
- tasks: `1`
- attempts: `1`
- concurrency: `1`
- trace content: `full`

State that the model, provider Chat endpoint, task resource, task-neutral system
prompt, shell tool, loop policy, runtime, sampling, and attempt are fixed. State
that only the transport profile varies. Report wire-protocol conformance,
Agent-loop operational behavior, and visible-schema deterministic task outcomes
as separate result layers. Include tool-call integrity, retries, compaction
errors and fallbacks, stop reason, provider or bridge errors, latency, tokens,
and observed cost; never substitute admission reserve for unavailable cost.

The preview must resolve to exactly three serial cells. Do not prepare, approve,
start, retry, or modify anything. Return the full eligibility result, blockers,
estimated calls, estimated cost, experiment id, and exact preview digest. Then
wait for an operator to provide the approval digest.

If live preflight reports `responses-proxy` unsupported, return that blocker and
stop. Do not request approval or start a partial cohort.

After the operator supplies that approval digest in this same task, start only
the unchanged preview with idempotency key
`start-wba-transport-qualification-v2-001`. Follow durable event cursors until it
is terminal. Never retry a cell. Inspect the exact outcome, transport receipts,
identity reconciliation, Agent links, accounting, and infrastructure evidence.
Record one bounded, sourced Study Result. Keep task outcomes separate from
evidence eligibility and do not infer transport equivalence from a null
difference.

If and only if all three cells are terminal and evidence-eligible, preview one
child experiment:

- proposal id: `wba-transport-primary-v2-001`
- stage: `primary`
- registered experiment: `wba-transport-ablation-v2`
- parent experiment: the exact terminal qualification experiment
- parent outcome: its exact outcome artifact
- decision rationale: the compatible harness and three locked transports
  produced reconciled qualification evidence, so the preregistered task cohort
  can proceed unchanged
- model: `wandb/zai-org/GLM-5.2`
- workload: `transport`
- harness: `wba-responses`
- context system: `none`
- variants: `responses-proxy`, `responses-inline`, `chat-inline`
- tasks: `8`
- attempts: `2`
- concurrency: `1`
- trace content: `full`

Require exactly 48 serial cells and V2's cumulative $1,275 campaign cap.
Return the second experiment id, eligibility, blockers, estimate, and exact
preview digest, then stop. Do not approve or start it.

If the operator later supplies the primary approval digest in this same task,
start only that unchanged preview with idempotency key
`start-wba-transport-primary-v2-001`. Watch it with resumable cursors until
terminal, without retries. Record a sourced Result that reports arm totals as
passes / 16, aligned task-attempt discordance, the preregistered
`responses-inline - responses-proxy` and `chat-inline - responses-inline`
contrasts, task-cluster bootstrap intervals, protocol-integrity metrics, and
available cost. If all task outcomes are identical, call the cohort
non-discriminating rather than evidence of equivalence. State the locked scope
and limitations. Do not declare equivalence or a universal transport winner.

If qualification is ineligible, incomplete, or has an infrastructure or
evidence-contract failure, record that state and stop. Do not preview or launch
the primary.
