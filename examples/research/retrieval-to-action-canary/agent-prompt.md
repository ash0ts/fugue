Use $optimize-agent-with-fugue and the configured Fugue MCP server.

Create or recover Study `retrieval-to-action-demo` in campaign
`retrieval-to-action-v1` with this question: “Does repository search improve a
software-repair Agent by itself, or only when the Agent must inspect and verify
what search returns?” Read the Study context and safe catalog before proposing
the experiment.

Preview exactly one registered experiment with these dimensions:

- proposal id: `retrieval-to-action-qualification-001`
- stage: `qualification`
- registered experiment: `retrieval-to-action`
- model: `wandb/zai-org/GLM-5.2`
- workload: `canary`
- harnesses: `codex`, `claude-code`
- context systems: `none`, `rag-dense`
- variants: `baseline`, `memory-only`, `policy-only`, `memory-policy`
- tasks: `1`
- attempts: `1`
- concurrency: `1`
- trace content: `full`

State the fixed dimensions as model, task, prompt base, runtime, and attempt.
State the varied dimensions as repository search, inspect-and-verify
instructions, and harness. Measure repair pass, search invocation, returned and
opened evidence, relevant-code changes, errors, latency, tokens, and cost.

The preview must resolve to eight cells. Do not prepare, approve, start, or
retry anything. Return the full eligibility result, blockers, estimated calls,
estimated cost, preview digest, and experiment id, then wait for an operator to
provide the approval digest.

After the operator provides that digest in this same task, start only the
unchanged preview with idempotency key
`start-retrieval-to-action-qualification-001`. Watch it using resumable event
cursors until terminal. Inspect the exact outcome and record only a bounded,
sourced Study Result. Treat repair passes as observations, keep behavioral
measurements separate, and do not claim a universal harness or search winner.
