# Experimental evaluation-provider protocol

Fugue Evaluation Provider V1 is an experimental, language-neutral JSON
subprocess protocol. It lets an external evaluation system prove that it can
describe and freeze its candidates, suites, private truth, preparation, and
one-cell result contract without making that external system a Fugue runtime.

Provider conformance is deliberately offline. It is not a Fugue experiment,
does not run a paid cohort, and never qualifies task outcomes.

## Supported boundary

A provider accepts one operation argument and one JSON object on standard
input, then emits exactly one JSON object:

- `describe`
- `resolve-candidate`
- `resolve-suite`
- `prepare`
- `run-cell`

Unknown fields fail closed. Candidate, suite, public task, private evaluation,
preparation, cell request, and cell result artifacts use the strict V1 schemas
under `schemas/fugue/providers/`.

The protocol preserves the important separation between public task content
and host-private evaluation truth. Private bundles are written with mode
`0600`. Commands are launched without a shell, provider source and executable
bytes are locked, mutable descriptors are rejected, and the offline cell
exercise refuses every credential-bearing candidate or task.

## Scaffold, validate, and lock

```bash
uv run fugue provider schema \
  --destination schemas/fugue/providers

uv run fugue provider scaffold ./my-provider \
  --provider-id my-provider

uv run fugue provider validate \
  --command "python ./my-provider/provider.py"

uv run fugue provider lock \
  --command "python ./my-provider/provider.py" \
  --output /tmp/my-provider.lock.json
```

The lock binds the resolved executable, executable digest, provider
descriptor, provider source digest, and protocol version.

## Credential-free conformance

Fugue includes a dependency-free fake provider that covers single-turn,
multi-turn, structured context, lifecycle declarations, deterministic
evaluation, preparation, cancellation-shaped results, and cleanup:

```bash
uv run fugue provider lock \
  --command "python -m fugue.providers.fake" \
  --output /tmp/fugue-fake-provider.lock.json

uv run fugue provider conformance \
  --provider /tmp/fugue-fake-provider.lock.json \
  --candidate reference \
  --suite conformance \
  --exercise-run-cell \
  --output /tmp/fugue-fake-provider-conformance.json
```

`--exercise-run-cell` invokes only the locked provider in a temporary local
workspace. It injects no credentials, model route, Sandbox, or Fugue evaluator.
The receipt therefore always has:

```text
scope = offline_protocol_conformance
task_outcomes_qualified = false
```

## Explicit non-goals

Fugue no longer includes a provider-backed Harbor Agent or a WBAF-specific
launch path. It does not call an external matrix runner, score provider cells
as Fugue task outcomes, or treat protocol success as Agent improvement.

WBAF, Aria, CoreWeave, Modal, and direct OpenAI credentials are not dependencies
of this experimental protocol. A future live provider must first map into the
canonical `ExperimentSpec → ResolvedRunPlan → cell → prediction → result` path
and pass Fugue's isolation, identity, approval, and evidence contracts; this
protocol alone is not that integration.

## Identity and safe claims

Provider source, descriptor, candidate bundle, behavior assets, suite, private
evaluation, preparation, request, result, and cleanup are independently
digested. Conformance can support only these claims:

- the executable matches its lock;
- the provider emits valid, internally bound V1 artifacts;
- public and private bundles remain separate;
- selected offline lifecycle operations complete and report cleanup.

It cannot support task quality, evaluator quality, production compatibility,
Serverless isolation, or a release recommendation.
