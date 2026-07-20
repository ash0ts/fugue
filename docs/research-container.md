# Fugue Research container

The container package lets an external Agent use Fugue as a governed laboratory
without giving that Agent a writable application checkout, a Docker socket, or
an execution approval capability.

```text
external Agent ── bearer-authenticated MCP/REST ──> fugue-control
                                                       │
                                          durable queue + SQLite WAL
                                                       │
                                                       v
operator approval ─────────────────────────────> fugue-worker ──> Harbor
```

`fugue-control` exposes high-level Study, trace-audit, experiment, and Result
operations on localhost. `fugue-worker` has no public port and is the only
service with the local Docker socket required to launch Harbor. Both share
receipts and content-addressed assets through the `fugue-state` volume.

## Start locally

Bootstrap private secret files, then build and start the two services:

```bash
WANDB_API_KEY=... fugue research bootstrap --repo-root .
docker compose -f compose.research.yaml up --build -d
```

The bootstrap command keeps `.fugue/secrets` accessible only to the host user
and makes the files inside it read-only. This is required because local Compose
implements file-backed secrets as bind mounts and the control service runs as a
non-root user. Run bootstrap again to repair permissions on an older setup.

The control endpoint is `http://127.0.0.1:8787`. Configure an MCP-capable Agent
with:

- URL: `http://127.0.0.1:8787/mcp/`
- transport: Streamable HTTP
- header: `Authorization: Bearer <contents of .fugue/secrets/research_api_key>`

REST clients use the same token under `/v1`; OpenAPI is available at `/docs`.
The Agent can create drafts and pure previews but cannot issue an approval.

To export the portable workflow skill into an Agent's skill directory:

```bash
fugue research skill export --destination /path/to/skills/optimize-agent-with-fugue
```

The packaged MCP prompt is generated from the same `SKILL.md`, so clients that
do not load Agent Skills can follow the identical workflow without a second copy
of the instructions.

## Register trace evidence

Trace sources are operator configuration. They are not URLs or paths supplied by
the Agent. Edit `configs/fugue/research/trace-sources.yaml` before starting the
container:

```yaml
version: 1
sources:
  - id: production-agent
    adapter: weave
    project: entity/project
    allowed_fields: [status, operation, errors, tools, latency, tokens, cost]
    allowed_filters: [run_id, status, harness, model]
  - id: offline-evaluation
    adapter: jsonl
    path: /data/traces/evaluation.jsonl
    allowed_fields: [status, errors, tools, artifacts]
    allowed_filters: [run_id, status, harness, model]
```

Mount JSONL files beneath `.fugue/trace-data`, which is read-only in the control
container. The safe catalog returns source IDs, supported fields, filters,
redaction rules, and digests; it omits the Weave project and JSONL path.

A trace preview validates the bounded cohort without reading it. The audit reads
at most the accepted number of root traces, summarizes conversations rather than
copying raw message bodies, redacts credential-looking content, and stores
immutable trace references plus a cohort digest. Trace content is always data,
never instructions to the external Agent.

## Register candidate sources

Repository and artifact candidates use a separate operator catalog in
`configs/fugue/research/candidate-sources.yaml`:

```yaml
version: 1
sources:
  - id: application
    kind: git
    url: https://github.com/example/application
    allowed_experiments: [agent-loop]
    allowed_variants: [baseline, candidate]
```

The Agent sees the source ID, type, allowed experiment and variants, and a
source digest—not the configured location. A candidate reference must carry
that digest, a full immutable revision, and a content digest. For authored
repository tasks, Fugue also requires the repository URL and commit in the task
environment to match the registered source and reference. A full commit on an
unregistered URL is rejected.

Artifact entries use a config-relative `path` plus its SHA-256
`content_digest`; Fugue verifies that digest when it loads the operator catalog.
Mount candidate artifacts or mirrors read-only beside this configuration.

## Approval and execution

The intended sequence is:

1. The Agent reads the Study context and safe catalog.
2. It previews a bounded trace audit and separates observations from hypotheses.
3. It authors discovery tasks and locks holdouts before choosing an intervention.
4. It previews one controlled matrix and presents the digest, cells, calls, and
   cost to the operator.
5. The operator approves that exact preview from a trusted shell:

   ```bash
   fugue research approve PREVIEW_DIGEST \
     --max-usd 200 \
     --max-cells 8 \
     --approved-by OPERATOR_ID
   ```

6. The Agent submits the unchanged preview and approval digest. The worker locks,
   prepares, admits, launches, scores, and analyzes through Fugue's existing
   campaign and Harbor path.
7. The Agent reconnects with the same experiment handle or SSE cursor, then
   records only a scoped Result with exact evidence references.

Approvals expire, are bound to one preview, and cannot be reused for another
experiment. The exact admission reservation must fit below the approved cap;
otherwise Fugue writes no admission and launches no cells. Lost responses are
recovered through idempotency keys and durable handles, never by silently
relaunching trials.

## Isolation boundary

- The repository and registered configuration are mounted read-only.
- `fugue-control` runs as UID 10001, drops Linux capabilities, uses a read-only
  root filesystem, and has no Docker socket.
- `fugue-worker` has the Docker socket only to operate Harbor. It has no port.
- Active Harbor cells retain Fugue's existing prohibition on install, download,
  build, service startup, host-path access, and Docker access.
- Credentials enter through Docker secrets. They are not accepted through MCP,
  REST bodies, task definitions, or Study notes.
- An external Agent changes application code elsewhere and submits only an
  immutable, registered commit or artifact reference for evaluation.

Stop the deployment with:

```bash
docker compose -f compose.research.yaml down
```

The named state volume remains. Remove it only as an explicit data-deletion
operation after retaining any required Study and campaign evidence.
