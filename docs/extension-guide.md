# Extending Fugue

Fugue compares controlled agent treatments. Memory is one treatment family,
not the system boundary. The matrix expands:

```text
workload × task × harness × variant × trial
```

A variant can change a prompt, one or more Agent Skills, context systems,
MCP/HTTP integrations, agent arguments, environment settings, verifier
settings, retry policy, and requested artifacts. Harbor 0.18.0 owns sandboxing,
task provisioning, native skills/MCP injection, service composition, artifact
collection, and verification. Fugue owns selection, safe preparation,
provenance, comparison identity, and availability classification.

Harbor remains in its own pinned `uv tool` environment. This is deliberate:
Harbor's model-provider dependency range currently conflicts with some optional
memory-provider dependencies. `fugue setup --check` verifies both
`harbor==0.18.0` and that `fugue.agents` imports under Harbor's Python, so the
environments stay isolated without weakening the execution contract.

## Current support

| Extension | Implemented now | Important boundary |
| --- | --- | --- |
| Local skills | Full skill directories under `configs/fugue/skills/<id>` | Whole bundle is hashed; duplicate declared names are rejected |
| Remote skills | Public GitHub source, exact subdirectory, review, approval, lock, immutable cache | Git objects are inspected without checkout; repository code is never installed or run |
| Context providers | `preflight`, `prepare`, `bind`, retrieval and optional ingestion/sequence | Task/commit cache identity and per-trial bindings remain separate |
| MCP integrations | Stdio, SSE, and streamable HTTP Harbor bindings | Stdio can enforce per-tool allowlists; HTTP requires a future policy gateway for per-tool filtering |
| HTTP services | Compose or external URL exposed through a deterministic endpoint environment variable | Compose image digest and external host allowlist are mandatory |
| Repositories | Legacy `repo`/`base_commit` and typed immutable Git repository input | Typed form requires a full commit SHA |
| Datasets/workloads | Harbor refs/paths, existing materializers, typed HTTP source metadata | A new benchmark still needs a Harbor dataset/environment/verifier contract |

Full third-party plugin installation, repository lifecycle hooks, HTTP MCP
per-tool filtering, private Git credentials, signed-source verification, and a
remote integration-image builder are not implemented. The TUI can select local
and declared remote skills, but integration declarations are currently edited
in YAML. These gaps should not be simulated with setup shell commands.

## Add a remote skill

Declare exactly one skill directory. This example is already present as
`configs/fugue/skill-sources/hallmark.yaml`:

```yaml
id: hallmark
source:
  type: git
  url: https://github.com/Nutlope/hallmark
  ref: aeb42fb354ff4efa36ab475773a082315a3af2ce
  path: skills/hallmark
```

Select it with the canonical variant field:

```yaml
variants:
  - id: hallmark
    label: Hallmark
    skills: [hallmark]
```

Then inspect and approve:

```bash
fugue setup --experiment my-study --skills --json
fugue setup --approve-skill hallmark=sha256:REVIEWED_DIGEST \
  --acknowledge-risk network-access
```

Inspection rejects symlinks, submodules, Git LFS pointers, unsupported binary
files, missing `SKILL.md`, path traversal, duplicate injected names, and size
limit violations. It inventories every file and flags scripts, package
manifests, network/credential instructions, destructive commands, repository
mutation, servers, telemetry, and subagent control. Acknowledgement is tied to
the exact digest and source commit in `configs/fugue/skills.lock.yaml`.

Fugue includes pinned declarations for one skill directory from Taste Skill,
Superpowers, Hallmark, and Emil Kowalski's skills repository. Only the selected
directory is admitted. For example, selecting Superpowers brainstorming does
not install its plugin, session hooks, extensions, or other skills. The
`multica-ai/andrej-karpathy-skills` repository currently presents a
`CLAUDE.md`/plugin-style guideline rather than an Agent Skills `SKILL.md`
bundle, so it is not silently treated as a skill; add a reviewed local adapter
if that treatment is desired.

## Add an MCP or HTTP integration

Integration declarations are reviewed project code under
`configs/fugue/integrations/<id>.yaml`. A local service looks like:

```yaml
id: repository-search
version: "1"
support: experimental
runtime:
  type: compose
  image: ghcr.io/example/repository-search@sha256:FULL_64_HEX_DIGEST
  service: repository-search
  port: 8000
  healthcheck: {path: /health}
interfaces:
  - type: mcp
    name: repository-search
    transport: streamable-http
    path: /mcp
required_env: [SEARCH_API_KEY]
instructions: [usage.md]
artifacts:
  - source: /logs/retrieval.jsonl
    service: repository-search
config_schema:
  top_k: {type: integer}
```

The image must be digest-pinned. Fugue renders read-only, non-root Compose
services with all Linux capabilities dropped, `no-new-privileges`, a bounded
temporary filesystem, no published host ports, and optional resource and
health checks. Secret values are not written into generated job configs;
Harbor resolves `${NAME}` templates from the trial process environment.

An external integration requires HTTPS and its hostname in `allowed_hosts`.
Fugue passes that list to Harbor's `extra_allowed_hosts`; it does not widen
network access globally. Selecting an integration is the server-level allow
decision. Stdio MCP declarations can additionally use `allowed_tools`, which
the Fugue proxy enforces and records. `allowed_tools` on HTTP MCP is rejected
until a reviewed filtering gateway exists.

Plain HTTP interfaces are exposed as
`FUGUE_INTEGRATION_<INTEGRATION>_<INTERFACE>_URL`. Instructions should document
the protocol and credentials without embedding secret values.

## Add a repository or benchmark

The typed task form makes immutable repository provenance explicit:

```yaml
tasks:
  - id: example-task
    repository:
      type: git
      url: https://github.com/example/project
      commit: 0123456789abcdef0123456789abcdef01234567
      path: packages/core
```

The legacy `repo` and `base_commit` fields remain readable. Do not mix the two
forms in one task. The typed form currently admits public GitHub HTTPS inputs;
private credential brokers are a future source-provider extension. `path` is
immutable source provenance for an adapter or materializer; it does not by
itself limit Harbor's workspace to that subdirectory.

For another task in an existing Harbor dataset, add the task and repository
identity. For a new benchmark type, implement its task environment,
instructions/submission contract, artifact contract, and verifier in Harbor or
a Fugue workload runner. Fugue should not duplicate those execution semantics.

Typed dataset sources may declare `type: http`, an HTTPS URL, and a SHA-256.
Git dataset sources are normalized but require a Git-aware materializer. This
keeps source acquisition explicit rather than allowing arbitrary install
commands in a dataset declaration.

## Conformance checklist

Before changing a support level to `supported`:

1. Validate the declaration with unknown-field rejection and pinned versions.
2. Prove setup is inert or isolated and idempotent.
3. Render and validate the generated configuration with Harbor 0.18.0.
4. Verify secrets are templates in artifacts and values exist only in process
   environments.
5. Test unsupported harness/transport combinations become `not_applicable`.
6. Capture version, digest, selection config, support level, availability,
   invocation, errors, latency, and artifacts in run provenance.
7. Run at least one baseline/treatment pair in fresh trials.

Use `experimental` for a runnable adapter that has not completed this matrix.
Use `not_applicable` for a known harness or transport mismatch and `disabled`
for a deliberately unavailable adapter. Neither should be counted as a failed
agent trial.
