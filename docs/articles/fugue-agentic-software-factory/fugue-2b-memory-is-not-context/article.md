# Fugue 2B — Memory Is Not Context: Preregistering the Repository-Memory Study

> **Fugue: Evals for the Agentic Software Factory · Part 2B**  
> A standalone preregistration for context, RAG, MCP, and developer-tool
> engineers. **Status:** draft preregistration; no accepted preview and no result. **Reading time:**
> about 16 minutes.

The complete causal vocabulary, 2×2 intervention, measurement denominators,
and privacy boundary appear here. No previous series installment is required.

The misconception is that retrieving the right repository path proves memory
helped an agent solve the task.

Our concrete failure was a treatment that looked healthy in its retrieval
table and inert in its Agent traces. The system had indexed the repository.
The query returned a gold-relevant file. The file name appeared in telemetry.
The agent did not open the relevant content before editing. We had measured a
search system and described an agent improvement.

Our falsifiable thesis is:

> Giving an agent memory does not prove that useful evidence was retrieved,
> opened, used, or responsible for success.

This thesis is weakened if retrieval assignment reliably implies use, or if
mechanism stages add no explanatory value beyond official task resolution.
Our 2×2 design lets the memory system and the instruction to inspect evidence
vary independently so we can observe where the chain breaks.

**Status:** draft preregistration. Historical Fugue memory runs are
operational evidence only. The mutable design below requires a new exact
preview, accepted article digest, and untouched holdout before it becomes a
preregistration or can support an efficacy claim.

## Scope and terms

**Stored** means evidence exists in the memory corpus. **Assigned** means the
candidate was configured to receive the treatment. **Returned** means a query
produced a source. **Opened** means the Agent inspected its content. **Used**
means an auditable decision or output depends on that content under the
declared rubric. **Outcome** is the official task result.

These are related events, not synonyms. The study measures their funnel; it
does not infer causality from a path appearing in a prompt or trace.

Weng’s agent taxonomy separates short-term context, long-term memory, and
retrieval-mediated access; that distinction prevents us from labeling every
piece of supplied text “memory.” [@weng-agent] Her harness survey also makes
the delivery mechanism part of the observed Agent rather than an invisible
preprocessing step. [@weng-harness] Hamel’s trace-first eval practice supplies
the operational test: open the actual retrieval and file-read sequence before
automating a proxy for use. [@hamel-evals-skills]

## Seven events hidden inside “the agent had memory”

“Memory” can refer to stored documents, an index, a retrieval service, a
prompt injection, a tool, or durable state from earlier work. We separate
seven events:

1. **Stored:** the source exists in the locked memory corpus.
2. **Retrieved:** a query returns a source reference.
3. **Exposed:** the harness makes that reference or content available to the
   Agent.
4. **Invoked:** the Agent chooses the memory tool or follows the injected
   context.
5. **Opened:** the Agent reads enough source content to inspect the relevant
   evidence.
6. **Used:** the answer or patch depends on supported facts from that source.
7. **Outcome:** the official task verifier or authored evaluation passes.

```mermaid
flowchart LR
    A["Assigned"] --> R["Returned"]
    R --> E["Exposed"]
    E --> I["Invoked"]
    I --> O["Opened"]
    O --> U["Used"]
    U --> T["Task outcome"]
```

Each arrow can fail. A repository can be indexed but queried poorly. A source
can rank at one and be dropped during context construction. A path can enter a
prompt but carry no useful content. The Agent can call the tool and ignore the
response. It can cite a file while making a change supported only by nearby
code. It can use the right evidence and still fail implementation.

This funnel is mechanism evidence, not a replacement outcome. The primary
question remains whether the repository task was resolved.

## Names and paths are weak evidence

Path overlap is tempting because it is easy to score. If private grading says
the fix touches `package/cache.py` and search returns that path, we can compute
recall@10.

That metric is useful for retrieval qualification. It is weak evidence of
agent use for three reasons.

First, the path may be obvious from the issue text. Second, a broad repository
query may return dozens of files, including the relevant one by chance.
Third, the final patch may touch the file without relying on the retrieved
content.

We therefore preserve path-based localization as a secondary retrieval
measure while requiring source-use evidence from allowed traces. “Opened”
requires a content read associated with the same source identity after it was
returned. “Used” requires a predeclared evidence relation: for example, the
answer states a source-supported fact, or the patch changes the discovered
contract consistently with its tests and caller evidence.

We do not expose private gold paths to the Agent to make this join easier.
They remain in a host-only evaluation lock. Publication contains derived
localization metrics and the lock digest, never the raw labels.

## The 2×2 intervention

Our core design crosses two binary treatments:

- **M:** a locked repository-memory system is available;
- **P:** an evidence-use policy tells the Agent to search, open, and verify
  relevant repository evidence before editing.

| Arm | Memory system | Evidence-use policy | Question                                                                      |
| --- | ------------- | ------------------- | ----------------------------------------------------------------------------- |
| A   | No            | Standard            | What does the native Agent do?                                                |
| B   | Yes           | Standard            | Does availability alone change behavior or outcome?                           |
| C   | No            | Yes                 | Does an inspection policy help with ordinary repository tools?                |
| D   | Yes           | Yes                 | Does policy create uptake of the memory system, and does that affect outcome? |

```mermaid
quadrantChart
    title Repository-memory intervention
    x-axis Standard policy --> Evidence-use policy
    y-axis No memory --> Locked memory
    quadrant-1 "D: memory + policy"
    quadrant-2 "B: memory only"
    quadrant-3 "A: neither"
    quadrant-4 "C: policy only"
```

This design prevents a common attribution mistake. If D beats A, the memory
system did not necessarily cause the difference. C may perform just as well,
showing that explicit source inspection—not the memory index—was the active
ingredient. B may improve localization without changing completion, showing
that availability is not uptake. D may cost more without increasing either.

The interaction also matters. The policy could be helpful only when memory is
available, or memory could distract a policy-guided Agent with irrelevant
retrievals.

## The factorial question

The 2×2 is not four unrelated demos. It estimates three bounded contrasts:

- **memory availability:** B − A under standard policy, and D − C under the
  evidence-use policy;
- **evidence-use policy:** C − A without memory, and D − B with memory;
- **interaction:** whether the policy contrast changes when memory is
  available.

For deterministic binary task resolution, we report paired task differences
within harness rather than fit a grand model to a tiny cohort. For mechanism
counts, we show raw distributions and aligned deltas. If later cohorts are
large enough for a model-based interaction estimate, that analysis must be
declared in the accepted preview; it is not added because one plot looks
interesting.

An outcome such as A=2, B=2, C=4, D=4 solved tasks supports a policy
observation on these tasks, not a memory effect. A=2, B=2, C=2, D=4 suggests
an interaction worth replicating. A=2, B=4, C=2, D=2 may indicate that the
policy interfered with a memory benefit. With only a few tasks, all remain
fragile and should be shown as aligned rows.

We predeclare no universal “memory score.” Official resolution, mechanism
uptake, latency, and cost can disagree. The decision depends on the product
question: a treatment that improves source use but not completion may be
valuable for auditability and unjustified for default deployment. That is a
product decision after the Study, not a weight hidden in analysis.

## What is locked

The final Study freezes:

- the public task briefs and task ordering;
- the host-only private evaluation lock;
- the base repositories and task images;
- the model route;
- each native harness build;
- prompt bytes for standard and evidence-use policies;
- memory-system source, dependencies, index, model, and vector dimensions;
- delivery mode and exact tool manifest;
- CPU, memory, storage, networking, timeout, and concurrency;
- attempt count and scheduling seed;
- deterministic verifier and scorer versions;
- evidence event schema and analysis code.

Behavior fingerprint changes include memory system, policy, prompt, tool
manifest, model, harness, task, and prepared runtime assets. Execution policy
changes include the environment that schedules an otherwise identical locked
candidate. A vector candidate cannot fall back to lexical search and retain
the same identity.

Fugue’s current managed GitNexus hybrid-vector boundary, for example, pins its
runtime dependencies and 384-dimensional embedding path. Setup executes an
offline lexical-mismatch semantic probe. If vector execution fails during a
candidate cell, the cell is not relabeled as a successful vector treatment.
The explicit BM25 candidate remains a different legal arm.

The principle applies to every memory system: graceful product fallback may
be good user experience, but hidden fallback is invalid experiment identity.

## Qualifying the measurement itself

Before testing memory efficacy, we qualify whether the evidence pipeline can
observe its mechanism.

The contract canary uses a task that explicitly requires a semantic lookup.
It must demonstrate:

1. the exact memory runtime initialized;
2. the native MCP tool was registered under the expected schema;
3. the Agent invoked the tool;
4. vector telemetry reported nonzero 384-dimensional execution;
5. the returned source was linked to a subsequent content open;
6. the official verifier ran offline;
7. the Agent conversation and Evaluation prediction reconciled.

This canary is deliberately biased toward tool use. It proves transport and
observability, not spontaneous uptake. Its row is excluded from efficacy.

We also run direct retrieval diagnostics against a frozen 225-query source.
Those queries characterize recall, rank, latency, errors, and vector
contribution without Agent variance. Preparation materializes and locks the
source; trials may not download it. Direct measurements remain ordinary Weave
operations and never synthesize Agent conversations.

The measurement is ready only if deterministic event joins survive negative
cases:

- a returned source that is never opened;
- an ordinary repository read with no memory invocation;
- a vector request that fails rather than falls back;
- a tool error with no result payload;
- multiple opens of the same source;
- an answer that mentions a path from the public task rather than retrieval;
- missing usage that remains missing.

Without these controls, the beautiful assigned-to-outcome funnel can be a
logging artifact.

## Discovery, selection, holdout

Fugue’s hard-memory program separates discovery from holdout.

The checked-in `repo-memory-impact` experiment contains:

- an eight-task hard calibration cohort with no memory;
- an eight-task Latin-square discovery cohort;
- a versioned treatment-selection analysis;
- a four-task untouched holdout with three attempts;
- two easy control tasks with three attempts;
- hard repository-QA, direct retrieval, and continuity diagnostics;
- an uptake diagnostic kept outside efficacy denominators.

The broader program evaluates several repository-memory systems. The 2×2
article Study is a frozen subdesign: discovery selects one qualifying memory
variant according to predeclared rules, then crosses it with the evidence-use
policy on untouched tasks. Selection cannot use holdout outcomes.

```mermaid
flowchart LR
    Q["Qualification<br/>runtime and contract"] --> D["Discovery<br/>rank treatments"]
    D --> L["TreatmentSelectionLockV1"]
    L --> H["Untouched 2×2 holdout"]
    H --> R["Aligned analysis"]
    E["Easy controls"] --> R
    U["Uptake diagnostic"] -. "mechanism only" .-> R
```

Discovery ranks variants within the same task, harness, and trial baseline.
Official resolution is primary. Localization recall@10, mean reciprocal rank,
recoverable-error rate, observed cost, and stable variant ID are tie-breaking
or secondary inputs exactly as declared. The selection lock records the full
ranking and chosen treatment. Holdout refuses a manually substituted variant.

Easy controls detect a treatment that adds overhead or breaks obvious tasks.
They do not enter the hard-task efficacy estimate. The uptake diagnostic can
require a semantic lookup to prove the end-to-end contract; because that
instruction changes behavior and its tasks were used during qualification, it
does not enter the unbiased treatment denominator.

## Primary and secondary outcomes

The primary outcome is official task resolution under the pinned offline
verifier. We pair arms within task, harness, and attempt policy.

Secondary outcomes are:

- localization recall@10 and mean reciprocal rank;
- assigned, returned, opened, and used source counts;
- required and spontaneous tool invocation;
- broad versus projected reads;
- no-action turns;
- recoverable errors by source;
- latency, observed tokens, and observed cost;
- patch size and files changed;
- evidence-honesty and maintainer-usefulness judgments.

We will show the funnel for every arm:

```text
assigned → returned → opened → used → official outcome
```

The denominators remain explicit. If 12 sources are returned across four
attempts, three are opened, one is used, and zero tasks pass, we do not say the
memory had “75% uptake” by choosing the convenient middle denominator.

Retrieval-only diagnostics remain in a separate evidence table. Nine hundred
direct queries can qualify a search system more precisely than a handful of
Agent tasks, but they are not Agent conversations and do not become task
outcomes.

## Context budget and displacement

Memory can hurt without returning a wrong source. It can consume the context
budget that would otherwise hold task instructions, code, tool output, or the
Agent’s own plan. It can also increase early confidence and reduce exploration.

We record:

- bytes or tokens injected before the first turn;
- retrieved content exposed per turn;
- truncation and compaction events;
- repository reads displaced or added;
- time between retrieval and first edit;
- whether required task or error content survived compaction.

These measures are harness-sensitive. One harness may inject retrieved content
directly; another may expose only an MCP reference. We do not normalize away
that product behavior. The candidate identity includes the delivery
interface, and analysis facets by harness.

A result can therefore show “memory improved localization but increased
truncation and reduced official resolution under harness H.” Calling the
retrieval component good does not make the integrated treatment good.

## What “used” can honestly mean

Causal source use is hard to infer from traces. We predeclare three evidence
tiers:

1. **Mechanically linked:** the exact returned source was subsequently opened
   in the same attempt.
2. **Semantically supported:** an allowed scorer finds that a material answer
   or patch decision matches inspected source evidence.
3. **Causally attributed:** the 2×2 contrast supports a difference associated
   with the locked treatment bundle.

Tier one is deterministic but shallow. Tier two is interpretive and requires
calibration. Tier three is a Study-level claim and still belongs to the whole
bundle, not a single file or tool call.

We avoid the stronger sentence “this source caused the patch” unless an
intervention actually manipulates source availability while controlling the
rest. Trace order is not causality.

A worked aligned row makes the denominator concrete:

| Coordinate | Assigned | Returned | Opened | Used | Official outcome |
| --- | ---: | ---: | ---: | ---: | ---: |
| `task-07 / harness-H / attempt-2 / arm-D` | 1/1 | 1/1 | 1/1 | 0/1 | pass |

This row supports “the locked memory returned evidence and the Agent opened
it.” It does not support “memory caused the pass,” because the declared use
rubric found no material dependence. Funnel denominators are the number of
eligible aligned coordinates at each stage, not the number of convenient
events in a trace. Long-context research also warns that putting relevant
information into a context does not guarantee effective use, especially as
position and competing content change. [@lost-in-middle]

## Failure and fallback semantics

Memory systems add infrastructure: builders, indexes, databases, MCP servers,
embedding models, context injectors, and caches. Their failures must not
become Agent failures.

Each attempt records:

- whether the treatment was applicable;
- whether required registration succeeded;
- whether the tool manifest matched;
- whether vector or lexical execution actually occurred;
- whether any fallback was declared and identity-preserving;
- whether sources were available to the Agent;
- whether task, trace, usage, and Evaluation evidence reconciled.

An unavailable optional product may be `not_applicable`. A required vector
runtime that silently uses BM25 is an identity violation. A timeout is not a
zero localization score. Missing OpenClaw usage is not free execution. A
memory service that starts after the Agent turn cannot claim availability.

Structured errors are themselves mechanism evidence. An agent that receives a
bounded “index unavailable” result can respond honestly; one that receives an
empty success may hallucinate completeness.

## The execution artifact

The repository exposes the existing cohort previews:

```bash
uv sync --python 3.13 --frozen --extra dev

uv run fugue run repo-memory-impact \
  --preset hard-calibration \
  --preview

uv run fugue run repo-memory-impact \
  --preset hard-discovery \
  --preview

uv run fugue analyze \
  --saved repo-memory-discovery-selection \
  --yes

uv run fugue run repo-memory-impact \
  --preset hard-holdout \
  --selection-lock REPORT_DIR/treatment-selection-lock.json \
  --preview

uv run fugue run repo-memory-impact \
  --preset uptake-diagnostic \
  --preview
```

These previews describe the broader managed-memory program. Before executing
the article’s 2×2 holdout, the exact standard and evidence-use policy variants
must appear in one generated preview, and its matrix must demonstrate all four
arms for every selected task–harness pair. The published artifact includes
that preview digest.

Preparation happens before preview:

- task repositories, verifier assets, and images are pinned;
- the selected memory runtime and index are built;
- vector artifacts and models are verified offline;
- tool registration and a nonzero semantic probe pass;
- the host-only evaluation lock is materialized;
- no trial is allowed to download or install these assets.

A human approves the exact cell and dollar cap. No discovery process can
approve its own selected holdout.

## Predeclared decision policy

The Study does not automatically make one memory system the product default.
It produces evidence for a decision policy.

We require all four arms to have complete eligible aligned rows, the selected
memory runtime to prove its declared retrieval mode, and no critical privacy
or evidence-honesty regression. We then consider:

1. official resolution contrasts within each harness;
2. the assigned-to-used funnel;
3. easy-control regressions;
4. observed latency, usage, and cost;
5. structured failure and fallback behavior;
6. maintainer usefulness and uncertainty.

A default-treatment recommendation requires a replicated resolution benefit
or an explicitly approved auditability benefit, no critical regressions, and
an operational cost the product owner accepts. A mechanism-only improvement
can justify further study or an opt-in mode; it cannot be described as a
coding-outcome win. A harness reversal blocks a universal default and may
support harness-specific configuration only if that decision was in scope.

We reject the selected treatment if it leaks private facts, hides vector
fallback, breaks easy controls, or creates unsupported completeness claims,
even when hard-task passes rise. We issue no decision if required rows,
usage, or source-use evidence are missing.

This policy exposes values that statistics cannot supply. How much latency is
acceptable for better auditability is a product judgment. Whether one
critical honesty regression outweighs two additional solved tasks is an
authority decision made in advance. Fugue records those choices; it does not
derive them from a composite score.

If no arm is distinguishable, we preserve the null. A successor may target
tasks with stronger repository-discovery demands, but it receives new private
labels, locks, preview, and approval. It cannot borrow the current holdout’s
identity.

The decision record names who owns the next action by workstream: memory
runtime maintainers for retrieval defects, harness maintainers for delivery
interactions, task authors for saturated cases, and evaluation owners for
ambiguous rubrics. We do not invent individual owners in an article. Clear
workstream ownership prevents a null from becoming a vague request to “improve
the agent.”

Each owner receives the exact aligned rows and lock identities, so remediation
begins from inspectable evidence instead of a generalized treatment ranking.

## Analysis and useful nulls

The primary paired analysis reports each harness separately. We do not pool a
memory win in one harness with a loss in another.

The following are useful publishable outcomes:

- **Better localization, no completion change.** Retrieval works, but the
  downstream Agent mechanism or task bottleneck remains.
- **More opening, no source use.** The policy changes behavior without
  improving decisions.
- **Policy benefit, no memory benefit.** Ordinary repository tools plus a
  stronger inspection contract explain the gain.
- **Memory benefit only with policy.** Availability needs an uptake
  intervention; the interaction is the result.
- **Higher cost, equal outcome.** The treatment is not justified for this
  taskset despite richer traces.
- **Harness reversal.** Context delivery or tool use differs by native
  harness; no universal memory claim is supported.
- **No interpretable result.** Missing locks, fallback, or evidence
  reconciliation prevent analysis.

We will not respond to a null by editing the holdout. A new, harder taskset
receives a new Study identity and preregistration.

## Privacy and leakage

Repository-memory evaluation is unusually vulnerable to label leakage because
the objects under test are sources.

Raw gold paths, expected patches, hidden test facts, and selection labels must
not enter:

- Agent prompts or environment;
- context bundles or indexes;
- MCP responses;
- task images accessible to the Agent;
- Weave trace inputs or outputs;
- W&B Run configuration;
- Study events or public snapshots.

The lock digest may appear. Derived localization counts may appear. A safe
deep link may appear only when its object contains no private label. We run
value-based leakage scans against generated jobs, snapshots, traces, logs, and
exports—not merely filename searches.

Credential values have the same rule. Their names can define a secret
contract; their values belong only in the trusted operator and runtime secret
boundary.

## Try this in 15 minutes

Open one Agent trace that used retrieval. Mark the exact event for assigned,
returned, exposed, invoked, opened, and used. Write `unobserved` rather than
guessing when a stage has no evidence. Then record the official task outcome
on a separate line.

Repeat for five attempts. If your “used” label is inferred only from a path
appearing somewhere in the trace, rename it “path observed” and design the
content-read or semantic-support evidence you actually need.

## When a memory study is unnecessary or insufficient

If a repository simply lacks navigable documentation, fix the documentation
before testing a retrieval treatment. A memory Study is useful when the
uncertainty concerns retrieval or evidence use. It is insufficient when the
gold corpus leaks into Agent inputs, lexical fallback is mislabeled as vector
retrieval, or the task can be solved without consulting repository evidence.

## What this does not show

This preregistration does not prove repository memory helps. It does not prove
the selected memory system is representative of RAG, code graphs, generated
maps, or longitudinal memory. It does not make trace-based “use” causal.

Four untouched tasks with three attempts can produce aligned evidence, not a
universal effect estimate. The official SWE verifier measures task
resolution, not long-term maintainability. The evidence-use policy is a real
treatment bundle; any benefit belongs to its exact wording and harness
integration.

Historical Fugue runs established that memory candidates, Agent links, and
direct retrieval measurements could execute. Some historical exports contain
known publication issues or missing usage. They are preserved as smoke
evidence and excluded from this Study.

## Results appendix — intentionally empty

The dated results appendix must record:

```text
source commit/tree and preview digest:
selected memory treatment and TreatmentSelectionLockV1:
task/evaluation/runtime/model/harness locks:
all four arm fingerprints:
planned/started/completed/excluded/missing by arm:
official paired outcomes by harness:
assigned/returned/opened/used funnel:
localization, error, latency, usage, and cost:
fallback and vector-conformance evidence:
privacy scan receipt:
reversals, nulls, and limitations:
canonical Weave/Study links:
supported claim:
```

If exact vector execution or private-label isolation cannot be proven, the
Study is ineligible.

## The bridge: an MCP release is an agent change

Memory made one lesson unavoidable: changing the information and actions
available to an Agent changes the candidate even when the model is untouched.
An MCP server is a general version of the same intervention. Descriptions,
projections, pagination, and errors can change what the Agent notices and
does.

In the next installment, **Fugue 3**, we apply the
Study primitives to an exact W&B MCP release comparison using genuine hosted
evidence and W&B Serverless execution.

## References

Complete source metadata and numbered citations are generated from this
article package’s `sources.json`.
