# Governed task authoring

Fugue lets an outer research loop propose tasks without giving that loop an
experiment runner, a shell, or authority over execution. The same campaign
boundary introduced for loop engineering remains in control:

```text
catalog → task draft → task preview → task lock → proposal → plan
        → admission → run → evaluation → meta-analysis
```

The draft is untrusted. Preview is pure: it validates the complete suite,
resolves only registered profiles, computes capability and cost estimates, and
writes nothing under `.fugue`. Locking is the first mutation. It materializes a
content-addressed task definition, private evaluation contract, resources, and
Harbor manifest. Proposals carry only the resulting suite digest.

## What an outer loop can define

`TaskSuiteDraftV1` contains tasks, scenarios, and criteria sets. A task has a
bounded prompt, a registered environment profile, an interaction plan, a
criteria reference, tags, and a qualification, discovery, or holdout
partition.

Prompt parts are typed. Text is bounded inline content. Files, images,
artifacts, and generic resources must refer to a registered profile whose
bytes and target path are already locked. Raw host paths, environment maps,
setup commands, dependency declarations, and downloader instructions are not
part of the schema and fail validation.

Environment profiles describe a pinned repository workspace, a blank artifact
workspace, or a typed live-service workspace. A live integration must also be
registered in Fugue and explicitly allowed by the environment profile. Active
Agent trials cannot install dependencies, build images, download assets, start
services, or reach the Docker socket. Preparation remains an Operator-owned
step before admission.

Scenarios are the V1 analysis hierarchy. They assign weights to tasks and may
mark individual tasks as must-pass. The task definition digest deliberately
excludes criteria: changing a scoring revision does not change the Agent cell,
and changing the task does.

## Interaction without hidden leakage

The V1 interaction modes are single turn, locked scripted follow-ups, and
model-generated follow-ups through an approved interactor profile. Multi-turn
tasks preserve one planned cell and one native harness session. Hermes uses
session continuation, OpenClaw uses a stable session key, Claude Code resumes
its current session, and Codex resumes the last session.

Future scripted turns and interactor directions are not copied into the Harbor
task image. Fugue's controller reveals one follow-up only after observing the
current Agent reply. A model interactor receives the Agent-visible task and
conversation plus its approved directions. It never receives criteria,
private references, expected answers, gold data, or treatment labels.

Interactor calls have their own route receipts and cost records. They do not
become child calls of the Agent trace. Exact-head qualification still has to
prove that each supported adapter reconciles to one Weave Agent conversation,
one `invoke_agent` root, and one normalized prediction row.

## Evaluation is a separate revision

`CriteriaSetV1` supports registered deterministic checks, answer and artifact
contracts, tool or trace evidence, repository path checks, blind judge
profiles, and inline Python scorers. A criterion declares the evidence it may
see, a weight, a threshold, and whether it is required.

Scores are normalized to `0..1`. The task score is the weighted mean of
available criteria. Passing requires the criteria-set threshold and every
required criterion. If a required evaluator is unavailable or broken, the
evaluation is unavailable; it is not rewritten as an Agent failure.

Judge profiles must blind harness, model, variant, context system, candidate,
and treatment labels. Judge calls receive separate route receipts and are
admitted against the campaign budget before execution. Inline scorer source
runs only in a pinned Docker image with no network, no secrets, no task
checkout, a read-only filesystem, dropped capabilities, and explicit CPU,
memory, process, time, and output limits. Only normalized evidence and the
criterion's private reference object are mounted.

`TaskEvaluationV1` is immutable and points to a scoring revision, so the same
prediction rows can be rescored without another Agent run. Deterministic
benchmark outcome and authored-criteria outcome remain separate fields.

## Meta-analysis

`TaskStudyAnalysisV1` reports raw evaluation provenance plus aggregates by
task, scenario, harness, attempt, interaction mode, environment, and tag. It
includes aligned within-task harness contrasts, harness-by-scenario effects,
2,000-sample task-cluster bootstrap intervals, judge disagreement, failures,
turn/tool activity, latency, tokens, and observed cost.

The analysis does not produce a universal harness ranking. Its useful output
is conditional: which task shapes or interaction modes appear sensitive to a
harness under the locked model, resources, runtime, attempts, and criteria.

## Adaptive authoring and holdouts

An adaptive suite records its parent outcome and decision rationale. Only
discovery tasks may be authored after observing that outcome. Qualification
and confirmatory holdout tasks must have been locked earlier. Admission fails
if the policy, catalog, source, task suite, integration, runtime, or compiled
plan has drifted.

Operation IDs apply to suite locking, scoring, and analysis just as they do to
campaign preparation and launch. Repeating an operation with identical input
returns the same artifact; reusing the ID with different input fails.

## Initial studies

The first four uses are deliberately diagnostic rather than a leaderboard:

1. **Evaluation-layer qualification.** Apply final-answer-only and richer
   artifact/diff/tool evidence views plus two blind judges to the existing 32
   Kimi harness predictions. Compare benchmark agreement, false positives and
   negatives, score stability, disagreement, and judge cost. Historical rows
   must first be imported with their exact run and evidence provenance; they
   are never presented as a new Agent run.
2. **Harness × task shape.** Author 12 tasks across retrieval, diagnosis,
   evidence comparison, and synthesis scenarios. Hold one model and locked
   resources fixed, use two attempts, qualify 16 cells, then run 96 cells.
3. **Interaction mode.** Hold four tasks and their criteria fixed while
   comparing single turn, scripted clarification, and approved
   model-generated clarification across four harnesses and two attempts.
4. **Outer-loop discovery.** Let an outer loop identify a recurring discovery
   failure, author new discovery tasks for it, and test whether those tasks
   reproduce the failure and predict an intervention effect on a holdout suite
   that was locked before the intervention was selected.

These are study templates, not checked-in results. Every live cohort still
requires a clean exact head, serial canary, complete evidence reconciliation,
and explicit admission under its campaign policy.
