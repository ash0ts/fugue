# North-star cases

Use the smallest research shape that can change an engineering decision.

## Improve an existing Agent from traces

Start with a bounded production or evaluation trace cohort. Separate recurring
observations from explanations, turn one failure cluster into discovery tasks, and
lock a distinct holdout before selecting an intervention. Evaluate immutable prompt,
tool, retrieval, harness, or code candidates on aligned tasks. The result should say
whether the intervention helped this use case and where it failed, not whether one
Agent is universally better.

## Isolate an Agent-loop component

Use a registered factorial or ablation when the question is about a component such
as repository search, inspect-and-verify instructions, tool access, context delivery,
or harness behavior. Hold model, task, runtime, and attempt identity fixed. Measure
both task outcome and the behavioral chain that could explain it. A component being
available is not evidence that the Agent invoked or used it.

## Learn which task shapes are sensitive

Author scenarios with explicit criteria when aggregate benchmark scores hide where
systems differ. Compare aligned outcomes by task, scenario, harness, interaction
mode, and attempt. Preserve deterministic outcomes separately from judge- or
criteria-based evaluations, and inspect scoring disagreement before drawing a
product conclusion.

## Advance a research branch

Read the parent outcome and its limitations from the Study. Choose one next branch:
replicate an uncertain effect, confirm it on a pre-locked holdout, or run an ablation
that separates competing explanations. Record the exact parent and rationale, create
one pure preview, and stop for operator approval. If the evidence cannot justify one
branch, record that ambiguity instead of searching until a favorable result appears.
