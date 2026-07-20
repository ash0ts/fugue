# Analysis Contract

Before previewing an experiment, write down:

- the audience and single research question;
- the operational decision the result should inform;
- authoritative evidence sources and known coverage gaps;
- fixed, varied, and measured dimensions;
- discovery, evaluation, and holdout partitions;
- the unit of alignment, such as task × harness × attempt;
- primary outcome and behavioral mechanism measurements;
- cancellation, cost, and evidence-eligibility rules;
- claims the design cannot support.

## Task construction

Use trace clusters to propose tasks, not to define truth automatically. A task needs a
clear environment, bounded resources, stopping policy, criteria, and permitted
evidence. Keep evaluator-only references away from the Agent. Freeze confirmatory
holdouts before selecting the intervention they will evaluate.

## Interpretation

Report observations before explanations. Compare aligned coordinates and show raw
task rows alongside aggregates. Preserve unavailable values as unavailable. Separate:

- task success;
- criteria success;
- tool availability and invocation;
- evidence returned, opened, and used;
- relevant and off-target changes;
- failures and interruptions;
- latency, token usage, and total cost.

Bound conclusions to the model, tasks, attempts, harnesses, runtime, and dates that
were actually tested. When effects reverse across tasks or harnesses, report the
interaction or replicate instead of selecting a convenient winner.
