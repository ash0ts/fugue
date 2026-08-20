# Fugue 4B draft qualification runbook

Status: mutable working draft. No accepted preview, paid Study result, selected
intervention, holdout result, or qualified source tree is claimed.

## Required identities

- Clean reviewed Fugue and candidate trees
- One human-reviewed failed attempt with resolved native evidence
- Fixed Claude Code harness and Anthropic route
- Production and patched Skill locks
- Current and repaired MCP locks
- Discovery and host-private holdout task digests
- Local Harbor runtime and policy identities
- Dedicated result project `wandb/fugue-claude-loop-engineering-v1`

## Ordered gates

1. Reconcile and lock the original failed attempt.
2. Prepare discovery tasks, private holdouts, Skills, MCP runtimes, Claude Code,
   task images, and evaluation assets.
3. Preview exactly two tasks × four Skill/MCP arms × one attempt = eight
   discovery cells.
4. Have a human approve the exact discovery digest and cap.
5. Execute through local Harbor and reconcile outcome, intervention-use,
   Agent/Evaluation, privacy, policy, cleanup, and zero-orphan evidence.
6. Run the saved selection analysis. Stop if no arm is eligible.
7. Persist the treatment-selection lock before opening holdout.
8. Preview exactly four private tasks × production/selected arm × one attempt
   = eight holdout cells.
9. Obtain a separate human approval, execute through local Harbor, and
   reconcile every row.
10. Qualify or reject the source tree; publish a bounded human memo.

## Optional Aria shell

Aria may read safe Study and result projections for presentation. It is not a
runtime dependency and cannot approve, start, retry, select, mutate holdout,
or issue a release decision.

## Fail closed on

- unresolved source failure or native evidence;
- changed source, task, Skill, MCP, model, harness, or runtime identity;
- missing confirmed intervention use;
- holdout opened before a durable selection lock;
- private truth or credential exposure;
- incomplete Agent/Evaluation reconciliation;
- missing local Harbor policy, cleanup, or zero-orphan receipt;
- hidden retry, cross-Study pooling, or cross-project result mixing.

## Evidence boundaries

The no-key replay proves installation and deterministic result projection. A
direct MCP receipt proves server and tool mechanism only. Neither is a live
Agent Study.

Local Harbor evidence supports only the declared local execution policy; it
does not certify Serverless isolation. The separate MCP package decision
remains pending its source-isolated `main` versus final-staging Study and
independent package sign-off.
