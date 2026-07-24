# Investigate a reviewed enterprise evidence failure

Use the Fugue research tools and the `optimize-agent-with-fugue` workflow.

The four selected Weave conversations were reviewed as the same failure class:
the current document appeared in search results, but the Agent did not open it
and answered from an older source.

1. Create or open the `Enterprise evidence use` Research.
2. Audit exactly the four selected calls from the registered
   `enterprise-evidence-agent` source. Do not copy conversation bodies, prompts,
   annotations, or document contents into Fugue.
3. Record:
   - What we saw: search returned the current document, but the Agent answered
     from an older source.
   - Plausible alternatives: weak ranking, unclear instructions, harness
     interaction, and run-to-run variation.
4. Derive the reviewed `enterprise-evidence-use-v1` task recipe.
5. Preview the `enterprise-evidence-use-v1` canary:
   one task × four treatments × two harnesses × one attempt.
6. Explain what stays fixed, what changes, the deterministic pass rule, and the
   maximum cells and reserved cost.
7. Stop and return the exact preview digest. Do not approve or start it.

After an operator supplies an approval digest for that unchanged preview,
resume from the durable Study, start it once, watch from the saved event cursor,
and record only the observation supported by the reconciled rows. Separate task
pass, infrastructure health, evaluation status, and evidence health. If the
effect reverses by harness or the cohort is non-discriminating, recommend
replication instead of a winner.
