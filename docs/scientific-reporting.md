# Scientific report publication

Fugue publishes scientific reports only after a comparison has produced a
canonical `ComparisonResultV3`. Report generation is offline; W&B publication
is a separate, explicit operator action.

## Build offline

```bash
uv run fugue result STUDY_ID \
  --report-action build \
  --report-out .fugue/reports/STUDY_ID
```

The immutable bundle contains:

- `result.json`: the complete canonical `ComparisonResultV3`;
- `report.json`: the derived `ScientificReportV1` and claim ledger;
- `report.md`: the decision-oriented human rendering; and
- `bundle.json`: independent file hashes and the result/report digests.

Reading the bundle recomputes the result, report, Markdown, and every file
digest. A different result cannot be placed behind an existing report.

Generate the exact data contract for article graphics separately:

```bash
uv run fugue result \
  --report-action visual-data \
  --report-bundle .fugue/reports/STUDY_ID \
  --report-out .fugue/reports/STUDY_ID/visual-data.json
```

Observed panels contain only values derived from the bound result. The next
stage is a distinct `planned` panel with a description and no observed value,
so an article cannot accidentally render a proposed cohort as completed data.
The manifest digest binds every panel to the result and report digests.

The narrative answers what was compared, why it was run, how it was run, what
was found, and what should happen next. Named deterministic blockers are
shown directly. Judge, Skill-use mechanism, efficiency, and evidence-integrity
claims remain separate; an advisory judge or Grade A evidence cannot turn a
failed deterministic outcome into an improvement.

## Add observed visual evidence

Generate graphics only after the canonical result exists. Pass a strict
`VisualAssetManifestV1` whose `source_result_digest` matches that result:

```bash
uv run fugue result STUDY_ID --report-action build \
  --visual-assets PUBLICATION/visual-assets.json \
  --report-out .fugue/reports/STUDY_ID
```

Each visual file lives below `assets/`, carries its SHA-256, size, media type,
alt text, and the report claim IDs it explains. A provenance animation also
requires reduced-motion, transcript, and source-hash companions in the same
group. Fugue verifies and copies the exact bytes; it does not generate or
invent observed graphics during report building.

## Publish one Study

Publication creates exactly one W&B Run with `job_type=scientific-report`, one
artifact containing the complete offline bundle, and one W&B Report. The Run
is marked `report_only` and excluded from task inputs and Evaluation counts.

```bash
uv run fugue result --report-action publish \
  --report-bundle .fugue/reports/STUDY_ID \
  --result-url https://wandb.ai/ENTITY/RESULT_PROJECT/artifacts/RESULT \
  --study-console-url https://STUDY-CONSOLE/...?study_id=STUDY_ID \
  --weave-url https://wandb.ai/ENTITY/RESULT_PROJECT/weave \
  --visibility organization \
  --receipt-out .fugue/reports/STUDY_ID/publication-receipt.json \
  --index-out .fugue/reports/STUDY_ID/study-report-index.json \
  --yes
```

`--yes` is mandatory because publication changes external W&B state. Repeating
the exact request is idempotent. A changed artifact, result, project, or link
creates a different publication identity instead of overwriting evidence.

## Build and publish the campaign index

The campaign index references independent Study indexes without pooling or
ranking their outcomes:

```bash
uv run fugue result --report-action campaign-index \
  --campaign-membership examples/comparisons/community-skill-selected-v1/campaign-membership.lock.json \
  --study-index .fugue/reports/SUPERPOWERS/study-report-index.json \
  --study-index .fugue/reports/ANTHROPIC/study-report-index.json \
  --study-index .fugue/reports/VERCEL/study-report-index.json \
  --campaign-id community-skill-case-studies-v1 \
  --campaign-project wandb/fugue-community-skill-case-studies-v1 \
  --report-out .fugue/reports/community-skill-case-studies-v1.json

uv run fugue result --report-action publish-campaign \
  --report-bundle .fugue/reports/community-skill-case-studies-v1.json \
  --receipt-out .fugue/reports/community-skill-case-studies-v1.receipt.json \
  --yes
```

The public claim remains bounded to each exact pair of revisions, task set,
model route, harness, and runtime. The campaign index does not calculate a
cross-repository winner.
