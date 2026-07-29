import {
  el,
  formatNumber,
  formatPercent,
  metric,
  safeExternalLink,
  tierLabel
} from "./common.js";
import { renderStudyAnalysis, studyCohorts } from "./analysis.js";
import { experimentById } from "./data.js";

const root = document.querySelector("#experiment-detail");
const parameters = new URLSearchParams(window.location.search);
const id = parameters.get("id") || "";
const requestedCohort = parameters.get("cohort") || "";
const experiment = experimentById(id);

if (!experiment) {
  root.append(el("section", { className: "empty-state" }, [
    el("p", { className: "eyebrow", text: "Not found" }),
    el("h1", { text: "This study is not in the reviewed index." }),
    el("a", { href: "./experiments.html", className: "text-link", text: "Return to Studies" })
  ]));
} else {
  renderStudy(experiment);
}

function renderStudy(study) {
  document.title = `${study.title} · Fugue`;
  const metrics = study.metrics;
  const incomplete = ["active", "blocked"].includes(study.evidence_tier);
  const cohorts = studyCohorts(study);

  const hero = el("section", { className: "experiment-hero ruled" }, [
    el("div", { className: "evidence-line" }, [
      el("span", {
        className: `tier tier-${study.evidence_tier}`,
        text: tierLabel(study.evidence_tier)
      }),
      el("span", { className: "status", text: study.status.replaceAll("_", " ") })
    ]),
    el("h1", { text: study.title }),
    el("p", { className: "lede", text: study.summary }),
    el("div", { className: "metric-strip" }, [
      metric(
        incomplete ? "Planned" : "Published",
        incomplete
          ? String(metrics.expected_predictions)
          : `${metrics.predictions}/${metrics.expected_predictions}`,
        incomplete ? "exact cells" : "predictions"
      ),
      metric(
        "Resolved",
        metrics.scored_predictions
          ? `${metrics.passed_predictions}/${metrics.scored_predictions}`
          : "Unavailable",
        "official verifier"
      ),
      metric("Pass rate", formatPercent(metrics.pass_rate), "not a composite"),
      metric("Agent links", metrics.agent_links ? formatNumber(metrics.agent_links) : "Unavailable", "verified")
    ])
  ]);

  const navigation = el("nav", { className: "study-nav", "aria-label": "Study sections" }, [
    el("a", { href: "#decision", text: "Decision" }),
    el("a", { href: "#question", text: "Question" }),
    el("a", { href: "#design", text: "Design" }),
    el("a", { href: "#results", text: "Results" }),
    el("a", { href: "#analysis", text: "Analysis" }),
    el("a", { href: "#evidence", text: "Evidence" }),
    el("a", { href: "#limitations", text: "Limitations" })
  ]);

  const decision = incomplete
    ? unresolvedDecision(study)
    : supportedDecision(study);

  const question = el("section", { id: "question", className: "narrative-grid anchored-section" }, [
    el("div", {}, [
      el("p", { className: "eyebrow", text: "Question and hypothesis" }),
      el("h2", { text: study.question })
    ]),
    el("div", {}, [
      el("h3", { text: "Hypothesis" }),
      el("p", { text: study.hypothesis }),
      el("h3", { text: "Why it matters" }),
      el("p", { text: study.why_it_matters })
    ])
  ]);

  const design = el("section", { id: "design", className: "section-block anchored-section" }, [
    el("div", { className: "section-heading" }, [
      el("div", {}, [
        el("p", { className: "eyebrow", text: "Controlled comparison" }),
        el("h2", { text: "Fixed, varied, and measured" })
      ]),
      el("p", { text: study.task_selection })
    ]),
    el("div", { className: "design-columns" }, [
      definitionGroup("Fixed", [
        ["Tasks", `${study.matrix.tasks.length} locked cases`],
        ["Attempts", String(study.matrix.attempts)],
        ["Workload", study.matrix.workload_id],
        ["Route proof", routingSummary(study.cells)]
      ]),
      definitionGroup("Varied", [
        ["Models", study.matrix.models.join(", ")],
        ["Harnesses", study.matrix.harnesses.join(", ")],
        ["Treatments", study.matrix.treatments.join(", ")]
      ]),
      definitionGroup("Measured", [
        ["Deterministic", "Official task outcome"],
        ["Operational", "Completion, errors, latency, usage, cost"],
        ["Mechanism", "Turns, tool calls, retrieval when available"],
        ["Evidence", "Run, prediction, evaluation, and Agent links"]
      ])
    ])
  ]);

  const results = el("section", { id: "results", className: "section-block evidence-block anchored-section" }, [
    el("div", { className: "section-heading" }, [
      el("div", {}, [
        el("p", { className: "eyebrow", text: "Results" }),
        el("h2", { text: incomplete ? "No decision yet" : "Reconciled outcome ledgers" })
      ]),
      el("p", {
        text: incomplete
          ? `${metrics.expected_predictions} cells are planned. Unpublished coordinates do not count as failures.`
          : `${metrics.predictions}/${metrics.expected_predictions} terminal predictions are public.`
      })
    ]),
    incomplete
      ? el("div", { className: "unresolved-receipt" }, [
          el("strong", { text: "Unavailable" }),
          el("span", { text: "No supported winner or pass-rate conclusion." })
        ])
      : resultLedgers(metrics)
  ]);

  const analysisRoot = el("div", { id: "study-analysis-results" });
  const cohortSelect = el("select", { id: "cohort-select", "aria-label": "Compatible cohort" });
  for (const cohort of cohorts) {
    cohortSelect.append(el("option", {
      value: cohort.id,
      text: cohort.label,
      selected: cohort.id === requestedCohort ? "selected" : null
    }));
  }
  const analysis = el("section", { id: "analysis", className: "section-block anchored-section" }, [
    el("div", { className: "section-heading" }, [
      el("div", {}, [
        el("p", { className: "eyebrow", text: "Analysis" }),
        el("h2", { text: "Compatible evidence stays inside this study." })
      ]),
      el("label", { className: "cohort-control" }, [
        el("span", { text: "Cohort" }),
        cohortSelect
      ])
    ]),
    analysisRoot
  ]);

  const evidence = el("section", { id: "evidence", className: "section-block anchored-section" }, [
    el("div", { className: "section-heading" }, [
      el("div", {}, [
        el("p", { className: "eyebrow", text: "Evidence and provenance" }),
        el("h2", { text: "Inspect the identities behind the result." })
      ]),
      el("p", { text: "Raw Agent content remains in authenticated Weave; the Atlas publishes safe coordinates and verified links." })
    ]),
    study.cells.length ? taskEvidence(study.cells) : el("p", {
      className: "empty-copy",
      text: "No task-level predictions have reconciled to the public evidence layer."
    }),
    el("div", { className: "provenance-grid" }, [
      definitionList([
        ["Source", shortIdentity(study.provenance.source_commit)],
        ["Dataset", study.provenance.dataset_id],
        ["Dataset digest", shortIdentity(study.provenance.dataset_digest)],
        ["Snapshot", shortIdentity(study.provenance.snapshot_digest)],
        ["Run IDs", study.provenance.run_ids.join(", ") || "Pending"]
      ]),
      el("div", { className: "link-row" }, provenanceLinks(study))
    ])
  ]);

  const limitations = el("section", { id: "limitations", className: "notes-grid anchored-section" }, [
    listSection("Supported findings", incomplete ? [] : study.findings, "No result has been declared."),
    listSection("Limitations", study.caveats, "No additional limitations are recorded.")
  ]);

  root.append(hero, navigation, decision, question, design, results, analysis, evidence, limitations);

  const selected = renderStudyAnalysis(analysisRoot, study, requestedCohort);
  if (selected) {
    cohortSelect.value = selected;
    updateCohortUrl(selected, false);
  }
  cohortSelect.addEventListener("change", () => {
    renderStudyAnalysis(analysisRoot, study, cohortSelect.value);
    updateCohortUrl(cohortSelect.value, true);
  });
}

function supportedDecision(study) {
  return el("section", { id: "decision", className: "decision-summary anchored-section" }, [
    el("div", {}, [
      el("p", { className: "eyebrow", text: study.evidence_tier === "contract" ? "Qualified contract" : "Decision summary" }),
      el("h2", { text: study.findings[0] || "The planned evidence reconciled." })
    ]),
    el("aside", {}, [
      el("span", { text: "READ WITH" }),
      el("p", { text: study.caveats[0] || "No additional limitation is recorded." })
    ])
  ]);
}

function unresolvedDecision(study) {
  return el("section", { id: "decision", className: "decision-summary unresolved anchored-section" }, [
    el("div", {}, [
      el("p", { className: "eyebrow", text: "Unresolved question" }),
      el("h2", { text: study.question }),
      el("p", { text: `${study.metrics.expected_predictions} exact cells are planned.` })
    ]),
    el("aside", {}, [
      el("span", { text: study.evidence_tier === "blocked" ? "BLOCKER" : "EVIDENCE STATUS" }),
      el("p", { text: study.caveats[0] || "The planned denominator has not reconciled." })
    ])
  ]);
}

function resultLedgers(metrics) {
  return el("div", { className: "ledger-grid" }, [
    metric("Infrastructure", `${metrics.predictions}/${metrics.expected_predictions}`, "published attempts"),
    metric(
      "Deterministic",
      metrics.scored_predictions ? `${metrics.passed_predictions}/${metrics.scored_predictions}` : "Unavailable",
      "official outcomes"
    ),
    metric(
      "Usage evidence",
      metrics.measured_usage_predictions === metrics.predictions
        ? `${metrics.measured_usage_predictions}/${metrics.predictions}`
        : "Unavailable",
      `${metrics.measured_usage_predictions}/${metrics.predictions} measured`
    ),
    metric(
      "Cost evidence",
      metrics.measured_cost_predictions === metrics.predictions
        ? `$${formatNumber(metrics.total_cost_usd, { maximumFractionDigits: 2 })}`
        : "Unavailable",
      `${metrics.measured_cost_predictions}/${metrics.predictions} measured`
    ),
    metric(
      "Latency",
      metrics.measured_latency_predictions === metrics.predictions
        ? `${formatNumber(metrics.median_wall_time_sec, { maximumFractionDigits: 0 })}s`
        : "Unavailable",
      `${metrics.measured_latency_predictions}/${metrics.predictions} measured`
    ),
    metric(
      "Mechanism",
      metrics.tool_calls === null ? "Unavailable" : formatNumber(metrics.tool_calls),
      "verified tool calls"
    )
  ]);
}

function definitionGroup(title, rows) {
  return el("article", { className: "design-group" }, [
    el("h3", { text: title }),
    definitionList(rows)
  ]);
}

function definitionList(rows) {
  const list = el("dl", { className: "definition-grid" });
  for (const [term, value] of rows) {
    list.append(el("div", {}, [
      el("dt", { text: term }),
      el("dd", { text: value })
    ]));
  }
  return list;
}

function listSection(title, values, empty) {
  const list = el("ul", { className: "note-list" });
  for (const value of values.length ? values : [empty]) list.append(el("li", { text: value }));
  return el("div", {}, [
    el("p", { className: "eyebrow", text: title }),
    el("h2", { text: title }),
    list
  ]);
}

function taskEvidence(cells) {
  const details = el("details", { className: "task-evidence" });
  details.append(el("summary", { text: `Inspect ${cells.length} task-level predictions` }));
  const table = el("table", { className: "evidence-table" });
  table.append(el("caption", { text: "Safe task-level outcomes; raw Agent content remains in Weave" }));
  table.append(el("thead", {}, el("tr", {}, [
    "Task", "Model", "Harness", "Treatment", "Wire", "Route proof", "Outcome", "Time", "Cost", "Evidence"
  ].map((text) => el("th", { scope: "col", text })))));
  const body = el("tbody");
  for (const cell of cells) {
    body.append(el("tr", {}, [
      el("td", { text: cell.task_id }),
      el("td", { text: cell.model }),
      el("td", { text: cell.harness }),
      el("td", { text: cell.treatment }),
      el("td", { text: routeLabel(cell) }),
      el("td", { text: routeEvidenceLabel(cell.route_evidence) }),
      el("td", {
        text: cell.pass === true
          ? "Resolved"
          : cell.pass === false ? "Not resolved" : "Unscored"
      }),
      el("td", {
        text: cell.wall_time_sec === null
          ? "Unavailable"
          : `${formatNumber(cell.wall_time_sec, { maximumFractionDigits: 0 })}s`
      }),
      el("td", {
        text: cell.cost_usd === null
          ? "Unavailable"
          : `$${formatNumber(cell.cost_usd, { maximumFractionDigits: 2 })}`
      }),
      el(
        "td",
        {},
        cell.agent_link
          ? safeExternalLink("Evaluation — sign-in required", cell.agent_link)
          : el("span", { text: "Unavailable" })
      )
    ]));
  }
  table.append(body);
  details.append(el("div", { className: "table-scroll" }, table));
  return details;
}

function provenanceLinks(study) {
  const links = [];
  if (study.links.project) links.push(safeExternalLink("Open project in Weave — sign-in required", study.links.project));
  if (study.provenance.source_url) {
    links.push(safeExternalLink(
      study.provenance.source_commit === "pending" ? "Source repository" : "Source commit",
      study.provenance.source_url
    ));
  }
  for (const [index, href] of study.links.evaluations.entries()) {
    links.push(safeExternalLink(`Evaluation ${index + 1} — sign-in required`, href));
  }
  return links;
}

function updateCohortUrl(cohortId, setHash) {
  const url = new URL(window.location.href);
  url.searchParams.set("cohort", cohortId);
  if (setHash) url.hash = "analysis";
  window.history.replaceState({}, "", url);
}

function routeLabel(cell) {
  if (!cell.wire_protocol || !cell.endpoint_kind) return "Unavailable";
  const protocol = {
    chat_completions: "Chat Completions",
    messages: "Messages",
    responses: "Responses"
  }[cell.wire_protocol] || cell.wire_protocol;
  const endpoint = cell.endpoint_kind === "provider_direct" ? "direct" : "bridge";
  return `${protocol} · ${endpoint}`;
}

function routeEvidenceLabel(value) {
  return {
    runtime_attested: "Runtime attested",
    snapshot_attested: "Snapshot attested",
    configured_only: "Configured only"
  }[value] || "Unavailable";
}

function routingSummary(cells) {
  if (!cells.length) return "Pending";
  const counts = new Map();
  for (const cell of cells) {
    const label = routeEvidenceLabel(cell.route_evidence);
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  return [...counts.entries()]
    .map(([label, count]) => `${count} ${label.toLowerCase()}`)
    .join("; ");
}

function shortIdentity(value) {
  if (!value || value === "pending") return "Pending";
  return value.slice(0, 12);
}
