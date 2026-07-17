import { el, formatNumber, formatPercent, metric, safeExternalLink, tierLabel } from "./common.js";
import { experimentById } from "./data.js";

const root = document.querySelector("#experiment-detail");
const id = new URLSearchParams(window.location.search).get("id") || "";
const experiment = experimentById(id);

if (!experiment) {
  root.append(el("section", { className: "empty-state" }, [
    el("p", { className: "eyebrow", text: "Not found" }),
    el("h1", { text: "This experiment is not in the reviewed index." }),
    el("a", { href: "./", className: "text-link", text: "Return to experiments" })
  ]));
} else {
  document.title = `${experiment.title} · Fugue Atlas`;
  const metrics = experiment.metrics;
  const header = el("section", { className: "experiment-hero ruled" }, [
    el("div", { className: "evidence-line" }, [
      el("span", { className: `tier tier-${experiment.evidence_tier}`, text: tierLabel(experiment.evidence_tier) }),
      el("span", { className: "status", text: experiment.status.replaceAll("_", " ") })
    ]),
    el("h1", { text: experiment.title }),
    el("p", { className: "lede", text: experiment.summary }),
    el("div", { className: "metric-strip" }, [
      metric("Published", `${metrics.predictions}/${metrics.expected_predictions}`, "predictions"),
      metric("Resolved", `${metrics.passed_predictions}/${metrics.scored_predictions}`, "official verifier"),
      metric("Pass rate", formatPercent(metrics.pass_rate), "no composite"),
      metric("Agent links", formatNumber(metrics.agent_links), "verified")
    ])
  ]);
  const rationale = el("section", { className: "narrative-grid" }, [
    el("div", {}, [el("p", { className: "eyebrow", text: "Question" }), el("h2", { text: experiment.question })]),
    el("div", {}, [el("h3", { text: "Hypothesis" }), el("p", { text: experiment.hypothesis }), el("h3", { text: "Why it matters" }), el("p", { text: experiment.why_it_matters })])
  ]);
  const matrix = el("section", { className: "section-block" }, [
    el("div", { className: "section-heading" }, [
      el("div", {}, [el("p", { className: "eyebrow", text: "Frozen matrix" }), el("h2", { text: "What varied—and what did not" })]),
      el("p", { text: experiment.task_selection })
    ]),
    definitionList([
      ["Models", experiment.matrix.models.join(", ")],
      ["Harnesses", experiment.matrix.harnesses.join(", ")],
      ["Treatments", experiment.matrix.treatments.join(", ")],
      ["Tasks", `${experiment.matrix.tasks.length} locked cases`],
      ["Attempts", String(experiment.matrix.attempts)],
      ["Workload", experiment.matrix.workload_id]
    ])
  ]);
  const evidence = el("section", { className: "section-block evidence-block" }, [
    el("p", { className: "eyebrow", text: "Evidence" }),
    el("h2", { text: metrics.predictions ? "Task-level outcomes" : "The score is still silent" }),
    metrics.predictions ? resultTable(experiment) : el("p", { className: "empty-copy", text: "No partial rows are public. This experiment remains visible for context, but it cannot be ranked until its exact planned denominator is reconciled." }),
    metrics.predictions ? taskEvidence(experiment.cells) : null,
    el("div", { className: "metric-strip compact" }, [
      metric("Cost", metrics.total_cost_usd === null ? "Unavailable" : `$${formatNumber(metrics.total_cost_usd, { maximumFractionDigits: 2 })}`, `${metrics.measured_cost_predictions}/${metrics.predictions} measured`),
      metric("Input tokens", formatNumber(metrics.input_tokens), "complete rows only"),
      metric("Output tokens", formatNumber(metrics.output_tokens), "complete rows only"),
      metric("Median time", metrics.median_wall_time_sec === null ? "Unavailable" : `${formatNumber(metrics.median_wall_time_sec, { maximumFractionDigits: 0 })}s`, "per prediction")
    ]),
    el("div", { className: "metric-strip compact" }, [
      metric("Tool calls", formatNumber(metrics.tool_calls), "complete rows only"),
      metric("Median turns", formatNumber(metrics.median_turns), "per prediction"),
      metric("Recoverable errors", formatNumber(metrics.recoverable_errors), "reported separately"),
      metric("Refusals", formatNumber(metrics.refusals), "not merged with misses")
    ])
  ]);
  const notes = el("section", { className: "notes-grid" }, [
    listSection("Findings", experiment.findings, "No result has been declared."),
    listSection("Caveats", experiment.caveats, "No additional caveats."),
    el("div", {}, [
      el("p", { className: "eyebrow", text: "Provenance" }),
      el("h2", { text: "Reproduce the evidence" }),
      definitionList([
        ["Source", experiment.provenance.source_commit.slice(0, 12)],
        ["Dataset", experiment.provenance.dataset_id],
        ["Dataset digest", experiment.provenance.dataset_digest.slice(0, 12)],
        ["Snapshot", experiment.provenance.snapshot_digest.slice(0, 12)],
        ["Run IDs", experiment.provenance.run_ids.join(", ") || "Pending"]
      ]),
      el("div", { className: "link-row" }, [
        safeExternalLink("Open project in Weave — sign-in required", experiment.links.project),
        safeExternalLink(experiment.provenance.source_commit === "pending" ? "Source repository" : "Source commit", experiment.provenance.source_url),
        ...experiment.links.evaluations.map((href, index) => safeExternalLink(`Evaluation ${index + 1} — sign-in required`, href))
      ])
    ])
  ]);
  root.append(header, rationale, matrix, evidence, notes);
}

function definitionList(rows) {
  const dl = el("dl", { className: "definition-grid" });
  for (const [term, value] of rows) dl.append(el("div", {}, [el("dt", { text: term }), el("dd", { text: value })]));
  return dl;
}

function listSection(title, values, empty) {
  const list = el("ul", { className: "note-list" });
  for (const value of values.length ? values : [empty]) list.append(el("li", { text: value }));
  return el("div", {}, [el("p", { className: "eyebrow", text: title }), el("h2", { text: title }), list]);
}

function resultTable(value) {
  const table = el("table", { className: "evidence-table" });
  table.append(el("caption", { text: "Official task outcomes by model, harness and treatment" }));
  table.append(el("thead", {}, el("tr", {}, ["Model", "Harness", "Treatment", "Resolved", "Trials"].map((text) => el("th", { scope: "col", text })))));
  const body = el("tbody");
  for (const group of value.groups) {
    body.append(el("tr", {}, [
      el("td", { text: group.model }), el("td", { text: group.harness }), el("td", { text: group.treatment }),
      el("td", { text: `${group.metrics.passed_predictions}/${group.metrics.scored_predictions}` }), el("td", { text: String(group.metrics.predictions) })
    ]));
  }
  table.append(body);
  return el("div", { className: "table-scroll" }, table);
}

function taskEvidence(cells) {
  const details = el("details", { className: "task-evidence" });
  details.append(el("summary", { text: `Inspect ${cells.length} task-level predictions` }));
  const table = el("table", { className: "evidence-table" });
  table.append(el("caption", { text: "Safe task-level outcomes; raw Agent content remains in Weave" }));
  table.append(el("thead", {}, el("tr", {}, ["Task", "Model", "Harness", "Treatment", "Outcome", "Time", "Cost", "Evidence"].map((text) => el("th", { scope: "col", text })))));
  const body = el("tbody");
  for (const cell of cells) body.append(el("tr", {}, [
    el("td", { text: cell.task_id }),
    el("td", { text: cell.model }),
    el("td", { text: cell.harness }),
    el("td", { text: cell.treatment }),
    el("td", { text: cell.pass === true ? "Resolved" : cell.pass === false ? "Not resolved" : "Unscored" }),
    el("td", { text: cell.wall_time_sec === null ? "Unavailable" : `${formatNumber(cell.wall_time_sec, { maximumFractionDigits: 0 })}s` }),
    el("td", { text: cell.cost_usd === null ? "Unavailable" : `$${formatNumber(cell.cost_usd, { maximumFractionDigits: 2 })}` }),
    el("td", {}, cell.agent_link ? safeExternalLink("Evaluation — sign-in required", cell.agent_link) : el("span", { text: "Not applicable" }))
  ]));
  table.append(body);
  details.append(el("div", { className: "table-scroll" }, table));
  return details;
}
