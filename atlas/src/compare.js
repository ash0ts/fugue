import { el, formatPercent, tierLabel } from "./common.js";
import { allExperiments } from "./data.js";

const select = document.querySelector("#experiment-select");
const cohortSelect = document.querySelector("#cohort-select");
const root = document.querySelector("#comparison");
const experiments = allExperiments();

for (const experiment of experiments) {
  select.append(el("option", { value: experiment.id, text: `${experiment.title} · ${tierLabel(experiment.evidence_tier)}` }));
}
select.addEventListener("change", () => {
  populateCohorts();
  render();
});
cohortSelect.addEventListener("change", render);
populateCohorts();
render();

function populateCohorts() {
  cohortSelect.replaceChildren();
  const experiment = experiments.find((item) => item.id === select.value) || experiments[0];
  for (const cohort of experiment?.matrix.cohorts || []) {
    cohortSelect.append(el("option", { value: cohort.id, text: cohort.label }));
  }
}

function render() {
  root.replaceChildren();
  const experiment = experiments.find((item) => item.id === select.value) || experiments[0];
  if (!experiment) return;
  const cohort = experiment.matrix.cohorts.find((item) => item.id === cohortSelect.value) || experiment.matrix.cohorts[0];
  const view = compatibleView(experiment, cohort);
  root.append(
    el("section", { className: "comparison-header" }, [
      el("div", {}, [el("p", { className: "eyebrow", text: experiment.matrix.workload_id }), el("h2", { text: experiment.title })]),
      el("p", { text: `${cohort.label} · ${cohort.tasks.length} tasks · ${experiment.matrix.attempts} attempt${experiment.matrix.attempts === 1 ? "" : "s"} · ${view.cells.length}/${cohort.expected_predictions} published` })
    ]),
    counterpoint(experiment, view),
    pairedLift(view),
    costLatencyFrontier(view),
    groupTable(view)
  );
}

function counterpoint(experiment, view) {
  const harnesses = view.harnesses;
  const tasks = view.tasks;
  const voices = view.treatments.length > 1 ? view.treatments : view.models;
  const width = Math.max(720, 170 + tasks.length * 96);
  const height = 100 + harnesses.length * 92;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Counterpoint ribbon for ${experiment.title}: harness lanes with ${voices.join(", ")} voices across ${tasks.length} tasks.`);
  svg.classList.add("counterpoint-svg");
  tasks.forEach((task, taskIndex) => {
    svg.append(svgText(154 + taskIndex * 96, 32, shortTask(task), "task-label", "middle"));
  });
  harnesses.forEach((harness, laneIndex) => {
    const y = 82 + laneIndex * 92;
    svg.append(svgText(12, y + 5, harness, "lane-label", "start"));
    const line = svgNode("line", { x1: 148, x2: width - 28, y1: y, y2: y, class: "staff-line" });
    svg.append(line);
    voices.forEach((voice, voiceIndex) => {
      const points = [];
      const marks = [];
      tasks.forEach((task, taskIndex) => {
        const x = 154 + taskIndex * 96;
        const row = view.cells.find((cell) => cell.harness === harness && cell.task_id === task && (view.treatments.length > 1 ? cell.treatment === voice : cell.model === voice));
        const offset = (voiceIndex - (voices.length - 1) / 2) * 10;
        points.push(`${x},${y + offset}`);
        const mark = svgNode("circle", { cx: x, cy: y + offset, r: 5.5, class: row ? (row.pass === true ? "note pass" : row.pass === false ? "note fail" : "note unscored") : "note pending" });
        mark.append(svgNode("title", {}, `${harness} · ${voice} · ${task}: ${row ? row.pass === true ? "resolved" : row.pass === false ? "not resolved" : "unscored" : "not published"}`));
        marks.push(mark);
      });
      svg.append(svgNode("polyline", { points: points.join(" "), class: `voice-line voice-${voiceIndex % 5}` }));
      svg.append(...marks);
    });
  });
  const legend = el("div", { className: "ribbon-legend" }, voices.map((voice, index) => el("span", {}, [el("i", { className: `voice-swatch voice-${index % 5}` }), el("span", { text: voice })])));
  const table = accessibleRibbonTable(view);
  return el("section", { className: "ribbon-panel" }, [
    el("div", { className: "ribbon-heading" }, [el("div", {}, [el("p", { className: "eyebrow", text: "Signature view" }), el("h2", { text: "Task-level counterpoint ribbon" })]), el("p", { text: "Each lane is a harness. Each voice is a treatment or model. Green notes resolve; coral notes miss; hollow notes are not yet public." })]),
    legend,
    el("div", { className: "ribbon-scroll" }, svg),
    table
  ]);
}

function groupTable(view) {
  if (!view.groups.length) return el("section", { className: "empty-state compact" }, [el("h2", { text: "No partial comparison" }), el("p", { text: "The planned staves are visible, but the atlas will not rank an active or blocked cohort." })]);
  const table = el("table", { className: "evidence-table" });
  table.append(el("caption", { text: "Compatible group outcomes" }));
  table.append(el("thead", {}, el("tr", {}, ["Model", "Harness", "Treatment", "Resolved", "Pass rate", "Median time"].map((text) => el("th", { scope: "col", text })))));
  const body = el("tbody");
  for (const group of view.groups) body.append(el("tr", {}, [
    el("td", { text: group.model }), el("td", { text: group.harness }), el("td", { text: group.treatment }),
    el("td", { text: `${group.metrics.passed_predictions}/${group.metrics.scored_predictions}` }),
    el("td", { text: formatPercent(group.metrics.pass_rate) }),
    el("td", { text: group.metrics.median_wall_time_sec === null ? "Unavailable" : `${Math.round(group.metrics.median_wall_time_sec)}s` })
  ]));
  table.append(body);
  return el("section", { className: "section-block" }, el("div", { className: "table-scroll" }, table));
}

function pairedLift(view) {
  const baseline = ["none", "baseline"].find((value) => view.treatments.includes(value));
  const rows = [];
  if (baseline) {
    for (const model of view.models) for (const harness of view.harnesses) {
      const control = taskOutcomes(view.cells, model, harness, baseline);
      for (const treatment of view.treatments.filter((value) => value !== baseline)) {
        const candidate = taskOutcomes(view.cells, model, harness, treatment);
        const paired = [...control.keys()].filter((key) => candidate.has(key));
        if (!paired.length) continue;
        const lift = paired.reduce((sum, key) => sum + candidate.get(key) - control.get(key), 0) / paired.length;
        rows.push({ model, harness, baseline, treatment, lift, paired: paired.length });
      }
    }
  }
  const section = el("section", { className: "section-block analysis-panel" }, [
    el("div", { className: "section-heading" }, [
      el("div", {}, [el("p", { className: "eyebrow", text: "Paired intervention" }), el("h2", { text: "Memory lift against the exact baseline" })]),
      el("p", { text: "Each delta compares the same model, harness, task and trial. One-attempt lift is directional and carries no confidence claim." })
    ])
  ]);
  if (!rows.length) {
    const hasIntervention = baseline && view.treatments.some((value) => value !== baseline);
    section.append(el("p", { className: "empty-copy", text: hasIntervention ? "No complete treatment pairs are public yet." : "This cohort has no intervention, so paired lift is not applicable." }));
    return section;
  }
  const table = el("table", { className: "evidence-table" });
  table.append(el("caption", { text: "Paired resolution lift by compatible model and harness" }));
  table.append(el("thead", {}, el("tr", {}, ["Model", "Harness", "Treatment", "Baseline", "Paired tasks", "Resolution lift"].map((text) => el("th", { scope: "col", text })))));
  const body = el("tbody");
  for (const row of rows) body.append(el("tr", {}, [
    el("td", { text: row.model }), el("td", { text: row.harness }), el("td", { text: row.treatment }),
    el("td", { text: row.baseline }), el("td", { text: String(row.paired) }),
    el("td", { text: `${row.lift >= 0 ? "+" : ""}${formatPercent(row.lift)}` })
  ]));
  table.append(body);
  section.append(el("div", { className: "table-scroll" }, table));
  return section;
}

function costLatencyFrontier(view) {
  const points = view.groups.filter((group) => group.metrics.mean_cost_usd !== null && group.metrics.median_wall_time_sec !== null);
  const section = el("section", { className: "section-block analysis-panel" }, [
    el("div", { className: "section-heading" }, [
      el("div", {}, [el("p", { className: "eyebrow", text: "Efficiency" }), el("h2", { text: "Cost and latency frontier" })]),
      el("p", { text: "Lower and farther left is cheaper and faster. Resolution is encoded by color, never blended into the axes." })
    ])
  ]);
  if (!points.length) {
    section.append(el("p", { className: "empty-copy", text: "No compatible group has complete measured cost and latency yet." }));
    return section;
  }
  const width = 760;
  const height = 390;
  const margin = { left: 72, right: 28, top: 30, bottom: 66 };
  const maxCost = Math.max(...points.map((point) => point.metrics.mean_cost_usd)) || 1;
  const maxTime = Math.max(...points.map((point) => point.metrics.median_wall_time_sec)) || 1;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Cost and latency frontier for compatible model, harness and treatment groups.");
  svg.classList.add("frontier-svg");
  svg.append(
    svgNode("line", { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: "axis-line" }),
    svgNode("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, class: "axis-line" }),
    svgText(width / 2, height - 20, "Mean measured cost (USD)", "axis-label", "middle"),
    svgText(margin.left, 18, "Median wall time ↑", "axis-label", "start")
  );
  for (const point of points) {
    const x = margin.left + (point.metrics.mean_cost_usd / maxCost) * (width - margin.left - margin.right);
    const y = height - margin.bottom - (point.metrics.median_wall_time_sec / maxTime) * (height - margin.top - margin.bottom);
    const circle = svgNode("circle", { cx: x, cy: y, r: 8, class: point.metrics.pass_rate > 0 ? "frontier-point pass" : "frontier-point fail" });
    circle.append(svgNode("title", {}, `${point.model} · ${point.harness} · ${point.treatment}: $${point.metrics.mean_cost_usd.toFixed(2)}, ${Math.round(point.metrics.median_wall_time_sec)}s, ${formatPercent(point.metrics.pass_rate)} resolved`));
    svg.append(circle);
  }
  const details = el("details", { className: "chart-alternative" });
  details.append(el("summary", { text: "Text alternative for cost and latency frontier" }));
  const list = el("ul");
  for (const point of points) list.append(el("li", { text: `${point.model}, ${point.harness}, ${point.treatment}: $${point.metrics.mean_cost_usd.toFixed(2)} mean cost; ${Math.round(point.metrics.median_wall_time_sec)} seconds median; ${formatPercent(point.metrics.pass_rate)} resolved.` }));
  details.append(list);
  section.append(el("div", { className: "frontier-chart" }, svg), details);
  return section;
}

function taskOutcomes(cells, model, harness, treatment) {
  return new Map(cells
    .filter((cell) => cell.model === model && cell.harness === harness && cell.treatment === treatment && cell.pass !== null)
    .map((cell) => [`${cell.comparison_example_id}:${cell.trial_index}`, cell.pass ? 1 : 0]));
}

function accessibleRibbonTable(view) {
  const details = el("details", { className: "chart-alternative" });
  details.append(el("summary", { text: "Text alternative for counterpoint ribbon" }));
  const list = el("ul");
  if (!view.cells.length) list.append(el("li", { text: `${view.expectedPredictions} predictions are planned; none are public yet.` }));
  for (const cell of view.cells) list.append(el("li", { text: `${cell.harness}, ${cell.treatment}, ${cell.task_id}: ${cell.pass === true ? "resolved" : cell.pass === false ? "not resolved" : "unscored"}.` }));
  details.append(list);
  return details;
}

function svgNode(tag, attributes = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text) node.textContent = text;
  return node;
}

function svgText(x, y, text, className, anchor) {
  return svgNode("text", { x, y, class: className, "text-anchor": anchor }, text);
}

function shortTask(task) {
  const [repo, issue] = task.split("__");
  return issue ? `${repo.replace("-doc", "")}·${issue.split("-").at(-1)}` : task;
}

function compatibleView(experiment, cohort) {
  const cells = experiment.cells.filter((cell) => cohort.models.includes(cell.model) && cohort.harnesses.includes(cell.harness) && cohort.treatments.includes(cell.treatment) && cohort.tasks.includes(cell.task_id));
  const groups = experiment.groups.filter((group) => cohort.models.includes(group.model) && cohort.harnesses.includes(group.harness) && cohort.treatments.includes(group.treatment));
  return {
    cells,
    groups,
    models: cohort.models,
    harnesses: cohort.harnesses,
    treatments: cohort.treatments,
    tasks: cohort.tasks,
    expectedPredictions: cohort.expected_predictions
  };
}
