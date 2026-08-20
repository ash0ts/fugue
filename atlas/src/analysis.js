import { el, formatPercent } from "./common.js";

export function renderStudyAnalysis(root, experiment, requestedCohortId = "") {
  root.replaceChildren();
  const cohort = experiment.matrix.cohorts.find((item) => item.id === requestedCohortId)
    || experiment.matrix.cohorts[0];

  if (!cohort) {
    root.append(el("p", { className: "empty-copy", text: "No compatible cohort is defined for this study." }));
    return null;
  }

  const view = compatibleView(experiment, cohort);
  root.append(
    el("div", { className: "analysis-receipt" }, [
      el("span", { text: cohort.label }),
      el("strong", { text: `${view.cells.length}/${cohort.expected_predictions} published` }),
      el("span", { text: `${cohort.tasks.length} tasks · ${experiment.matrix.attempts} attempt${experiment.matrix.attempts === 1 ? "" : "s"}` })
    ])
  );

  if (!view.cells.length) {
    root.append(el("section", { className: "empty-state compact" }, [
      el("h3", { text: "The question is still unresolved." }),
      el("p", { text: `${cohort.expected_predictions} cells are planned for this cohort; none have reconciled to public evidence.` })
    ]));
    return cohort.id;
  }

  root.append(taskCounterpoint(experiment, view), groupOutcomeTable(view));

  const lift = pairedLift(view);
  if (lift) root.append(lift);

  const frontier = costLatencyFrontier(view);
  if (frontier) root.append(frontier);

  return cohort.id;
}

export function studyCohorts(experiment) {
  return experiment.matrix.cohorts || [];
}

function taskCounterpoint(experiment, view) {
  const harnesses = view.harnesses;
  const tasks = view.tasks;
  const voices = view.treatments.length > 1 ? view.treatments : view.models;
  const width = Math.max(720, 170 + tasks.length * 96);
  const height = 100 + harnesses.length * 92;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Task outcomes for ${experiment.title}: ${harnesses.length} harness lanes across ${tasks.length} tasks.`);
  svg.classList.add("counterpoint-svg");

  tasks.forEach((task, taskIndex) => {
    svg.append(svgText(154 + taskIndex * 96, 32, shortTask(task), "task-label", "middle"));
  });
  harnesses.forEach((harness, laneIndex) => {
    const y = 82 + laneIndex * 92;
    svg.append(
      svgText(12, y + 5, harness, "lane-label", "start"),
      svgNode("line", { x1: 148, x2: width - 28, y1: y, y2: y, class: "staff-line" })
    );
    voices.forEach((voice, voiceIndex) => {
      const points = [];
      const marks = [];
      tasks.forEach((task, taskIndex) => {
        const x = 154 + taskIndex * 96;
        const row = view.cells.find((cell) =>
          cell.harness === harness &&
          cell.task_id === task &&
          (view.treatments.length > 1 ? cell.treatment === voice : cell.model === voice)
        );
        const offset = (voiceIndex - (voices.length - 1) / 2) * 10;
        points.push(`${x},${y + offset}`);
        const state = row
          ? row.pass === true ? "pass" : row.pass === false ? "fail" : "unscored"
          : "pending";
        const mark = svgNode("circle", { cx: x, cy: y + offset, r: 5.5, class: `note ${state}` });
        mark.append(svgNode("title", {}, `${harness} · ${voice} · ${task}: ${outcomeLabel(row)}`));
        marks.push(mark);
      });
      svg.append(svgNode("polyline", { points: points.join(" "), class: `voice-line voice-${voiceIndex % 5}` }));
      svg.append(...marks);
    });
  });

  const legend = el(
    "div",
    { className: "ribbon-legend" },
    voices.map((voice, index) => el("span", {}, [
      el("i", { className: `voice-swatch voice-${index % 5}` }),
      el("span", { text: voice })
    ]))
  );

  return el("section", { className: "ribbon-panel" }, [
    el("div", { className: "ribbon-heading" }, [
      el("div", {}, [
        el("p", { className: "eyebrow", text: "Task-level outcomes" }),
        el("h3", { text: "Inspect the rows before the aggregate." })
      ]),
      el("p", { text: "Each lane is a harness. Green resolves, coral does not resolve, and hollow marks are unavailable—not zero." })
    ]),
    legend,
    el("div", { className: "ribbon-scroll" }, svg),
    accessibleTaskAlternative(view)
  ]);
}

function groupOutcomeTable(view) {
  const table = el("table", { className: "evidence-table" });
  table.append(el("caption", { text: "Aggregate outcomes for compatible groups only" }));
  table.append(el("thead", {}, el("tr", {}, [
    "Model", "Harness", "Treatment", "Resolved", "Pass rate", "Median time"
  ].map((text) => el("th", { scope: "col", text })))));
  const body = el("tbody");
  for (const group of view.groups) {
    body.append(el("tr", {}, [
      el("td", { text: group.model }),
      el("td", { text: group.harness }),
      el("td", { text: group.treatment }),
      el("td", { text: scoredFraction(group.metrics) }),
      el("td", { text: formatPercent(group.metrics.pass_rate) }),
      el("td", {
        text: group.metrics.median_wall_time_sec === null
          ? "Unavailable"
          : `${Math.round(group.metrics.median_wall_time_sec)}s`
      })
    ]));
  }
  table.append(body);
  return el("section", { className: "analysis-panel" }, [
    el("div", { className: "analysis-panel-heading" }, [
      el("p", { className: "eyebrow", text: "Compatible aggregate" }),
      el("h3", { text: "Group outcomes" })
    ]),
    el("div", { className: "table-scroll" }, table)
  ]);
}

function pairedLift(view) {
  const baseline = ["none", "baseline"].find((value) => view.treatments.includes(value));
  if (!baseline || !view.treatments.some((value) => value !== baseline)) return null;

  const rows = [];
  const expectedPairs = view.tasks.length * view.attempts;
  for (const model of view.models) {
    for (const harness of view.harnesses) {
      const control = taskOutcomes(view.cells, model, harness, baseline);
      for (const treatment of view.treatments.filter((value) => value !== baseline)) {
        const candidate = taskOutcomes(view.cells, model, harness, treatment);
        const paired = [...control.keys()].filter((key) => candidate.has(key));
        if (paired.length !== expectedPairs) continue;
        const lift = paired.reduce(
          (sum, key) => sum + candidate.get(key) - control.get(key),
          0
        ) / paired.length;
        rows.push({ model, harness, baseline, treatment, lift, paired: paired.length });
      }
    }
  }
  if (!rows.length) return null;

  const table = el("table", { className: "evidence-table" });
  table.append(el("caption", { text: "Paired resolution lift for complete treatment and control coordinates" }));
  table.append(el("thead", {}, el("tr", {}, [
    "Model", "Harness", "Treatment", "Control", "Paired cells", "Resolution lift"
  ].map((text) => el("th", { scope: "col", text })))));
  const body = el("tbody");
  for (const row of rows) {
    body.append(el("tr", {}, [
      el("td", { text: row.model }),
      el("td", { text: row.harness }),
      el("td", { text: row.treatment }),
      el("td", { text: row.baseline }),
      el("td", { text: String(row.paired) }),
      el("td", { text: `${row.lift >= 0 ? "+" : ""}${formatPercent(row.lift)}` })
    ]));
  }
  table.append(body);
  return el("section", { className: "analysis-panel" }, [
    el("div", { className: "analysis-panel-heading" }, [
      el("p", { className: "eyebrow", text: "Paired intervention" }),
      el("h3", { text: "Lift against the exact control" }),
      el("p", { text: "Every displayed delta has a complete model, harness, task, trial, treatment, and control pair." })
    ]),
    el("div", { className: "table-scroll" }, table)
  ]);
}

function costLatencyFrontier(view) {
  if (!view.groups.length || !view.groups.every(hasCompleteEfficiency)) return null;
  const points = view.groups;
  const width = 760;
  const height = 390;
  const margin = { left: 72, right: 28, top: 30, bottom: 66 };
  const maxCost = Math.max(...points.map((point) => point.metrics.mean_cost_usd)) || 1;
  const maxTime = Math.max(...points.map((point) => point.metrics.median_wall_time_sec)) || 1;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Cost and latency for groups with complete measurement denominators.");
  svg.classList.add("frontier-svg");
  svg.append(
    svgNode("line", { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: "axis-line" }),
    svgNode("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, class: "axis-line" }),
    svgText(width / 2, height - 20, "Mean measured cost (USD)", "axis-label", "middle"),
    svgText(margin.left, 18, "Median wall time ↑", "axis-label", "start")
  );
  for (const point of points) {
    const x = margin.left + (point.metrics.mean_cost_usd / maxCost) * (width - margin.left - margin.right);
    const y = height - margin.bottom -
      (point.metrics.median_wall_time_sec / maxTime) * (height - margin.top - margin.bottom);
    const circle = svgNode("circle", {
      cx: x,
      cy: y,
      r: 8,
      class: point.metrics.pass_rate > 0 ? "frontier-point pass" : "frontier-point fail"
    });
    circle.append(svgNode("title", {}, `${point.model} · ${point.harness} · ${point.treatment}: $${point.metrics.mean_cost_usd.toFixed(2)}, ${Math.round(point.metrics.median_wall_time_sec)}s, ${formatPercent(point.metrics.pass_rate)} resolved`));
    svg.append(circle);
  }
  return el("section", { className: "analysis-panel" }, [
    el("div", { className: "analysis-panel-heading" }, [
      el("p", { className: "eyebrow", text: "Complete efficiency evidence" }),
      el("h3", { text: "Cost and latency" }),
      el("p", { text: "Shown only because every compatible group has a complete measurement denominator." })
    ]),
    el("div", { className: "frontier-chart" }, svg),
    efficiencyAlternative(points)
  ]);
}

function compatibleView(experiment, cohort) {
  const cells = experiment.cells.filter((cell) =>
    cohort.models.includes(cell.model) &&
    cohort.harnesses.includes(cell.harness) &&
    cohort.treatments.includes(cell.treatment) &&
    cohort.tasks.includes(cell.task_id)
  );
  const groups = experiment.groups.filter((group) =>
    cohort.models.includes(group.model) &&
    cohort.harnesses.includes(group.harness) &&
    cohort.treatments.includes(group.treatment)
  );
  return {
    cells,
    groups,
    models: cohort.models,
    harnesses: cohort.harnesses,
    treatments: cohort.treatments,
    tasks: cohort.tasks,
    attempts: experiment.matrix.attempts,
    expectedPredictions: cohort.expected_predictions
  };
}

function taskOutcomes(cells, model, harness, treatment) {
  return new Map(cells
    .filter((cell) =>
      cell.model === model &&
      cell.harness === harness &&
      cell.treatment === treatment &&
      cell.pass !== null
    )
    .map((cell) => [`${cell.comparison_example_id}:${cell.trial_index}`, cell.pass ? 1 : 0]));
}

function accessibleTaskAlternative(view) {
  const details = el("details", { className: "chart-alternative" });
  details.append(el("summary", { text: "Text alternative for task outcomes" }));
  const list = el("ul");
  for (const cell of view.cells) {
    list.append(el("li", {
      text: `${cell.harness}, ${cell.treatment}, ${cell.task_id}: ${outcomeLabel(cell)}.`
    }));
  }
  details.append(list);
  return details;
}

function efficiencyAlternative(points) {
  const details = el("details", { className: "chart-alternative" });
  details.append(el("summary", { text: "Text alternative for cost and latency" }));
  const list = el("ul");
  for (const point of points) {
    list.append(el("li", {
      text: `${point.model}, ${point.harness}, ${point.treatment}: $${point.metrics.mean_cost_usd.toFixed(2)} mean cost; ${Math.round(point.metrics.median_wall_time_sec)} seconds median; ${formatPercent(point.metrics.pass_rate)} resolved.`
    }));
  }
  details.append(list);
  return details;
}

function hasCompleteEfficiency(group) {
  const metrics = group.metrics;
  return metrics.mean_cost_usd !== null &&
    metrics.median_wall_time_sec !== null &&
    metrics.measured_cost_predictions === metrics.predictions &&
    metrics.measured_latency_predictions === metrics.predictions;
}

function scoredFraction(metrics) {
  return metrics.scored_predictions
    ? `${metrics.passed_predictions}/${metrics.scored_predictions}`
    : "Unavailable";
}

function outcomeLabel(row) {
  if (!row) return "not published";
  if (row.pass === true) return "resolved";
  if (row.pass === false) return "not resolved";
  return "unscored";
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
