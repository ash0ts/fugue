import { el, formatPercent, tierLabel } from "./common.js";
import { experimentIndex } from "./data.js";

const list = document.querySelector("#experiment-list");
const index = experimentIndex();

for (const [position, experiment] of index.experiments.entries()) {
  const href = `./experiment.html?id=${encodeURIComponent(experiment.id)}`;
  const metrics = experiment.metrics;
  const outcome = metrics.scored_predictions
    ? `${metrics.passed_predictions}/${metrics.scored_predictions} resolved`
    : `${metrics.predictions}/${metrics.expected_predictions} published`;
  const article = el("article", { className: "score-entry" }, [
    el("div", { className: "score-index", text: ["active", "blocked"].includes(experiment.evidence_tier) ? "—" : String(position + 1).padStart(2, "0") }),
    el("div", { className: "score-main" }, [
      el("div", { className: "evidence-line" }, [
        el("span", { className: `tier tier-${experiment.evidence_tier}`, text: tierLabel(experiment.evidence_tier) }),
        el("span", { className: "status", text: experiment.status.replaceAll("_", " ") })
      ]),
      el("h3", {}, el("a", { href, text: experiment.title })),
      el("p", { text: experiment.summary }),
      el("div", { className: "voice-list" }, [
        el("span", { text: `${experiment.models.length} model${experiment.models.length === 1 ? "" : "s"}` }),
        el("span", { text: `${experiment.harnesses.length} harness${experiment.harnesses.length === 1 ? "" : "es"}` }),
        el("span", { text: `${experiment.treatments.length} voice${experiment.treatments.length === 1 ? "" : "s"}` })
      ])
    ]),
    el("div", { className: "score-result" }, [
      el("span", { className: "result-primary", text: outcome }),
      el("span", { text: metrics.pass_rate === null ? "No result yet" : `${formatPercent(metrics.pass_rate)} pass rate` }),
      el("a", { href, className: "text-link", text: "Read the score →" })
    ])
  ]);
  list.append(article);
}
