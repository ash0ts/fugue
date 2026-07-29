import { el, tierLabel } from "./common.js";
import { allExperiments } from "./data.js";

const studies = allExperiments();
const root = document.querySelector("#study-groups");
const search = document.querySelector("#study-search");
const statusFilter = document.querySelector("#status-filter");
const tierFilter = document.querySelector("#tier-filter");
const summary = document.querySelector("#study-filter-summary");

for (const tier of [...new Set(studies.map((study) => study.evidence_tier))]) {
  tierFilter.append(el("option", { value: tier, text: tierLabel(tier) }));
}

search.addEventListener("input", render);
statusFilter.addEventListener("change", render);
tierFilter.addEventListener("change", render);
render();

function render() {
  const query = search.value.trim().toLowerCase();
  const visible = studies.filter((study) =>
    matchesStatus(study, statusFilter.value) &&
    (tierFilter.value === "all" || study.evidence_tier === tierFilter.value) &&
    (!query || searchText(study).includes(query))
  );

  root.replaceChildren();
  summary.textContent = `${visible.length} of ${studies.length} studies shown.`;

  const groups = [
    ["results", "Results available", "Complete evidence with a bounded finding and its limitation."],
    ["contract", "Contract qualification", "Canaries that qualify identity, routing, execution, or evidence plumbing."],
    ["incomplete", "Active or incomplete", "Unresolved questions shown without a winner or implied pass-rate conclusion."]
  ];

  for (const [groupId, title, description] of groups) {
    const members = visible.filter((study) => statusGroup(study) === groupId);
    if (!members.length) continue;
    root.append(
      el("section", { className: "study-group", "aria-labelledby": `group-${groupId}` }, [
        el("div", { className: "study-group-heading" }, [
          el("div", {}, [
            el("p", { className: "eyebrow", text: groupId.replaceAll("_", " ") }),
            el("h2", { id: `group-${groupId}`, text: title })
          ]),
          el("p", { text: description })
        ]),
        el("div", { className: "study-card-grid" }, members.map(studyCard))
      ])
    );
  }

  if (!visible.length) {
    root.append(el("section", { className: "empty-state compact" }, [
      el("h2", { text: "No studies match these filters." }),
      el("p", { text: "Clear the search or choose a different evidence state." })
    ]));
  }
}

function studyCard(study) {
  const complete = statusGroup(study) !== "incomplete";
  const contract = statusGroup(study) === "contract";
  const dimensions = [
    `${study.matrix.models.length} model${study.matrix.models.length === 1 ? "" : "s"}`,
    `${study.matrix.harnesses.length} harness${study.matrix.harnesses.length === 1 ? "" : "es"}`,
    `${study.matrix.treatments.length} treatment${study.matrix.treatments.length === 1 ? "" : "s"}`
  ].join(" · ");
  const denominator = complete
    ? `${study.metrics.predictions}/${study.metrics.expected_predictions} published`
    : `${study.metrics.expected_predictions} planned cells`;
  const finding = contract
    ? study.findings[0]
    : complete
      ? study.findings[0]
      : "No result has been declared.";
  const limitation = complete
    ? study.caveats[0]
    : blocker(study);

  return el("article", { className: `study-card study-${statusGroup(study)}` }, [
    el("div", { className: "evidence-line" }, [
      el("span", { className: `tier tier-${study.evidence_tier}`, text: tierLabel(study.evidence_tier) }),
      el("span", { className: "status", text: study.status.replaceAll("_", " ") })
    ]),
    el("h3", { text: study.title }),
    el("p", { className: "study-question", text: study.question }),
    el("dl", { className: "study-facts" }, [
      el("div", {}, [el("dt", { text: "Comparison" }), el("dd", { text: dimensions })]),
      el("div", {}, [el("dt", { text: "Denominator" }), el("dd", { text: denominator })]),
      el("div", {}, [el("dt", { text: complete ? "Supported finding" : "Evidence status" }), el("dd", { text: finding })]),
      el("div", {}, [el("dt", { text: complete ? "Limitation" : "Blocker" }), el("dd", { text: limitation })])
    ]),
    el("a", {
      className: "text-link",
      href: `./experiment.html?id=${encodeURIComponent(study.id)}`,
      text: "Open study →"
    })
  ]);
}

function statusGroup(study) {
  if (study.evidence_tier === "contract") return "contract";
  if (["active", "blocked"].includes(study.evidence_tier)) return "incomplete";
  return "results";
}

function matchesStatus(study, value) {
  return value === "all" || statusGroup(study) === value;
}

function searchText(study) {
  return [
    study.title,
    study.summary,
    study.question,
    study.matrix.models.join(" "),
    study.matrix.harnesses.join(" "),
    study.matrix.treatments.join(" "),
    tierLabel(study.evidence_tier)
  ].join(" ").toLowerCase();
}

function blocker(study) {
  if (study.caveats.length) return study.caveats[0];
  if (study.evidence_tier === "blocked") return "The planned denominator is incomplete.";
  return "The planned cells have not reconciled to public evidence.";
}
