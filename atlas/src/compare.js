import { el, tierLabel } from "./common.js";
import { allExperiments, experimentById } from "./data.js";

const parameters = new URLSearchParams(window.location.search);
const requestedId = parameters.get("id") || "";
const requestedCohort = parameters.get("cohort") || "";
const requestedStudy = experimentById(requestedId);

if (requestedStudy) {
  const target = new URL("./experiment.html", window.location.href);
  target.searchParams.set("id", requestedStudy.id);
  if (requestedCohort) target.searchParams.set("cohort", requestedCohort);
  target.hash = "analysis";
  window.location.replace(target);
} else {
  const select = document.querySelector("#legacy-study");
  const link = document.querySelector("#legacy-open");
  for (const study of allExperiments()) {
    select.append(el("option", {
      value: study.id,
      text: `${study.title} · ${tierLabel(study.evidence_tier)}`
    }));
  }
  const updateLink = () => {
    link.href = `./experiment.html?id=${encodeURIComponent(select.value)}#analysis`;
  };
  select.addEventListener("change", updateLink);
  updateLink();
}
