import { el, safeExternalLink } from "./common.js";
import { evidenceInventory } from "./data.js";

const entries = evidenceInventory().entries;
const root = document.querySelector("#inventory-ledger");
const search = document.querySelector("#inventory-search");
const summary = document.querySelector("#inventory-summary");
const buttons = [...document.querySelectorAll("[data-category]")];
let category = "all";

search.addEventListener("input", render);
for (const button of buttons) {
  button.addEventListener("click", () => {
    category = button.dataset.category;
    for (const item of buttons) item.setAttribute("aria-pressed", String(item === button));
    render();
  });
}
render();

function render() {
  const query = search.value.trim().toLowerCase();
  const visible = entries.filter((entry) =>
    (category === "all" || entry.category === category) &&
    (!query || searchable(entry).includes(query))
  );
  summary.textContent = `${visible.length} of ${entries.length} reviewed entries shown · ${runCount(entries)} canonical run IDs inventoried.`;
  root.replaceChildren(...visible.map(inventoryRow));
}

function inventoryRow(entry, index) {
  const identity = entry.run_ids.length ? entry.run_ids.join(", ") : "No run ID — not executed";
  const links = [];
  for (const studyId of entry.study_ids) {
    links.push(el("a", { className: "text-link", href: `./experiment.html?id=${encodeURIComponent(studyId)}`, text: "Open Atlas record" }));
  }
  if (entry.source_reference) links.push(safeExternalLink("Inspect source", entry.source_reference));
  return el("article", { className: `inventory-row inventory-${entry.category}` }, [
    el("div", { className: "inventory-index", text: String(index + 1).padStart(2, "0") }),
    el("div", { className: "inventory-main" }, [
      el("div", { className: "evidence-line" }, [
        el("span", { className: "inventory-category", text: entry.category.replaceAll("_", " ") }),
        el("span", { className: "status", text: entry.lifecycle_state.replaceAll("_", " ") }),
        el("span", { className: "status", text: entry.publication_level.replaceAll("_", " ") })
      ]),
      el("h2", { text: entry.label }),
      el("code", { className: "inventory-identity", text: identity }),
      el("p", { text: entry.claim_boundary })
    ]),
    el("div", { className: "inventory-links" }, links)
  ]);
}

function searchable(entry) {
  return [entry.id, entry.label, entry.category, entry.lifecycle_state, entry.publication_level, ...entry.run_ids, entry.claim_boundary].join(" ").toLowerCase();
}

function runCount(values) {
  return values.reduce((count, entry) => count + entry.run_ids.length, 0);
}
