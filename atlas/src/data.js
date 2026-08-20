import index from "../public/data/index.json";
import inventory from "../public/data/inventory.json";
import product from "../public/data/product.json";

const modules = import.meta.glob("../public/data/experiments/*.json", {
  eager: true,
  import: "default"
});

const experiments = new Map(
  Object.values(modules).map((experiment) => [experiment.id, normalizeExperiment(experiment)])
);

export function experimentIndex() {
  return index;
}

export function allExperiments() {
  return index.experiments.map((item) => experiments.get(item.id)).filter(Boolean);
}

export function experimentById(id) {
  return experiments.get(id);
}

export function productInfo() {
  return product;
}

export function evidenceInventory() {
  return inventory;
}

function normalizeExperiment(experiment) {
  if (experiment.schema_version === 1) {
    return {
      ...experiment,
      study_kind: experiment.evidence_tier === "contract" ? "contract" : "benchmark",
      publication_level: experiment.cells.length ? "full" : "summary",
      primary_outcome: {
        id: "official_verifier_resolution",
        label: "Official verifier resolution",
        success_label: "Resolved",
        failure_label: "Not resolved"
      }
    };
  }
  return {
    ...experiment,
    metrics: {
      ...experiment.metrics,
      expected_predictions: experiment.metrics.expected_cells,
      predictions: experiment.metrics.published_cells,
      scored_predictions: experiment.metrics.outcome_observed_cells,
      passed_predictions: experiment.metrics.outcome_successes,
      pass_rate: experiment.metrics.outcome_rate,
      agent_links: experiment.cells.filter((cell) => cell.evidence_link).length
    }
  };
}
