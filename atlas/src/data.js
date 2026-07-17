import index from "../public/data/index.json";

const modules = import.meta.glob("../public/data/experiments/*.json", {
  eager: true,
  import: "default"
});

const experiments = new Map(
  Object.values(modules).map((experiment) => [experiment.id, experiment])
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
