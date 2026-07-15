from __future__ import annotations

import json
from pathlib import Path

from fugue.bench.catalog import ExperimentCatalog


def test_catalog_filters_and_facets_skill_and_integration_variants(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "configs" / "fugue" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "study.yaml").write_text(
        """
id: study
title: Study
variants:
  - {id: baseline, label: Baseline}
  - id: treatment
    label: Treatment
    skills: [reviewed-skill]
    integrations: [search-api]
"""
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    rows = [
        {
            "record_type": "trial",
            "experiment_id": "study",
            "variant_id": "baseline",
            "run_id": "run-1",
            "task_name": "task-a",
            "harness": "codex",
            "skill_ids": [],
            "integration_ids": [],
            "reward": 0,
        },
        {
            "record_type": "trial",
            "experiment_id": "study",
            "variant_id": "treatment",
            "run_id": "run-1",
            "task_name": "task-a",
            "harness": "codex",
            "skill_ids": ["reviewed-skill"],
            "integration_ids": ["search-api"],
            "reward": 1,
        },
    ]
    (reports / "trials.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )

    catalog = ExperimentCatalog(tmp_path)
    status = catalog.refresh()
    integration_rows = catalog.records(filters={"integration_id": "search-api"})
    skill_rows = catalog.records(filters={"skill_id": "reviewed-skill"})
    facets = catalog.facets()

    assert status.records == 2
    assert [row["variant_id"] for row in integration_rows] == ["treatment"]
    assert [row["variant_id"] for row in skill_rows] == ["treatment"]
    assert facets["integration_id"] == {"none": 1, "search-api": 1}
    assert facets["skill_id"] == {"none": 1, "reviewed-skill": 1}
