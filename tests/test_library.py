from __future__ import annotations

import pytest

from fugue.bench.library import (
    get_experiment,
    get_prompt,
    save_experiment,
    save_prompt,
    validate_id,
)


def test_prompt_save_reload_and_hash(tmp_path):
    item = save_prompt("prompt-a", "# Prompt A\n\nUse tests.\n", tmp_path)

    loaded = get_prompt("prompt-a", tmp_path)

    assert loaded.body == "# Prompt A\n\nUse tests.\n"
    assert loaded.title == "Prompt A"
    assert loaded.sha256 == item.sha256
    assert len(loaded.sha256) == 64


def test_experiment_save_reload_with_feature_variants(tmp_path):
    save_experiment(
        "experiment-a",
        """
id: experiment-a
title: Experiment A
manifest: datasets/pilot.yaml
variants:
  - id: baseline
    label: Baseline
    memory: none
  - id: prompt-skill
    label: Prompt plus skill
    prompt_id: prompt-a
    skill_ids: [skill-a]
    memory: agentsmd
    agent_kwargs:
      temperature: 0
""",
        tmp_path,
    )

    experiment = get_experiment("experiment-a", tmp_path)

    assert experiment.id == "experiment-a"
    assert [variant.id for variant in experiment.variants] == [
        "baseline",
        "prompt-skill",
    ]
    assert [variant.memory for variant in experiment.variants] == ["none", "agentsmd"]
    assert experiment.variants[1].prompt_id == "prompt-a"
    assert experiment.variants[1].skill_ids == ["skill-a"]
    assert experiment.variants[1].agent_kwargs == {"temperature": 0}


def test_experiment_defaults_to_baseline_variant(tmp_path):
    save_experiment(
        "experiment-b",
        """
id: experiment-b
title: Experiment B
manifest: datasets/pilot.yaml
""",
        tmp_path,
    )

    experiment = get_experiment("experiment-b", tmp_path)

    assert experiment.id == "experiment-b"
    assert [variant.id for variant in experiment.variants] == ["baseline"]
    assert experiment.variants[0].memory == "none"


def test_ids_reject_path_traversal():
    with pytest.raises(ValueError):
        validate_id("../secret", kind="prompt id")
