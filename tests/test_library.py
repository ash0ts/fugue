from __future__ import annotations

import pytest

from fugue.bench.library import (
    experiment_from_data,
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
    context: {system_id: none}
  - id: prompt-skill
    label: Prompt plus skill
    prompt_id: prompt-a
    skill_ids: [skill-a]
    context:
      system_id: agentsmd
      config: {depth: 2}
    agent_kwargs:
      temperature: 0
workloads:
  - id: coding
    runner: harbor
    manifest: datasets/pilot.yaml
presets:
  smoke:
    workloads: [coding]
    workload_overrides:
      coding: {n_tasks: 1}
""",
        tmp_path,
    )

    experiment = get_experiment("experiment-a", tmp_path)

    assert experiment.id == "experiment-a"
    assert [variant.id for variant in experiment.variants] == [
        "baseline",
        "prompt-skill",
    ]
    assert [variant.context.system_id for variant in experiment.variants] == [
        "none",
        "agentsmd",
    ]
    assert experiment.variants[1].context.config == {"depth": 2}
    assert experiment.variants[1].prompt_id == "prompt-a"
    assert experiment.variants[1].skill_ids == ["skill-a"]
    assert experiment.variants[1].agent_kwargs == {"temperature": 0}
    assert experiment.presets[0].workload_overrides == {"coding": {"n_tasks": 1}}


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
    assert experiment.variants[0].context.system_id == "none"


def test_unknown_variant_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown variant field.*memory"):
        save_experiment(
            "old-experiment",
            """
id: old-experiment
variants:
  - id: baseline
    memory: none
""",
            tmp_path,
        )


@pytest.mark.parametrize("field", ["k", "invented"])
def test_unknown_experiment_fields_are_rejected(tmp_path, field):
    with pytest.raises(ValueError, match=f"unknown experiment field.*{field}"):
        save_experiment(
            "strict",
            f"""
id: strict
{field}: 2
""",
            tmp_path,
        )


def test_preset_rejects_unknown_workload_override(tmp_path):
    with pytest.raises(ValueError, match="unknown workload"):
        save_experiment(
            "bad-preset",
            """
id: bad-preset
workloads:
  - {id: coding, runner: harbor}
presets:
  smoke:
    workload_overrides:
      typo: {n_tasks: 1}
""",
            tmp_path,
        )


def test_preset_rejects_unknown_workload_override_setting(tmp_path):
    with pytest.raises(ValueError, match="unknown override.*typo"):
        save_experiment(
            "bad-setting",
            """
id: bad-setting
workloads:
  - {id: coding, runner: harbor}
presets:
  smoke:
    workloads: [coding]
    workload_overrides:
      coding: {typo: 1}
""",
            tmp_path,
        )


def test_evaluation_generation_and_scorers_are_strictly_parsed():
    experiment = experiment_from_data(
        {
            "id": "generated",
            "judge_model": "openai/gpt-5-mini",
            "evaluation_generation": {
                "size": 8,
                "sources": [
                    {"kind": "seed", "text": "Test skill behavior."},
                    {"kind": "file", "path": "README.md"},
                    {
                        "kind": "mcp",
                        "server": "github",
                        "tools": ["search_code"],
                        "resources": ["repo://schema"],
                    },
                ],
            },
            "workloads": [
                {
                    "id": "capabilities",
                    "runner": "harbor",
                    "scorers": [
                        "builtin:harbor-outcome",
                        "configs/fugue/evaluations/generated/rubric.yaml",
                    ],
                }
            ],
        }
    )

    assert experiment.evaluation_generation is not None
    assert experiment.evaluation_generation.size == 8
    assert [source.kind for source in experiment.evaluation_generation.sources] == [
        "seed",
        "file",
        "mcp",
    ]
    assert experiment.workloads[0].scorers[-1].endswith("rubric.yaml")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sources", [{"kind": "file", "path": "../secret"}], "repository-relative"),
        ("sources", [{"kind": "seed"}], "seed text is required"),
        ("sources", [{"kind": "mcp"}], "MCP server is required"),
    ],
)
def test_evaluation_generation_rejects_invalid_sources(field, value, message):
    with pytest.raises(ValueError, match=message):
        experiment_from_data(
            {
                "id": "invalid-generated",
                "evaluation_generation": {field: value},
            }
        )


@pytest.mark.parametrize(
    "scorer",
    [
        "../rubric.yaml",
        "/tmp/rubric.yaml",
        "configs/fugue/rubrics/not-evaluation.yaml",
        "configs/fugue/evaluations/suite/rubric.json",
    ],
)
def test_workload_rejects_unsafe_or_unsupported_scorer_refs(scorer):
    with pytest.raises(ValueError, match="scorer"):
        experiment_from_data(
            {
                "id": "invalid-scorer",
                "workloads": [
                    {
                        "id": "capabilities",
                        "runner": "harbor",
                        "scorers": [scorer],
                    }
                ],
            }
        )


def test_workload_rejects_duplicate_scorers():
    with pytest.raises(ValueError, match="duplicate workload capabilities scorer"):
        experiment_from_data(
            {
                "id": "duplicate-scorer",
                "workloads": [
                    {
                        "id": "capabilities",
                        "runner": "harbor",
                        "scorers": [
                            "builtin:harbor-outcome",
                            "builtin:harbor-outcome",
                        ],
                    }
                ],
            }
        )


def test_experiment_rejects_duplicate_ids_and_nonpositive_counts(tmp_path):
    with pytest.raises(ValueError, match="duplicate variant"):
        save_experiment(
            "duplicates",
            """
id: duplicates
variants:
  - {id: same, label: One}
  - {id: same, label: Two}
""",
            tmp_path,
        )
    with pytest.raises(ValueError, match="n_attempts must be positive"):
        save_experiment(
            "bad-count",
            """
id: bad-count
n_attempts: 0
""",
            tmp_path,
        )


def test_preset_rejects_unknown_workload_reference(tmp_path):
    with pytest.raises(ValueError, match="unknown workload"):
        save_experiment(
            "bad-workload",
            """
id: bad-workload
workloads:
  - {id: coding, runner: harbor}
presets:
  smoke:
    workloads: [typo]
""",
            tmp_path,
        )


def test_ids_reject_path_traversal():
    with pytest.raises(ValueError):
        validate_id("../secret", kind="prompt id")


def test_experiment_file_rejects_mismatched_declared_id(tmp_path):
    path = tmp_path / "configs" / "fugue" / "experiments" / "expected.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("id: different\ntitle: Different\n")

    with pytest.raises(ValueError, match="mismatched id"):
        get_experiment("expected", tmp_path)
