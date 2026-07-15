from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from fugue.bench.candidates import CANDIDATE_IDENTITY_SCHEMA_VERSION, resolve_candidate
from fugue.bench.job_config import _comparison_example_id
from fugue.bench.runtime_provenance import resolve_fugue_source_provenance


def _candidate_inputs() -> dict:
    return {
        "harness": "codex",
        "harness_version": "codex@0.143.0+fugue-flat-mcp.1",
        "model_route": {"provider": "openai", "model_id": "gpt-5"},
        "prompt_digest": "prompt-a",
        "skills": [{"id": "reviewed", "sha256": "skill-a"}],
        "context": {
            "id": "none",
            "version": "1",
            "config_hash": "context-a",
            "delivery": "portable",
        },
        "integrations": [{"id": "search", "version": "1"}],
        "agent": {"agent_kwargs": {"reasoning": "high"}},
        "execution": {
            "harbor_version": "0.18.0",
            "trace_content": "full",
        },
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("harness", "claude-code"),
        ("model_route", {"provider": "anthropic", "model_id": "claude"}),
        ("prompt_digest", "prompt-b"),
        ("skills", [{"id": "reviewed", "sha256": "skill-b"}]),
        (
            "context",
            {
                "id": "none",
                "version": "1",
                "config_hash": "context-a",
                "delivery": "native_mcp",
            },
        ),
        ("integrations", [{"id": "browser", "version": "1"}]),
        ("agent", {"agent_kwargs": {"reasoning": "low"}}),
    ),
)
def test_every_behavioral_input_changes_candidate_identity(
    field: str, replacement: object
) -> None:
    original = _candidate_inputs()
    changed = deepcopy(original)
    changed[field] = replacement

    assert resolve_candidate(**original).candidate_id != resolve_candidate(
        **changed
    ).candidate_id


def test_tool_result_modalities_are_candidate_behavior() -> None:
    original = _candidate_inputs()
    original["model_route"]["tool_result_modalities"] = ["text", "image"]
    changed = deepcopy(original)
    changed["model_route"]["tool_result_modalities"] = ["text"]

    first = resolve_candidate(**original)
    second = resolve_candidate(**changed)

    assert CANDIDATE_IDENTITY_SCHEMA_VERSION == 3
    assert first.candidate_id != second.candidate_id
    assert first.definition["identity_schema_version"] == 3


def test_harness_behavior_version_changes_candidate_identity() -> None:
    original = _candidate_inputs()
    changed = deepcopy(original)
    changed["harness_version"] = "codex@0.143.0+fugue-flat-mcp.2"

    assert resolve_candidate(**original).candidate_id != resolve_candidate(
        **changed
    ).candidate_id


@pytest.mark.parametrize("harness", [None, "", object()])
def test_candidate_harness_must_be_a_non_empty_string(harness: object) -> None:
    inputs = _candidate_inputs()
    inputs["harness"] = harness

    with pytest.raises(ValueError, match="harness must be a non-empty string"):
        resolve_candidate(**inputs)


@pytest.mark.parametrize("version", [None, "", object()])
def test_candidate_harness_version_must_be_non_empty(version: object) -> None:
    inputs = _candidate_inputs()
    inputs["harness_version"] = version

    with pytest.raises(ValueError, match="harness_version must be a non-empty string"):
        resolve_candidate(**inputs)


def test_execution_policy_changes_fingerprint_not_candidate_identity() -> None:
    original = _candidate_inputs()
    changed = deepcopy(original)
    changed["execution"]["trace_content"] = "metadata"

    first = resolve_candidate(**original)
    second = resolve_candidate(**changed)

    assert first.candidate_id == second.candidate_id
    assert first.execution_fingerprint != second.execution_fingerprint


def test_context_runtime_topology_changes_only_execution_identity() -> None:
    original = _candidate_inputs()
    original["execution"]["context_runtime"] = {
        "schema_version": 1,
        "network": "compose_project",
    }
    changed = deepcopy(original)
    changed["execution"]["context_runtime"]["network"] = "shared_namespace"

    first = resolve_candidate(**original)
    second = resolve_candidate(**changed)

    assert first.candidate_id == second.candidate_id
    assert first.execution_fingerprint != second.execution_fingerprint


def test_fugue_source_commit_changes_only_execution_identity() -> None:
    original = _candidate_inputs()
    original["execution"]["fugue_source"] = {
        "schema_version": 1,
        "kind": "git",
        "commit": "a" * 40,
        "dirty": False,
    }
    changed = deepcopy(original)
    changed["execution"]["fugue_source"]["commit"] = "b" * 40

    first = resolve_candidate(**original)
    second = resolve_candidate(**changed)

    assert first.candidate_id == second.candidate_id
    assert first.execution_fingerprint != second.execution_fingerprint


def test_source_provenance_distinguishes_clean_and_dirty_trees(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "fugue.py"
    source.write_text("VERSION = 1\n")
    subprocess.run(["git", "add", "fugue.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fugue Tests",
            "-c",
            "user.email=fugue@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )

    clean = resolve_fugue_source_provenance(tmp_path)
    source.write_text("VERSION = 2\n")
    first_dirty = resolve_fugue_source_provenance(tmp_path)
    source.write_text("VERSION = 3\n")
    second_dirty = resolve_fugue_source_provenance(tmp_path)

    assert clean["kind"] == "git"
    assert clean["dirty"] is False
    assert len(clean["commit"]) == 40
    assert "dirty_digest" not in clean
    assert first_dirty["commit"] == clean["commit"]
    assert first_dirty["dirty"] is True
    assert first_dirty["dirty_digest"] != second_dirty["dirty_digest"]


def test_unversioned_source_provenance_ignores_secrets_and_runtime_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fugue.py"
    source.write_text("VERSION = 1\n")
    first = resolve_fugue_source_provenance(tmp_path)
    (tmp_path / ".env").write_text("API_KEY=secret-value\n")
    runtime = tmp_path / ".fugue" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "run.json").write_text("{}\n")
    second = resolve_fugue_source_provenance(tmp_path)
    source.write_text("VERSION = 2\n")
    changed = resolve_fugue_source_provenance(tmp_path)

    assert first == second
    assert first["kind"] == "unversioned"
    assert first["dirty"] is True
    assert first["digest"] != changed["digest"]


def test_resolved_candidate_does_not_expose_mutable_internal_state() -> None:
    resolved = resolve_candidate(**_candidate_inputs())
    definition = resolved.definition
    definition["harness"] = "mutated"

    assert resolved.definition["harness"] == "codex"


def test_comparison_example_identity_excludes_trial_index() -> None:
    identity = _comparison_example_id(
        dataset_id="dataset", workload_id="workload", task_id="task"
    )

    assert identity == _comparison_example_id(
        dataset_id="dataset", workload_id="workload", task_id="task"
    )
    assert len(identity) == 64
