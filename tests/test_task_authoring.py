from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from fugue.bench.manifest import load_manifest
from fugue.bench.task_authoring import (
    AuthoredTaskMaterializer,
    analyze_task_evaluation,
    evaluate_task_rows,
    materialize_task_suite_lock,
    preview_task_suite,
    run_inline_scorer,
    scoring_revision_from_dict,
    task_authoring_policy_from_dict,
    task_evaluation_from_dict,
    task_profile_catalog_from_dict,
    task_study_analysis_from_dict,
    task_suite_draft_from_dict,
    task_suite_lock_from_dict,
    task_suite_preview_from_dict,
    verify_task_suite_lock,
)


def _policy():
    policy = task_authoring_policy_from_dict(
        {
            "enabled_stages": ["qualification", "discovery"],
            "allowed_partitions": ["qualification", "discovery"],
            "allowed_environment_profiles": ["artifact-v1"],
            "allowed_resource_profiles": ["reference-v1"],
            "allowed_interactor_profiles": ["scripted-v1"],
            "allowed_judge_profiles": ["judge-v1"],
            "allowed_scorer_runtimes": ["scorer-v1"],
            "allowed_prompt_parts": ["text", "resource"],
            "adaptive_discovery": True,
            "limits": {
                "max_tasks": 4,
                "max_scenarios": 4,
                "max_prompt_bytes": 4096,
                "max_authored_asset_bytes": 4096,
                "max_user_turns": 2,
                "max_agent_turns": 3,
                "max_interactor_calls": 4,
                "max_judge_calls": 8,
                "scorer_timeout_sec": 10,
                "scorer_memory_mb": 128,
                "scorer_cpus": 0.5,
                "scorer_output_bytes": 4096,
            },
        }
    )
    assert policy is not None
    return policy


def _profiles(tmp_path: Path):
    resource = tmp_path / "reference.md"
    resource.write_text("Locked reference.\n")
    import hashlib

    return task_profile_catalog_from_dict(
        {
            "schema_version": 1,
            "environments": [
                {
                    "id": "artifact-v1",
                    "title": "Artifact workspace",
                    "kind": "artifact",
                    "base_image": "python:3.12.10-slim-bookworm",
                    "supported_harnesses": [
                        "hermes",
                        "openclaw",
                        "claude-code",
                        "codex",
                    ],
                    "capabilities": ["text", "resource", "artifact"],
                    "cpus": 1,
                    "memory_mb": 1024,
                    "storage_mb": 2048,
                }
            ],
            "resources": [
                {
                    "id": "reference-v1",
                    "title": "Locked reference",
                    "kind": "markdown",
                    "path": "reference.md",
                    "sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
                    "media_type": "text/markdown",
                    "target": "/workspace/resources/reference.md",
                }
            ],
            "interactors": [
                {
                    "id": "scripted-v1",
                    "title": "Scripted user",
                    "kind": "scripted",
                    "directions": [],
                    "supported_harnesses": [
                        "hermes",
                        "openclaw",
                        "claude-code",
                        "codex",
                    ],
                }
            ],
            "judges": [
                {
                    "id": "judge-v1",
                    "title": "Blind judge",
                    "model": "openai/gpt-5",
                    "prompt": "Judge only the supplied evidence.",
                    "evidence": ["answer"],
                    "blind_fields": [
                        "harness",
                        "model",
                        "variant_id",
                        "context_system_id",
                        "candidate_id",
                        "treatment",
                    ],
                    "input_cost_per_million": 1,
                    "output_cost_per_million": 2,
                }
            ],
            "scorer_runtimes": [
                {
                    "id": "scorer-v1",
                    "title": "Pinned scorer",
                    "image": "example/scorer@sha256:" + "b" * 64,
                    "command": ["python", "/input/scorer.py", "/input/input.json"],
                }
            ],
        },
        source_sha256="a" * 64,
    )


def _draft(*, interaction: str = "single_turn"):
    interaction_value: dict[str, object] = {
        "type": "single_turn",
        "max_user_turns": 1,
        "max_agent_turns": 1,
        "timeout_sec": 300,
    }
    if interaction == "scripted":
        interaction_value = {
            "type": "scripted",
            "profile_id": "scripted-v1",
            "scripted_turns": ["Show the evidence behind that conclusion."],
            "max_user_turns": 1,
            "max_agent_turns": 2,
            "timeout_sec": 300,
        }
    return task_suite_draft_from_dict(
        {
            "schema_version": 1,
            "id": "suite-one",
            "title": "Qualification suite",
            "objective": "Measure whether the Agent produces a grounded answer.",
            "stage_id": "qualification",
            "tasks": [
                {
                    "id": "task-one",
                    "title": "Explain the contract",
                    "prompt": [
                        {"type": "text", "text": "Explain the supplied contract."},
                        {"type": "resource", "resource_profile_id": "reference-v1"},
                    ],
                    "environment": {"profile_id": "artifact-v1"},
                    "interaction": interaction_value,
                    "criteria_set_id": "grounded",
                    "tags": ["explanation"],
                    "partition": "qualification",
                }
            ],
            "scenarios": [
                {
                    "id": "explanation",
                    "title": "Explanation",
                    "tasks": [{"task_id": "task-one", "weight": 1, "must_pass": True}],
                }
            ],
            "criteria_sets": [
                {
                    "id": "grounded",
                    "title": "Grounded answer",
                    "pass_threshold": 1,
                    "criteria": [
                        {
                            "id": "benchmark",
                            "description": "The deterministic verifier passes.",
                            "evaluator": {"type": "benchmark_outcome", "config": {}},
                            "evidence": ["benchmark"],
                            "weight": 1,
                            "threshold": 1,
                            "required": True,
                        }
                    ],
                }
            ],
        }
    )


def _preview(tmp_path: Path, *, interaction: str = "single_turn"):
    profiles = _profiles(tmp_path)
    preview = preview_task_suite(
        campaign_id="campaign-one",
        catalog_digest="c" * 64,
        policy_digest="d" * 64,
        draft=_draft(interaction=interaction),
        policy=_policy(),
        profiles=profiles,
        harnesses=("hermes", "openclaw", "claude-code", "codex"),
        repo_root=tmp_path,
    )
    return profiles, preview


def test_task_artifacts_are_strict_canonical_and_preview_is_pure(
    tmp_path: Path,
) -> None:
    profiles, preview = _preview(tmp_path)

    assert preview.eligible
    assert preview.task_count == 1
    assert preview.scenario_count == 1
    assert len(preview.capability_matrix) == 4
    assert all(item["applicable"] for item in preview.capability_matrix)
    assert task_suite_preview_from_dict(preview.to_dict()) == preview
    assert not (tmp_path / ".fugue").exists()

    destination = tmp_path / ".fugue/runtime/campaigns/campaign-one/assets"
    lock = materialize_task_suite_lock(
        preview,
        profiles=profiles,
        repo_root=tmp_path,
        destination=destination,
        harnesses=("hermes", "openclaw", "claude-code", "codex"),
    )
    assert task_suite_lock_from_dict(lock.to_dict()) == lock
    verify_task_suite_lock(tmp_path, lock)
    assert json.loads((destination / "private-evaluation.json").read_text())[
        "criteria_sets"
    ]
    public = (destination / "public-cases.jsonl").read_text()
    assert "criteria" not in public
    assert "expected" not in public


def test_task_draft_rejects_commands_paths_environment_and_dependencies() -> None:
    raw = _draft().to_dict()
    for field in ("command", "path", "environment_variables", "dependencies"):
        changed = json.loads(json.dumps(raw))
        changed["tasks"][0][field] = "unsafe"
        changed.pop("draft_digest")
        with pytest.raises(ValueError, match="unknown authored task field"):
            task_suite_draft_from_dict(changed)

    changed = json.loads(json.dumps(raw))
    changed["tasks"][0]["prompt"][0] = {
        "type": "text",
        "text": "safe",
        "path": "/etc/passwd",
    }
    changed.pop("draft_digest")
    with pytest.raises(ValueError, match="unknown prompt part field"):
        task_suite_draft_from_dict(changed)


def test_multi_turn_capability_is_explicit_per_harness(tmp_path: Path) -> None:
    _, preview = _preview(tmp_path, interaction="scripted")

    assert preview.eligible
    assert preview.estimated_calls["interactor"] == 0
    assert {item["harness"] for item in preview.capability_matrix} == {
        "hermes",
        "openclaw",
        "claude-code",
        "codex",
    }


def test_rescoring_is_immutable_and_keeps_benchmark_outcome_separate(
    tmp_path: Path,
) -> None:
    profiles, preview = _preview(tmp_path)
    destination = tmp_path / ".fugue/runtime/campaigns/campaign-one/assets"
    lock = materialize_task_suite_lock(
        preview,
        profiles=profiles,
        repo_root=tmp_path,
        destination=destination,
        harnesses=("hermes", "codex"),
    )
    revision = scoring_revision_from_dict(
        {
            "schema_version": 1,
            "id": "answer-only-v1",
            "evidence_view": "answer",
            "reason": "Qualification scoring view.",
        }
    )
    rows = [
        {
            "prediction_id": "prediction-hermes",
            "task_name": "task-one",
            "harness": "hermes",
            "trial_index": 1,
            "status": "passed",
            "pass": True,
            "agent_response": "Grounded answer.",
        },
        {
            "prediction_id": "prediction-codex",
            "task_name": "task-one",
            "harness": "codex",
            "trial_index": 1,
            "status": "failed",
            "pass": False,
            "agent_response": "Incomplete answer.",
        },
    ]
    evaluation = evaluate_task_rows(
        campaign_id="campaign-one",
        run_id="run-one",
        lock=lock,
        revision=revision,
        rows=rows,
        profiles=profiles,
        repo_root=tmp_path,
        env={},
    )

    assert task_evaluation_from_dict(evaluation.to_dict()) == evaluation
    assert evaluation.evaluated_predictions == 2
    assert evaluation.passed == 1
    assert evaluation.failed == 1
    assert [row["benchmark_pass"] for row in evaluation.prediction_results] == [
        True,
        False,
    ]
    richer_revision = scoring_revision_from_dict(
        {
            "schema_version": 1,
            "id": "answer-artifacts-v1",
            "evidence_view": "answer_artifacts_tools",
        }
    )
    rescored = evaluate_task_rows(
        campaign_id="campaign-one",
        run_id="run-one",
        lock=lock,
        revision=richer_revision,
        rows=rows,
        profiles=profiles,
        repo_root=tmp_path,
        env={},
    )
    assert rescored.task_suite_digest == evaluation.task_suite_digest
    assert rescored.evaluation_digest != evaluation.evaluation_digest
    analysis = analyze_task_evaluation(
        analysis_id="task-shape-v1",
        lock=lock,
        evaluation=evaluation,
        repo_root=tmp_path,
    )
    assert task_study_analysis_from_dict(analysis.to_dict()) == analysis
    assert len(analysis.contrasts) == 1
    assert "universal ranking" in " ".join(analysis.limitations)


def test_required_broken_evaluator_is_unavailable_not_agent_failure(
    tmp_path: Path,
) -> None:
    profiles, preview = _preview(tmp_path)
    destination = tmp_path / ".fugue/runtime/campaigns/campaign-one/assets"
    lock = materialize_task_suite_lock(
        preview,
        profiles=profiles,
        repo_root=tmp_path,
        destination=destination,
        harnesses=("codex",),
    )
    revision = scoring_revision_from_dict(
        {
            "schema_version": 1,
            "id": "answer-only-v1",
            "evidence_view": "answer",
        }
    )
    evaluation = evaluate_task_rows(
        campaign_id="campaign-one",
        run_id="run-one",
        lock=lock,
        revision=revision,
        rows=[
            {
                "prediction_id": "prediction-one",
                "task_name": "task-one",
                "harness": "codex",
                "status": "failed",
                "pass": None,
            }
        ],
        profiles=profiles,
        repo_root=tmp_path,
        env={},
    )

    assert evaluation.unavailable == 1
    assert evaluation.failed == 0
    assert evaluation.prediction_results[0]["criteria_status"] == "unavailable"


def test_future_scripted_turns_are_not_mounted_in_the_agent_environment(
    tmp_path: Path,
) -> None:
    profiles, preview = _preview(tmp_path, interaction="scripted")
    destination = tmp_path / ".fugue/runtime/campaigns/campaign-one/assets"
    lock = materialize_task_suite_lock(
        preview,
        profiles=profiles,
        repo_root=tmp_path,
        destination=destination,
        harnesses=("codex",),
    )
    task_root = tmp_path / "materialized"
    AuthoredTaskMaterializer().materialize(
        load_manifest(tmp_path / lock.manifest_path),
        task_root,
        tmp_path / lock.public_cases_path,
        repo_root=tmp_path,
    )

    environment = task_root / "task-one/environment"
    assert not (environment / "fugue-task-interaction.json").exists()
    rendered = "\n".join(
        path.read_text(errors="ignore")
        for path in (task_root / "task-one").rglob("*")
        if path.is_file()
    )
    assert "Show the evidence behind that conclusion." not in rendered


def test_every_task_artifact_rejects_unknown_fields_and_versions(
    tmp_path: Path,
) -> None:
    profiles, preview = _preview(tmp_path)
    lock = materialize_task_suite_lock(
        preview,
        profiles=profiles,
        repo_root=tmp_path,
        destination=tmp_path / ".fugue/runtime/campaigns/campaign-one/assets",
        harnesses=("codex",),
    )
    revision = scoring_revision_from_dict(
        {"schema_version": 1, "id": "answer-v1", "evidence_view": "answer"}
    )
    evaluation = evaluate_task_rows(
        campaign_id="campaign-one",
        run_id="run-one",
        lock=lock,
        revision=revision,
        rows=[
            {
                "prediction_id": "prediction-one",
                "task_name": "task-one",
                "harness": "codex",
                "status": "passed",
                "pass": True,
            }
        ],
        profiles=profiles,
        repo_root=tmp_path,
        env={},
    )
    analysis = analyze_task_evaluation(
        analysis_id="analysis-v1",
        lock=lock,
        evaluation=evaluation,
        repo_root=tmp_path,
    )
    artifacts = (
        (task_suite_preview_from_dict, preview),
        (task_suite_lock_from_dict, lock),
        (task_evaluation_from_dict, evaluation),
        (task_study_analysis_from_dict, analysis),
    )
    for parser, artifact in artifacts:
        unknown = artifact.to_dict()
        unknown["unexpected"] = True
        with pytest.raises(ValueError, match="unknown"):
            parser(unknown)

        unsupported = artifact.to_dict()
        unsupported["schema_version"] = 2
        with pytest.raises(ValueError, match="schema_version 1"):
            parser(unsupported)


def test_inline_scorer_runs_with_a_locked_isolated_docker_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = _profiles(tmp_path)
    profile = profiles.scorer_runtime("scorer-v1")
    limits = _policy().limits
    observed: dict[str, object] = {}

    monkeypatch.setattr("fugue.bench.task_authoring.shutil.which", lambda _: "/docker")

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"score": 1, "reason": "passed"}',
            stderr="",
        )

    monkeypatch.setattr("fugue.bench.task_authoring.subprocess.run", fake_run)
    payload = run_inline_scorer(
        source="print('{}')",
        evidence={"answer": "safe"},
        reference={"expected": "private"},
        profile=profile,
        limits=limits,
    )

    assert payload["score"] == 1
    command = observed["command"]
    assert isinstance(command, list)
    for expected in (
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "no-new-privileges",
        "--pids-limit",
    ):
        assert expected in command
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert set(kwargs["env"]) == {"PATH"}
    assert kwargs["timeout"] == limits.scorer_timeout_sec


def test_adaptive_task_preview_rejects_non_discovery_partitions(
    tmp_path: Path,
) -> None:
    profiles = _profiles(tmp_path)
    raw = _draft().to_dict()
    raw.pop("draft_digest")
    raw["parent_outcome_id"] = "earlier-outcome"
    raw["decision_rationale"] = "Target a failure observed in discovery."
    preview = preview_task_suite(
        campaign_id="campaign-one",
        catalog_digest="c" * 64,
        policy_digest="d" * 64,
        draft=task_suite_draft_from_dict(raw),
        policy=_policy(),
        profiles=profiles,
        harnesses=("codex",),
        repo_root=tmp_path,
    )

    assert not preview.eligible
    assert any("discovery tasks only" in failure for failure in preview.failures)


def test_blind_judge_has_a_separate_receipt_and_accounted_cost(
    tmp_path: Path,
) -> None:
    profiles = _profiles(tmp_path)
    raw = _draft().to_dict()
    raw.pop("draft_digest")
    raw["criteria_sets"][0] = {
        "id": "grounded",
        "title": "Grounded answer",
        "pass_threshold": 0.5,
        "criteria": [
            {
                "id": "blind-judge",
                "description": "The answer is grounded.",
                "evaluator": {
                    "type": "judge",
                    "profile_id": "judge-v1",
                    "config": {},
                },
                "evidence": ["answer"],
                "weight": 1,
                "threshold": 0.5,
                "required": True,
            }
        ],
    }
    preview = preview_task_suite(
        campaign_id="campaign-one",
        catalog_digest="c" * 64,
        policy_digest="d" * 64,
        draft=task_suite_draft_from_dict(raw),
        policy=_policy(),
        profiles=profiles,
        harnesses=("codex",),
        repo_root=tmp_path,
        paid_call_reserve_usd=2,
    )
    assert preview.estimated_calls["judge"] == 1
    assert preview.estimated_cost_usd == 2
    lock = materialize_task_suite_lock(
        preview,
        profiles=profiles,
        repo_root=tmp_path,
        destination=tmp_path / ".fugue/runtime/campaigns/campaign-one/judge-assets",
        harnesses=("codex",),
    )
    observed: dict[str, object] = {}

    def judge_request(*, profile, evidence, env):
        del profile, env
        observed["evidence"] = evidence
        return (
            {"score": 0.75, "reason": "Grounded."},
            {"input_tokens": 10, "output_tokens": 5},
            {"role": "judge", "trace_scope": "separate_from_agent"},
        )

    evaluation = evaluate_task_rows(
        campaign_id="campaign-one",
        run_id="run-one",
        lock=lock,
        revision=scoring_revision_from_dict(
            {"schema_version": 1, "id": "judge-v1", "evidence_view": "answer"}
        ),
        rows=[
            {
                "prediction_id": "prediction-one",
                "task_name": "task-one",
                "harness": "codex",
                "model": "secret-model-label",
                "variant_id": "secret-treatment",
                "agent_response": "Evidence-backed answer.",
            }
        ],
        profiles=profiles,
        repo_root=tmp_path,
        env={},
        judge_request=judge_request,
    )

    assert observed["evidence"] == {"answer": "Evidence-backed answer."}
    assert evaluation.judge_calls == 1
    assert evaluation.unmeasured_paid_calls == 0
    assert evaluation.observed_cost_usd == pytest.approx(0.00002)
    criterion = evaluation.prediction_results[0]["criteria"][0]
    assert criterion["route_receipt"]["trace_scope"] == "separate_from_agent"


@pytest.mark.parametrize(
    ("completed", "error"),
    [
        (
            subprocess.CompletedProcess(["docker"], 0, stdout="not-json", stderr=""),
            json.JSONDecodeError,
        ),
        (
            subprocess.CompletedProcess(["docker"], 0, stdout="[]", stderr=""),
            ValueError,
        ),
    ],
)
def test_inline_scorer_rejects_malformed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed: subprocess.CompletedProcess[str],
    error: type[Exception],
) -> None:
    profile = _profiles(tmp_path).scorer_runtime("scorer-v1")
    monkeypatch.setattr("fugue.bench.task_authoring.shutil.which", lambda _: "/docker")
    monkeypatch.setattr(
        "fugue.bench.task_authoring.subprocess.run", lambda *args, **kwargs: completed
    )
    with pytest.raises(error):
        run_inline_scorer(
            source="print('{}')",
            evidence={},
            reference={},
            profile=profile,
            limits=_policy().limits,
        )
