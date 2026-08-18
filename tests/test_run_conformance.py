from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import fugue.bench.run_conformance as conformance_module
from fugue.bench.candidates import stable_digest
from fugue.bench.run_conformance import (
    _HostedEvidenceSnapshot,
    _required_conversation_ids,
    capture_local_cell_conformance,
    capture_local_docker_inventory,
    read_harbor_run_conformance_receipt,
    read_hosted_evidence_privacy_receipt,
    write_harbor_run_conformance_receipt,
    write_hosted_evidence_privacy_receipt,
)
from fugue.bench.sandbox_policy import attest_harbor_job


def _local_job(repo_root: Path, run_id: str) -> SimpleNamespace:
    run_dir = repo_root / ".fugue/runtime" / run_id
    jobs_dir = repo_root / ".fugue/runtime/jobs/demo" / run_id
    config_path = run_dir / "job-configs/job.json"
    result_path = jobs_dir / "job/result.json"
    config_path.parent.mkdir(parents=True)
    result_path.parent.mkdir(parents=True)
    (result_path.parent / "task__AbC123").mkdir()
    candidate_id = "c" * 64
    execution_definition = {
        "identity_schema_version": 1,
        "candidate_id": candidate_id,
        "environment": "local-harbor-test",
    }
    execution_fingerprint = stable_digest(execution_definition)
    config = {
        "job_name": "job",
        "jobs_dir": jobs_dir.relative_to(repo_root).as_posix(),
        "fugue": {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "execution_fingerprint": execution_fingerprint,
            "agent_runtime": {
                "image": "fugue-agent:test",
                "image_id": "sha256:" + "1" * 64,
                "recipe_sha256": "2" * 64,
                "architecture": "amd64",
                "os": "linux",
                "version": "test",
            },
            "task_runtime": {
                "image": "fugue-task:test",
                "image_id": "sha256:" + "3" * 64,
                "recipe_sha256": "4" * 64,
                "architecture": "amd64",
                "os": "linux",
            },
        },
    }
    config["fugue"]["sandbox_attestation"] = attest_harbor_job(
        config,
        repo_root=repo_root,
        bridge_required=False,
        require_files=True,
    ).to_dict()
    config_path.write_text(json.dumps(config))
    result_path.write_text('{"status": "passed"}\n')
    evaluation_assets = run_dir / "evaluation-assets.json"
    evaluation_assets.write_text('{"schema_version": 1}\n')
    evaluation_assets.chmod(0o600)
    return SimpleNamespace(
        applicable=True,
        execution_kind="agent",
        command=["harbor", "run", "--config", config_path.as_posix()],
        config=config,
        config_path=config_path,
        result_path=result_path,
        job_name="job",
        run_id=run_id,
        resolved_candidate=SimpleNamespace(
            candidate_id=candidate_id,
            execution_fingerprint=execution_fingerprint,
            execution_definition=execution_definition,
        ),
    )


def _inventory(*container_ids: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "passed",
        "backend": "local_harbor_docker",
        "container_ids": list(container_ids),
    }


def _hosted_row() -> dict[str, object]:
    return {
        "attempt_id": "attempt-1",
        "weave_evaluation_root_call_id": "evaluation-root",
        "eval_predict_and_score_call_id": "predict-and-score",
        "weave_prediction_call_id": "prediction",
        "weave_agent_root_call_id": "agent-root",
        "weave_evaluation_root_trace_id": "trace-1",
        "planned_conversation_id": "planned-conversation-1",
        "observed_conversation_id": "conversation-1",
        "native_session_ids": ["conversation-1"],
        "weave_conversation_id": "planned-conversation-1",
        "weave_conversation_ids": ["conversation-1"],
        "weave_dataset_ref": (
            "weave:///wandb/fugue-demo/object/fugue-tasks:dataset-digest"
        ),
    }


def _hosted_snapshot(*payloads: tuple[str, object]) -> _HostedEvidenceSnapshot:
    return _HostedEvidenceSnapshot(
        status="passed",
        payloads=payloads,
        required_call_count=4,
        observed_required_call_count=4,
        descendant_call_count=2,
        agent_span_count=3,
        required_agent_conversation_count=1,
        observed_agent_conversation_count=1,
        required_dataset_count=1,
        observed_dataset_count=1,
        query_error_count=0,
    )


def test_hosted_privacy_requires_observed_not_planned_conversation() -> None:
    conversation_ids = _required_conversation_ids(
        [
            {
                "planned_conversation_id": "planned-conversation",
                "weave_conversation_id": "planned-conversation",
                "observed_conversation_id": "native-conversation",
                "native_session_ids": ["native-conversation"],
                "weave_conversation_ids": ["native-conversation"],
            }
        ]
    )

    assert conversation_ids == {"native-conversation"}


def test_hosted_privacy_receipt_scans_exact_evidence_without_raw_values(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "private-labels.jsonl"
    private_fact = "maintainer-private-fact-123"
    labels.write_text(
        json.dumps({"id": "task", "expected": {"fact": private_fact}}) + "\n"
    )
    secret = "wandb-secret-value"
    snapshot = _hosted_snapshot(
        ("call", {"id": "evaluation-root", "inputs": {"task": "public"}}),
        ("call", {"id": "predict-and-score", "output": {"passed": True}}),
        ("call", {"id": "prediction", "output": {"answer": "public"}}),
        ("call", {"id": "agent-root", "output": {"answer": "public"}}),
        ("call", {"id": "agent-child", "output": {"tool": "read"}}),
        ("agent_span", {"conversation_id": "conversation-1"}),
        ("dataset", {"rows": [{"input": "public"}]}),
    )

    receipt = write_hosted_evidence_privacy_receipt(
        repo_root=tmp_path,
        run_id="run-hosted",
        rows=[_hosted_row()],
        env={"WANDB_API_KEY": secret},
        evidence_project="wandb/fugue-demo",
        private_labels_path=labels,
        publication_payloads={
            "result": {"decision": "incomplete"},
            "study": {"state": "completed"},
        },
        fetcher=lambda **_kwargs: snapshot,
    )

    value = read_hosted_evidence_privacy_receipt(
        repo_root=tmp_path,
        run_id="run-hosted",
    )
    serialized = receipt.path.read_text()
    assert receipt.status == "passed"
    assert value["status"] == "passed"
    assert value["required_call_count"] == 4
    assert value["observed_required_call_count"] == 4
    assert value["descendant_call_count"] == 2
    assert value["result_payload_count"] == 1
    assert value["study_payload_count"] == 1
    assert value["secret_match_count"] == 0
    assert value["private_corpus_match_count"] == 0
    assert value["private_structure_match_count"] == 0
    assert secret not in serialized
    assert private_fact not in serialized
    assert "evaluation-root" not in serialized
    assert "conversation-1" not in serialized


def test_hosted_privacy_accepts_one_pretty_json_private_bundle(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "private-evaluation.json"
    labels.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "criteria_sets": [
                    {
                        "id": "deterministic",
                        "criteria": [{"id": "fact", "expected": "private-fact"}],
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )
    receipt = write_hosted_evidence_privacy_receipt(
        repo_root=tmp_path,
        run_id="run-pretty-private-bundle",
        rows=[_hosted_row()],
        env={"WANDB_API_KEY": "wandb-secret-value"},
        evidence_project="wandb/fugue-demo",
        private_labels_path=labels,
        publication_payloads={"result": {"decision": "incomplete"}},
        fetcher=lambda **_kwargs: _hosted_snapshot(
            ("call", {"id": "evaluation-root"}),
            ("call", {"id": "predict-and-score"}),
            ("call", {"id": "prediction"}),
            ("call", {"id": "agent-root"}),
            ("agent_span", {"conversation_id": "conversation-1"}),
            ("dataset", {"rows": [{"input": "public"}]}),
        ),
    )

    value = read_hosted_evidence_privacy_receipt(
        repo_root=tmp_path,
        run_id="run-pretty-private-bundle",
    )
    assert receipt.status == "passed"
    assert value["private_label_record_count"] == 1
    assert "private-fact" not in receipt.path.read_text()


@pytest.mark.parametrize("private_input", ["missing", "malformed"])
def test_hosted_privacy_fails_closed_without_parseable_private_truth(
    tmp_path: Path,
    private_input: str,
) -> None:
    labels = tmp_path / f"{private_input}.json"
    if private_input == "malformed":
        labels.write_text('{"expected":\n')
    receipt = write_hosted_evidence_privacy_receipt(
        repo_root=tmp_path,
        run_id=f"run-{private_input}-private-truth",
        rows=[_hosted_row()],
        env={"WANDB_API_KEY": "wandb-secret-value"},
        evidence_project="wandb/fugue-demo",
        private_labels_path=labels,
        publication_payloads={"result": {"decision": "blocked"}},
        fetcher=lambda **_kwargs: _hosted_snapshot(
            ("call", {"id": "evaluation-root"}),
            ("call", {"id": "predict-and-score"}),
            ("call", {"id": "prediction"}),
            ("call", {"id": "agent-root"}),
            ("agent_span", {"conversation_id": "conversation-1"}),
            ("dataset", {"rows": [{"input": "public"}]}),
        ),
    )

    value = read_hosted_evidence_privacy_receipt(
        repo_root=tmp_path,
        run_id=f"run-{private_input}-private-truth",
    )
    assert receipt.status == "unavailable"
    assert value["status"] == "unavailable"
    assert value["private_label_record_count"] == 0


def test_hosted_privacy_allows_explicit_no_private_corpus_but_requires_fetch(
    tmp_path: Path,
) -> None:
    common = {
        "repo_root": tmp_path,
        "rows": [_hosted_row()],
        "env": {"WANDB_API_KEY": "wandb-secret-value"},
        "evidence_project": "wandb/fugue-demo",
        "private_labels_path": None,
        "publication_payloads": {"result": {"decision": "incomplete"}},
        "private_corpus_applicable": False,
        "private_corpus_source_kind": "run_evaluation_asset_lock",
        "private_corpus_source_lock_sha256": "a" * 64,
        "private_corpus_reason": "verified lock contains no private corpus",
    }
    receipt = write_hosted_evidence_privacy_receipt(
        **common,
        run_id="run-private-na",
        fetcher=lambda **_kwargs: _hosted_snapshot(
            ("call", {"id": "evaluation-root"}),
            ("call", {"id": "predict-and-score"}),
            ("call", {"id": "prediction"}),
            ("call", {"id": "agent-root"}),
            ("agent_span", {"conversation_id": "conversation-1"}),
            ("dataset", {"rows": [{"input": "public"}]}),
        ),
    )
    value = read_hosted_evidence_privacy_receipt(
        repo_root=tmp_path,
        run_id="run-private-na",
    )
    assert receipt.status == "passed"
    assert value["private_corpus_applicable"] is False
    assert value["private_corpus_comparison_status"] == "not_applicable"
    assert value["private_corpus_match_count"] == 0

    unavailable = write_hosted_evidence_privacy_receipt(
        **common,
        run_id="run-private-na-no-hosted-scan",
        fetch_hosted=False,
    )
    assert unavailable.status == "unavailable"


def test_hosted_privacy_receipt_fails_on_secret_or_private_structure(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "private-labels.jsonl"
    private_fact = "private-secret-fact"
    labels.write_text(
        json.dumps({"id": "task", "expected": {"fact": private_fact}}) + "\n"
    )
    secret = "anthropic-secret-value"
    snapshot = _hosted_snapshot(
        ("call", {"id": "evaluation-root", "authorization": secret}),
        ("call", {"id": "predict-and-score"}),
        ("call", {"id": "prediction"}),
        ("call", {"id": "agent-root", "expected": {"fact": private_fact}}),
        ("agent_span", {"conversation_id": "conversation-1"}),
        ("dataset", {"rows": [{"input": "public"}]}),
    )

    receipt = write_hosted_evidence_privacy_receipt(
        repo_root=tmp_path,
        run_id="run-hosted-leak",
        rows=[_hosted_row()],
        env={"ANTHROPIC_API_KEY": secret},
        evidence_project="wandb/fugue-demo",
        private_labels_path=labels,
        publication_payloads={"result": {"decision": "blocked"}},
        fetcher=lambda **_kwargs: snapshot,
    )

    value = json.loads(receipt.path.read_text())
    assert receipt.status == "failed"
    assert value["secret_match_count"] == 1
    assert value["private_structure_match_count"] == 1
    assert value["affected_payload_count"] == 2
    assert secret not in receipt.path.read_text()
    assert private_fact not in receipt.path.read_text()


def test_harbor_receipt_records_exact_identity_and_zero_orphans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-local"
    job = _local_job(tmp_path, run_id)
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(conformance_module.subprocess, "run", run)

    result = write_harbor_run_conformance_receipt(
        repo_root=tmp_path,
        run_id=run_id,
        jobs=[job],
        env={"WANDB_API_KEY": "wandb-secret-value"},
        host_scorer_names=("comparison.deterministic.answer",),
        pre_execution_inventory=_inventory(),
    )

    value = json.loads(result.path.read_text())
    assert result.status == "passed"
    identity = value["execution_identity"]["jobs"][0]
    assert identity["execution_fingerprint_verified"] is True
    assert identity["execution_fingerprint"] == identity[
        "recomputed_execution_fingerprint"
    ]
    assert identity["policy"]["verification"] == "passed"
    assert value["execution_identity"]["jobs"][0]["agent_runtime"]["image_id"] == (
        "sha256:" + "1" * 64
    )
    local_scan = value["local_artifact_privacy_scan"]
    assert local_scan["files_with_matches"] == []
    assert local_scan["scope"]["kind"] == "exact_local_run_artifacts"
    assert "hosted Weave objects" in local_scan["scope"]["excluded"]
    assert value["docker_cleanup"]["matched_containers"] == []
    assert value["docker_cleanup"]["matched_networks"] == []
    assert value["docker_cleanup"]["scope"]["compose_projects"] == [
        "task__abc123__env"
    ]
    assert "wandb-secret-value" not in result.path.read_text()
    assert calls == [
        ["docker", "container", "ls", "--all", "--quiet"],
        [
            "docker",
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            "label=com.docker.compose.project=task__abc123__env",
        ],
        [
            "docker",
            "network",
            "ls",
            "--quiet",
            "--filter",
            "label=com.docker.compose.project=task__abc123__env",
        ],
    ]
    assert read_harbor_run_conformance_receipt(
        repo_root=tmp_path,
        run_id=run_id,
    ) == value


def test_first_cell_conformance_proves_cleanup_and_private_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-first-cell"
    job = _local_job(tmp_path, run_id)
    monkeypatch.setattr(
        conformance_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="",
            stderr="",
        ),
    )
    cell = SimpleNamespace(
        id="cell-1",
        run_id=run_id,
        attempt_id="a" * 64,
        config_path=job.config_path,
    )

    receipt = capture_local_cell_conformance(
        repo_root=tmp_path,
        cell=cell,
        job=job,
        env={"WANDB_API_KEY": "wandb-secret-value"},
        host_scorer_names=("comparison.deterministic.answer",),
        pre_execution_inventory=_inventory(),
    )

    assert receipt["status"] == "passed"
    assert receipt["execution_identity"]["status"] == "passed"
    assert receipt["local_artifact_privacy_scan"]["status"] == "passed"
    assert (
        receipt["local_artifact_privacy_scan"][
            "configured_sensitive_value_count"
        ]
        == 1
    )
    assert (
        "configured_secret_count"
        not in receipt["local_artifact_privacy_scan"]
    )
    assert receipt["private_label_boundary"]["status"] == "passed"
    assert receipt["docker_cleanup"]["status"] == "passed"
    assert receipt["docker_cleanup"]["matched_containers"] == []
    assert receipt["docker_cleanup"]["matched_networks"] == []
    assert len(receipt["receipt_sha256"]) == 64
    assert "wandb-secret-value" not in json.dumps(receipt)


def test_private_boundary_rejects_neutral_key_private_bundle_copy(
    tmp_path: Path,
) -> None:
    run_id = "run-neutral-private-copy"
    job = _local_job(tmp_path, run_id)
    private_bundle = {
        "prediction-a": {
            "task_id": "task-a",
            "expected_evidence_paths": ["host-only/answer.json"],
        }
    }
    asset_lock = tmp_path / ".fugue/runtime" / run_id / "evaluation-assets.json"
    asset_lock.write_text(
        json.dumps({"schema_version": 1, "predictions": private_bundle}) + "\n"
    )
    asset_lock.chmod(0o600)
    job.config["context"] = {"innocent_name": private_bundle}

    receipt = conformance_module._private_label_boundary(
        run_dir=asset_lock.parent,
        jobs=[job],
        host_scorer_names=("comparison.deterministic.answer",),
    )

    assert receipt["status"] == "failed"
    assert receipt["rendered_private_fields"] == []
    assert receipt["tainted_agent_inputs"]


@pytest.mark.parametrize(
    ("private_value", "copied_value"),
    [
        ("host-only-scalar-string", "host-only-scalar-string"),
        (314_159, 314_159),
        (False, False),
    ],
    ids=("string", "number", "boolean"),
)
def test_private_boundary_rejects_neutral_key_private_scalar_copy(
    tmp_path: Path,
    private_value: object,
    copied_value: object,
) -> None:
    run_id = "run-neutral-private-scalar-copy"
    job = _local_job(tmp_path, run_id)
    asset_lock = tmp_path / ".fugue/runtime" / run_id / "evaluation-assets.json"
    asset_lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictions": {
                    "prediction-a": {
                        "task_id": "public-task-a",
                        "expected": {"host_value": private_value},
                    }
                },
            }
        )
        + "\n"
    )
    asset_lock.chmod(0o600)
    job.config["agents"] = [{"kwargs": {"neutral_value": copied_value}}]

    receipt = conformance_module._private_label_boundary(
        run_dir=asset_lock.parent,
        jobs=[job],
        host_scorer_names=("comparison.deterministic.answer",),
    )

    assert receipt["status"] == "failed"
    assert receipt["rendered_private_fields"] == []
    assert any("neutral_value" in path for path in receipt["tainted_agent_inputs"])


def test_private_boundary_rejects_leaves_copied_from_nested_private_list(
    tmp_path: Path,
) -> None:
    run_id = "run-neutral-nested-private-copy"
    job = _local_job(tmp_path, run_id)
    asset_lock = tmp_path / ".fugue/runtime" / run_id / "evaluation-assets.json"
    asset_lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictions": {
                    "prediction-a": {
                        "task_id": "public-task-a",
                        "expected": {
                            "matrix": [
                                ["host-only-nested-leaf", 271_828],
                                [True],
                            ]
                        },
                    }
                },
            }
        )
        + "\n"
    )
    asset_lock.chmod(0o600)
    job.config["agents"] = [
        {
            "kwargs": {
                "neutral_copy": {
                    "message": "host-only-nested-leaf",
                    "threshold": 271_828,
                    "allowed": True,
                }
            }
        }
    ]

    receipt = conformance_module._private_label_boundary(
        run_dir=asset_lock.parent,
        jobs=[job],
        host_scorer_names=("comparison.deterministic.answer",),
    )

    assert receipt["status"] == "failed"
    assert receipt["rendered_private_fields"] == []
    assert {
        path.rsplit(".", 1)[-1] for path in receipt["tainted_agent_inputs"]
    } >= {"allowed", "message", "threshold"}


def test_private_boundary_rejects_host_only_bundle_mount(tmp_path: Path) -> None:
    run_id = "run-private-mount"
    job = _local_job(tmp_path, run_id)
    asset_lock = tmp_path / ".fugue/runtime" / run_id / "evaluation-assets.json"
    job.config["environment"] = {
        "mounts": [{"source": asset_lock.as_posix(), "target": "/input/data.json"}]
    }

    receipt = conformance_module._private_label_boundary(
        run_dir=asset_lock.parent,
        jobs=[job],
        host_scorer_names=("comparison.deterministic.answer",),
    )

    assert receipt["status"] == "failed"
    assert any("source" in path for path in receipt["tainted_agent_inputs"])


def test_private_boundary_does_not_treat_correct_output_as_input_leakage(
    tmp_path: Path,
) -> None:
    run_id = "run-correct-output"
    job = _local_job(tmp_path, run_id)
    asset_lock = tmp_path / ".fugue/runtime" / run_id / "evaluation-assets.json"
    private_bundle = {
        "prediction-a": {
            "task_id": "task-a",
            "expected_evidence_paths": ["host-only/answer.json"],
        }
    }
    asset_lock.write_text(
        json.dumps({"schema_version": 1, "predictions": private_bundle}) + "\n"
    )
    asset_lock.chmod(0o600)
    job.result_path.write_text(json.dumps({"answer": private_bundle}) + "\n")

    receipt = conformance_module._private_label_boundary(
        run_dir=asset_lock.parent,
        jobs=[job],
        host_scorer_names=("comparison.deterministic.answer",),
    )

    assert receipt["status"] == "passed"
    assert receipt["tainted_agent_inputs"] == []


def test_private_boundary_does_not_treat_correct_scalar_output_as_input_leakage(
    tmp_path: Path,
) -> None:
    run_id = "run-correct-scalar-output"
    job = _local_job(tmp_path, run_id)
    asset_lock = tmp_path / ".fugue/runtime" / run_id / "evaluation-assets.json"
    private_value = "host-only-correct-scalar"
    asset_lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictions": {
                    "prediction-a": {
                        "task_id": "public-task-a",
                        "expected": {"answer": private_value},
                    }
                },
            }
        )
        + "\n"
    )
    asset_lock.chmod(0o600)
    job.result_path.write_text(json.dumps({"answer": private_value}) + "\n")

    receipt = conformance_module._private_label_boundary(
        run_dir=asset_lock.parent,
        jobs=[job],
        host_scorer_names=("comparison.deterministic.answer",),
    )

    assert receipt["status"] == "passed"
    assert receipt["tainted_agent_inputs"] == []


def test_harbor_cleanup_attributes_new_container_from_exact_run_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-labeled"
    job = _local_job(tmp_path, run_id)
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(list(command))
        if command == ["docker", "container", "ls", "--all", "--quiet"]:
            return subprocess.CompletedProcess(command, 0, stdout="matching\n", stderr="")
        if command[-1] == "matching":
            value = [
                {
                    "Config": {
                        "Labels": {},
                        "Env": [f"FUGUE_RUN_ID={run_id}"],
                    },
                    "Mounts": [],
                }
            ]
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(value), stderr=""
            )
        if command[1:3] == ["container", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[1:3] == ["network", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(conformance_module.subprocess, "run", run)

    result = write_harbor_run_conformance_receipt(
        repo_root=tmp_path,
        run_id=run_id,
        jobs=[job],
        env={"WANDB_API_KEY": "wandb-secret-value"},
        pre_execution_inventory=_inventory("preexisting"),
    )

    value = json.loads(result.path.read_text())
    assert result.status == "failed"
    assert value["docker_cleanup"]["matched_containers"] == [
        {
            "attribution": "exact_fugue_run_environment",
            "compose_project": "",
            "container_id": "matching",
        }
    ]
    assert all("rm" not in command and "stop" not in command for command in calls)


def test_harbor_receipt_allows_agent_only_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-agent-runtime-only"
    job = _local_job(tmp_path, run_id)
    job.config["fugue"].pop("task_runtime")
    job.config_path.write_text(json.dumps(job.config))

    monkeypatch.setattr(
        conformance_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="",
            stderr="",
        ),
    )

    result = write_harbor_run_conformance_receipt(
        repo_root=tmp_path,
        run_id=run_id,
        jobs=[job],
        env={"WANDB_API_KEY": "wandb-secret-value"},
        pre_execution_inventory=_inventory(),
    )

    value = read_harbor_run_conformance_receipt(
        repo_root=tmp_path,
        run_id=run_id,
    )
    assert result.status == "passed"
    assert (
        value["execution_identity"]["jobs"][0]["task_runtime"]["status"]
        == "not_applicable"
    )


def test_harbor_receipt_fails_closed_without_docker_and_never_writes_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-no-docker"
    job = _local_job(tmp_path, run_id)
    secret = "wandb-secret-value"
    job.config["fugue"]["agent_runtime"]["image"] = f"registry.invalid/{secret}"
    job.config_path.write_text(json.dumps(job.config))

    def run(command, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(conformance_module.subprocess, "run", run)

    result = write_harbor_run_conformance_receipt(
        repo_root=tmp_path,
        run_id=run_id,
        jobs=[job],
        env={"WANDB_API_KEY": secret},
        pre_execution_inventory=_inventory(),
    )

    value = json.loads(result.path.read_text())
    assert result.status == "failed"
    assert value["docker_cleanup"]["status"] == "unavailable"
    assert value["redaction_applied"] is True
    assert value["execution_identity"]["jobs"][0]["agent_runtime"]["image"] == (
        "registry.invalid/[redacted]"
    )
    assert secret not in result.path.read_text()


def test_capture_local_inventory_is_read_only_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _local_job(tmp_path, "run-inventory")
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="container-b\ncontainer-a\n",
            stderr="",
        )

    monkeypatch.setattr(conformance_module.subprocess, "run", run)

    inventory = capture_local_docker_inventory([job])

    assert inventory == {
        "schema_version": 1,
        "status": "passed",
        "backend": "local_harbor_docker",
        "container_ids": ["container-a", "container-b"],
    }
    assert calls == [["docker", "container", "ls", "--all", "--quiet"]]


def test_harbor_cleanup_detects_exact_run_network_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _local_job(tmp_path, "run-network")

    def run(command, **kwargs):
        if command[1:3] == ["network", "ls"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="network-left\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(conformance_module.subprocess, "run", run)

    result = write_harbor_run_conformance_receipt(
        repo_root=tmp_path,
        run_id="run-network",
        jobs=[job],
        env={"WANDB_API_KEY": "wandb-secret-value"},
        pre_execution_inventory=_inventory(),
    )

    value = json.loads(result.path.read_text())
    assert result.status == "failed"
    assert value["docker_cleanup"]["matched_networks"] == [
        {
            "compose_project": "task__abc123__env",
            "network_id": "network-left",
        }
    ]


def test_harbor_receipt_rejects_execution_fingerprint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _local_job(tmp_path, "run-fingerprint-drift")
    job.config["fugue"]["execution_fingerprint"] = "f" * 64
    job.config_path.write_text(json.dumps(job.config))
    monkeypatch.setattr(
        conformance_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )

    result = write_harbor_run_conformance_receipt(
        repo_root=tmp_path,
        run_id="run-fingerprint-drift",
        jobs=[job],
        env={"WANDB_API_KEY": "wandb-secret-value"},
        pre_execution_inventory=_inventory(),
    )

    value = json.loads(result.path.read_text())
    assert result.status == "failed"
    [identity] = value["execution_identity"]["jobs"]
    assert identity["execution_fingerprint_verified"] is False
    assert any(
        "execution fingerprint does not match" in issue
        for issue in value["execution_identity"]["issues"]
    )


def test_harbor_receipt_reverifies_rendered_policy_after_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _local_job(tmp_path, "run-policy-drift")
    job.config["environment"] = {"privileged": True}
    job.config_path.write_text(json.dumps(job.config))
    monkeypatch.setattr(
        conformance_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )

    result = write_harbor_run_conformance_receipt(
        repo_root=tmp_path,
        run_id="run-policy-drift",
        jobs=[job],
        env={"WANDB_API_KEY": "wandb-secret-value"},
        pre_execution_inventory=_inventory(),
    )

    value = json.loads(result.path.read_text())
    assert result.status == "failed"
    [identity] = value["execution_identity"]["jobs"]
    assert identity["policy"]["verification"] == "failed"
    assert any(
        "policy attestation did not reverify" in issue
        for issue in value["execution_identity"]["issues"]
    )


def test_harbor_cleanup_is_unavailable_for_unattributed_new_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _local_job(tmp_path, "run-unattributed")

    def run(command, **kwargs):
        if command == ["docker", "container", "ls", "--all", "--quiet"]:
            return subprocess.CompletedProcess(command, 0, "unknown-new\n", "")
        if command == ["docker", "container", "inspect", "unknown-new"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    [{"Config": {"Labels": {}, "Env": []}, "Mounts": []}]
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(conformance_module.subprocess, "run", run)

    result = write_harbor_run_conformance_receipt(
        repo_root=tmp_path,
        run_id="run-unattributed",
        jobs=[job],
        env={"WANDB_API_KEY": "wandb-secret-value"},
        pre_execution_inventory=_inventory(),
    )

    value = json.loads(result.path.read_text())
    assert result.status == "unavailable"
    assert value["docker_cleanup"]["unattributed_new_containers"] == [
        "unknown-new"
    ]
