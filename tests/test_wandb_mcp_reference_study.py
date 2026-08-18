from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.component_imports import MCPImportLockV1
from fugue.reference_studies.wandb_mcp import (
    PREPARATION_RECEIPT_NAME,
    SOURCE_LOCK_NAME,
    WANDB_MCP_REFERENCE_ROOT,
    WANDB_MCP_RELEASE_REF,
    WANDB_MCP_REPOSITORY_URL,
    WBAF_TASK_DESIGN_PROVENANCE,
    GitSourceSnapshotV1,
    MaterializedReferenceArtifactV1,
    ReferenceMaterializationRequest,
    ReferenceStudyMaterializationV1,
    SubprocessGitReferenceTransport,
    WandbMCPReferencePreparationReceiptV1,
    WandbMCPReferenceSourceLockV1,
    _freeze_hosted_reference_source,
    _materialization_inventory,
    prepare_wandb_mcp_reference_study,
    read_wandb_mcp_reference_lock,
    read_wandb_mcp_reference_receipt,
)

COMMIT = "b" * 40
MOVED_COMMIT = "c" * 40
TREE = "d" * 40
BLOB = "e" * 40
RELEASE_NOTES = b"# W&B MCP 0.4.0\n\nA frozen release note.\n"
_BASELINE_FOR_TEST = "53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0"


class FakeGit:
    def __init__(self, observations: list[str] | None = None) -> None:
        self.observations = observations or [COMMIT, COMMIT]
        self.observe_calls: list[tuple[str, str, Path]] = []
        self.fetch_calls: list[tuple[str, str, Path]] = []

    def observe_ref(
        self, repository_url: str, requested_ref: str, *, cwd: Path
    ) -> str:
        self.observe_calls.append((repository_url, requested_ref, cwd))
        return self.observations.pop(0)

    def fetch_snapshot(
        self,
        repository_url: str,
        source_commit: str,
        *,
        bare_repository: Path,
    ) -> GitSourceSnapshotV1:
        self.fetch_calls.append((repository_url, source_commit, bare_repository))
        bare_repository.mkdir(parents=True)
        (bare_repository / "HEAD").write_text(source_commit, encoding="utf-8")
        return GitSourceSnapshotV1(
            source_commit=source_commit,
            source_tree=TREE,
            release_notes_blob=BLOB,
            release_notes=RELEASE_NOTES,
        )


def _materializer(request: ReferenceMaterializationRequest):
    body = b'{"candidate":"staging-0.4.0"}\n'
    path = request.staging_root / "candidate" / "lock.json"
    path.parent.mkdir()
    path.write_bytes(body)
    inventory = _materialization_inventory(request.staging_root)
    return ReferenceStudyMaterializationV1(
        schema_version=1,
        study_bundle_id="wandb-mcp-staging-0.4.0",
        behavior_inputs_digest="1" * 64,
        execution_inputs_digest="2" * 64,
        inventory_digest=inventory["inventory_digest"],
        total_files=inventory["total_files"],
        total_bytes=inventory["total_bytes"],
        artifacts=(
            MaterializedReferenceArtifactV1(
                schema_version=1,
                path="candidate/lock.json",
                sha256=hashlib.sha256(body).hexdigest(),
                byte_count=len(body),
                private=False,
            ),
        ),
    )


def test_prepare_fences_ref_and_publishes_one_immutable_directory(
    tmp_path: Path,
) -> None:
    git = FakeGit()
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=not-read-by-source-core\n", encoding="utf-8")
    env_file.chmod(0o600)

    receipt = prepare_wandb_mcp_reference_study(
        repo_root=tmp_path,
        env_file=env_file,
        platform="linux/amd64",
        git=git,
        materializer=_materializer,
    )

    destination = tmp_path / WANDB_MCP_REFERENCE_ROOT / COMMIT
    assert destination.is_dir()
    assert receipt == read_wandb_mcp_reference_receipt(destination)
    lock = read_wandb_mcp_reference_lock(destination)
    assert receipt.source_lock_digest == lock.lock_digest
    assert lock.source_commit == COMMIT
    assert lock.source_tree == TREE
    assert lock.release_notes.git_blob == BLOB
    assert lock.release_notes.byte_count == len(RELEASE_NOTES)
    assert lock.release_notes.sha256 == hashlib.sha256(RELEASE_NOTES).hexdigest()
    assert lock.task_design_provenance == (WBAF_TASK_DESIGN_PROVENANCE,)
    assert WBAF_TASK_DESIGN_PROVENANCE.runtime_dependency is False
    assert receipt.materialization is not None
    assert receipt.materialization.behavior_inputs_digest == "1" * 64
    assert receipt.materialization.execution_inputs_digest == "2" * 64
    assert len(git.observe_calls) == 2
    assert len(git.fetch_calls) == 1
    assert all(call[0] == WANDB_MCP_REPOSITORY_URL for call in git.observe_calls)
    assert all(call[1] == WANDB_MCP_RELEASE_REF for call in git.observe_calls)
    serialized = json.dumps(receipt.to_dict(), sort_keys=True)
    assert tmp_path.as_posix() not in serialized
    assert "not-read-by-source-core" not in serialized
    assert (destination / SOURCE_LOCK_NAME).stat().st_mode & 0o222 == 0
    assert (destination / PREPARATION_RECEIPT_NAME).stat().st_mode & 0o222 == 0


def test_default_materializer_builds_a_complete_check_ready_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported: list[tuple[str, str]] = []

    def fake_import(config_path, *, server, import_id, repo_root, **_kwargs):
        assert config_path == repo_root / "mcp.json"
        imported.append((server, import_id))
        return object()

    def fake_lock(
        import_id,
        repo_root,
        *,
        acknowledge_package_code,
        target_platform,
    ):
        assert acknowledge_package_code is True
        assert target_platform == "linux/amd64"
        commit = _BASELINE_FOR_TEST if "main" in import_id else COMMIT
        lock = MCPImportLockV1(
            schema_version=1,
            id=import_id,
            transport="stdio",
            source="mcp.json",
            source_digest=f"sha256:{'3' * 64}",
            runtime_digest=f"sha256:{'4' * 64}",
            runtime_platform=target_platform,
            command=("python", "-m", "wandb_mcp_server"),
            version_identity=f"git:{commit}",
            required_env=("WANDB_API_KEY",),
            fixed_env=(("WANDB_MCP_READ_ONLY", "true"),),
            allowed_hosts=("api.wandb.ai",),
            allowed_tools=("query_wandb_tool",),
            tool_manifest=(
                {
                    "name": "query_wandb_tool",
                    "description": "read-only query",
                    "input_schema": {"type": "object"},
                },
            ),
            tool_manifest_digest=f"sha256:{'5' * 64}",
            server_info={"name": "wandb-mcp", "version": commit[:8]},
            support="supported",
        )
        lock_path = repo_root / ".fugue/imports/mcp/locks" / f"{import_id}.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock.to_dict()), encoding="utf-8")
        integration = (
            repo_root / ".fugue/imports/integrations" / f"{import_id}.yaml"
        )
        integration.parent.mkdir(parents=True, exist_ok=True)
        integration.write_text(
            f"id: {import_id}\nversion: {'6' * 64}\nsupport: supported\n",
            encoding="utf-8",
        )
        return lock

    monkeypatch.setattr(
        "fugue.bench.component_imports.import_mcp_config", fake_import
    )
    monkeypatch.setattr("fugue.bench.component_imports.lock_mcp_import", fake_lock)

    def fake_freeze(*, staging_root, credential):
        assert credential == "unit-test-wandb-key"
        evidence = {"evidence_lock_digest": "7" * 64}
        conformance = {"receipt_digest": "8" * 64}
        (staging_root / "source-evidence.lock.json").write_text(
            json.dumps(evidence), encoding="utf-8"
        )
        (staging_root / "source-conformance-receipt.json").write_text(
            json.dumps(conformance), encoding="utf-8"
        )
        return evidence, conformance

    monkeypatch.setattr(
        "fugue.reference_studies.wandb_mcp._freeze_hosted_reference_source",
        fake_freeze,
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=unit-test-wandb-key\n", encoding="utf-8")
    env_file.chmod(0o600)

    receipt = prepare_wandb_mcp_reference_study(
        repo_root=tmp_path,
        env_file=env_file,
        platform="linux/amd64",
        git=FakeGit(),
    )

    destination = tmp_path / WANDB_MCP_REFERENCE_ROOT / COMMIT
    assert imported == [
        ("wandb-main", f"wandb-mcp-main-{_BASELINE_FOR_TEST[:12]}"),
        ("wandb-0-4-staging", f"wandb-mcp-staging-{COMMIT[:12]}"),
    ]
    comparison = (destination / "comparison.yaml").read_text(encoding="utf-8")
    assert "{{" not in comparison
    assert COMMIT in comparison
    assert f"mcp-main-vs-0-4-{COMMIT[:7]}-harbor-canary-v11" in comparison
    assert f"wandb-mcp-main-{_BASELINE_FOR_TEST[:12]}" in comparison
    assert f"wandb-mcp-staging-{COMMIT[:12]}" in comparison
    assert "evidence_lock: source-evidence.lock.json" in comparison
    assert "source_evidence_project: wandb/fugue-mcp-release-source-v2" in comparison
    assert "research_id: fugue-mcp-release-qualification-v1" in comparison
    assert "study_console_base_url: http://127.0.0.1:18080" in comparison
    assert json.loads((destination / ".fugue-study.json").read_text()) == {
        "kind": "fugue_standalone_study",
        "schema_version": 1,
        "template": "mcp-change",
    }
    for name in (
        "source-evidence.lock.json",
        "source-conformance-receipt.json",
        "release-notes.lock.json",
        "mechanism-receipt.json",
        "tasks.jsonl",
        "tool_surface_scorer_v7.py",
    ):
        assert (destination / name).is_file()
    assert (destination / "private-labels.jsonl").stat().st_mode & 0o777 == 0o600
    assert receipt.materialization is not None
    assert receipt.materialization.total_files > 10
    assert "unit-test-wandb-key" not in json.dumps(receipt.to_dict())


def test_hosted_source_freeze_reads_and_rechecks_exact_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fugue.bench import mcp_release_qualification as qualification

    inventory = {
        "complete": True,
        "inventory_digest": "1" * 64,
        "runs": [{"id": "run-1"}],
    }
    evidence = {
        "evidence_lock_digest": "2" * 64,
        "objects": {"evaluations": [{"call_id": "evaluation-1"}]},
    }
    conformance = {"status": "passed", "receipt_digest": "3" * 64}
    calls: list[str] = []

    monkeypatch.setattr(
        qualification,
        "_qualification_endpoint_binding",
        lambda env: {
            "api_base_url": "https://api.wandb.ai",
            "trace_base_url": "https://trace.wandb.ai",
            "endpoint_digest": "4" * 64,
        },
    )
    monkeypatch.setattr(
        qualification,
        "verify_private_project_topology",
        lambda **_kwargs: {
            "wandb/fugue-mcp-release-source-v2": "PRIVATE"
        },
    )
    monkeypatch.setattr(
        qualification,
        "_stable_hosted_source_inventory",
        lambda *_args, **_kwargs: calls.append("inventory") or inventory,
    )
    monkeypatch.setattr(
        qualification,
        "_inventory_weave_receipts",
        lambda _inventory: {},
    )
    monkeypatch.setattr(
        qualification,
        "_evidence_lock",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(
        qualification,
        "validate_evidence_lock",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        qualification,
        "_fetch_hosted_source_calls",
        lambda **_kwargs: ([{"id": "evaluation-1"}], {"evaluation-1": []}),
    )
    monkeypatch.setattr(
        qualification,
        "build_hosted_source_conformance_receipt",
        lambda **_kwargs: conformance,
    )
    monkeypatch.setattr(
        qualification,
        "verify_hosted_source_drift",
        lambda **_kwargs: calls.append("drift")
        or SimpleNamespace(status="matched"),
    )

    observed = _freeze_hosted_reference_source(
        staging_root=tmp_path,
        credential="operator-key-not-persisted",
    )

    assert observed == (evidence, conformance)
    assert calls == ["inventory", "drift"]
    assert json.loads((tmp_path / "source-evidence.lock.json").read_text()) == evidence
    assert json.loads(
        (tmp_path / "source-conformance-receipt.json").read_text()
    ) == conformance
    assert "operator-key-not-persisted" not in json.dumps(observed)

    inventory["complete"] = False
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(RuntimeError, match="source cohort is incomplete"):
        _freeze_hosted_reference_source(
            staging_root=incomplete,
            credential="operator-key-not-persisted",
        )
    assert list(incomplete.iterdir()) == []


def test_prepare_is_idempotent_but_never_replaces_existing_content(
    tmp_path: Path,
) -> None:
    first = prepare_wandb_mcp_reference_study(
        repo_root=tmp_path, git=FakeGit(), materializer=_materializer
    )
    second = prepare_wandb_mcp_reference_study(
        repo_root=tmp_path, git=FakeGit(), materializer=_materializer
    )
    assert first == second

    destination = tmp_path / WANDB_MCP_REFERENCE_ROOT / COMMIT
    receipt_path = destination / PREPARATION_RECEIPT_NAME
    receipt_path.chmod(0o600)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["receipt_digest"] = "9" * 64
    receipt_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt digest"):
        prepare_wandb_mcp_reference_study(
            repo_root=tmp_path, git=FakeGit(), materializer=_materializer
        )


def test_moving_ref_fails_before_durable_destination(tmp_path: Path) -> None:
    git = FakeGit([COMMIT, MOVED_COMMIT])
    with pytest.raises(RuntimeError, match="moved during preparation"):
        prepare_wandb_mcp_reference_study(
            repo_root=tmp_path, git=git, materializer=_materializer
        )

    reference_root = tmp_path / WANDB_MCP_REFERENCE_ROOT
    assert not (reference_root / COMMIT).exists()
    assert not (reference_root / MOVED_COMMIT).exists()
    assert list(reference_root.iterdir()) == []


def test_materializer_cannot_publish_undeclared_or_secret_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "test-secret-that-must-not-be-persisted"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"ANTHROPIC_API_KEY={secret}\n", encoding="utf-8")
    env_file.chmod(0o600)

    def leaking_materializer(request: ReferenceMaterializationRequest):
        body = secret.encode()
        path = request.staging_root / "leak.txt"
        path.write_bytes(body)
        inventory = _materialization_inventory(request.staging_root)
        return ReferenceStudyMaterializationV1(
            schema_version=1,
            study_bundle_id="leak-check",
            behavior_inputs_digest="1" * 64,
            execution_inputs_digest="2" * 64,
            inventory_digest=inventory["inventory_digest"],
            total_files=inventory["total_files"],
            total_bytes=inventory["total_bytes"],
            artifacts=(
                MaterializedReferenceArtifactV1(
                    schema_version=1,
                    path="leak.txt",
                    sha256=hashlib.sha256(body).hexdigest(),
                    byte_count=len(body),
                    private=False,
                ),
            ),
        )

    with pytest.raises(ValueError, match="host credential"):
        prepare_wandb_mcp_reference_study(
            repo_root=tmp_path,
            env_file=env_file,
            git=FakeGit(),
            materializer=leaking_materializer,
        )
    assert list((tmp_path / WANDB_MCP_REFERENCE_ROOT).iterdir()) == []


def test_lock_and_receipt_parsers_fail_closed_on_unknown_or_changed_content(
    tmp_path: Path,
) -> None:
    prepare_wandb_mcp_reference_study(
        repo_root=tmp_path, git=FakeGit(), materializer=_materializer
    )
    destination = tmp_path / WANDB_MCP_REFERENCE_ROOT / COMMIT
    lock = read_wandb_mcp_reference_lock(destination)
    receipt = read_wandb_mcp_reference_receipt(destination)

    lock_unknown = lock.to_dict()
    lock_unknown["future"] = True
    with pytest.raises(ValueError, match="unknown=future"):
        WandbMCPReferenceSourceLockV1.from_dict(lock_unknown)

    receipt_changed = receipt.to_dict()
    receipt_changed["destination"] = ".fugue/reference-studies/wandb-mcp/other"
    with pytest.raises(ValueError, match="destination"):
        WandbMCPReferencePreparationReceiptV1.from_dict(receipt_changed)

    with pytest.raises(ValueError, match="digest"):
        WandbMCPReferenceSourceLockV1.from_dict(
            {**lock.to_dict(), "candidate_source_digest": "f" * 64}
        )


def test_subprocess_transport_is_noninteractive_bounded_and_shell_free(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def runner(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{COMMIT}\t{WANDB_MCP_RELEASE_REF}\n".encode(),
            stderr=b"",
        )

    transport = SubprocessGitReferenceTransport(
        timeout_seconds=7, fetch_timeout_seconds=11, runner=runner
    )
    assert (
        transport.observe_ref(
            WANDB_MCP_REPOSITORY_URL, WANDB_MCP_RELEASE_REF, cwd=tmp_path
        )
        == COMMIT
    )
    assert calls[0]["timeout"] == 7
    assert calls[0]["check"] is False
    assert "shell" not in calls[0]
    environment = calls[0]["env"]
    assert isinstance(environment, dict)
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    command = calls[0]["command"]
    assert isinstance(command, list)
    assert "credential.helper=" in command
    assert "protocol.file.allow=never" in command


def test_snapshot_commit_mismatch_and_unsafe_platform_fail_closed(
    tmp_path: Path,
) -> None:
    class WrongSnapshotGit(FakeGit):
        def fetch_snapshot(self, repository_url, source_commit, *, bare_repository):
            snapshot = super().fetch_snapshot(
                repository_url, source_commit, bare_repository=bare_repository
            )
            return replace(snapshot, source_commit=MOVED_COMMIT)

    with pytest.raises(ValueError, match="observations do not reconcile|task-design|source"):
        prepare_wandb_mcp_reference_study(repo_root=tmp_path, git=WrongSnapshotGit())
    with pytest.raises(ValueError, match="platform"):
        prepare_wandb_mcp_reference_study(
            repo_root=tmp_path, platform="darwin/arm64", git=FakeGit()
        )


def test_wbaf_provenance_is_task_design_only_and_digest_bound() -> None:
    value = WBAF_TASK_DESIGN_PROVENANCE.to_dict()
    assert value["runtime_dependency"] is False
    assert value["role"] == "task_design_reference"
    assert value["source_commit"] == "e2d8d670017bc426b68a311c5777c3b9084023f3"
    assert value["source_tree"] == "7776ac62bbe32da5c6824809893d70ea6725a42e"
    assert value["provenance_digest"] == (
        "06217bae1c31aa445045dc4022d1cd65d98ce8581d87a924f38fdc81fd05dfaa"
    )
    assert value["files"] == [
        {
            "schema_version": 1,
            "path": "data/evals/mcp-all.yaml",
            "git_blob": "6165211befa7b5af60468f41b195d29c7c29a7ed",
            "sha256": "d0dc3ea830cb9ccb2e5d57bbef54712f46e335ca731bb873072201c43a305624",
            "byte_count": 1016,
        },
        {
            "schema_version": 1,
            "path": "data/evals/mcp-ci.yaml",
            "git_blob": "201a0f0cb1b77845378e834e4e2db08b89570d2d",
            "sha256": "777985c511d405795e93b2df75ba6c6d6f7d723e6a43123b0d483fa84f91ba6e",
            "byte_count": 471,
        },
        {
            "schema_version": 1,
            "path": "docs/tasks.md",
            "git_blob": "52bf9f093503ae3ab022942302a54d5d1df142a7",
            "sha256": "133bd8924378dede4ba73a30a9cd5298a9beb741f6d8bdd9e57e8a4939b59033",
            "byte_count": 8985,
        },
    ]
    assert "API_KEY" not in json.dumps(value)
    assert os.path.isabs(value["repository_url"]) is False
