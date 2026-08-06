from __future__ import annotations

import copy
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    _public_source_publication_receipt_path,
    _verify_local_public_source_drift,
    load_comparison,
    materialize_comparison,
    prepare_comparison,
    scaffold_comparison,
)
from fugue.bench.operator import OperatorService
from fugue.bench.public_source import (
    PublicTaskSourceError,
    PublicTaskSourceManifestV1,
    WandbPublicTaskSourceRemote,
    public_task_source_manifest_from_dict,
    public_task_source_publication_receipt_from_dict,
    publish_public_task_source,
    verify_public_task_source_publication,
)
from fugue.model_plane import EvidenceDestinationV1


def _destination(project: str = "source-project") -> EvidenceDestinationV1:
    return EvidenceDestinationV1(
        entity="wandb",
        project=project,
        api_base_url="https://api.wandb.ai",
        trace_base_url="https://trace.wandb.ai",
        app_base_url="https://wandb.ai",
    )


def _manifest() -> PublicTaskSourceManifestV1:
    unsigned_lock = {
        "schema_version": 1,
        "kind": "public_task_source_lock",
        "public_tasks_sha256": "1" * 64,
        "compiled_public_cases_sha256": "2" * 64,
        "task_resources": [],
    }
    source_lock = {**unsigned_lock, "lock_digest": stable_digest(unsigned_lock)}
    return PublicTaskSourceManifestV1(
        comparison_id="skill-upgrade",
        destination=_destination(),
        source_lock=source_lock,
        public_cases=(
            {
                "id": "task-a",
                "input": {"question": "Repair the bounded public fixture."},
                "attachments": [],
            },
        ),
    )


class _FakeRemote:
    def __init__(self) -> None:
        self.publishes = 0
        self.resolved: dict[str, Any] | None = None

    def publish(
        self, *, manifest: PublicTaskSourceManifestV1, publication_id: str
    ) -> Mapping[str, Any]:
        self.publishes += 1
        source_object = {
            "artifact_name": f"fugue-task-source-{publication_id[:20]}",
            "artifact_version": "v0",
            "artifact_digest": "wandb-digest-v0",
            "qualified_name": (
                f"{manifest.destination.project_slug}/"
                f"fugue-task-source-{publication_id[:20]}:v0"
            ),
            "artifact_ref": (
                "wandb-artifact://"
                f"{manifest.destination.project_slug}/"
                f"fugue-task-source-{publication_id[:20]}:v0"
            ),
            "artifact_url": (
                f"https://wandb.ai/{manifest.destination.project_slug}/artifacts/"
                f"fugue-task-source-{publication_id[:20]}/v0"
            ),
        }
        self.resolved = source_object
        return {"publication_run_id": "source-run", "source_object": source_object}

    def resolve(
        self, *, artifact_ref: str, destination: EvidenceDestinationV1
    ) -> Mapping[str, Any]:
        assert artifact_ref.startswith(
            f"wandb-artifact://{destination.project_slug}/"
        )
        assert self.resolved is not None
        return self.resolved


def test_public_source_contract_is_strict_idempotent_and_drift_sensitive() -> None:
    manifest = _manifest()
    remote = _FakeRemote()

    first = publish_public_task_source(manifest, remote=remote)
    parsed = public_task_source_publication_receipt_from_dict(first.to_dict())

    assert parsed.receipt_digest == first.receipt_digest
    assert parsed.source_lock_digest == manifest.source_lock["lock_digest"]
    assert parsed.content_digest == manifest.content_digest
    assert parsed.destination.project_slug == "wandb/source-project"

    unsafe = copy.deepcopy(first.to_dict())
    unsafe["source_object"]["artifact_url"] += "?token=secret"
    with pytest.raises(PublicTaskSourceError, match="artifact URL is invalid"):
        public_task_source_publication_receipt_from_dict(unsafe)

    remote.resolved = {
        **dict(remote.resolved or {}),
        "artifact_digest": "different-remote-digest",
    }
    with pytest.raises(PublicTaskSourceError, match="drifted"):
        verify_public_task_source_publication(
            parsed,
            manifest=manifest,
            remote=remote,
        )


def test_public_source_manifest_rejects_private_truth_and_digest_tampering() -> None:
    raw = _manifest().to_dict()
    private = copy.deepcopy(raw)
    private["public_cases"][0]["expected"] = {"answer": "secret"}
    private["content_digest"] = ""
    with pytest.raises(PublicTaskSourceError, match="private evaluation"):
        public_task_source_manifest_from_dict(private)

    tampered = copy.deepcopy(raw)
    tampered["source_lock"]["public_tasks_sha256"] = "f" * 64
    with pytest.raises(PublicTaskSourceError, match="lock digest"):
        public_task_source_manifest_from_dict(tampered)


class CommError(Exception):
    pass


class _FakeArtifact:
    def __init__(self, name: str, metadata: Mapping[str, Any]) -> None:
        self.name = name
        self.metadata = dict(metadata)
        self.version = "v0"
        self.digest = "artifact-digest-v0"
        self.qualified_name = f"wandb/source-project/{name}:v0"
        self.url = f"https://wandb.ai/wandb/source-project/artifacts/{name}/v0"

    def new_file(self, _name: str, *, mode: str) -> io.StringIO:
        assert mode == "w"
        return io.StringIO()

    def wait(self) -> None:
        return None


class _FakeApi:
    def __init__(self, module: _FakeWandb) -> None:
        self.module = module

    def artifact(self, path: str, *, type: str) -> _FakeArtifact:
        assert type == "fugue-public-task-source"
        artifact = self.module.artifacts.get(path)
        if artifact is None:
            artifact = next(
                (
                    item
                    for key, item in self.module.artifacts.items()
                    if key.replace(":v0", ":locked") == path
                ),
                None,
            )
        if artifact is None:
            raise CommError(path)
        return artifact


class _FakeRun:
    def __init__(self, module: _FakeWandb, run_id: str) -> None:
        self.module = module
        self.id = run_id
        self.summary: dict[str, Any] = {}

    def log_artifact(
        self, artifact: _FakeArtifact, *, aliases: tuple[str, ...]
    ) -> _FakeArtifact:
        assert aliases == ("locked",)
        self.module.log_count += 1
        self.module.artifacts[artifact.qualified_name] = artifact
        return artifact

    def finish(self, *, exit_code: int) -> None:
        assert exit_code in {0, 1}


class _FakeWandb:
    def __init__(self) -> None:
        self.run: _FakeRun | None = None
        self.artifacts: dict[str, _FakeArtifact] = {}
        self.log_count = 0
        self.init_calls: list[dict[str, Any]] = []

    def Settings(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_calls.append(kwargs)
        if self.run is None:
            self.run = _FakeRun(self, str(kwargs["id"]))
        return self.run

    def Artifact(
        self, *, name: str, type: str, metadata: Mapping[str, Any]
    ) -> _FakeArtifact:
        assert type == "fugue-public-task-source"
        return _FakeArtifact(name, metadata)

    def Api(self, **kwargs: Any) -> _FakeApi:
        assert kwargs == {"overrides": {"base_url": "https://api.wandb.ai"}}
        return _FakeApi(self)


def test_wandb_source_adapter_reuses_one_exact_artifact_version() -> None:
    wandb = _FakeWandb()
    remote = WandbPublicTaskSourceRemote(wandb_module=wandb)
    manifest = _manifest()

    first = publish_public_task_source(manifest, remote=remote)
    second = publish_public_task_source(manifest, remote=remote)

    assert second == first
    assert wandb.log_count == 1
    assert all(call["project"] == "source-project" for call in wandb.init_calls)
    assert all(
        call["config"]["fugue"]["excluded_from_task_inputs"] is True
        for call in wandb.init_calls
    )


def test_comparison_preparation_binds_source_publication_into_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_path = scaffold_comparison(tmp_path / "comparison")
    raw = yaml.safe_load(comparison_path.read_text())
    raw["schema_version"] = 3
    raw["execution"].update(
        evidence_project="wandb/result-project",
        evidence_destination={
            "entity": "wandb",
            "project": "result-project",
            "api_base_url": "https://api.wandb.ai",
            "trace_base_url": "https://trace.wandb.ai",
            "app_base_url": "https://wandb.ai",
        },
        source_evidence_project="wandb/source-project",
        source_evidence_destination={
            "entity": "wandb",
            "project": "source-project",
            "api_base_url": "https://api.wandb.ai",
            "trace_base_url": "https://trace.wandb.ai",
            "app_base_url": "https://wandb.ai",
        },
        approval_required=False,
        preparation_required=True,
    )
    comparison_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    root = comparison_path.parent
    spec = load_comparison(comparison_path, repo_root=root)
    monkeypatch.setattr(
        "fugue.bench.comparison._runtime_readiness",
        lambda *_args, **_kwargs: (
            {"agent:codex:amd64": "a" * 64, "task:policy-limit:amd64": "b" * 64},
            [],
        ),
    )
    monkeypatch.setattr(OperatorService, "prepare", lambda *_args, **_kwargs: None)
    remote = _FakeRemote()

    receipt, preview, _path = prepare_comparison(
        spec,
        repo_root=root,
        operator=OperatorService(root),
        source_remote=remote,
    )
    source_receipt = public_task_source_publication_receipt_from_dict(
        __import__("json").loads(
            _public_source_publication_receipt_path(spec, repo_root=root).read_text()
        )
    )
    digests = preview.readiness["qualification_input_digests"]

    assert receipt["qualification_input_digests"] == digests
    assert digests["public_source_publication"] == source_receipt.receipt_digest
    assert digests["public_source_object"] == source_receipt.source_object.identity_digest
    assert digests["public_source_content"] == source_receipt.content_digest
    assert preview.matrix["source_evidence_project"] == "wandb/source-project"
    assert preview.matrix["evidence_project"] == "wandb/result-project"
    _experiment, request = materialize_comparison(
        preview,
        repo_root=root,
        operator=OperatorService(root),
        approval_digest="",
    )
    approved = request.approved_comparison
    assert approved["approved_inputs"]["public_source_publication"] == (
        source_receipt.to_dict()
    )
    assert approved["evidence_project"] == "wandb/result-project"

    monkeypatch.setattr(
        "fugue.bench.comparison.WandbPublicTaskSourceRemote",
        lambda: remote,
    )
    matched = _verify_local_public_source_drift(
        spec,
        readiness=preview.readiness,
        repo_root=root,
    )
    assert matched.status == "matched"

    remote.resolved = {
        **dict(remote.resolved or {}),
        "artifact_version": "v1",
    }
    drifted = _verify_local_public_source_drift(
        spec,
        readiness=preview.readiness,
        repo_root=root,
    )
    assert drifted.status == "unavailable"
    assert "public source verification" in str(drifted.reason)
