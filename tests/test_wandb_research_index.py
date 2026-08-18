from __future__ import annotations

import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

import fugue.bench.research_index as research_index_module
import fugue.bench.wandb_research_index as adapter
from fugue import weave_support
from fugue.bench.candidates import stable_digest
from fugue.bench.local_publication import (
    StudyPublicationScopeV1,
    WeavePublicationTargetV1,
)
from fugue.bench.research_index import (
    ResearchCandidateAssignmentV1,
    ResearchCandidateDefinitionV1,
    ResearchEvidenceRefV1,
    ResearchIndexPublicationTargetV1,
    ResearchIndexV1,
    ResearchStudyIndexEntryV1,
)
from fugue.bench.wandb_research_index import (
    WANDB_STUDY_INDEX_JOB_TYPE,
    WANDB_STUDY_INDEX_RECORD_KIND,
)


def _digest(seed: int) -> str:
    return "0123456789abcdef"[seed % 16] * 64


@dataclass(frozen=True)
class _EmbeddedAttempt:
    attempt_id: str
    identity: dict[str, str]


@dataclass(frozen=True)
class _EmbeddedPair:
    baseline: _EmbeddedAttempt
    candidate: _EmbeddedAttempt
    harness: str = "claude-code"


@dataclass(frozen=True)
class _EmbeddedResultV3:
    comparison_id: str
    result_digest: str
    qualification_digest: str
    rows: int
    paired_cases: tuple[_EmbeddedPair, ...]
    candidate_definitions: dict[str, dict[str, str]]
    behavioral_summary: SimpleNamespace
    decision: SimpleNamespace
    task_validity: tuple[SimpleNamespace, ...]
    evidence_backend: str = "local"
    local_chain_integrity: str = "reconciled"
    hosted_chain_integrity: str = "not_applicable"


@pytest.fixture(autouse=True)
def _canonical_embedded_source_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parse_result(payload: str):
        raw = json.loads(payload)
        attempts = tuple(str(item) for item in raw["attempt_ids"])
        candidate_id = str(raw["candidate_id"])
        pairs = tuple(
            _EmbeddedPair(
                baseline=_EmbeddedAttempt(
                    attempt_id=attempts[position],
                    identity={
                        "candidate": candidate_id,
                        "harness": "claude-code",
                    },
                ),
                candidate=_EmbeddedAttempt(
                    attempt_id=attempts[position + 1],
                    identity={
                        "candidate": candidate_id,
                        "harness": "claude-code",
                    },
                ),
            )
            for position in range(0, len(attempts), 2)
        )
        return _EmbeddedResultV3(
            comparison_id=str(raw["comparison_id"]),
            result_digest=str(raw["result_digest"]),
            qualification_digest=str(raw["qualification_digest"]),
            rows=int(raw["rows"]),
            paired_cases=pairs,
            candidate_definitions={
                candidate_id: dict(raw["candidate_definition"])
            },
            behavioral_summary=SimpleNamespace(
                status=str(raw["behavioral_status"]),
                recommendation=str(raw["recommendation"]),
            ),
            decision=SimpleNamespace(
                evidence_grade="A",
                status=str(raw["decision_status"]),
                recommendation=str(raw["decision_recommendation"]),
            ),
            task_validity=(
                SimpleNamespace(status=str(raw["task_validity_status"])),
            ),
        )

    def parse_receipt(raw: dict[str, Any]):
        target_raw = dict(raw["target"])
        scope_raw = dict(target_raw["study_scope"])
        target = WeavePublicationTargetV1(
            entity=str(target_raw["entity"]),
            project=str(target_raw["project"]),
            study_scope=StudyPublicationScopeV1(
                research_id=str(scope_raw["research_id"]),
                study_id=str(scope_raw["study_id"]),
            ),
        )
        return SimpleNamespace(
            target=target,
            result_digest=str(raw["result_digest"]),
            qualification_digest=str(raw["qualification_digest"]),
            result_file_sha256=str(raw["result_file_sha256"]),
            receipt_digest=str(raw["receipt_digest"]),
            hosted_objects=tuple(
                SimpleNamespace(**dict(item)) for item in raw["hosted_objects"]
            ),
        )

    monkeypatch.setattr(
        research_index_module,
        "ComparisonResultV3",
        _EmbeddedResultV3,
    )
    monkeypatch.setattr(
        research_index_module,
        "comparison_result_from_json",
        parse_result,
    )
    monkeypatch.setattr(
        research_index_module,
        "weave_publication_receipt_from_dict",
        parse_receipt,
    )


def _entry(
    *,
    study_id: str,
    rows: int,
    seed: int,
    research_id: str = "community-skill-studies-v1",
) -> ResearchStudyIndexEntryV1:
    definition = {"arm": "candidate", "study": study_id}
    candidate_id = stable_digest(definition)
    attempts = tuple(
        stable_digest({"study": study_id, "attempt": position})
        for position in range(rows)
    )
    result_digest = _digest(seed + 1)
    qualification_digest = _digest(seed + 2)
    recommendation = (
        "Advance to confirmation."
        if study_id == "study-a"
        else "Use harder frozen tasks."
    )
    behavioral_status = "improved" if study_id == "study-a" else "unchanged"
    evidence_refs = tuple(
        sorted(
            (
                ResearchEvidenceRefV1(
                    attempt_id=attempt,
                    kind=kind,
                    ref=(
                        f"weave:///wandb/{study_id}/object/{attempt}-{kind}"
                        if kind == "dataset"
                        else f"weave:///wandb/{study_id}/call/{attempt}-{kind}"
                    ),
                )
                for attempt in attempts
                for kind in (
                    "agent_evidence_receipt",
                    "dataset",
                    "evaluation_root",
                    "prediction",
                    "prediction_and_score",
                )
            ),
            key=lambda item: (item.attempt_id, item.kind),
        )
    )
    result_json = json.dumps(
        {
            "schema_version": 3,
            "comparison_id": f"comparison-{study_id}",
            "result_digest": result_digest,
            "qualification_digest": qualification_digest,
            "rows": rows,
            "attempt_ids": list(attempts),
            "candidate_id": candidate_id,
            "candidate_definition": definition,
            "behavioral_status": behavioral_status,
            "recommendation": recommendation,
            "decision_status": "inconclusive",
            "decision_recommendation": "Package release was not evaluated.",
            "task_validity_status": "valid",
        },
        sort_keys=True,
    )
    result_file_sha256 = __import__("hashlib").sha256(
        result_json.encode("utf-8")
    ).hexdigest()
    publication_receipt_digest = _digest(seed + 4)
    publication_receipt_json = json.dumps(
        {
            "schema_version": 1,
            "receipt_digest": publication_receipt_digest,
            "result_digest": result_digest,
            "qualification_digest": qualification_digest,
            "result_file_sha256": result_file_sha256,
            "target": {
                "entity": "wandb",
                "project": study_id,
                "study_scope": {
                    "research_id": research_id,
                    "study_id": study_id,
                },
            },
            "hosted_objects": [item.to_dict() for item in evidence_refs],
        },
        sort_keys=True,
    )
    return ResearchStudyIndexEntryV1(
        research_id=research_id,
        study_id=study_id,
        comparison_id=f"comparison-{study_id}",
        result_digest=result_digest,
        qualification_digest=qualification_digest,
        result_file_sha256=result_file_sha256,
        publication_receipt_digest=publication_receipt_digest,
        publication_receipt_file_sha256=__import__("hashlib")
        .sha256(publication_receipt_json.encode("utf-8"))
        .hexdigest(),
        project=f"wandb/{study_id}",
        behavioral_status=behavioral_status,
        behavioral_recommendation=recommendation,
        decision_status="inconclusive",
        decision_recommendation="Package release was not evaluated.",
        task_validity_status="valid",
        rows=rows,
        evidence_integrity_grade="A",
        evidence_backend="local",
        local_chain_integrity="reconciled",
        result_hosted_chain_integrity="not_applicable",
        published_chain_integrity="reconciled",
        candidate_ids=(candidate_id,),
        candidate_definitions=(
            ResearchCandidateDefinitionV1.from_definition(
                candidate_id,
                definition,
            ),
        ),
        candidate_assignments=(
            ResearchCandidateAssignmentV1(
                role="baseline",
                harness="claude-code",
                candidate_id=candidate_id,
            ),
            ResearchCandidateAssignmentV1(
                role="candidate",
                harness="claude-code",
                candidate_id=candidate_id,
            ),
        ),
        evidence_refs=evidence_refs,
        result_json=result_json,
        publication_receipt_json=publication_receipt_json,
    )


def _index(
    research_id: str = "community-skill-studies-v1",
) -> ResearchIndexV1:
    studies = (
        _entry(study_id="study-a", rows=4, seed=1, research_id=research_id),
        _entry(study_id="study-b", rows=8, seed=8, research_id=research_id),
    )
    return ResearchIndexV1(
        research_id=research_id,
        title="Community Skill studies",
        objective="Compare exact Skill revisions on locked tasks.",
        studies=studies,
        study_count=2,
        total_rows=12,
    )


def _index_bytes(index: ResearchIndexV1) -> bytes:
    return (
        json.dumps(index.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")


class _FakeTable:
    def __init__(
        self,
        *,
        columns: list[str],
        data: list[list[Any]],
        log_mode: str,
    ) -> None:
        self.columns = columns
        self.data = data
        self.log_mode = log_mode


class _FakeRunFile:
    def __init__(self, sdk: _FakeWandb, name: str, body: bytes) -> None:
        self._sdk = sdk
        self.name = name
        self._body = body

    def download(self, *, root: str, replace: bool):
        self._sdk.table_download_calls.append((root, replace))
        destination = Path(root) / self.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not replace:
            raise ValueError("table download already exists")
        destination.write_bytes(self._body)
        return destination.open(encoding="utf-8")


class _FakeArtifact:
    def __init__(
        self,
        sdk: _FakeWandb,
        name: str,
        *,
        type: str,
        metadata: dict[str, Any],
    ) -> None:
        self._sdk = sdk
        self.base_name = name
        self.type = type
        self.metadata = dict(metadata)
        self.files: dict[str, bytes] = {}
        self.file_policies: dict[str, str | None] = {}
        self.entity = ""
        self.project = ""
        self.version = ""
        self.name = name
        self.qualified_name = ""
        self.url = ""

    def add_file(
        self,
        local_path: str,
        *,
        name: str,
        policy: str | None = None,
    ) -> None:
        self.files[name] = Path(local_path).read_bytes()
        self.file_policies[name] = policy

    def wait(self) -> _FakeArtifact:
        self._sdk.wait_calls += 1
        return self

    def download(self, *, root: str, skip_cache: bool) -> str:
        self._sdk.download_calls.append((root, skip_cache))
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=False)
        for name, body in self.files.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        return str(destination)


class _FakeRun:
    def __init__(self, sdk: _FakeWandb, kwargs: dict[str, Any]) -> None:
        self._sdk = sdk
        self.kwargs = dict(kwargs)
        self.entity = str(kwargs["entity"])
        self.project = str(kwargs["project"])
        self.id = str(kwargs["id"])
        self.job_type = str(kwargs["job_type"])
        self.config = dict(kwargs["config"])
        app_origin = os.environ.get("WANDB_APP_BASE_URL", "https://wandb.ai").rstrip(
            "/"
        )
        self.url = (
            f"{app_origin}/{self.entity}/{self.project}/runs/{self.id}"
        )
        self.summary: dict[str, Any] = {}
        self.files: dict[str, bytes] = {}
        self.logs: list[tuple[dict[str, Any], bool]] = []
        self.finish_calls: list[int | None] = []

    def log(self, payload: dict[str, Any], *, commit: bool) -> None:
        self.logs.append((dict(payload), commit))
        table = payload.get(adapter._TABLE_KEY)
        if isinstance(table, _FakeTable):
            table_payload = {
                "columns": table.columns,
                "data": table.data,
            }
            table_body = json.dumps(
                table_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            table_sha256 = __import__("hashlib").sha256(table_body).hexdigest()
            table_path = (
                f"media/table/studies_0_{table_sha256[:20]}.table.json"
            )
            self.files[table_path] = table_body
            self.summary[adapter._TABLE_KEY] = {
                "_type": "table-file",
                "ncols": len(table.columns),
                "nrows": len(table.data),
                "log_mode": table.log_mode,
                "path": table_path,
                "sha256": table_sha256,
                "size": len(table_body),
            }
            if self._sdk.raise_after_table_log:
                raise RuntimeError("injected failure after table log")

    def file(self, name: str) -> _FakeRunFile:
        self._sdk.run_file_paths.append(name)
        body = self.files[name]
        return _FakeRunFile(self._sdk, name, body)

    def log_artifact(self, artifact: _FakeArtifact) -> _FakeArtifact:
        artifact.entity = self.entity
        artifact.project = self.project
        artifact.version = "v0"
        artifact.name = artifact.base_name
        artifact.qualified_name = (
            f"{self.entity}/{self.project}/{artifact.base_name}:v0"
        )
        app_origin = os.environ.get("WANDB_APP_BASE_URL", "https://wandb.ai").rstrip(
            "/"
        )
        artifact.url = (
            f"{app_origin}/{self.entity}/{self.project}/artifacts/"
            f"{artifact.type}/{artifact.base_name}/v0"
        )
        self._sdk.logged_artifact = artifact
        if self._sdk.raise_after_artifact_log:
            raise RuntimeError("injected failure after artifact log")
        return artifact

    def finish(self, *, exit_code: int | None = None) -> None:
        self.finish_calls.append(exit_code)
        if exit_code == 0 and self._sdk.raise_after_success_finish:
            raise RuntimeError("injected failure after successful finish")


class _FakeApi:
    def __init__(self, sdk: _FakeWandb) -> None:
        self._sdk = sdk

    def run(self, path: str) -> _FakeRun:
        self._sdk.api_run_paths.append(path)
        if self._sdk.run is None:
            raise AssertionError("fake Run was not initialized")
        if self._sdk.mutate_run_config:
            self._sdk.run.config["index_digest"] = _digest(15)
        if self._sdk.mutate_table_summary:
            self._sdk.run.summary[adapter._TABLE_KEY]["nrows"] = 999
        if self._sdk.mutate_table_cell:
            table = self._sdk.run.summary[adapter._TABLE_KEY]
            old_path = str(table["path"])
            payload = json.loads(self._sdk.run.files.pop(old_path))
            payload["data"][0][0] = "changed-study"
            body = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            sha256 = __import__("hashlib").sha256(body).hexdigest()
            path = f"media/table/studies_0_{sha256[:20]}.table.json"
            self._sdk.run.files[path] = body
            table.update(path=path, sha256=sha256, size=len(body))
        if self._sdk.mutate_table_path:
            table = self._sdk.run.summary[adapter._TABLE_KEY]
            old_path = str(table["path"])
            path = f"media/table/changed_0_{str(table['sha256'])[:20]}.table.json"
            self._sdk.run.files[path] = self._sdk.run.files.pop(old_path)
            table["path"] = path
        if self._sdk.mutate_table_sha:
            self._sdk.run.summary[adapter._TABLE_KEY]["sha256"] = _digest(14)
        if self._sdk.mutate_run_url_project:
            app_origin = os.environ["WANDB_APP_BASE_URL"].rstrip("/")
            self._sdk.run.url = (
                f"{app_origin}/{self._sdk.run.entity}/other-project/runs/"
                f"{self._sdk.run.id}"
            )
        return self._sdk.run

    def runs(
        self,
        path: str,
        *,
        filters: dict[str, Any],
        per_page: int,
    ) -> list[_FakeRun]:
        self._sdk.api_runs_queries.append((path, dict(filters), per_page))
        if self._sdk.run is None:
            return []
        return [self._sdk.run] if filters == {"id": self._sdk.run.id} else []

    def artifacts(
        self,
        *,
        type_name: str,
        name: str,
        per_page: int,
    ) -> list[_FakeArtifact]:
        self._sdk.api_artifacts_queries.append((type_name, name, per_page))
        if self._sdk.logged_artifact is None:
            return []
        copies = 2 if self._sdk.duplicate_artifact else 1
        return [self._sdk.logged_artifact] * copies

    def artifact(self, ref: str) -> _FakeArtifact:
        self._sdk.api_artifact_refs.append(ref)
        if self._sdk.logged_artifact is None:
            raise AssertionError("fake artifact was not logged")
        if self._sdk.mutate_artifact_bytes:
            self._sdk.logged_artifact.files["research-index.json"] = b"{}\n"
        if self._sdk.mutate_artifact_url_project:
            app_origin = os.environ["WANDB_APP_BASE_URL"].rstrip("/")
            artifact = self._sdk.logged_artifact
            artifact.url = (
                f"{app_origin}/{artifact.entity}/other-project/artifacts/"
                f"{artifact.type}/{artifact.base_name}/v0"
            )
        return self._sdk.logged_artifact


class _FakeWandb:
    __version__ = "9.9.9"

    def __init__(self) -> None:
        self.init_calls: list[dict[str, Any]] = []
        self.tables: list[_FakeTable] = []
        self.artifacts: list[_FakeArtifact] = []
        self.run: _FakeRun | None = None
        self.logged_artifact: _FakeArtifact | None = None
        self.api_run_paths: list[str] = []
        self.api_runs_queries: list[tuple[str, dict[str, Any], int]] = []
        self.api_artifact_refs: list[str] = []
        self.api_artifacts_queries: list[tuple[str, str, int]] = []
        self.download_calls: list[tuple[str, bool]] = []
        self.run_file_paths: list[str] = []
        self.table_download_calls: list[tuple[str, bool]] = []
        self.wait_calls = 0
        self.mutate_run_config = False
        self.mutate_table_summary = False
        self.mutate_table_cell = False
        self.mutate_table_path = False
        self.mutate_table_sha = False
        self.mutate_artifact_bytes = False
        self.mutate_run_url_project = False
        self.mutate_artifact_url_project = False
        self.duplicate_artifact = False
        self.raise_after_init = False
        self.raise_after_table_log = False
        self.raise_after_artifact_log = False
        self.raise_after_success_finish = False
        self.environment_at_init: dict[str, str | None] = {}
        self.environment_at_api: dict[str, str | None] = {}
        self.environments_at_api: list[dict[str, str | None]] = []
        self.init_entered: Event | None = None
        self.init_release: Event | None = None

    def init(self, **kwargs: Any) -> _FakeRun:
        self.environment_at_init = {
            name: os.environ.get(name) for name in adapter._SCOPED_ENV
        }
        if self.init_entered is not None:
            self.init_entered.set()
        if self.init_release is not None:
            assert self.init_release.wait(timeout=10)
        self.init_calls.append(dict(kwargs))
        if kwargs.get("resume") == "must" and self.run is not None:
            if self.run.id != str(kwargs["id"]):
                raise AssertionError("fake resume selected another Run")
        else:
            self.run = _FakeRun(self, kwargs)
        if self.raise_after_init:
            raise RuntimeError("injected initialization failure")
        return self.run

    def Table(self, **kwargs: Any) -> _FakeTable:  # noqa: N802
        table = _FakeTable(**kwargs)
        self.tables.append(table)
        return table

    def Artifact(self, name: str, **kwargs: Any) -> _FakeArtifact:  # noqa: N802
        artifact = _FakeArtifact(self, name, **kwargs)
        self.artifacts.append(artifact)
        return artifact

    def Api(self) -> _FakeApi:  # noqa: N802
        self.environment_at_api = {
            name: os.environ.get(name) for name in adapter._SCOPED_ENV
        }
        self.environments_at_api.append(dict(self.environment_at_api))
        return _FakeApi(self)


def _publisher(
    monkeypatch: pytest.MonkeyPatch,
    sdk: _FakeWandb,
    *,
    env: dict[str, str] | None = None,
):
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: sdk)
    return adapter.wandb_research_index_publisher_from_environment(
        env or _publication_env()
    )


def _publication_env() -> dict[str, str]:
    return {
        "WANDB_API_KEY": "publication-api-key",
        "WANDB_BASE_URL": "https://api.wandb.example",
        "WANDB_APP_BASE_URL": "https://app.wandb.example",
    }


def _target(
    env: dict[str, str] | None = None,
) -> ResearchIndexPublicationTargetV1:
    return adapter.wandb_research_index_target_from_environment(
        "wandb/community-studies",
        env or _publication_env(),
    )


def test_wandb_import_is_lazy_and_missing_extra_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    real_import = importlib.import_module

    def spy(name: str, package: str | None = None):
        calls.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", spy)
    importlib.reload(adapter)
    assert "wandb" not in calls

    def missing(name: str):
        assert name == "wandb"
        raise ModuleNotFoundError("No module named 'wandb'", name="wandb")

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    with pytest.raises(
        adapter.MissingWandbIndexExtraError,
        match=r"fugue\[wandb-index\]",
    ):
        adapter.wandb_research_index_publisher_from_environment({})


def test_publisher_writes_one_deterministic_safe_run_and_reads_it_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index()
    body = _index_bytes(index)
    sdk = _FakeWandb()
    previous = {
        name: f"original-{position}"
        for position, name in enumerate(adapter._SCOPED_ENV)
    }
    for name, value in previous.items():
        monkeypatch.setenv(name, value)
    publisher = _publisher(monkeypatch, sdk)

    target = _target()
    outcome = publisher(index, body, target)

    assert len(sdk.init_calls) == 1
    call = sdk.init_calls[0]
    assert call["entity"] == "wandb"
    assert call["project"] == "community-studies"
    assert call["job_type"] == WANDB_STUDY_INDEX_JOB_TYPE
    assert call["resume"] == "never"
    assert call["save_code"] is False
    assert call["config"] == {
        "fugue_research_id": index.research_id,
        "research_title": index.title,
        "research_objective": index.objective,
        "record_kind": WANDB_STUDY_INDEX_RECORD_KIND,
        "index_digest": index.index_digest,
        "index_file_sha256": __import__("hashlib").sha256(body).hexdigest(),
        "study_table_file_sha256": __import__("hashlib")
        .sha256(
            adapter._study_table_bytes(
                index,
                adapter._study_table_data(
                    index,
                    app_base_url=target.app_base_url,
                ),
            )
        )
        .hexdigest(),
        "study_count": 2,
        "total_rows": 12,
    }
    assert "wandb_research_id" not in call["config"]
    assert "wandb_study_id" not in call["config"]
    expected_environment = {
        "WANDB_ENTITY": "wandb",
        "WANDB_PROJECT": "community-studies",
        "WANDB_RUN_ID": call["id"],
        "WANDB_NAME": call["name"],
        "WANDB_JOB_TYPE": WANDB_STUDY_INDEX_JOB_TYPE,
        "WANDB_RUN_GROUP": index.research_id,
        "WANDB_API_KEY": "publication-api-key",
        "WANDB_BASE_URL": "https://api.wandb.example",
        "WANDB_APP_BASE_URL": "https://app.wandb.example",
    }
    assert sdk.environment_at_init == expected_environment
    assert sdk.environment_at_api == expected_environment
    assert {name: os.environ.get(name) for name in previous} == previous
    assert sdk.run is not None
    assert sdk.run.finish_calls == [0]
    assert len(sdk.run.logs) == 1
    logged, commit = sdk.run.logs[0]
    assert commit is True
    table = logged["studies"]
    assert table.columns == list(adapter._TABLE_COLUMNS)
    assert [row[0] for row in table.data] == ["study-a", "study-b"]
    assert all(len(row) == len(table.columns) for row in table.data)
    serialized_table = json.dumps(table.data, sort_keys=True)
    assert "weave:///" not in serialized_table
    assert "evidence_refs" not in serialized_table
    assert "private" not in serialized_table.lower()

    assert len(sdk.artifacts) == 1
    artifact = sdk.artifacts[0]
    assert len(artifact.base_name) <= 128
    assert artifact.base_name.endswith(f"-{index.index_digest[:12]}")
    assert artifact.type == WANDB_STUDY_INDEX_JOB_TYPE
    assert artifact.files["research-index.json"] == body
    assert set(artifact.files) == {"research-index.json", "studies-table.json"}
    assert set(artifact.file_policies.values()) == {"immutable"}
    assert artifact.metadata == {
        **call["config"],
        "artifact_file_sha256s": {
            name: __import__("hashlib").sha256(value).hexdigest()
            for name, value in sorted(artifact.files.items())
        },
    }
    public_payload = json.dumps(
        {"config": call["config"], "table": table.data, "metadata": artifact.metadata},
        sort_keys=True,
    )
    assert "publication-api-key" not in public_payload
    assert sdk.wait_calls == 1
    assert sdk.api_run_paths == [f"wandb/community-studies/{call['id']}"]
    assert sdk.api_runs_queries == [
        ("wandb/community-studies", {"id": call["id"]}, 2)
    ]
    assert len(sdk.api_artifacts_queries) == 1
    assert sdk.api_artifact_refs == [artifact.qualified_name]
    assert sdk.download_calls and sdk.download_calls[0][1] is True
    assert sdk.run_file_paths == [sdk.run.summary[adapter._TABLE_KEY]["path"]]
    assert len(sdk.table_download_calls) == 1
    assert sdk.table_download_calls[0][1] is True

    baseline_column = adapter._TABLE_COLUMNS.index(
        "baseline_candidate_assignments"
    )
    candidate_column = adapter._TABLE_COLUMNS.index(
        "candidate_candidate_assignments"
    )
    expected_candidate = index.studies[0].candidate_ids[0]
    assert json.loads(table.data[0][baseline_column]) == [
        {"candidate_id": expected_candidate, "harness": "claude-code"}
    ]
    assert json.loads(table.data[0][candidate_column]) == [
        {"candidate_id": expected_candidate, "harness": "claude-code"}
    ]

    assert outcome.target == target
    assert outcome.run_url == sdk.run.url
    assert outcome.artifact_url == artifact.url
    assert outcome.report_url is None
    assert outcome.report_status == "unavailable"
    assert outcome.publisher_revision == "v1+wandb-9.9.9"


@pytest.mark.parametrize("mismatch", ["run", "artifact", "table"])
def test_publisher_rejects_authoritative_readback_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    index = _index()
    sdk = _FakeWandb()
    sdk.mutate_run_config = mismatch == "run"
    sdk.mutate_table_summary = mismatch == "table"
    sdk.mutate_artifact_bytes = mismatch == "artifact"
    publisher = _publisher(monkeypatch, sdk)

    with pytest.raises(
        adapter.WandbResearchIndexPublicationError,
        match="W&B",
    ):
        publisher(index, _index_bytes(index), _target())


@pytest.mark.parametrize(
    "mismatch",
    ["cell", "path", "sha", "run_url", "artifact_url"],
)
def test_publisher_rejects_authoritative_table_or_destination_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    index = _index()
    sdk = _FakeWandb()
    sdk.mutate_table_cell = mismatch == "cell"
    sdk.mutate_table_path = mismatch == "path"
    sdk.mutate_table_sha = mismatch == "sha"
    sdk.mutate_run_url_project = mismatch == "run_url"
    sdk.mutate_artifact_url_project = mismatch == "artifact_url"
    publisher = _publisher(monkeypatch, sdk)

    with pytest.raises(
        adapter.WandbResearchIndexPublicationError,
        match="authoritative W&B|another W&B object",
    ):
        publisher(index, _index_bytes(index), _target())


def test_artifact_name_is_bounded_and_preserves_digest_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index("r" * 256)
    sdk = _FakeWandb()

    _publisher(monkeypatch, sdk)(index, _index_bytes(index), _target())

    assert len(sdk.artifacts) == 1
    name = sdk.artifacts[0].base_name
    assert len(name) == 128
    assert name.endswith(f"-{index.index_digest[:12]}")


def test_publisher_resumes_exact_crash_window_without_duplicate_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index()
    body = _index_bytes(index)
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)
    target = _target()

    first = publisher(index, body, target)
    initial_counts = (
        len(sdk.init_calls),
        len(sdk.tables),
        len(sdk.artifacts),
        len(sdk.run.logs) if sdk.run is not None else 0,
        sdk.wait_calls,
    )

    second = publisher(index, body, target)

    assert second == first
    assert initial_counts == (1, 1, 1, 1, 1)
    assert (
        len(sdk.init_calls),
        len(sdk.tables),
        len(sdk.artifacts),
        len(sdk.run.logs) if sdk.run is not None else 0,
        sdk.wait_calls,
    ) == initial_counts
    assert len(sdk.api_runs_queries) == 2
    assert len(sdk.api_artifacts_queries) == 2
    assert len(sdk.download_calls) == 2
    assert len(sdk.table_download_calls) == 2


def test_publisher_rejects_artifact_without_deterministic_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index()
    body = _index_bytes(index)
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)
    target = _target()
    publisher(index, body, target)
    initial_init_count = len(sdk.init_calls)

    sdk.run = None

    with pytest.raises(
        adapter.WandbResearchIndexPublicationError,
        match="artifact without its deterministic Run",
    ):
        publisher(index, body, target)
    assert len(sdk.init_calls) == initial_init_count


@pytest.mark.parametrize("phase", ["init", "table", "artifact", "finish"])
def test_publisher_recovers_verified_deterministic_crash_windows(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    index = _index()
    body = _index_bytes(index)
    sdk = _FakeWandb()
    sdk.raise_after_init = phase == "init"
    sdk.raise_after_table_log = phase == "table"
    sdk.raise_after_artifact_log = phase == "artifact"
    sdk.raise_after_success_finish = phase == "finish"
    publisher = _publisher(monkeypatch, sdk)
    target = _target()

    with pytest.raises(RuntimeError, match="injected"):
        publisher(index, body, target)

    sdk.raise_after_init = False
    sdk.raise_after_table_log = False
    sdk.raise_after_artifact_log = False
    sdk.raise_after_success_finish = False
    outcome = publisher(index, body, target)

    assert outcome.target == target
    assert sdk.run is not None
    assert len(sdk.run.logs) == 1
    if phase in {"init", "table"}:
        assert len(sdk.init_calls) == 2
        assert sdk.init_calls[1]["resume"] == "must"
        assert len(sdk.artifacts) == 1
    else:
        assert len(sdk.init_calls) == 1
        assert len(sdk.artifacts) == 1


def test_publisher_rejects_unverifiable_run_only_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index()
    body = _index_bytes(index)
    sdk = _FakeWandb()
    sdk.raise_after_table_log = True
    publisher = _publisher(monkeypatch, sdk)
    target = _target()

    with pytest.raises(RuntimeError, match="after table log"):
        publisher(index, body, target)
    assert sdk.run is not None
    sdk.raise_after_table_log = False
    sdk.run.summary[adapter._TABLE_KEY]["nrows"] = 999

    with pytest.raises(
        adapter.WandbResearchIndexPublicationError,
        match="Table summary changed shape",
    ):
        publisher(index, body, target)
    assert len(sdk.init_calls) == 1


def test_publisher_rejects_conflicting_artifact_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index()
    body = _index_bytes(index)
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)
    target = _target()
    publisher(index, body, target)
    sdk.duplicate_artifact = True

    with pytest.raises(
        adapter.WandbResearchIndexPublicationError,
        match="conflicting Research-index artifact revisions",
    ):
        publisher(index, body, target)


def test_concurrent_publishers_isolate_destination_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index()
    body = _index_bytes(index)
    first_sdk = _FakeWandb()
    second_sdk = _FakeWandb()
    first_entered = Event()
    first_release = Event()
    second_started = Event()
    first_sdk.init_entered = first_entered
    first_sdk.init_release = first_release
    first_env = {
        "WANDB_API_KEY": "first-key",
        "WANDB_BASE_URL": "https://api.first.example",
        "WANDB_APP_BASE_URL": "https://app.first.example",
    }
    second_env = {
        "WANDB_API_KEY": "second-key",
        "WANDB_BASE_URL": "https://api.second.example",
        "WANDB_APP_BASE_URL": "https://app.second.example",
    }
    first_target = adapter.wandb_research_index_target_from_environment(
        "wandb/first-index",
        first_env,
    )
    second_target = adapter.wandb_research_index_target_from_environment(
        "wandb/second-index",
        second_env,
    )
    previous = {
        name: f"original-{position}"
        for position, name in enumerate(adapter._SCOPED_ENV)
    }
    for name, value in previous.items():
        monkeypatch.setenv(name, value)

    def publish_second():
        second_started.set()
        return adapter._publish_with_wandb_sdk(
            wandb=second_sdk,
            index=index,
            index_bytes=body,
            target=second_target,
            publication_env=second_env,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            adapter._publish_with_wandb_sdk,
            wandb=first_sdk,
            index=index,
            index_bytes=body,
            target=first_target,
            publication_env=first_env,
        )
        assert first_entered.wait(timeout=10)
        second_future = pool.submit(publish_second)
        assert second_started.wait(timeout=10)
        assert second_sdk.environments_at_api == []
        first_release.set()
        first_outcome = first_future.result(timeout=10)
        second_outcome = second_future.result(timeout=10)

    assert first_outcome.target == first_target
    assert second_outcome.target == second_target
    assert first_outcome.run_url.startswith(first_target.app_base_url)
    assert second_outcome.run_url.startswith(second_target.app_base_url)
    assert first_sdk.environment_at_init["WANDB_API_KEY"] == "first-key"
    assert second_sdk.environment_at_init["WANDB_API_KEY"] == "second-key"
    assert all(
        observed["WANDB_BASE_URL"] == first_target.api_base_url
        for observed in first_sdk.environments_at_api
    )
    assert all(
        observed["WANDB_BASE_URL"] == second_target.api_base_url
        for observed in second_sdk.environments_at_api
    )
    assert {name: os.environ.get(name) for name in previous} == previous


def test_wandb_and_weave_publishers_share_the_process_destination_lock() -> None:
    assert adapter._PUBLICATION_LOCK is weave_support.EVIDENCE_ROUTING_LOCK


def test_publisher_restores_routing_environment_when_wandb_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index()
    sdk = _FakeWandb()
    sdk.raise_after_init = True
    publication_env = {
        "WANDB_API_KEY": "new-api-key",
        "WANDB_BASE_URL": "https://new-api.wandb.example",
        "WANDB_APP_BASE_URL": "https://new-app.wandb.example",
    }
    publisher = _publisher(monkeypatch, sdk, env=publication_env)
    previous = {
        name: f"original-{position}"
        for position, name in enumerate(adapter._SCOPED_ENV)
    }
    for name, value in previous.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match="injected initialization failure"):
        publisher(index, _index_bytes(index), _target(publication_env))

    assert sdk.environment_at_init["WANDB_API_KEY"] == "new-api-key"
    assert (
        sdk.environment_at_init["WANDB_BASE_URL"]
        == "https://new-api.wandb.example"
    )
    assert sdk.environment_at_init["WANDB_ENTITY"] == "wandb"
    assert {name: os.environ.get(name) for name in previous} == previous
