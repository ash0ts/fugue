from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, urlsplit

from fugue.bench.candidates import stable_digest
from fugue.bench.research_index import (
    ResearchIndexPublicationOutcomeV1,
    ResearchIndexPublicationTargetV1,
    ResearchIndexPublisher,
    ResearchIndexV1,
    ResearchStudyIndexEntryV1,
)
from fugue.weave_support import EVIDENCE_ROUTING_LOCK

WANDB_STUDY_INDEX_JOB_TYPE = "study-index"
WANDB_STUDY_INDEX_RECORD_KIND = "research_index_v1"
WANDB_STUDY_INDEX_PUBLISHER_ID = "fugue-wandb-research-index"
DEFAULT_WANDB_API_BASE_URL = "https://api.wandb.ai"
DEFAULT_WANDB_APP_BASE_URL = "https://wandb.ai"

_PROJECT_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARTIFACT_PART = re.compile(r"[^A-Za-z0-9_.-]+")
_ROUTING_ENV = (
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WANDB_RUN_ID",
    "WANDB_NAME",
    "WANDB_JOB_TYPE",
    "WANDB_RUN_GROUP",
)
_AUTH_ENV = (
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
    "WANDB_APP_BASE_URL",
)
_SCOPED_ENV = (*_ROUTING_ENV, *_AUTH_ENV)
_INDEX_FILENAME = "research-index.json"
_TABLE_FILENAME = "studies-table.json"
_TABLE_RECORD_KIND = "research_index_table_v1"
_TABLE_KEY = "studies"
_TABLE_COLUMNS = (
    "study_id",
    "comparison_id",
    "project",
    "behavioral_status",
    "behavioral_recommendation",
    "decision_status",
    "decision_recommendation",
    "task_validity_status",
    "rows",
    "evidence_integrity_grade",
    "evidence_backend",
    "local_chain_integrity",
    "published_chain_integrity",
    "evidence_object_count",
    "evidence_project_url",
    "primary_evidence_url",
    "result_digest",
    "qualification_digest",
    "candidate_count",
    "baseline_candidate_assignments",
    "candidate_candidate_assignments",
)
_MAX_ARTIFACT_NAME_LENGTH = 128
_TABLE_MEDIA_PATH = re.compile(
    r"^media/table/studies_[0-9]+_([0-9a-f]{20})\.table\.json$"
)
_PUBLICATION_LOCK = EVIDENCE_ROUTING_LOCK


class MissingWandbIndexExtraError(RuntimeError):
    """The optional W&B SDK needed for Research-index publication is absent."""


class WandbResearchIndexPublicationError(RuntimeError):
    """W&B did not authoritatively preserve a Research index publication."""


def wandb_research_index_publisher_from_environment(
    env: Mapping[str, str],
) -> ResearchIndexPublisher:
    """Build a publisher using explicitly supplied W&B connection settings."""

    try:
        wandb = importlib.import_module("wandb")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MissingWandbIndexExtraError(
            "W&B Research-index publication requires the optional W&B SDK. "
            'Install it with `python -m pip install "fugue[wandb-index]"`.'
        ) from exc
    publication_env = {
        name: str(env[name]) for name in _AUTH_ENV if name in env
    }

    def publish(
        index: ResearchIndexV1,
        index_bytes: bytes,
        target: ResearchIndexPublicationTargetV1,
    ) -> ResearchIndexPublicationOutcomeV1:
        expected = wandb_research_index_target_from_environment(target.project, env)
        if target != expected:
            raise WandbResearchIndexPublicationError(
                "W&B Research-index target disagrees with the publisher environment"
            )
        return _publish_with_wandb_sdk(
            wandb=wandb,
            index=index,
            index_bytes=index_bytes,
            target=target,
            publication_env=publication_env,
        )

    return cast(ResearchIndexPublisher, publish)


def wandb_research_index_target_from_environment(
    project: str,
    env: Mapping[str, str],
) -> ResearchIndexPublicationTargetV1:
    """Resolve and bind the exact W&B control and application destinations."""

    return ResearchIndexPublicationTargetV1(
        project=project,
        api_base_url=_normalized_origin(
            env.get("WANDB_BASE_URL") or DEFAULT_WANDB_API_BASE_URL,
            "WANDB_BASE_URL",
        ),
        app_base_url=_normalized_origin(
            env.get("WANDB_APP_BASE_URL") or DEFAULT_WANDB_APP_BASE_URL,
            "WANDB_APP_BASE_URL",
        ),
    )


def _publish_with_wandb_sdk(
    *,
    wandb: Any,
    index: ResearchIndexV1,
    index_bytes: bytes,
    target: ResearchIndexPublicationTargetV1,
    publication_env: Mapping[str, str],
) -> ResearchIndexPublicationOutcomeV1:
    entity, project_id = _project_parts(target.project)
    _verify_index_bytes(index, index_bytes)

    run_id = stable_digest(
        {
            "schema_version": 1,
            "record_kind": WANDB_STUDY_INDEX_RECORD_KIND,
            "target": target.to_dict(),
            "index_digest": index.index_digest,
        }
    )[:32]
    run_name = f"study-index-{_safe_name(index.research_id)}-{index.index_digest[:12]}"
    artifact_name = _bounded_name(
        prefix="fugue-research-index-",
        value=index.research_id,
        digest=index.index_digest,
        max_length=_MAX_ARTIFACT_NAME_LENGTH,
    )
    index_file_sha256 = hashlib.sha256(index_bytes).hexdigest()
    table_data = _study_table_data(index, app_base_url=target.app_base_url)
    table_bytes = _study_table_bytes(index, table_data)
    table_file_sha256 = hashlib.sha256(table_bytes).hexdigest()
    artifact_files = _artifact_files(
        index,
        index_bytes=index_bytes,
        table_bytes=table_bytes,
    )
    config = _run_config(
        index,
        index_file_sha256=index_file_sha256,
        table_file_sha256=table_file_sha256,
    )
    artifact_metadata = _artifact_metadata(
        index,
        index_file_sha256=index_file_sha256,
        table_file_sha256=table_file_sha256,
        artifact_files=artifact_files,
    )
    routing = {
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project_id,
        "WANDB_RUN_ID": run_id,
        "WANDB_NAME": run_name,
        "WANDB_JOB_TYPE": WANDB_STUDY_INDEX_JOB_TYPE,
        "WANDB_RUN_GROUP": index.research_id,
    }

    scoped_environment = {
        **publication_env,
        "WANDB_BASE_URL": target.api_base_url,
        "WANDB_APP_BASE_URL": target.app_base_url,
        **routing,
    }
    with _PUBLICATION_LOCK, _scoped_environment(scoped_environment):
        with tempfile.TemporaryDirectory(
            prefix="fugue-wandb-research-index-"
        ) as raw_tmp:
            temp_root = Path(raw_tmp)
            _write_artifact_files(temp_root, artifact_files)
            api = wandb.Api()
            existing_run = _find_run(
                api,
                entity=entity,
                project_id=project_id,
                run_id=run_id,
            )
            existing_artifact = _find_artifact(
                api,
                entity=entity,
                project_id=project_id,
                artifact_name=artifact_name,
            )
            if existing_run is None and existing_artifact is not None:
                raise WandbResearchIndexPublicationError(
                    "existing W&B Research-index publication has an artifact "
                    "without its deterministic Run"
                )
            if existing_run is not None and existing_artifact is not None:
                return _read_back_publication(
                    api=api,
                    wandb=wandb,
                    entity=entity,
                    project_id=project_id,
                    target=target,
                    run_id=run_id,
                    artifact_name=artifact_name,
                    config=config,
                    artifact_ref=_artifact_ref(
                        existing_artifact,
                        project=target.project,
                        artifact_name=artifact_name,
                    ),
                    artifact_metadata=artifact_metadata,
                    artifact_files=artifact_files,
                    expected_table_data=table_data,
                    temp_root=temp_root,
                )
            resume_existing = existing_run is not None
            existing_table = False
            if existing_run is not None:
                authoritative_run = api.run(f"{entity}/{project_id}/{run_id}")
                existing_table = _verify_run(
                    run=authoritative_run,
                    entity=entity,
                    project_id=project_id,
                    target=target,
                    run_id=run_id,
                    config=config,
                    expected_table_data=table_data,
                    temp_root=temp_root,
                    allow_missing_table=True,
                )
            run: Any | None = None
            finished = False
            logged_artifact: Any | None = None
            try:
                run = wandb.init(
                    entity=entity,
                    project=project_id,
                    dir=str(temp_root),
                    id=run_id,
                    name=run_name,
                    group=index.research_id,
                    job_type=WANDB_STUDY_INDEX_JOB_TYPE,
                    config=config,
                    mode="online",
                    reinit="create_new",
                    resume="must" if resume_existing else "never",
                    save_code=False,
                )
                if run is None:
                    raise WandbResearchIndexPublicationError(
                        "wandb.init did not return a Research-index Run"
                    )
                _verify_active_run(
                    run,
                    entity=entity,
                    project_id=project_id,
                    run_id=run_id,
                    config=config,
                )
                if not existing_table:
                    table = wandb.Table(
                        columns=list(_TABLE_COLUMNS),
                        data=[list(row) for row in table_data],
                        log_mode="IMMUTABLE",
                    )
                    run.log({_TABLE_KEY: table}, commit=True)
                artifact = wandb.Artifact(
                    artifact_name,
                    type=WANDB_STUDY_INDEX_JOB_TYPE,
                    metadata=artifact_metadata,
                )
                for relative, body in artifact_files.items():
                    source = temp_root / relative
                    if source.read_bytes() != body:
                        raise WandbResearchIndexPublicationError(
                            "Research-index artifact source changed before upload"
                        )
                    artifact.add_file(
                        str(source),
                        name=relative,
                        policy="immutable",
                    )
                logged_artifact = run.log_artifact(artifact)
                if logged_artifact is None:
                    raise WandbResearchIndexPublicationError(
                        "W&B did not return the logged Research-index artifact"
                    )
                wait = getattr(logged_artifact, "wait", None)
                if not callable(wait):
                    raise WandbResearchIndexPublicationError(
                        "the installed W&B SDK cannot wait for artifact finalization"
                    )
                logged_artifact = wait()
                if logged_artifact is None:
                    raise WandbResearchIndexPublicationError(
                        "W&B artifact finalization returned no artifact"
                    )
                run.finish(exit_code=0)
                finished = True
            finally:
                if run is not None and not finished:
                    try:
                        run.finish(exit_code=1)
                    except Exception:
                        pass

            if logged_artifact is None:
                raise WandbResearchIndexPublicationError(
                    "Research-index artifact was not finalized"
                )
            artifact_ref = _artifact_ref(
                logged_artifact,
                project=target.project,
                artifact_name=artifact_name,
            )
            return _read_back_publication(
                api=wandb.Api(),
                wandb=wandb,
                entity=entity,
                project_id=project_id,
                target=target,
                run_id=run_id,
                artifact_name=artifact_name,
                config=config,
                artifact_ref=artifact_ref,
                artifact_metadata=artifact_metadata,
                artifact_files=artifact_files,
                expected_table_data=table_data,
                temp_root=temp_root,
            )


def _read_back_publication(
    *,
    api: Any,
    wandb: Any,
    entity: str,
    project_id: str,
    target: ResearchIndexPublicationTargetV1,
    run_id: str,
    artifact_name: str,
    config: Mapping[str, Any],
    artifact_ref: str,
    artifact_metadata: Mapping[str, Any],
    artifact_files: Mapping[str, bytes],
    expected_table_data: tuple[tuple[Any, ...], ...],
    temp_root: Path,
) -> ResearchIndexPublicationOutcomeV1:
    run = api.run(f"{entity}/{project_id}/{run_id}")
    _verify_run(
        run=run,
        entity=entity,
        project_id=project_id,
        target=target,
        run_id=run_id,
        config=config,
        expected_table_data=expected_table_data,
        temp_root=temp_root,
        allow_missing_table=False,
    )
    run_url = _destination_url(
        getattr(run, "url", None),
        "W&B Research-index Run URL",
        target=target,
        entity=entity,
        project_id=project_id,
        resource_path=("runs", run_id),
    )

    artifact = api.artifact(artifact_ref)
    if (
        str(getattr(artifact, "entity", "")) != entity
        or str(getattr(artifact, "project", "")) != project_id
        or str(getattr(artifact, "type", "")) != WANDB_STUDY_INDEX_JOB_TYPE
    ):
        raise WandbResearchIndexPublicationError(
            "authoritative W&B artifact readback returned another project"
        )
    observed_ref = _artifact_ref(
        artifact,
        project=target.project,
        artifact_name=artifact_name,
    )
    if observed_ref != artifact_ref:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B artifact readback changed the immutable revision"
        )
    observed_metadata = dict(getattr(artifact, "metadata", {}) or {})
    if observed_metadata != dict(artifact_metadata):
        raise WandbResearchIndexPublicationError(
            "authoritative W&B artifact readback changed the Research-index metadata"
        )
    download = getattr(artifact, "download", None)
    if not callable(download):
        raise WandbResearchIndexPublicationError(
            "the installed W&B SDK cannot read back artifact files"
        )
    download_root = temp_root / "readback"
    downloaded = Path(download(root=str(download_root), skip_cache=True))
    observed_files = {
        path.relative_to(downloaded).as_posix(): path.read_bytes()
        for path in downloaded.rglob("*")
        if path.is_file()
    }
    if observed_files != dict(artifact_files):
        raise WandbResearchIndexPublicationError(
            "authoritative W&B artifact readback changed the bound publication files"
        )
    _verify_table_file(
        observed_files[_TABLE_FILENAME],
        expected_table_data=expected_table_data,
    )
    artifact_url = _destination_url(
        getattr(artifact, "url", None),
        "W&B Research-index artifact URL",
        target=target,
        entity=entity,
        project_id=project_id,
        resource_path=(
            "artifacts",
            WANDB_STUDY_INDEX_JOB_TYPE,
            artifact_name,
        ),
    )
    version = str(getattr(wandb, "__version__", "unknown") or "unknown")
    return ResearchIndexPublicationOutcomeV1(
        target=target,
        run_url=run_url,
        artifact_url=artifact_url,
        report_url=None,
        report_status="unavailable",
        publisher_id=WANDB_STUDY_INDEX_PUBLISHER_ID,
        publisher_revision=f"v1+wandb-{version}",
    )


def _run_config(
    index: ResearchIndexV1,
    *,
    index_file_sha256: str,
    table_file_sha256: str,
) -> dict[str, Any]:
    return {
        "fugue_research_id": index.research_id,
        "research_title": index.title,
        "research_objective": index.objective,
        "record_kind": WANDB_STUDY_INDEX_RECORD_KIND,
        "index_digest": index.index_digest,
        "index_file_sha256": index_file_sha256,
        "study_table_file_sha256": table_file_sha256,
        "study_count": index.study_count,
        "total_rows": index.total_rows,
    }


def _artifact_metadata(
    index: ResearchIndexV1,
    *,
    index_file_sha256: str,
    table_file_sha256: str,
    artifact_files: Mapping[str, bytes],
) -> dict[str, Any]:
    return {
        "fugue_research_id": index.research_id,
        "research_title": index.title,
        "research_objective": index.objective,
        "record_kind": WANDB_STUDY_INDEX_RECORD_KIND,
        "index_digest": index.index_digest,
        "index_file_sha256": index_file_sha256,
        "study_table_file_sha256": table_file_sha256,
        "artifact_file_sha256s": {
            name: hashlib.sha256(body).hexdigest()
            for name, body in sorted(artifact_files.items())
        },
        "study_count": index.study_count,
        "total_rows": index.total_rows,
    }


def _study_table_bytes(
    index: ResearchIndexV1,
    table_data: tuple[tuple[Any, ...], ...],
) -> bytes:
    payload = {
        "schema_version": 1,
        "record_kind": _TABLE_RECORD_KIND,
        "index_digest": index.index_digest,
        "columns": list(_TABLE_COLUMNS),
        "data": [list(row) for row in table_data],
        "log_mode": "IMMUTABLE",
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _artifact_files(
    index: ResearchIndexV1,
    *,
    index_bytes: bytes,
    table_bytes: bytes,
) -> dict[str, bytes]:
    # The canonical Research index is the reviewed, sanitized sharing boundary.
    # It embeds each exact result and scoped publication receipt as SHA-bound
    # JSON text. The table file is only a derivative browsing surface.
    del index
    return {
        _INDEX_FILENAME: index_bytes,
        _TABLE_FILENAME: table_bytes,
    }


def _write_artifact_files(root: Path, files: Mapping[str, bytes]) -> None:
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def _find_run(
    api: Any,
    *,
    entity: str,
    project_id: str,
    run_id: str,
) -> Any | None:
    runs = getattr(api, "runs", None)
    if not callable(runs):
        raise WandbResearchIndexPublicationError(
            "the installed W&B SDK cannot preflight deterministic Runs"
        )
    observed = tuple(
        runs(
            f"{entity}/{project_id}",
            filters={"name": run_id},
            per_page=2,
        )
    )
    matching = tuple(item for item in observed if str(getattr(item, "id", "")) == run_id)
    if len(matching) > 1:
        raise WandbResearchIndexPublicationError(
            "W&B returned duplicate deterministic Research-index Runs"
        )
    if observed and not matching:
        raise WandbResearchIndexPublicationError(
            "W&B Run preflight ignored the deterministic identity filter"
        )
    return matching[0] if matching else None


def _find_artifact(
    api: Any,
    *,
    entity: str,
    project_id: str,
    artifact_name: str,
) -> Any | None:
    artifact_ref = f"{entity}/{project_id}/{artifact_name}:latest"
    artifact_exists = getattr(api, "artifact_exists", None)
    artifact = getattr(api, "artifact", None)
    if callable(artifact_exists) and callable(artifact):
        if not artifact_exists(
            artifact_ref,
            type=WANDB_STUDY_INDEX_JOB_TYPE,
        ):
            return None
        observed = artifact(
            artifact_ref,
            type=WANDB_STUDY_INDEX_JOB_TYPE,
        )
        if observed is None:
            raise WandbResearchIndexPublicationError(
                "W&B reported that the Research-index artifact exists but "
                "did not return it"
            )
        return observed

    artifacts = getattr(api, "artifacts", None)
    if not callable(artifacts):
        raise WandbResearchIndexPublicationError(
            "the installed W&B SDK cannot preflight immutable artifacts"
        )
    observed = tuple(
        artifacts(
            type_name=WANDB_STUDY_INDEX_JOB_TYPE,
            name=f"{entity}/{project_id}/{artifact_name}",
            per_page=2,
        )
    )
    if len(observed) > 1:
        raise WandbResearchIndexPublicationError(
            "W&B returned conflicting Research-index artifact revisions"
        )
    return observed[0] if observed else None


def _verify_active_run(
    run: Any,
    *,
    entity: str,
    project_id: str,
    run_id: str,
    config: Mapping[str, Any],
) -> None:
    if (
        str(getattr(run, "entity", "")) != entity
        or str(getattr(run, "project", "")) != project_id
        or str(getattr(run, "id", "")) != run_id
    ):
        raise WandbResearchIndexPublicationError(
            "active W&B Run disagrees with the deterministic destination"
        )
    if str(getattr(run, "job_type", "")) != WANDB_STUDY_INDEX_JOB_TYPE:
        raise WandbResearchIndexPublicationError(
            "active W&B Run changed the Research-index job type"
        )
    observed_config = {
        str(key): value
        for key, value in dict(getattr(run, "config", {}) or {}).items()
        if not str(key).startswith("_")
    }
    if observed_config != dict(config):
        raise WandbResearchIndexPublicationError(
            "active W&B Run changed the Research-index config"
        )


def _verify_run(
    *,
    run: Any,
    entity: str,
    project_id: str,
    target: ResearchIndexPublicationTargetV1,
    run_id: str,
    config: Mapping[str, Any],
    expected_table_data: tuple[tuple[Any, ...], ...],
    temp_root: Path,
    allow_missing_table: bool,
) -> bool:
    _verify_active_run(
        run,
        entity=entity,
        project_id=project_id,
        run_id=run_id,
        config=config,
    )
    _destination_url(
        getattr(run, "url", None),
        "W&B Research-index Run URL",
        target=target,
        entity=entity,
        project_id=project_id,
        resource_path=("runs", run_id),
    )
    raw_summary = getattr(run, "summary", {})
    try:
        summary = dict(raw_summary or {})
    except (TypeError, ValueError) as exc:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Run has no readable summary"
        ) from exc
    if _TABLE_KEY not in summary:
        if allow_missing_table:
            return False
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Run is missing the Research-index Table summary"
        )
    table_path, table_sha256, table_size = _verify_table_summary(
        summary,
        rows=len(expected_table_data),
    )
    _verify_run_table_media(
        run,
        table_path=table_path,
        table_sha256=table_sha256,
        table_size=table_size,
        expected_table_data=expected_table_data,
        temp_root=temp_root,
    )
    return True


def _verify_table_summary(
    raw_summary: Any,
    *,
    rows: int,
) -> tuple[str, str, int | None]:
    try:
        summary = dict(raw_summary or {})
    except (TypeError, ValueError) as exc:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Run has no readable summary"
        ) from exc
    raw_table = summary.get(_TABLE_KEY)
    if raw_table is None:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Run is missing the Research-index Table summary"
        )
    try:
        table = dict(raw_table)
    except (TypeError, ValueError) as exc:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Run has an unreadable Research-index Table summary"
        ) from exc
    if (
        table.get("_type") != "table-file"
        or table.get("ncols") != len(_TABLE_COLUMNS)
        or table.get("nrows") != rows
        or table.get("log_mode") != "IMMUTABLE"
    ):
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table summary changed shape"
        )
    table_path = str(table.get("path", ""))
    table_sha = str(table.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", table_sha):
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table has no content digest"
        )
    path_match = _TABLE_MEDIA_PATH.fullmatch(table_path)
    if path_match is None or path_match.group(1) != table_sha[:20]:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table path disagrees with its digest"
        )
    raw_size = table.get("size")
    if raw_size is not None and (
        not isinstance(raw_size, int)
        or isinstance(raw_size, bool)
        or raw_size < 1
    ):
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table has an invalid size"
        )
    return table_path, table_sha, raw_size


def _verify_run_table_media(
    run: Any,
    *,
    table_path: str,
    table_sha256: str,
    table_size: int | None,
    expected_table_data: tuple[tuple[Any, ...], ...],
    temp_root: Path,
) -> None:
    file_method = getattr(run, "file", None)
    if not callable(file_method):
        raise WandbResearchIndexPublicationError(
            "the installed W&B SDK cannot read the authoritative Run Table"
        )
    remote_file = file_method(table_path)
    if str(getattr(remote_file, "name", "")) != table_path:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table returned another path"
        )
    download = getattr(remote_file, "download", None)
    if not callable(download):
        raise WandbResearchIndexPublicationError(
            "the installed W&B SDK cannot download the authoritative Run Table"
        )
    download_root = Path(
        tempfile.mkdtemp(prefix="run-table-readback-", dir=temp_root)
    )
    downloaded = download(root=str(download_root), replace=True)
    try:
        raw_path = getattr(downloaded, "name", downloaded)
        downloaded_path = Path(str(raw_path))
        try:
            downloaded_path.resolve().relative_to(download_root.resolve())
        except ValueError as exc:
            raise WandbResearchIndexPublicationError(
                "authoritative W&B Run Table download escaped the readback directory"
            ) from exc
        body = downloaded_path.read_bytes()
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, WandbResearchIndexPublicationError):
            raise
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Run Table could not be read"
        ) from exc
    finally:
        close = getattr(downloaded, "close", None)
        if callable(close):
            close()
    if table_size is not None and len(body) != table_size:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table changed size"
        )
    if hashlib.sha256(body).hexdigest() != table_sha256:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table changed its reported content"
        )
    try:
        raw = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table is not valid JSON"
        ) from exc
    expected = {
        "columns": list(_TABLE_COLUMNS),
        "data": [list(row) for row in expected_table_data],
    }
    if raw != expected:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table changed columns or data"
        )


def _verify_table_file(
    body: bytes,
    *,
    expected_table_data: tuple[tuple[Any, ...], ...],
) -> None:
    try:
        raw = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table artifact is not valid JSON"
        ) from exc
    expected = {
        "schema_version": 1,
        "record_kind": _TABLE_RECORD_KIND,
        "columns": list(_TABLE_COLUMNS),
        "data": [list(row) for row in expected_table_data],
        "log_mode": "IMMUTABLE",
    }
    observed = dict(raw) if isinstance(raw, Mapping) else {}
    observed.pop("index_digest", None)
    if observed != expected:
        raise WandbResearchIndexPublicationError(
            "authoritative W&B Research-index Table artifact changed content"
        )


def _study_table_data(
    index: ResearchIndexV1,
    *,
    app_base_url: str,
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            study.study_id,
            study.comparison_id,
            study.project,
            study.behavioral_status,
            study.behavioral_recommendation,
            study.decision_status,
            study.decision_recommendation,
            study.task_validity_status,
            study.rows,
            study.evidence_integrity_grade,
            study.evidence_backend,
            study.local_chain_integrity,
            study.published_chain_integrity,
            len(study.evidence_refs),
            f"{app_base_url}/{study.project}/weave",
            _primary_evidence_url(study, app_base_url=app_base_url),
            study.result_digest,
            study.qualification_digest,
            len(study.candidate_ids),
            _candidate_assignments(study, role="baseline"),
            _candidate_assignments(study, role="candidate"),
        )
        for study in index.studies
    )


def _candidate_assignments(
    study: ResearchStudyIndexEntryV1,
    *,
    role: str,
) -> str:
    assignments = [
        {
            "harness": item.harness,
            "candidate_id": item.candidate_id,
        }
        for item in study.candidate_assignments
        if item.role == role
    ]
    return json.dumps(assignments, sort_keys=True, separators=(",", ":"))


def _primary_evidence_url(
    study: ResearchStudyIndexEntryV1,
    *,
    app_base_url: str,
) -> str:
    ref = next(
        item.ref
        for item in study.evidence_refs
        if item.kind == "prediction_and_score"
    )
    call_id = ref.rsplit("/", 1)[-1]
    return (
        f"{app_base_url}/{quote(study.project, safe='/')}/weave/calls/"
        f"{quote(call_id, safe='')}"
    )


def _normalized_origin(raw: str, label: str) -> str:
    value = str(raw).rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise WandbResearchIndexPublicationError(
            f"{label} must be a safe HTTPS origin"
        )
    return value


def _verify_index_bytes(index: ResearchIndexV1, index_bytes: bytes) -> None:
    expected = (
        json.dumps(index.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")
    if index_bytes != expected:
        raise WandbResearchIndexPublicationError(
            "Research-index bytes do not match the canonical ResearchIndexV1"
        )


def _project_parts(project: str) -> tuple[str, str]:
    parts = str(project).split("/")
    if len(parts) != 2 or any(not _PROJECT_PART.fullmatch(part) for part in parts):
        raise WandbResearchIndexPublicationError(
            "W&B Research-index project must be ENTITY/PROJECT"
        )
    return parts[0], parts[1]


def _safe_name(value: str) -> str:
    rendered = _ARTIFACT_PART.sub("-", value).strip("-._")
    if not rendered:
        raise WandbResearchIndexPublicationError(
            "Research id cannot produce a safe W&B object name"
        )
    return rendered[:120]


def _bounded_name(
    *,
    prefix: str,
    value: str,
    digest: str,
    max_length: int,
) -> str:
    suffix = f"-{digest[:12]}"
    available = max_length - len(prefix) - len(suffix)
    if available < 1:
        raise WandbResearchIndexPublicationError(
            "W&B object-name limit cannot preserve the digest suffix"
        )
    human = _safe_name(value)[:available].rstrip("-._")
    if not human:
        human = "index"[:available]
    rendered = f"{prefix}{human}{suffix}"
    if len(rendered) > max_length or not rendered.endswith(suffix):
        raise WandbResearchIndexPublicationError(
            "W&B object name is not safely digest-qualified"
        )
    return rendered


def _artifact_ref(
    artifact: Any,
    *,
    project: str,
    artifact_name: str,
) -> str:
    qualified_name = str(getattr(artifact, "qualified_name", "") or "")
    if not qualified_name:
        name = str(getattr(artifact, "name", "") or "")
        version = str(getattr(artifact, "version", "") or "")
        if name and ":" in name:
            qualified_name = name
        elif name and version:
            qualified_name = f"{project}/{name}:{version}"
    if not qualified_name:
        raise WandbResearchIndexPublicationError(
            "finalized W&B artifact has no immutable qualified name"
        )
    if qualified_name.count("/") == 0:
        qualified_name = f"{project}/{qualified_name}"
    if qualified_name.count("/") != 2 or ":" not in qualified_name.rsplit("/", 1)[-1]:
        raise WandbResearchIndexPublicationError(
            "finalized W&B artifact name is not revision-qualified"
        )
    entity, project_id = _project_parts(project)
    qualified_parts = qualified_name.split("/")
    collection, revision = qualified_parts[2].rsplit(":", 1)
    if (
        qualified_parts[:2] != [entity, project_id]
        or collection != artifact_name
        or not revision
    ):
        raise WandbResearchIndexPublicationError(
            "finalized W&B artifact name disagrees with the deterministic destination"
        )
    return qualified_name


def _https_url(raw: Any, label: str) -> str:
    value = str(raw or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise WandbResearchIndexPublicationError(f"{label} is not a safe HTTPS URL")
    return value


def _destination_url(
    raw: Any,
    label: str,
    *,
    target: ResearchIndexPublicationTargetV1,
    entity: str,
    project_id: str,
    resource_path: tuple[str, ...],
) -> str:
    value = _https_url(raw, label)
    parsed = urlsplit(value)
    base = urlsplit(target.app_base_url)
    if (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
        raise WandbResearchIndexPublicationError(
            f"{label} returned another W&B application origin"
        )
    path_parts = [part for part in parsed.path.split("/") if part]
    required = [entity, project_id, *resource_path]
    if path_parts[: len(required)] != required:
        raise WandbResearchIndexPublicationError(
            f"{label} returned another W&B object path"
        )
    return value


@contextmanager
def _scoped_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in _SCOPED_ENV}
    try:
        for name in _SCOPED_ENV:
            if name in values:
                os.environ[name] = str(values[name])
            else:
                os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
