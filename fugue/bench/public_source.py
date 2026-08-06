from __future__ import annotations

import importlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from fugue.bench.candidates import stable_digest
from fugue.model_plane import (
    EvidenceDestinationV1,
    evidence_destination_from_dict,
)

_SUMMARY_KEY = "fugue.public_task_source_publication_v1"
_ARTIFACT_TYPE = "fugue-public-task-source"


class PublicTaskSourceError(ValueError):
    """A strict public-task source publication or resolution failure."""


@dataclass(frozen=True)
class PublicTaskSourceManifestV1:
    comparison_id: str
    destination: EvidenceDestinationV1
    source_lock: dict[str, Any]
    public_cases: tuple[dict[str, Any], ...]
    schema_version: int = 1
    kind: str = "public_task_source_manifest"
    content_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "public_task_source_manifest":
            raise PublicTaskSourceError("unsupported public source manifest schema")
        _identifier(self.comparison_id, "comparison id")
        source_lock = _source_lock(self.source_lock)
        if not self.public_cases:
            raise PublicTaskSourceError("public source manifest requires cases")
        task_ids: list[str] = []
        for raw in self.public_cases:
            case = _mapping(raw, "public source case")
            task_id = str(case.get("id") or "")
            _identifier(task_id, "public source task id")
            task_ids.append(task_id)
            if _contains_private_key(case):
                raise PublicTaskSourceError(
                    "public source manifest cannot contain private evaluation data"
                )
        if task_ids != sorted(set(task_ids)):
            raise PublicTaskSourceError(
                "public source task identities must be sorted and unique"
            )
        computed = stable_digest(self.unsigned_dict())
        if self.content_digest and self.content_digest != computed:
            raise PublicTaskSourceError("public source content digest does not match")
        if not self.content_digest:
            object.__setattr__(self, "content_digest", computed)
        object.__setattr__(self, "source_lock", source_lock)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "comparison_id": self.comparison_id,
            "destination": self.destination.to_dict(),
            "source_lock": self.source_lock,
            "public_cases": [dict(item) for item in self.public_cases],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "content_digest": self.content_digest}


@dataclass(frozen=True)
class PublicTaskSourceObjectV1:
    artifact_name: str
    artifact_version: str
    artifact_digest: str
    qualified_name: str
    artifact_ref: str
    artifact_url: str

    def __post_init__(self) -> None:
        for label, value in (
            ("artifact name", self.artifact_name),
            ("artifact version", self.artifact_version),
            ("artifact digest", self.artifact_digest),
            ("qualified artifact name", self.qualified_name),
            ("artifact reference", self.artifact_ref),
            ("artifact URL", self.artifact_url),
        ):
            if not value or len(value) > 2_000:
                raise PublicTaskSourceError(f"{label} is required")
        if not self.artifact_ref.startswith("wandb-artifact://"):
            raise PublicTaskSourceError("public source artifact ref is not immutable")
        if not re.fullmatch(r"v[0-9]+", self.artifact_version):
            raise PublicTaskSourceError("public source artifact version is not immutable")
        if self.qualified_name != (
            f"{self.artifact_name}:{self.artifact_version}"
        ) and not self.qualified_name.endswith(
            f"/{self.artifact_name}:{self.artifact_version}"
        ):
            raise PublicTaskSourceError("public source qualified artifact disagrees")
        if self.artifact_ref != f"wandb-artifact://{self.qualified_name}":
            raise PublicTaskSourceError("public source artifact ref disagrees")
        parsed_url = urlsplit(self.artifact_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise PublicTaskSourceError("public source artifact URL is invalid")

    @property
    def identity_digest(self) -> str:
        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_name": self.artifact_name,
            "artifact_version": self.artifact_version,
            "artifact_digest": self.artifact_digest,
            "qualified_name": self.qualified_name,
            "artifact_ref": self.artifact_ref,
            "artifact_url": self.artifact_url,
        }


@dataclass(frozen=True)
class PublicTaskSourcePublicationReceiptV1:
    comparison_id: str
    destination: EvidenceDestinationV1
    source_lock_digest: str
    content_digest: str
    publication_id: str
    publication_run_id: str
    source_object: PublicTaskSourceObjectV1
    schema_version: int = 1
    kind: str = "public_task_source_publication"
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "public_task_source_publication":
            raise PublicTaskSourceError("unsupported source publication receipt")
        _identifier(self.comparison_id, "comparison id")
        for value, label in (
            (self.source_lock_digest, "source lock digest"),
            (self.content_digest, "source content digest"),
            (self.publication_id, "source publication id"),
        ):
            _digest(value, label)
        _identifier(self.publication_run_id, "publication run id")
        if not self.source_object.artifact_ref.startswith(
            f"wandb-artifact://{self.destination.project_slug}/"
        ):
            raise PublicTaskSourceError(
                "public source artifact ref targets a different destination"
            )
        expected_name = f"fugue-task-source-{self.publication_id[:20]}"
        expected_qualified = (
            f"{self.destination.project_slug}/{expected_name}:"
            f"{self.source_object.artifact_version}"
        )
        if (
            self.source_object.artifact_name != expected_name
            or self.source_object.qualified_name != expected_qualified
            or self.source_object.artifact_ref
            != f"wandb-artifact://{expected_qualified}"
        ):
            raise PublicTaskSourceError(
                "public source artifact identity disagrees with publication"
            )
        _validate_destination_url(
            self.source_object.artifact_url,
            self.destination,
            version=self.source_object.artifact_version,
        )
        expected_publication_id = stable_digest(
            {
                "comparison_id": self.comparison_id,
                "destination_digest": self.destination.destination_digest,
                "source_lock_digest": self.source_lock_digest,
                "content_digest": self.content_digest,
            }
        )
        if self.publication_id != expected_publication_id:
            raise PublicTaskSourceError(
                "public source publication identity does not recompute"
            )
        computed = stable_digest(self.unsigned_dict())
        if self.receipt_digest and self.receipt_digest != computed:
            raise PublicTaskSourceError("source publication receipt digest changed")
        if not self.receipt_digest:
            object.__setattr__(self, "receipt_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "comparison_id": self.comparison_id,
            "destination": self.destination.to_dict(),
            "source_lock_digest": self.source_lock_digest,
            "content_digest": self.content_digest,
            "publication_id": self.publication_id,
            "publication_run_id": self.publication_run_id,
            "source_object": self.source_object.to_dict(),
            "source_object_identity_digest": self.source_object.identity_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "receipt_digest": self.receipt_digest}


class PublicTaskSourceRemote(Protocol):
    def publish(
        self, *, manifest: PublicTaskSourceManifestV1, publication_id: str
    ) -> Mapping[str, Any]: ...

    def resolve(
        self, *, artifact_ref: str, destination: EvidenceDestinationV1
    ) -> Mapping[str, Any]: ...


def publish_public_task_source(
    manifest: PublicTaskSourceManifestV1,
    *,
    remote: PublicTaskSourceRemote,
) -> PublicTaskSourcePublicationReceiptV1:
    """Publish one public-only manifest and bind the immutable returned object."""

    accepted = public_task_source_manifest_from_dict(manifest.to_dict())
    publication_id = stable_digest(
        {
            "comparison_id": accepted.comparison_id,
            "destination_digest": accepted.destination.destination_digest,
            "source_lock_digest": accepted.source_lock["lock_digest"],
            "content_digest": accepted.content_digest,
        }
    )
    response = _mapping(
        remote.publish(manifest=accepted, publication_id=publication_id),
        "public source publication response",
    )
    receipt = _receipt_from_response(accepted, publication_id, response)
    return verify_public_task_source_publication(
        receipt,
        manifest=accepted,
        remote=remote,
    )


def verify_public_task_source_publication(
    receipt: PublicTaskSourcePublicationReceiptV1 | Mapping[str, Any],
    *,
    manifest: PublicTaskSourceManifestV1,
    remote: PublicTaskSourceRemote,
) -> PublicTaskSourcePublicationReceiptV1:
    accepted = (
        receipt
        if isinstance(receipt, PublicTaskSourcePublicationReceiptV1)
        else public_task_source_publication_receipt_from_dict(receipt)
    )
    if (
        accepted.comparison_id != manifest.comparison_id
        or accepted.destination != manifest.destination
        or accepted.source_lock_digest != manifest.source_lock["lock_digest"]
        or accepted.content_digest != manifest.content_digest
    ):
        raise PublicTaskSourceError(
            "public source publication disagrees with the prepared manifest"
        )
    resolved = _source_object(
        remote.resolve(
            artifact_ref=accepted.source_object.artifact_ref,
            destination=accepted.destination,
        )
    )
    if resolved != accepted.source_object:
        raise PublicTaskSourceError("published public source object drifted")
    return accepted


class WandbPublicTaskSourceRemote:
    """Late-bound W&B adapter used only by trusted preparation and drift checks."""

    def __init__(self, *, wandb_module: Any | None = None) -> None:
        self._wandb_module = wandb_module

    def publish(
        self, *, manifest: PublicTaskSourceManifestV1, publication_id: str
    ) -> Mapping[str, Any]:
        wandb = self._wandb()
        destination = manifest.destination
        artifact_name = f"fugue-task-source-{publication_id[:20]}"
        run_id = f"fugue-task-source-{publication_id[:20]}"
        settings_factory = getattr(wandb, "Settings", None)
        if not callable(settings_factory):
            raise PublicTaskSourceError(
                "public task source publication requires destination-aware Settings"
            )
        settings = settings_factory(
            base_url=destination.api_base_url,
            console="off",
            disable_git=True,
            silent=True,
        )
        run = wandb.init(
            entity=destination.entity,
            project=destination.project,
            id=run_id,
            name=f"Fugue task source · {manifest.comparison_id}",
            job_type="task-source-manifest",
            resume="allow",
            reinit="create_new",
            config={
                "fugue": {
                    "run_kind": "task_source_manifest",
                    "excluded_from_task_inputs": True,
                    "excluded_from_evaluation_counts": True,
                    "publication_id": publication_id,
                    "source_lock_digest": manifest.source_lock["lock_digest"],
                    "content_digest": manifest.content_digest,
                }
            },
            settings=settings,
        )
        if run is None:
            raise PublicTaskSourceError("W&B did not create the task-source Run")
        _validate_run_destination(run, destination)
        try:
            prior = run.summary.get(_SUMMARY_KEY)
            if isinstance(prior, Mapping):
                response = _published_response(prior, publication_id)
                run.finish(exit_code=0)
                return response
            artifact = self._recover(
                wandb,
                destination=destination,
                artifact_name=artifact_name,
                publication_id=publication_id,
                content_digest=manifest.content_digest,
            )
            if artifact is None:
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type=_ARTIFACT_TYPE,
                    metadata={
                        "schema_version": 1,
                        "publication_id": publication_id,
                        "comparison_id": manifest.comparison_id,
                        "source_lock_digest": manifest.source_lock["lock_digest"],
                        "content_digest": manifest.content_digest,
                    },
                )
                with artifact.new_file("public-task-source.json", mode="w") as stream:
                    json.dump(manifest.to_dict(), stream, sort_keys=True)
                    stream.write("\n")
                artifact = run.log_artifact(artifact, aliases=("locked",))
                if hasattr(artifact, "wait"):
                    waited = artifact.wait()
                    if waited is not None:
                        artifact = waited
            response = {
                "publication_run_id": str(run.id),
                "source_object": _wandb_source_object(
                    artifact,
                    destination=destination,
                    artifact_name=artifact_name,
                ).to_dict(),
            }
            run.summary.update(
                {
                    _SUMMARY_KEY: {
                        "schema_version": 1,
                        "publication_id": publication_id,
                        "response": response,
                    }
                }
            )
            run.finish(exit_code=0)
            return response
        except Exception:
            run.finish(exit_code=1)
            raise

    def resolve(
        self, *, artifact_ref: str, destination: EvidenceDestinationV1
    ) -> Mapping[str, Any]:
        path = _artifact_path_from_ref(artifact_ref)
        wandb = self._wandb()
        api_factory = getattr(wandb, "Api", None)
        if not callable(api_factory):
            raise PublicTaskSourceError("public source resolution API is unavailable")
        api = api_factory(overrides={"base_url": destination.api_base_url})
        artifact = api.artifact(path, type=_ARTIFACT_TYPE)
        entity, project, name_version = path.split("/", 2)
        if f"{entity}/{project}" != destination.project_slug:
            raise PublicTaskSourceError(
                "public source artifact ref targets a different project"
            )
        name, _separator, _version = name_version.rpartition(":")
        return _wandb_source_object(
            artifact,
            destination=destination,
            artifact_name=name,
        ).to_dict()

    def _recover(
        self,
        wandb: Any,
        *,
        destination: EvidenceDestinationV1,
        artifact_name: str,
        publication_id: str,
        content_digest: str,
    ) -> Any | None:
        try:
            api_factory = getattr(wandb, "Api", None)
            if not callable(api_factory):
                raise PublicTaskSourceError(
                    "public source recovery API is unavailable"
                )
            api = api_factory(overrides={"base_url": destination.api_base_url})
            artifact = api.artifact(
                f"{destination.project_slug}/{artifact_name}:locked",
                type=_ARTIFACT_TYPE,
            )
        except Exception as exc:
            if type(exc).__name__ not in {
                "CommError",
                "UsageError",
                "ValueError",
            }:
                raise
            return None
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        if (
            metadata.get("publication_id") != publication_id
            or metadata.get("content_digest") != content_digest
        ):
            raise PublicTaskSourceError(
                "existing public source artifact alias has different content"
            )
        return artifact

    def _wandb(self) -> Any:
        if self._wandb_module is not None:
            return self._wandb_module
        try:
            return importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            raise PublicTaskSourceError(
                "public task source publication requires the optional wandb package"
            ) from exc


def public_task_source_manifest_from_dict(
    raw: Mapping[str, Any],
) -> PublicTaskSourceManifestV1:
    value = _exact(
        raw,
        {
            "schema_version",
            "kind",
            "comparison_id",
            "destination",
            "source_lock",
            "public_cases",
            "content_digest",
        },
        "public source manifest",
    )
    return PublicTaskSourceManifestV1(
        schema_version=int(value["schema_version"]),
        kind=str(value["kind"]),
        comparison_id=str(value["comparison_id"]),
        destination=evidence_destination_from_dict(
            _mapping(value["destination"], "public source destination")
        ),
        source_lock=dict(_mapping(value["source_lock"], "public source lock")),
        public_cases=tuple(
            dict(_mapping(item, "public source case"))
            for item in _sequence(value["public_cases"], "public source cases")
        ),
        content_digest=str(value["content_digest"]),
    )


def public_task_source_publication_receipt_from_dict(
    raw: Mapping[str, Any],
) -> PublicTaskSourcePublicationReceiptV1:
    value = _exact(
        raw,
        {
            "schema_version",
            "kind",
            "comparison_id",
            "destination",
            "source_lock_digest",
            "content_digest",
            "publication_id",
            "publication_run_id",
            "source_object",
            "source_object_identity_digest",
            "receipt_digest",
        },
        "public source publication receipt",
    )
    source_object = _source_object(value["source_object"])
    if value["source_object_identity_digest"] != source_object.identity_digest:
        raise PublicTaskSourceError("public source object identity digest changed")
    return PublicTaskSourcePublicationReceiptV1(
        schema_version=int(value["schema_version"]),
        kind=str(value["kind"]),
        comparison_id=str(value["comparison_id"]),
        destination=evidence_destination_from_dict(
            _mapping(value["destination"], "public source destination")
        ),
        source_lock_digest=str(value["source_lock_digest"]),
        content_digest=str(value["content_digest"]),
        publication_id=str(value["publication_id"]),
        publication_run_id=str(value["publication_run_id"]),
        source_object=source_object,
        receipt_digest=str(value["receipt_digest"]),
    )


def _receipt_from_response(
    manifest: PublicTaskSourceManifestV1,
    publication_id: str,
    response: Mapping[str, Any],
) -> PublicTaskSourcePublicationReceiptV1:
    if set(response) != {"publication_run_id", "source_object"}:
        raise PublicTaskSourceError("public source publication response fields changed")
    return PublicTaskSourcePublicationReceiptV1(
        comparison_id=manifest.comparison_id,
        destination=manifest.destination,
        source_lock_digest=str(manifest.source_lock["lock_digest"]),
        content_digest=manifest.content_digest,
        publication_id=publication_id,
        publication_run_id=str(response["publication_run_id"]),
        source_object=_source_object(response["source_object"]),
    )


def _published_response(raw: Mapping[str, Any], publication_id: str) -> Mapping[str, Any]:
    if set(raw) != {"schema_version", "publication_id", "response"}:
        raise PublicTaskSourceError("stored source publication state fields changed")
    if raw.get("schema_version") != 1 or raw.get("publication_id") != publication_id:
        raise PublicTaskSourceError("stored source publication identity changed")
    response = _mapping(raw.get("response"), "stored source publication response")
    if set(response) != {"publication_run_id", "source_object"}:
        raise PublicTaskSourceError("stored source publication response changed")
    _source_object(response["source_object"])
    return response


def _wandb_source_object(
    artifact: Any,
    *,
    destination: EvidenceDestinationV1,
    artifact_name: str,
) -> PublicTaskSourceObjectV1:
    version = str(getattr(artifact, "version", "") or "")
    qualified = str(getattr(artifact, "qualified_name", "") or "")
    value = PublicTaskSourceObjectV1(
        artifact_name=artifact_name,
        artifact_version=version,
        artifact_digest=str(getattr(artifact, "digest", "") or ""),
        qualified_name=qualified,
        artifact_ref=f"wandb-artifact://{qualified}",
        artifact_url=str(getattr(artifact, "url", "") or ""),
    )
    expected_qualified = (
        f"{destination.project_slug}/{artifact_name}:{value.artifact_version}"
    )
    if value.qualified_name != expected_qualified:
        raise PublicTaskSourceError("W&B public source artifact identity disagrees")
    _validate_destination_url(
        value.artifact_url,
        destination,
        version=value.artifact_version,
    )
    return value


def _source_object(raw: Any) -> PublicTaskSourceObjectV1:
    value = _exact(
        _mapping(raw, "public source object"),
        {
            "artifact_name",
            "artifact_version",
            "artifact_digest",
            "qualified_name",
            "artifact_ref",
            "artifact_url",
        },
        "public source object",
    )
    return PublicTaskSourceObjectV1(**{key: str(item) for key, item in value.items()})


def _source_lock(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "kind",
        "public_tasks_sha256",
        "compiled_public_cases_sha256",
        "task_resources",
        "lock_digest",
    }
    value = _exact(raw, expected, "public source lock")
    if value["schema_version"] != 1 or value["kind"] != "public_task_source_lock":
        raise PublicTaskSourceError("public source lock identity is invalid")
    _digest(str(value["public_tasks_sha256"]), "public tasks digest")
    _digest(str(value["compiled_public_cases_sha256"]), "compiled cases digest")
    _digest(str(value["lock_digest"]), "public source lock digest")
    resources = _sequence(
        value["task_resources"], "public source resources", allow_empty=True
    )
    normalized: list[dict[str, str]] = []
    for raw_resource in resources:
        resource = _exact(
            _mapping(raw_resource, "public source resource"),
            {"locked_relative", "sha256"},
            "public source resource",
        )
        relative = str(resource["locked_relative"])
        if (
            not relative
            or relative.startswith("/")
            or ".." in relative.split("/")
        ):
            raise PublicTaskSourceError("public source resource path is unsafe")
        digest = str(resource["sha256"])
        _digest(digest, "public source resource digest")
        normalized.append({"locked_relative": relative, "sha256": digest})
    if normalized != sorted(
        normalized, key=lambda item: (item["locked_relative"], item["sha256"])
    ) or len({(item["locked_relative"], item["sha256"]) for item in normalized}) != len(
        normalized
    ):
        raise PublicTaskSourceError(
            "public source resources must be sorted and unique"
        )
    unsigned = {
        "schema_version": 1,
        "kind": "public_task_source_lock",
        "public_tasks_sha256": str(value["public_tasks_sha256"]),
        "compiled_public_cases_sha256": str(
            value["compiled_public_cases_sha256"]
        ),
        "task_resources": normalized,
    }
    if stable_digest(unsigned) != value["lock_digest"]:
        raise PublicTaskSourceError("public source lock digest does not match")
    return dict(value)


def _artifact_path_from_ref(value: str) -> str:
    prefix = "wandb-artifact://"
    if not value.startswith(prefix):
        raise PublicTaskSourceError("public source artifact ref is invalid")
    path = value.removeprefix(prefix)
    if len(path.split("/")) != 3 or ":" not in path.rsplit("/", 1)[-1]:
        raise PublicTaskSourceError("public source artifact ref is incomplete")
    return path


def _validate_run_destination(
    run: Any,
    destination: EvidenceDestinationV1,
) -> None:
    observed_entity = str(getattr(run, "entity", "") or "")
    observed_project = str(getattr(run, "project", "") or "")
    if observed_entity and observed_entity != destination.entity:
        raise PublicTaskSourceError("public source Run entity disagrees")
    if observed_project and observed_project != destination.project:
        raise PublicTaskSourceError("public source Run project disagrees")


def _validate_destination_url(
    value: str,
    destination: EvidenceDestinationV1,
    *,
    version: str,
) -> None:
    parsed = urlsplit(value)
    app = urlsplit(destination.app_base_url)
    try:
        origin = (parsed.scheme, parsed.hostname, parsed.port)
        expected_origin = (app.scheme, app.hostname, app.port)
    except ValueError as exc:
        raise PublicTaskSourceError("public source artifact URL is invalid") from exc
    project_path = (
        app.path.rstrip("/")
        + f"/{destination.entity}/{destination.project}"
    )
    if (
        origin != expected_origin
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith(project_path + "/")
        or version not in parsed.path.split("/")
    ):
        raise PublicTaskSourceError(
            "public source artifact URL targets a different destination"
        )


def _contains_private_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if any(marker in normalized for marker in ("expected", "private", "gold")):
                return True
            if _contains_private_key(item):
                return True
    elif isinstance(value, list | tuple):
        return any(_contains_private_key(item) for item in value)
    return False


def _exact(
    value: Mapping[str, Any], expected: set[str], label: str
) -> Mapping[str, Any]:
    if set(value) != expected:
        raise PublicTaskSourceError(f"{label} fields do not match")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicTaskSourceError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str, *, allow_empty: bool = False) -> list[Any]:
    if not isinstance(value, list | tuple) or (not value and not allow_empty):
        suffix = "an array" if allow_empty else "a non-empty array"
        raise PublicTaskSourceError(f"{label} must be {suffix}")
    return list(value)


def _digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PublicTaskSourceError(f"{label} must be a SHA-256 digest")


def _identifier(value: str, label: str) -> None:
    if not value or len(value) > 200 or any(character.isspace() for character in value):
        raise PublicTaskSourceError(f"{label} is invalid")
