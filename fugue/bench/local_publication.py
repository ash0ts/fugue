from __future__ import annotations

import hashlib
import importlib
import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import ComparisonResultV3, read_comparison_result
from fugue.bench.files import atomic_write_json
from fugue.bench.local_evidence import (
    LocalEvidenceManifestV1,
    LocalEvidenceStore,
)
from fugue.redaction import redact_text

WEAVE_PUBLICATION_SCHEMA_VERSION = 1

HostedObjectKind = Literal[
    "evaluation_root",
    "prediction_and_score",
    "prediction",
    "agent_evidence_receipt",
    "dataset",
]
PublicationStatus = Literal["published"]

_HOSTED_KINDS = frozenset(
    {
        "evaluation_root",
        "prediction_and_score",
        "prediction",
        "agent_evidence_receipt",
        "dataset",
    }
)
_CALL_KINDS = _HOSTED_KINDS - {"dataset"}
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SLUG_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OBJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,511}$")


class LocalResultPublicationError(ValueError):
    """A canonical local result cannot be published without changing meaning."""


class MissingWeaveExtraError(LocalResultPublicationError):
    """The user requested optional publication without Fugue's Weave extra."""


@dataclass(frozen=True)
class WeavePublicationTargetV1:
    entity: str
    project: str
    schema_version: Literal[1] = WEAVE_PUBLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WEAVE_PUBLICATION_SCHEMA_VERSION:
            raise ValueError("unsupported Weave publication target schema")
        for label, value in (("entity", self.entity), ("project", self.project)):
            if not _SLUG_PART.fullmatch(value):
                raise ValueError(f"invalid Weave publication {label}")

    @property
    def project_slug(self) -> str:
        return f"{self.entity}/{self.project}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeaveHostedObjectRefV1:
    attempt_id: str
    kind: HostedObjectKind
    target: WeavePublicationTargetV1
    object_id: str
    ref: str
    system: Literal["weave"] = "weave"
    native_agent_call: Literal[False] = False

    def __post_init__(self) -> None:
        _digest(self.attempt_id, "hosted object attempt id")
        if self.kind not in _HOSTED_KINDS:
            raise ValueError("unsupported hosted evidence object kind")
        if self.system != "weave":
            raise ValueError("hosted evidence object system must be weave")
        if self.native_agent_call is not False:
            raise ValueError(
                "a published local Agent receipt is not a native Weave Agent call"
            )
        if not _OBJECT_ID.fullmatch(self.object_id):
            raise ValueError("invalid hosted evidence object id")
        object_type = "call" if self.kind in _CALL_KINDS else "object"
        expected_ref = (
            f"weave:///{self.target.project_slug}/{object_type}/{self.object_id}"
        )
        if self.ref != expected_ref:
            raise ValueError(
                "hosted evidence ref disagrees with its exact Weave target"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target"] = self.target.to_dict()
        return value


@dataclass(frozen=True)
class WeavePublicationOutcomeV1:
    target: WeavePublicationTargetV1
    objects: tuple[WeaveHostedObjectRefV1, ...]
    publisher_id: str
    publisher_revision: str
    status: PublicationStatus = "published"

    def __post_init__(self) -> None:
        if self.status != "published":
            raise ValueError("Weave publication did not complete")
        _required_text(self.publisher_id, "Weave publisher id")
        _required_text(self.publisher_revision, "Weave publisher revision")
        if not self.objects:
            raise ValueError("Weave publication returned no hosted objects")
        if tuple(sorted(self.objects, key=_object_sort_key)) != self.objects:
            raise ValueError("hosted evidence objects must be canonically sorted")
        if any(item.target != self.target for item in self.objects):
            raise ValueError("hosted evidence object was published to another project")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "objects": [item.to_dict() for item in self.objects],
            "publisher_id": self.publisher_id,
            "publisher_revision": self.publisher_revision,
            "status": self.status,
        }


class LocalResultWeavePublisher(Protocol):
    def __call__(
        self,
        result: ComparisonResultV3,
        manifest: LocalEvidenceManifestV1,
        target: WeavePublicationTargetV1,
    ) -> WeavePublicationOutcomeV1 | Mapping[str, Any]: ...


def weave_publisher_from_environment(
    env: Mapping[str, str],
) -> LocalResultWeavePublisher:
    """Return the optional SDK publisher without importing Weave at startup."""

    try:
        weave_module = importlib.import_module("weave")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MissingWeaveExtraError(
            'Weave publication requires the optional extra; install '
            '`pip install "fugue[weave]"` and retry.'
        ) from exc

    def publish(
        result: ComparisonResultV3,
        manifest: LocalEvidenceManifestV1,
        target: WeavePublicationTargetV1,
    ) -> WeavePublicationOutcomeV1:
        return _publish_with_weave_sdk(
            weave_module,
            result=result,
            manifest=manifest,
            target=target,
            env=env,
        )

    return publish


@dataclass(frozen=True)
class WeavePublicationReceiptV1:
    publication_id: str
    target: WeavePublicationTargetV1
    result_digest: str
    qualification_digest: str
    result_file_sha256: str
    local_manifest_digest: str
    local_manifest_file_sha256: str
    hosted_objects: tuple[WeaveHostedObjectRefV1, ...]
    publisher_id: str
    publisher_revision: str
    status: PublicationStatus
    published_at: str
    schema_version: Literal[1] = WEAVE_PUBLICATION_SCHEMA_VERSION
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != WEAVE_PUBLICATION_SCHEMA_VERSION:
            raise ValueError("unsupported Weave publication receipt schema")
        for label, value in (
            ("publication id", self.publication_id),
            ("result digest", self.result_digest),
            ("qualification digest", self.qualification_digest),
            ("result file digest", self.result_file_sha256),
            ("local manifest digest", self.local_manifest_digest),
            ("local manifest file digest", self.local_manifest_file_sha256),
        ):
            _digest(value, label)
        _required_text(self.publisher_id, "Weave publisher id")
        _required_text(self.publisher_revision, "Weave publisher revision")
        if self.status != "published":
            raise ValueError("unsupported Weave publication receipt status")
        _timestamp(self.published_at)
        if not self.hosted_objects:
            raise ValueError("Weave publication receipt requires hosted objects")
        if tuple(sorted(self.hosted_objects, key=_object_sort_key)) != (
            self.hosted_objects
        ):
            raise ValueError("publication receipt objects must be canonically sorted")
        if any(item.target != self.target for item in self.hosted_objects):
            raise ValueError("publication receipt contains a cross-project object")
        expected_publication_id = stable_digest(
            {
                "schema_version": self.schema_version,
                "target": self.target.to_dict(),
                "result_digest": self.result_digest,
                "local_manifest_digest": self.local_manifest_digest,
            }
        )
        if self.publication_id != expected_publication_id:
            raise ValueError("Weave publication identity does not match")
        computed = self.computed_digest()
        if self.receipt_digest and self.receipt_digest != computed:
            raise ValueError("Weave publication receipt digest does not match")
        if not self.receipt_digest:
            object.__setattr__(self, "receipt_digest", computed)

    def computed_digest(self) -> str:
        return stable_digest(self._unsigned_dict())

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "target": self.target.to_dict(),
            "result_digest": self.result_digest,
            "qualification_digest": self.qualification_digest,
            "result_file_sha256": self.result_file_sha256,
            "local_manifest_digest": self.local_manifest_digest,
            "local_manifest_file_sha256": self.local_manifest_file_sha256,
            "hosted_objects": [item.to_dict() for item in self.hosted_objects],
            "publisher_id": self.publisher_id,
            "publisher_revision": self.publisher_revision,
            "status": self.status,
            "published_at": self.published_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned_dict(), "receipt_digest": self.computed_digest()}


def publish_local_result_to_weave(
    result_path: Path,
    manifest_path: Path,
    *,
    target: str | WeavePublicationTargetV1,
    publisher: LocalResultWeavePublisher,
    receipt_path: Path | None = None,
    secret_values: Sequence[str] = (),
    clock: Callable[[], datetime] | None = None,
) -> WeavePublicationReceiptV1:
    """Publish one immutable local result without changing its meaning.

    ``publisher`` is the only network-capable seam. It receives the already
    verified V3 result, complete local manifest, and exact target. Fugue then
    verifies the returned five-object chain for every attempt before writing
    a digest-bound receipt. The canonical result remains local and unchanged.
    """

    result_path = _regular_file(result_path, "comparison result")
    manifest_path = _regular_file(manifest_path, "local evidence manifest")
    selected_target = (
        target if isinstance(target, WeavePublicationTargetV1) else parse_target(target)
    )
    result_before = result_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    _assert_secret_free(result_before, secret_values, "comparison result")
    _assert_secret_free(manifest_before, secret_values, "local evidence manifest")
    _assert_secret_free(
        json.dumps(selected_target.to_dict(), sort_keys=True).encode(),
        secret_values,
        "Weave publication target",
    )

    result = read_comparison_result(result_path)
    if not isinstance(result, ComparisonResultV3):
        raise LocalResultPublicationError(
            "optional Weave publication requires ComparisonResultV3"
        )
    result_file_sha256 = hashlib.sha256(result_before).hexdigest()
    manifest_file_sha256 = hashlib.sha256(manifest_before).hexdigest()
    manifest, local_binding = _read_complete_manifest(
        manifest_path,
        manifest_file_sha256=manifest_file_sha256,
    )
    attempt_ids = _validate_local_result(
        result,
        manifest,
        local_binding=local_binding,
    )
    requested_receipt_path = (
        receipt_path
        if receipt_path is not None
        else result_path.with_name("weave-publication-receipt.json")
    )
    if requested_receipt_path.is_symlink():
        raise LocalResultPublicationError("publication receipt cannot be a symlink")
    selected_receipt_path = requested_receipt_path.resolve()

    lock = selected_receipt_path.with_name(f".{selected_receipt_path.name}.lock")
    with FileLock(lock, timeout=120):
        if selected_receipt_path.exists():
            _assert_secret_free(
                selected_receipt_path.read_bytes(),
                secret_values,
                "Weave publication receipt",
            )
            receipt = read_weave_publication_receipt(selected_receipt_path)
            _verify_existing_receipt(
                receipt,
                target=selected_target,
                result=result,
                manifest=manifest,
                result_file_sha256=result_file_sha256,
                manifest_file_sha256=manifest_file_sha256,
                attempt_ids=attempt_ids,
            )
            return receipt

        try:
            raw_outcome = publisher(result, manifest, selected_target)
        finally:
            _verify_sources_unchanged(
                result_path=result_path,
                result_before=result_before,
                manifest_path=manifest_path,
                manifest_before=manifest_before,
            )
        outcome = (
            raw_outcome
            if isinstance(raw_outcome, WeavePublicationOutcomeV1)
            else weave_publication_outcome_from_dict(raw_outcome)
        )
        _validate_outcome(outcome, selected_target, attempt_ids)
        _assert_secret_free(
            json.dumps(outcome.to_dict(), sort_keys=True).encode(),
            secret_values,
            "Weave publication outcome",
        )
        publication_id = stable_digest(
            {
                "schema_version": WEAVE_PUBLICATION_SCHEMA_VERSION,
                "target": selected_target.to_dict(),
                "result_digest": result.result_digest,
                "local_manifest_digest": manifest.manifest_digest,
            }
        )
        now = (clock or (lambda: datetime.now(UTC)))()
        if now.tzinfo is None:
            raise ValueError("publication clock must return a timezone-aware value")
        receipt = WeavePublicationReceiptV1(
            publication_id=publication_id,
            target=selected_target,
            result_digest=result.result_digest,
            qualification_digest=result.qualification_digest,
            result_file_sha256=result_file_sha256,
            local_manifest_digest=manifest.manifest_digest,
            local_manifest_file_sha256=manifest_file_sha256,
            hosted_objects=outcome.objects,
            publisher_id=outcome.publisher_id,
            publisher_revision=outcome.publisher_revision,
            status=outcome.status,
            published_at=now.astimezone(UTC).isoformat(),
        )
        _write_immutable_receipt(selected_receipt_path, receipt)
        reloaded = read_weave_publication_receipt(selected_receipt_path)
        if reloaded != receipt:
            raise LocalResultPublicationError(
                "persisted Weave publication receipt did not round-trip"
            )
        return receipt


def parse_target(value: str) -> WeavePublicationTargetV1:
    parts = value.strip().split("/")
    if len(parts) != 2:
        raise ValueError("Weave target must be exactly ENTITY/PROJECT")
    return WeavePublicationTargetV1(entity=parts[0], project=parts[1])


def read_weave_publication_receipt(path: Path) -> WeavePublicationReceiptV1:
    value = _read_mapping(path, "Weave publication receipt")
    allowed = {
        "schema_version",
        "publication_id",
        "target",
        "result_digest",
        "qualification_digest",
        "result_file_sha256",
        "local_manifest_digest",
        "local_manifest_file_sha256",
        "hosted_objects",
        "publisher_id",
        "publisher_revision",
        "status",
        "published_at",
        "receipt_digest",
    }
    _strict_fields(value, allowed, "Weave publication receipt")
    status = str(value["status"])
    if status != "published":
        raise ValueError("unsupported Weave publication receipt status")
    return WeavePublicationReceiptV1(
        schema_version=_literal_one(value["schema_version"], "receipt"),
        publication_id=str(value["publication_id"]),
        target=_target_from_mapping(value["target"]),
        result_digest=str(value["result_digest"]),
        qualification_digest=str(value["qualification_digest"]),
        result_file_sha256=str(value["result_file_sha256"]),
        local_manifest_digest=str(value["local_manifest_digest"]),
        local_manifest_file_sha256=str(value["local_manifest_file_sha256"]),
        hosted_objects=tuple(
            _object_from_mapping(item)
            for item in _sequence(value["hosted_objects"], "hosted objects")
        ),
        publisher_id=str(value["publisher_id"]),
        publisher_revision=str(value["publisher_revision"]),
        status=cast(PublicationStatus, status),
        published_at=str(value["published_at"]),
        receipt_digest=str(value["receipt_digest"]),
    )


def weave_publication_outcome_from_dict(
    raw: Mapping[str, Any],
) -> WeavePublicationOutcomeV1:
    value = dict(raw)
    _strict_fields(
        value,
        {"target", "objects", "publisher_id", "publisher_revision", "status"},
        "Weave publication outcome",
    )
    status = str(value["status"])
    if status != "published":
        raise ValueError("Weave publication did not complete")
    objects = tuple(
        sorted(
            (
                _object_from_mapping(item)
                for item in _sequence(value["objects"], "hosted objects")
            ),
            key=_object_sort_key,
        )
    )
    return WeavePublicationOutcomeV1(
        target=_target_from_mapping(value["target"]),
        objects=objects,
        publisher_id=str(value["publisher_id"]),
        publisher_revision=str(value["publisher_revision"]),
        status=cast(PublicationStatus, status),
    )


def _publish_with_weave_sdk(
    weave_module: Any,
    *,
    result: ComparisonResultV3,
    manifest: LocalEvidenceManifestV1,
    target: WeavePublicationTargetV1,
    env: Mapping[str, str],
) -> WeavePublicationOutcomeV1:
    from fugue.weave_support import initialize_weave

    active_weave = initialize_weave(target.project_slug, env)
    if active_weave is not weave_module:
        weave_module = active_weave
    get_client = getattr(weave_module, "get_client", None)
    publish_object = getattr(weave_module, "publish", None)
    if not callable(get_client) or not callable(publish_object):
        raise RuntimeError(
            "installed Weave SDK lacks the client or object publication API"
        )
    client = get_client()
    if client is None or not callable(getattr(client, "create_call", None)) or not callable(
        getattr(client, "finish_call", None)
    ):
        raise RuntimeError("installed Weave SDK cannot publish Fugue evidence Calls")
    client_project = str(getattr(client, "project_id", "") or "")
    if client_project and client_project != target.project_slug:
        raise RuntimeError("active Weave client disagrees with the requested project")

    attempts = {
        attempt.attempt_id: (pair, arm, attempt)
        for pair in result.paired_cases
        for arm, attempt in (
            ("baseline", pair.baseline),
            ("candidate", pair.candidate),
        )
        if attempt is not None
    }
    objects: list[WeaveHostedObjectRefV1] = []
    for attempt_id in manifest.planned_attempt_ids:
        pair, arm, attempt = attempts[attempt_id]
        dataset_payload = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "task_id": pair.task_id,
            "harness": pair.harness,
            "attempt": pair.attempt,
            "arm": arm,
            "result_digest": result.result_digest,
            "local_manifest_digest": manifest.manifest_digest,
        }
        published_dataset = publish_object(
            dataset_payload,
            name=f"fugue-local-{result.comparison_id[:48]}-{attempt_id[:12]}",
        )
        dataset_ref = _weave_ref_uri(published_dataset)
        dataset_object_id = _weave_object_id(
            dataset_ref,
            target=target,
            object_type="object",
        )
        objects.append(
            WeaveHostedObjectRefV1(
                attempt_id=attempt_id,
                kind="dataset",
                target=target,
                object_id=dataset_object_id,
                ref=dataset_ref,
            )
        )

        attributes = {
            "fugue.publication.schema_version": 1,
            "fugue.evidence.source": "canonical_local_result",
            "fugue.result_digest": result.result_digest,
            "fugue.qualification_digest": result.qualification_digest,
            "fugue.local_manifest_digest": manifest.manifest_digest,
            "fugue.attempt_id": attempt_id,
            "fugue.task_id": pair.task_id,
            "fugue.arm": arm,
            "fugue.harness": pair.harness,
            "fugue.trial_index": pair.attempt,
            "fugue.candidate_id": str(attempt.identity["candidate"]),
            "fugue.execution_fingerprint": str(attempt.identity["runtime"]),
        }
        root = _create_weave_call(
            client,
            target=target,
            publication_id=result.result_digest,
            attempt_id=attempt_id,
            kind="evaluation_root",
            op="fugue.local_evaluation",
            inputs={"dataset_ref": dataset_ref, "task_id": pair.task_id},
            attributes=attributes,
            display_name=f"Fugue local evaluation · {pair.task_id}",
        )
        predict_and_score = _create_weave_call(
            client,
            target=target,
            publication_id=result.result_digest,
            attempt_id=attempt_id,
            kind="prediction_and_score",
            op="fugue.local_predict_and_score",
            inputs={"attempt_id": attempt_id, "arm": arm},
            attributes=attributes,
            display_name=f"Fugue predict and score · {pair.task_id}",
            parent=root,
        )
        prediction = _create_weave_call(
            client,
            target=target,
            publication_id=result.result_digest,
            attempt_id=attempt_id,
            kind="prediction",
            op="fugue.local_prediction",
            inputs={"attempt_id": attempt_id},
            attributes=attributes,
            display_name=f"Fugue prediction · {pair.task_id}",
            parent=predict_and_score,
        )
        agent_receipt = _create_weave_call(
            client,
            target=target,
            publication_id=result.result_digest,
            attempt_id=attempt_id,
            kind="agent_evidence_receipt",
            op="fugue.local_agent_evidence_receipt",
            inputs={
                "attempt_id": attempt_id,
                "native_agent_call": False,
                "local_manifest_digest": manifest.manifest_digest,
            },
            attributes={
                **attributes,
                "fugue.evidence.kind": "local_agent_cross_transport_receipt_v1",
                "fugue.native_agent_call": False,
            },
            display_name=f"Fugue Agent evidence receipt · {pair.task_id}",
            parent=prediction,
        )
        _finish_weave_call(
            client,
            agent_receipt,
            {
                "status": attempt.execution_status,
                "evidence_status": attempt.evidence_status,
                "native_agent_call": False,
            },
        )
        _finish_weave_call(
            client,
            prediction,
            {
                "passed": attempt.passed,
                "scores": dict(attempt.scores),
                "score_explanations": dict(attempt.score_explanations),
                "sanitized_answer_excerpt": attempt.sanitized_answer_excerpt,
            },
        )
        _finish_weave_call(
            client,
            predict_and_score,
            {
                "passed": attempt.passed,
                "scores": dict(attempt.scores),
                "evidence_status": attempt.evidence_status,
            },
        )
        _finish_weave_call(
            client,
            root,
            {
                "status": "published",
                "result_digest": result.result_digest,
                "local_manifest_digest": manifest.manifest_digest,
            },
        )
        flush = getattr(client, "flush", None)
        if callable(flush):
            flush()
        for kind, call in (
            ("evaluation_root", root),
            ("prediction_and_score", predict_and_score),
            ("prediction", prediction),
            ("agent_evidence_receipt", agent_receipt),
        ):
            call_id = str(getattr(call, "id", "") or "")
            ref = _weave_ref_uri(call) or (
                f"weave:///{target.project_slug}/call/{call_id}"
            )
            objects.append(
                WeaveHostedObjectRefV1(
                    attempt_id=attempt_id,
                    kind=cast(HostedObjectKind, kind),
                    target=target,
                    object_id=call_id,
                    ref=ref,
                )
            )
    version = str(getattr(weave_module, "__version__", "") or "unknown")
    return WeavePublicationOutcomeV1(
        target=target,
        objects=tuple(sorted(objects, key=_object_sort_key)),
        publisher_id="fugue-local-result-weave",
        publisher_revision=f"v1+weave-{version}",
    )


def _create_weave_call(
    client: Any,
    *,
    target: WeavePublicationTargetV1,
    publication_id: str,
    attempt_id: str,
    kind: str,
    op: str,
    inputs: Mapping[str, Any],
    attributes: Mapping[str, Any],
    display_name: str,
    parent: Any | None = None,
) -> Any:
    call_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"fugue:{publication_id}:{attempt_id}:{kind}",
        )
    )
    call = client.create_call(
        op,
        dict(inputs),
        parent=parent,
        attributes=dict(attributes),
        display_name=display_name,
        use_stack=False,
        _call_id_override=call_id,
    )
    if str(getattr(call, "id", "") or "") != call_id:
        raise RuntimeError("Weave publication changed a deterministic Call identity")
    project_id = str(getattr(call, "project_id", "") or "")
    if project_id != target.project_slug:
        raise RuntimeError("published Weave Call belongs to another project")
    if parent is not None and str(getattr(call, "parent_id", "") or "") != str(
        getattr(parent, "id", "") or ""
    ):
        raise RuntimeError("published Weave Call lost its evidence parent")
    return call


def _finish_weave_call(client: Any, call: Any, output: Mapping[str, Any]) -> None:
    client.finish_call(call, output=dict(output))


def _weave_ref_uri(value: Any) -> str:
    ref = getattr(value, "ref", None)
    uri = getattr(ref, "uri", None)
    if ref is not None:
        selected = uri() if callable(uri) else uri or ref
        if selected:
            return str(selected)
    direct = getattr(value, "uri", None)
    selected = direct() if callable(direct) else direct
    return str(selected or "")


def _weave_object_id(
    ref: str,
    *,
    target: WeavePublicationTargetV1,
    object_type: Literal["call", "object"],
) -> str:
    prefix = f"weave:///{target.project_slug}/{object_type}/"
    if not ref.startswith(prefix):
        raise RuntimeError("published Weave object belongs to another project")
    object_id = ref.removeprefix(prefix)
    if not _OBJECT_ID.fullmatch(object_id):
        raise RuntimeError("published Weave object has an invalid identity")
    return object_id


def _read_complete_manifest(
    path: Path,
    *,
    manifest_file_sha256: str,
) -> tuple[LocalEvidenceManifestV1, dict[str, Any]]:
    resolved = path.resolve()
    current_manifest_file_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if current_manifest_file_sha256 != manifest_file_sha256:
        raise LocalResultPublicationError(
            "local evidence manifest changed during publication verification"
        )
    store: LocalEvidenceStore | None = None
    for ancestor in resolved.parents:
        expected = (
            ancestor
            / ".fugue"
            / "runtime"
            / resolved.parent.parent.name
            / "evidence"
            / "manifest.json"
        )
        if expected.resolve() == resolved:
            store = LocalEvidenceStore(ancestor, resolved.parent.parent.name)
            break
    if store is None or store.manifest_path.resolve() != resolved:
        raise LocalResultPublicationError(
            "local evidence manifest is not in the canonical run store"
        )
    try:
        manifest = store.read_manifest()
    except (OSError, ValueError) as exc:
        raise LocalResultPublicationError(
            "local evidence manifest failed canonical recomputation"
        ) from exc
    if manifest.status != "complete":
        raise LocalResultPublicationError(
            f"local evidence chain is {manifest.status}, not complete"
        )
    if not manifest.run_conformance_required or manifest.run_conformance is None:
        raise LocalResultPublicationError(
            "local publication requires an enforced Harbor conformance receipt"
        )
    receipt_path = store.run_conformance_path
    try:
        receipt_raw = receipt_path.read_bytes()
        receipt = _read_mapping(receipt_path, "Harbor conformance receipt")
    except (OSError, ValueError) as exc:
        raise LocalResultPublicationError(
            "Harbor conformance receipt failed canonical recomputation"
        ) from exc
    receipt_digest = str(receipt.get("receipt_sha256") or "")
    receipt_unsigned = {**receipt, "receipt_sha256": ""}
    recomputed_receipt_digest = stable_digest(receipt_unsigned)
    receipt_file_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    if (
        receipt.get("run_id") != manifest.run_id
        or receipt.get("backend") != "local_harbor_docker"
        or receipt.get("status") != "passed"
        or receipt.get("enforced") is not True
        or receipt_digest != recomputed_receipt_digest
        or manifest.run_conformance.status != "passed"
        or not manifest.run_conformance.enforced
        or manifest.run_conformance.receipt_sha256 != receipt_digest
        or manifest.run_conformance.sha256 != receipt_file_sha256
    ):
        raise LocalResultPublicationError(
            "Harbor conformance receipt does not match the complete local manifest"
        )
    return manifest, {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "manifest_digest": manifest.manifest_digest,
        "manifest_file_sha256": current_manifest_file_sha256,
        "plan_digest": manifest.plan_digest,
        "attempt_record_set_digest": manifest.attempt_record_set_digest,
        "prediction_row_set_digest": manifest.prediction_row_set_digest,
        "run_conformance_receipt_digest": receipt_digest,
        "run_conformance_file_sha256": receipt_file_sha256,
    }


def _validate_local_result(
    result: ComparisonResultV3,
    manifest: LocalEvidenceManifestV1,
    *,
    local_binding: Mapping[str, Any],
) -> tuple[str, ...]:
    if result.evidence_backend != "local":
        raise LocalResultPublicationError("comparison result is not local evidence")
    if result.publication_status != "not_requested":
        raise LocalResultPublicationError(
            "canonical local result has an invalid embedded publication status"
        )
    if result.local_chain_integrity != "reconciled":
        raise LocalResultPublicationError(
            "comparison result local evidence chain is not reconciled"
        )
    if result.hosted_chain_integrity != "not_applicable":
        raise LocalResultPublicationError(
            "unpublished local result cannot claim hosted evidence integrity"
        )
    if result.evidence_topology.result_destination != manifest.destination:
        raise LocalResultPublicationError(
            "comparison result and local manifest destinations disagree"
        )
    if result.source != manifest.run_id:
        raise LocalResultPublicationError(
            "comparison result and local manifest run identities disagree"
        )
    if result.local_evidence is None:
        raise LocalResultPublicationError(
            "comparison result is missing its canonical local evidence binding"
        )
    if dict(result.local_evidence) != dict(local_binding):
        mismatches = tuple(
            key
            for key in sorted(set(result.local_evidence) | set(local_binding))
            if result.local_evidence.get(key) != local_binding.get(key)
        )
        detail = ", ".join(mismatches) or "unknown field"
        raise LocalResultPublicationError(
            "comparison result local evidence binding disagrees with recomputed "
            f"evidence: {detail}"
        )
    attempts = tuple(
        sorted(
            attempt.attempt_id
            for pair in result.paired_cases
            for attempt in (pair.baseline, pair.candidate)
            if attempt is not None
        )
    )
    if len(set(attempts)) != len(attempts):
        raise LocalResultPublicationError(
            "comparison result contains duplicate attempt identities"
        )
    if attempts != manifest.planned_attempt_ids:
        raise LocalResultPublicationError(
            "comparison result and local manifest attempt sets disagree"
        )
    if attempts != manifest.terminal_attempt_ids or result.rows != len(attempts):
        raise LocalResultPublicationError(
            "comparison result rows do not match complete local attempts"
        )
    records = {item.attempt_id: item for item in manifest.attempt_records}
    for attempt in (
        item
        for pair in result.paired_cases
        for item in (pair.baseline, pair.candidate)
        if item is not None
    ):
        record = records[attempt.attempt_id]
        if attempt.local_evidence_record_digest != record.record_digest:
            raise LocalResultPublicationError(
                "comparison attempt local evidence record digest disagrees with "
                f"its manifest: {attempt.attempt_id}"
            )
        if (
            record.prediction_row_sha256 is None
            or attempt.local_prediction_row_sha256
            != record.prediction_row_sha256
        ):
            raise LocalResultPublicationError(
                "comparison attempt local prediction digest disagrees with its "
                f"manifest: {attempt.attempt_id}"
            )
    if not _HEX_SHA256.fullmatch(result.result_digest):
        raise LocalResultPublicationError("comparison result digest is invalid")
    if not _HEX_SHA256.fullmatch(result.qualification_digest):
        raise LocalResultPublicationError("comparison qualification digest is invalid")
    return attempts


def _validate_outcome(
    outcome: WeavePublicationOutcomeV1,
    target: WeavePublicationTargetV1,
    attempt_ids: tuple[str, ...],
) -> None:
    if outcome.target != target:
        raise LocalResultPublicationError(
            "publisher returned a different Weave target"
        )
    expected = {
        (attempt_id, kind)
        for attempt_id in attempt_ids
        for kind in _HOSTED_KINDS
    }
    observed = {(item.attempt_id, item.kind) for item in outcome.objects}
    if len(observed) != len(outcome.objects):
        raise LocalResultPublicationError(
            "publisher returned duplicate hosted evidence objects"
        )
    if observed != expected:
        raise LocalResultPublicationError(
            "publisher did not return one complete five-object chain per attempt"
        )


def _verify_existing_receipt(
    receipt: WeavePublicationReceiptV1,
    *,
    target: WeavePublicationTargetV1,
    result: ComparisonResultV3,
    manifest: LocalEvidenceManifestV1,
    result_file_sha256: str,
    manifest_file_sha256: str,
    attempt_ids: tuple[str, ...],
) -> None:
    expected_publication_id = stable_digest(
        {
            "schema_version": WEAVE_PUBLICATION_SCHEMA_VERSION,
            "target": target.to_dict(),
            "result_digest": result.result_digest,
            "local_manifest_digest": manifest.manifest_digest,
        }
    )
    if (
        receipt.publication_id != expected_publication_id
        or receipt.target != target
        or receipt.result_digest != result.result_digest
        or receipt.qualification_digest != result.qualification_digest
        or receipt.result_file_sha256 != result_file_sha256
        or receipt.local_manifest_digest != manifest.manifest_digest
        or receipt.local_manifest_file_sha256 != manifest_file_sha256
    ):
        raise LocalResultPublicationError(
            "conflicting immutable Weave publication receipt already exists"
        )
    _validate_outcome(
        WeavePublicationOutcomeV1(
            target=receipt.target,
            objects=receipt.hosted_objects,
            publisher_id=receipt.publisher_id,
            publisher_revision=receipt.publisher_revision,
            status=receipt.status,
        ),
        target,
        attempt_ids,
    )


def _write_immutable_receipt(
    path: Path, receipt: WeavePublicationReceiptV1
) -> None:
    if path.exists():
        existing = read_weave_publication_receipt(path)
        if existing != receipt:
            raise LocalResultPublicationError(
                "conflicting immutable Weave publication receipt already exists"
            )
        return
    atomic_write_json(path, receipt.to_dict())


def _verify_sources_unchanged(
    *,
    result_path: Path,
    result_before: bytes,
    manifest_path: Path,
    manifest_before: bytes,
) -> None:
    if not result_path.is_file() or result_path.read_bytes() != result_before:
        raise LocalResultPublicationError(
            "publisher mutated the canonical comparison result"
        )
    if not manifest_path.is_file() or manifest_path.read_bytes() != manifest_before:
        raise LocalResultPublicationError(
            "publisher mutated the canonical local evidence manifest"
        )


def _object_from_mapping(raw: Any) -> WeaveHostedObjectRefV1:
    value = dict(_mapping(raw, "hosted evidence object"))
    _strict_fields(
        value,
        {
            "attempt_id",
            "kind",
            "target",
            "object_id",
            "ref",
            "system",
            "native_agent_call",
        },
        "hosted evidence object",
    )
    kind = str(value["kind"])
    if kind not in _HOSTED_KINDS:
        raise ValueError("unsupported hosted evidence object kind")
    return WeaveHostedObjectRefV1(
        attempt_id=str(value["attempt_id"]),
        kind=cast(HostedObjectKind, kind),
        target=_target_from_mapping(value["target"]),
        object_id=str(value["object_id"]),
        ref=str(value["ref"]),
        system=cast(Literal["weave"], str(value["system"])),
        native_agent_call=cast(Literal[False], value["native_agent_call"]),
    )


def _target_from_mapping(raw: Any) -> WeavePublicationTargetV1:
    value = dict(_mapping(raw, "Weave target"))
    _strict_fields(value, {"schema_version", "entity", "project"}, "Weave target")
    return WeavePublicationTargetV1(
        schema_version=_literal_one(value["schema_version"], "target"),
        entity=str(value["entity"]),
        project=str(value["project"]),
    )


def _object_sort_key(item: WeaveHostedObjectRefV1) -> tuple[str, str]:
    return item.attempt_id, item.kind


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise LocalResultPublicationError(f"{label} must be a regular file")
    return resolved


def _assert_secret_free(
    payload: bytes,
    secret_values: Sequence[str],
    label: str,
) -> None:
    text = payload.decode("utf-8", errors="replace")
    for secret in secret_values:
        if len(secret) >= 8 and secret in text:
            raise LocalResultPublicationError(
                f"{label} contains a configured secret value"
            )
    if redact_text(text) != text:
        raise LocalResultPublicationError(
            f"{label} contains secret-shaped content"
        )


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalResultPublicationError(f"{label} is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise LocalResultPublicationError(f"{label} must be an object")
    return dict(raw)


def _mapping(raw: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    return raw


def _sequence(raw: Any, label: str) -> Sequence[Any]:
    if not isinstance(raw, list | tuple):
        raise ValueError(f"{label} must be a list")
    return raw


def _strict_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        if missing:
            details.append("missing " + ", ".join(missing))
        raise ValueError(f"invalid {label}: " + "; ".join(details))


def _literal_one(raw: Any, label: str) -> Literal[1]:
    if raw != 1:
        raise ValueError(f"unsupported {label} schema")
    return 1


def _digest(value: str, label: str) -> None:
    if not _HEX_SHA256.fullmatch(value):
        raise ValueError(f"{label} must be an exact SHA-256 digest")


def _required_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{label} must be non-empty and bounded")


def _timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("publication timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("publication timestamp must include a timezone")
