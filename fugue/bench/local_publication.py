from __future__ import annotations

import base64
import hashlib
import importlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import ComparisonResultV3, read_comparison_result
from fugue.bench.evaluation_semantics import behavioral_task_output_available
from fugue.bench.files import atomic_write_json
from fugue.bench.local_evidence import (
    LocalEvidenceManifestV1,
    LocalEvidenceStore,
    local_result_attempt_projection_v1,
    local_result_projection_digest_candidates_v1,
)
from fugue.bench.task_presentation import (
    task_presentation_from_dict,
    task_result_from_dict,
)
from fugue.model_plane import (
    EvidenceDestinationV1,
    default_evidence_destination,
    evidence_destination_from_dict,
    resolve_evidence_destination,
    trace_project_environment,
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
_WEAVE_CONTENT_HASH = re.compile(r"^(?:[A-Za-z0-9]{43}|[0-9a-f]{64})$")
_SLUG_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OBJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,511}$")
_STUDY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class LocalResultPublicationError(ValueError):
    """A canonical local result cannot be published without changing meaning."""


class MissingWeaveExtraError(LocalResultPublicationError):
    """The user requested optional publication without Fugue's Weave extra."""


def _attempt_score_details(attempt: Any) -> dict[str, dict[str, str]]:
    """Serialize optional score detail contracts, including legacy test doubles."""

    if _attempt_is_nonbehavioral(attempt):
        return {}
    raw = getattr(attempt, "score_details", {})
    if not isinstance(raw, Mapping):
        raise LocalResultPublicationError("comparison score details must be an object")
    details: dict[str, dict[str, str]] = {}
    for dimension, item in raw.items():
        if callable(getattr(item, "to_dict", None)):
            value = item.to_dict()
        elif isinstance(item, Mapping):
            value = dict(item)
        else:
            raise LocalResultPublicationError(
                f"comparison score detail {dimension!r} is invalid"
            )
        details[str(dimension)] = value
    return details


def _attempt_judge_reviews(attempt: Any) -> dict[str, dict[str, Any]]:
    """Serialize optional anchored reviews, including legacy test doubles."""

    if _attempt_is_nonbehavioral(attempt):
        return {}
    raw = getattr(attempt, "judge_reviews", {})
    if not isinstance(raw, Mapping):
        raise LocalResultPublicationError("comparison judge reviews must be an object")
    reviews: dict[str, dict[str, Any]] = {}
    for judge_id, item in raw.items():
        if callable(getattr(item, "to_dict", None)):
            value = item.to_dict()
        elif isinstance(item, Mapping):
            value = dict(item)
        else:
            raise LocalResultPublicationError(
                f"comparison judge review {judge_id!r} is invalid"
            )
        reviews[str(judge_id)] = value
    return reviews


def _attempt_task_presentation(
    result: Any,
    pair: Any,
    attempt: Any,
) -> dict[str, Any] | None:
    """Return the exact public task contract without reconstructing it."""

    direct = getattr(attempt, "task_presentation", None)
    if direct is not None:
        value = (
            direct.to_dict() if callable(getattr(direct, "to_dict", None)) else direct
        )
        if not isinstance(value, Mapping):
            raise LocalResultPublicationError(
                "comparison task presentation must be an object"
            )
        return task_presentation_from_dict(value).to_dict()
    for item in getattr(result, "task_catalogue", ()) or ():
        task_id = str(getattr(item, "task_id", "") or "")
        if task_id != str(pair.task_id):
            continue
        value = item.to_dict() if callable(getattr(item, "to_dict", None)) else item
        if not isinstance(value, Mapping):
            raise LocalResultPublicationError(
                "comparison task catalogue entry must be an object"
            )
        return task_presentation_from_dict(value).to_dict()
    return None


def _attempt_agent_execution_status(attempt: Any) -> str:
    execution_status = str(
        getattr(attempt, "execution_status", "not_applicable") or "not_applicable"
    ).lower()
    return {
        "passed": "completed",
        "success": "completed",
        "succeeded": "completed",
        "completed": "completed",
        "timeout": "timed_out",
        "timed_out": "timed_out",
        "error": "failed",
        "failed": "failed",
        "infrastructure_failed": "failed",
        "unknown": "not_started",
    }.get(execution_status, execution_status)


def _attempt_is_nonbehavioral(attempt: Any) -> bool:
    execution_status = _attempt_agent_execution_status(attempt)
    infrastructure = getattr(attempt, "infrastructure", {})
    terminal_kind = (
        infrastructure.get("terminal_kind")
        if isinstance(infrastructure, Mapping)
        else None
    )
    if execution_status in {
        "not_started",
        "cancelled",
        "interrupted",
        "not_applicable",
    }:
        return True
    if terminal_kind:
        return not behavioral_task_output_available(
            {
                "runtime_outcome": (
                    execution_status
                    if execution_status in {"completed", "timed_out"}
                    else None
                ),
                "terminal_kind": terminal_kind,
            }
        )
    return False


def _attempt_evidence_integrity_status(attempt: Any) -> str:
    return (
        "verified"
        if str(getattr(attempt, "evidence_status", ""))
        in {"resolved", "reconciled"}
        else "incomplete"
    )


def _task_result_display_verdict(task_result: Mapping[str, Any] | None) -> str:
    if task_result is None:
        return "UNSCORED"
    if task_result["task_passed"] is True:
        return "PASSED"
    if task_result["task_passed"] is False:
        return "DID NOT PASS"
    return "INVALID"


def _failed_required_check_ids(
    task_result: Mapping[str, Any] | None,
) -> list[str]:
    if task_result is None:
        return []
    failed_checks = task_result.get("failed_required_checks")
    if not isinstance(failed_checks, Sequence) or isinstance(
        failed_checks, (str, bytes, bytearray)
    ):
        raise LocalResultPublicationError(
            "comparison task result failed checks must be a sequence"
        )
    blocker_ids: set[str] = set()
    for item in failed_checks:
        if not isinstance(item, Mapping):
            raise LocalResultPublicationError(
                "comparison task result failed check must be an object"
            )
        blocker_id = str(item.get("id") or "").strip()
        if not blocker_id:
            raise LocalResultPublicationError(
                "comparison task result failed check requires an id"
            )
        blocker_ids.add(blocker_id)
    return sorted(blocker_ids)


def _attempt_task_result(attempt: Any) -> dict[str, Any] | None:
    if _attempt_is_nonbehavioral(attempt):
        # A nonbehavioral terminal deliberately has no task verdict. Ignore a
        # stale or malformed legacy field at this defensive publication seam.
        return None
    raw = getattr(attempt, "task_result", None)
    if raw is not None:
        value = raw.to_dict() if callable(getattr(raw, "to_dict", None)) else raw
        if not isinstance(value, Mapping):
            raise LocalResultPublicationError(
                "comparison task result must be an object"
            )
        return task_result_from_dict(value).to_dict()
    passed = getattr(attempt, "passed", None)
    failed = (
        []
        if passed is not False
        else [
            {
                "id": "task_outcome_not_satisfied",
                "label": "Required task outcome",
                "explanation": (
                    "This legacy result did not publish a more specific failed check."
                ),
                "critical": True,
            }
        ]
    )
    execution_status = _attempt_agent_execution_status(attempt)
    return task_result_from_dict(
        {
            "schema_version": 1,
            "task_passed": passed if isinstance(passed, bool) else None,
            "outcome_summary": (
                "The Agent satisfied the required task outcome."
                if passed is True
                else "The Agent did not satisfy the required task outcome."
                if passed is False
                else "This legacy result does not contain a task verdict."
            ),
            "failed_required_checks": failed,
            "answer_digest": None,
            "agent_execution_status": execution_status,
            "evidence_integrity_status": _attempt_evidence_integrity_status(attempt),
        }
    ).to_dict()


def _attempt_arm_label(arm: str, attempt: Any) -> str:
    explicit = str(getattr(attempt, "arm_label", "") or "").strip()
    return explicit or arm.title()


def _attempt_treatment_summary(arm: str, attempt: Any) -> str:
    explicit = str(getattr(attempt, "treatment_summary", "") or "").strip()
    if explicit:
        return explicit
    candidate_id = str(getattr(attempt, "identity", {}).get("candidate") or "")
    return f"Exact {arm} candidate {candidate_id[:8] or 'unknown'}."


def _score_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return key or "score"


def _published_scores(
    attempt: Any,
    dimension_roles: Mapping[str, str],
) -> dict[str, Any]:
    if _attempt_is_nonbehavioral(attempt):
        return {}
    result: dict[str, Any] = {
        "task_passed": getattr(attempt, "passed", None),
    }
    for name, value in dict(getattr(attempt, "scores", {})).items():
        role = str(dimension_roles.get(str(name)) or "")
        prefix = {
            "outcome": "outcome",
            "safety_gate": "safety",
            "mechanism": "mechanism",
            "infrastructure": "infrastructure",
            "efficiency": "efficiency",
        }.get(role, "score")
        result[f"{prefix}__{_score_key(str(name).rsplit('.', 1)[-1])}"] = value
    return {key: value for key, value in result.items() if value is not None}


def _weave_row_digest(row: Mapping[str, Any]) -> str:
    """Compute Weave 0.53.6's canonical Dataset-row digest."""

    payload = json.dumps(dict(row), sort_keys=True).encode()
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode()
    return encoded.replace("-", "X").replace("_", "Y").rstrip("=")


def _published_judge_evidence(attempt: Any) -> dict[str, Any]:
    if _attempt_is_nonbehavioral(attempt):
        return {
            "status": "not_applicable",
            "advisory": True,
            "reason": "No behavioral task result was available for judge review.",
        }
    reviews = _attempt_judge_reviews(attempt)
    if not reviews:
        return {
            "status": "unavailable",
            "advisory": True,
            "reason": "This attempt has no published judge review.",
        }
    return {
        "status": "available",
        "advisory": True,
        "reviews": reviews,
    }


@dataclass(frozen=True)
class StudyPublicationScopeV1:
    research_id: str
    study_id: str
    schema_version: Literal[1] = WEAVE_PUBLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WEAVE_PUBLICATION_SCHEMA_VERSION:
            raise ValueError("unsupported Study publication scope schema")
        for label, value in (
            ("research id", self.research_id),
            ("study id", self.study_id),
        ):
            if not _STUDY_ID.fullmatch(value):
                raise ValueError(f"invalid Study publication {label}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeavePublicationTargetV1:
    entity: str
    project: str
    schema_version: Literal[1] = WEAVE_PUBLICATION_SCHEMA_VERSION
    study_scope: StudyPublicationScopeV1 | None = None
    destination: EvidenceDestinationV1 | None = None

    def __post_init__(self) -> None:
        if self.schema_version != WEAVE_PUBLICATION_SCHEMA_VERSION:
            raise ValueError("unsupported Weave publication target schema")
        for label, value in (("entity", self.entity), ("project", self.project)):
            if not _SLUG_PART.fullmatch(value):
                raise ValueError(f"invalid Weave publication {label}")
        if self.study_scope is not None and not isinstance(
            self.study_scope, StudyPublicationScopeV1
        ):
            raise ValueError("invalid Weave publication Study scope")
        if self.destination is not None:
            if not isinstance(self.destination, EvidenceDestinationV1):
                raise ValueError("invalid Weave publication evidence destination")
            if self.destination.project_slug != self.project_slug:
                raise ValueError(
                    "Weave publication target and evidence destination disagree"
                )

    @property
    def project_slug(self) -> str:
        return f"{self.entity}/{self.project}"

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "entity": self.entity,
            "project": self.project,
            "schema_version": self.schema_version,
        }
        if self.study_scope is not None:
            value["study_scope"] = self.study_scope.to_dict()
        if self.destination is not None:
            value["destination"] = self.destination.to_dict()
        return value


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
        if self.kind == "dataset":
            _immutable_dataset_object_id(self.object_id)
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
            "Weave publication requires the optional extra; install "
            '`pip install "fugue[weave]"` and retry.'
        ) from exc

    bound_env = dict(env)

    def publish(
        result: ComparisonResultV3,
        manifest: LocalEvidenceManifestV1,
        target: WeavePublicationTargetV1,
    ) -> WeavePublicationOutcomeV1:
        destination = _require_current_publication_destination(target)
        observed = resolve_evidence_destination(
            trace_project_environment(target.project_slug, bound_env)
        )
        if observed != destination:
            raise LocalResultPublicationError(
                "Weave publisher environment disagrees with the immutable "
                "publication destination"
            )
        return _publish_with_weave_sdk(
            weave_module,
            result=result,
            manifest=manifest,
            target=target,
            env=bound_env,
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
    study_scope: StudyPublicationScopeV1 | None = None,
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
    if isinstance(target, WeavePublicationTargetV1):
        if study_scope is not None and target.study_scope != study_scope:
            raise LocalResultPublicationError(
                "Study publication scope disagrees with the Weave target"
            )
        selected_target = target
    else:
        selected_target = parse_target(target, study_scope=study_scope)
    _require_current_publication_destination(selected_target)
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


def parse_target(
    value: str,
    *,
    study_scope: StudyPublicationScopeV1 | None = None,
) -> WeavePublicationTargetV1:
    parts = value.strip().split("/")
    if len(parts) != 2:
        raise ValueError("Weave target must be exactly ENTITY/PROJECT")
    return WeavePublicationTargetV1(
        entity=parts[0],
        project=parts[1],
        study_scope=study_scope,
        destination=default_evidence_destination(value.strip()),
    )


def weave_publication_target_from_environment(
    value: str,
    env: Mapping[str, str],
    *,
    study_scope: StudyPublicationScopeV1 | None = None,
) -> WeavePublicationTargetV1:
    """Resolve one endpoint-bound publication target from an exact environment."""

    parts = value.strip().split("/")
    if len(parts) != 2:
        raise ValueError("Weave target must be exactly ENTITY/PROJECT")
    destination = resolve_evidence_destination(
        trace_project_environment(value.strip(), env)
    )
    return WeavePublicationTargetV1(
        entity=parts[0],
        project=parts[1],
        study_scope=study_scope,
        destination=destination,
    )


def read_weave_publication_receipt(path: Path) -> WeavePublicationReceiptV1:
    value = _read_mapping(path, "Weave publication receipt")
    return weave_publication_receipt_from_dict(value)


def weave_publication_receipt_from_dict(
    raw: Mapping[str, Any],
) -> WeavePublicationReceiptV1:
    """Parse and verify a canonical receipt without requiring a temporary file."""

    value = dict(raw)
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
    from fugue.weave_support import weave_destination_session

    destination = _require_current_publication_destination(target)
    with weave_destination_session(destination, env) as active_weave:
        return _publish_with_active_weave_sdk(
            active_weave if active_weave is not weave_module else weave_module,
            result=result,
            manifest=manifest,
            target=target,
        )


def _publish_with_active_weave_sdk(
    weave_module: Any,
    *,
    result: ComparisonResultV3,
    manifest: LocalEvidenceManifestV1,
    target: WeavePublicationTargetV1,
) -> WeavePublicationOutcomeV1:
    get_client = getattr(weave_module, "get_client", None)
    publish_object = getattr(weave_module, "publish", None)
    if not callable(get_client) or not callable(publish_object):
        raise RuntimeError(
            "installed Weave SDK lacks the client or object publication API"
        )
    client = get_client()
    required_client_methods = (
        "create_call",
        "finish_call",
        "flush",
        "get_call",
        "get_calls",
        "get",
    )
    if client is None or any(
        not callable(getattr(client, method, None))
        for method in required_client_methods
    ):
        raise RuntimeError(
            "installed Weave SDK cannot publish and authoritatively read back "
            "Fugue evidence"
        )
    ref_factory = getattr(weave_module, "ref", None)
    if not callable(ref_factory):
        raise RuntimeError(
            "installed Weave SDK lacks the immutable object readback API"
        )
    client_project = str(getattr(client, "project_id", "") or "")
    if client_project != target.project_slug:
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
    publication_id = stable_digest(
        {
            "schema_version": WEAVE_PUBLICATION_SCHEMA_VERSION,
            "target": target.to_dict(),
            "result_digest": result.result_digest,
            "local_manifest_digest": manifest.manifest_digest,
        }
    )
    objects: list[WeaveHostedObjectRefV1] = []
    dataset_rows_by_task: dict[str, dict[str, Any]] = {}
    for pair, _arm, attempt in attempts.values():
        task_presentation = _attempt_task_presentation(result, pair, attempt)
        public_row = {
            "example_id": str(pair.task_id),
            "task_id": str(pair.task_id),
            "task_title": str((task_presentation or {}).get("title") or pair.task_id),
            "task_presentation": task_presentation,
        }
        public_row["row_payload_digest"] = _weave_row_digest(public_row)
        existing = dataset_rows_by_task.get(str(pair.task_id))
        if existing is not None and existing != public_row:
            raise LocalResultPublicationError(
                "comparison result contains conflicting public task definitions"
            )
        dataset_rows_by_task[str(pair.task_id)] = public_row
    dataset_payload = {
        "schema_version": 1,
        "publication_id": publication_id,
        "evidence_destination_digest": (
            _require_current_publication_destination(target).destination_digest
        ),
        **(
            {
                "fugue_research_id": target.study_scope.research_id,
                "fugue_study_key": target.study_scope.study_id,
            }
            if target.study_scope is not None
            else {}
        ),
        "result_digest": result.result_digest,
        "qualification_digest": result.qualification_digest,
        "local_manifest_digest": manifest.manifest_digest,
        "rows": [dataset_rows_by_task[key] for key in sorted(dataset_rows_by_task)],
    }
    dataset_name = f"fugue-local-{stable_digest(dataset_payload)}"
    published_dataset = publish_object(dataset_payload, name=dataset_name)
    dataset_ref = _weave_ref_uri(published_dataset)
    dataset_object_id = _weave_object_id(
        dataset_ref,
        target=target,
        object_type="object",
    )
    _verify_weave_dataset_readback(
        client,
        ref_factory=ref_factory,
        target=target,
        dataset_ref=dataset_ref,
        expected=dataset_payload,
    )
    for attempt_id in manifest.planned_attempt_ids:
        pair, arm, attempt = attempts[attempt_id]
        task_presentation = _attempt_task_presentation(result, pair, attempt)
        task_title = str((task_presentation or {}).get("title") or pair.task_id)
        arm_label = _attempt_arm_label(arm, attempt)
        treatment_summary = _attempt_treatment_summary(arm, attempt)
        task_result = _attempt_task_result(attempt)
        dimension_roles = {
            str(change.id): str(change.role)
            for change in getattr(pair, "dimension_changes", ())
        }
        published_scores = _published_scores(attempt, dimension_roles)
        score_details = _attempt_score_details(attempt)
        judge_evidence = _published_judge_evidence(attempt)
        failed_required_check_ids = _failed_required_check_ids(task_result)
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
            "fugue.publication_id": publication_id,
            "fugue.evidence.destination_digest": (
                _require_current_publication_destination(target).destination_digest
            ),
            "fugue.evidence.source": "canonical_local_result",
            "fugue.evidence.kind": "immutable_local_result_replay_v1",
            "fugue.native_evaluation_call": False,
            "fugue.comparison_id": result.comparison_id,
            "fugue.result_digest": result.result_digest,
            "fugue.qualification_digest": result.qualification_digest,
            "fugue.local_manifest_digest": manifest.manifest_digest,
            "fugue.attempt_id": attempt_id,
            "fugue.task_id": pair.task_id,
            "fugue.task_title": task_title,
            "fugue.arm": arm,
            "fugue.arm_label": arm_label,
            "fugue.treatment_summary": treatment_summary,
            "fugue.failed_required_check_ids": failed_required_check_ids,
            **(
                {
                    "fugue.task_verdict": (
                        "passed"
                        if task_result["task_passed"] is True
                        else "did_not_pass"
                        if task_result["task_passed"] is False
                        else "invalid"
                    )
                }
                if task_result is not None
                else {}
            ),
            "fugue.harness": pair.harness,
            "fugue.trial_index": pair.attempt,
            "fugue.candidate_id": str(attempt.identity["candidate"]),
            "fugue.execution_fingerprint": str(attempt.identity["runtime"]),
            **(
                {
                    "fugue.research_id": target.study_scope.research_id,
                    "fugue.study_key": target.study_scope.study_id,
                }
                if target.study_scope is not None
                else {}
            ),
        }
        root_inputs = {
            "dataset_ref": dataset_ref,
            "task_id": pair.task_id,
            "task_title": task_title,
            "task_presentation": task_presentation,
        }
        root_output = {
            "status": "published",
            "publication_semantics": "immutable_local_result_replay_v1",
            "native_evaluation_call": False,
            "publication_id": publication_id,
            "result_digest": result.result_digest,
            "local_manifest_digest": manifest.manifest_digest,
        }
        predict_and_score_inputs = {
            "attempt_id": attempt_id,
            "arm": arm,
            "arm_label": arm_label,
            "treatment_summary": treatment_summary,
            "task_presentation": task_presentation,
        }
        predict_and_score_output = {
            **({"task_result": task_result} if task_result is not None else {}),
            "scores": published_scores,
            "score_details": score_details,
            "judge_evidence": judge_evidence,
            "evidence_status": attempt.evidence_status,
        }
        prediction_inputs = {
            "attempt_id": attempt_id,
            "task_title": task_title,
            "arm_label": arm_label,
            "treatment_summary": treatment_summary,
        }
        prediction_output = {
            **({"task_result": task_result} if task_result is not None else {}),
            "scores": published_scores,
            "score_explanations": (
                {} if _attempt_is_nonbehavioral(attempt) else dict(attempt.score_explanations)
            ),
            "score_details": score_details,
            "judge_evidence": judge_evidence,
            "sanitized_answer_excerpt": (
                None
                if _attempt_is_nonbehavioral(attempt)
                else attempt.sanitized_answer_excerpt
            ),
        }
        agent_inputs = {
            "attempt_id": attempt_id,
            "native_agent_call": False,
            "local_manifest_digest": manifest.manifest_digest,
        }
        agent_output = {
            "agent_execution_status": (
                task_result["agent_execution_status"]
                if task_result is not None
                else _attempt_agent_execution_status(attempt)
            ),
            "evidence_integrity_status": (
                task_result["evidence_integrity_status"]
                if task_result is not None
                else _attempt_evidence_integrity_status(attempt)
            ),
            "native_agent_call": False,
        }
        root_display_name = (
            "Published evaluation record: "
            f"{_task_result_display_verdict(task_result)} · {task_title} · "
            f"{arm_label} · Attempt {pair.attempt}"
        )
        predict_and_score_display_name = (
            "Published scored attempt: "
            f"{_task_result_display_verdict(task_result)}"
            f" · {task_title} · {arm_label} · Attempt {pair.attempt}"
        )
        prediction_display_name = (
            "Published prediction record: "
            + (
                f"Agent execution (no scored answer) · {task_title} · "
                if _attempt_is_nonbehavioral(attempt)
                else f"Agent answer · {task_title} · "
            )
            + f"{arm_label} · Attempt {pair.attempt}"
        )
        agent_receipt_display_name = (
            f"Receipt: Agent execution · {task_title} · "
            f"{arm_label} · Attempt {pair.attempt}"
        )
        root = _ensure_weave_call_started(
            client,
            target=target,
            publication_id=publication_id,
            attempt_id=attempt_id,
            kind="evaluation_root",
            op="fugue.local_evaluation",
            inputs=root_inputs,
            attributes=attributes,
            display_name=root_display_name,
        )
        predict_and_score = _ensure_weave_call_started(
            client,
            target=target,
            publication_id=publication_id,
            attempt_id=attempt_id,
            kind="prediction_and_score",
            op="fugue.local_predict_and_score",
            inputs=predict_and_score_inputs,
            attributes=attributes,
            display_name=predict_and_score_display_name,
            parent=root,
        )
        prediction = _ensure_weave_call_started(
            client,
            target=target,
            publication_id=publication_id,
            attempt_id=attempt_id,
            kind="prediction",
            op="fugue.local_prediction",
            inputs=prediction_inputs,
            attributes=attributes,
            display_name=prediction_display_name,
            parent=predict_and_score,
        )
        agent_receipt = _ensure_weave_call_started(
            client,
            target=target,
            publication_id=publication_id,
            attempt_id=attempt_id,
            kind="agent_evidence_receipt",
            op="fugue.local_agent_evidence_receipt",
            inputs=agent_inputs,
            attributes={
                **attributes,
                "fugue.evidence.kind": "local_agent_cross_transport_receipt_v1",
                "fugue.native_agent_call": False,
            },
            display_name=agent_receipt_display_name,
            parent=prediction,
        )
        call_specs = (
            (
                "agent_evidence_receipt",
                agent_receipt,
                agent_inputs,
                agent_output,
                prediction,
                {
                    **attributes,
                    "fugue.evidence.kind": "local_agent_cross_transport_receipt_v1",
                    "fugue.native_agent_call": False,
                },
                agent_receipt_display_name,
            ),
            (
                "prediction",
                prediction,
                prediction_inputs,
                prediction_output,
                predict_and_score,
                attributes,
                prediction_display_name,
            ),
            (
                "prediction_and_score",
                predict_and_score,
                predict_and_score_inputs,
                predict_and_score_output,
                root,
                attributes,
                predict_and_score_display_name,
            ),
            (
                "evaluation_root",
                root,
                root_inputs,
                root_output,
                None,
                attributes,
                root_display_name,
            ),
        )
        # Weave 0.53.6 uses calls-complete batching by default.  An open Call
        # exists only in the local batch processor until its matching end is
        # queued, and flush() waits for that end.  Build and close the complete
        # chain first, then flush it once as paired start/end records.
        for (
            _kind,
            call,
            _inputs,
            output,
            _parent,
            _attributes,
            _display_name,
        ) in call_specs:
            _finish_or_verify_weave_call(client, call, output)
        client.flush()

        # Only authoritative API readback may turn the submission into a
        # published outcome. Local Call objects and flush completion are not
        # evidence that the exact project persisted a finished chain.
        readback: dict[str, Any] = {}
        for (
            kind,
            call,
            inputs,
            output,
            parent,
            expected_attributes,
            expected_display_name,
        ) in call_specs:
            call_id = str(getattr(call, "id", "") or "")
            matches = _read_back_weave_publication_calls(
                client,
                publication_id=publication_id,
                attempt_id=attempt_id,
                kind=kind,
            )
            if len(matches) > 1:
                raise RuntimeError(
                    "Weave publication found a duplicate Call race for one "
                    "immutable evidence identity"
                )
            if not matches:
                raise RuntimeError(
                    "authoritative Weave readback could not resolve a created Call"
                )
            if str(getattr(matches[0], "id", "") or "") != call_id:
                raise RuntimeError(
                    "authoritative Weave readback returned another Call for a "
                    "created evidence identity"
                )
            observed = _read_back_weave_call(client, call_id)
            _verify_weave_call(
                observed,
                target=target,
                call_id=call_id,
                parent_id=(str(getattr(parent, "id", "") or "") if parent else None),
                inputs=inputs,
                attributes=expected_attributes,
                display_name=expected_display_name,
                output=output,
                require_finished=True,
            )
            readback[kind] = observed
        root_trace_id = str(
            getattr(readback["evaluation_root"], "trace_id", "") or ""
        )
        if not root_trace_id:
            raise RuntimeError(
                "authoritative Weave readback omitted the evidence trace identity"
            )
        for observed in readback.values():
            if str(getattr(observed, "trace_id", "") or "") != root_trace_id:
                raise RuntimeError(
                    "authoritative Weave readback lost the evidence trace identity"
                )

        for kind in (
            "evaluation_root",
            "prediction_and_score",
            "prediction",
            "agent_evidence_receipt",
        ):
            call = readback[kind]
            call_id = str(getattr(call, "id", "") or "")
            ref = f"weave:///{target.project_slug}/call/{call_id}"
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
        publisher_revision=f"v4-blocker-filter-display-readback+weave-{version}",
    )


def _ensure_weave_call_started(
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
    expected_attributes = {
        **dict(attributes),
        "fugue.evidence.object_kind": kind,
    }
    matches = _query_weave_publication_calls(
        client,
        publication_id=publication_id,
        attempt_id=attempt_id,
        kind=kind,
    )
    if len(matches) > 1:
        raise RuntimeError(
            "Weave publication found duplicate Calls for one immutable "
            "evidence identity"
        )
    if matches:
        call = matches[0]
        if getattr(call, "ended_at", None) is None:
            raise RuntimeError(
                "Weave publication found an unfinished remote Call that cannot "
                "be resumed safely with the public calls-complete API; inspect "
                "the partial publication and select a clean target"
            )
    else:
        created = client.create_call(
            op,
            dict(inputs),
            parent=parent,
            attributes=expected_attributes,
            display_name=display_name,
            use_stack=False,
        )
        created_id = str(getattr(created, "id", "") or "")
        if not created_id:
            raise RuntimeError("Weave did not return an identity for a created Call")
        call = created
    call_id = str(getattr(call, "id", "") or "")
    if not call_id:
        raise RuntimeError("published Weave Call has no identity")
    _verify_weave_call(
        call,
        target=target,
        call_id=call_id,
        parent_id=(str(getattr(parent, "id", "") or "") if parent else None),
        inputs=inputs,
        attributes=expected_attributes,
        display_name=display_name,
        output=None,
        require_finished=False,
    )
    return call


def _finish_or_verify_weave_call(
    client: Any,
    call: Any,
    output: Mapping[str, Any],
) -> None:
    existing_output = getattr(call, "output", None)
    ended_at = getattr(call, "ended_at", None)
    if ended_at is not None:
        if not _canonical_values_equal(existing_output, output):
            raise RuntimeError(
                "existing deterministic Weave Call has a conflicting output"
            )
        return
    if existing_output is not None:
        raise RuntimeError(
            "existing deterministic Weave Call has output without completion"
        )
    client.finish_call(call, output=dict(output))


def _query_weave_publication_calls(
    client: Any,
    *,
    publication_id: str,
    attempt_id: str,
    kind: str,
) -> list[Any]:
    # Fugue stores namespaced attributes as flat keys (for example,
    # ``{"fugue.publication_id": ...}``).  Weave's public query grammar uses
    # an unescaped dot as a JSON path separator, so literal dots in one key
    # must be escaped.  Without the escapes, the server looks for a nested
    # ``attributes["fugue"]["publication_id"]`` value and cannot recover a
    # terminal publication after a controller retry.
    query = {
        "$expr": {
            "$and": [
                {
                    "$eq": [
                        {"$getField": r"attributes.fugue\.publication_id"},
                        {"$literal": publication_id},
                    ]
                },
                {
                    "$eq": [
                        {"$getField": r"attributes.fugue\.attempt_id"},
                        {"$literal": attempt_id},
                    ]
                },
                {
                    "$eq": [
                        {
                            "$getField": (
                                r"attributes.fugue\.evidence\.object_kind"
                            )
                        },
                        {"$literal": kind},
                    ]
                },
            ]
        }
    }
    return list(client.get_calls(query=query, limit=2))


def _read_back_weave_publication_calls(
    client: Any,
    *,
    publication_id: str,
    attempt_id: str,
    kind: str,
) -> list[Any]:
    for retry in range(5):
        matches = _query_weave_publication_calls(
            client,
            publication_id=publication_id,
            attempt_id=attempt_id,
            kind=kind,
        )
        if matches or retry == 4:
            return matches
        time.sleep(0.05 * (retry + 1))
    raise AssertionError("unreachable Weave publication query retry state")


def _read_back_weave_call(client: Any, call_id: str) -> Any:
    for retry in range(5):
        try:
            return client.get_call(call_id)
        except ValueError as exc:
            if "call not found" not in str(exc).lower() or retry == 4:
                raise RuntimeError(
                    "authoritative Weave readback could not resolve a published Call"
                ) from exc
            time.sleep(0.05 * (retry + 1))
    raise AssertionError("unreachable Weave readback retry state")


def _verify_weave_call(
    call: Any,
    *,
    target: WeavePublicationTargetV1,
    call_id: str,
    parent_id: str | None,
    inputs: Mapping[str, Any],
    attributes: Mapping[str, Any],
    display_name: str,
    output: Mapping[str, Any] | None,
    require_finished: bool,
) -> None:
    if str(getattr(call, "id", "") or "") != call_id:
        raise RuntimeError("Weave publication changed a deterministic Call identity")
    if str(getattr(call, "project_id", "") or "") != target.project_slug:
        raise RuntimeError("published Weave Call belongs to another project")
    observed_parent = getattr(call, "parent_id", None)
    if (str(observed_parent) if observed_parent is not None else None) != parent_id:
        raise RuntimeError("published Weave Call lost its evidence parent")
    if not _canonical_values_equal(getattr(call, "inputs", None), inputs):
        raise RuntimeError("published Weave Call inputs disagree with local evidence")
    if getattr(call, "display_name", None) != display_name:
        raise RuntimeError(
            "published Weave Call display name disagrees with local evidence"
        )
    observed_attributes = _mapping_value(getattr(call, "attributes", None))
    missing_or_changed = {
        key: value
        for key, value in attributes.items()
        if key not in observed_attributes
        or not _canonical_values_equal(observed_attributes[key], value)
    }
    if missing_or_changed:
        raise RuntimeError(
            "published Weave Call attributes disagree with local identities"
        )
    if not require_finished:
        return
    if getattr(call, "ended_at", None) is None:
        raise RuntimeError("authoritative Weave readback returned an unfinished Call")
    if getattr(call, "exception", None) not in (None, ""):
        raise RuntimeError("authoritative Weave readback returned a failed Call")
    if output is None or not _canonical_values_equal(
        getattr(call, "output", None), output
    ):
        raise RuntimeError(
            "authoritative Weave readback returned a conflicting Call output"
        )


def _verify_weave_dataset_readback(
    client: Any,
    *,
    ref_factory: Callable[[str], Any],
    target: WeavePublicationTargetV1,
    dataset_ref: str,
    expected: Mapping[str, Any],
) -> None:
    try:
        parsed_ref = ref_factory(dataset_ref)
    except Exception as exc:
        raise RuntimeError("published Weave Dataset ref could not be parsed") from exc
    if (
        str(getattr(parsed_ref, "entity", "") or "") != target.entity
        or str(getattr(parsed_ref, "project", "") or "") != target.project
    ):
        raise RuntimeError("published Weave Dataset belongs to another project")
    try:
        observed = client.get(parsed_ref, objectify=False)
    except Exception as exc:
        raise RuntimeError(
            "authoritative Weave readback could not resolve the published Dataset"
        ) from exc
    if not _canonical_values_equal(observed, expected):
        raise RuntimeError(
            "authoritative Weave Dataset readback disagrees with local identities"
        )


def _mapping_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {}


def _canonical_values_equal(left: Any, right: Any) -> bool:
    def normalize(value: Any) -> Any:
        # Weave's public Call readback wraps inputs in a WeaveDict whose
        # ``items()`` method objectifies stored refs. Compare the persisted
        # wire identity instead: read raw WeaveDict entries and normalize
        # immutable Weave Ref objects to their URI before model serialization.
        weave_uri = _immutable_weave_ref_uri(value)
        if weave_uri is not None:
            return weave_uri
        if isinstance(value, Mapping):
            items = _wire_mapping_items(value)
            return {
                str(key): normalize(item)
                for key, item in sorted(items, key=lambda item: str(item[0]))
            }
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [normalize(item) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return normalize(model_dump())
        return value

    return normalize(left) == normalize(right)


def _immutable_weave_ref_uri(value: Any) -> str | None:
    ref_type = _optional_weave_type("weave.trace.refs", "Ref")
    if ref_type is None or not isinstance(value, ref_type):
        return None
    direct = getattr(value, "uri", None)
    selected = direct() if callable(direct) else direct
    uri = str(selected or "")
    return uri if uri.startswith("weave:///") else None


def _wire_mapping_items(value: Mapping[Any, Any]) -> Any:
    weave_dict_type = _optional_weave_type("weave.trace.vals", "WeaveDict")
    if weave_dict_type is not None and isinstance(value, weave_dict_type):
        return dict.items(value)
    return value.items()


def _optional_weave_type(module_name: str, name: str) -> type[Any] | None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    selected = getattr(module, name, None)
    return selected if isinstance(selected, type) else None


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
    if object_type == "object":
        try:
            _immutable_dataset_object_id(object_id)
        except ValueError as exc:
            raise RuntimeError(
                "published Weave object ref is not an immutable content revision"
            ) from exc
    return object_id


def _immutable_dataset_object_id(value: str) -> None:
    """Require the exact immutable ``name:content-revision`` Weave identity."""

    if value.count(":") != 1:
        raise ValueError(
            "hosted Dataset object id must contain one name and content revision"
        )
    name, revision = value.split(":", 1)
    if not name or not _WEAVE_CONTENT_HASH.fullmatch(revision):
        raise ValueError(
            "hosted Dataset object id must use a positive content hash"
        )


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
    binding = {
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
    if manifest.result_row_projection_set_digest is not None:
        binding["result_row_projection_set_digest"] = (
            manifest.result_row_projection_set_digest
        )
    return manifest, binding


def verify_comparison_result_local_evidence(
    result: ComparisonResultV3,
    manifest_path: Path,
) -> LocalEvidenceManifestV1:
    """Recompute and reconcile the local ledger bound by a V3 result.

    This is the network-free verification boundary shared by result display and
    optional hosted publication.  It deliberately accepts both local-only and
    dual-chain V3 results: hosted status is irrelevant to whether the canonical
    local attempt records still match the decision-bearing result.
    """

    if not isinstance(result, ComparisonResultV3):
        raise LocalResultPublicationError(
            "local evidence verification requires ComparisonResultV3"
        )
    try:
        manifest_raw = manifest_path.resolve().read_bytes()
    except OSError as exc:
        raise LocalResultPublicationError(
            "comparison result local evidence manifest is unavailable"
        ) from exc
    manifest, local_binding = _read_complete_manifest(
        manifest_path,
        manifest_file_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    )
    _validate_result_local_evidence_binding(
        result,
        manifest,
        local_binding=local_binding,
    )
    return manifest


def _validate_result_local_evidence_binding(
    result: ComparisonResultV3,
    manifest: LocalEvidenceManifestV1,
    *,
    local_binding: Mapping[str, Any],
) -> tuple[str, ...]:
    if result.local_chain_integrity != "reconciled":
        raise LocalResultPublicationError(
            "comparison result local evidence chain is not reconciled"
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
            or attempt.local_prediction_row_sha256 != record.prediction_row_sha256
        ):
            raise LocalResultPublicationError(
                "comparison attempt local prediction digest disagrees with its "
                f"manifest: {attempt.attempt_id}"
            )
        if record.result_row_projection_digest is not None and (
            attempt.local_result_row_projection_digest
            != record.result_row_projection_digest
        ):
            raise LocalResultPublicationError(
                "comparison attempt decision projection digest disagrees with "
                f"its manifest: {attempt.attempt_id}"
            )
        if record.result_row_projection_digest is not None:
            projection = local_result_attempt_projection_v1(
                attempt_id=attempt.attempt_id,
                prediction_id=attempt.prediction_id,
                passed=attempt.passed,
                execution_status=attempt.execution_status,
                evaluation_status=attempt.evaluation_status,
                cost_usd=attempt.cost_usd,
                latency_sec=attempt.latency_sec,
                input_tokens=attempt.input_tokens,
                output_tokens=attempt.output_tokens,
                tool_calls=attempt.tool_calls,
                tools=attempt.tools,
                queried_projects=attempt.queried_projects,
                scores=attempt.scores,
                score_explanations=attempt.score_explanations,
                sanitized_answer_excerpt=attempt.sanitized_answer_excerpt,
                actual_query_scope=attempt.actual_query_scope,
                reported_project_identity=attempt.reported_project_identity,
                execution_fingerprint=attempt.execution_fingerprint,
                runtime_lock_digest=attempt.runtime_lock_digest,
                cost_reconciliation_status=attempt.cost_reconciliation_status,
                latency_reconciliation_status=attempt.latency_reconciliation_status,
                usage_reconciliation_status=attempt.usage_reconciliation_status,
                score_details=_attempt_score_details(attempt),
                judge_reviews=_attempt_judge_reviews(attempt),
                task_presentation=(
                    attempt.task_presentation.to_dict()
                    if getattr(attempt, "task_presentation", None) is not None
                    else None
                ),
                arm_label=getattr(attempt, "arm_label", None),
                treatment_summary=getattr(attempt, "treatment_summary", None),
                task_result=(
                    attempt.task_result.to_dict()
                    if getattr(attempt, "task_result", None) is not None
                    else None
                ),
            )
            if record.result_row_projection_digest not in (
                local_result_projection_digest_candidates_v1(projection)
            ):
                raise LocalResultPublicationError(
                    "comparison attempt decision fields disagree with its immutable "
                    f"prediction artifact: {attempt.attempt_id}"
                )
    if not _HEX_SHA256.fullmatch(result.result_digest):
        raise LocalResultPublicationError("comparison result digest is invalid")
    if not _HEX_SHA256.fullmatch(result.qualification_digest):
        raise LocalResultPublicationError("comparison qualification digest is invalid")
    return attempts


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
    if result.hosted_chain_integrity != "not_applicable":
        raise LocalResultPublicationError(
            "unpublished local result cannot claim hosted evidence integrity"
        )
    if result.evidence_topology.result_destination != manifest.destination:
        raise LocalResultPublicationError(
            "comparison result and local manifest destinations disagree"
        )
    return _validate_result_local_evidence_binding(
        result,
        manifest,
        local_binding=local_binding,
    )


def _validate_outcome(
    outcome: WeavePublicationOutcomeV1,
    target: WeavePublicationTargetV1,
    attempt_ids: tuple[str, ...],
) -> None:
    if outcome.target != target:
        raise LocalResultPublicationError("publisher returned a different Weave target")
    expected = {
        (attempt_id, kind) for attempt_id in attempt_ids for kind in _HOSTED_KINDS
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


def _write_immutable_receipt(path: Path, receipt: WeavePublicationReceiptV1) -> None:
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
    required = {"schema_version", "entity", "project"}
    unknown = set(value) - required - {"study_scope", "destination"}
    missing = required - set(value)
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        raise ValueError("invalid Weave target: " + "; ".join(details))
    raw_scope = value.get("study_scope")
    scope = None
    if raw_scope is not None:
        scope_value = dict(_mapping(raw_scope, "Study publication scope"))
        _strict_fields(
            scope_value,
            {"schema_version", "research_id", "study_id"},
            "Study publication scope",
        )
        scope = StudyPublicationScopeV1(
            schema_version=_literal_one(
                scope_value["schema_version"], "Study publication scope"
            ),
            research_id=str(scope_value["research_id"]),
            study_id=str(scope_value["study_id"]),
        )
    raw_destination = value.get("destination")
    destination = (
        evidence_destination_from_dict(
            _mapping(raw_destination, "Weave publication evidence destination")
        )
        if raw_destination is not None
        else None
    )
    target = WeavePublicationTargetV1(
        schema_version=_literal_one(value["schema_version"], "target"),
        entity=str(value["entity"]),
        project=str(value["project"]),
        study_scope=scope,
        destination=destination,
    )
    if value != target.to_dict():
        raise ValueError("persisted Weave target must use its canonical values")
    return target


def _require_current_publication_destination(
    target: WeavePublicationTargetV1,
) -> EvidenceDestinationV1:
    if target.destination is None:
        raise LocalResultPublicationError(
            "new Weave publication requires an endpoint-bound evidence destination"
        )
    return target.destination


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
        raise LocalResultPublicationError(f"{label} contains secret-shaped content")


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
    if type(raw) is not int or raw != 1:
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
