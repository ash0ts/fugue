from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from fugue.bench.candidates import stable_digest
from fugue.bench.library import validate_id
from fugue.bench.task_authoring import TaskAuthoringLimitsV1

POST_TRIAL_VERIFIER_SOURCE_ROOT = Path(
    ".fugue/private/comparison-inputs/verifier"
)
POST_TRIAL_VERIFIER_RESOURCE_ROOT = Path(
    ".fugue/runtime/comparison-inputs/resources"
)
POST_TRIAL_VERIFIER_RUNTIME_ROOT = Path(
    ".fugue/private/comparison-inputs/verifier-runtime"
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_PLATFORMS = frozenset({"linux/amd64", "linux/arm64"})


class ScorerRuntimeProfile(Protocol):
    id: str
    profile_digest: str
    image: str
    platform: str


VerifierRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class PostTrialVerifierV1:
    """A trusted, frozen verifier that runs only after an Agent trial."""

    type: Literal["node_test", "skill_package"]
    source: str
    runtime_lock: str
    runtime: str
    dimension: str

    def __post_init__(self) -> None:
        if self.type not in {"node_test", "skill_package"}:
            raise ValueError(
                "post-trial verifier type must be node_test or skill_package"
            )
        _safe_relative_path(self.source, "post-trial verifier source")
        _safe_relative_path(self.runtime_lock, "post-trial verifier runtime lock")
        validate_id(self.runtime, kind="post-trial verifier runtime id")
        validate_id(self.dimension, kind="post-trial verifier dimension")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PostTrialVerifierLockV1:
    schema_version: int
    kind: Literal["post_trial_verifier_lock"]
    evaluator_id: str
    declaration_digest: str
    source_sha256: str
    runtime_lock_digest: str
    runtime_profile_id: str
    runtime_profile_digest: str
    runtime_image: str
    runtime_platform: str
    dimension: str
    dimension_role: Literal["outcome", "safety_gate"]
    dimension_digest: str
    lock_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PostTrialVerifierResultV1:
    passed: bool
    reason: str
    receipt: dict[str, Any] = field(default_factory=dict)

    def scorer_evidence(self) -> dict[str, Any]:
        """Return the bounded receipt a public deterministic scorer may inspect."""

        return {
            "host_verifier": dict(
                _mapping(
                    self.receipt.get("verifier_result"),
                    "post-trial verifier result",
                )
            )
        }


def post_trial_verifier_from_dict(raw: Any) -> PostTrialVerifierV1:
    value = _mapping(raw, "post-trial verifier")
    _reject_unknown(
        value,
        {"type", "source", "runtime_lock", "runtime", "dimension"},
        "verifier",
    )
    return PostTrialVerifierV1(
        type=str(value.get("type") or ""),  # type: ignore[arg-type]
        source=str(value.get("source") or ""),
        runtime_lock=str(value.get("runtime_lock") or ""),
        runtime=str(value.get("runtime") or ""),
        dimension=str(value.get("dimension") or ""),
    )


def resolve_post_trial_verifier_lock(
    verifier: PostTrialVerifierV1,
    *,
    evaluator_id: str,
    repo_root: Path,
    runtime_profile: ScorerRuntimeProfile,
    dimension_role: Literal["outcome", "safety_gate"],
) -> PostTrialVerifierLockV1:
    """Purely resolve source, runtime, and outcome dimension identity for preview."""

    evaluator = validate_id(evaluator_id, kind="deterministic evaluator id")
    if dimension_role not in {"outcome", "safety_gate"}:
        raise ValueError("post-trial verifier dimension must be outcome or safety_gate")
    _verify_runtime_profile(verifier, runtime_profile)
    source_path = _repository_file(
        repo_root,
        verifier.source,
        label="post-trial verifier source",
    )
    source = source_path.read_bytes()
    _validate_source(source)
    source_digest = hashlib.sha256(source).hexdigest()
    runtime_lock = _runtime_lock_value(verifier, repo_root=repo_root)
    runtime_lock_digest = stable_digest(runtime_lock)
    _verify_declared_runtime_lock(
        runtime_lock,
        source_sha256=source_digest,
        runtime_profile=runtime_profile,
    )
    unsigned = PostTrialVerifierLockV1(
        schema_version=1,
        kind="post_trial_verifier_lock",
        evaluator_id=evaluator,
        declaration_digest=stable_digest(verifier.to_dict()),
        source_sha256=source_digest,
        runtime_lock_digest=runtime_lock_digest,
        runtime_profile_id=runtime_profile.id,
        runtime_profile_digest=runtime_profile.profile_digest,
        runtime_image=runtime_profile.image,
        runtime_platform=runtime_profile.platform,
        dimension=verifier.dimension,
        dimension_role=dimension_role,
        dimension_digest=stable_digest(
            {"dimension": verifier.dimension, "role": dimension_role}
        ),
    )
    return replace(unsigned, lock_digest=_artifact_digest(unsigned.to_dict()))


def post_trial_verifier_lock_from_dict(raw: Any) -> PostTrialVerifierLockV1:
    value = _mapping(raw, "post-trial verifier lock")
    required = {
        "schema_version",
        "kind",
        "evaluator_id",
        "declaration_digest",
        "source_sha256",
        "runtime_lock_digest",
        "runtime_profile_id",
        "runtime_profile_digest",
        "runtime_image",
        "runtime_platform",
        "dimension",
        "dimension_role",
        "dimension_digest",
        "lock_digest",
    }
    if set(value) != required:
        raise ValueError("post-trial verifier lock fields do not match")
    lock = PostTrialVerifierLockV1(**value)  # type: ignore[arg-type]
    _verify_lock(lock)
    return lock


def prepare_post_trial_verifier(
    verifier: PostTrialVerifierV1,
    lock: PostTrialVerifierLockV1,
    *,
    repo_root: Path,
    runtime_profile: ScorerRuntimeProfile,
    dimension_role: Literal["outcome", "safety_gate"],
) -> Path:
    """Materialize the previewed verifier into ignored, immutable preparation state."""

    current = resolve_post_trial_verifier_lock(
        verifier,
        evaluator_id=lock.evaluator_id,
        repo_root=repo_root,
        runtime_profile=runtime_profile,
        dimension_role=dimension_role,
    )
    if current != lock:
        raise ValueError("post-trial verifier changed after its preview was approved")
    source = _repository_file(
        repo_root, verifier.source, label="post-trial verifier source"
    ).read_bytes()
    target = _prepared_source_path(repo_root, lock.source_sha256)
    _write_immutable(
        target,
        source,
        repo_root=repo_root,
        expected_sha256=lock.source_sha256,
    )
    runtime_lock = _runtime_lock_value(verifier, repo_root=repo_root)
    _write_immutable(
        _prepared_runtime_lock_path(repo_root, lock.runtime_lock_digest),
        _canonical_json(runtime_lock),
        repo_root=repo_root,
        expected_sha256=lock.runtime_lock_digest,
    )
    verify_prepared_post_trial_verifier(lock, repo_root=repo_root)
    return target


def verify_prepared_post_trial_verifier(
    lock: PostTrialVerifierLockV1,
    *,
    repo_root: Path,
) -> None:
    """Verify only the immutable, ignored assets selected by an approved lock."""

    _verify_lock(lock)
    source = _locked_file(
        _prepared_source_path(repo_root, lock.source_sha256),
        repo_root=repo_root,
        expected_sha256=lock.source_sha256,
        label="prepared post-trial verifier",
    ).read_bytes()
    _validate_source(source)
    runtime_path = _locked_file(
        _prepared_runtime_lock_path(repo_root, lock.runtime_lock_digest),
        repo_root=repo_root,
        expected_sha256=lock.runtime_lock_digest,
        label="prepared post-trial verifier runtime lock",
    )
    try:
        runtime_lock = _mapping(
            json.loads(runtime_path.read_text(encoding="utf-8")),
            "prepared post-trial verifier runtime lock",
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "prepared post-trial verifier runtime lock is invalid JSON"
        ) from exc
    _verify_runtime_lock_identity(
        runtime_lock,
        source_sha256=lock.source_sha256,
        runtime_image=lock.runtime_image,
    )


def run_post_trial_verifier(
    verifier: PostTrialVerifierV1,
    lock: PostTrialVerifierLockV1,
    *,
    evaluator_id: str,
    task: Mapping[str, Any],
    output: Any,
    expected: Mapping[str, Any],
    evidence: Mapping[str, Any],
    repo_root: Path,
    runtime_profile: ScorerRuntimeProfile,
    runner: VerifierRunner | None = None,
    prepared: bool = True,
) -> PostTrialVerifierResultV1:
    """Run a locked verifier with no trial network, credentials, or writable checkout."""

    _verify_lock(lock)
    if lock.evaluator_id != evaluator_id:
        raise ValueError("post-trial verifier lock belongs to another evaluator")
    _verify_declaration_lock(verifier, lock, runtime_profile)
    task_id = validate_id(str(task.get("id") or ""), kind="task id")
    attempt = _logical_attempt_id(evidence, task_id=task_id)
    source_path = (
        _prepared_source_path(repo_root, lock.source_sha256)
        if prepared
        else _repository_file(
            repo_root,
            verifier.source,
            label="post-trial verifier source",
        )
    )
    source = _locked_file(
        source_path,
        repo_root=repo_root,
        expected_sha256=lock.source_sha256,
        label="prepared post-trial verifier",
    ).read_bytes()
    _validate_source(source)
    archive, archive_digest = _base_archive(
        task,
        expected=expected,
        repo_root=repo_root,
        prepared=prepared,
    )
    runtime_lock_path = (
        _prepared_runtime_lock_path(repo_root, lock.runtime_lock_digest)
        if prepared
        else _repository_file(
            repo_root,
            verifier.runtime_lock,
            label="post-trial verifier runtime lock",
        )
    )
    runtime_lock_bytes = _locked_file(
        runtime_lock_path,
        repo_root=repo_root,
        expected_sha256=(lock.runtime_lock_digest if prepared else None),
        label=(
            "prepared post-trial verifier runtime lock"
            if prepared
            else "post-trial verifier runtime lock"
        ),
    ).read_bytes()
    runtime_lock = _mapping(
        json.loads(runtime_lock_bytes), "prepared post-trial verifier runtime lock"
    )
    if stable_digest(runtime_lock) != lock.runtime_lock_digest:
        raise ValueError("post-trial verifier runtime lock digest changed")
    _verify_declared_runtime_lock(
        runtime_lock,
        source_sha256=lock.source_sha256,
        runtime_profile=runtime_profile,
    )
    allowed_paths = _allowed_output_paths(expected)
    output_bytes = _canonical_json(output)
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    input_contract = {
        "schema_version": 1,
        "task_id": task_id,
        "task_archive": {
            "path": "/input/task.tar",
            "sha256": archive_digest,
        },
        "agent_output": {
            "path": "/input/agent-output.json",
            "sha256": output_sha256,
        },
        "runtime_lock_digest": lock.runtime_lock_digest,
        "workspace": "/work",
        "allowed_paths": list(allowed_paths),
    }
    files = {
        "verifier.cjs": source,
        "task.tar": archive,
        "agent-output.json": output_bytes,
        "input.json": _canonical_json(input_contract),
    }
    execute = runner or _default_runner()
    payload = execute(
        files=files,
        profile=runtime_profile,
        limits=_verifier_limits(),
        writable_workdir=True,
        accepted_exit_codes=(0, 1),
        output_kind="object",
    )
    if not isinstance(payload, Mapping):
        raise ValueError("post-trial verifier must return one JSON object")
    verifier_result = {
        key: item
        for key, item in payload.items()
        if key not in {"fugue_input_receipt", "fugue_runtime_receipt"}
    }
    _verify_verifier_result(
        verifier_result,
        task_id=task_id,
        task_archive_sha256=archive_digest,
        agent_output_sha256=output_sha256,
        output=output,
        allowed_paths=allowed_paths,
        runtime_lock_digest=lock.runtime_lock_digest,
        verifier_id=_runtime_lock_verifier_id(runtime_lock),
        command=_runtime_lock_command(runtime_lock),
    )
    input_receipt = _verify_input_receipt(
        payload.get("fugue_input_receipt"),
        files=files,
        runtime_profile=runtime_profile,
    )
    cleanup_receipt = _verify_cleanup_receipt(
        payload.get("fugue_runtime_receipt"), runtime_profile=runtime_profile
    )
    passed = verifier_result["status"] == "passed"
    unsigned = {
        "schema_version": 1,
        "kind": "post_trial_verifier_receipt",
        "evaluator_id": evaluator_id,
        "task_id": task_id,
        "logical_attempt_id": attempt,
        "status": "passed" if passed else "failed",
        "dimension": lock.dimension,
        "dimension_role": lock.dimension_role,
        "dimension_digest": lock.dimension_digest,
        "normalized_output_digest": stable_digest(output),
        "agent_output_sha256": output_sha256,
        "task_archive_sha256": archive_digest,
        "verifier_source_sha256": lock.source_sha256,
        "runtime_lock_digest": lock.runtime_lock_digest,
        "verifier_lock_digest": lock.lock_digest,
        "runtime_profile_id": lock.runtime_profile_id,
        "runtime_profile_digest": lock.runtime_profile_digest,
        "runtime_image": lock.runtime_image,
        "runtime_platform": lock.runtime_platform,
        "verifier_result": verifier_result,
        "verifier_result_digest": stable_digest(verifier_result),
        "runtime_input_receipt": input_receipt,
        "runtime_cleanup_receipt": cleanup_receipt,
    }
    receipt = {**unsigned, "receipt_digest": stable_digest(unsigned)}
    verify_post_trial_verifier_receipt(
        receipt,
        lock=lock,
        task_id=task_id,
        logical_attempt_id=attempt,
        normalized_output_digest=stable_digest(output),
        agent_output_sha256=output_sha256,
        task_archive_sha256=archive_digest,
    )
    return PostTrialVerifierResultV1(
        passed=passed,
        reason=(
            "frozen public test passed"
            if passed
            else "frozen public test failed"
        ),
        receipt=receipt,
    )


def verify_post_trial_verifier_receipt(
    raw: Any,
    *,
    lock: PostTrialVerifierLockV1,
    task_id: str,
    logical_attempt_id: str,
    normalized_output_digest: str,
    agent_output_sha256: str,
    task_archive_sha256: str,
) -> dict[str, Any]:
    value = _mapping(raw, "post-trial verifier receipt")
    required = {
        "schema_version",
        "kind",
        "evaluator_id",
        "task_id",
        "logical_attempt_id",
        "status",
        "dimension",
        "dimension_role",
        "dimension_digest",
        "normalized_output_digest",
        "agent_output_sha256",
        "task_archive_sha256",
        "verifier_source_sha256",
        "runtime_lock_digest",
        "verifier_lock_digest",
        "runtime_profile_id",
        "runtime_profile_digest",
        "runtime_image",
        "runtime_platform",
        "verifier_result",
        "verifier_result_digest",
        "runtime_input_receipt",
        "runtime_cleanup_receipt",
        "receipt_digest",
    }
    if set(value) != required:
        raise ValueError("post-trial verifier receipt fields do not match")
    unsigned = {key: item for key, item in value.items() if key != "receipt_digest"}
    identities_match = (
        value.get("schema_version") == 1
        and value.get("kind") == "post_trial_verifier_receipt"
        and value.get("evaluator_id") == lock.evaluator_id
        and value.get("task_id") == task_id
        and value.get("logical_attempt_id") == logical_attempt_id
        and value.get("status") in {"passed", "failed"}
        and value.get("dimension") == lock.dimension
        and value.get("dimension_role") == lock.dimension_role
        and value.get("dimension_digest") == lock.dimension_digest
        and value.get("normalized_output_digest") == normalized_output_digest
        and value.get("agent_output_sha256") == agent_output_sha256
        and value.get("task_archive_sha256") == task_archive_sha256
        and value.get("verifier_source_sha256") == lock.source_sha256
        and value.get("runtime_lock_digest") == lock.runtime_lock_digest
        and value.get("verifier_lock_digest") == lock.lock_digest
        and value.get("runtime_profile_id") == lock.runtime_profile_id
        and value.get("runtime_profile_digest") == lock.runtime_profile_digest
        and value.get("runtime_image") == lock.runtime_image
        and value.get("runtime_platform") == lock.runtime_platform
        and value.get("receipt_digest") == stable_digest(unsigned)
    )
    if not identities_match:
        raise ValueError("post-trial verifier receipt identity does not match")
    for name in (
        "logical_attempt_id",
        "dimension_digest",
        "normalized_output_digest",
        "agent_output_sha256",
        "task_archive_sha256",
        "verifier_source_sha256",
        "runtime_lock_digest",
        "verifier_lock_digest",
        "runtime_profile_digest",
        "verifier_result_digest",
    ):
        if not _DIGEST.fullmatch(str(value.get(name) or "")):
            raise ValueError(f"post-trial verifier receipt {name} is invalid")
    verifier_result = _mapping(
        value.get("verifier_result"), "post-trial verifier result"
    )
    if value.get("verifier_result_digest") != stable_digest(verifier_result):
        raise ValueError("post-trial verifier result digest does not match")
    if value.get("status") != verifier_result.get("status"):
        raise ValueError("post-trial verifier status does not match its result")
    _verify_cleanup_receipt(
        value.get("runtime_cleanup_receipt"), runtime_profile=lock
    )
    return value


def _verify_lock(lock: PostTrialVerifierLockV1) -> None:
    if (
        lock.schema_version != 1
        or lock.kind != "post_trial_verifier_lock"
        or lock.dimension_role not in {"outcome", "safety_gate"}
        or lock.lock_digest != _artifact_digest(lock.to_dict())
    ):
        raise ValueError("post-trial verifier lock digest does not match")
    validate_id(lock.evaluator_id, kind="deterministic evaluator id")
    validate_id(lock.runtime_profile_id, kind="post-trial verifier runtime id")
    validate_id(lock.dimension, kind="post-trial verifier dimension")
    for value in (
        lock.declaration_digest,
        lock.source_sha256,
        lock.runtime_lock_digest,
        lock.runtime_profile_digest,
        lock.dimension_digest,
        lock.lock_digest,
    ):
        if not _DIGEST.fullmatch(value):
            raise ValueError("post-trial verifier lock contains an invalid digest")
    if not _IMAGE.fullmatch(lock.runtime_image) or lock.runtime_platform not in _PLATFORMS:
        raise ValueError("post-trial verifier runtime lock is invalid")
    expected_dimension = stable_digest(
        {"dimension": lock.dimension, "role": lock.dimension_role}
    )
    if lock.dimension_digest != expected_dimension:
        raise ValueError("post-trial verifier outcome dimension digest changed")


def _verify_declaration_lock(
    verifier: PostTrialVerifierV1,
    lock: PostTrialVerifierLockV1,
    runtime_profile: ScorerRuntimeProfile,
) -> None:
    _verify_runtime_profile(verifier, runtime_profile)
    expected = {
        "declaration_digest": stable_digest(verifier.to_dict()),
        "runtime_profile_id": runtime_profile.id,
        "runtime_profile_digest": runtime_profile.profile_digest,
        "runtime_image": runtime_profile.image,
        "runtime_platform": runtime_profile.platform,
        "dimension": verifier.dimension,
    }
    observed = {name: getattr(lock, name) for name in expected}
    if observed != expected:
        raise ValueError("post-trial verifier declaration or runtime drifted")


def _verify_runtime_profile(
    verifier: PostTrialVerifierV1, profile: ScorerRuntimeProfile
) -> None:
    if profile.id != verifier.runtime:
        raise ValueError("post-trial verifier runtime profile does not match")
    if not _DIGEST.fullmatch(str(profile.profile_digest)):
        raise ValueError("post-trial verifier runtime profile is not digest locked")
    if not _IMAGE.fullmatch(str(profile.image)):
        raise ValueError("post-trial verifier runtime image is not digest pinned")
    if str(profile.platform) not in _PLATFORMS:
        raise ValueError("post-trial verifier runtime platform is unsupported")


def _base_archive(
    task: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    repo_root: Path,
    prepared: bool,
) -> tuple[bytes, str]:
    declared_digest = str(expected.get("task_archive_sha256") or "")
    attachments = task.get("attachments") or ()
    if attachments and (
        not isinstance(attachments, Sequence)
        or isinstance(attachments, str | bytes)
    ):
        raise ValueError("post-trial verifier task attachments must be a list")
    attachment_rows = [
        _mapping(raw, "post-trial verifier task attachment")
        for raw in attachments
    ]
    attachment_digests = {
        str(row.get("sha256") or "")
        for row in attachment_rows
        if _DIGEST.fullmatch(str(row.get("sha256") or ""))
    }
    if declared_digest:
        if not _DIGEST.fullmatch(declared_digest):
            raise ValueError("post-trial verifier base archive digest is invalid")
        if attachment_digests and declared_digest not in attachment_digests:
            raise ValueError(
                "post-trial verifier private archive digest disagrees with the public task"
            )
        candidate_digests = (declared_digest,)
    elif len(attachment_digests) == 1:
        candidate_digests = tuple(attachment_digests)
    else:
        candidate_digests = ()

    resources = task.get("resources") or ()
    if not isinstance(resources, Sequence) or isinstance(resources, str | bytes):
        raise ValueError("post-trial verifier task resources must be a list")
    source_rows = [
        _mapping(raw, "post-trial verifier task resource") for raw in resources
    ]
    if not candidate_digests and source_rows:
        source_digests = {
            hashlib.sha256(
                _repository_file(
                    repo_root,
                    _safe_relative_path(
                        str(row.get("path") or ""),
                        "post-trial verifier task resource",
                    ),
                    label="post-trial verifier task resource",
                ).read_bytes()
            ).hexdigest()
            for row in source_rows
        }
        if len(source_digests) == 1:
            candidate_digests = tuple(source_digests)
    if len(candidate_digests) != 1:
        raise ValueError("post-trial verifier base archive digest is unavailable")
    digest = candidate_digests[0]

    combined = [*attachment_rows, *source_rows]
    for resource in combined:
        if "locked_relative" in resource:
            relative = _safe_relative_path(
                str(resource.get("locked_relative") or ""),
                "post-trial verifier task attachment",
            )
        else:
            relative = _safe_relative_path(
                str(resource.get("path") or ""),
                "post-trial verifier task resource",
            )
        candidate = (
            _prepared_resource_path(repo_root, digest, Path(relative).name)
            if prepared
            else repo_root.resolve() / relative
        )
        try:
            path = _locked_file(
                candidate,
                repo_root=repo_root,
                expected_sha256=digest,
                label=(
                    "prepared post-trial verifier base archive"
                    if prepared
                    else "post-trial verifier source base archive"
                ),
            )
        except (FileNotFoundError, ValueError):
            continue
        content = path.read_bytes()
        if not content or len(content) > 64_000_000:
            raise ValueError("post-trial verifier base archive exceeds its bound")
        return content, digest
    boundary = "frozen" if prepared else "source"
    raise ValueError(
        f"post-trial verifier exact {boundary} base archive is unavailable"
    )


def _verify_input_receipt(
    raw: Any,
    *,
    files: Mapping[str, bytes],
    runtime_profile: ScorerRuntimeProfile,
) -> dict[str, Any]:
    value = _mapping(raw, "post-trial verifier input receipt")
    input_files = [
        {
            "path": name,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in sorted(files.items())
    ]
    unsigned = {
        "schema_version": 1,
        "kind": "isolated_evaluator_input",
        "status": "bound",
        "files": input_files,
        "files_digest": stable_digest(input_files),
        "runtime_profile_id": runtime_profile.id,
        "runtime_profile_digest": runtime_profile.profile_digest,
        "runtime_image": runtime_profile.image,
        "runtime_platform": runtime_profile.platform,
        "writable_workdir": True,
    }
    expected = {**unsigned, "receipt_digest": stable_digest(unsigned)}
    if value != expected:
        raise ValueError("post-trial verifier input receipt disagrees with invocation")
    return value


def _verify_verifier_result(
    value: Mapping[str, Any],
    *,
    task_id: str,
    task_archive_sha256: str,
    agent_output_sha256: str,
    output: Any,
    allowed_paths: tuple[str, ...],
    runtime_lock_digest: str,
    verifier_id: str,
    command: list[str],
) -> None:
    required = {
        "schema_version",
        "verifier_id",
        "task_id",
        "task_archive_sha256",
        "agent_output_sha256",
        "output_files_sha256",
        "allowed_paths_digest",
        "runtime_lock_digest",
        "observed_node_version",
        "command",
        "status",
        "exit_code",
        "stdout_sha256",
        "stderr_sha256",
        "receipt_digest",
    }
    if set(value) != required:
        raise ValueError("post-trial verifier result fields do not match")
    unsigned = {key: item for key, item in value.items() if key != "receipt_digest"}
    files = output.get("files") if isinstance(output, Mapping) else None
    file_digests = (
        {
            str(path): hashlib.sha256(content.encode()).hexdigest()
            for path, content in files.items()
            if isinstance(path, str) and isinstance(content, str)
        }
        if isinstance(files, Mapping)
        else {}
    )
    status = value.get("status")
    if (
        value.get("schema_version") != 1
        or value.get("verifier_id") != verifier_id
        or value.get("task_id") != task_id
        or value.get("task_archive_sha256") != task_archive_sha256
        or value.get("agent_output_sha256") != agent_output_sha256
        or value.get("output_files_sha256") != stable_digest(file_digests)
        or value.get("allowed_paths_digest") != stable_digest(list(allowed_paths))
        or value.get("runtime_lock_digest") != runtime_lock_digest
        or value.get("command") != command
        or status not in {"passed", "failed"}
        or (status == "passed") != (value.get("exit_code") == 0)
        or value.get("receipt_digest") != stable_digest(unsigned)
    ):
        raise ValueError("post-trial verifier result identity does not match")
    for name in (
        "task_archive_sha256",
        "agent_output_sha256",
        "output_files_sha256",
        "allowed_paths_digest",
        "runtime_lock_digest",
        "stdout_sha256",
        "stderr_sha256",
        "receipt_digest",
    ):
        if not _DIGEST.fullmatch(str(value.get(name) or "")):
            raise ValueError(f"post-trial verifier result {name} is invalid")


def _allowed_output_paths(expected: Mapping[str, Any]) -> tuple[str, ...]:
    raw = expected.get("allowed_paths")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or not raw:
        raise ValueError("post-trial verifier allowed paths are unavailable")
    paths = tuple(
        sorted(
            _safe_relative_path(str(item), "post-trial verifier allowed output")
            for item in raw
        )
    )
    if len(paths) != len(set(paths)):
        raise ValueError("post-trial verifier allowed paths must be unique")
    return paths


def _runtime_lock_value(
    verifier: PostTrialVerifierV1, *, repo_root: Path
) -> dict[str, Any]:
    path = _repository_file(
        repo_root,
        verifier.runtime_lock,
        label="post-trial verifier runtime lock",
    )
    try:
        return _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            "post-trial verifier runtime lock",
        )
    except json.JSONDecodeError as exc:
        raise ValueError("post-trial verifier runtime lock is invalid JSON") from exc


def _verify_declared_runtime_lock(
    value: Mapping[str, Any],
    *,
    source_sha256: str,
    runtime_profile: ScorerRuntimeProfile,
) -> None:
    _verify_runtime_lock_identity(
        value,
        source_sha256=source_sha256,
        runtime_image=runtime_profile.image,
    )


def _verify_runtime_lock_identity(
    value: Mapping[str, Any],
    *,
    source_sha256: str,
    runtime_image: str,
) -> None:
    runtime = _mapping(value.get("runtime"), "post-trial verifier locked runtime")
    _runtime_lock_verifier_id(value)
    _runtime_lock_command(value)
    if (
        value.get("schema_version") != 1
        or value.get("verifier_source_sha256") != source_sha256
        or runtime.get("kind") != "oci"
        or runtime.get("image") != runtime_image
        or runtime.get("network") != "none"
        or runtime.get("read_only_base") is not True
    ):
        raise ValueError("post-trial verifier runtime lock does not match its source")


def _runtime_lock_verifier_id(value: Mapping[str, Any]) -> str:
    return validate_id(
        str(value.get("id") or ""), kind="post-trial verifier implementation id"
    )


def _runtime_lock_command(value: Mapping[str, Any]) -> list[str]:
    raw = value.get("command")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, str | bytes)
        or not raw
        or len(raw) > 16
        or any(
            not isinstance(item, str) or not item.strip() or len(item) > 200
            for item in raw
        )
    ):
        raise ValueError("post-trial verifier locked command is invalid")
    return list(raw)


def _verify_cleanup_receipt(
    raw: Any, *, runtime_profile: ScorerRuntimeProfile | PostTrialVerifierLockV1
) -> dict[str, Any]:
    value = _mapping(raw, "post-trial verifier cleanup receipt")
    required = {
        "schema_version",
        "kind",
        "status",
        "container_name_sha256",
        "runtime_profile_id",
        "runtime_profile_digest",
        "runtime_image",
        "runtime_platform",
        "receipt_digest",
    }
    if set(value) != required:
        raise ValueError("post-trial verifier cleanup receipt fields do not match")
    unsigned = {key: item for key, item in value.items() if key != "receipt_digest"}
    profile_id = (
        runtime_profile.runtime_profile_id
        if isinstance(runtime_profile, PostTrialVerifierLockV1)
        else runtime_profile.id
    )
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "isolated_evaluator_cleanup"
        or value.get("status") != "verified_absent"
        or value.get("runtime_profile_id") != profile_id
    ):
        raise ValueError("post-trial verifier cleanup receipt identity does not match")
    profile_digest = (
        runtime_profile.runtime_profile_digest
        if isinstance(runtime_profile, PostTrialVerifierLockV1)
        else runtime_profile.profile_digest
    )
    image = (
        runtime_profile.runtime_image
        if isinstance(runtime_profile, PostTrialVerifierLockV1)
        else runtime_profile.image
    )
    platform = (
        runtime_profile.runtime_platform
        if isinstance(runtime_profile, PostTrialVerifierLockV1)
        else runtime_profile.platform
    )
    if (
        value.get("runtime_profile_digest") != profile_digest
        or value.get("runtime_image") != image
        or value.get("runtime_platform") != platform
        or value.get("receipt_digest") != stable_digest(unsigned)
        or not _DIGEST.fullmatch(str(value.get("container_name_sha256") or ""))
    ):
        raise ValueError("post-trial verifier cleanup receipt does not match runtime")
    return value


def _logical_attempt_id(evidence: Mapping[str, Any], *, task_id: str) -> str:
    attempt = str(evidence.get("attempt_id") or "")
    if not _DIGEST.fullmatch(attempt):
        raise ValueError("post-trial verifier logical attempt identity is unavailable")
    if str(evidence.get("task_id") or "") != task_id:
        raise ValueError("post-trial verifier evidence belongs to another task")
    return attempt


def _default_runner() -> VerifierRunner:
    from fugue.bench.task_authoring import run_isolated_evaluator

    return run_isolated_evaluator


def _verifier_limits() -> TaskAuthoringLimitsV1:
    return TaskAuthoringLimitsV1(
        max_tasks=1,
        max_scenarios=1,
        max_prompt_bytes=1,
        max_authored_asset_bytes=1,
        max_user_turns=1,
        max_agent_turns=1,
        max_interactor_calls=0,
        max_judge_calls=0,
        scorer_timeout_sec=30,
        scorer_memory_mb=256,
        scorer_cpus=1.0,
        scorer_output_bytes=64_000,
    )


def _prepared_source_path(repo_root: Path, digest: str) -> Path:
    return repo_root.resolve() / POST_TRIAL_VERIFIER_SOURCE_ROOT / f"{digest}.cjs"


def _prepared_runtime_lock_path(repo_root: Path, digest: str) -> Path:
    return repo_root.resolve() / POST_TRIAL_VERIFIER_RUNTIME_ROOT / f"{digest}.json"


def _prepared_resource_path(repo_root: Path, digest: str, name: str) -> Path:
    return repo_root.resolve() / POST_TRIAL_VERIFIER_RESOURCE_ROOT / digest / name


def _repository_file(repo_root: Path, relative: str, *, label: str) -> Path:
    safe = _safe_relative_path(relative, label)
    root = repo_root.resolve()
    selected = root / safe
    return _locked_file(selected, repo_root=root, label=label)


def _locked_file(
    path: Path,
    *,
    repo_root: Path,
    label: str,
    expected_sha256: str | None = None,
) -> Path:
    root = repo_root.resolve()
    _reject_symlink_components(path, root)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} not found: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    if expected_sha256 and hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"{label} digest changed")
    return resolved


def _write_immutable(
    path: Path,
    content: bytes,
    *,
    repo_root: Path,
    expected_sha256: str,
) -> None:
    root = repo_root.resolve()
    _reject_symlink_components(path, root)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _reject_symlink_components(path, root)
    if path.exists():
        _locked_file(
            path,
            repo_root=root,
            expected_sha256=expected_sha256,
            label="prepared post-trial verifier",
        )
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
        temporary.chmod(0o400)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_symlink_components(path: Path, root: Path) -> None:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise ValueError("prepared verifier path must be inside the repository") from exc
    current = root.absolute()
    for part in relative.parts:
        current /= part
        if current.exists() and current.is_symlink():
            raise ValueError("prepared verifier path must not contain symlinks")


def _safe_relative_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a safe repository-relative file")
    return path.as_posix()


def _validate_source(source: bytes) -> None:
    if not source or len(source) > 64_000 or b"\0" in source:
        raise ValueError("post-trial verifier source must be non-empty and at most 64 KiB")
    try:
        source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("post-trial verifier source must be UTF-8") from exc


def _artifact_digest(value: Mapping[str, Any]) -> str:
    return stable_digest({**value, "lock_digest": ""})


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], known: set[str], label: str) -> None:
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")
