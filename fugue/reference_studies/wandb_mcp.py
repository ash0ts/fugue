from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from fugue.bench.files import atomic_write_json

WANDB_MCP_REPOSITORY_URL = "https://github.com/wandb/wandb-mcp-server.git"
WANDB_MCP_RELEASE_REF = "refs/heads/staging/0.4.0"
WANDB_MCP_RELEASE_NOTES_PATH = "docs/releases/v0.4.0.md"
WANDB_MCP_REFERENCE_ROOT = (
    Path(".fugue") / "reference-studies" / "wandb-mcp"
)
SOURCE_LOCK_NAME = "source.lock.json"
PREPARATION_RECEIPT_NAME = "preparation.receipt.json"
HUMAN_READABLE_COMPARISON_NAME = "human-readable-evidence-canary.yaml"
HUMAN_READABLE_TASKS_NAME = "human-readable-evidence-tasks.jsonl"
HUMAN_READABLE_PRIVATE_LABELS_NAME = (
    "human-readable-evidence-private-labels.jsonl"
)
HUMAN_READABLE_CANARY_LOCK_NAME = "human-readable-evidence-canary.lock.json"

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SECRET_ENV_NAME = re.compile(
    r"(?:api.?key|credential|password|secret|token)", re.IGNORECASE
)
_MAX_RELEASE_NOTES_BYTES = 2 * 1024 * 1024
_DEFAULT_GIT_TIMEOUT_SECONDS = 60
_FETCH_GIT_TIMEOUT_SECONDS = 180
_REFERENCE_RESOURCE_ROOT = ("resources", "reference-studies", "wandb-mcp")
_REFERENCE_RESOURCE_NAMES = (
    "README.md",
    "comparison.yaml.template",
    "human-readable-evidence-canary.yaml.template",
    "configs/fugue/task-authoring/profiles.yaml",
    "mcp.json.template",
    "private-labels.jsonl",
    "release-contract-v1.json",
    "tasks.jsonl",
    "tool_surface_scorer_v7.py",
    "wbaf-task-design-provenance-v1.json",
)
_BASELINE_COMMIT = "53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0"
_SOURCE_PROJECT = "wandb/fugue-mcp-release-source-v2"
_HUMAN_READABLE_TASK_IDS = (
    "run-inventory-projection",
    "evaluation-summary-accuracy",
)
_PACKAGED_RELEASE_NOTE_COVERAGE = (
    {
        "release_note": "selective-server-side-fields",
        "status": "unqualified",
        "task_ids": ["run-inventory-projection"],
        "dimensions": [
            "tool-surface.answer_correct",
            "tool-surface.bounded_evidence",
        ],
        "infrastructure_gates": [],
        "rationale": (
            "The locked Run-inventory task assesses selective fields. "
            "The canary does not qualify this release-note behavior by itself."
        ),
    },
    {
        "release_note": "cursor-continuation-pagination",
        "status": "unqualified",
        "task_ids": ["filtered-failure-triage"],
        "dimensions": [
            "tool-surface.answer_correct",
            "tool-surface.bounded_evidence",
        ],
        "infrastructure_gates": [],
        "rationale": (
            "The locked failure-triage task assesses bounded continuation. "
            "The canary does not qualify this release-note behavior by itself."
        ),
    },
    {
        "release_note": "bounded-history",
        "status": "unqualified",
        "task_ids": ["exact-history-target"],
        "dimensions": [
            "tool-surface.answer_correct",
            "tool-surface.bounded_evidence",
            "tool-surface.evidence_honesty",
        ],
        "infrastructure_gates": [],
        "rationale": (
            "The locked history task assesses exact-axis targeting and bounds. "
            "The canary does not qualify this release-note behavior by itself."
        ),
    },
    {
        "release_note": "evaluation-prediction-reconciliation",
        "status": "unqualified",
        "task_ids": ["evaluation-summary-accuracy"],
        "dimensions": [
            "tool-surface.answer_correct",
            "tool-surface.target_behavior_satisfied",
        ],
        "infrastructure_gates": [],
        "rationale": (
            "The locked Evaluation task assesses direct prediction-child "
            "reconciliation. The canary does not qualify this release-note "
            "behavior by itself."
        ),
    },
)


def _stable_digest(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_exact_fields(
    value: Mapping[str, Any],
    fields: frozenset[str],
    *,
    artifact: str,
) -> None:
    unknown = set(value) - fields
    missing = fields - set(value)
    if not unknown and not missing:
        return
    details: list[str] = []
    if unknown:
        details.append("unknown=" + ",".join(sorted(unknown)))
    if missing:
        details.append("missing=" + ",".join(sorted(missing)))
    raise ValueError(f"invalid {artifact} fields: {'; '.join(details)}")


def _require_version(value: object, *, artifact: str) -> None:
    if type(value) is not int or value != 1:
        raise ValueError(f"unsupported {artifact} schema version")


def _require_sha(value: str, *, label: str, length: int) -> str:
    expression = _HEX_40 if length == 40 else _HEX_64
    if not expression.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase {length}-character hex digest")
    return value


def _safe_relative_path(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return path.as_posix()


@dataclass(frozen=True)
class ImmutableGitFileV1:
    schema_version: int
    path: str
    git_blob: str
    sha256: str
    byte_count: int | None

    def __post_init__(self) -> None:
        _require_version(self.schema_version, artifact="immutable git file")
        _safe_relative_path(self.path, label="immutable git file path")
        _require_sha(self.git_blob, label="immutable git file blob", length=40)
        _require_sha(self.sha256, label="immutable git file SHA-256", length=64)
        if self.byte_count is not None and (
            type(self.byte_count) is not int or self.byte_count < 0
        ):
            raise ValueError("immutable git file byte_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ImmutableGitFileV1:
        _require_exact_fields(
            value,
            frozenset({"schema_version", "path", "git_blob", "sha256", "byte_count"}),
            artifact="immutable git file",
        )
        byte_count = value["byte_count"]
        if byte_count is not None and type(byte_count) is not int:
            raise ValueError("immutable git file byte_count must be an integer or null")
        return cls(
            schema_version=value["schema_version"],
            path=_strict_string(value["path"], label="immutable git file path"),
            git_blob=_strict_string(
                value["git_blob"], label="immutable git file blob"
            ),
            sha256=_strict_string(
                value["sha256"], label="immutable git file SHA-256"
            ),
            byte_count=byte_count,
        )


@dataclass(frozen=True)
class ReviewedTaskDesignProvenanceV1:
    schema_version: int
    repository_url: str
    source_commit: str
    source_tree: str
    role: str
    runtime_dependency: bool
    files: tuple[ImmutableGitFileV1, ...]
    provenance_digest: str

    def __post_init__(self) -> None:
        _require_version(self.schema_version, artifact="task-design provenance")
        if self.repository_url != "https://github.com/wandb/WandBAgentFactory.git":
            raise ValueError("task-design provenance repository is not allowlisted")
        _require_sha(self.source_commit, label="provenance commit", length=40)
        _require_sha(self.source_tree, label="provenance tree", length=40)
        if self.role != "task_design_reference":
            raise ValueError("task-design provenance role is invalid")
        if type(self.runtime_dependency) is not bool or self.runtime_dependency:
            raise ValueError("WBAF provenance must not be a runtime dependency")
        if not self.files or tuple(sorted(item.path for item in self.files)) != tuple(
            item.path for item in self.files
        ):
            raise ValueError("task-design provenance files must be non-empty and sorted")
        _require_sha(
            self.provenance_digest,
            label="task-design provenance digest",
            length=64,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["files"] = [item.to_dict() for item in self.files]
        return value

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> ReviewedTaskDesignProvenanceV1:
        _require_exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "repository_url",
                    "source_commit",
                    "source_tree",
                    "role",
                    "runtime_dependency",
                    "files",
                    "provenance_digest",
                }
            ),
            artifact="task-design provenance",
        )
        files_raw = value["files"]
        if not isinstance(files_raw, list) or not all(
            isinstance(item, dict) for item in files_raw
        ):
            raise ValueError("task-design provenance files must be a list of objects")
        result = cls(
            schema_version=value["schema_version"],
            repository_url=_strict_string(
                value["repository_url"], label="provenance repository"
            ),
            source_commit=_strict_string(
                value["source_commit"], label="provenance commit"
            ),
            source_tree=_strict_string(
                value["source_tree"], label="provenance tree"
            ),
            role=_strict_string(value["role"], label="provenance role"),
            runtime_dependency=_strict_bool(
                value["runtime_dependency"], label="runtime_dependency"
            ),
            files=tuple(ImmutableGitFileV1.from_dict(item) for item in files_raw),
            provenance_digest=_strict_string(
                value["provenance_digest"], label="provenance digest"
            ),
        )
        expected = _provenance_digest(result)
        if expected != result.provenance_digest:
            raise ValueError("task-design provenance digest does not match its content")
        return result


def _provenance_digest(value: ReviewedTaskDesignProvenanceV1) -> str:
    payload = value.to_dict()
    payload.pop("provenance_digest")
    return _stable_digest(payload)


def _build_wbaf_provenance() -> ReviewedTaskDesignProvenanceV1:
    base = ReviewedTaskDesignProvenanceV1(
        schema_version=1,
        repository_url="https://github.com/wandb/WandBAgentFactory.git",
        source_commit="e2d8d670017bc426b68a311c5777c3b9084023f3",
        source_tree="7776ac62bbe32da5c6824809893d70ea6725a42e",
        role="task_design_reference",
        runtime_dependency=False,
        files=(
            ImmutableGitFileV1(
                schema_version=1,
                path="data/evals/mcp-all.yaml",
                git_blob="6165211befa7b5af60468f41b195d29c7c29a7ed",
                sha256="d0dc3ea830cb9ccb2e5d57bbef54712f46e335ca731bb873072201c43a305624",
                byte_count=1016,
            ),
            ImmutableGitFileV1(
                schema_version=1,
                path="data/evals/mcp-ci.yaml",
                git_blob="201a0f0cb1b77845378e834e4e2db08b89570d2d",
                sha256="777985c511d405795e93b2df75ba6c6d6f7d723e6a43123b0d483fa84f91ba6e",
                byte_count=471,
            ),
            ImmutableGitFileV1(
                schema_version=1,
                path="docs/tasks.md",
                git_blob="52bf9f093503ae3ab022942302a54d5d1df142a7",
                sha256="133bd8924378dede4ba73a30a9cd5298a9beb741f6d8bdd9e57e8a4939b59033",
                byte_count=8985,
            ),
        ),
        provenance_digest="0" * 64,
    )
    return replace(base, provenance_digest=_provenance_digest(base))


WBAF_TASK_DESIGN_PROVENANCE = _build_wbaf_provenance()


@dataclass(frozen=True)
class WandbMCPReferenceSourceLockV1:
    schema_version: int
    kind: str
    repository_url: str
    requested_ref: str
    source_commit: str
    source_tree: str
    release_notes: ImmutableGitFileV1
    task_design_provenance: tuple[ReviewedTaskDesignProvenanceV1, ...]
    candidate_source_digest: str
    lock_digest: str

    def __post_init__(self) -> None:
        _require_version(self.schema_version, artifact="W&B MCP reference lock")
        if self.kind != "wandb-mcp-reference-source-lock":
            raise ValueError("W&B MCP reference lock kind is invalid")
        if self.repository_url != WANDB_MCP_REPOSITORY_URL:
            raise ValueError("W&B MCP reference repository is not allowlisted")
        if self.requested_ref != WANDB_MCP_RELEASE_REF:
            raise ValueError("W&B MCP reference ref must be staging/0.4.0")
        _require_sha(self.source_commit, label="W&B MCP source commit", length=40)
        _require_sha(self.source_tree, label="W&B MCP source tree", length=40)
        if self.release_notes.path != WANDB_MCP_RELEASE_NOTES_PATH:
            raise ValueError("W&B MCP release-note path is invalid")
        if self.release_notes.byte_count is None or self.release_notes.byte_count == 0:
            raise ValueError("W&B MCP release notes must be non-empty")
        if self.task_design_provenance != (WBAF_TASK_DESIGN_PROVENANCE,):
            raise ValueError("W&B MCP task-design provenance is not the reviewed lock")
        _require_sha(
            self.candidate_source_digest,
            label="candidate source digest",
            length=64,
        )
        _require_sha(self.lock_digest, label="reference lock digest", length=64)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["release_notes"] = self.release_notes.to_dict()
        value["task_design_provenance"] = [
            item.to_dict() for item in self.task_design_provenance
        ]
        return value

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> WandbMCPReferenceSourceLockV1:
        _require_exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "repository_url",
                    "requested_ref",
                    "source_commit",
                    "source_tree",
                    "release_notes",
                    "task_design_provenance",
                    "candidate_source_digest",
                    "lock_digest",
                }
            ),
            artifact="W&B MCP reference lock",
        )
        release_raw = value["release_notes"]
        provenance_raw = value["task_design_provenance"]
        if not isinstance(release_raw, dict):
            raise ValueError("reference release_notes must be an object")
        if not isinstance(provenance_raw, list) or not all(
            isinstance(item, dict) for item in provenance_raw
        ):
            raise ValueError("reference task_design_provenance must be a list")
        result = cls(
            schema_version=value["schema_version"],
            kind=_strict_string(value["kind"], label="reference lock kind"),
            repository_url=_strict_string(
                value["repository_url"], label="reference repository"
            ),
            requested_ref=_strict_string(
                value["requested_ref"], label="reference ref"
            ),
            source_commit=_strict_string(
                value["source_commit"], label="reference source commit"
            ),
            source_tree=_strict_string(
                value["source_tree"], label="reference source tree"
            ),
            release_notes=ImmutableGitFileV1.from_dict(release_raw),
            task_design_provenance=tuple(
                ReviewedTaskDesignProvenanceV1.from_dict(item)
                for item in provenance_raw
            ),
            candidate_source_digest=_strict_string(
                value["candidate_source_digest"], label="candidate source digest"
            ),
            lock_digest=_strict_string(
                value["lock_digest"], label="reference lock digest"
            ),
        )
        expected_source = _candidate_source_digest(result)
        if expected_source != result.candidate_source_digest:
            raise ValueError("candidate source digest does not match source content")
        expected_lock = _reference_lock_digest(result)
        if expected_lock != result.lock_digest:
            raise ValueError("reference lock digest does not match its content")
        return result


def _candidate_source_digest(value: WandbMCPReferenceSourceLockV1) -> str:
    return _stable_digest(
        {
            "schema_version": 1,
            "repository_url": value.repository_url,
            "source_commit": value.source_commit,
            "source_tree": value.source_tree,
            "release_notes": value.release_notes.to_dict(),
        }
    )


def _reference_lock_digest(value: WandbMCPReferenceSourceLockV1) -> str:
    payload = value.to_dict()
    payload.pop("lock_digest")
    return _stable_digest(payload)


@dataclass(frozen=True)
class MaterializedReferenceArtifactV1:
    schema_version: int
    path: str
    sha256: str
    byte_count: int
    executable: bool = False
    private: bool = False

    def __post_init__(self) -> None:
        _require_version(self.schema_version, artifact="reference materialized artifact")
        _safe_relative_path(self.path, label="materialized artifact path")
        _require_sha(self.sha256, label="materialized artifact SHA-256", length=64)
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise ValueError("materialized artifact byte_count must be non-negative")
        if type(self.executable) is not bool:
            raise ValueError("materialized artifact executable must be boolean")
        if type(self.private) is not bool:
            raise ValueError("materialized artifact private must be boolean")
        if self.private and self.executable:
            raise ValueError("a private reference artifact may not be executable")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> MaterializedReferenceArtifactV1:
        _require_exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "path",
                    "sha256",
                    "byte_count",
                    "executable",
                    "private",
                }
            ),
            artifact="reference materialized artifact",
        )
        return cls(
            schema_version=value["schema_version"],
            path=_strict_string(value["path"], label="materialized artifact path"),
            sha256=_strict_string(
                value["sha256"], label="materialized artifact SHA-256"
            ),
            byte_count=_strict_int(
                value["byte_count"], label="materialized artifact byte_count"
            ),
            executable=_strict_bool(
                value["executable"], label="materialized artifact executable"
            ),
            private=_strict_bool(
                value["private"], label="materialized artifact private"
            ),
        )


@dataclass(frozen=True)
class ReferenceStudyMaterializationV1:
    """Prepared study inputs with behavior and execution identities separated.

    These are component-input digests, not reconstructed Fugue candidate IDs.
    The ordinary comparison resolver remains the sole owner of canonical
    candidate identity.
    """

    schema_version: int
    study_bundle_id: str
    behavior_inputs_digest: str
    execution_inputs_digest: str
    inventory_digest: str
    total_files: int
    total_bytes: int
    artifacts: tuple[MaterializedReferenceArtifactV1, ...]

    def __post_init__(self) -> None:
        _require_version(self.schema_version, artifact="reference materialization")
        if not _SAFE_ID.fullmatch(self.study_bundle_id):
            raise ValueError("reference materialization study_bundle_id is invalid")
        _require_sha(
            self.behavior_inputs_digest,
            label="behavior inputs digest",
            length=64,
        )
        _require_sha(
            self.execution_inputs_digest,
            label="execution inputs digest",
            length=64,
        )
        _require_sha(
            self.inventory_digest,
            label="reference materialization inventory digest",
            length=64,
        )
        if type(self.total_files) is not int or self.total_files < 1:
            raise ValueError("reference materialization total_files must be positive")
        if type(self.total_bytes) is not int or self.total_bytes < 1:
            raise ValueError("reference materialization total_bytes must be positive")
        paths = tuple(item.path for item in self.artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("materialized artifacts must have unique sorted paths")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifacts"] = [item.to_dict() for item in self.artifacts]
        return value

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> ReferenceStudyMaterializationV1:
        _require_exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "study_bundle_id",
                    "behavior_inputs_digest",
                    "execution_inputs_digest",
                    "inventory_digest",
                    "total_files",
                    "total_bytes",
                    "artifacts",
                }
            ),
            artifact="reference materialization",
        )
        artifacts_raw = value["artifacts"]
        if not isinstance(artifacts_raw, list) or not all(
            isinstance(item, dict) for item in artifacts_raw
        ):
            raise ValueError("reference materialization artifacts must be a list")
        return cls(
            schema_version=value["schema_version"],
            study_bundle_id=_strict_string(
                value["study_bundle_id"], label="materialized study bundle id"
            ),
            behavior_inputs_digest=_strict_string(
                value["behavior_inputs_digest"],
                label="behavior inputs digest",
            ),
            execution_inputs_digest=_strict_string(
                value["execution_inputs_digest"], label="execution inputs digest"
            ),
            inventory_digest=_strict_string(
                value["inventory_digest"], label="materialization inventory digest"
            ),
            total_files=_strict_int(
                value["total_files"], label="materialization total_files"
            ),
            total_bytes=_strict_int(
                value["total_bytes"], label="materialization total_bytes"
            ),
            artifacts=tuple(
                MaterializedReferenceArtifactV1.from_dict(item)
                for item in artifacts_raw
            ),
        )


@dataclass(frozen=True)
class WandbMCPReferencePreparationReceiptV1:
    schema_version: int
    kind: str
    source_lock_digest: str
    candidate_source_digest: str
    source_commit: str
    first_observed_commit: str
    second_observed_commit: str
    destination: str
    materialization: ReferenceStudyMaterializationV1
    receipt_digest: str

    def __post_init__(self) -> None:
        _require_version(self.schema_version, artifact="reference preparation receipt")
        if self.kind != "wandb-mcp-reference-preparation":
            raise ValueError("reference preparation receipt kind is invalid")
        for label, value in (
            ("source lock digest", self.source_lock_digest),
            ("candidate source digest", self.candidate_source_digest),
            ("receipt digest", self.receipt_digest),
        ):
            _require_sha(value, label=label, length=64)
        for label, value in (
            ("source commit", self.source_commit),
            ("first observed commit", self.first_observed_commit),
            ("second observed commit", self.second_observed_commit),
        ):
            _require_sha(value, label=label, length=40)
        if not (
            self.source_commit
            == self.first_observed_commit
            == self.second_observed_commit
        ):
            raise ValueError("reference preparation observations do not reconcile")
        expected_destination = (
            WANDB_MCP_REFERENCE_ROOT / self.source_commit
        ).as_posix()
        if self.destination != expected_destination:
            raise ValueError("reference preparation destination is not canonical")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["materialization"] = self.materialization.to_dict()
        return value

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> WandbMCPReferencePreparationReceiptV1:
        _require_exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "source_lock_digest",
                    "candidate_source_digest",
                    "source_commit",
                    "first_observed_commit",
                    "second_observed_commit",
                    "destination",
                    "materialization",
                    "receipt_digest",
                }
            ),
            artifact="reference preparation receipt",
        )
        raw_materialization = value["materialization"]
        if not isinstance(raw_materialization, dict):
            raise ValueError("reference receipt materialization must be an object")
        result = cls(
            schema_version=value["schema_version"],
            kind=_strict_string(value["kind"], label="reference receipt kind"),
            source_lock_digest=_strict_string(
                value["source_lock_digest"], label="source lock digest"
            ),
            candidate_source_digest=_strict_string(
                value["candidate_source_digest"], label="candidate source digest"
            ),
            source_commit=_strict_string(
                value["source_commit"], label="source commit"
            ),
            first_observed_commit=_strict_string(
                value["first_observed_commit"], label="first observed commit"
            ),
            second_observed_commit=_strict_string(
                value["second_observed_commit"], label="second observed commit"
            ),
            destination=_strict_string(value["destination"], label="destination"),
            materialization=ReferenceStudyMaterializationV1.from_dict(
                raw_materialization
            ),
            receipt_digest=_strict_string(
                value["receipt_digest"], label="receipt digest"
            ),
        )
        if _receipt_digest(result) != result.receipt_digest:
            raise ValueError("reference preparation receipt digest does not match")
        return result


def _receipt_digest(value: WandbMCPReferencePreparationReceiptV1) -> str:
    payload = value.to_dict()
    payload.pop("receipt_digest")
    return _stable_digest(payload)


@dataclass(frozen=True)
class GitSourceSnapshotV1:
    source_commit: str
    source_tree: str
    release_notes_blob: str
    release_notes: bytes


class GitReferenceTransport(Protocol):
    def observe_ref(
        self, repository_url: str, requested_ref: str, *, cwd: Path
    ) -> str: ...

    def fetch_snapshot(
        self,
        repository_url: str,
        source_commit: str,
        *,
        bare_repository: Path,
    ) -> GitSourceSnapshotV1: ...


class SubprocessGitReferenceTransport:
    """Non-interactive, bounded git transport for the trusted preparation step."""

    def __init__(
        self,
        *,
        timeout_seconds: int = _DEFAULT_GIT_TIMEOUT_SECONDS,
        fetch_timeout_seconds: int = _FETCH_GIT_TIMEOUT_SECONDS,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        if timeout_seconds <= 0 or fetch_timeout_seconds <= 0:
            raise ValueError("git timeouts must be positive")
        self._timeout_seconds = timeout_seconds
        self._fetch_timeout_seconds = fetch_timeout_seconds
        self._runner = runner

    def observe_ref(
        self, repository_url: str, requested_ref: str, *, cwd: Path
    ) -> str:
        _validate_reference_coordinates(repository_url, requested_ref)
        result = self._run(
            [
                "ls-remote",
                "--exit-code",
                "--refs",
                repository_url,
                requested_ref,
            ],
            cwd=cwd,
        )
        lines = result.stdout.decode("utf-8", errors="strict").splitlines()
        expected_suffix = f"\t{requested_ref}"
        if len(lines) != 1 or not lines[0].endswith(expected_suffix):
            raise RuntimeError("git ls-remote returned an ambiguous release ref")
        commit = lines[0][: -len(expected_suffix)]
        return _require_sha(commit, label="observed release commit", length=40)

    def fetch_snapshot(
        self,
        repository_url: str,
        source_commit: str,
        *,
        bare_repository: Path,
    ) -> GitSourceSnapshotV1:
        _validate_reference_coordinates(repository_url, WANDB_MCP_RELEASE_REF)
        _require_sha(source_commit, label="requested source commit", length=40)
        bare_repository.mkdir(mode=0o700, parents=True, exist_ok=False)
        self._run(["init", "--bare", "."], cwd=bare_repository)
        self._run(
            ["fetch", "--no-tags", "--depth=1", repository_url, source_commit],
            cwd=bare_repository,
            timeout=self._fetch_timeout_seconds,
        )
        resolved = self._text(
            ["rev-parse", "--verify", f"{source_commit}^{{commit}}"],
            cwd=bare_repository,
        )
        if resolved != source_commit:
            raise RuntimeError("fetched commit does not match the observed ref")
        source_tree = _require_sha(
            self._text(
                ["rev-parse", "--verify", f"{source_commit}^{{tree}}"],
                cwd=bare_repository,
            ),
            label="fetched source tree",
            length=40,
        )
        blob = _require_sha(
            self._text(
                [
                    "rev-parse",
                    "--verify",
                    f"{source_commit}:{WANDB_MCP_RELEASE_NOTES_PATH}",
                ],
                cwd=bare_repository,
            ),
            label="release-note blob",
            length=40,
        )
        object_type = self._text(["cat-file", "-t", blob], cwd=bare_repository)
        if object_type != "blob":
            raise RuntimeError("release-note object is not a git blob")
        size_text = self._text(["cat-file", "-s", blob], cwd=bare_repository)
        try:
            size = int(size_text)
        except ValueError as exc:
            raise RuntimeError("release-note blob size is invalid") from exc
        if size <= 0 or size > _MAX_RELEASE_NOTES_BYTES:
            raise RuntimeError("release-note blob is empty or exceeds the size limit")
        release_notes = self._run(
            ["cat-file", "blob", blob], cwd=bare_repository
        ).stdout
        if len(release_notes) != size:
            raise RuntimeError("release-note byte count changed while reading")
        try:
            release_notes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError("release notes are not valid UTF-8") from exc
        return GitSourceSnapshotV1(
            source_commit=resolved,
            source_tree=source_tree,
            release_notes_blob=blob,
            release_notes=release_notes,
        )

    def _text(self, arguments: Sequence[str], *, cwd: Path) -> str:
        return self._run(arguments, cwd=cwd).stdout.decode(
            "utf-8", errors="strict"
        ).strip()

    def _run(
        self,
        arguments: Sequence[str],
        *,
        cwd: Path,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        command = [
            "git",
            "-c",
            "credential.helper=",
            "-c",
            "protocol.file.allow=never",
            "-c",
            "core.hooksPath=/dev/null",
            *arguments,
        ]
        try:
            result = self._runner(
                command,
                cwd=cwd,
                env=_git_environment(),
                capture_output=True,
                timeout=timeout or self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("git reference preparation timed out") from exc
        except OSError as exc:
            raise RuntimeError("git is unavailable for reference preparation") from exc
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                "git reference preparation failed"
                + (f": {detail[:500]}" if detail else "")
            )
        return result


@dataclass(frozen=True)
class ReferenceMaterializationRequest:
    staging_root: Path
    source_lock: WandbMCPReferenceSourceLockV1
    bare_repository: Path
    env_file: Path | None
    target_platform: str | None


ReferenceMaterializer = Callable[
    [ReferenceMaterializationRequest], ReferenceStudyMaterializationV1
]


def _materialize_packaged_reference_study(
    request: ReferenceMaterializationRequest,
) -> ReferenceStudyMaterializationV1:
    """Render and lock the installed W&B MCP reference into the staging root."""

    if request.target_platform is None:
        raise ValueError("W&B MCP reference preparation requires a target platform")
    resources = _load_packaged_reference_resources()
    _validate_packaged_wbaf_provenance(
        json.loads(resources["wbaf-task-design-provenance-v1.json"])
    )
    _validate_packaged_release_contract(
        json.loads(resources["release-contract-v1.json"])
    )

    candidate_sha = request.source_lock.source_commit
    baseline_id = f"wandb-mcp-main-{_BASELINE_COMMIT[:12]}"
    candidate_id = f"wandb-mcp-staging-{candidate_sha[:12]}"
    replacements = {
        "{{CANDIDATE_SHA}}": candidate_sha,
        "{{CANDIDATE_SHORT}}": candidate_sha[:7],
        "{{CANDIDATE_TREE}}": request.source_lock.source_tree,
        "{{FUGUE_SCORER_PLATFORM}}": request.target_platform,
        "{{TARGET_PLATFORM}}": request.target_platform,
        "{{MCP_BASELINE_LOCK_ID}}": baseline_id,
        "{{MCP_CANDIDATE_LOCK_ID}}": candidate_id,
    }
    mcp_config = _render_resource(
        resources["mcp.json.template"], replacements, name="mcp.json.template"
    )
    _write_prepared_file(request.staging_root / "mcp.json", mcp_config)
    scorer_profiles = _render_resource(
        resources["configs/fugue/task-authoring/profiles.yaml"],
        replacements,
        name="configs/fugue/task-authoring/profiles.yaml",
    )
    _write_prepared_file(
        request.staging_root / "configs/fugue/task-authoring/profiles.yaml",
        scorer_profiles,
    )
    for resource_name, destination_name in (
        ("README.md", "README.md"),
        ("tasks.jsonl", "tasks.jsonl"),
        ("private-labels.jsonl", "private-labels.jsonl"),
        ("release-contract-v1.json", "release-contract-v1.json"),
        ("tool_surface_scorer_v7.py", "tool_surface_scorer_v7.py"),
        (
            "wbaf-task-design-provenance-v1.json",
            "wbaf-task-design-provenance-v1.json",
        ),
    ):
        _write_prepared_file(
            request.staging_root / destination_name,
            resources[resource_name],
            mode=0o600 if destination_name == "private-labels.jsonl" else 0o644,
        )

    human_tasks, human_private_labels = _select_aligned_reference_rows(
        public_body=resources["tasks.jsonl"],
        private_body=resources["private-labels.jsonl"],
        selected_ids=_HUMAN_READABLE_TASK_IDS,
    )
    _write_prepared_file(
        request.staging_root / HUMAN_READABLE_TASKS_NAME,
        human_tasks,
    )
    _write_prepared_file(
        request.staging_root / HUMAN_READABLE_PRIVATE_LABELS_NAME,
        human_private_labels,
        mode=0o600,
    )

    from fugue.bench.component_imports import import_mcp_config, lock_mcp_import

    import_mcp_config(
        request.staging_root / "mcp.json",
        server="wandb-main",
        import_id=baseline_id,
        repo_root=request.staging_root,
    )
    import_mcp_config(
        request.staging_root / "mcp.json",
        server="wandb-0-4-staging",
        import_id=candidate_id,
        repo_root=request.staging_root,
    )
    credential = _read_operator_credential(request.env_file, "WANDB_API_KEY")
    with _scoped_environment({"WANDB_API_KEY": credential}):
        baseline_lock = lock_mcp_import(
            baseline_id,
            request.staging_root,
            acknowledge_package_code=True,
            target_platform=request.target_platform,
        )
        candidate_lock = lock_mcp_import(
            candidate_id,
            request.staging_root,
            acknowledge_package_code=True,
            target_platform=request.target_platform,
        )

    source_evidence, source_conformance = _freeze_hosted_reference_source(
        staging_root=request.staging_root,
        credential=credential,
    )

    comparison = _render_resource(
        resources["comparison.yaml.template"],
        replacements,
        name="comparison.yaml.template",
    )
    _write_prepared_file(request.staging_root / "comparison.yaml", comparison)
    human_readable_comparison = _render_resource(
        resources["human-readable-evidence-canary.yaml.template"],
        replacements,
        name="human-readable-evidence-canary.yaml.template",
    )
    _write_prepared_file(
        request.staging_root / HUMAN_READABLE_COMPARISON_NAME,
        human_readable_comparison,
    )
    _write_prepared_file(
        request.staging_root / ".fugue-study.json",
        _json_bytes(
            {
                "schema_version": 1,
                "kind": "fugue_standalone_study",
                "template": "mcp-change",
            }
        ),
    )
    _write_prepared_file(
        request.staging_root / ".gitignore",
        (
            b".env\nprivate-labels.jsonl\n"
            b"human-readable-evidence-private-labels.jsonl\n"
            b".fugue/results/\n"
        ),
    )

    task_digest = hashlib.sha256(resources["tasks.jsonl"]).hexdigest()
    private_digest = hashlib.sha256(resources["private-labels.jsonl"]).hexdigest()
    scorer_digest = hashlib.sha256(
        resources["tool_surface_scorer_v7.py"]
    ).hexdigest()
    human_readable_canary_lock = _digest_bound_document(
        {
            "schema_version": 1,
            "kind": "wandb-mcp-human-readable-evidence-canary",
            "study_id": (
                "mcp-main-vs-0-4-"
                f"{candidate_sha[:7]}-human-readable-evidence-canary-v1"
            ),
            "source_lock_digest": request.source_lock.lock_digest,
            "candidate_source_digest": (
                request.source_lock.candidate_source_digest
            ),
            "comparison": {
                "path": HUMAN_READABLE_COMPARISON_NAME,
                "sha256": hashlib.sha256(
                    human_readable_comparison
                ).hexdigest(),
            },
            "tasks": {
                "path": HUMAN_READABLE_TASKS_NAME,
                "sha256": hashlib.sha256(human_tasks).hexdigest(),
                "task_ids": list(_HUMAN_READABLE_TASK_IDS),
            },
            "private_labels": {
                "path": HUMAN_READABLE_PRIVATE_LABELS_NAME,
                "sha256": hashlib.sha256(human_private_labels).hexdigest(),
                "task_ids": list(_HUMAN_READABLE_TASK_IDS),
            },
            "scorer": {
                "path": "tool_surface_scorer_v7.py",
                "sha256": scorer_digest,
            },
            "arm_count": 2,
            "attempts_per_coordinate": 1,
            "logical_cell_count": 4,
        },
        digest_field="lock_digest",
    )
    _write_prepared_file(
        request.staging_root / HUMAN_READABLE_CANARY_LOCK_NAME,
        _json_bytes(human_readable_canary_lock),
    )
    release_notes = _digest_bound_document(
        {
            "schema_version": 1,
            "kind": "wandb-mcp-release-notes-lock",
            "repository_url": request.source_lock.repository_url,
            "requested_ref": request.source_lock.requested_ref,
            "commit": request.source_lock.source_commit,
            "tree": request.source_lock.source_tree,
            "release_notes": request.source_lock.release_notes.to_dict(),
        },
        digest_field="lock_digest",
    )
    _write_prepared_file(
        request.staging_root / "release-notes.lock.json",
        _json_bytes(release_notes),
    )

    baseline_lock_value = baseline_lock.to_dict()
    candidate_lock_value = candidate_lock.to_dict()
    mechanism = _digest_bound_document(
        {
            "schema_version": 1,
            "kind": "wandb-mcp-reference-mechanism-preparation",
            "target_platform": request.target_platform,
            "source_lock_digest": request.source_lock.lock_digest,
            "release_notes_lock_digest": release_notes["lock_digest"],
            "profiles": [
                _mechanism_profile("baseline", _BASELINE_COMMIT, baseline_lock_value),
                _mechanism_profile("candidate", candidate_sha, candidate_lock_value),
            ],
            "release_note_coverage": [
                dict(item) for item in _PACKAGED_RELEASE_NOTE_COVERAGE
            ],
            "runtime_dependency": False,
        },
        digest_field="receipt_digest",
    )
    _write_prepared_file(
        request.staging_root / "mechanism-receipt.json",
        _json_bytes(mechanism),
    )

    key_paths = (
        ".fugue-study.json",
        f".fugue/imports/mcp/locks/{baseline_id}.json",
        f".fugue/imports/mcp/locks/{candidate_id}.json",
        f".fugue/imports/integrations/{baseline_id}.yaml",
        f".fugue/imports/integrations/{candidate_id}.yaml",
        "comparison.yaml",
        HUMAN_READABLE_CANARY_LOCK_NAME,
        HUMAN_READABLE_COMPARISON_NAME,
        HUMAN_READABLE_PRIVATE_LABELS_NAME,
        HUMAN_READABLE_TASKS_NAME,
        "mechanism-receipt.json",
        "private-labels.jsonl",
        "release-notes.lock.json",
        "source-conformance-receipt.json",
        "source-evidence.lock.json",
        "tasks.jsonl",
        "tool_surface_scorer_v7.py",
        "wbaf-task-design-provenance-v1.json",
    )
    artifacts = tuple(
        _materialized_artifact(
            request.staging_root,
            path,
            private=path
            in {"private-labels.jsonl", HUMAN_READABLE_PRIVATE_LABELS_NAME},
        )
        for path in sorted(key_paths)
    )
    behavior_inputs_digest = _stable_digest(
        {
            "schema_version": 1,
            "source_lock_digest": request.source_lock.lock_digest,
            "release_notes_lock_digest": release_notes["lock_digest"],
            "baseline": {
                "commit": _BASELINE_COMMIT,
                "source_digest": baseline_lock_value["source_digest"],
            },
            "candidate": {
                "commit": candidate_sha,
                "source_digest": candidate_lock_value["source_digest"],
            },
            "tasks_sha256": task_digest,
            "private_labels_sha256": private_digest,
            "scorer_sha256": scorer_digest,
            "human_readable_canary_lock": human_readable_canary_lock[
                "lock_digest"
            ],
            "hosted_source_evidence_lock": source_evidence[
                "evidence_lock_digest"
            ],
            "hosted_source_conformance": source_conformance[
                "receipt_digest"
            ],
        }
    )
    execution_inputs_digest = _stable_digest(
        {
            "schema_version": 1,
            "target_platform": request.target_platform,
            "mechanism_receipt_digest": mechanism["receipt_digest"],
            "profiles": [
                {
                    "id": value["id"],
                    "runtime_digest": value["runtime_digest"],
                    "runtime_platform": value["runtime_platform"],
                    "tool_manifest_digest": value["tool_manifest_digest"],
                }
                for value in (baseline_lock_value, candidate_lock_value)
            ],
        }
    )
    inventory = _materialization_inventory(request.staging_root)
    return ReferenceStudyMaterializationV1(
        schema_version=1,
        study_bundle_id="wandb-mcp-release",
        behavior_inputs_digest=behavior_inputs_digest,
        execution_inputs_digest=execution_inputs_digest,
        inventory_digest=str(inventory["inventory_digest"]),
        total_files=int(inventory["total_files"]),
        total_bytes=int(inventory["total_bytes"]),
        artifacts=artifacts,
    )


def _freeze_hosted_reference_source(
    *,
    staging_root: Path,
    credential: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze and verify the existing hosted task source without writing it.

    The W&B project is an input to the MCP under test, not Fugue's result
    backend.  Preparation therefore reads the complete source cohort twice,
    binds every immutable W&B/Weave identity, verifies the two Evaluation
    roots and their direct children, and writes only local lock artifacts.
    Missing, extra, mutable, cross-project, or non-private source evidence is
    fatal.  This function never seeds or repairs the hosted project.
    """

    from fugue.reference_studies import (
        wandb_mcp_qualification_core as qualification,
    )

    runtime_env = {"WANDB_API_KEY": credential}
    endpoint_binding = qualification._qualification_endpoint_binding(runtime_env)
    project_access = qualification.verify_private_project_topology(
        source_project=_SOURCE_PROJECT,
        # Results are canonical local artifacts.  The legacy alias in the
        # evidence-lock schema therefore points back to the source project;
        # no W&B result project is required or initialized.
        result_project=_SOURCE_PROJECT,
        env=runtime_env,
    )
    entity, project = _SOURCE_PROJECT.split("/", 1)
    selected_env = {
        "WANDB_API_KEY": credential,
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project,
        "WANDB_BASE_URL": endpoint_binding["api_base_url"],
        "WANDB_MODE": "online",
        "WANDB_RESUME": "never",
        "WANDB_SILENT": "true",
        "WEAVE_ALLOW_UNSAFE_CUSTOM_OBJ_DECODE": "false",
    }
    with _scoped_environment(selected_env):
        inventory = qualification._stable_hosted_source_inventory(
            entity,
            project,
            source_project=_SOURCE_PROJECT,
        )
    if inventory.get("complete") is not True:
        raise RuntimeError(
            "the W&B MCP reference source cohort is incomplete; prepare the "
            "source separately before freezing this study"
        )
    inventory_digest = str(inventory.get("inventory_digest") or "")
    _require_sha(
        inventory_digest,
        label="hosted source inventory digest",
        length=64,
    )
    evidence = qualification._evidence_lock(
        _SOURCE_PROJECT,
        inventory["runs"],
        qualification._inventory_weave_receipts(inventory),
        result_project=_SOURCE_PROJECT,
        # These deterministic provenance values keep a pure re-preparation
        # byte-identical while still naming the exact inventory that was
        # observed.  They are identities, not wall-clock claims.
        created_at=f"source-inventory:{inventory_digest}",
        preparation_id=f"wandb-mcp-reference-{inventory_digest[:24]}",
        source_inventory_digest=inventory_digest,
    )
    evidence = qualification.validate_evidence_lock(
        evidence,
        expected_project=None,
        expected_source_project=_SOURCE_PROJECT,
        expected_result_project=_SOURCE_PROJECT,
    )
    evidence_path = staging_root / "source-evidence.lock.json"
    _write_prepared_file(evidence_path, _json_bytes(evidence))

    roots, children = qualification._fetch_hosted_source_calls(
        source_project=_SOURCE_PROJECT,
        trace_base_url=endpoint_binding["trace_base_url"],
        api_key=credential,
        evaluation_call_ids=[
            str(item["call_id"])
            for item in evidence["objects"]["evaluations"]
        ],
    )
    conformance = qualification.build_hosted_source_conformance_receipt(
        evidence_lock=evidence,
        evaluation_roots=roots,
        direct_children=children,
        project_access=project_access,
        endpoint_binding=endpoint_binding,
        created_at=f"source-inventory:{inventory_digest}",
    )
    if conformance.get("status") != "passed":
        blockers = conformance.get("blockers") or ()
        raise RuntimeError(
            "the W&B MCP reference source conformance check failed: "
            + ", ".join(str(item) for item in blockers)
        )
    conformance_path = staging_root / "source-conformance-receipt.json"
    _write_prepared_file(conformance_path, _json_bytes(conformance))

    drift = qualification.verify_hosted_source_drift(
        evidence_lock=evidence_path,
        env=runtime_env,
    )
    if drift.status != "matched":
        raise RuntimeError(
            "the W&B MCP reference source changed while its lock was being "
            "prepared"
        )
    return evidence, conformance


def _load_packaged_reference_resources() -> dict[str, bytes]:
    root = files("fugue").joinpath(*_REFERENCE_RESOURCE_ROOT)
    if not root.is_dir():
        raise FileNotFoundError("packaged W&B MCP reference resources are missing")
    resources: dict[str, bytes] = {}
    for name in _REFERENCE_RESOURCE_NAMES:
        item = root.joinpath(name)
        symlink_check = getattr(item, "is_symlink", None)
        if callable(symlink_check) and symlink_check():
            raise ValueError(f"packaged W&B MCP resource may not be a symlink: {name}")
        if not item.is_file():
            raise FileNotFoundError(f"packaged W&B MCP resource is missing: {name}")
        resources[name] = item.read_bytes()
    return resources


def _validate_packaged_wbaf_provenance(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("packaged WBAF provenance must be an object")
    # Accept only the strict Fugue-owned representation. The package resource
    # is checked against the in-code reviewed constants rather than becoming a
    # second authority.
    observed = ReviewedTaskDesignProvenanceV1.from_dict(value)
    if observed != WBAF_TASK_DESIGN_PROVENANCE:
        raise ValueError("packaged WBAF provenance differs from reviewed constants")


def _validate_packaged_release_contract(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("packaged W&B MCP release contract must be an object")
    repository = value.get("repository")
    study = value.get("study")
    if not isinstance(repository, dict) or not isinstance(study, dict):
        raise ValueError("packaged W&B MCP release contract is incomplete")
    baseline = repository.get("baseline")
    candidate = repository.get("candidate")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError("packaged W&B MCP release coordinates are incomplete")
    if (
        repository.get("url") != WANDB_MCP_REPOSITORY_URL
        or baseline.get("commit") != _BASELINE_COMMIT
        or candidate.get("ref") != WANDB_MCP_RELEASE_REF
        or repository.get("release_notes_path") != WANDB_MCP_RELEASE_NOTES_PATH
        or study.get("adapter_id") != "wandb-mcp-release"
        or study.get("adapter_version") != 1
        or study.get("intent") != "python-package-release-qualification"
        or study.get("result_evidence_mode") != "local"
    ):
        raise ValueError("packaged W&B MCP release contract coordinates drifted")


def _render_resource(
    body: bytes,
    replacements: Mapping[str, str],
    *,
    name: str,
) -> bytes:
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"packaged reference template is not UTF-8: {name}") from exc
    for marker, replacement in replacements.items():
        text = text.replace(marker, replacement)
    remaining = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))
    if remaining:
        raise ValueError(
            f"packaged reference template {name} has unresolved markers: {remaining}"
        )
    return text.encode("utf-8")


def _jsonl_rows_by_id(body: bytes, *, name: str) -> dict[str, bytes]:
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"packaged reference JSONL is not UTF-8: {name}") from exc
    rows: dict[str, bytes] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(
                f"packaged reference JSONL has a blank row: {name}:{line_number}"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"packaged reference JSONL is invalid: {name}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                "packaged reference JSONL row must be an object: "
                f"{name}:{line_number}"
            )
        row_id = value.get("id")
        if not isinstance(row_id, str) or not _SAFE_ID.fullmatch(row_id):
            raise ValueError(
                "packaged reference JSONL row id is invalid: "
                f"{name}:{line_number}"
            )
        if row_id in rows:
            raise ValueError(
                f"packaged reference JSONL row id is duplicated: {name}:{row_id}"
            )
        rows[row_id] = (line + "\n").encode("utf-8")
    if not rows:
        raise ValueError(f"packaged reference JSONL is empty: {name}")
    return rows


def _select_aligned_reference_rows(
    *,
    public_body: bytes,
    private_body: bytes,
    selected_ids: Sequence[str],
) -> tuple[bytes, bytes]:
    """Select an exact task subset during trusted preparation.

    The prepared comparison points directly at these complete files. No run,
    preview, CLI caller, or presentation client truncates the four-task bundle
    after preparation.
    """

    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("reference task selection must contain unique task ids")
    public = _jsonl_rows_by_id(public_body, name="tasks.jsonl")
    private = _jsonl_rows_by_id(private_body, name="private-labels.jsonl")
    if set(public) != set(private):
        raise ValueError("packaged reference tasks and private labels are not aligned")
    missing = [task_id for task_id in selected_ids if task_id not in public]
    if missing:
        raise ValueError(
            "human-readable evidence canary task ids are missing: "
            + ", ".join(missing)
        )
    return (
        b"".join(public[task_id] for task_id in selected_ids),
        b"".join(private[task_id] for task_id in selected_ids),
    )


def _write_prepared_file(path: Path, body: bytes, *, mode: int = 0o644) -> None:
    if path.is_symlink():
        raise ValueError(f"reference materialization may not overwrite a symlink: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"reference materialization path already exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    path.chmod(mode)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _digest_bound_document(
    value: Mapping[str, Any], *, digest_field: str
) -> dict[str, Any]:
    if digest_field in value:
        raise ValueError(f"digest field is already populated: {digest_field}")
    payload = dict(value)
    payload[digest_field] = _stable_digest(payload)
    return payload


def _mechanism_profile(
    role: str,
    commit: str,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    expected_version = f"git:{commit}"
    if lock.get("version_identity") != expected_version:
        raise ValueError(f"{role} MCP lock does not bind {expected_version}")
    runtime_digest = str(lock.get("runtime_digest") or "")
    manifest_digest = str(lock.get("tool_manifest_digest") or "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_digest):
        raise ValueError(f"{role} MCP runtime digest is invalid")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_digest):
        raise ValueError(f"{role} MCP tool-manifest digest is invalid")
    tools = lock.get("tool_manifest")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{role} MCP lock has no initialized tool manifest")
    return {
        "role": role,
        "id": lock.get("id"),
        "source_commit": commit,
        "source_digest": lock.get("source_digest"),
        "runtime_digest": runtime_digest,
        "runtime_platform": lock.get("runtime_platform"),
        "tool_manifest_digest": manifest_digest,
        "initialized_manifest_matches_lock": True,
    }


def _read_operator_credential(env_file: Path | None, name: str) -> str:
    current = os.environ.get(name)
    if current:
        return current
    if env_file is None:
        raise ValueError(f"reference preparation requires {name} or --env-file")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw = stripped.split("=", 1)
        if key.strip() == name:
            value = raw.strip().strip("'\"")
            if value:
                return value
    raise ValueError(f"reference-study env file does not define {name}")


@contextmanager
def _scoped_environment(values: Mapping[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _materialized_artifact(
    root: Path,
    relative: str,
    *,
    private: bool = False,
) -> MaterializedReferenceArtifactV1:
    _safe_relative_path(relative, label="materialized artifact path")
    path = root / PurePosixPath(relative)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"materialized key artifact is missing: {relative}")
    body = path.read_bytes()
    return MaterializedReferenceArtifactV1(
        schema_version=1,
        path=relative,
        sha256=hashlib.sha256(body).hexdigest(),
        byte_count=len(body),
        executable=bool(path.stat().st_mode & 0o111),
        private=private,
    )


def _materialization_inventory(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    excluded = {SOURCE_LOCK_NAME, PREPARATION_RECEIPT_NAME}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("reference materialization may not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("reference materialization contains a non-regular object")
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        body = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(body).hexdigest(),
                "byte_count": len(body),
            }
        )
    entries.sort(key=lambda item: str(item["path"]))
    if not entries:
        raise ValueError("reference materialization inventory is empty")
    return {
        "inventory_digest": _stable_digest(
            {"schema_version": 1, "files": entries}
        ),
        "total_files": len(entries),
        "total_bytes": sum(int(item["byte_count"]) for item in entries),
        "paths": tuple(str(item["path"]) for item in entries),
    }


def prepare_wandb_mcp_reference_study(
    *,
    repo_root: Path,
    env_file: Path | None = None,
    platform: str | None = None,
    git: GitReferenceTransport | None = None,
    materializer: ReferenceMaterializer | None = None,
) -> WandbMCPReferencePreparationReceiptV1:
    """Freeze the exact staging head before creating any runnable study assets.

    Network resolution and source inspection happen only in this trusted
    preparation function. The first and second ref observations fence the
    complete staging/materialization transaction. A moving ref therefore
    leaves no durable candidate directory behind.
    """

    root = repo_root.resolve()
    if not root.is_dir():
        raise ValueError(f"study repository does not exist: {root}")
    if env_file is not None:
        env_file = env_file.resolve()
        if not env_file.is_file() or env_file.is_symlink():
            raise ValueError("reference-study env file must be a regular file")
    target_platform = _validate_platform(platform)
    transport = git or SubprocessGitReferenceTransport()
    first_commit = transport.observe_ref(
        WANDB_MCP_REPOSITORY_URL,
        WANDB_MCP_RELEASE_REF,
        cwd=root,
    )
    _require_sha(first_commit, label="first observed release commit", length=40)

    reference_parent = root / WANDB_MCP_REFERENCE_ROOT
    reference_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{first_commit}.", dir=reference_parent)
    )
    bare_parent: tempfile.TemporaryDirectory[str] | None = None
    try:
        bare_parent = tempfile.TemporaryDirectory(prefix="fugue-wandb-mcp-source-")
        bare_repository = Path(bare_parent.name) / "source.git"
        snapshot = transport.fetch_snapshot(
            WANDB_MCP_REPOSITORY_URL,
            first_commit,
            bare_repository=bare_repository,
        )
        if snapshot.source_commit != first_commit:
            raise ValueError(
                "fetched source commit differs from the first ref observation"
            )
        source_lock = _build_source_lock(snapshot)
        atomic_write_json(
            staging / SOURCE_LOCK_NAME,
            source_lock.to_dict(),
            mode=0o600,
        )

        selected_materializer = materializer or _materialize_packaged_reference_study
        materialization = selected_materializer(
            ReferenceMaterializationRequest(
                staging_root=staging,
                source_lock=source_lock,
                bare_repository=bare_repository,
                env_file=env_file,
                target_platform=target_platform,
            )
        )
        if not isinstance(materialization, ReferenceStudyMaterializationV1):
            raise TypeError(
                "reference materializer must return "
                "ReferenceStudyMaterializationV1"
            )
        _verify_materialized_artifacts(
            staging,
            materialization,
            additional_secret_values=_secret_env_file_values(env_file),
        )

        second_commit = transport.observe_ref(
            WANDB_MCP_REPOSITORY_URL,
            WANDB_MCP_RELEASE_REF,
            cwd=root,
        )
        _require_sha(second_commit, label="second observed release commit", length=40)
        if second_commit != first_commit:
            raise RuntimeError(
                "staging/0.4.0 moved during preparation; no reference lock was published"
            )

        receipt = _build_receipt(
            source_lock=source_lock,
            first_commit=first_commit,
            second_commit=second_commit,
            materialization=materialization,
        )
        atomic_write_json(
            staging / PREPARATION_RECEIPT_NAME,
            receipt.to_dict(),
            mode=0o600,
        )
        _freeze_staging_files(staging, materialization)
        destination = reference_parent / first_commit
        return _publish_immutable_destination(
            staging=staging,
            destination=destination,
            expected_lock=source_lock,
            expected_receipt=receipt,
        )
    finally:
        if bare_parent is not None:
            bare_parent.cleanup()
        if staging.exists():
            shutil.rmtree(staging)


def _validate_reference_coordinates(repository_url: str, requested_ref: str) -> None:
    if repository_url != WANDB_MCP_REPOSITORY_URL:
        raise ValueError("W&B MCP reference repository is not allowlisted")
    if requested_ref != WANDB_MCP_RELEASE_REF:
        raise ValueError("W&B MCP reference ref must be staging/0.4.0")


def _build_source_lock(
    snapshot: GitSourceSnapshotV1,
) -> WandbMCPReferenceSourceLockV1:
    _require_sha(snapshot.source_commit, label="fetched source commit", length=40)
    _require_sha(snapshot.source_tree, label="fetched source tree", length=40)
    _require_sha(
        snapshot.release_notes_blob, label="fetched release-note blob", length=40
    )
    if not snapshot.release_notes or len(snapshot.release_notes) > _MAX_RELEASE_NOTES_BYTES:
        raise ValueError("fetched release notes are empty or exceed the size limit")
    try:
        snapshot.release_notes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("fetched release notes are not valid UTF-8") from exc
    release_notes = ImmutableGitFileV1(
        schema_version=1,
        path=WANDB_MCP_RELEASE_NOTES_PATH,
        git_blob=snapshot.release_notes_blob,
        sha256=hashlib.sha256(snapshot.release_notes).hexdigest(),
        byte_count=len(snapshot.release_notes),
    )
    base = WandbMCPReferenceSourceLockV1(
        schema_version=1,
        kind="wandb-mcp-reference-source-lock",
        repository_url=WANDB_MCP_REPOSITORY_URL,
        requested_ref=WANDB_MCP_RELEASE_REF,
        source_commit=snapshot.source_commit,
        source_tree=snapshot.source_tree,
        release_notes=release_notes,
        task_design_provenance=(WBAF_TASK_DESIGN_PROVENANCE,),
        candidate_source_digest="0" * 64,
        lock_digest="0" * 64,
    )
    with_source = replace(
        base, candidate_source_digest=_candidate_source_digest(base)
    )
    return replace(with_source, lock_digest=_reference_lock_digest(with_source))


def _build_receipt(
    *,
    source_lock: WandbMCPReferenceSourceLockV1,
    first_commit: str,
    second_commit: str,
    materialization: ReferenceStudyMaterializationV1,
) -> WandbMCPReferencePreparationReceiptV1:
    base = WandbMCPReferencePreparationReceiptV1(
        schema_version=1,
        kind="wandb-mcp-reference-preparation",
        source_lock_digest=source_lock.lock_digest,
        candidate_source_digest=source_lock.candidate_source_digest,
        source_commit=source_lock.source_commit,
        first_observed_commit=first_commit,
        second_observed_commit=second_commit,
        destination=(
            WANDB_MCP_REFERENCE_ROOT / source_lock.source_commit
        ).as_posix(),
        materialization=materialization,
        receipt_digest="0" * 64,
    )
    return replace(base, receipt_digest=_receipt_digest(base))


def _verify_materialized_artifacts(
    staging: Path,
    materialization: ReferenceStudyMaterializationV1,
    *,
    additional_secret_values: tuple[bytes, ...] = (),
) -> None:
    inventory = _materialization_inventory(staging)
    if inventory["inventory_digest"] != materialization.inventory_digest:
        raise ValueError("reference materialization inventory digest differs")
    if inventory["total_files"] != materialization.total_files:
        raise ValueError("reference materialization file count differs")
    if inventory["total_bytes"] != materialization.total_bytes:
        raise ValueError("reference materialization byte count differs")
    secret_values = tuple(
        sorted(set(_secret_environment_values()) | set(additional_secret_values))
    )
    declared_paths = {item.path for item in materialization.artifacts}
    observed_paths = set(inventory["paths"])
    missing = sorted(declared_paths - observed_paths)
    if missing:
        raise ValueError(f"reference materialization key artifacts are missing: {missing}")
    for item in materialization.artifacts:
        path = staging / PurePosixPath(item.path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"materialized artifact is not a regular file: {item.path}")
        body = path.read_bytes()
        if len(body) != item.byte_count:
            raise ValueError(f"materialized artifact byte count differs: {item.path}")
        if hashlib.sha256(body).hexdigest() != item.sha256:
            raise ValueError(f"materialized artifact digest differs: {item.path}")
        for secret in secret_values:
            if secret in body:
                raise ValueError(
                    f"materialized artifact contains a host credential: {item.path}"
                )
    for relative in inventory["paths"]:
        body = (staging / PurePosixPath(relative)).read_bytes()
        for secret in secret_values:
            if secret in body:
                raise ValueError(
                    f"materialized artifact contains a host credential: {relative}"
                )


def _secret_environment_values() -> tuple[bytes, ...]:
    values = {
        value.encode("utf-8")
        for name, value in os.environ.items()
        if _SECRET_ENV_NAME.search(name) and len(value) >= 8
    }
    return tuple(sorted(values))


def _secret_env_file_values(env_file: Path | None) -> tuple[bytes, ...]:
    if env_file is None:
        return ()
    values: set[bytes] = set()
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, raw = stripped.split("=", 1)
        value = raw.strip().strip("'\"")
        if _SECRET_ENV_NAME.search(name.strip()) and len(value) >= 8:
            values.add(value.encode("utf-8"))
    return tuple(sorted(values))


def _freeze_staging_files(
    staging: Path,
    materialization: ReferenceStudyMaterializationV1,
) -> None:
    private = {
        item.path
        for item in materialization.artifacts
        if item.private
    }
    for path in staging.rglob("*"):
        if path.is_file():
            relative = path.relative_to(staging).as_posix()
            if relative in private:
                path.chmod(0o600)
                continue
            mode = path.stat().st_mode & 0o777
            path.chmod((mode & 0o555) or 0o444)


def _publish_immutable_destination(
    *,
    staging: Path,
    destination: Path,
    expected_lock: WandbMCPReferenceSourceLockV1,
    expected_receipt: WandbMCPReferencePreparationReceiptV1,
) -> WandbMCPReferencePreparationReceiptV1:
    if destination.exists():
        return _verify_existing_destination(
            destination, expected_lock=expected_lock, expected_receipt=expected_receipt
        )
    try:
        os.replace(staging, destination)
    except OSError:
        if not destination.exists():
            raise
        return _verify_existing_destination(
            destination, expected_lock=expected_lock, expected_receipt=expected_receipt
        )
    return read_wandb_mcp_reference_receipt(destination)


def _verify_existing_destination(
    destination: Path,
    *,
    expected_lock: WandbMCPReferenceSourceLockV1,
    expected_receipt: WandbMCPReferencePreparationReceiptV1,
) -> WandbMCPReferencePreparationReceiptV1:
    current_lock = read_wandb_mcp_reference_lock(destination)
    current_receipt = read_wandb_mcp_reference_receipt(destination)
    if current_lock != expected_lock or current_receipt != expected_receipt:
        raise ValueError(
            f"immutable W&B MCP reference destination already differs: {destination}"
        )
    _verify_materialized_artifacts(destination, current_receipt.materialization)
    return current_receipt


def read_wandb_mcp_reference_lock(
    path: Path,
) -> WandbMCPReferenceSourceLockV1:
    source = path / SOURCE_LOCK_NAME if path.is_dir() else path
    value = _read_json_object(source, artifact="W&B MCP reference lock")
    return WandbMCPReferenceSourceLockV1.from_dict(value)


def read_wandb_mcp_reference_receipt(
    path: Path,
) -> WandbMCPReferencePreparationReceiptV1:
    source = path / PREPARATION_RECEIPT_NAME if path.is_dir() else path
    value = _read_json_object(source, artifact="W&B MCP preparation receipt")
    receipt = WandbMCPReferencePreparationReceiptV1.from_dict(value)
    lock_path = source.parent / SOURCE_LOCK_NAME
    if lock_path.is_file():
        lock = read_wandb_mcp_reference_lock(lock_path)
        if (
            receipt.source_lock_digest != lock.lock_digest
            or receipt.candidate_source_digest != lock.candidate_source_digest
            or receipt.source_commit != lock.source_commit
        ):
            raise ValueError("reference receipt does not match its source lock")
    return receipt


def _read_json_object(path: Path, *, artifact: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{artifact} is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{artifact} must be a JSON object")
    return value


def _strict_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _strict_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _strict_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _validate_platform(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in {"linux/amd64", "linux/arm64"}:
        raise ValueError("reference-study platform must be linux/amd64 or linux/arm64")
    return value


def _git_environment() -> dict[str, str]:
    allowed = {
        "ALL_PROXY",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return environment
