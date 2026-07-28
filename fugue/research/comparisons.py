from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.redaction import redact_text, secrets_from_env
from fugue.research.approvals import ApprovalLedger
from fugue.research.contracts import (
    AttributionV1,
    EvidenceRefV1,
    ResearchError,
    StudyUpdateV1,
)
from fugue.research.experiment_views import (
    build_comparison_design_view,
    build_comparison_evaluation_view,
    build_comparison_progress_view,
)
from fugue.research.records import ResearchEvidenceRefV1
from fugue.research.store import StudyStore

if TYPE_CHECKING:
    from fugue.bench.comparison import ComparisonPreviewV1, ComparisonResultV1

COMPARISON_REGISTRY = Path("configs/fugue/research/comparisons.yaml")
COMPARISON_CONTROL_ROOT = Path(".fugue/private/research-comparisons")
COMPARISON_PUBLIC_ROOT = Path(".fugue/research-comparisons")
COMPARISON_RESULT_ROOT = Path(".fugue/comparison-results")
_TERMINAL = frozenset({"completed", "failed", "interrupted"})


@dataclass(frozen=True)
class RegisteredComparisonV1:
    id: str
    path: str
    digest: str
    question: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ComparisonRegistry:
    def __init__(
        self,
        repo_root: Path,
        entries: Mapping[str, RegisteredComparisonV1],
    ) -> None:
        self.repo_root = repo_root.resolve()
        self._entries = dict(entries)

    @classmethod
    def from_file(
        cls,
        repo_root: Path,
        path: Path | None = None,
    ) -> ComparisonRegistry:
        from fugue.bench.comparison import load_comparison

        root = repo_root.resolve()
        selected = (path or root / COMPARISON_REGISTRY).resolve()
        if not selected.is_file() or selected.is_symlink():
            return cls(root, {})
        raw = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping) or set(raw) != {
            "schema_version",
            "comparisons",
        }:
            raise RuntimeError("comparison registry has an invalid shape")
        if raw["schema_version"] != 1 or not isinstance(raw["comparisons"], list):
            raise RuntimeError("comparison registry version must be 1")
        entries: dict[str, RegisteredComparisonV1] = {}
        paths: set[str] = set()
        for index, item in enumerate(raw["comparisons"], start=1):
            if not isinstance(item, Mapping) or set(item) != {"id", "path", "digest"}:
                raise RuntimeError(
                    f"comparison registry entry {index} has an invalid shape"
                )
            comparison_id = str(item["id"])
            relative = _safe_relative_path(str(item["path"]))
            if comparison_id in entries or relative in paths:
                raise RuntimeError("comparison registry ids and paths must be unique")
            spec = load_comparison(Path(relative), repo_root=root)
            digest = str(item["digest"])
            if spec.id != comparison_id or spec.spec_digest != digest:
                raise RuntimeError(f"registered comparison drift: {comparison_id}")
            entries[comparison_id] = RegisteredComparisonV1(
                id=comparison_id,
                path=relative,
                digest=digest,
                question=spec.question,
            )
            paths.add(relative)
        return cls(root, entries)

    def catalog(self) -> tuple[RegisteredComparisonV1, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def resolve(self, comparison_id: str) -> RegisteredComparisonV1:
        from fugue.bench.comparison import load_comparison

        try:
            entry = self._entries[comparison_id]
        except KeyError as exc:
            raise ResearchError(
                "comparison_not_registered",
                f"comparison is not registered: {comparison_id}",
                category="policy",
            ) from exc
        spec = load_comparison(Path(entry.path), repo_root=self.repo_root)
        if spec.id != entry.id or spec.spec_digest != entry.digest:
            raise ResearchError(
                "comparison_registry_drift",
                f"registered comparison changed: {entry.id}",
                category="policy",
            )
        return entry


class ComparisonControlService:
    """Bounded Research control over the canonical comparison façade."""

    def __init__(
        self,
        repo_root: Path,
        *,
        store: StudyStore,
        approvals: ApprovalLedger,
        env_file: Path | None = None,
        registry: ComparisonRegistry | None = None,
        state_root: Path | None = None,
        public_root: Path | None = None,
        launch_worker: Callable[[Path], int] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.store = store
        self.approvals = approvals
        self.env_file = env_file.resolve() if env_file is not None else None
        self.registry = registry or ComparisonRegistry.from_file(self.repo_root)
        self.state_root = (
            state_root.resolve()
            if state_root is not None
            else self.repo_root / COMPARISON_CONTROL_ROOT
        )
        self.public_root = (
            public_root.resolve()
            if public_root is not None
            else self.repo_root / COMPARISON_PUBLIC_ROOT
        )
        self._launch_worker = launch_worker or self._popen_worker

    def catalog(self) -> tuple[dict[str, str], ...]:
        return tuple(item.to_dict() for item in self.registry.catalog())

    def readiness(self, comparison_id: str) -> dict[str, Any]:
        from fugue.bench.comparison import check_comparison

        entry, spec = self._registered_spec(comparison_id)
        readiness = check_comparison(spec, repo_root=self.repo_root).to_dict()
        readiness.pop("private_labels_digest", None)
        return {
            "registration": entry.to_dict(),
            "readiness": readiness,
        }

    def preview(self, study_id: str, comparison_id: str) -> dict[str, Any]:
        from fugue.bench.comparison import preview_comparison

        self.store.get_study(study_id)
        entry, spec = self._registered_spec(comparison_id)
        preview = preview_comparison(spec, repo_root=self.repo_root)
        directory = self._run_dir(preview.preview_digest)
        directory.mkdir(parents=True, exist_ok=True)
        preview_path = directory / "preview.json"
        atomic_write_json(preview_path, preview.to_dict())
        preview_path.chmod(0o600)
        public = self._public_preview(study_id, entry, preview)
        public_path = self._public_preview_path(study_id, preview.preview_digest)
        public_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(public_path, public)
        return public

    def request_approval(
        self,
        study_id: str,
        comparison_id: str,
        preview_digest: str,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        entry, preview = self._accepted_preview(study_id, comparison_id, preview_digest)
        readiness = preview.readiness
        if readiness.get("status") != "ready":
            raise ResearchError(
                "preview_ineligible",
                "only a ready comparison preview may request approval",
                category="policy",
                details={"blockers": list(readiness.get("blockers") or [])},
            )
        resource_id = f"comparison-preview-{preview.preview_digest[:20]}"
        study = self.store.get_study(study_id)
        if resource_id not in {item.id for item in study.resources}:
            study = self.store.update_study(
                study_id,
                StudyUpdateV1(
                    message=(
                        f"Comparison {entry.id} is awaiting operator approval for "
                        f"exact preview {preview.preview_digest}; "
                        f"{readiness['estimated_cells']} cells, estimated "
                        f"${float(readiness['estimated_cost_usd']):.2f}."
                    ),
                    note_kind="decision",
                    resources=(
                        {
                            "id": resource_id,
                            "uri": self._relative(
                                self._public_preview_path(
                                    study_id, preview.preview_digest
                                )
                            ),
                            "kind": "comparison_preview",
                            "digest": self._public_preview(study_id, entry, preview)[
                                "artifact_digest"
                            ],
                            "version": preview.preview_digest,
                            "title": entry.question,
                            "summary": (
                                f"{readiness['estimated_cells']} exact cells; "
                                "human approval required before execution."
                            ),
                        },
                    ),
                    attribution=_service_attribution(),
                ),
                operation_id=idempotency_key,
            )
        view = build_comparison_design_view(preview.to_dict())
        self.store.record_experiment_view_event(
            research_id=study_id,
            experiment_id=_comparison_experiment_id(preview.preview_digest),
            producer_event_id=(
                f"fugue:{study_id}:{_comparison_experiment_id(preview.preview_digest)}:"
                f"comparison-design-{preview.preview_digest}"
            ),
            classification="decision",
            state="awaiting_approval",
            message="Exact comparison preview is awaiting operator approval.",
            reserved_cost_usd=float(readiness["estimated_cost_usd"]),
            evidence=(
                ResearchEvidenceRefV1(
                    kind="artifact",
                    ref=f"comparison-preview:{preview.preview_digest}",
                    system="fugue",
                    digest=preview.preview_digest,
                ),
            ),
            view=view,
            attribution=_service_attribution(),
        )
        return {
            **self._public_preview(study_id, entry, preview),
            "approval_state": "awaiting_approval",
            "study_revision": study.revision,
        }

    def start(
        self,
        study_id: str,
        comparison_id: str,
        preview_digest: str,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        entry, preview = self._accepted_preview(study_id, comparison_id, preview_digest)
        try:
            approval = self.approvals.get_for_preview(
                subject_kind="experiment",
                preview_digest=preview.preview_digest,
            )
        except ResearchError as exc:
            if exc.code != "approval_not_found":
                raise
            raise ResearchError(
                "approval_required",
                "the exact comparison preview requires prior operator approval",
                category="policy",
            ) from exc
        readiness = preview.readiness
        approval = self.approvals.claim(
            approval_digest=approval.approval_digest,
            subject_kind="experiment",
            preview_digest=preview.preview_digest,
            subject_id=f"comparison-{preview.preview_digest[:20]}",
            estimated_cells=int(readiness["estimated_cells"]),
            estimated_cost_usd=float(readiness["estimated_cost_usd"]),
        )
        directory = self._run_dir(preview.preview_digest)
        directory.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(directory / "start.lock"))
        with lock:
            state_path = directory / "state.json"
            if state_path.is_file():
                state = _read_state(state_path)
                if state["comparison_id"] != entry.id or state["study_id"] != study_id:
                    raise ResearchError(
                        "comparison_start_conflict",
                        "preview is already attached to another comparison or Study",
                        category="conflict",
                    )
                return _public_state(state)
            worker_input = directory / "worker-input.json"
            atomic_write_json(
                worker_input,
                {
                    "schema_version": 1,
                    "repo_root": self.repo_root.as_posix(),
                    "study_id": study_id,
                    "comparison_id": entry.id,
                    "spec_digest": entry.digest,
                    "preview_digest": preview.preview_digest,
                    "approval_digest": approval.approval_digest,
                    "env_file": (
                        self.env_file.as_posix() if self.env_file is not None else None
                    ),
                    "state_path": state_path.as_posix(),
                },
            )
            worker_input.chmod(0o600)
            state = _signed_state(
                {
                    "schema_version": 1,
                    "study_id": study_id,
                    "comparison_id": entry.id,
                    "spec_digest": entry.digest,
                    "preview_digest": preview.preview_digest,
                    "status": "starting",
                    "pid": None,
                    "created_at": _now(),
                    "updated_at": _now(),
                    "result_digest": None,
                    "rows": None,
                    "result_path": None,
                    "error": None,
                    "operation_id": idempotency_key,
                }
            )
            atomic_write_json(state_path, state)
            state_path.chmod(0o600)
            try:
                pid = self._launch_worker(worker_input)
            except Exception as exc:
                state = _signed_state(
                    {
                        **state,
                        "status": "failed",
                        "updated_at": _now(),
                        "error": {
                            "code": "worker_launch_failed",
                            "message": type(exc).__name__,
                        },
                    }
                )
                atomic_write_json(state_path, state)
                raise
            state = _signed_state(
                {**state, "status": "running", "pid": pid, "updated_at": _now()}
            )
            atomic_write_json(state_path, state)
        self._project_started(study_id, entry, preview, idempotency_key)
        return _public_state(state)

    def watch(
        self,
        study_id: str,
        comparison_id: str,
        preview_digest: str,
    ) -> dict[str, Any]:
        self._accepted_preview(study_id, comparison_id, preview_digest)
        state_path = self._run_dir(preview_digest) / "state.json"
        if not state_path.is_file():
            raise ResearchError(
                "comparison_not_started",
                "comparison preview has not been started",
            )
        state = _read_state(state_path)
        pid = state.get("pid")
        if (
            state["status"] not in _TERMINAL
            and isinstance(pid, int)
            and not _process_exists(pid)
        ):
            state = _signed_state(
                {
                    **state,
                    "status": "interrupted",
                    "updated_at": _now(),
                    "error": {
                        "code": "worker_interrupted",
                        "message": "comparison worker exited without a terminal result",
                    },
                }
            )
            atomic_write_json(state_path, state)
        return _public_state(state)

    def result(
        self,
        study_id: str,
        comparison_id: str,
        preview_digest: str,
    ) -> dict[str, Any]:
        state = self.watch(study_id, comparison_id, preview_digest)
        if state["status"] != "completed":
            raise ResearchError(
                "comparison_result_unavailable",
                f"comparison is {state['status']}; no completed result is available",
            )
        path = (self.repo_root / str(state["result_path"])).resolve()
        result_root = (self.repo_root / COMPARISON_RESULT_ROOT).resolve()
        if not path.is_relative_to(result_root) or not path.is_file():
            raise ResearchError(
                "comparison_result_path_invalid",
                "comparison result is outside the governed result store",
                category="evidence",
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        unsigned = dict(raw)
        unsigned["result_digest"] = ""
        if stable_digest(unsigned) != raw.get("result_digest"):
            raise ResearchError(
                "comparison_result_drift",
                "comparison result digest does not match",
                category="evidence",
            )
        return raw

    def _registered_spec(self, comparison_id: str) -> tuple[Any, Any]:
        from fugue.bench.comparison import load_comparison

        entry = self.registry.resolve(comparison_id)
        spec = load_comparison(Path(entry.path), repo_root=self.repo_root)
        return entry, spec

    def _accepted_preview(
        self,
        study_id: str,
        comparison_id: str,
        preview_digest: str,
    ) -> tuple[RegisteredComparisonV1, ComparisonPreviewV1]:
        from fugue.bench.comparison import preview_comparison

        self.store.get_study(study_id)
        entry, spec = self._registered_spec(comparison_id)
        current = preview_comparison(spec, repo_root=self.repo_root)
        if current.preview_digest != preview_digest:
            raise ResearchError(
                "comparison_preview_drift",
                "the accepted comparison preview no longer matches registered inputs",
                category="policy",
            )
        path = self._run_dir(preview_digest) / "preview.json"
        if not path.is_file():
            raise ResearchError(
                "comparison_preview_not_found",
                "preview must be created before it can be approved or started",
            )
        if stable_digest(json.loads(path.read_text(encoding="utf-8"))) != stable_digest(
            current.to_dict()
        ):
            raise ResearchError(
                "comparison_preview_drift",
                "stored comparison preview differs from current registered inputs",
                category="policy",
            )
        return entry, current

    def _public_preview(
        self,
        study_id: str,
        entry: RegisteredComparisonV1,
        preview: ComparisonPreviewV1,
    ) -> dict[str, Any]:
        readiness = dict(preview.readiness)
        readiness.pop("private_labels_digest", None)
        public = json.loads(
            json.dumps(
                {
                    "schema_version": 1,
                    "study_id": study_id,
                    "comparison": entry.to_dict(),
                    "preview_digest": preview.preview_digest,
                    "readiness": readiness,
                    "matrix": preview.matrix,
                },
                sort_keys=True,
                default=str,
            )
        )
        return {**public, "artifact_digest": stable_digest(public)}

    def _run_dir(self, preview_digest: str) -> Path:
        if len(preview_digest) != 64 or any(
            char not in "0123456789abcdef" for char in preview_digest
        ):
            raise ResearchError("invalid_digest", "preview digest must be sha256")
        return self.state_root / preview_digest

    def _public_preview_path(
        self,
        study_id: str,
        preview_digest: str,
    ) -> Path:
        self._run_dir(preview_digest)
        return self.public_root / study_id / preview_digest / "preview.json"

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    @staticmethod
    def _popen_worker(input_path: Path) -> int:
        directory = input_path.parent
        stdout_path = directory / "worker.stdout.log"
        stderr_path = directory / "worker.stderr.log"
        with (
            stdout_path.open("ab") as stdout,
            stderr_path.open("ab") as stderr,
        ):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "fugue.research.comparison_worker",
                    input_path.as_posix(),
                ],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        stdout_path.chmod(0o600)
        stderr_path.chmod(0o600)
        return process.pid

    def _project_started(
        self,
        study_id: str,
        entry: RegisteredComparisonV1,
        preview: ComparisonPreviewV1,
        idempotency_key: str,
    ) -> None:
        source = EvidenceRefV1(
            kind="artifact",
            ref=f"comparison-preview:{preview.preview_digest}",
            digest=preview.preview_digest,
        )
        self.store.update_study(
            study_id,
            StudyUpdateV1(
                message=(
                    f"Started approved comparison {entry.id} at exact preview "
                    f"{preview.preview_digest}."
                ),
                note_kind="execution",
                note_sources=(source,),
                attribution=_service_attribution(),
            ),
            operation_id=f"{idempotency_key}-project",
        )
        view = build_comparison_progress_view(preview.to_dict(), phase="running")
        self.store.record_experiment_view_event(
            research_id=study_id,
            experiment_id=_comparison_experiment_id(preview.preview_digest),
            producer_event_id=(
                f"fugue:{study_id}:{_comparison_experiment_id(preview.preview_digest)}:"
                f"comparison-progress-{preview.preview_digest}"
            ),
            classification="lifecycle",
            state="running",
            message="Approved comparison execution started.",
            progress={
                "completed": 0,
                "total": int(preview.readiness["estimated_cells"]),
            },
            reserved_cost_usd=float(preview.readiness["estimated_cost_usd"]),
            evidence=(
                ResearchEvidenceRefV1(
                    kind="artifact",
                    ref=f"comparison-preview:{preview.preview_digest}",
                    system="fugue",
                    digest=preview.preview_digest,
                ),
            ),
            view=view,
            attribution=_service_attribution(),
        )


def run_comparison_worker(input_path: Path) -> int:
    from fugue.bench.comparison import execute_comparison

    raw = json.loads(input_path.resolve(strict=True).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "repo_root",
        "study_id",
        "comparison_id",
        "spec_digest",
        "preview_digest",
        "approval_digest",
        "env_file",
        "state_path",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw["schema_version"] != 1
    ):
        raise RuntimeError("comparison worker input has an invalid shape")
    repo_root = Path(str(raw["repo_root"])).resolve(strict=True)
    state_path = Path(str(raw["state_path"])).resolve()
    store = StudyStore(repo_root)
    service = ComparisonControlService(
        repo_root,
        store=store,
        approvals=ApprovalLedger(store.path),
        env_file=Path(str(raw["env_file"])) if raw["env_file"] else None,
    )
    entry, preview = service._accepted_preview(
        str(raw["study_id"]),
        str(raw["comparison_id"]),
        str(raw["preview_digest"]),
    )
    if entry.digest != raw["spec_digest"]:
        raise RuntimeError("comparison worker input no longer matches the registry")
    current = _read_state(state_path)
    atomic_write_json(
        state_path,
        _signed_state(
            {
                **current,
                "status": "running",
                "pid": os.getpid(),
                "updated_at": _now(),
            }
        ),
    )
    try:
        result, json_path, _ = execute_comparison(
            preview,
            approval_digest=str(raw["approval_digest"]),
            repo_root=repo_root,
            env_file=Path(str(raw["env_file"])) if raw["env_file"] else None,
        )
    except BaseException as exc:
        safe_message = redact_text(str(exc), secrets_from_env(os.environ))[:1000]
        failed = _signed_state(
            {
                **_read_state(state_path),
                "status": "failed",
                "updated_at": _now(),
                "error": {
                    "code": "comparison_execution_failed",
                    "message": safe_message or type(exc).__name__,
                },
            }
        )
        atomic_write_json(state_path, failed)
        _project_failure(store, str(raw["study_id"]), preview, failed["error"])
        raise
    completed = _signed_state(
        {
            **_read_state(state_path),
            "status": "completed",
            "updated_at": _now(),
            "result_digest": result.result_digest,
            "rows": result.rows,
            "result_path": json_path.relative_to(repo_root).as_posix(),
            "error": None,
        }
    )
    atomic_write_json(state_path, completed)
    _project_result(store, str(raw["study_id"]), result, json_path, repo_root)
    return 0


def _project_result(
    store: StudyStore,
    study_id: str,
    result: ComparisonResultV1,
    json_path: Path,
    repo_root: Path,
) -> None:
    source = EvidenceRefV1(
        kind="artifact",
        ref=json_path.relative_to(repo_root).as_posix(),
        digest=result.result_digest,
    )
    outcome = (
        "incomplete"
        if result.incomplete or result.required_evaluations_incomplete
        else "completed"
    )
    store.update_study(
        study_id,
        StudyUpdateV1(
            message=(
                f"Comparison {result.comparison_id} completed with "
                f"{result.improved} improved, {result.regressed} regressed, and "
                f"{result.unchanged} unchanged aligned pairs."
            ),
            note_kind="result",
            note_sources=(source,),
            resources=(
                {
                    "id": f"comparison-result-{result.result_digest[:20]}",
                    "uri": source.ref,
                    "kind": "comparison_result",
                    "digest": result.result_digest,
                    "version": result.preview_digest,
                    "title": f"{result.comparison_id} result",
                    "summary": (
                        f"{result.rows} rows; {result.improved} improved; "
                        f"{result.regressed} regressed."
                    ),
                },
            ),
            results=(
                {
                    "id": f"comparison-{result.result_digest[:20]}",
                    "statement": (
                        f"{result.candidate_passed} candidate attempts and "
                        f"{result.baseline_passed} baseline attempts passed; "
                        f"{result.improved} aligned pairs improved and "
                        f"{result.regressed} regressed."
                    ),
                    "kind": "comparison",
                    "outcome": outcome,
                    "estimate": {
                        "value": result.candidate_passed - result.baseline_passed,
                        "kind": "deterministic-pass-count-delta",
                        "unit": "attempts",
                    },
                    "comparison": {
                        "condition": "candidate",
                        "comparator": "baseline",
                        "condition_sources": [source.to_dict()],
                        "comparator_sources": [source.to_dict()],
                    },
                    "population": "locked comparison tasks, harnesses, and attempts",
                    "conditions": {
                        "comparison_id": result.comparison_id,
                        "preview_digest": result.preview_digest,
                    },
                    "sample_size": result.rows,
                    "aggregation": "aligned task-harness-attempt pairs",
                    "exclusions": list(result.limitations),
                    "sources": [source.to_dict()],
                },
            ),
            run_refs=(
                EvidenceRefV1(
                    kind="run",
                    ref=result.source,
                    selector={"comparison_id": result.comparison_id},
                ),
            ),
            attribution=_service_attribution(),
        ),
        operation_id=f"comparison-result-{result.result_digest[:20]}",
    )
    result_ref = json_path.relative_to(repo_root).as_posix()
    view = build_comparison_evaluation_view(
        result.to_dict(),
        result_ref=result_ref,
    )
    store.record_experiment_view_event(
        research_id=study_id,
        experiment_id=_comparison_experiment_id(result.preview_digest),
        producer_event_id=(
            f"fugue:{study_id}:{_comparison_experiment_id(result.preview_digest)}:"
            f"comparison-evaluation-{result.result_digest}"
        ),
        classification="result",
        state="completed",
        message="Comparison result reconciled into the canonical Study view.",
        progress={"completed": result.rows, "total": result.rows},
        observed_cost_usd=(
            float(result.operational_summary["observed_cost_usd"])
            if result.operational_summary.get("observed_cost_usd") is not None
            else None
        ),
        evidence=(
            ResearchEvidenceRefV1(
                kind="artifact",
                ref=result_ref,
                system="fugue",
                digest=result.result_digest,
            ),
        ),
        view=view,
        attribution=_service_attribution(),
    )


def _project_failure(
    store: StudyStore,
    study_id: str,
    preview: ComparisonPreviewV1,
    error: Mapping[str, Any],
) -> None:
    store.update_study(
        study_id,
        StudyUpdateV1(
            message=(
                f"Comparison preview {preview.preview_digest} failed: "
                f"{error.get('message') or error.get('code')}."
            ),
            note_kind="execution_error",
            note_sources=(
                EvidenceRefV1(
                    kind="artifact",
                    ref=f"comparison-preview:{preview.preview_digest}",
                    digest=preview.preview_digest,
                ),
            ),
            attribution=_service_attribution(),
        ),
        operation_id=f"comparison-failed-{preview.preview_digest[:20]}",
    )
    view = build_comparison_progress_view(
        preview.to_dict(),
        phase="failed",
        state_counts={"failed": 1},
    )
    store.record_experiment_view_event(
        research_id=study_id,
        experiment_id=_comparison_experiment_id(preview.preview_digest),
        producer_event_id=(
            f"fugue:{study_id}:{_comparison_experiment_id(preview.preview_digest)}:"
            f"comparison-failed-{preview.preview_digest}"
        ),
        classification="limitation",
        state="failed",
        message="Comparison execution failed before a terminal evaluation was available.",
        progress={
            "completed": 0,
            "total": int(preview.readiness["estimated_cells"]),
        },
        reserved_cost_usd=float(preview.readiness["estimated_cost_usd"]),
        evidence=(
            ResearchEvidenceRefV1(
                kind="artifact",
                ref=f"comparison-preview:{preview.preview_digest}",
                system="fugue",
                digest=preview.preview_digest,
            ),
        ),
        view=view,
        attribution=_service_attribution(),
    )


def _comparison_experiment_id(preview_digest: str) -> str:
    return f"comparison-{preview_digest[:20]}"


def _read_state(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ResearchError(
            "comparison_state_corrupt",
            "comparison state is not an object",
            category="evidence",
        )
    expected = raw.get("state_digest")
    if expected != stable_digest(
        {key: value for key, value in raw.items() if key != "state_digest"}
    ):
        raise ResearchError(
            "comparison_state_drift",
            "comparison state digest does not match",
            category="evidence",
        )
    return raw


def _signed_state(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in raw.items() if key != "state_digest"}
    return {**value, "state_digest": stable_digest(value)}


def _public_state(state: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "schema_version",
        "study_id",
        "comparison_id",
        "spec_digest",
        "preview_digest",
        "status",
        "pid",
        "created_at",
        "updated_at",
        "result_digest",
        "rows",
        "result_path",
        "error",
        "state_digest",
    )
    return {key: state[key] for key in allowed if state.get(key) is not None}


def _safe_relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError("comparison registry paths must be repository-relative")
    return path.as_posix()


def _service_attribution() -> AttributionV1:
    return AttributionV1(actor_type="service", name="fugue-comparison-control")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
