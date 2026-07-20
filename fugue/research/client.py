from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.research.contracts import (
    AttributionV1,
    EvidenceRefV1,
    ExperimentDraftV1,
    ExperimentPreviewV1,
    StudyContextV1,
    StudyV1,
    build_experiment_draft,
    evidence_ref_from_dict,
    study_update_from_dict,
)
from fugue.research.service import ExperimentHandle, ResearchService, ResearchWorker


class FugueResearchClient:
    """Python surface for governed outer-loop research."""

    def __init__(
        self,
        service: ResearchService,
        *,
        worker: ResearchWorker | None = None,
    ) -> None:
        self.service = service
        self.worker = worker
        self.studies = Studies(self)

    @classmethod
    def local(
        cls,
        repo_root: str | Path,
        *,
        env_file: str | Path | None = None,
        auto_worker: bool = True,
        poll_interval: float = 1.0,
    ) -> FugueResearchClient:
        service = ResearchService(
            Path(repo_root), Path(env_file) if env_file is not None else None
        )
        worker = (
            ResearchWorker(service, poll_interval=poll_interval).start()
            if auto_worker
            else None
        )
        return cls(service, worker=worker)

    def close(self) -> None:
        if self.worker:
            self.worker.stop()

    def __enter__(self) -> FugueResearchClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def experiment(self, experiment_id: str) -> ExperimentHandle:
        self.service.store.get_experiment(experiment_id)
        return ExperimentHandle(self.service, experiment_id)


class Studies:
    def __init__(self, client: FugueResearchClient) -> None:
        self.client = client

    def create(
        self,
        *,
        title: str,
        question: str,
        campaign_id: str,
        study_id: str | None = None,
        background: str = "",
        parent_study_ids: Iterable[str] = (),
        attribution: AttributionV1 | Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> StudyHandle:
        study_id = study_id or f"study-{uuid.uuid4().hex[:12]}"
        operation_id = idempotency_key or f"create-{study_id}"
        actor = _attribution(attribution)
        self.client.service.store.create_study(
            study_id=study_id,
            title=title,
            campaign_id=campaign_id,
            question=question,
            background=background,
            parent_study_ids=parent_study_ids,
            attribution=actor,
            operation_id=operation_id,
        )
        return StudyHandle(self.client, study_id)

    def get(self, study_id: str) -> StudyHandle:
        self.client.service.store.get_study(study_id)
        return StudyHandle(self.client, study_id)

    def list(self, *, limit: int = 100) -> tuple[StudyV1, ...]:
        return self.client.service.store.list_studies(limit=limit)


class StudyHandle:
    def __init__(self, client: FugueResearchClient, study_id: str) -> None:
        self.client = client
        self.id = study_id
        self.experiments = Experiments(self)

    @property
    def revision(self) -> int:
        return self.get().revision

    def get(self) -> StudyV1:
        return self.client.service.store.get_study(self.id)

    def context(
        self,
        *,
        max_experiments: int = 20,
        max_results: int = 20,
        max_notes: int = 20,
        max_chars: int = 32000,
    ) -> StudyContextV1:
        return self.client.service.store.context(
            self.id,
            max_experiments=max_experiments,
            max_results=max_results,
            max_notes=max_notes,
            max_chars=max_chars,
        )

    def record(
        self,
        message: str | None = None,
        *,
        runs: Iterable[EvidenceRefV1 | Mapping[str, Any] | str] = (),
        results: Iterable[Mapping[str, Any]] = (),
        resources: Iterable[Mapping[str, Any]] = (),
        brief_patch: Mapping[str, Any] | None = None,
        brief_sources: Mapping[str, Iterable[EvidenceRefV1 | Mapping[str, Any]]]
        | None = None,
        baselines: Iterable[EvidenceRefV1 | Mapping[str, Any] | str] | None = None,
        primary_baseline: EvidenceRefV1 | Mapping[str, Any] | str | None = None,
        note_kind: str = "observation",
        note_sources: Iterable[EvidenceRefV1 | Mapping[str, Any]] = (),
        supersedes_note: str | None = None,
        attribution: AttributionV1 | Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
    ) -> StudyV1:
        raw: dict[str, Any] = {
            "message": message,
            "note_kind": note_kind,
            "note_sources": [_ref(item).to_dict() for item in note_sources],
            "note_supersedes": supersedes_note,
            "brief_patch": dict(brief_patch or {}),
            "brief_sources": {
                key: [_ref(item).to_dict() for item in values]
                for key, values in (brief_sources or {}).items()
            },
            "resources": [dict(item) for item in resources],
            "results": [dict(item) for item in results],
            "run_refs": [_ref(item, default_kind="run").to_dict() for item in runs],
            "attribution": _attribution(attribution).to_dict(),
        }
        if baselines is not None:
            raw["baseline_refs"] = [
                _ref(item, default_kind="outcome").to_dict() for item in baselines
            ]
        if primary_baseline is not None:
            raw["primary_baseline_ref"] = _ref(
                primary_baseline, default_kind="outcome"
            ).to_dict()
        update = study_update_from_dict(raw)
        operation_id = idempotency_key or f"record-{uuid.uuid4().hex[:12]}"
        return self.client.service.store.update_study(
            self.id,
            update,
            operation_id=operation_id,
            expected_revision=expected_revision,
        )


class Experiments:
    def __init__(self, study: StudyHandle) -> None:
        self.study = study

    def preview(
        self,
        draft: ExperimentDraftV1 | Mapping[str, Any] | None = None,
        **values: Any,
    ) -> ExperimentPreviewV1:
        study = self.study.get()
        if draft is None:
            design = values.pop("design", None)
            task_suite = values.pop("task_suite", None)
            if design is not None:
                if not isinstance(design, Mapping):
                    raise ValueError("experiment design must be a mapping")
                overlap = set(values) & set(design)
                if overlap:
                    raise ValueError(
                        f"experiment fields provided twice: {', '.join(sorted(overlap))}"
                    )
                values = {**dict(design), **values}
            if task_suite is not None:
                if "task_suite_digest" in values or "task_suite_draft" in values:
                    raise ValueError("task_suite was provided more than once")
                if isinstance(task_suite, str):
                    values["task_suite_digest"] = task_suite
                elif isinstance(task_suite, Mapping):
                    values["task_suite_draft"] = dict(task_suite)
                elif hasattr(task_suite, "to_dict"):
                    values["task_suite_draft"] = task_suite.to_dict()
                else:
                    raise ValueError(
                        "task_suite must be a digest, mapping, or V1 artifact"
                    )
            values.setdefault("study_id", study.id)
            values.setdefault("campaign_id", study.campaign_id)
            values.setdefault("proposal_id", f"proposal-{stable_digest(values)[:12]}")
            draft_value = build_experiment_draft(**values)
        elif isinstance(draft, ExperimentDraftV1):
            if values:
                raise ValueError("pass an experiment draft or keyword fields, not both")
            draft_value = draft
        else:
            if values:
                raise ValueError("pass an experiment draft or keyword fields, not both")
            raw = dict(draft)
            raw.setdefault("study_id", study.id)
            raw.setdefault("campaign_id", study.campaign_id)
            raw.setdefault("schema_version", 1)
            draft_value = build_experiment_draft(
                **{key: value for key, value in raw.items() if key != "schema_version"}
            )
        return self.study.client.service.preview_experiment(study.id, draft_value)

    def start(
        self, preview: ExperimentPreviewV1, *, idempotency_key: str
    ) -> ExperimentHandle:
        record = self.study.client.service.start_experiment(
            preview, idempotency_key=idempotency_key
        )
        return ExperimentHandle(self.study.client.service, record.id)

    def get(self, experiment_id: str) -> ExperimentHandle:
        return self.study.client.experiment(experiment_id)

    def list(self, *, limit: int = 100) -> tuple[Any, ...]:
        return self.study.client.service.store.list_experiments(
            self.study.id, limit=limit
        )


def _attribution(
    value: AttributionV1 | Mapping[str, Any] | None,
) -> AttributionV1:
    if value is None:
        return AttributionV1()
    if isinstance(value, AttributionV1):
        return value
    from fugue.research.contracts import attribution_from_dict

    return attribution_from_dict(value)


def _ref(
    value: EvidenceRefV1 | Mapping[str, Any] | str,
    *,
    default_kind: str | None = None,
) -> EvidenceRefV1:
    if isinstance(value, EvidenceRefV1):
        return value
    if isinstance(value, str):
        if not default_kind:
            raise ValueError("string evidence references require a default kind")
        return evidence_ref_from_dict({"kind": default_kind, "ref": value})
    return evidence_ref_from_dict(value)
