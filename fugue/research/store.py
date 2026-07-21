from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.library import validate_id
from fugue.research.contracts import (
    RESEARCH_SCHEMA_VERSION,
    TERMINAL_EXPERIMENT_STATES,
    AttributionV1,
    EvidenceRefV1,
    ExperimentEventV1,
    ExperimentRecordV1,
    ResearchError,
    StudyBriefV1,
    StudyContextV1,
    StudyExperimentRefV1,
    StudyNoteV1,
    StudyResourceV1,
    StudyUpdateV1,
    StudyV1,
    brief_from_dict,
    experiment_event_from_dict,
    experiment_record_from_dict,
    now,
    resource_from_dict,
    result_from_dict,
    sign_context,
    sign_event,
    sign_record,
    sign_study,
    study_from_dict,
)
from fugue.research.database import connect_database


class StudyStore:
    """Transactional research memory and operational experiment index."""

    def __init__(self, repo_root: Path, database: Path | None = None) -> None:
        self.repo_root = repo_root.resolve()
        self.path = (database or self.repo_root / ".fugue" / "research.db").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS studies (
                    study_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS study_events (
                    study_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    operation_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (study_id, revision),
                    UNIQUE (study_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS operations (
                    scope_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    input_digest TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, operation_id)
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS experiments_study
                    ON experiments(study_id, created_at);
                CREATE INDEX IF NOT EXISTS experiments_queue
                    ON experiments(state, lease_expires_at, created_at);
                CREATE TABLE IF NOT EXISTS experiment_events (
                    experiment_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (experiment_id, sequence),
                    UNIQUE (experiment_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS schema_info (
                    schema_version INTEGER NOT NULL
                );
                """
            )
            version = conn.execute("SELECT schema_version FROM schema_info").fetchone()
            if version is None:
                conn.execute(
                    "INSERT INTO schema_info(schema_version) VALUES (?)",
                    (RESEARCH_SCHEMA_VERSION,),
                )
            elif int(version[0]) != RESEARCH_SCHEMA_VERSION:
                raise RuntimeError("unsupported Fugue research database schema")

    def _connect(self) -> Any:
        return connect_database(self.path)

    def create_study(
        self,
        *,
        study_id: str,
        title: str,
        campaign_id: str,
        question: str,
        background: str = "",
        parent_study_ids: Iterable[str] = (),
        attribution: AttributionV1 | None = None,
        operation_id: str,
    ) -> StudyV1:
        study_id = validate_id(study_id, kind="study id")
        operation_id = validate_id(operation_id, kind="operation id")
        actor = attribution or AttributionV1()
        parents = tuple(parent_study_ids)
        created_at = now()
        unsigned = StudyV1(
            schema_version=RESEARCH_SCHEMA_VERSION,
            id=study_id,
            title=title,
            campaign_id=campaign_id,
            brief=StudyBriefV1(question=question, background=background),
            revision=1,
            notes=(),
            resources=(),
            results=(),
            experiments=(),
            run_refs=(),
            baseline_refs=(),
            primary_baseline_ref=None,
            parent_study_ids=parents,
            created_at=created_at,
            updated_at=created_at,
            created_by=actor,
            updated_by=actor,
        )
        study = study_from_dict(sign_study(unsigned).to_dict())
        request_digest = stable_digest(
            {
                "action": "create_study",
                "study_id": study_id,
                "title": title,
                "campaign_id": campaign_id,
                "question": question,
                "background": background,
                "parent_study_ids": list(parents),
                "attribution": actor.to_dict(),
            }
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = self._operation(conn, study_id, operation_id)
            if existing:
                return study_from_dict(
                    self._operation_response(
                        existing, "create_study", request_digest, operation_id
                    )
                )
            if conn.execute(
                "SELECT 1 FROM studies WHERE study_id=?", (study_id,)
            ).fetchone():
                raise ResearchError(
                    "study_exists",
                    f"study already exists: {study_id}",
                    category="conflict",
                )
            for parent_id in parents:
                if not conn.execute(
                    "SELECT 1 FROM studies WHERE study_id=?", (parent_id,)
                ).fetchone():
                    raise ResearchError(
                        "unknown_parent",
                        f"parent Study does not exist: {parent_id}",
                    )
            payload = self._json(study.to_dict())
            conn.execute(
                "INSERT INTO studies VALUES (?, ?, ?, ?, ?)",
                (study.id, study.revision, payload, study.created_at, study.updated_at),
            )
            self._append_study_event(
                conn, study, "study_created", operation_id, {"study": study.to_dict()}
            )
            self._record_operation(
                conn,
                study_id,
                operation_id,
                "create_study",
                request_digest,
                study.to_dict(),
            )
            conn.commit()
        self._checkpoint(study)
        return study

    def get_study(self, study_id: str) -> StudyV1:
        study_id = validate_id(study_id, kind="study id")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT snapshot_json FROM studies WHERE study_id=?", (study_id,)
            ).fetchone()
        if row is None:
            raise ResearchError("study_not_found", f"study not found: {study_id}")
        return study_from_dict(json.loads(row[0]))

    def list_studies(self, *, limit: int = 100) -> tuple[StudyV1, ...]:
        if limit < 1 or limit > 1000:
            raise ValueError("study limit must be between 1 and 1000")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT snapshot_json FROM studies ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(study_from_dict(json.loads(row[0])) for row in rows)

    def update_study(
        self,
        study_id: str,
        update: StudyUpdateV1,
        *,
        operation_id: str,
        expected_revision: int | None = None,
    ) -> StudyV1:
        study_id = validate_id(study_id, kind="study id")
        operation_id = validate_id(operation_id, kind="operation id")
        request_digest = stable_digest(
            {
                "action": "update_study",
                "study_id": study_id,
                "update": update.to_dict(),
                "expected_revision": expected_revision,
            }
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = self._operation(conn, study_id, operation_id)
            if existing:
                return study_from_dict(
                    self._operation_response(
                        existing, "update_study", request_digest, operation_id
                    )
                )
            row = conn.execute(
                "SELECT snapshot_json FROM studies WHERE study_id=?", (study_id,)
            ).fetchone()
            if row is None:
                raise ResearchError("study_not_found", f"study not found: {study_id}")
            current = study_from_dict(json.loads(row[0]))
            if expected_revision is not None and current.revision != expected_revision:
                raise ResearchError(
                    "revision_conflict",
                    f"study revision is {current.revision}, not {expected_revision}",
                    category="conflict",
                    retryable=True,
                )
            updated = self._apply_update(current, update)
            conn.execute(
                "UPDATE studies SET revision=?, snapshot_json=?, updated_at=? WHERE study_id=?",
                (
                    updated.revision,
                    self._json(updated.to_dict()),
                    updated.updated_at,
                    study_id,
                ),
            )
            self._append_study_event(
                conn,
                updated,
                "study_updated",
                operation_id,
                {
                    "update": update.to_dict(),
                    "study": updated.to_dict(),
                    "study_digest": updated.study_digest,
                },
            )
            self._record_operation(
                conn,
                study_id,
                operation_id,
                "update_study",
                request_digest,
                updated.to_dict(),
            )
            conn.commit()
        self._checkpoint(updated)
        return updated

    def context(
        self,
        study_id: str,
        *,
        max_experiments: int = 20,
        max_results: int = 20,
        max_notes: int = 20,
        max_chars: int = 32000,
    ) -> StudyContextV1:
        for label, value in (
            ("max_experiments", max_experiments),
            ("max_results", max_results),
            ("max_notes", max_notes),
        ):
            if value < 0 or value > 1000:
                raise ValueError(f"{label} must be between 0 and 1000")
        if max_chars < 1000 or max_chars > 1_000_000:
            raise ValueError("max_chars must be between 1000 and 1000000")
        study = self.get_study(study_id)
        superseded_results = {
            item.supersedes for item in study.results if item.supersedes
        }
        visible_results = [
            item for item in study.results if item.id not in superseded_results
        ]
        visible_results = visible_results[-max_results:] if max_results else []
        superseded_notes = {item.supersedes for item in study.notes if item.supersedes}
        cited_notes = {
            ref.ref
            for refs in study.brief.provenance.values()
            for ref in refs
            if ref.kind == "note"
        }
        current_notes = [
            item for item in study.notes if item.id not in superseded_notes
        ]
        selected_notes = [item for item in current_notes if item.id in cited_notes]
        selected_ids = {item.id for item in selected_notes}
        selected_notes.extend(
            item
            for item in (current_notes[-max_notes:] if max_notes else [])
            if item.id not in selected_ids
        )
        selected_notes = selected_notes[-max_notes:] if max_notes else []
        experiments = (
            list(study.experiments[-max_experiments:]) if max_experiments else []
        )
        experiment_context = tuple(
            self._experiment_context(item) for item in experiments
        )
        cited_resources = {
            ref.ref
            for refs in study.brief.provenance.values()
            for ref in refs
            if ref.kind == "resource"
        }
        resources = [item for item in study.resources if item.id in cited_resources]
        resource_ids = {item.id for item in resources}
        resources.extend(
            item
            for item in (study.resources[-max_notes:] if max_notes else [])
            if item.id not in resource_ids
        )
        resources = resources[-max_notes:] if max_notes else []
        context = StudyContextV1(
            schema_version=RESEARCH_SCHEMA_VERSION,
            study_id=study.id,
            revision=study.revision,
            title=study.title,
            campaign_id=study.campaign_id,
            brief=study.brief.to_dict(),
            baseline={
                "primary": (
                    study.primary_baseline_ref.to_dict()
                    if study.primary_baseline_ref
                    else None
                ),
                "references": [item.to_dict() for item in study.baseline_refs],
            },
            experiments=experiment_context,
            results=tuple(item.to_dict() for item in visible_results),
            notes=tuple(item.to_dict() for item in selected_notes),
            resources=tuple(item.to_dict() for item in resources),
            omissions={
                "experiments": max(0, len(study.experiments) - len(experiments)),
                "results": max(0, len(study.results) - len(visible_results)),
                "notes": max(0, len(study.notes) - len(selected_notes)),
                "resources": max(0, len(study.resources) - len(resources)),
            },
        )
        while len(self._json(context.to_dict())) > max_chars:
            if context.notes:
                context = replace(
                    context,
                    notes=context.notes[1:],
                    omissions={
                        **context.omissions,
                        "notes": context.omissions["notes"] + 1,
                    },
                )
            elif context.resources:
                context = replace(
                    context,
                    resources=context.resources[1:],
                    omissions={
                        **context.omissions,
                        "resources": context.omissions["resources"] + 1,
                    },
                )
            elif context.results:
                context = replace(
                    context,
                    results=context.results[1:],
                    omissions={
                        **context.omissions,
                        "results": context.omissions["results"] + 1,
                    },
                )
            elif context.experiments:
                context = replace(
                    context,
                    experiments=context.experiments[1:],
                    omissions={
                        **context.omissions,
                        "experiments": context.omissions["experiments"] + 1,
                    },
                )
            else:
                raise ResearchError(
                    "context_too_large",
                    "the current Study brief exceeds the context character limit",
                )
        return sign_context(context)

    def _experiment_context(self, item: StudyExperimentRefV1) -> dict[str, Any]:
        """Project enough experiment meaning for an outer loop to follow lineage."""
        values = item.to_dict()
        record = self.get_experiment(item.experiment_id)
        draft = record.draft
        for key in (
            "stage_id",
            "question",
            "hypothesis",
            "decision_rationale",
            "parent_outcome_id",
        ):
            value = draft.get(key)
            if value not in (None, "", [], {}):
                values[key] = value
        return values

    def study_events(self, study_id: str) -> tuple[dict[str, Any], ...]:
        study_id = validate_id(study_id, kind="study id")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT revision, event_id, event_type, operation_id, payload_json, "
                "created_at FROM study_events WHERE study_id=? ORDER BY revision",
                (study_id,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for expected, row in enumerate(rows, 1):
            if int(row["revision"]) != expected:
                raise ResearchError(
                    "study_event_sequence_invalid",
                    "Study event revisions are not contiguous",
                    category="evidence",
                )
            events.append(
                {
                    "study_id": study_id,
                    "revision": expected,
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "operation_id": row["operation_id"],
                    "payload": json.loads(row["payload_json"]),
                    "created_at": row["created_at"],
                }
            )
        return tuple(events)

    def reconstruct_study(
        self, study_id: str, *, revision: int | None = None
    ) -> StudyV1:
        events = self.study_events(study_id)
        selected = [
            item for item in events if revision is None or item["revision"] <= revision
        ]
        if not selected:
            raise ResearchError(
                "study_revision_not_found",
                f"Study {study_id} has no requested revision",
            )
        raw = selected[-1]["payload"].get("study")
        if not isinstance(raw, Mapping):
            raise ResearchError(
                "study_event_corrupt",
                "Study event does not contain its immutable revision snapshot",
                category="evidence",
            )
        return study_from_dict(raw)

    def insert_experiment(
        self,
        record: ExperimentRecordV1,
        *,
        operation_id: str,
        input_digest: str,
    ) -> ExperimentRecordV1:
        operation_id = validate_id(operation_id, kind="operation id")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = self._operation(conn, record.study_id, operation_id)
            if existing:
                return experiment_record_from_dict(
                    self._operation_response(
                        existing, "start_experiment", input_digest, operation_id
                    )
                )
            if conn.execute(
                "SELECT 1 FROM experiments WHERE experiment_id=?", (record.id,)
            ).fetchone():
                raise ResearchError(
                    "experiment_exists",
                    f"experiment already exists: {record.id}",
                    category="conflict",
                )
            self._validate_parents(conn, record)
            payload = self._json(record.to_dict())
            conn.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)",
                (
                    record.id,
                    record.study_id,
                    record.state,
                    payload,
                    record.created_at,
                    record.updated_at,
                ),
            )
            self._append_experiment_event(
                conn,
                record,
                state=record.state,
                event_type="experiment_queued",
                message="Experiment accepted into the governed execution queue.",
            )
            self._record_operation(
                conn,
                record.study_id,
                operation_id,
                "start_experiment",
                input_digest,
                record.to_dict(),
            )
            conn.commit()
        self.sync_experiment_reference(record, terminal=False)
        return record

    def get_experiment(self, experiment_id: str) -> ExperimentRecordV1:
        experiment_id = validate_id(experiment_id, kind="experiment record id")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT record_json FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            raise ResearchError(
                "experiment_not_found", f"experiment not found: {experiment_id}"
            )
        return experiment_record_from_dict(json.loads(row[0]))

    def list_experiments(
        self, study_id: str, *, limit: int = 100
    ) -> tuple[ExperimentRecordV1, ...]:
        study_id = validate_id(study_id, kind="study id")
        if limit < 1 or limit > 1000:
            raise ValueError("experiment limit must be between 1 and 1000")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record_json FROM experiments WHERE study_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (study_id, limit),
            ).fetchall()
        return tuple(experiment_record_from_dict(json.loads(row[0])) for row in rows)

    def update_experiment(
        self,
        record: ExperimentRecordV1,
        *,
        worker_id: str | None = None,
        event_type: str,
        message: str,
        artifact_type: str | None = None,
        artifact_digest: str | None = None,
        release: bool = False,
        lease_seconds: float = 30,
    ) -> ExperimentRecordV1:
        self._validate_lease_seconds(lease_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT lease_owner, lease_expires_at FROM experiments "
                "WHERE experiment_id=?",
                (record.id,),
            ).fetchone()
            if row is None:
                raise ResearchError(
                    "experiment_not_found", f"experiment not found: {record.id}"
                )
            if worker_id and (
                row[0] != worker_id or not self._lease_is_current(row[1])
            ):
                raise ResearchError(
                    "lease_lost",
                    "experiment worker no longer owns a current lease",
                    category="conflict",
                    retryable=True,
                )
            conn.execute(
                "UPDATE experiments SET state=?, record_json=?, updated_at=?, "
                "lease_owner=?, lease_expires_at=? WHERE experiment_id=?",
                (
                    record.state,
                    self._json(record.to_dict()),
                    record.updated_at,
                    None if release else row[0],
                    None if release else self._lease_expiry(lease_seconds),
                    record.id,
                ),
            )
            self._append_experiment_event(
                conn,
                record,
                state=record.state,
                event_type=event_type,
                message=message,
                artifact_type=artifact_type,
                artifact_digest=artifact_digest,
            )
            conn.commit()
        if record.state in TERMINAL_EXPERIMENT_STATES:
            self.sync_experiment_reference(record, terminal=True)
        return record

    def record_cancellation(
        self,
        record: ExperimentRecordV1,
        *,
        operation_id: str,
        input_digest: str,
        prelaunch: bool,
    ) -> ExperimentRecordV1:
        operation_id = validate_id(operation_id, kind="operation id")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = self._operation(conn, record.id, operation_id)
            if existing:
                return experiment_record_from_dict(
                    self._operation_response(
                        existing, "cancel_experiment", input_digest, operation_id
                    )
                )
            row = conn.execute(
                "SELECT record_json FROM experiments WHERE experiment_id=?",
                (record.id,),
            ).fetchone()
            if row is None:
                raise ResearchError(
                    "experiment_not_found", f"experiment not found: {record.id}"
                )
            current = experiment_record_from_dict(json.loads(row[0]))
            if current.state in TERMINAL_EXPERIMENT_STATES:
                updated = current
            else:
                updated = replace(
                    current,
                    state="cancelled" if prelaunch else "cancelling",
                    updated_at=now(),
                )
                updated = experiment_record_from_dict(sign_record(updated).to_dict())
                conn.execute(
                    "UPDATE experiments SET state=?, record_json=?, updated_at=?, "
                    "lease_owner=NULL, lease_expires_at=NULL WHERE experiment_id=?",
                    (
                        updated.state,
                        self._json(updated.to_dict()),
                        updated.updated_at,
                        updated.id,
                    ),
                )
                self._append_experiment_event(
                    conn,
                    updated,
                    state=updated.state,
                    event_type=(
                        "experiment_cancelled"
                        if prelaunch
                        else "cancellation_requested"
                    ),
                    message=(
                        "Experiment cancelled before a run was launched."
                        if prelaunch
                        else "Cancellation requested; terminal evidence will be reconciled."
                    ),
                )
            self._record_operation(
                conn,
                record.id,
                operation_id,
                "cancel_experiment",
                input_digest,
                updated.to_dict(),
            )
            conn.commit()
        if updated.state in TERMINAL_EXPERIMENT_STATES:
            self.sync_experiment_reference(updated, terminal=True)
        return updated

    def claim_experiment(
        self, worker_id: str, *, lease_seconds: float = 30
    ) -> ExperimentRecordV1 | None:
        worker_id = validate_id(worker_id, kind="worker id")
        self._validate_lease_seconds(lease_seconds)
        cutoff = now()
        terminal = tuple(TERMINAL_EXPERIMENT_STATES)
        placeholders = ",".join("?" for _ in terminal)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"SELECT experiment_id, record_json FROM experiments "
                f"WHERE state NOT IN ({placeholders}) "
                "AND (lease_owner IS NULL OR lease_expires_at < ?) "
                "ORDER BY created_at LIMIT 1",
                (*terminal, cutoff),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            expiry = (
                (datetime.now(UTC) + timedelta(seconds=lease_seconds))
                .isoformat()
                .replace("+00:00", "Z")
            )
            conn.execute(
                "UPDATE experiments SET lease_owner=?, lease_expires_at=? "
                "WHERE experiment_id=?",
                (worker_id, expiry, row[0]),
            )
            conn.commit()
        return experiment_record_from_dict(json.loads(row[1]))

    def renew_lease(
        self, experiment_id: str, worker_id: str, *, lease_seconds: float = 30
    ) -> None:
        self._validate_lease_seconds(lease_seconds)
        cutoff = now()
        expiry = (
            (datetime.now(UTC) + timedelta(seconds=lease_seconds))
            .isoformat()
            .replace("+00:00", "Z")
        )
        with self._connect() as conn:
            changed = conn.execute(
                "UPDATE experiments SET lease_expires_at=? "
                "WHERE experiment_id=? AND lease_owner=? AND lease_expires_at>=?",
                (expiry, experiment_id, worker_id, cutoff),
            ).rowcount
        if changed != 1:
            raise ResearchError(
                "lease_lost",
                "experiment worker lease was lost",
                category="conflict",
                retryable=True,
            )

    def assert_lease(self, experiment_id: str, worker_id: str) -> None:
        cutoff = now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM experiments WHERE experiment_id=? "
                "AND lease_owner=? AND lease_expires_at>=?",
                (experiment_id, worker_id, cutoff),
            ).fetchone()
        if row is None:
            raise ResearchError(
                "lease_lost",
                "experiment worker lease was lost",
                category="conflict",
                retryable=True,
            )

    def release_lease(self, experiment_id: str, worker_id: str) -> None:
        with self._connect() as conn:
            changed = conn.execute(
                "UPDATE experiments SET lease_owner=NULL, lease_expires_at=NULL "
                "WHERE experiment_id=? AND lease_owner=?",
                (experiment_id, worker_id),
            ).rowcount
        if changed != 1:
            raise ResearchError(
                "lease_lost",
                "experiment worker lease was lost",
                category="conflict",
                retryable=True,
            )

    def events(
        self, experiment_id: str, *, after: int = 0, limit: int = 1000
    ) -> tuple[ExperimentEventV1, ...]:
        experiment_id = validate_id(experiment_id, kind="experiment record id")
        if after < 0:
            raise ValueError("event cursor must be non-negative")
        if limit < 1 or limit > 1000:
            raise ValueError("event limit must be between 1 and 1000")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_json FROM experiment_events "
                "WHERE experiment_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                (experiment_id, after, limit),
            ).fetchall()
        return tuple(experiment_event_from_dict(json.loads(row[0])) for row in rows)

    def latest_event(self, experiment_id: str) -> ExperimentEventV1 | None:
        experiment_id = validate_id(experiment_id, kind="experiment record id")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT event_json FROM experiment_events "
                "WHERE experiment_id=? ORDER BY sequence DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
        return experiment_event_from_dict(json.loads(row[0])) if row else None

    def sync_experiment_reference(
        self, record: ExperimentRecordV1, *, terminal: bool
    ) -> None:
        study = self.get_study(record.study_id)
        ref = self._study_experiment_ref(record)
        current = {item.experiment_id: item for item in study.experiments}
        if current.get(record.id) == ref:
            return
        operation_id = f"sync-{record.id}-{'terminal' if terminal else 'created'}"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            latest_row = conn.execute(
                "SELECT snapshot_json FROM studies WHERE study_id=?", (study.id,)
            ).fetchone()
            if latest_row is None:
                raise ResearchError("study_not_found", f"study not found: {study.id}")
            latest = study_from_dict(json.loads(latest_row[0]))
            latest_refs = {item.experiment_id: item for item in latest.experiments}
            if latest_refs.get(record.id) == ref:
                conn.commit()
                return
            latest_refs[record.id] = ref
            ids = [
                item.experiment_id
                for item in latest.experiments
                if item.experiment_id != record.id
            ]
            ids.append(record.id)
            updated = sign_study(
                replace(
                    latest,
                    revision=latest.revision + 1,
                    experiments=tuple(latest_refs[item] for item in ids),
                    run_refs=self._terminal_run_refs(latest, record, terminal),
                    updated_at=now(),
                    updated_by=AttributionV1(
                        actor_type="service", name="fugue-research"
                    ),
                )
            )
            conn.execute(
                "UPDATE studies SET revision=?, snapshot_json=?, updated_at=? WHERE study_id=?",
                (
                    updated.revision,
                    self._json(updated.to_dict()),
                    updated.updated_at,
                    updated.id,
                ),
            )
            self._append_study_event(
                conn,
                updated,
                "experiment_attached" if not terminal else "experiment_terminal",
                operation_id,
                {"experiment": ref.to_dict(), "study": updated.to_dict()},
            )
            conn.commit()
        self._checkpoint(updated)

    def _apply_update(self, current: StudyV1, update: StudyUpdateV1) -> StudyV1:
        revision = current.revision + 1
        timestamp = now()
        notes = list(current.notes)
        resources = list(current.resources)
        results = list(current.results)
        note: StudyNoteV1 | None = None
        if update.note_supersedes and update.note_supersedes not in {
            item.id for item in notes
        }:
            raise ResearchError("unknown_source", "superseded note is not in the Study")
        if update.message:
            note = StudyNoteV1(
                id=f"note-{revision}",
                revision=revision,
                text=update.message,
                kind=update.note_kind,
                sources=update.note_sources,
                supersedes=update.note_supersedes,
                created_at=timestamp,
                attribution=update.attribution,
            )
            notes.append(note)

        resource_ids = {item.id for item in resources}
        for draft in update.resources:
            unknown = set(draft) - {
                "id",
                "uri",
                "kind",
                "digest",
                "version",
                "title",
                "summary",
            }
            if unknown:
                raise ResearchError(
                    "unknown_field",
                    f"study resource has unknown fields: {', '.join(sorted(unknown))}",
                )
            resource_id = validate_id(str(draft.get("id") or ""), kind="resource id")
            if resource_id in resource_ids:
                raise ResearchError(
                    "immutable_resource", f"resource already exists: {resource_id}"
                )
            resource = StudyResourceV1(
                id=resource_id,
                uri=str(draft.get("uri") or ""),
                kind=str(draft.get("kind") or "other"),
                digest=str(draft.get("digest") or "") or None,
                version=str(draft.get("version") or "") or None,
                title=str(draft.get("title") or ""),
                summary=str(draft.get("summary") or ""),
                added_revision=revision,
                added_at=timestamp,
                attribution=update.attribution,
            )
            resource = resource_from_dict(resource.to_dict())
            resources.append(resource)
            resource_ids.add(resource.id)

        result_ids = {item.id for item in results}
        for draft in update.results:
            result_id = validate_id(str(draft.get("id") or ""), kind="result id")
            if result_id in result_ids:
                raise ResearchError(
                    "immutable_result", f"result already exists: {result_id}"
                )
            value = {
                **draft,
                "revision": revision,
                "created_at": timestamp,
                "attribution": update.attribution.to_dict(),
            }
            result = result_from_dict(value)
            if result.supersedes and result.supersedes not in result_ids:
                raise ResearchError(
                    "unknown_source", "superseded result is not in the Study"
                )
            results.append(result)
            result_ids.add(result.id)

        brief_values = current.brief.to_dict()
        provenance = dict(current.brief.provenance)
        for key, value in update.brief_patch.items():
            sources = update.brief_sources.get(key)
            if sources is None:
                sources = (EvidenceRefV1(kind="note", ref=note.id),) if note else ()
            if not sources:
                raise ResearchError(
                    "missing_provenance",
                    f"brief field {key} requires an exact source",
                )
            brief_values[key] = value
            provenance[key] = sources
        brief_values["provenance"] = {
            key: [item.to_dict() for item in value] for key, value in provenance.items()
        }
        brief = brief_from_dict(brief_values)

        run_refs = self._merge_refs(current.run_refs, update.run_refs)
        baseline_refs = (
            current.baseline_refs
            if update.baseline_refs is None
            else tuple(update.baseline_refs)
        )
        primary = (
            current.primary_baseline_ref
            if update.baseline_refs is None and update.primary_baseline_ref is None
            else update.primary_baseline_ref
        )
        if baseline_refs and primary is None:
            if len(baseline_refs) == 1:
                primary = baseline_refs[0]
            else:
                raise ResearchError(
                    "baseline_primary_required",
                    "multiple baseline references require a primary baseline",
                )
        if primary and self._ref_key(primary) not in {
            self._ref_key(item) for item in baseline_refs
        }:
            raise ResearchError(
                "baseline_mismatch", "primary baseline must be one of baseline_refs"
            )

        candidate = sign_study(
            replace(
                current,
                brief=brief,
                revision=revision,
                notes=tuple(notes),
                resources=tuple(resources),
                results=tuple(results),
                run_refs=run_refs,
                baseline_refs=baseline_refs,
                primary_baseline_ref=primary,
                updated_at=timestamp,
                updated_by=update.attribution,
            )
        )
        parsed = study_from_dict(candidate.to_dict())
        self._validate_local_sources(parsed)
        return parsed

    def _validate_local_sources(self, study: StudyV1) -> None:
        ids = {
            "note": {item.id for item in study.notes},
            "resource": {item.id for item in study.resources},
            "result": {item.id for item in study.results},
        }
        refs = [
            *[item for values in study.brief.provenance.values() for item in values],
            *[item for note in study.notes for item in note.sources],
            *[item for result in study.results for item in result.sources],
        ]
        for ref in refs:
            if ref.kind in ids and ref.ref not in ids[ref.kind]:
                raise ResearchError(
                    "unknown_source",
                    f"{ref.kind} source is not in the Study: {ref.ref}",
                )

    def _validate_parents(
        self, conn: sqlite3.Connection, record: ExperimentRecordV1
    ) -> None:
        if record.id in record.parent_experiment_ids:
            raise ResearchError("lineage_cycle", "an experiment cannot parent itself")
        for parent_id in record.parent_experiment_ids:
            row = conn.execute(
                "SELECT study_id FROM experiments WHERE experiment_id=?", (parent_id,)
            ).fetchone()
            if row is None or row[0] != record.study_id:
                raise ResearchError(
                    "unknown_parent",
                    f"parent experiment is not in this Study: {parent_id}",
                )

    def _study_experiment_ref(self, record: ExperimentRecordV1) -> StudyExperimentRefV1:
        return StudyExperimentRefV1(
            experiment_id=record.id,
            state=record.state,
            proposal_digest=(record.proposal or {}).get("proposal_digest"),
            preview_digest=str(record.preview["preview_digest"]),
            task_suite_digest=(record.task_suite_lock or {}).get("suite_digest")
            or record.draft.get("task_suite_digest"),
            plan_digest=(record.plan or {}).get("plan_digest"),
            parent_experiment_ids=record.parent_experiment_ids,
            run_id=record.run_id,
            outcome_digest=(record.outcome or {}).get("outcome_digest"),
            evaluation_digest=(record.evaluation or {}).get("evaluation_digest"),
            analysis_digest=(record.analysis or {}).get("analysis_digest"),
            updated_at=record.updated_at,
        )

    def _terminal_run_refs(
        self, study: StudyV1, record: ExperimentRecordV1, terminal: bool
    ) -> tuple[EvidenceRefV1, ...]:
        if not terminal or not record.run_id or not record.outcome:
            return study.run_refs
        additions = [
            EvidenceRefV1(
                kind="run",
                ref=record.run_id,
                digest=record.outcome.get("run_snapshot_sha256"),
            ),
            EvidenceRefV1(
                kind="outcome",
                ref=str(record.outcome.get("outcome_id") or record.id),
                digest=record.outcome.get("outcome_digest"),
            ),
        ]
        return self._merge_refs(study.run_refs, additions)

    def _append_study_event(
        self,
        conn: sqlite3.Connection,
        study: StudyV1,
        event_type: str,
        operation_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        conn.execute(
            "INSERT INTO study_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                study.id,
                study.revision,
                f"study-{study.revision}",
                event_type,
                operation_id,
                self._json(payload),
                study.updated_at,
            ),
        )

    def _append_experiment_event(
        self,
        conn: sqlite3.Connection,
        record: ExperimentRecordV1,
        *,
        state: str,
        event_type: str,
        message: str,
        artifact_type: str | None = None,
        artifact_digest: str | None = None,
    ) -> ExperimentEventV1:
        row = conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM experiment_events WHERE experiment_id=?",
            (record.id,),
        ).fetchone()
        sequence = int(row[0]) + 1
        unsigned = ExperimentEventV1(
            schema_version=RESEARCH_SCHEMA_VERSION,
            event_id=f"event-{sequence}",
            sequence=sequence,
            study_id=record.study_id,
            experiment_id=record.id,
            state=state,
            event_type=event_type,
            message=message,
            artifact_type=artifact_type,
            artifact_digest=artifact_digest,
            created_at=now(),
        )
        event = sign_event(unsigned)
        conn.execute(
            "INSERT INTO experiment_events VALUES (?, ?, ?, ?, ?)",
            (
                record.id,
                sequence,
                event.event_id,
                self._json(event.to_dict()),
                event.created_at,
            ),
        )
        return event

    def _operation(
        self, conn: sqlite3.Connection, scope_id: str, operation_id: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM operations WHERE scope_id=? AND operation_id=?",
            (scope_id, operation_id),
        ).fetchone()

    def _operation_response(
        self,
        row: sqlite3.Row,
        action: str,
        input_digest: str,
        operation_id: str,
    ) -> dict[str, Any]:
        if row["action"] != action or row["input_digest"] != input_digest:
            raise ResearchError(
                "idempotency_conflict",
                f"operation id {operation_id!r} was already used with different input",
                category="conflict",
            )
        return json.loads(row["response_json"])

    def _record_operation(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        operation_id: str,
        action: str,
        input_digest: str,
        response: Mapping[str, Any],
    ) -> None:
        conn.execute(
            "INSERT INTO operations VALUES (?, ?, ?, ?, ?, ?)",
            (scope_id, operation_id, action, input_digest, self._json(response), now()),
        )

    def _checkpoint(self, study: StudyV1) -> None:
        destination = self.repo_root / ".fugue" / "studies" / study.id / "study.json"
        atomic_write_json(destination, study.to_dict())

    def _lease_expiry(self, seconds: float = 30) -> str:
        return (
            (datetime.now(UTC) + timedelta(seconds=seconds))
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _validate_lease_seconds(seconds: float) -> None:
        if not 0 < seconds <= 3600:
            raise ValueError("lease duration must be between 0 and 3600 seconds")

    @staticmethod
    def _lease_is_current(expires_at: str | None) -> bool:
        return bool(expires_at and expires_at >= now())

    @staticmethod
    def _json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @staticmethod
    def _ref_key(value: EvidenceRefV1) -> str:
        return json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":"))

    def _merge_refs(
        self,
        existing: Iterable[EvidenceRefV1],
        additions: Iterable[EvidenceRefV1],
    ) -> tuple[EvidenceRefV1, ...]:
        values: dict[str, EvidenceRefV1] = {
            self._ref_key(item): item for item in existing
        }
        for item in additions:
            values.setdefault(self._ref_key(item), item)
        return tuple(values.values())
