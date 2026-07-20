from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fugue.bench.candidates import stable_digest
from fugue.bench.library import validate_id
from fugue.research.agent_contracts import (
    ExecutionApprovalV1,
    execution_approval_from_dict,
    sign_execution_approval,
)
from fugue.research.contracts import RESEARCH_SCHEMA_VERSION, ResearchError, now


class ApprovalLedger:
    """Operator-owned approvals bound to immutable preview digests."""

    def __init__(self, database: Path) -> None:
        self.path = database.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS execution_approvals (
                    approval_digest TEXT PRIMARY KEY,
                    preview_digest TEXT NOT NULL,
                    subject_kind TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    consumed_by TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS approval_subject
                    ON execution_approvals(subject_kind, preview_digest);
                CREATE TABLE IF NOT EXISTS approval_operations (
                    operation_id TEXT PRIMARY KEY,
                    input_digest TEXT NOT NULL,
                    approval_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def approve(
        self,
        *,
        subject_kind: str,
        preview_digest: str,
        maximum_cost_usd: float,
        maximum_cells: int | None,
        approved_by: str,
        operation_id: str,
        expires_in_seconds: int = 3600,
    ) -> ExecutionApprovalV1:
        operation_id = validate_id(operation_id, kind="approval operation id")
        if expires_in_seconds < 60 or expires_in_seconds > 86_400:
            raise ValueError("approval expiry must be between 60 and 86400 seconds")
        created = datetime.now(UTC)
        approval_id = f"approval-{preview_digest[:20]}"
        unsigned = ExecutionApprovalV1(
            schema_version=RESEARCH_SCHEMA_VERSION,
            approval_id=approval_id,
            subject_kind=subject_kind,  # type: ignore[arg-type]
            preview_digest=preview_digest,
            maximum_cost_usd=float(maximum_cost_usd),
            maximum_cells=int(maximum_cells) if maximum_cells is not None else None,
            approved_by=approved_by,
            operation_id=operation_id,
            created_at=created.isoformat().replace("+00:00", "Z"),
            expires_at=(created + timedelta(seconds=expires_in_seconds))
            .isoformat()
            .replace("+00:00", "Z"),
        )
        approval = execution_approval_from_dict(
            sign_execution_approval(unsigned).to_dict()
        )
        input_digest = stable_digest(
            {
                "action": "approve_execution",
                "subject_kind": subject_kind,
                "preview_digest": preview_digest,
                "maximum_cost_usd": maximum_cost_usd,
                "maximum_cells": maximum_cells,
                "approved_by": approved_by,
                "expires_in_seconds": expires_in_seconds,
            }
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute(
                "SELECT input_digest, approval_digest FROM approval_operations "
                "WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if prior:
                if prior[0] != input_digest:
                    raise ResearchError(
                        "operation_conflict",
                        "approval operation id was reused with different input",
                        category="conflict",
                    )
                conn.commit()
                return self.get(str(prior[1]))
            existing = conn.execute(
                "SELECT receipt_json FROM execution_approvals "
                "WHERE subject_kind=? AND preview_digest=?",
                (subject_kind, preview_digest),
            ).fetchone()
            if existing:
                prior_approval = execution_approval_from_dict(json.loads(existing[0]))
                if prior_approval.to_dict() != approval.to_dict():
                    raise ResearchError(
                        "approval_conflict",
                        "the preview already has a different approval",
                        category="conflict",
                    )
                return prior_approval
            conn.execute(
                "INSERT INTO execution_approvals VALUES (?, ?, ?, ?, NULL, ?)",
                (
                    approval.approval_digest,
                    approval.preview_digest,
                    approval.subject_kind,
                    json.dumps(approval.to_dict(), sort_keys=True),
                    approval.created_at,
                ),
            )
            conn.execute(
                "INSERT INTO approval_operations VALUES (?, ?, ?, ?)",
                (
                    operation_id,
                    input_digest,
                    approval.approval_digest,
                    now(),
                ),
            )
            conn.commit()
        return approval

    def get(self, approval_digest: str) -> ExecutionApprovalV1:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT receipt_json FROM execution_approvals WHERE approval_digest=?",
                (approval_digest,),
            ).fetchone()
        if row is None:
            raise ResearchError(
                "approval_not_found",
                "execution approval was not found",
                category="policy",
            )
        return execution_approval_from_dict(json.loads(row[0]))

    def claim(
        self,
        *,
        approval_digest: str,
        subject_kind: str,
        preview_digest: str,
        subject_id: str,
        estimated_cells: int = 0,
        estimated_cost_usd: float = 0.0,
    ) -> ExecutionApprovalV1:
        subject_id = validate_id(subject_id, kind="approved subject id")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT receipt_json, consumed_by FROM execution_approvals "
                "WHERE approval_digest=?",
                (approval_digest,),
            ).fetchone()
            if row is None:
                raise ResearchError(
                    "approval_not_found",
                    "execution approval was not found",
                    category="policy",
                )
            approval = execution_approval_from_dict(json.loads(row[0]))
            if approval.subject_kind != subject_kind:
                raise ResearchError(
                    "approval_subject_mismatch",
                    "execution approval belongs to another operation kind",
                    category="policy",
                )
            if approval.preview_digest != preview_digest:
                raise ResearchError(
                    "approval_preview_mismatch",
                    "execution approval does not match the accepted preview",
                    category="policy",
                )
            if _parse_time(approval.expires_at) <= datetime.now(UTC):
                raise ResearchError(
                    "approval_expired",
                    "execution approval has expired",
                    category="policy",
                )
            if (
                approval.maximum_cells is not None
                and estimated_cells > approval.maximum_cells
            ):
                raise ResearchError(
                    "approval_cell_limit",
                    "accepted preview exceeds the approved cell limit",
                    category="policy",
                )
            if estimated_cost_usd > approval.maximum_cost_usd + 1e-9:
                raise ResearchError(
                    "approval_cost_limit",
                    "accepted preview exceeds the approved cost limit",
                    category="policy",
                )
            consumed_by = row[1]
            if consumed_by and consumed_by != subject_id:
                raise ResearchError(
                    "approval_consumed",
                    "execution approval was already consumed by another operation",
                    category="policy",
                )
            if not consumed_by:
                conn.execute(
                    "UPDATE execution_approvals SET consumed_by=? "
                    "WHERE approval_digest=? AND consumed_by IS NULL",
                    (subject_id, approval_digest),
                )
            conn.commit()
        return approval

    def require_cost(self, approval_digest: str, reserved_cost_usd: float) -> None:
        approval = self.get(approval_digest)
        if reserved_cost_usd > approval.maximum_cost_usd + 1e-9:
            raise ResearchError(
                "approval_cost_limit",
                "campaign reservation exceeds the operator-approved cost limit",
                category="admission",
                details={
                    "approved_cost_usd": approval.maximum_cost_usd,
                    "reserved_cost_usd": reserved_cost_usd,
                },
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
