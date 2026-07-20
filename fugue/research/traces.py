from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.library import validate_id
from fugue.research.agent_contracts import (
    TraceAuditDraftV1,
    TraceAuditPreviewV1,
    TraceAuditV1,
    TraceSourceRefV1,
    sign_trace_audit,
    sign_trace_audit_preview,
    trace_audit_draft_from_dict,
    trace_audit_from_dict,
    trace_audit_preview_from_dict,
)
from fugue.research.approvals import ApprovalLedger
from fugue.research.contracts import RESEARCH_SCHEMA_VERSION, ResearchError, now
from fugue.research.store import StudyStore
from fugue.weave_support import resolved_weave_trace_server_url

_SOURCE_FIELDS = {
    "adapter",
    "allowed_fields",
    "allowed_filters",
    "id",
    "path",
    "project",
    "redactions",
}
_SECRET = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|password|secret|token)\s*[:=]\s*\S+"
)
_DEFAULT_FIELDS = (
    "status",
    "operation",
    "errors",
    "tools",
    "latency",
    "tokens",
    "cost",
)
_DEFAULT_REDACTIONS = (
    "credentials",
    "hidden_reasoning",
    "prompt_instructions",
    "unrequested_content",
)


class TraceSourceAdapter(Protocol):
    @property
    def source(self) -> TraceSourceRefV1: ...

    @property
    def available_fields(self) -> tuple[str, ...]: ...

    @property
    def allowed_filters(self) -> tuple[str, ...]: ...

    @property
    def redactions(self) -> tuple[str, ...]: ...

    def read(self, draft: TraceAuditDraftV1) -> tuple[dict[str, Any], ...]: ...


@dataclass(frozen=True)
class _SourceConfig:
    id: str
    adapter: str
    allowed_fields: tuple[str, ...]
    allowed_filters: tuple[str, ...]
    redactions: tuple[str, ...]
    path: Path | None = None
    project: str | None = None

    @property
    def safe_digest(self) -> str:
        return stable_digest(
            {
                "id": self.id,
                "adapter": self.adapter,
                "allowed_fields": list(self.allowed_fields),
                "allowed_filters": list(self.allowed_filters),
                "redactions": list(self.redactions),
                "project": self.project,
            }
        )


class JsonlTraceSource:
    def __init__(self, config: _SourceConfig) -> None:
        if config.path is None:
            raise ValueError("jsonl trace source requires an operator-configured path")
        self.config = config

    @property
    def source(self) -> TraceSourceRefV1:
        return TraceSourceRefV1(
            RESEARCH_SCHEMA_VERSION,
            self.config.id,
            "jsonl",
            self.config.safe_digest,
        )

    @property
    def available_fields(self) -> tuple[str, ...]:
        return self.config.allowed_fields

    @property
    def allowed_filters(self) -> tuple[str, ...]:
        return self.config.allowed_filters

    @property
    def redactions(self) -> tuple[str, ...]:
        return self.config.redactions

    def read(self, draft: TraceAuditDraftV1) -> tuple[dict[str, Any], ...]:
        if not self.config.path.is_file():
            raise ResearchError(
                "trace_source_unavailable",
                "registered JSONL trace source is unavailable",
                category="evidence",
                retryable=True,
            )
        records: list[dict[str, Any]] = []
        with self.config.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if len(line) > 2_000_000:
                    raise ResearchError(
                        "trace_record_too_large",
                        f"trace record {line_number} exceeds the size limit",
                        category="evidence",
                    )
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResearchError(
                        "trace_source_invalid",
                        f"registered trace source contains invalid JSON at line {line_number}",
                        category="evidence",
                    ) from exc
                if not isinstance(raw, dict):
                    raise ResearchError(
                        "trace_source_invalid",
                        f"registered trace source line {line_number} is not an object",
                        category="evidence",
                    )
                normalized = _normalize_trace(raw, draft.fields)
                if _matches(normalized, draft):
                    records.append(normalized)
                if len(records) >= draft.max_traces:
                    break
        return tuple(records)


class WeaveTraceSource:
    def __init__(
        self,
        config: _SourceConfig,
        *,
        env: Mapping[str, str] | None = None,
        fetcher: Callable[[dict[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        if not config.project or "/" not in config.project:
            raise ValueError("weave trace source requires an entity/project")
        self.config = config
        self.env = dict(os.environ if env is None else env)
        self.fetcher = fetcher

    @property
    def source(self) -> TraceSourceRefV1:
        return TraceSourceRefV1(
            RESEARCH_SCHEMA_VERSION,
            self.config.id,
            "weave",
            self.config.safe_digest,
        )

    @property
    def available_fields(self) -> tuple[str, ...]:
        return self.config.allowed_fields

    @property
    def allowed_filters(self) -> tuple[str, ...]:
        return self.config.allowed_filters

    @property
    def redactions(self) -> tuple[str, ...]:
        return self.config.redactions

    def read(self, draft: TraceAuditDraftV1) -> tuple[dict[str, Any], ...]:
        payload = {
            "project_id": self.config.project,
            "filter": {"trace_roots_only": True},
            "limit": draft.max_traces,
        }
        if draft.started_after:
            payload["filter"]["started_at_from"] = draft.started_after
        if draft.started_before:
            payload["filter"]["started_at_to"] = draft.started_before
        if self.fetcher is not None:
            raw_records = self.fetcher(payload)
        else:
            raw_records = self._fetch(payload)
        records = []
        for raw in raw_records:
            normalized = _normalize_trace(dict(raw), draft.fields)
            if _matches(normalized, draft):
                records.append(normalized)
            if len(records) >= draft.max_traces:
                break
        return tuple(records)

    def _fetch(self, payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        api_key = self.env.get("WANDB_API_KEY", "").strip()
        if not api_key:
            raise ResearchError(
                "trace_credentials_unavailable",
                "the registered Weave source has no runtime credentials",
                category="evidence",
                retryable=True,
            )
        base_url = resolved_weave_trace_server_url(self.env)
        try:
            with httpx.Client(
                timeout=30,
                headers={"Authorization": f"Bearer {api_key}"},
            ) as client:
                response = client.post(f"{base_url}/calls/stream_query", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ResearchError(
                "trace_source_unavailable",
                "the registered Weave source could not be read",
                category="evidence",
                retryable=True,
            ) from exc
        return tuple(_decode_stream(response.text))


class TraceSourceRegistry:
    def __init__(self, adapters: Iterable[TraceSourceAdapter] = ()) -> None:
        values = tuple(adapters)
        self._adapters = {adapter.source.source_id: adapter for adapter in values}
        if len(self._adapters) != len(values):
            raise ValueError("trace source ids must be unique")

    @classmethod
    def from_file(
        cls,
        path: Path | None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> TraceSourceRegistry:
        if path is None:
            return cls()
        root = path.resolve().parent
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_mapping(raw, root=root, env=env)

    @classmethod
    def from_mapping(
        cls,
        raw: Any,
        *,
        root: Path,
        env: Mapping[str, str] | None = None,
        weave_fetchers: Mapping[
            str, Callable[[dict[str, Any]], Sequence[Mapping[str, Any]]]
        ]
        | None = None,
    ) -> TraceSourceRegistry:
        if not isinstance(raw, dict) or set(raw) != {"version", "sources"}:
            raise ValueError("trace source config requires only version and sources")
        if raw["version"] != 1 or not isinstance(raw["sources"], list):
            raise ValueError("trace source config version must be 1")
        adapters: list[TraceSourceAdapter] = []
        for value in raw["sources"]:
            config = _source_config(value, root)
            if config.adapter == "jsonl":
                adapters.append(JsonlTraceSource(config))
            else:
                adapters.append(
                    WeaveTraceSource(
                        config,
                        env=env,
                        fetcher=(weave_fetchers or {}).get(config.id),
                    )
                )
        return cls(adapters)

    def get(self, source_id: str) -> TraceSourceAdapter:
        source_id = validate_id(source_id, kind="trace source id")
        try:
            return self._adapters[source_id]
        except KeyError as exc:
            raise ResearchError(
                "trace_source_not_registered",
                f"trace source is not registered: {source_id}",
                category="policy",
            ) from exc

    def catalog(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "source": adapter.source.to_dict(),
                "available_fields": list(adapter.available_fields),
                "allowed_filters": list(adapter.allowed_filters),
                "redactions": list(adapter.redactions),
            }
            for adapter in sorted(
                self._adapters.values(), key=lambda item: item.source.source_id
            )
        )


class TraceAuditStore:
    def __init__(self, database: Path) -> None:
        self.path = database.resolve()
        self._initialize()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trace_audits (
                    audit_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    preview_digest TEXT NOT NULL UNIQUE,
                    audit_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trace_audit_operations (
                    operation_id TEXT PRIMARY KEY,
                    input_digest TEXT NOT NULL,
                    audit_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def put(
        self, audit: TraceAuditV1, *, operation_id: str, input_digest: str
    ) -> TraceAuditV1:
        operation_id = validate_id(operation_id, kind="trace audit operation id")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute(
                "SELECT input_digest, audit_id FROM trace_audit_operations "
                "WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if prior:
                if prior[0] != input_digest:
                    raise ResearchError(
                        "operation_conflict",
                        "trace audit operation id was reused with different input",
                        category="conflict",
                    )
                conn.commit()
                return self.get(str(prior[1]))
            existing = conn.execute(
                "SELECT audit_json FROM trace_audits WHERE preview_digest=?",
                (audit.preview_digest,),
            ).fetchone()
            if existing:
                return trace_audit_from_dict(json.loads(existing[0]))
            conn.execute(
                "INSERT INTO trace_audits VALUES (?, ?, ?, ?, ?)",
                (
                    audit.id,
                    audit.study_id,
                    audit.preview_digest,
                    json.dumps(audit.to_dict(), sort_keys=True),
                    audit.created_at,
                ),
            )
            conn.execute(
                "INSERT INTO trace_audit_operations VALUES (?, ?, ?, ?)",
                (operation_id, input_digest, audit.id, now()),
            )
            conn.commit()
        return audit

    def get(self, audit_id: str) -> TraceAuditV1:
        audit_id = validate_id(audit_id, kind="trace audit id")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT audit_json FROM trace_audits WHERE audit_id=?", (audit_id,)
            ).fetchone()
        if row is None:
            raise ResearchError("trace_audit_not_found", "trace audit was not found")
        return trace_audit_from_dict(json.loads(row[0]))

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


class TraceAuditService:
    def __init__(
        self,
        studies: StudyStore,
        registry: TraceSourceRegistry,
        approvals: ApprovalLedger,
    ) -> None:
        self.studies = studies
        self.registry = registry
        self.approvals = approvals
        self.store = TraceAuditStore(studies.path)

    def preview(self, study_id: str, draft: TraceAuditDraftV1) -> TraceAuditPreviewV1:
        study = self.studies.get_study(study_id)
        draft = trace_audit_draft_from_dict(draft.to_dict())
        if draft.study_id != study.id:
            raise ResearchError(
                "study_mismatch", "trace audit belongs to another Study"
            )
        source = self.registry.get(draft.source_id)
        blockers = []
        unavailable = sorted(set(draft.fields) - set(source.available_fields))
        if unavailable:
            blockers.append("source does not permit fields: " + ", ".join(unavailable))
        disallowed_filters = sorted(set(draft.filters) - set(source.allowed_filters))
        if disallowed_filters:
            blockers.append(
                "source does not permit filters: " + ", ".join(disallowed_filters)
            )
        audit_id = f"audit-{draft.draft_digest[:20]}"
        unsigned = TraceAuditPreviewV1(
            schema_version=RESEARCH_SCHEMA_VERSION,
            study_id=study.id,
            audit_id=audit_id,
            source=source.source,
            draft=draft.to_dict(),
            maximum_traces=draft.max_traces,
            available_fields=source.available_fields,
            estimated_calls={},
            estimated_cost_usd=0.0,
            approval_required=False,
            redactions=source.redactions,
            eligible=not blockers,
            blockers=tuple(blockers),
        )
        return sign_trace_audit_preview(unsigned)

    def run(
        self,
        preview: TraceAuditPreviewV1,
        *,
        operation_id: str,
        approval_digest: str | None = None,
    ) -> TraceAuditV1:
        preview = trace_audit_preview_from_dict(preview.to_dict())
        if not preview.eligible:
            raise ResearchError(
                "preview_ineligible",
                "an ineligible trace audit preview cannot run",
                category="policy",
            )
        self.studies.get_study(preview.study_id)
        draft = trace_audit_draft_from_dict(preview.draft)
        source = self.registry.get(draft.source_id)
        if source.source != preview.source:
            raise ResearchError(
                "trace_source_drift",
                "registered trace source changed after preview",
                category="policy",
            )
        if preview.approval_required:
            if not approval_digest:
                raise ResearchError(
                    "approval_required",
                    "paid trace analysis requires operator approval",
                    category="policy",
                )
            self.approvals.claim(
                approval_digest=approval_digest,
                subject_kind="trace_audit",
                preview_digest=preview.preview_digest,
                subject_id=preview.audit_id,
                estimated_cost_usd=preview.estimated_cost_usd,
            )
        records = source.read(draft)
        snapshot_digest = stable_digest(records)
        clusters = _clusters(records)
        fields = {field: 0 for field in draft.fields}
        for record in records:
            for field in fields:
                if record.get(field) not in (None, [], {}, ""):
                    fields[field] += 1
        warnings = () if records else ("No traces matched the locked cohort.",)
        audit = sign_trace_audit(
            TraceAuditV1(
                schema_version=RESEARCH_SCHEMA_VERSION,
                id=preview.audit_id,
                study_id=preview.study_id,
                preview_digest=preview.preview_digest,
                source=preview.source,
                source_snapshot_digest=snapshot_digest,
                cohort_count=len(records),
                coverage={
                    "requested": draft.max_traces,
                    "returned": len(records),
                    "fields": fields,
                },
                clusters=clusters,
                trace_refs=tuple(_trace_ref(item) for item in records),
                suggested_tasks=_suggested_tasks(clusters),
                redactions=source.redactions,
                warnings=warnings,
                created_at=now(),
            )
        )
        input_digest = stable_digest(
            {
                "action": "run_trace_audit",
                "preview_digest": preview.preview_digest,
                "approval_digest": approval_digest,
            }
        )
        return self.store.put(
            audit, operation_id=operation_id, input_digest=input_digest
        )


def _source_config(raw: Any, root: Path) -> _SourceConfig:
    if not isinstance(raw, dict):
        raise ValueError("trace source entry must be an object")
    unknown = sorted(set(raw) - _SOURCE_FIELDS)
    if unknown:
        raise ValueError("unknown trace source fields: " + ", ".join(unknown))
    source_id = validate_id(str(raw.get("id") or ""), kind="trace source id")
    adapter = str(raw.get("adapter") or "")
    if adapter not in {"jsonl", "weave"}:
        raise ValueError("trace source adapter must be jsonl or weave")
    allowed_fields = _string_tuple(raw.get("allowed_fields") or _DEFAULT_FIELDS)
    allowed_filters = _string_tuple(
        raw.get("allowed_filters") or ("run_id", "status", "harness", "model")
    )
    redactions = _string_tuple(raw.get("redactions") or _DEFAULT_REDACTIONS)
    configured_path = raw.get("path")
    path = None
    if configured_path is not None:
        if adapter != "jsonl" or not isinstance(configured_path, str):
            raise ValueError("path is permitted only for JSONL trace sources")
        path = (root / configured_path).resolve()
    project = raw.get("project")
    if project is not None and (adapter != "weave" or not isinstance(project, str)):
        raise ValueError("project is permitted only for Weave trace sources")
    return _SourceConfig(
        id=source_id,
        adapter=adapter,
        allowed_fields=allowed_fields,
        allowed_filters=allowed_filters,
        redactions=redactions,
        path=path,
        project=project,
    )


def _normalize_trace(raw: dict[str, Any], requested: Sequence[str]) -> dict[str, Any]:
    attributes = (
        raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
    )
    summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
    usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
    exception = raw.get("exception") if isinstance(raw.get("exception"), dict) else {}
    operation = raw.get("op_name") or raw.get("operation") or raw.get("name")
    trace_id = raw.get("trace_id") or raw.get("id") or raw.get("call_id")
    if not trace_id:
        trace_id = stable_digest(raw)[:32]
    error_type = (
        raw.get("error_type")
        or exception.get("type")
        or summary.get("error_type")
        or attributes.get("error.type")
    )
    status = raw.get("status") or summary.get("status")
    if not status:
        status = "error" if error_type or raw.get("exception") else "unknown"
    tools = raw.get("tool_names") or summary.get("tool_names") or []
    if not isinstance(tools, list):
        tools = []
    result: dict[str, Any] = {
        "trace_id": _clean_text(trace_id, 300),
        "conversation_id": _clean_optional(
            raw.get("conversation_id") or attributes.get("gen_ai.conversation.id"), 300
        ),
        "run_id": _clean_optional(
            raw.get("run_id") or attributes.get("fugue.run_id"), 300
        ),
        "harness": _clean_optional(
            raw.get("harness") or attributes.get("fugue.harness"), 100
        ),
        "model": _clean_optional(
            raw.get("model") or attributes.get("gen_ai.request.model"), 300
        ),
        "started_at": _clean_optional(raw.get("started_at"), 64),
    }
    if "status" in requested:
        result["status"] = _clean_text(status, 100)
    if "operation" in requested:
        result["operation"] = _clean_optional(operation, 300)
    if "errors" in requested:
        result["errors"] = {
            "type": _clean_optional(error_type, 300),
            "message": _clean_optional(
                raw.get("error_message") or exception.get("message"), 1000
            ),
        }
    if "tools" in requested:
        result["tools"] = sorted({_clean_text(item, 300) for item in tools})[:100]
    if "latency" in requested:
        result["latency"] = _finite_or_none(
            raw.get("latency_ms") or summary.get("latency_ms")
        )
    if "tokens" in requested:
        result["tokens"] = {
            "input": _int_or_none(usage.get("input_tokens") or raw.get("input_tokens")),
            "output": _int_or_none(
                usage.get("output_tokens") or raw.get("output_tokens")
            ),
        }
    if "cost" in requested:
        result["cost"] = _finite_or_none(summary.get("cost_usd") or raw.get("cost_usd"))
    if "final_output" in requested:
        result["final_output"] = _clean_optional(
            raw.get("output") or raw.get("final_output"), 2000
        )
    if "artifacts" in requested:
        artifacts = (
            raw.get("artifacts") if isinstance(raw.get("artifacts"), list) else []
        )
        result["artifacts"] = [
            _clean_text(item, 500) for item in artifacts if isinstance(item, str)
        ][:100]
    if "conversation" in requested:
        result["conversation"] = _conversation_summary(raw.get("conversation"))
    return _remove_empty(result)


def _conversation_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    roles = Counter()
    tool_messages = 0
    for item in value[:1000]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "unknown")
        roles[role] += 1
        if role == "tool" or item.get("tool_call_id"):
            tool_messages += 1
    return {
        "message_count": sum(roles.values()),
        "roles": dict(sorted(roles.items())),
        "tool_message_count": tool_messages,
    }


def _matches(record: Mapping[str, Any], draft: TraceAuditDraftV1) -> bool:
    for key, expected in draft.filters.items():
        actual = record.get(key)
        values = expected if isinstance(expected, list) else [expected]
        if actual not in values:
            return False
    started = str(record.get("started_at") or "")
    if draft.started_after and (not started or started < draft.started_after):
        return False
    if draft.started_before and (not started or started >= draft.started_before):
        return False
    return True


def _clusters(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        errors = record.get("errors") if isinstance(record.get("errors"), dict) else {}
        key = str(errors.get("type") or record.get("status") or "unknown")
        if key.lower() in {"ok", "passed", "success", "succeeded"}:
            continue
        grouped.setdefault(key, []).append(record)
    return tuple(
        {
            "id": f"cluster-{stable_digest(key)[:12]}",
            "label": _clean_text(key, 300),
            "count": len(values),
            "representative_trace_ids": [str(item["trace_id"]) for item in values[:3]],
            "harnesses": sorted(
                {str(item["harness"]) for item in values if item.get("harness")}
            ),
            "operations": sorted(
                {str(item["operation"]) for item in values if item.get("operation")}
            ),
        }
        for key, values in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0])
        )
    )


def _trace_ref(record: Mapping[str, Any]) -> dict[str, Any]:
    return _remove_empty(
        {
            "trace_id": record.get("trace_id"),
            "conversation_id": record.get("conversation_id"),
            "run_id": record.get("run_id"),
            "status": record.get("status"),
            "harness": record.get("harness"),
            "model": record.get("model"),
            "summary_digest": stable_digest(record),
        }
    )


def _suggested_tasks(
    clusters: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "id": f"task-{str(cluster['id']).removeprefix('cluster-')}",
            "purpose": f"Reproduce the observed {cluster['label']} failure pattern.",
            "source_cluster_id": cluster["id"],
            "representative_trace_ids": cluster["representative_trace_ids"],
            "status": "candidate",
            "warning": "Validate the task and lock a separate holdout before confirmation.",
        }
        for cluster in clusters
    )


def _decode_stream(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        values = []
        for line in stripped.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResearchError(
                    "trace_source_invalid",
                    "Weave returned invalid trace data",
                    category="evidence",
                ) from exc
            if isinstance(item, dict):
                values.append(item)
        return values
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("calls"), list):
        return [item for item in value["calls"] if isinstance(item, dict)]
    return [value] if isinstance(value, dict) else []


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("trace source string collections must be lists")
    values = tuple(str(item).strip() for item in value)
    if any(not item for item in values) or len(values) != len(set(values)):
        raise ValueError("trace source string collections must be unique and non-empty")
    return values


def _clean_text(value: Any, limit: int) -> str:
    text = " ".join(str(value).replace("\x00", " ").split())[:limit]
    return _SECRET.sub("[REDACTED]", text)


def _clean_optional(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = _clean_text(value, limit)
    return text or None


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 and number < float("inf") else None


def _int_or_none(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _remove_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item for key, item in value.items() if item not in (None, "", [], {}, ())
    }
