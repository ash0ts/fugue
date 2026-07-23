from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from filelock import FileLock

from fugue.bench.candidates import stable_digest
from fugue.research.contracts import (
    RESEARCH_SCHEMA_VERSION,
    AttributionV1,
    JsonValue,
    ResearchError,
    attribution_from_dict,
)

ResearchLogClassification = Literal[
    "lifecycle",
    "observation",
    "decision",
    "evidence",
    "result",
    "budget",
    "limitation",
]
ResearchLogState = Literal[
    "proposed",
    "awaiting_approval",
    "preparing",
    "running",
    "evaluating",
    "completed",
    "failed",
    "cancelled",
    "paused",
]
ResearchRelationshipKind = Literal[
    "derived_from",
    "compares_to",
    "uses_baseline",
    "uses_evidence",
    "supersedes",
]

_CLASSIFICATIONS = {
    "lifecycle",
    "observation",
    "decision",
    "evidence",
    "result",
    "budget",
    "limitation",
}
_STATES = {
    "proposed",
    "awaiting_approval",
    "preparing",
    "running",
    "evaluating",
    "completed",
    "failed",
    "cancelled",
    "paused",
}
_RELATIONSHIPS = {
    "derived_from",
    "compares_to",
    "uses_baseline",
    "uses_evidence",
    "supersedes",
}
RESEARCH_LOG_MAX_BYTES = 65_536
_PRIVATE_KEYS = {
    "credential",
    "credentials",
    "expected",
    "expected_answer",
    "expected_answers",
    "expected_output",
    "expected_outputs",
    "expected_path",
    "expected_paths",
    "expected_reference",
    "expected_references",
    "expected_value",
    "expected_values",
    "gold",
    "gold_path",
    "gold_paths",
    "hidden_reasoning",
    "private_criteria",
    "prompt",
    "prompt_body",
    "prompt_content",
    "prompt_messages",
    "prompt_text",
    "reasoning_body",
    "reasoning_content",
    "reasoning_text",
    "secret",
    "secrets",
    "trace_body",
}
_PRIVATE_KEY_PREFIXES = ("credential_", "gold_", "private_", "secret_")
_PRIVATE_KEY_SUFFIXES = ("_credential", "_secret")
_PUBLIC_SELECTOR_KEYS = {
    "analysis_id",
    "artifact_name",
    "artifact_version",
    "call_id",
    "commit",
    "dataset_row_id",
    "entity",
    "evaluation_id",
    "operation_id",
    "project",
    "row_id",
    "run_id",
    "trace_id",
}


@dataclass(frozen=True)
class ResearchRelationshipV1:
    kind: ResearchRelationshipKind
    target: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchEvidenceRefV1:
    system: str
    kind: str
    ref: str
    uri: str | None = None
    digest: str | None = None
    version: str | None = None
    selector: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: item
            for key, item in asdict(self).items()
            if item not in (None, "", (), [], {})
        }


@dataclass(frozen=True)
class ResearchLogEventV1:
    schema_version: int
    producer_event_id: str
    sequence: int
    timestamp: str
    source: str
    actor: AttributionV1
    research_id: str
    study_id: str | None
    classification: ResearchLogClassification
    state: ResearchLogState
    message: str
    progress: dict[str, JsonValue] = field(default_factory=dict)
    reserved_cost_usd: float | None = None
    observed_cost_usd: float | None = None
    relationships: tuple[ResearchRelationshipV1, ...] = ()
    evidence: tuple[ResearchEvidenceRefV1, ...] = ()
    summary: dict[str, JsonValue] = field(default_factory=dict)
    event_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {
            key: item
            for key, item in value.items()
            if item not in (None, "", (), [], {})
        }


def research_log_event_from_dict(
    raw: Mapping[str, Any], *, require_digest: bool = True
) -> ResearchLogEventV1:
    fields = {item.name for item in ResearchLogEventV1.__dataclass_fields__.values()}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(
            "research log event has unknown fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    schema_version = _positive_int(raw.get("schema_version"), "schema_version")
    if schema_version != RESEARCH_SCHEMA_VERSION:
        raise ValueError("unsupported research log event schema")
    classification = str(raw.get("classification") or "")
    if classification not in _CLASSIFICATIONS:
        raise ValueError("unknown research log classification")
    state = str(raw.get("state") or "")
    if state not in _STATES:
        raise ValueError("unknown research log state")
    relationships = tuple(
        _relationship(item)
        for item in _sequence(raw.get("relationships"), "relationships")
    )
    evidence = tuple(
        _evidence_ref(item) for item in _sequence(raw.get("evidence"), "evidence")
    )
    event = ResearchLogEventV1(
        schema_version=schema_version,
        producer_event_id=_text(
            raw.get("producer_event_id"), "producer_event_id", 1000
        ),
        sequence=_positive_int(raw.get("sequence"), "sequence"),
        timestamp=_text(raw.get("timestamp"), "timestamp", 100),
        source=_text(raw.get("source"), "source", 300),
        actor=attribution_from_dict(_mapping(raw.get("actor"), "actor")),
        research_id=_text(raw.get("research_id"), "research_id", 1000),
        study_id=_optional_text(raw.get("study_id"), "study_id", 1000),
        classification=classification,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        message=_text(raw.get("message"), "message", 4000),
        progress=_json_mapping(raw.get("progress"), "progress"),
        reserved_cost_usd=_cost(raw.get("reserved_cost_usd"), "reserved cost"),
        observed_cost_usd=_cost(raw.get("observed_cost_usd"), "observed cost"),
        relationships=relationships,
        evidence=evidence,
        summary=_json_mapping(raw.get("summary"), "summary"),
        event_digest=str(raw.get("event_digest") or ""),
    )
    unsigned = event.to_dict()
    unsigned.pop("event_digest", None)
    if (
        len(json.dumps(unsigned, separators=(",", ":")).encode())
        > RESEARCH_LOG_MAX_BYTES
    ):
        raise ValueError("research log event exceeds the publication size limit")
    digest = stable_digest(unsigned)
    if event.event_digest and event.event_digest != digest:
        raise ValueError("event_digest does not match research log event")
    if require_digest and event.event_digest != digest:
        raise ValueError("event_digest is required")
    return replace(event, event_digest=digest)


def sign_research_log_event(event: ResearchLogEventV1) -> ResearchLogEventV1:
    return research_log_event_from_dict(event.to_dict(), require_digest=False)


def event_state(value: str) -> ResearchLogState:
    if value in {"queued", "planning", "preparing", "admitting", "launching"}:
        return "preparing"
    if value in {"running", "cancelling"}:
        return "running"
    if value in {"scoring", "analyzing"}:
        return "evaluating"
    if value == "completed":
        return "completed"
    if value == "cancelled":
        return "cancelled"
    if value in {"blocked", "interrupted", "failed"}:
        return "failed"
    return "proposed"


def public_evidence_selector(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Keep only bounded public identities; evidence bodies remain at the source."""

    selector: dict[str, JsonValue] = {}
    for key, item in value.items():
        if key not in _PUBLIC_SELECTOR_KEYS:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            selector[key] = item
        elif isinstance(item, list) and len(item) <= 50 and all(
            isinstance(member, (str, int, float, bool)) or member is None
            for member in item
        ):
            selector[key] = item
    return selector


class ResearchRecordSink(Protocol):
    @property
    def sink_id(self) -> str: ...

    def publish(self, event: ResearchLogEventV1) -> None: ...


class JsonlResearchRecordSink:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    @property
    def sink_id(self) -> str:
        return f"jsonl:{stable_digest(str(self.path))[:20]}"

    def publish(self, event: ResearchLogEventV1) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        with FileLock(lock_path):
            # The append-only log is authoritative. Rebuilding the compact index
            # makes a crash after fsync but before the index rename recoverable
            # without appending the producer event a second time.
            records = self._records_from_log()
            prior = records.get(event.producer_event_id)
            if prior:
                if prior != event.event_digest:
                    raise ResearchError(
                        "publication_conflict",
                        "producer event id was replayed with different content",
                        category="conflict",
                    )
                self._write_index(records)
                return
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            records[event.producer_event_id] = event.event_digest
            self._write_index(records)

    def _records_from_log(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        size = self.path.stat().st_size
        index = self.path.with_suffix(f"{self.path.suffix}.index.json")
        if index.is_file():
            try:
                cached = json.loads(index.read_text(encoding="utf-8"))
                offset = int(cached["offset"])
                raw_records = cached["records"]
                if (
                    cached.get("version") == 1
                    and 0 <= offset <= size
                    and isinstance(raw_records, dict)
                    and all(
                        isinstance(key, str) and isinstance(value, str)
                        for key, value in raw_records.items()
                    )
                ):
                    records = dict(raw_records)
                    if offset == size:
                        return records
                    with self.path.open("rb") as stream:
                        stream.seek(offset)
                        tail = stream.read().decode("utf-8")
                    return self._records_from_lines(
                        tail.splitlines(), records=records, first_line=0
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                # The index is a disposable cache. Rebuild it from the durable
                # append-only log rather than turning cache damage into data loss.
                pass
        return self._records_from_lines(
            self.path.read_text(encoding="utf-8").splitlines(),
            records={},
            first_line=1,
        )

    @staticmethod
    def _records_from_lines(
        lines: Iterable[str], *, records: dict[str, str], first_line: int
    ) -> dict[str, str]:
        for line_number, line in enumerate(lines, start=first_line):
            if not line.strip():
                continue
            try:
                event = research_log_event_from_dict(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"research record JSONL is invalid at line {line_number}"
                ) from exc
            prior = records.get(event.producer_event_id)
            if prior and prior != event.event_digest:
                raise ResearchError(
                    "publication_conflict",
                    "producer event id appears with conflicting content",
                    category="conflict",
                )
            records[event.producer_event_id] = event.event_digest
        return records

    def _write_index(self, records: Mapping[str, str]) -> None:
        index = self.path.with_suffix(f"{self.path.suffix}.index.json")
        temporary = index.with_suffix(f"{index.suffix}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": 1,
                    "offset": self.path.stat().st_size,
                    "records": dict(records),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(index)


class HttpResearchRecordSink:
    def __init__(self, url: str, token: str, *, timeout: float = 10.0) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("research record HTTP sink must use http or https")
        if not token:
            raise ValueError("research record HTTP sink requires an ingest token")

    @property
    def sink_id(self) -> str:
        return f"http:{stable_digest(self.url)[:20]}"

    def publish(self, event: ResearchLogEventV1) -> None:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(event.to_dict(), separators=(",", ":")).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Idempotency-Key": event.producer_event_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status not in {200, 201, 202, 204}:
                    raise RuntimeError(
                        f"research record sink returned HTTP {response.status}"
                    )
        except urllib.error.HTTPError as exc:
            body = exc.read(2048).decode("utf-8", "replace")
            raise RuntimeError(
                f"research record sink returned HTTP {exc.code}: {body}"
            ) from exc


class ResearchRecordPublisher:
    def __init__(self, store: Any, sinks: Iterable[ResearchRecordSink]) -> None:
        self.store = store
        self.sinks = tuple(sinks)

    @classmethod
    def from_environment(
        cls, store: Any, *, env: Mapping[str, str] | None = None
    ) -> ResearchRecordPublisher:
        values = dict(os.environ if env is None else env)
        sinks: list[ResearchRecordSink] = []
        jsonl = values.get("FUGUE_RESEARCH_RECORD_JSONL", "").strip()
        if jsonl:
            sinks.append(JsonlResearchRecordSink(Path(jsonl)))
        url = values.get("FUGUE_RESEARCH_RECORD_HTTP_URL", "").strip()
        if url:
            token = _secret(values, "FUGUE_RESEARCH_RECORD_TOKEN")
            sinks.append(HttpResearchRecordSink(url, token))
        return cls(store, sinks)

    def flush(self, *, limit: int = 100) -> dict[str, int]:
        delivered = 0
        failed = 0
        for sink in self.sinks:
            for event in self.store.pending_research_log_events(
                sink.sink_id, limit=limit
            ):
                try:
                    sink.publish(event)
                except Exception as exc:  # publication must not affect a Run
                    self.store.mark_research_log_failed(
                        sink.sink_id, event.sequence, str(exc)
                    )
                    failed += 1
                    continue
                self.store.mark_research_log_delivered(sink.sink_id, event.sequence)
                delivered += 1
        return {"delivered": delivered, "failed": failed}


def _secret(env: Mapping[str, str], name: str) -> str:
    path = env.get(f"{name}_FILE", "").strip()
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return env.get(name, "").strip()


def _relationship(raw: Any) -> ResearchRelationshipV1:
    value = _mapping(raw, "relationship")
    unknown = set(value) - {"kind", "target"}
    if unknown:
        raise ValueError("research relationship has unknown fields")
    kind = str(value.get("kind") or "")
    if kind not in _RELATIONSHIPS:
        raise ValueError("unknown research relationship kind")
    return ResearchRelationshipV1(
        kind=kind,  # type: ignore[arg-type]
        target=_text(value.get("target"), "relationship target", 2000),
    )


def _evidence_ref(raw: Any) -> ResearchEvidenceRefV1:
    value = _mapping(raw, "evidence reference")
    unknown = set(value) - {
        "system",
        "kind",
        "ref",
        "uri",
        "digest",
        "version",
        "selector",
    }
    if unknown:
        raise ValueError("research evidence reference has unknown fields")
    digest = _optional_text(value.get("digest"), "evidence digest", 64)
    if digest and (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("evidence digest must be sha256")
    uri = _optional_text(value.get("uri"), "evidence uri", 4000)
    if uri and not uri.startswith(("http://", "https://")):
        raise ValueError("evidence uri must use http or https")
    return ResearchEvidenceRefV1(
        system=_text(value.get("system"), "evidence system", 300),
        kind=_text(value.get("kind"), "evidence kind", 300),
        ref=_text(value.get("ref"), "evidence ref", 2000),
        uri=uri,
        digest=digest,
        version=_optional_text(value.get("version"), "evidence version", 300),
        selector=_json_mapping(value.get("selector"), "evidence selector"),
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    return tuple(value)


def _text(value: Any, label: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum:
        raise ValueError(f"{label} must contain 1 to {maximum} characters")
    return text


def _optional_text(value: Any, label: str, maximum: int) -> str | None:
    if value in (None, ""):
        return None
    return _text(value, label, maximum)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{label} must be a positive integer")
    return integer


def _cost(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be non-negative")
    number = float(value)
    if number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _json_mapping(value: Any, label: str) -> dict[str, JsonValue]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    serialized = json.loads(json.dumps(value))
    if not isinstance(serialized, dict):
        raise ValueError(f"{label} must be an object")
    _reject_private_keys(serialized, label)
    return serialized


def _reject_private_keys(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if (
                normalized in _PRIVATE_KEYS
                or normalized.startswith(_PRIVATE_KEY_PREFIXES)
                or normalized.endswith(_PRIVATE_KEY_SUFFIXES)
            ):
                raise ValueError(f"{label} contains a private field")
            _reject_private_keys(item, label)
    elif isinstance(value, list):
        for item in value:
            _reject_private_keys(item, label)
