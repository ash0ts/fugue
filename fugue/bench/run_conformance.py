from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx

from fugue.bench.files import atomic_write_json
from fugue.model_plane import trace_api_key
from fugue.redaction import redact_value, secrets_from_env
from fugue.weave_support import WEAVE_AGENTS_BASE_URL

if TYPE_CHECKING:
    from fugue.bench.execution import PlannedCell
    from fugue.bench.job_config import RenderedJob


ReceiptStatus = Literal["passed", "failed", "unavailable", "not_applicable"]

HARBOR_CONFORMANCE_RECEIPT = "harbor-conformance.json"
HOSTED_EVIDENCE_PRIVACY_RECEIPT = "hosted-evidence-privacy.json"
PRIVACY_CONTRACT_VERSION = 2
_MAX_SCAN_FILES = 20_000
_MAX_SCAN_FILE_BYTES = 16 * 1024 * 1024
_MAX_SCAN_TOTAL_BYTES = 128 * 1024 * 1024
_MAX_NEW_CONTAINER_INSPECTIONS = 512
_MAX_HOSTED_PAYLOADS = 50_000
_MAX_HOSTED_PAYLOAD_BYTES = 256 * 1024 * 1024
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_PRIVATE_INPUT_KEYS = frozenset(
    {
        "answer_key",
        "expected",
        "gold",
        "private",
        "private_expected_values",
        "private_labels",
        "reference_answer",
    }
)


@dataclass(frozen=True)
class HarborRunConformance:
    path: Path
    sha256: str
    status: ReceiptStatus
    enforced: bool

    def manifest_reference(self, repo_root: Path) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "path": self.path.relative_to(repo_root).as_posix(),
            "sha256": self.sha256,
            "status": self.status,
            "enforced": self.enforced,
        }


@dataclass(frozen=True)
class HostedEvidencePrivacy:
    path: Path
    sha256: str
    status: ReceiptStatus


@dataclass(frozen=True)
class _HostedEvidenceSnapshot:
    status: ReceiptStatus
    payloads: tuple[tuple[str, Any], ...]
    required_call_count: int
    observed_required_call_count: int
    descendant_call_count: int
    agent_span_count: int
    required_agent_conversation_count: int
    observed_agent_conversation_count: int
    required_dataset_count: int
    observed_dataset_count: int
    query_error_count: int


def capture_local_docker_inventory(
    jobs: Sequence[RenderedJob],
) -> dict[str, Any]:
    """Capture a bounded, read-only pre-run Docker container inventory.

    The inventory is used only to distinguish containers that remained from
    this execution window. It is not evidence that unrelated host resources
    are part of the Fugue run.
    """

    _local_jobs, backend = _local_agent_jobs(jobs)
    if backend != "local_harbor_docker":
        return {
            "schema_version": 1,
            "status": "not_applicable",
            "backend": backend,
            "container_ids": [],
        }
    container_ids, error = _docker_ids(
        ["docker", "container", "ls", "--all", "--quiet"],
        resource="container",
    )
    if error is not None:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "backend": backend,
            "container_ids": [],
            "reason": error,
        }
    return {
        "schema_version": 1,
        "status": "passed",
        "backend": backend,
        "container_ids": container_ids,
    }


def capture_local_cell_conformance(
    *,
    repo_root: Path,
    cell: PlannedCell,
    job: RenderedJob,
    env: Mapping[str, str],
    host_scorer_names: Sequence[str] = (),
    pre_execution_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Prove the first Harbor cell is clean before releasing a queued cell."""

    root = repo_root.resolve()
    run_dir = root / ".fugue" / "runtime" / cell.run_id
    if Path(job.config_path).resolve() != Path(cell.config_path).resolve():
        raise ValueError("cell conformance job does not match its rendered config")
    secrets = _configured_secrets(env, [job])
    execution_identity = _execution_identity(cell.run_id, [job], repo_root=root)
    privacy_scan = _scan_run_artifacts(
        repo_root=root,
        run_id=cell.run_id,
        run_dir=run_dir,
        jobs=[job],
        secrets=secrets,
    )
    private_boundary = _private_label_boundary(
        run_dir=run_dir,
        jobs=[job],
        host_scorer_names=host_scorer_names,
    )
    cleanup = _docker_cleanup_audit(
        repo_root=root,
        run_dir=run_dir,
        run_id=cell.run_id,
        jobs=[job],
        pre_execution_inventory=pre_execution_inventory,
    )
    components = {
        "execution_identity": execution_identity,
        "local_artifact_privacy_scan": privacy_scan,
        "private_label_boundary": private_boundary,
        "docker_cleanup": cleanup,
    }
    statuses = {str(value.get("status") or "") for value in components.values()}
    status: ReceiptStatus
    if "failed" in statuses:
        status = "failed"
    elif "unavailable" in statuses:
        status = "unavailable"
    elif statuses <= {"passed", "not_applicable"}:
        status = "passed"
    else:
        status = "unavailable"
    payload = {
        "schema_version": PRIVACY_CONTRACT_VERSION,
        "run_id": cell.run_id,
        "cell_id": cell.id,
        "attempt_id": cell.attempt_id,
        "backend": "local_harbor_docker",
        "status": status,
        "enforced": True,
        **components,
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = _stable_digest(payload)
    return payload


def read_harbor_run_conformance_receipt(
    *,
    repo_root: Path,
    run_id: str,
) -> dict[str, Any]:
    path = repo_root.resolve() / ".fugue" / "runtime" / run_id / HARBOR_CONFORMANCE_RECEIPT
    _read_existing_receipt(path, run_id=run_id)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):  # pragma: no cover - validated above
        raise ValueError("Harbor conformance receipt must be a mapping")
    return value


def write_harbor_run_conformance_receipt(
    *,
    repo_root: Path,
    run_id: str,
    jobs: Sequence[RenderedJob],
    env: Mapping[str, str],
    host_scorer_names: Sequence[str] = (),
    pre_execution_inventory: Mapping[str, Any] | None = None,
    enforce: bool = True,
) -> HarborRunConformance:
    """Write one run-scoped, read-only audit for a local Harbor execution.

    Docker discovery reads only the list projection. Detailed container
    inspection is restricted to rows whose labels already reference this
    repository's exact run directory or exact run ID. This function never
    removes, stops, or otherwise mutates a container.
    """

    root = repo_root.resolve()
    run_dir = root / ".fugue" / "runtime" / run_id
    receipt_path = run_dir / HARBOR_CONFORMANCE_RECEIPT
    if receipt_path.is_file():
        return _read_existing_receipt(receipt_path, run_id=run_id)

    configured_secrets = _configured_secrets(env, jobs)
    local_jobs, backend = _local_agent_jobs(jobs)
    generated_at = datetime.now(UTC).isoformat()
    if backend != "local_harbor_docker":
        payload = {
            "schema_version": PRIVACY_CONTRACT_VERSION,
            "run_id": run_id,
            "backend": backend,
            "status": "not_applicable",
            "enforced": False,
            "generated_at": generated_at,
            "reason": "receipt applies only to local Docker/Harbor Agent runs",
            "receipt_sha256": "",
        }
        return _write_receipt(
            receipt_path,
            payload,
            secrets=configured_secrets,
        )

    if not enforce:
        payload = {
            "schema_version": PRIVACY_CONTRACT_VERSION,
            "run_id": run_id,
            "backend": backend,
            "status": "not_applicable",
            "enforced": False,
            "generated_at": generated_at,
            "reason": (
                "an injected cell runner did not establish local Docker execution"
            ),
            "receipt_sha256": "",
        }
        return _write_receipt(
            receipt_path,
            payload,
            secrets=configured_secrets,
        )

    identity = _execution_identity(run_id, local_jobs, repo_root=root)
    secret_scan = _scan_run_artifacts(
        repo_root=root,
        run_id=run_id,
        run_dir=run_dir,
        jobs=local_jobs,
        secrets=configured_secrets,
    )
    private_boundary = _private_label_boundary(
        run_dir=run_dir,
        jobs=local_jobs,
        host_scorer_names=host_scorer_names,
    )
    cleanup = _docker_cleanup_audit(
        repo_root=root,
        run_dir=run_dir,
        run_id=run_id,
        jobs=local_jobs,
        pre_execution_inventory=pre_execution_inventory,
    )
    statuses = (
        str(identity["status"]),
        str(secret_scan["status"]),
        str(private_boundary["status"]),
        str(cleanup["status"]),
    )
    status: ReceiptStatus
    if "failed" in statuses:
        status = "failed"
    elif "unavailable" in statuses:
        status = "unavailable"
    else:
        status = "passed"
    payload = {
        "schema_version": PRIVACY_CONTRACT_VERSION,
        "run_id": run_id,
        "backend": backend,
        "status": status,
        "enforced": True,
        "generated_at": generated_at,
        "execution_identity": identity,
        "local_artifact_privacy_scan": secret_scan,
        "private_label_boundary": private_boundary,
        "docker_cleanup": cleanup,
        "limitations": [
            "This is local Harbor evidence, not managed-sandbox isolation evidence.",
            (
                "The private-label check proves the rendered Agent-input boundary; "
                "it does not infer unknown private values from output text."
            ),
            (
                "The secret-value scan covers only the exact local run directory "
                "and rendered Harbor job artifact directories. It does not inspect "
                "hosted Weave objects, external services, or removed container "
                "filesystems."
            ),
            (
                "The cleanup audit covers only Docker Compose projects derived "
                "from this run's rendered Harbor job directories. It does not make "
                "a host-wide no-orphan claim."
            ),
            (
                "Runtime identity is verified against the prepared lock embedded "
                "in the exact rendered Harbor config. This receipt is not a "
                "hardware-backed container image execution attestation."
            ),
        ],
        "receipt_sha256": "",
    }
    return _write_receipt(
        receipt_path,
        payload,
        secrets=configured_secrets,
    )


def _configured_secrets(
    env: Mapping[str, str],
    jobs: Sequence[RenderedJob],
) -> tuple[str, ...]:
    values = set(secrets_from_env(env))
    for job in jobs:
        job_env = getattr(job, "env", None)
        if isinstance(job_env, Mapping):
            values.update(
                secrets_from_env(
                    {
                        str(key): str(value)
                        for key, value in job_env.items()
                    }
                )
            )
    return tuple(sorted(values, key=len, reverse=True))


def write_hosted_evidence_privacy_receipt(
    *,
    repo_root: Path,
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
    env: Mapping[str, str],
    evidence_project: str,
    private_labels_path: Path | None,
    publication_payloads: Mapping[str, Any],
    private_corpus_applicable: bool = True,
    private_corpus_source_kind: str = "private_labels",
    private_corpus_source_lock_sha256: str = "",
    private_corpus_reason: str = "",
    fetch_hosted: bool = True,
    fetcher: Callable[..., _HostedEvidenceSnapshot] | None = None,
) -> HostedEvidencePrivacy:
    """Scan exact published evidence without persisting evidence content.

    The scan runs after live Evaluation/Dataset publication. Raw hosted
    payloads, configured secret values, and private labels exist only in
    process memory. The receipt records hashes, counts, and statuses only.
    """

    root = repo_root.resolve()
    receipt_path = (
        root / ".fugue" / "runtime" / run_id / HOSTED_EVIDENCE_PRIVACY_RECEIPT
    )
    secrets = tuple(sorted(set(secrets_from_env(env)), key=len, reverse=True))
    source_kind = str(private_corpus_source_kind or "").strip()
    if not source_kind:
        raise ValueError("private corpus source kind is required")
    if (
        private_corpus_source_lock_sha256
        and not _HEX_SHA256.fullmatch(private_corpus_source_lock_sha256)
    ):
        raise ValueError("private corpus source lock must be a SHA-256 digest")
    if not private_corpus_applicable:
        labels_bytes = b""
        label_rows: list[Any] = []
        private_input_status: ReceiptStatus = "not_applicable"
    else:
        labels_path = (
            private_labels_path.resolve()
            if private_labels_path is not None
            else None
        )
        try:
            if labels_path is None:
                raise FileNotFoundError("private corpus path is unavailable")
            labels_path.relative_to(root)
            labels_bytes = labels_path.read_bytes()
            label_rows = _private_label_records(labels_bytes)
        except (
            FileNotFoundError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            labels_bytes = b""
            label_rows = []
            private_input_status = "unavailable"
        else:
            private_input_status = "passed" if label_rows else "unavailable"
    private_patterns = _private_label_patterns(labels_bytes, label_rows)
    selected_fetcher = fetcher or _fetch_hosted_evidence_snapshot
    try:
        if not fetch_hosted:
            raise RuntimeError("hosted evidence fetching was not requested")
        snapshot = selected_fetcher(
            rows=rows,
            project=evidence_project,
            env=env,
        )
    except Exception:
        snapshot = _HostedEvidenceSnapshot(
            status="unavailable",
            payloads=(),
            required_call_count=_required_call_count(rows),
            observed_required_call_count=0,
            descendant_call_count=0,
            agent_span_count=0,
            required_agent_conversation_count=len(
                _required_conversation_ids(rows)
            ),
            observed_agent_conversation_count=0,
            required_dataset_count=len(_required_dataset_refs(rows)),
            observed_dataset_count=0,
            query_error_count=1,
        )

    payloads = [
        *snapshot.payloads,
        *(
            (str(kind), payload)
            for kind, payload in publication_payloads.items()
            if payload is not None
        ),
    ]
    scan = _scan_hosted_payloads(
        payloads,
        secrets=secrets,
        private_patterns=private_patterns,
    )
    publication_kinds = [kind for kind, _payload in payloads]
    required_publication_payloads = {
        "result": publication_kinds.count("result"),
        "study": publication_kinds.count("study"),
    }
    missing_required_objects = sum(
        max(required - observed, 0)
        for required, observed in (
            (
                snapshot.required_call_count,
                snapshot.observed_required_call_count,
            ),
            (
                snapshot.required_dataset_count,
                snapshot.observed_dataset_count,
            ),
            (
                snapshot.required_agent_conversation_count,
                snapshot.observed_agent_conversation_count,
            ),
        )
    )
    inputs_available = bool(secrets) and private_input_status in {
        "passed",
        "not_applicable",
    }
    if scan["match_count"]:
        status: ReceiptStatus = "failed"
    elif (
        snapshot.status != "passed"
        or not inputs_available
        or missing_required_objects
        or required_publication_payloads["result"] != 1
        or scan["status"] != "passed"
    ):
        status = "unavailable"
    else:
        status = "passed"
    attempt_ids = sorted(
        str(row.get("attempt_id") or "")
        for row in rows
        if str(row.get("attempt_id") or "")
    )
    payload = {
        "schema_version": PRIVACY_CONTRACT_VERSION,
        "contract_digest": _stable_digest(
            {
                "name": "hosted-evidence-privacy",
                "version": PRIVACY_CONTRACT_VERSION,
                "required_calls": [
                    "evaluation_root",
                    "prediction_and_score",
                    "prediction",
                    "native_agent_root_and_descendants",
                ],
                "required_objects": ["dataset"],
                "publication_payloads": ["result", "study_when_present"],
            }
        ),
        "status": status,
        "run_scope_sha256": _stable_digest(
            {
                "run_id": run_id,
                "attempt_ids": attempt_ids,
                "project": evidence_project,
            }
        ),
        "project_sha256": hashlib.sha256(evidence_project.encode()).hexdigest(),
        "attempt_count": len(attempt_ids),
        "attempt_set_sha256": _stable_digest(attempt_ids),
        "configured_secret_count": len(secrets),
        "configured_secret_set_sha256": _stable_digest(
            sorted(hashlib.sha256(value.encode()).hexdigest() for value in secrets)
        ),
        "private_label_record_count": len(label_rows),
        "private_label_corpus_sha256": hashlib.sha256(labels_bytes).hexdigest(),
        "private_label_pattern_count": len(private_patterns),
        "private_corpus_applicable": private_corpus_applicable,
        "private_corpus_comparison_status": private_input_status,
        "private_corpus_source_kind": source_kind,
        "private_corpus_source_lock_sha256": private_corpus_source_lock_sha256,
        "private_corpus_reason": str(private_corpus_reason or "").strip(),
        "required_call_count": snapshot.required_call_count,
        "observed_required_call_count": snapshot.observed_required_call_count,
        "descendant_call_count": snapshot.descendant_call_count,
        "agent_span_count": snapshot.agent_span_count,
        "required_agent_conversation_count": (
            snapshot.required_agent_conversation_count
        ),
        "observed_agent_conversation_count": (
            snapshot.observed_agent_conversation_count
        ),
        "required_dataset_count": snapshot.required_dataset_count,
        "observed_dataset_count": snapshot.observed_dataset_count,
        "result_payload_count": required_publication_payloads["result"],
        "study_payload_count": required_publication_payloads["study"],
        "payload_count": scan["payload_count"],
        "payload_bytes_scanned": scan["payload_bytes_scanned"],
        "payload_set_sha256": scan["payload_set_sha256"],
        "secret_match_count": scan["secret_match_count"],
        "private_corpus_match_count": scan["private_corpus_match_count"],
        "private_structure_match_count": scan["private_structure_match_count"],
        "affected_payload_count": scan["affected_payload_count"],
        "missing_required_object_count": max(missing_required_objects, 0),
        "query_error_count": snapshot.query_error_count,
        "receipt_sha256": "",
    }
    redacted = redact_value(payload, secrets=secrets)
    if not isinstance(redacted, dict):  # pragma: no cover - fixed mapping
        raise TypeError("hosted evidence privacy receipt must be a mapping")
    unsigned = {**redacted, "receipt_sha256": ""}
    digest = _stable_digest(unsigned)
    atomic_write_json(receipt_path, {**unsigned, "receipt_sha256": digest})
    return HostedEvidencePrivacy(path=receipt_path, sha256=digest, status=status)


def _private_label_records(raw: bytes) -> list[dict[str, Any]]:
    """Read one JSON object or a nonempty JSONL mapping corpus."""

    text = raw.decode("utf-8")
    if not text.strip():
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    else:
        records = [value]
    if not records or any(not isinstance(item, Mapping) for item in records):
        raise TypeError("private labels must contain JSON object records")
    return [dict(item) for item in records]


def read_hosted_evidence_privacy_receipt(
    *,
    repo_root: Path,
    run_id: str,
) -> dict[str, Any]:
    path = (
        repo_root.resolve()
        / ".fugue"
        / "runtime"
        / run_id
        / HOSTED_EVIDENCE_PRIVACY_RECEIPT
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != PRIVACY_CONTRACT_VERSION
        or value.get("receipt_sha256")
        != _stable_digest({**value, "receipt_sha256": ""})
        or value.get("status") not in {
            "passed",
            "failed",
            "unavailable",
            "not_applicable",
        }
    ):
        raise ValueError("hosted evidence privacy receipt is invalid")
    return value


def _fetch_hosted_evidence_snapshot(
    *,
    rows: Sequence[Mapping[str, Any]],
    project: str,
    env: Mapping[str, str],
) -> _HostedEvidenceSnapshot:
    api_key = trace_api_key(env)
    required_calls = _required_call_ids(rows)
    trace_ids = _required_trace_ids(rows)
    dataset_refs = _required_dataset_refs(rows)
    conversation_ids = _required_conversation_ids(rows)
    if not api_key or project.count("/") != 1 or not required_calls or not dataset_refs:
        return _HostedEvidenceSnapshot(
            status="unavailable",
            payloads=(),
            required_call_count=len(required_calls),
            observed_required_call_count=0,
            descendant_call_count=0,
            agent_span_count=0,
            required_agent_conversation_count=len(conversation_ids),
            observed_agent_conversation_count=0,
            required_dataset_count=len(dataset_refs),
            observed_dataset_count=0,
            query_error_count=1,
        )
    base_url = (
        env.get("FUGUE_WEAVE_TRACE_SERVER_URL")
        or env.get("WF_TRACE_SERVER_URL")
        or WEAVE_AGENTS_BASE_URL
    ).rstrip("/")
    agents_base_url = (env.get("WEAVE_AGENTS_BASE_URL") or base_url).rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    query_errors = 0
    calls: list[dict[str, Any]] = []
    agent_spans: list[dict[str, Any]] = []
    datasets: list[Any] = []
    with httpx.Client(timeout=30.0, headers=headers) as client:
        try:
            calls.extend(
                _query_weave_calls(
                    client,
                    base_url=base_url,
                    project=project,
                    call_ids=sorted(required_calls),
                )
            )
            if trace_ids:
                calls.extend(
                    _query_weave_calls(
                        client,
                        base_url=base_url,
                        project=project,
                        trace_ids=sorted(trace_ids),
                    )
                )
        except (httpx.HTTPError, RuntimeError, ValueError):
            query_errors += 1
        try:
            agent_spans.extend(
                _query_agent_spans(
                    client,
                    base_url=agents_base_url,
                    project=project,
                    conversation_ids=sorted(conversation_ids),
                )
            )
        except (httpx.HTTPError, RuntimeError, ValueError):
            query_errors += 1
        try:
            datasets.extend(
                _read_weave_refs(
                    client,
                    base_url=base_url,
                    refs=sorted(dataset_refs),
                )
            )
        except (httpx.HTTPError, RuntimeError, ValueError):
            query_errors += 1
    unique_calls: dict[str, dict[str, Any]] = {}
    for value in calls:
        call_id = str(value.get("id") or value.get("call_id") or "")
        if call_id:
            unique_calls[call_id] = value
    observed_required = required_calls & set(unique_calls)
    project_mismatch = sum(
        1
        for value in unique_calls.values()
        if (
            (observed := _hosted_project_id(value))
            and observed != project
        )
    )
    descendant_calls = set(unique_calls) - required_calls
    observed_conversations = {
        str(value)
        for span in agent_spans
        if (value := span.get("conversation_id"))
    }
    status: ReceiptStatus = (
        "passed"
        if (
            not query_errors
            and not project_mismatch
            and observed_required == required_calls
            and len(datasets) == len(dataset_refs)
            and observed_conversations == conversation_ids
        )
        else "unavailable"
    )
    payloads = tuple(
        [
            *(("call", value) for value in unique_calls.values()),
            *(("agent_span", value) for value in agent_spans),
            *(("dataset", value) for value in datasets),
        ]
    )
    return _HostedEvidenceSnapshot(
        status=status,
        payloads=payloads,
        required_call_count=len(required_calls),
        observed_required_call_count=len(observed_required),
        descendant_call_count=len(descendant_calls),
        agent_span_count=len(agent_spans),
        required_agent_conversation_count=len(conversation_ids),
        observed_agent_conversation_count=len(observed_conversations),
        required_dataset_count=len(dataset_refs),
        observed_dataset_count=len(datasets),
        query_error_count=query_errors + project_mismatch,
    )


def _query_weave_calls(
    client: httpx.Client,
    *,
    base_url: str,
    project: str,
    call_ids: Sequence[str] = (),
    trace_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    call_filter: dict[str, Any] = {"trace_roots_only": False}
    if call_ids:
        call_filter["call_ids"] = list(call_ids)
    if trace_ids:
        call_filter["trace_ids"] = list(trace_ids)
    response = client.post(
        f"{base_url}/calls/stream_query",
        json={"project_id": project, "filter": call_filter},
    )
    if response.status_code >= 400:
        raise RuntimeError("Weave Calls query failed")
    return _decode_call_stream(response.text)


def _query_agent_spans(
    client: httpx.Client,
    *,
    base_url: str,
    project: str,
    conversation_ids: Sequence[str],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for conversation_id in conversation_ids:
        response = client.post(
            f"{base_url}/agents/spans/query",
            json={
                "project_id": project,
                "query": {
                    "$expr": {
                        "$eq": [
                            {"$getField": "conversation_id"},
                            {"$literal": conversation_id},
                        ]
                    }
                },
                "include_details": True,
                "include_costs": True,
                "limit": 10_000,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError("Weave Agents query failed")
        value = response.json()
        records = value if isinstance(value, list) else value.get("spans", [])
        if not isinstance(records, list):
            raise ValueError("Weave Agents query returned malformed data")
        spans.extend(item for item in records if isinstance(item, dict))
    return spans


def _read_weave_refs(
    client: httpx.Client,
    *,
    base_url: str,
    refs: Sequence[str],
) -> list[Any]:
    response = client.post(f"{base_url}/refs/read_batch", json={"refs": list(refs)})
    if response.status_code >= 400:
        raise RuntimeError("Weave reference read failed")
    value = response.json()
    records = value.get("vals") if isinstance(value, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("Weave reference read returned malformed data")
    return [record for record in records if record is not None]


def _decode_call_stream(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[:1] in {"[", "{"}:
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, Mapping) and isinstance(value.get("calls"), list):
            return [
                item for item in value["calls"] if isinstance(item, dict)
            ]
    result: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        value = json.loads(line)
        if isinstance(value, dict):
            result.append(value)
    return result


def _required_call_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    fields = (
        "weave_evaluation_root_call_id",
        "eval_predict_and_score_call_id",
        "weave_prediction_call_id",
        "prediction_call_id",
        "weave_agent_root_call_id",
        "native_agent_root_call_id",
    )
    return {
        str(value)
        for row in rows
        for field in fields
        if (value := row.get(field))
    }


def _required_call_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len(_required_call_ids(rows))


def _required_trace_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    fields = (
        "weave_evaluation_root_trace_id",
        "eval_predict_and_score_trace_id",
        "weave_prediction_trace_id",
        "weave_agent_bridge_trace_id",
    )
    return {
        str(value)
        for row in rows
        for field in fields
        if (value := row.get(field))
    }


def _required_dataset_refs(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for row in rows:
        value = (
            row.get("weave_dataset_ref")
            or row.get("weave_dataset_id")
            or row.get("dataset_id")
        )
        if value and str(value).startswith("weave:///"):
            refs.add(str(value))
    return refs


def _required_conversation_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        observed = row.get("observed_conversation_id")
        if observed:
            result.add(str(observed))
        for field in ("native_session_ids", "weave_conversation_ids"):
            for value in row.get(field) or ():
                if value:
                    result.add(str(value))
    return result


def _hosted_project_id(value: Mapping[str, Any]) -> str:
    direct = value.get("project_id")
    if direct:
        return str(direct)
    project = value.get("project")
    if isinstance(project, Mapping) and project.get("id"):
        return str(project["id"])
    return ""


def _private_label_patterns(
    raw: bytes,
    rows: Sequence[Any],
) -> tuple[bytes, ...]:
    patterns = {
        line.strip()
        for line in raw.splitlines()
        if len(line.strip()) >= 8
    }
    patterns.update(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        for row in rows
    )
    return tuple(sorted(patterns, key=len, reverse=True))


def _scan_hosted_payloads(
    payloads: Sequence[tuple[str, Any]],
    *,
    secrets: Sequence[str],
    private_patterns: Sequence[bytes],
) -> dict[str, Any]:
    if len(payloads) > _MAX_HOSTED_PAYLOADS:
        return {
            "status": "unavailable",
            "payload_count": 0,
            "payload_bytes_scanned": 0,
            "payload_set_sha256": _stable_digest([]),
            "secret_match_count": 0,
            "private_corpus_match_count": 0,
            "private_structure_match_count": 0,
            "affected_payload_count": 0,
            "match_count": 0,
        }
    secret_values = tuple(value.encode() for value in secrets)
    payload_digests: list[str] = []
    total_bytes = 0
    secret_matches = 0
    private_matches = 0
    structure_matches = 0
    affected = 0
    for kind, value in payloads:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        total_bytes += len(encoded)
        if total_bytes > _MAX_HOSTED_PAYLOAD_BYTES:
            return {
                "status": "unavailable",
                "payload_count": len(payload_digests),
                "payload_bytes_scanned": total_bytes,
                "payload_set_sha256": _stable_digest(payload_digests),
                "secret_match_count": secret_matches,
                "private_corpus_match_count": private_matches,
                "private_structure_match_count": structure_matches,
                "affected_payload_count": affected,
                "match_count": secret_matches + private_matches + structure_matches,
            }
        payload_digests.append(
            _stable_digest(
                {
                    "kind": kind,
                    "payload_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        )
        secret_count = sum(encoded.count(secret) for secret in secret_values)
        private_count = sum(
            encoded.count(pattern) for pattern in private_patterns
        )
        structural_count = len(_private_key_paths(value))
        secret_matches += secret_count
        private_matches += private_count
        structure_matches += structural_count
        if secret_count or private_count or structural_count:
            affected += 1
    return {
        "status": "passed",
        "payload_count": len(payloads),
        "payload_bytes_scanned": total_bytes,
        "payload_set_sha256": _stable_digest(sorted(payload_digests)),
        "secret_match_count": secret_matches,
        "private_corpus_match_count": private_matches,
        "private_structure_match_count": structure_matches,
        "affected_payload_count": affected,
        "match_count": secret_matches + private_matches + structure_matches,
    }


def _local_agent_jobs(
    jobs: Sequence[RenderedJob],
) -> tuple[list[RenderedJob], str]:
    agents = [
        job
        for job in jobs
        if bool(job.applicable) and str(job.execution_kind) == "agent"
    ]
    if not agents:
        return [], "none"
    if any(
        isinstance(job.config.get("fugue"), Mapping)
        and (job.config.get("fugue") or {}).get("wandb_serverless_runtime")
        for job in agents
    ):
        return [], "wandb_serverless"
    if not all(job.command and Path(str(job.command[0])).name == "harbor" for job in agents):
        return [], "unknown"
    return agents, "local_harbor_docker"


def _execution_identity(
    run_id: str,
    jobs: Sequence[RenderedJob],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    from fugue.bench.sandbox_policy import verify_harbor_job_attestation

    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for job in jobs:
        config = job.config
        fugue = config.get("fugue")
        if not isinstance(fugue, Mapping):
            issues.append(f"{job.job_name}: missing fugue configuration")
            continue
        if str(fugue.get("run_id") or "") != run_id:
            issues.append(f"{job.job_name}: run identity disagrees")
        if str(getattr(job, "run_id", "") or "") != run_id:
            issues.append(f"{job.job_name}: rendered job run identity disagrees")
        attestation = fugue.get("sandbox_attestation")
        policy = (
            dict(attestation)
            if isinstance(attestation, Mapping)
            else {}
        )
        required_policy = (
            "attestation_digest",
            "policy_digest",
            "policy_version",
        )
        if policy.get("source") != "rendered_harbor_config" or any(
            not policy.get(key) for key in required_policy
        ):
            issues.append(f"{job.job_name}: local Harbor policy attestation is incomplete")
        policy_verified = True
        try:
            verify_harbor_job_attestation(job.config_path, repo_root)
        except (OSError, TypeError, ValueError):
            policy_verified = False
            issues.append(
                f"{job.job_name}: rendered Harbor policy attestation did not reverify"
            )
        agent_runtime = _runtime_identity(fugue.get("agent_runtime"))
        raw_task_runtime = fugue.get("task_runtime")
        task_runtime = _runtime_identity(raw_task_runtime)
        if agent_runtime["status"] != "passed":
            issues.append(f"{job.job_name}: Agent runtime identity is incomplete")
        if raw_task_runtime is not None and task_runtime["status"] != "passed":
            issues.append(f"{job.job_name}: task runtime identity is incomplete")
        fingerprint = str(fugue.get("execution_fingerprint") or "")
        resolved = getattr(job, "resolved_candidate", None)
        locked_fingerprint = str(
            getattr(resolved, "execution_fingerprint", "") or ""
        )
        execution_definition = getattr(resolved, "execution_definition", None)
        recomputed_fingerprint = (
            _stable_digest(execution_definition)
            if isinstance(execution_definition, Mapping)
            else ""
        )
        fingerprint_verified = bool(
            _HEX_SHA256.fullmatch(fingerprint)
            and fingerprint == locked_fingerprint
            and fingerprint == recomputed_fingerprint
        )
        if not fingerprint:
            issues.append(f"{job.job_name}: execution fingerprint is missing")
        elif not fingerprint_verified:
            issues.append(
                f"{job.job_name}: execution fingerprint does not match the "
                "resolved execution definition"
            )
        candidate_id = str(fugue.get("candidate_id") or "")
        locked_candidate_id = str(getattr(resolved, "candidate_id", "") or "")
        if (
            not _HEX_SHA256.fullmatch(candidate_id)
            or candidate_id != locked_candidate_id
        ):
            issues.append(
                f"{job.job_name}: candidate identity does not match the resolved "
                "candidate"
            )
        rows.append(
            {
                "job_name": str(job.job_name),
                "config_sha256": _file_digest(job.config_path),
                "execution_fingerprint": fingerprint,
                "locked_execution_fingerprint": locked_fingerprint,
                "recomputed_execution_fingerprint": recomputed_fingerprint,
                "execution_fingerprint_verified": fingerprint_verified,
                "candidate_id": candidate_id,
                "candidate_id_verified": candidate_id == locked_candidate_id,
                "policy": {
                    "source": policy.get("source"),
                    "attestation_digest": policy.get("attestation_digest"),
                    "policy_digest": policy.get("policy_digest"),
                    "policy_version": policy.get("policy_version"),
                    "verification": "passed" if policy_verified else "failed",
                    "verification_scope": (
                        "exact rendered Harbor config and referenced Compose assets"
                    ),
                },
                "agent_runtime": agent_runtime,
                "task_runtime": task_runtime,
            }
        )
    return {
        "status": "failed" if issues else "passed",
        "job_count": len(jobs),
        "jobs": rows,
        "issues": issues,
    }


def _runtime_identity(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {
            "status": "not_applicable",
            "identity_sha256": None,
            "image": None,
            "image_id": None,
            "recipe_sha256": None,
            "architecture": None,
            "os": None,
            "version": None,
        }
    value = dict(raw) if isinstance(raw, Mapping) else {}
    image = str(value.get("image") or "")
    image_id = str(value.get("image_id") or "")
    recipe_sha256 = str(value.get("recipe_sha256") or "")
    architecture = str(value.get("architecture") or "")
    os_name = str(value.get("os") or "")
    status = (
        "passed"
        if (
            image
            and _IMAGE_SHA256.fullmatch(image_id)
            and _HEX_SHA256.fullmatch(recipe_sha256)
            and architecture
            and os_name
        )
        else "unavailable"
    )
    return {
        "status": status,
        "identity_basis": "prepared runtime lock in exact rendered Harbor config",
        "identity_sha256": _stable_digest(value) if value else None,
        "image": image or None,
        "image_id": image_id or None,
        "recipe_sha256": recipe_sha256 or None,
        "architecture": architecture or None,
        "os": os_name or None,
        "version": value.get("version"),
    }


def _scan_run_artifacts(  # noqa: C901 - one bounded audit reports every gap
    *,
    repo_root: Path,
    run_id: str,
    run_dir: Path,
    jobs: Sequence[RenderedJob],
    secrets: Sequence[str],
) -> dict[str, Any]:
    if not secrets:
        return {
            "status": "unavailable",
            "reason": "no configured secret values were available to verify",
            "scope": {
                "kind": "exact_local_run_artifacts",
                "included": [],
                "excluded": [
                    "hosted Weave objects",
                    "external services",
                    "removed container filesystems",
                ],
            },
            "configured_secret_count": 0,
            "files_scanned": 0,
            "bytes_scanned": 0,
            "files_with_matches": [],
        }
    roots: set[Path] = {run_dir}
    errors: list[str] = []
    runtime_root = (repo_root / ".fugue" / "runtime").resolve()
    for job in jobs:
        raw = job.config.get("jobs_dir")
        if not isinstance(raw, str) or not raw:
            errors.append(f"{job.job_name}: jobs_dir is unavailable")
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        resolved = path.resolve()
        if runtime_root not in resolved.parents or run_id not in resolved.parts:
            errors.append(f"{job.job_name}: jobs_dir is outside the exact run scope")
            continue
        roots.add(resolved)
    files: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            errors.append(f"run-scoped artifact directory is unavailable: {_safe_path(root, repo_root)}")
            continue
        for path in root.rglob("*"):
            if path.name == HARBOR_CONFORMANCE_RECEIPT:
                continue
            if path.is_symlink():
                errors.append(
                    f"run-scoped symlink was not scanned: {_safe_path(path, repo_root)}"
                )
                continue
            if path.is_file():
                files.add(path)
                if len(files) > _MAX_SCAN_FILES:
                    errors.append("run-scoped secret scan exceeded the file limit")
                    break
    matched: list[dict[str, Any]] = []
    scanned = 0
    total_bytes = 0
    secret_bytes = tuple(secret.encode("utf-8") for secret in secrets)
    for path in sorted(files):
        try:
            size = path.stat().st_size
        except OSError:
            errors.append(f"could not stat {_safe_path(path, repo_root)}")
            continue
        if size > _MAX_SCAN_FILE_BYTES:
            errors.append(f"file exceeded scan limit: {_safe_path(path, repo_root)}")
            continue
        total_bytes += size
        if total_bytes > _MAX_SCAN_TOTAL_BYTES:
            errors.append("run-scoped secret scan exceeded the total byte limit")
            break
        try:
            content = path.read_bytes()
        except OSError:
            errors.append(f"could not read {_safe_path(path, repo_root)}")
            continue
        count = sum(content.count(secret) for secret in secret_bytes)
        scanned += 1
        if count:
            matched.append(
                {
                    "path": _safe_path(path, repo_root),
                    "match_count": count,
                }
            )
    status: ReceiptStatus
    if matched:
        status = "failed"
    elif errors:
        status = "unavailable"
    else:
        status = "passed"
    return {
        "status": status,
        "scope": {
            "kind": "exact_local_run_artifacts",
            "included": sorted(_safe_path(root, repo_root) for root in roots),
            "excluded": [
                "hosted Weave objects",
                "external services",
                "removed container filesystems",
            ],
        },
        "configured_secret_count": len(secrets),
        "files_scanned": scanned,
        "bytes_scanned": total_bytes,
        "files_with_matches": matched,
        "errors": errors,
    }


def _private_label_boundary(  # noqa: C901 - one fail-closed taint boundary
    *,
    run_dir: Path,
    jobs: Sequence[RenderedJob],
    host_scorer_names: Sequence[str],
) -> dict[str, Any]:
    def canonical(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    def structural_taints(value: Any) -> set[str]:
        taints: set[str] = set()
        if isinstance(value, Mapping):
            if value:
                taints.add(canonical(value))
            for item in value.values():
                taints.update(structural_taints(item))
        elif isinstance(value, list):
            if value:
                taints.add(canonical(value))
            for item in value:
                taints.update(structural_taints(item))
        return taints

    def private_label_digests(value: Any) -> set[str]:
        digests: set[str] = set()
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key).lower()
                if key in {"private_labels_sha256", "private_labels_digest"}:
                    rendered = str(item or "")
                    if _HEX_SHA256.fullmatch(rendered):
                        digests.add(rendered)
                digests.update(private_label_digests(item))
        elif isinstance(value, list):
            for item in value:
                digests.update(private_label_digests(item))
        return digests

    def scan_agent_input(
        value: Any,
        *,
        prefix: str,
        taints: set[str],
        private_paths: set[Path],
        repo_root: Path,
    ) -> list[str]:
        matches: list[str] = []
        if isinstance(value, Mapping):
            if value and canonical(value) in taints:
                matches.append(prefix)
            for raw_key, item in value.items():
                matches.extend(
                    scan_agent_input(
                        item,
                        prefix=f"{prefix}.{raw_key}",
                        taints=taints,
                        private_paths=private_paths,
                        repo_root=repo_root,
                    )
                )
        elif isinstance(value, list):
            if value and canonical(value) in taints:
                matches.append(prefix)
            for index, item in enumerate(value):
                matches.extend(
                    scan_agent_input(
                        item,
                        prefix=f"{prefix}[{index}]",
                        taints=taints,
                        private_paths=private_paths,
                        repo_root=repo_root,
                    )
                )
        elif isinstance(value, str):
            rendered = value.strip()
            if rendered in taints:
                matches.append(prefix)
            if ".fugue/private/" in rendered.replace("\\", "/"):
                matches.append(prefix)
            try:
                candidate = Path(rendered).expanduser()
            except (OSError, RuntimeError, ValueError):
                return matches
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                resolved = candidate.absolute()
            if resolved in private_paths:
                matches.append(prefix)
        return matches

    if not host_scorer_names:
        return {
            "status": "not_applicable",
            "reason": "no host evaluator was configured",
            "scope": "rendered Harbor Agent-input structures only",
            "rendered_private_fields": [],
        }
    findings: list[str] = []
    for job in jobs:
        for path in _private_key_paths(job.config):
            findings.append(f"{job.job_name}:{path}")
    asset_lock = run_dir / "evaluation-assets.json"
    if not asset_lock.is_file():
        return {
            "status": "unavailable",
            "reason": "the host-only evaluation asset lock is unavailable",
            "scope": "rendered Harbor Agent-input structures only",
            "rendered_private_fields": findings,
        }
    repo_root = run_dir.resolve().parents[2]
    try:
        asset_payload = json.loads(asset_lock.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "status": "unavailable",
            "reason": "the host-only evaluation asset lock is malformed",
            "scope": "rendered Harbor Agent-input structures only",
            "rendered_private_fields": sorted(findings),
        }
    if not isinstance(asset_payload, Mapping):
        return {
            "status": "unavailable",
            "reason": "the host-only evaluation asset lock is not an object",
            "scope": "rendered Harbor Agent-input structures only",
            "rendered_private_fields": sorted(findings),
        }

    private_paths = {asset_lock.resolve()}
    taints = structural_taints(asset_payload.get("predictions") or {})
    declared_private_missing: list[str] = []
    input_lock = run_dir / "input-lock.json"
    if input_lock.is_file():
        try:
            input_payload = json.loads(input_lock.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {
                "status": "unavailable",
                "reason": "the run input lock is malformed",
                "scope": "rendered Harbor Agent-input structures only",
                "rendered_private_fields": sorted(findings),
            }
        for digest in sorted(private_label_digests(input_payload)):
            path = (
                repo_root
                / ".fugue/private/comparison-inputs/labels"
                / f"{digest}.jsonl"
            )
            if not path.is_file():
                declared_private_missing.append(digest)
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return {
                    "status": "failed",
                    "reason": "a frozen private-label bundle failed digest verification",
                    "scope": "rendered Harbor Agent-input structures only",
                    "rendered_private_fields": sorted(findings),
                }
            private_paths.add(path.resolve())
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        taints.update(structural_taints(json.loads(line)))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return {
                    "status": "failed",
                    "reason": "a frozen private-label bundle is malformed",
                    "scope": "rendered Harbor Agent-input structures only",
                    "rendered_private_fields": sorted(findings),
                }
    if declared_private_missing:
        return {
            "status": "unavailable",
            "reason": "a declared frozen private-label bundle is unavailable",
            "scope": "rendered Harbor Agent-input structures only",
            "rendered_private_fields": sorted(findings),
            "missing_private_bundle_digests": declared_private_missing,
        }

    tainted_inputs: list[str] = []
    for job in jobs:
        tainted_inputs.extend(
            f"{job.job_name}:{path}"
            for path in scan_agent_input(
                job.config,
                prefix="$",
                taints=taints,
                private_paths=private_paths,
                repo_root=repo_root,
            )
        )
    mode = asset_lock.stat().st_mode & 0o777
    restricted = mode & 0o077 == 0
    if findings or tainted_inputs or not restricted:
        return {
            "status": "failed",
            "method": "rendered Agent-input structural and host-taint scan",
            "scope": "rendered Harbor Agent-input structures only",
            "host_asset_lock": asset_lock.name,
            "host_asset_lock_mode": oct(mode),
            "rendered_private_fields": sorted(findings),
            "tainted_agent_inputs": sorted(set(tainted_inputs)),
        }
    return {
        "status": "passed",
        "method": "rendered Agent-input structural and host-taint scan",
        "scope": "rendered Harbor Agent-input structures only",
        "host_asset_lock": asset_lock.name,
        "host_asset_lock_mode": oct(mode),
        "rendered_private_fields": [],
        "tainted_agent_inputs": [],
    }


def _private_key_paths(value: Any, prefix: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            child = f"{prefix}.{key}"
            if key.lower() in _PRIVATE_INPUT_KEYS:
                findings.append(child)
            findings.extend(_private_key_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_private_key_paths(item, f"{prefix}[{index}]"))
    return findings


def _docker_cleanup_audit(
    *,
    repo_root: Path,
    run_dir: Path,
    run_id: str,
    jobs: Sequence[RenderedJob],
    pre_execution_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    projects, derivation_errors = _compose_projects_for_jobs(
        repo_root=repo_root,
        run_id=run_id,
        jobs=jobs,
    )
    base = {
        "matched_containers": [],
        "matched_networks": [],
        "mutations_performed": False,
        "scope": {
            "kind": "exact_run_compose_projects",
            "run_id": run_id,
            "run_directory": run_dir.resolve().as_posix(),
            "compose_projects": projects,
            "resource_types": ["container", "network"],
            "selector": "com.docker.compose.project=<exact project>",
            "excluded": [
                "other Compose projects",
                "non-Docker resources",
            ],
        },
    }
    inventory = (
        dict(pre_execution_inventory)
        if isinstance(pre_execution_inventory, Mapping)
        else {}
    )
    if (
        inventory.get("schema_version") != 1
        or inventory.get("backend") != "local_harbor_docker"
        or inventory.get("status") != "passed"
        or not isinstance(inventory.get("container_ids"), list)
    ):
        return {
            **base,
            "status": "unavailable",
            "reason": "the pre-execution Docker container inventory is unavailable",
            "errors": derivation_errors,
            "pre_execution_inventory_status": str(
                inventory.get("status") or "missing"
            ),
        }
    if derivation_errors or not projects:
        return {
            **base,
            "status": "unavailable",
            "reason": (
                "exact run Compose projects could not be established"
                if not projects
                else "some exact run Compose projects could not be established"
            ),
            "errors": derivation_errors,
            "pre_execution_inventory_status": "passed",
        }

    pre_container_ids = {
        str(value) for value in inventory["container_ids"] if str(value)
    }
    post_container_ids, post_error = _docker_ids(
        ["docker", "container", "ls", "--all", "--quiet"],
        resource="container",
    )
    if post_error is not None:
        return {
            **base,
            "status": "unavailable",
            "reason": post_error,
            "errors": [],
            "pre_execution_inventory_status": "passed",
        }
    new_container_ids = sorted(set(post_container_ids) - pre_container_ids)
    if len(new_container_ids) > _MAX_NEW_CONTAINER_INSPECTIONS:
        return {
            **base,
            "status": "unavailable",
            "reason": "new Docker containers exceeded the bounded inspection limit",
            "errors": [],
            "pre_execution_inventory_status": "passed",
            "new_remaining_container_count": len(new_container_ids),
        }

    job_directories = sorted(
        {Path(job.result_path).parent.resolve().as_posix() for job in jobs}
    )
    matched_containers: list[dict[str, str]] = []
    matched_networks: list[dict[str, str]] = []
    unattributed_new_containers: list[str] = []
    for container_id in new_container_ids:
        attribution, inspect_error = _container_run_attribution(
            container_id,
            run_id=run_id,
            run_directory=run_dir.resolve().as_posix(),
            job_directories=job_directories,
            compose_projects=projects,
        )
        if inspect_error is not None:
            return {
                **base,
                "status": "unavailable",
                "reason": inspect_error,
                "matched_containers": matched_containers,
                "matched_networks": matched_networks,
                "errors": [],
                "pre_execution_inventory_status": "passed",
            }
        if attribution:
            matched_containers.append(
                {
                    "container_id": container_id[:12],
                    "compose_project": str(attribution.get("compose_project") or ""),
                    "attribution": ",".join(attribution["reasons"]),
                }
            )
        else:
            unattributed_new_containers.append(container_id[:12])
    for project in projects:
        container_ids, container_error = _docker_ids(
            [
                "docker",
                "container",
                "ls",
                "--all",
                "--quiet",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ],
            resource="container",
        )
        if container_error:
            return {
                **base,
                "status": "unavailable",
                "reason": container_error,
                "matched_containers": matched_containers,
                "matched_networks": matched_networks,
            }
        for value in container_ids:
            if not any(
                row["container_id"] == value[:12] for row in matched_containers
            ):
                matched_containers.append(
                    {
                        "container_id": value[:12],
                        "compose_project": project,
                        "attribution": "exact_compose_project_label",
                    }
                )
        network_ids, network_error = _docker_ids(
            [
                "docker",
                "network",
                "ls",
                "--quiet",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ],
            resource="network",
        )
        if network_error:
            return {
                **base,
                "status": "unavailable",
                "reason": network_error,
                "matched_containers": matched_containers,
                "matched_networks": matched_networks,
            }
        matched_networks.extend(
            {"network_id": value[:12], "compose_project": project}
            for value in network_ids
        )
    status: ReceiptStatus
    if matched_containers or matched_networks:
        status = "failed"
    elif unattributed_new_containers:
        # A concurrent unrelated launch is possible, so this is not a Fugue
        # orphan claim. It also means the inventory cannot prove that an
        # unlabeled Harbor container was cleaned up.
        status = "unavailable"
    else:
        status = "passed"
    return {
        **base,
        "status": status,
        "matched_containers": matched_containers,
        "matched_networks": matched_networks,
        "pre_execution_inventory_status": "passed",
        "new_remaining_container_count": len(new_container_ids),
        "unattributed_new_containers": unattributed_new_containers,
        "errors": [],
    }


def _compose_projects_for_jobs(
    *,
    repo_root: Path,
    run_id: str,
    jobs: Sequence[RenderedJob],
) -> tuple[list[str], list[str]]:
    allowed_roots = (
        (repo_root / "jobs").resolve(),
        (repo_root / ".fugue" / "runtime" / "jobs").resolve(),
    )
    projects: set[str] = set()
    errors: list[str] = []
    for job in jobs:
        job_dir = Path(job.result_path).parent.resolve()
        if (
            run_id not in job_dir.parts
            or not any(job_dir.is_relative_to(root) for root in allowed_roots)
        ):
            errors.append(f"{job.job_name}: Harbor result path is outside run scope")
            continue
        try:
            children = tuple(job_dir.iterdir())
        except OSError:
            errors.append(f"{job.job_name}: Harbor result directory is unavailable")
            continue
        job_projects = {
            f"{child.name.lower()}__env"
            for child in children
            if child.is_dir() and "__" in child.name
        }
        valid = {
            project
            for project in job_projects
            if re.fullmatch(
                r"[a-z0-9][a-z0-9_.-]*(?:__[a-z0-9_.-]+)+",
                project,
            )
        }
        if not valid:
            errors.append(
                f"{job.job_name}: no exact Harbor Compose project was recorded"
            )
            continue
        projects.update(valid)
    return sorted(projects), errors


def _docker_ids(
    command: list[str],
    *,
    resource: str,
) -> tuple[list[str], str | None]:
    try:
        listed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"Docker {resource} listing failed: {type(exc).__name__}"
    if listed.returncode:
        return [], f"Docker {resource} listing returned a nonzero status"
    return sorted({value for value in listed.stdout.splitlines() if value}), None


def _container_run_attribution(
    container_id: str,
    *,
    run_id: str,
    run_directory: str,
    job_directories: Sequence[str],
    compose_projects: Sequence[str],
) -> tuple[dict[str, Any], str | None]:
    try:
        inspected = subprocess.run(
            ["docker", "container", "inspect", container_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"Docker container inspection failed: {type(exc).__name__}"
    if inspected.returncode:
        return {}, "Docker container inspection returned a nonzero status"
    try:
        values = json.loads(inspected.stdout)
    except json.JSONDecodeError:
        values = None
    if not isinstance(values, list) or len(values) != 1:
        return {}, "Docker container inspection returned malformed JSON"
    value = values[0]
    config = value.get("Config") if isinstance(value.get("Config"), Mapping) else {}
    labels = config.get("Labels") if isinstance(config.get("Labels"), Mapping) else {}
    env = config.get("Env") if isinstance(config.get("Env"), list) else []
    mounts = value.get("Mounts") if isinstance(value.get("Mounts"), list) else []
    reasons: list[str] = []
    compose_project = str(labels.get("com.docker.compose.project") or "")
    if compose_project in compose_projects:
        reasons.append("exact_compose_project_label")
    if any(
        str(labels.get(key) or "") == run_id
        for key in ("fugue.run_id", "io.fugue.run_id")
    ):
        reasons.append("exact_fugue_run_label")
    if f"FUGUE_RUN_ID={run_id}" in {str(item) for item in env}:
        reasons.append("exact_fugue_run_environment")
    marker_paths = (run_directory, *job_directories)
    for mount in mounts:
        if not isinstance(mount, Mapping):
            continue
        source = str(mount.get("Source") or "")
        if any(
            source == marker or source.startswith(f"{marker}/")
            for marker in marker_paths
        ):
            reasons.append("exact_run_directory_mount")
            break
    return (
        {
            "compose_project": compose_project,
            "reasons": sorted(set(reasons)),
        }
        if reasons
        else {},
        None,
    )


def _write_receipt(
    path: Path,
    payload: dict[str, Any],
    *,
    secrets: Sequence[str],
) -> HarborRunConformance:
    redacted = redact_value(payload, secrets=secrets)
    if not isinstance(redacted, dict):  # pragma: no cover - payload is fixed above
        raise TypeError("Harbor conformance receipt must be a mapping")
    unsigned = {
        **redacted,
        "redaction_applied": redacted != payload,
        "receipt_sha256": "",
    }
    digest = _stable_digest(unsigned)
    value = {**unsigned, "receipt_sha256": digest}
    atomic_write_json(path, value)
    return HarborRunConformance(
        path=path,
        sha256=digest,
        status=str(value["status"]),  # type: ignore[arg-type]
        enforced=bool(value["enforced"]),
    )


def _read_existing_receipt(
    path: Path,
    *,
    run_id: str,
) -> HarborRunConformance:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported Harbor conformance receipt")
    if value.get("run_id") != run_id:
        raise ValueError("Harbor conformance receipt run identity disagrees")
    digest = str(value.get("receipt_sha256") or "")
    if not digest or _stable_digest({**value, "receipt_sha256": ""}) != digest:
        raise ValueError("Harbor conformance receipt digest does not match")
    status = str(value.get("status") or "")
    if status not in {"passed", "failed", "unavailable", "not_applicable"}:
        raise ValueError("Harbor conformance receipt status is invalid")
    return HarborRunConformance(
        path=path,
        sha256=digest,
        status=status,  # type: ignore[arg-type]
        enforced=bool(value.get("enforced")),
    )


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.name
