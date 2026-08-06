from __future__ import annotations

import asyncio
import json
import os
import statistics
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import weave
from filelock import FileLock

from fugue.bench.analysis_contracts import EvidenceDriftCheckV1
from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.operator import load_env
from fugue.bench.source_filters import task_evidence_wandb_runs
from fugue.model_plane import trace_api_key

QUALIFICATION_RESULT_PROJECT = "wandb/fugue-mcp-release-qualification-v1"
QUALIFICATION_SOURCE_PROJECT = "wandb/fugue-mcp-release-source-v2"
QUALIFICATION_API_BASE_URL = "https://api.wandb.ai"
QUALIFICATION_TRACE_BASE_URL = "https://trace.wandb.ai"
# Backward-compatible name for the original single-project contract.
QUALIFICATION_PROJECT = QUALIFICATION_RESULT_PROJECT
MCP_RELEASE_NOTES_LOCK = Path(
    "examples/comparisons/wandb-mcp-maintenance/release-notes.lock.json"
)
_MCP_RELEASE_NOTES_COMMIT = "29cc1b5b5cf4061afa1faa712021fa1b68ad0bf7"
_MCP_RELEASE_NOTES_SHA256 = (
    "257110b2542caec6bf30c24cfb59c1a3ee1074b69ad68497b9978ba130243e12"
)
_MCP_RELEASE_NOTES_BYTES = 8187
_CURRENT_MCP_RELEASE_NOTES_COMMIT = (
    "5c6cc1c9a1079296daf6613ea6d12daebdd8bcba"
)
_CURRENT_MCP_RELEASE_NOTES_SHA256 = (
    "1c90d85057d48fcb54e70e2b799566abf711f149318a86e44d45676f613d86d2"
)
_CURRENT_MCP_RELEASE_NOTES_BYTES = 9168
_MCP_RELEASE_NOTES_SOURCES = {
    _MCP_RELEASE_NOTES_COMMIT: (
        _MCP_RELEASE_NOTES_SHA256,
        _MCP_RELEASE_NOTES_BYTES,
    ),
    _CURRENT_MCP_RELEASE_NOTES_COMMIT: (
        _CURRENT_MCP_RELEASE_NOTES_SHA256,
        _CURRENT_MCP_RELEASE_NOTES_BYTES,
    ),
}
EVIDENCE_LOCK_SCHEMA_VERSION = 1
_REQUIRED_COUNTS = {
    "runs": 6,
    "source_conversations": 24,
    "tool_spans": 48,
    "dataset_rows": 8,
    "aligned_evaluation_pairs": 8,
    "evaluation_prediction_rows": 16,
}
SOURCE_CONFORMANCE_SCHEMA_VERSION = 2
SOURCE_PREPARATION_PROGRESS_SCHEMA_VERSION = 1
_SOURCE_CONFORMANCE_EXPECTATIONS = {
    "evaluation_roots": 2,
    "direct_children": 18,
    "predict_and_score_children": 16,
    "summarize_children": 2,
    "repaired_candidate_prediction_children": 16,
}
DEFAULT_MCP_RELEASE_CANDIDATES = (
    (
        "wandb-mcp-main",
        "git:53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0",
    ),
    (
        "wandb-mcp-0-4-staging",
        "git:29cc1b5b5cf4061afa1faa712021fa1b68ad0bf7",
    ),
)
# Backward-compatible private alias for callers that imported the original
# module-level tuple.
_MCP_RELEASE_CANDIDATES = DEFAULT_MCP_RELEASE_CANDIDATES

_MCP_PROJECT_TASK_TOOLS: dict[str, tuple[str, ...]] = {
    "maintainer-source-inventory": (
        "probe_project_tool",
        "query_wandb_tool",
    ),
    "maintainer-run-triage": (
        "compare_runs_tool",
        "diagnose_run_tool",
        "get_run_history_tool",
    ),
    "maintainer-evaluation-trace-topology": (
        "count_weave_traces_tool",
        "infer_trace_schema_tool",
        "query_weave_traces_tool",
        "resolve_trace_roots_tool",
        "summarize_evaluation_tool",
    ),
    "maintainer-artifact-provenance": (
        "compare_artifact_versions_tool",
        "get_artifact_details_tool",
        "list_artifact_versions_tool",
    ),
}
_MCP_PROJECT_TOOL_TASK_ALIASES: dict[str, tuple[str, ...]] = {
    "compare_artifact_versions_tool": ("artifact-provenance",),
    "compare_runs_tool": ("selective-run-comparison",),
    "count_weave_traces_tool": (
        "evaluation-summary-accuracy",
        "trace-source-use",
    ),
    "diagnose_run_tool": ("selective-run-comparison",),
    "get_artifact_details_tool": ("artifact-provenance",),
    "get_run_history_tool": ("exact-history-target",),
    "infer_trace_schema_tool": (
        "evaluation-summary-accuracy",
        "trace-source-use",
    ),
    "list_artifact_versions_tool": ("artifact-provenance",),
    "probe_project_tool": ("run-inventory-projection",),
    "query_wandb_tool": (
        "run-inventory-projection",
        "filtered-failure-triage",
        "missing-cost-honesty",
    ),
    "query_weave_traces_tool": (
        "evaluation-summary-accuracy",
        "trace-source-use",
    ),
    "resolve_trace_roots_tool": (
        "evaluation-summary-accuracy",
        "trace-source-use",
    ),
    "summarize_evaluation_tool": ("evaluation-summary-accuracy",),
}
_MCP_ZERO_MODEL_TOOLS: dict[str, str] = {
    "list_entities_tool": (
        "Account-scoped discovery is mutable and is checked by exact manifest "
        "and schema conformance, not a locked-project outcome task."
    ),
    "query_wandb_entity_projects": (
        "Entity-wide project discovery would exceed the immutable source scope."
    ),
    "list_registries_tool": ("Registry discovery is organization-scoped and mutable."),
    "list_registry_collections_tool": (
        "Registry collections are outside the locked project cohort."
    ),
    "list_wandb_automations_tool": ("Automations are mutable account configuration."),
    "list_wandb_integrations_tool": ("Integrations are mutable account configuration."),
    "search_wandb_docs_tool": (
        "Documentation search targets an external mutable corpus."
    ),
}


def _validate_mcp_release_candidates(
    candidates: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Return one exact baseline/candidate pair in declared role order."""

    if isinstance(candidates, (str, bytes)) or len(candidates) != 2:
        raise ValueError(
            "MCP release qualification requires exactly one baseline and "
            "one candidate"
        )
    normalized: list[tuple[str, str]] = []
    for role, item in zip(("baseline", "candidate"), candidates, strict=True):
        if (
            isinstance(item, (str, bytes))
            or not isinstance(item, Sequence)
            or len(item) != 2
        ):
            raise ValueError(
                f"MCP release {role} must contain import id and version identity"
            )
        import_id = str(item[0]).strip()
        version_identity = str(item[1]).strip()
        if not import_id or not version_identity:
            raise ValueError(
                f"MCP release {role} import id and version identity are required"
            )
        normalized.append((import_id, version_identity))
    if normalized[0][0] == normalized[1][0]:
        raise ValueError("MCP release baseline and candidate import ids must differ")
    if normalized[0][1] == normalized[1][1]:
        raise ValueError(
            "MCP release baseline and candidate version identities must differ"
        )
    return normalized[0], normalized[1]


def _mcp_release_candidate_commit(version_identity: str) -> str:
    prefix = "git:"
    commit = version_identity.removeprefix(prefix)
    if (
        not version_identity.startswith(prefix)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError(
            "MCP release qualification requires an exact lowercase git commit"
        )
    return commit


def _mcp_release_candidate_bindings(
    candidates: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    selected = _validate_mcp_release_candidates(candidates)
    return [
        {
            "role": role,
            "import_id": import_id,
            "version_identity": version_identity,
        }
        for role, (import_id, version_identity) in zip(
            ("baseline", "candidate"),
            selected,
            strict=True,
        )
    ]


def _validate_mcp_release_observations(
    observations: Sequence[Mapping[str, Any]],
    candidates: Sequence[tuple[str, str]],
) -> None:
    selected = _validate_mcp_release_candidates(candidates)
    observed = [
        (
            str(item.get("id") or ""),
            str(item.get("version_identity") or ""),
        )
        for item in observations
    ]
    if observed != list(selected):
        raise ValueError(
            "MCP qualification observations do not match the exact "
            "baseline/candidate bindings"
        )


def _mcp_release_candidates_from_receipt(
    receipt: Mapping[str, Any],
) -> tuple[tuple[str, str], tuple[str, str]]:
    raw = receipt.get("candidate_bindings")
    if raw is None:
        return DEFAULT_MCP_RELEASE_CANDIDATES
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or len(raw) != 2
    ):
        raise ValueError("MCP candidate bindings are invalid")
    expected_roles = ("baseline", "candidate")
    pairs: list[tuple[str, str]] = []
    for role, item in zip(expected_roles, raw, strict=True):
        if not isinstance(item, Mapping) or item.get("role") != role:
            raise ValueError("MCP candidate binding roles are invalid")
        pairs.append(
            (
                str(item.get("import_id") or ""),
                str(item.get("version_identity") or ""),
            )
        )
    selected = _validate_mcp_release_candidates(pairs)
    expected_digest = stable_digest(list(raw))
    if receipt.get("candidate_bindings_digest") != expected_digest:
        raise ValueError("MCP candidate bindings digest does not match")
    return selected


def _qualification_endpoint_binding(
    env: Mapping[str, str],
) -> dict[str, str]:
    api_override = str(env.get("WANDB_BASE_URL") or "").rstrip("/")
    trace_overrides = {
        str(env.get(name) or "").rstrip("/")
        for name in (
            "FUGUE_WEAVE_TRACE_SERVER_URL",
            "WF_TRACE_SERVER_URL",
        )
        if str(env.get(name) or "").strip()
    }
    if api_override and api_override != QUALIFICATION_API_BASE_URL:
        raise RuntimeError(
            "MCP qualification WANDB_BASE_URL disagrees with the locked "
            "public-cloud destination"
        )
    if trace_overrides and trace_overrides != {
        QUALIFICATION_TRACE_BASE_URL
    }:
        raise RuntimeError(
            "MCP qualification trace endpoint disagrees with the locked "
            "public-cloud destination"
        )
    unsigned = {
        "api_base_url": QUALIFICATION_API_BASE_URL,
        "trace_base_url": QUALIFICATION_TRACE_BASE_URL,
    }
    return {
        **unsigned,
        "endpoint_digest": stable_digest(unsigned),
    }

_RUNS: tuple[dict[str, Any], ...] = (
    {
        "id": "maint-r18-01",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "inventory-openclaw",
        "latency_ms": 820,
        "observed_cost_usd": 0.11,
        "deterministic_pass": True,
        "projected_reads": 3,
        "broad_reads": 0,
        "source_returned": 4,
        "source_opened": 4,
    },
    {
        "id": "maint-r18-02",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "coverage-claude",
        "latency_ms": 860,
        "observed_cost_usd": 0.14,
        "deterministic_pass": True,
        "projected_reads": 4,
        "broad_reads": 0,
        "source_returned": 4,
        "source_opened": 4,
    },
    {
        "id": "maint-r18-03",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "regression-openclaw",
        "latency_ms": 900,
        "observed_cost_usd": 0.16,
        "deterministic_pass": True,
        "projected_reads": 2,
        "broad_reads": 1,
        "source_returned": 4,
        "source_opened": 3,
    },
    {
        "id": "maint-r18-04",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "partial-evidence-claude",
        "latency_ms": 940,
        "observed_cost_usd": 0.19,
        "deterministic_pass": False,
        "projected_reads": 2,
        "broad_reads": 1,
        "source_returned": 4,
        "source_opened": 1,
    },
    {
        "id": "maint-r18-05",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "cost-audit-openclaw",
        "latency_ms": 980,
        "observed_cost_usd": 0.36,
        "deterministic_pass": True,
        "projected_reads": 3,
        "broad_reads": 0,
        "source_returned": 4,
        "source_opened": 4,
    },
    {
        "id": "maint-r18-06",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "broad-history-scan-claude",
        "latency_ms": 4200,
        "observed_cost_usd": None,
        "deterministic_pass": False,
        "projected_reads": 0,
        "broad_reads": 4,
        "source_returned": 4,
        "source_opened": 1,
    },
)

_EVALUATION_ROWS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "coverage-inventory",
        "question": "Count only inspected evidence.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
    {
        "case_id": "aligned-regression",
        "question": "Compare aligned evaluation revisions.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
    {
        "case_id": "latency-anomaly",
        "question": "Identify the observed latency outlier.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
    {
        "case_id": "cost-coverage",
        "question": "Do not replace missing observed cost with zero.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
    {
        "case_id": "source-opened",
        "question": "Separate source return from source use.",
        "baseline_pass": False,
        "candidate_pass": True,
    },
    {
        "case_id": "partial-evidence",
        "question": "Return a structured incomplete-evidence result.",
        "baseline_pass": True,
        "candidate_pass": False,
    },
    {
        "case_id": "maintenance-priority",
        "question": "Recommend one evidence-linked maintenance action.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
    {
        "case_id": "release-completeness",
        "question": "Refuse an unsupported release-wide claim.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
)


def qualification_seed(
    *,
    source_project: str = QUALIFICATION_PROJECT,
) -> dict[str, Any]:
    latencies = [int(item["latency_ms"]) for item in _RUNS]
    observed_costs = [
        float(item["observed_cost_usd"])
        for item in _RUNS
        if item["observed_cost_usd"] is not None
    ]
    baseline_passes = {
        str(item["case_id"]): bool(item["baseline_pass"]) for item in _EVALUATION_ROWS
    }
    candidate_passes = {
        str(item["case_id"]): bool(item["candidate_pass"]) for item in _EVALUATION_ROWS
    }
    regressions = sorted(
        case_id
        for case_id, passed in baseline_passes.items()
        if passed and not candidate_passes[case_id]
    )
    return {
        "schema_version": 1,
        "project": source_project,
        "runs": [dict(item) for item in _RUNS],
        "evaluation_rows": [dict(item) for item in _EVALUATION_ROWS],
        "facts": {
            "run_count": len(_RUNS),
            "source_conversation_count": len(_RUNS) * 4,
            "tool_span_count": len(_RUNS) * 8,
            "aligned_evaluation_pairs": len(_EVALUATION_ROWS),
            "evaluation_prediction_rows": len(_EVALUATION_ROWS) * 2,
            "latency_anomaly": {
                "attempt_label": "broad-history-scan-claude",
                "latency_ms": 4200,
                "cohort_median_ms": int(statistics.median(latencies)),
                "ratio": round(4200 / statistics.median(latencies), 4),
            },
            "cost_coverage": {
                "attempts": len(_RUNS),
                "attempts_with_observed_cost": len(observed_costs),
                "total_observed_usd": round(sum(observed_costs), 6),
                "complete": False,
            },
            "reviewed_failure_cohort": {
                "reviewed_examples": 4,
                "current_source_returned": 4,
                "current_source_opened": 1,
                "failure_class": "current-source-returned-mostly-not-opened",
            },
            "regressions": regressions,
            "missing_evidence": "maintainer-r18-partial-evaluation",
        },
    }


def qualification_seed_digest(
    *,
    source_project: str = QUALIFICATION_PROJECT,
) -> str:
    return stable_digest(qualification_seed(source_project=source_project))


def evidence_source_project(value: Mapping[str, Any]) -> str:
    """Return the immutable project containing the hosted task evidence."""

    return str(value.get("source_project") or value.get("project") or "")


def evidence_result_project(value: Mapping[str, Any]) -> str:
    """Return the destination where later experiment results may be written."""

    return str(value.get("result_project") or value.get("project") or "")


def validate_evidence_lock(
    value: Mapping[str, Any],
    *,
    expected_project: str | None = QUALIFICATION_PROJECT,
    expected_source_project: str | None = None,
    expected_result_project: str | None = None,
) -> dict[str, Any]:
    raw = dict(value)
    supplied = str(raw.get("evidence_lock_digest") or "")
    unsigned = dict(raw)
    unsigned["evidence_lock_digest"] = ""
    if len(supplied) != 64 or stable_digest(unsigned) != supplied:
        raise ValueError("evidence lock digest does not match")
    if raw.get("schema_version") != EVIDENCE_LOCK_SCHEMA_VERSION:
        raise ValueError("evidence lock schema is unsupported")
    source_project = evidence_source_project(raw)
    result_project = evidence_result_project(raw)
    _validate_evidence_project_binding(
        raw,
        source_project=source_project,
        result_project=result_project,
        expected_project=expected_project,
        expected_source_project=expected_source_project,
        expected_result_project=expected_result_project,
    )
    if raw.get("seed_digest") != qualification_seed_digest(
        source_project=source_project
    ):
        raise ValueError("evidence lock seed digest does not match")
    expected_facts_digest = stable_digest(
        qualification_seed(source_project=source_project)["facts"]
    )
    if raw.get("facts_digest") != expected_facts_digest:
        raise ValueError("evidence lock facts digest does not match")
    _validate_evidence_lock_counts(raw)
    _validate_evidence_lock_objects(raw, source_project=source_project)
    return raw


def _validate_evidence_project_binding(
    raw: Mapping[str, Any],
    *,
    source_project: str,
    result_project: str,
    expected_project: str | None,
    expected_source_project: str | None,
    expected_result_project: str | None,
) -> None:
    if expected_source_project is None:
        expected_source_project = expected_project
    if expected_result_project is None and expected_project is not None:
        expected_result_project = expected_project
    if (
        expected_source_project is not None
        and source_project != expected_source_project
    ):
        raise ValueError("evidence lock source project does not match")
    if (
        expected_result_project is not None
        and result_project != expected_result_project
    ):
        raise ValueError("evidence lock result project does not match")
    if raw.get("project") != source_project:
        raise ValueError("evidence lock legacy project alias does not match source")


def _validate_evidence_lock_counts(raw: Mapping[str, Any]) -> None:
    counts = raw.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("evidence lock counts are missing")
    for name, expected in _REQUIRED_COUNTS.items():
        if int(counts.get(name) or 0) != expected:
            raise ValueError(f"evidence lock count {name} must equal {expected}")
    if raw.get("source_snapshot_digest"):
        objects = raw.get("objects")
        if not isinstance(objects, Mapping):
            raise ValueError("evidence lock objects are missing")
        derived = _derived_evidence_counts(
            objects.get("runs") or (),
            {
                "dataset": objects.get("dataset"),
                "source_conversations": objects.get("source_conversations") or (),
                "evaluations": objects.get("evaluations") or (),
            },
        )
        if dict(counts) != derived:
            raise ValueError("evidence lock counts do not match remote receipts")


def _validate_evidence_lock_objects(
    raw: Mapping[str, Any],
    *,
    source_project: str,
) -> None:
    objects = raw.get("objects")
    if not isinstance(objects, Mapping):
        raise ValueError("evidence lock objects are missing")
    runs = objects.get("runs")
    conversations = objects.get("source_conversations")
    evaluations = objects.get("evaluations")
    if not isinstance(runs, list) or len(runs) != _REQUIRED_COUNTS["runs"]:
        raise ValueError("evidence lock must contain every W&B Run")
    if (
        not isinstance(conversations, list)
        or len(conversations) != _REQUIRED_COUNTS["source_conversations"]
    ):
        raise ValueError("evidence lock must contain every source conversation")
    if not isinstance(evaluations, list) or len(evaluations) != 2:
        raise ValueError("evidence lock must contain both evaluation revisions")
    dataset = objects.get("dataset")
    _validate_evidence_lock_refs(
        source_project=source_project,
        runs=runs,
        dataset=dataset,
        conversations=conversations,
        evaluations=evaluations,
    )
    snapshot_digest = raw.get("source_snapshot_digest")
    if snapshot_digest:
        _validate_evidence_lock_snapshot(
            raw,
            objects=objects,
            runs=runs,
            dataset=dataset,
            conversations=conversations,
            evaluations=evaluations,
            snapshot_digest=snapshot_digest,
        )


def _validate_evidence_lock_refs(
    *,
    source_project: str,
    runs: Sequence[Any],
    dataset: Any,
    conversations: Sequence[Any],
    evaluations: Sequence[Any],
) -> None:
    source_prefix = f"weave:///{source_project}/"
    if not isinstance(dataset, Mapping) or not str(dataset.get("ref") or "").startswith(
        source_prefix
    ):
        raise ValueError("evidence lock Dataset reference is not immutable")
    if any(
        not isinstance(run, Mapping)
        or not str(run.get("ref") or "").startswith(
            f"wandb-run:///{source_project}/"
        )
        for run in runs
    ):
        raise ValueError("evidence lock W&B Run reference is not immutable")
    if any(
        not isinstance(conversation, Mapping)
        or not str(conversation.get("ref") or "").startswith(source_prefix)
        for conversation in conversations
    ):
        raise ValueError(
            "evidence lock source conversation reference is not immutable"
        )
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise ValueError("evidence lock Evaluation entry is invalid")
        if not str(evaluation.get("ref") or "").startswith(source_prefix):
            raise ValueError("evidence lock Evaluation reference is not immutable")
        if not str(evaluation.get("call_ref") or "").startswith(source_prefix):
            raise ValueError("evidence lock Evaluation call is not immutable")
        call_id = str(evaluation.get("call_id") or "")
        if not call_id or not str(evaluation["call_ref"]).endswith(f"/{call_id}"):
            raise ValueError("evidence lock Evaluation call id does not match its ref")
        if int(evaluation.get("prediction_rows") or 0) < 1:
            raise ValueError("evidence lock Evaluation has no prediction rows")


def _validate_evidence_lock_snapshot(
    raw: Mapping[str, Any],
    *,
    objects: Mapping[str, Any],
    runs: Sequence[Any],
    dataset: Any,
    conversations: Sequence[Any],
    evaluations: Sequence[Any],
    snapshot_digest: Any,
) -> None:
    object_versions = objects.get("weave_object_versions")
    if (
        not isinstance(object_versions, list)
        or len(object_versions) != 13
        or any(
            not isinstance(item, Mapping)
            or item.get("kind") not in {"object", "op"}
            or not str(item.get("object_id") or "")
            or not str(item.get("digest") or "")
            or not str(item.get("content_digest") or "")
            for item in object_versions
        )
        or not _weave_object_versions_complete(object_versions)
        or len(
            {
                (
                    str(item["kind"]),
                    str(item["object_id"]),
                    str(item["digest"]),
                )
                for item in object_versions
            }
        )
        != len(object_versions)
    ):
        raise ValueError(
            "evidence lock must bind every source Weave object version"
        )
    if any(
        not isinstance(evaluation, Mapping)
        or not isinstance(evaluation.get("call_ids"), list)
        or len(evaluation["call_ids"]) != 26
        or evaluation.get("call_id") not in evaluation["call_ids"]
        for evaluation in evaluations
    ):
        raise ValueError(
            "evidence lock must bind every Evaluation call identity"
        )
    if (
        not isinstance(snapshot_digest, str)
        or len(snapshot_digest) != 64
        or stable_digest(objects) != snapshot_digest
    ):
        raise ValueError("evidence lock source snapshot digest does not match")
    if (
        not isinstance(raw.get("source_inventory_digest"), str)
        or len(str(raw["source_inventory_digest"])) != 64
        or not isinstance(raw.get("preparation_id"), str)
        or not raw["preparation_id"]
    ):
        raise ValueError("evidence lock preparation identity is incomplete")
    content_items = [
        *runs,
        dataset,
        *conversations,
        *evaluations,
        *object_versions,
    ]
    if any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("content_digest"), str)
        or len(str(item["content_digest"])) != 64
        for item in content_items
    ):
        raise ValueError("evidence lock remote content digest is missing")


def qualify_locked_mcp_revisions(
    *,
    repo_root: Path,
    evidence_lock: Path,
    env_file: Path,
    output: Path,
    source_project: str | None = None,
    result_project: str | None = None,
    candidates: Sequence[tuple[str, str]] = DEFAULT_MCP_RELEASE_CANDIDATES,
    release_notes_lock: Path = MCP_RELEASE_NOTES_LOCK,
) -> dict[str, Any]:
    """Exercise both exact MCP locks against the immutable hosted evidence.

    This is a bounded infrastructure/mechanism qualification, not an Agent
    outcome study. It deliberately avoids the blind judge and never treats a
    successful tool RPC as proof of task correctness.
    """

    root = repo_root.resolve()
    selected_candidates = _validate_mcp_release_candidates(candidates)
    raw_lock = json.loads(evidence_lock.resolve().read_text(encoding="utf-8"))
    resolved_source_project = source_project or evidence_source_project(raw_lock)
    resolved_result_project = result_project or evidence_result_project(raw_lock)
    lock = validate_evidence_lock(
        raw_lock,
        expected_project=None,
        expected_source_project=resolved_source_project,
        expected_result_project=resolved_result_project,
    )
    selected_release_notes_lock = (
        release_notes_lock
        if release_notes_lock.is_absolute()
        else root / release_notes_lock
    )
    release_notes = validate_release_notes_lock(
        json.loads(
            selected_release_notes_lock.resolve().read_text(encoding="utf-8")
        ),
        expected_commit=_mcp_release_candidate_commit(
            selected_candidates[1][1]
        ),
    )
    env = load_env(env_file.resolve())
    if not str(env.get("WANDB_API_KEY") or "").strip():
        raise RuntimeError("MCP qualification requires WANDB_API_KEY")
    endpoint_binding = _qualification_endpoint_binding(env)
    runtime_env = {
        "WANDB_API_KEY": str(env["WANDB_API_KEY"]),
        "WANDB_BASE_URL": endpoint_binding["api_base_url"],
    }
    observations = asyncio.run(
        _run_mcp_release_observations(
            root,
            lock,
            runtime_env,
            candidates=selected_candidates,
        )
    )
    _validate_mcp_release_observations(observations, selected_candidates)
    receipt = _mcp_release_qualification_receipt(
        lock,
        observations,
        release_notes=release_notes,
        endpoint_binding=endpoint_binding,
        candidates=selected_candidates,
    )
    serialized = json.dumps(receipt, sort_keys=True)
    for secret in (runtime_env["WANDB_API_KEY"],):
        if secret and secret in serialized:
            raise RuntimeError("MCP qualification receipt contains a credential")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output.resolve(), receipt)
    return receipt


async def _run_mcp_release_observations(
    repo_root: Path,
    evidence_lock: Mapping[str, Any],
    runtime_env: Mapping[str, str],
    *,
    candidates: Sequence[tuple[str, str]] = DEFAULT_MCP_RELEASE_CANDIDATES,
) -> list[dict[str, Any]]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise RuntimeError(
            "MCP qualification requires `uv sync --extra research-worker`"
        ) from exc
    from fugue.bench.component_imports import _managed_python_probe_command

    selected_candidates = _validate_mcp_release_candidates(candidates)
    project = evidence_source_project(evidence_lock)
    entity, project_name = project.split("/", 1)
    evaluations = list(evidence_lock["objects"]["evaluations"])
    observations: list[dict[str, Any]] = []
    candidate_import_id = selected_candidates[1][0]
    for import_id, version_identity in selected_candidates:
        path = repo_root / ".fugue" / "imports" / "mcp" / "locks" / f"{import_id}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            raw.get("id") != import_id
            or raw.get("version_identity") != version_identity
        ):
            raise ValueError(f"{import_id} lock does not bind {version_identity}")
        runtime_digest = str(raw.get("runtime_digest") or "")
        if not runtime_digest.startswith("sha256:"):
            raise ValueError(f"{import_id} has no managed runtime digest")
        runtime_source = (
            repo_root
            / ".fugue"
            / "runtime"
            / "mcp"
            / import_id
            / runtime_digest.removeprefix("sha256:")
        )
        if not (runtime_source / "bin" / "server").is_file():
            raise FileNotFoundError(f"{import_id} managed runtime is absent")
        command = _managed_python_probe_command(
            runtime_source,
            runtime_platform=str(raw["runtime_platform"]),
            required_env=tuple(raw["required_env"]),
            fixed_env=tuple(dict(raw["fixed_env"]).items()),
            allowed_hosts=tuple(raw["allowed_hosts"]),
        )
        process_env = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "HOME", "DOCKER_HOST"}
        }
        process_env.update(runtime_env)
        parameters = StdioServerParameters(
            command=command[0],
            args=list(command[1:]),
            env=process_env,
        )
        calls: dict[str, Any] = {}
        async with stdio_client(parameters) as streams:
            async with ClientSession(*streams) as session:
                initialized = await asyncio.wait_for(session.initialize(), timeout=90)
                initialized_tools = await asyncio.wait_for(
                    session.list_tools(),
                    timeout=90,
                )
                calls["count_weave_traces_tool"] = await _call_mcp_json(
                    session,
                    "count_weave_traces_tool",
                    {
                        "entity_name": entity,
                        "project_name": project_name,
                        "filters": {"trace_roots_only": True},
                    },
                )
                calls["probe_project_tool"] = await _call_mcp_json(
                    session,
                    "probe_project_tool",
                    {
                        "entity_name": entity,
                        "project_name": project_name,
                        "sample_runs": 4,
                    },
                )
                calls["count_evaluation_roots_tool"] = await _call_mcp_json(
                    session,
                    "count_weave_traces_tool",
                    {
                        "entity_name": entity,
                        "project_name": project_name,
                        "filters": {
                            "trace_roots_only": True,
                            "op_name_contains": "Evaluation.evaluate",
                        },
                    },
                )
                evaluation_root_count = int(
                    _successful_mcp_value(calls["count_evaluation_roots_tool"]).get(
                        "root_traces_count"
                    )
                    or 0
                )
                summary_limit = min(
                    max(evaluation_root_count, len(evaluations)),
                    50,
                )
                calls["summarize_evaluation_tool"] = await _call_mcp_json(
                    session,
                    "summarize_evaluation_tool",
                    {
                        "entity_name": entity,
                        "project_name": project_name,
                        "max_evals": summary_limit,
                        "include_per_task": False,
                    },
                )
                child_ops: dict[str, Any] = {}
                for evaluation in evaluations:
                    call_id = str(evaluation["call_id"])
                    child_ops[call_id] = await _call_mcp_json(
                        session,
                        "query_weave_traces_tool",
                        {
                            "entity_name": entity,
                            "project_name": project_name,
                            "filters": {"parent_ids": [call_id]},
                            "columns": [
                                "id",
                                "op_name",
                                "parent_id",
                                "trace_id",
                                "started_at",
                            ],
                            "metadata_only": True,
                            "include_costs": False,
                            "include_feedback": False,
                            "limit": 20,
                            "sort_by": "started_at",
                            "sort_direction": "asc",
                        },
                        timeout=150,
                    )
        profile_probes: dict[str, Any] = {}
        if import_id == candidate_import_id:
            profile_probes["package_default"] = await _probe_tool_profile(
                runtime_source=runtime_source,
                lock=raw,
                runtime_env=runtime_env,
                overrides={},
                unset_fixed_env=("WANDB_MCP_READ_ONLY",),
                use_package_entrypoint=True,
            )
            profile_probes["read_only"] = await _probe_tool_profile(
                runtime_source=runtime_source,
                lock=raw,
                runtime_env=runtime_env,
                overrides={"WANDB_MCP_READ_ONLY": "true"},
            )
            profile_probes["raw_graphql"] = await _probe_tool_profile(
                runtime_source=runtime_source,
                lock=raw,
                runtime_env=runtime_env,
                overrides={"WANDB_MCP_ENABLE_RAW_GRAPHQL": "true"},
                mutation_probe=True,
            )
        server = getattr(initialized, "serverInfo", None)
        initialized_tool_schemas = _initialized_tool_schema_contract(
            initialized_tools.tools
        )
        locked_tool_schemas = _locked_tool_schema_contract(raw.get("tool_manifest"))
        observations.append(
            {
                "id": import_id,
                "version_identity": version_identity,
                "runtime_digest": runtime_digest,
                "tool_manifest_digest": str(raw["tool_manifest_digest"]),
                "server": {
                    "name": str(getattr(server, "name", "") or ""),
                    "version": str(getattr(server, "version", "") or ""),
                },
                "initialized_tools": sorted(
                    str(tool.name) for tool in initialized_tools.tools
                ),
                "initialized_tool_schema_digest": stable_digest(
                    initialized_tool_schemas
                ),
                "locked_tools": sorted(
                    str(item.get("name") or "")
                    for item in raw.get("tool_manifest", [])
                    if isinstance(item, Mapping) and item.get("name")
                ),
                "locked_tool_schema_digest": stable_digest(locked_tool_schemas),
                "initialized_schema_matches_lock": (
                    initialized_tool_schemas == locked_tool_schemas
                ),
                "release_capabilities": _release_capabilities(raw.get("tool_manifest")),
                "calls": calls,
                "evaluation_child_ops": child_ops,
                "profile_probes": profile_probes,
            }
        )
    return observations


async def _probe_tool_profile(
    *,
    runtime_source: Path,
    lock: Mapping[str, Any],
    runtime_env: Mapping[str, str],
    overrides: Mapping[str, str],
    unset_fixed_env: tuple[str, ...] = (),
    use_package_entrypoint: bool = False,
    mutation_probe: bool = False,
) -> dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from fugue.bench.component_imports import _managed_python_probe_command

    fixed_env = _profile_fixed_env(
        lock,
        overrides=overrides,
        unset_fixed_env=unset_fixed_env,
    )
    command = _managed_python_probe_command(
        runtime_source,
        runtime_platform=str(lock["runtime_platform"]),
        required_env=tuple(lock["required_env"]),
        fixed_env=tuple(sorted(fixed_env.items())),
        allowed_hosts=tuple(lock["allowed_hosts"]),
    )
    if use_package_entrypoint:
        # The managed wrapper deliberately bakes in the experiment's reviewed
        # fixed environment. Package-default conformance must bypass only that
        # wrapper while executing the exact same locked package tree.
        command = (
            *command[:-1],
            "-c",
            (
                "import sys;"
                "sys.path.insert(0, '/fugue-component/site');"
                "from wandb_mcp_server.server import cli;"
                "raise SystemExit(cli())"
            ),
        )
    process_env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "DOCKER_HOST"}
    }
    process_env.update(runtime_env)
    parameters = StdioServerParameters(
        command=command[0],
        args=list(command[1:]),
        env=process_env,
    )
    mutation: dict[str, Any] | None = None
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as session:
            await asyncio.wait_for(session.initialize(), timeout=90)
            initialized_tools = await asyncio.wait_for(
                session.list_tools(),
                timeout=90,
            )
            names = sorted(str(tool.name) for tool in initialized_tools.tools)
            tool_schemas = _initialized_tool_schema_contract(initialized_tools.tools)
            if mutation_probe:
                mutation = await _call_mcp_json(
                    session,
                    "query_wandb_graphql_tool",
                    {"query": ("mutation FugueQualificationMutation { __typename }")},
                )
    return {
        "overrides": dict(sorted(overrides.items())),
        "unset_fixed_env": sorted(unset_fixed_env),
        "entrypoint": "package" if use_package_entrypoint else "locked_wrapper",
        "initialized_tools": names,
        "tool_schemas": tool_schemas,
        "tool_schema_digest": stable_digest(tool_schemas),
        "tool_manifest_digest": stable_digest(names),
        "mutation_probe": mutation,
    }


def _profile_fixed_env(
    lock: Mapping[str, Any],
    *,
    overrides: Mapping[str, str],
    unset_fixed_env: tuple[str, ...],
) -> dict[str, str]:
    fixed_env = {str(key): str(value) for key, value in dict(lock["fixed_env"]).items()}
    for name in unset_fixed_env:
        fixed_env.pop(name, None)
    fixed_env.update({str(key): str(value) for key, value in overrides.items()})
    return fixed_env


def _initialized_tool_schema_contract(tools: Sequence[Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "name": str(getattr(tool, "name", "") or ""),
                "input_schema": json.loads(
                    json.dumps(
                        getattr(tool, "inputSchema", {}) or {},
                        sort_keys=True,
                    )
                ),
            }
            for tool in tools
            if str(getattr(tool, "name", "") or "")
        ),
        key=lambda item: item["name"],
    )


def _locked_tool_schema_contract(raw: Any) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "name": str(item.get("name") or ""),
                "input_schema": json.loads(
                    json.dumps(item.get("input_schema") or {}, sort_keys=True)
                ),
            }
            for item in (raw or ())
            if isinstance(item, Mapping) and str(item.get("name") or "")
        ),
        key=lambda item: item["name"],
    )


async def _call_mcp_json(
    session: Any,
    name: str,
    arguments: Mapping[str, Any],
    *,
    timeout: float = 90,
) -> dict[str, Any]:
    try:
        result = await asyncio.wait_for(
            session.call_tool(name, dict(arguments)),
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    text = "\n".join(
        str(getattr(item, "text", "") or "")
        for item in result.content
        if getattr(item, "type", None) == "text"
    )
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        value = {"message": text[:1_000]}
    application_error = (
        str(value.get("error") or "") if isinstance(value, Mapping) else ""
    )
    return {
        "ok": not bool(result.isError) and not bool(application_error),
        "protocol_error": bool(result.isError),
        "value": value,
    }


def _mcp_release_qualification_receipt(
    evidence_lock: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    release_notes: Mapping[str, Any] | None = None,
    endpoint_binding: Mapping[str, str] | None = None,
    candidates: Sequence[tuple[str, str]] = DEFAULT_MCP_RELEASE_CANDIDATES,
) -> dict[str, Any]:
    selected_candidates = _validate_mcp_release_candidates(candidates)
    candidate_bindings = _mcp_release_candidate_bindings(selected_candidates)
    expected_runs = int(evidence_lock["counts"]["runs"])
    expected_rows = {
        str(item["call_id"]): int(item["prediction_rows"])
        for item in evidence_lock["objects"]["evaluations"]
    }
    candidates: list[dict[str, Any]] = []
    for raw in observations:
        calls = dict(raw["calls"])
        count = _successful_mcp_value(calls.get("count_weave_traces_tool"))
        probe = _successful_mcp_value(calls.get("probe_project_tool"))
        summary = _successful_mcp_value(calls.get("summarize_evaluation_tool"))
        evaluation_roots = _successful_mcp_value(
            calls.get("count_evaluation_roots_tool")
        )
        summaries = {
            str(item.get("eval_id") or ""): item
            for item in (summary.get("evaluations") or [])
            if isinstance(item, Mapping)
        }
        reconciliations = []
        for call_id, expected in expected_rows.items():
            child = _successful_mcp_value(
                dict(raw["evaluation_child_ops"]).get(call_id)
            )
            metadata = (
                dict(child.get("metadata") or {}) if isinstance(child, Mapping) else {}
            )
            op_distribution = dict(metadata.get("op_distribution") or {})
            observed_predictions = int(
                op_distribution.get("Evaluation.predict_and_score") or 0
            )
            observed_summaries = int(op_distribution.get("Evaluation.summarize") or 0)
            reported = summaries.get(call_id, {})
            reported_predictions = (
                int(reported.get("total_predictions") or 0)
                if isinstance(reported, Mapping)
                else 0
            )
            reconciliations.append(
                {
                    "evaluation_call_id": call_id,
                    "locked_prediction_rows": expected,
                    "observed_predict_and_score_children": observed_predictions,
                    "observed_summarize_children": observed_summaries,
                    "tool_reported_total_predictions": reported_predictions,
                    "prediction_rows_reconciled": (
                        expected == observed_predictions == reported_predictions
                    ),
                    "trace_children_reconciled": expected == observed_predictions,
                }
            )
        tool_calls_ok = all(
            bool(dict(calls.get(name) or {}).get("ok"))
            for name in (
                "count_weave_traces_tool",
                "probe_project_tool",
                "summarize_evaluation_tool",
                "count_evaluation_roots_tool",
            )
        )
        child_queries_ok = all(
            bool(dict(item or {}).get("ok"))
            for item in dict(raw["evaluation_child_ops"]).values()
        )
        candidate = {
            "id": raw["id"],
            "version_identity": raw["version_identity"],
            "runtime_digest": raw["runtime_digest"],
            "tool_manifest_digest": raw["tool_manifest_digest"],
            "server": dict(raw["server"]),
            "initialized_tools": list(raw.get("initialized_tools") or []),
            "locked_tools": list(raw.get("locked_tools") or []),
            "initialized_tool_schema_digest": str(
                raw.get("initialized_tool_schema_digest") or ""
            ),
            "locked_tool_schema_digest": str(
                raw.get("locked_tool_schema_digest") or ""
            ),
            "initialized_manifest_matches_lock": (
                list(raw.get("initialized_tools") or [])
                == list(raw.get("locked_tools") or [])
                and raw.get("initialized_schema_matches_lock") is True
            ),
            "initialized_schema_matches_lock": (
                raw.get("initialized_schema_matches_lock") is True
            ),
            "release_capabilities": dict(raw.get("release_capabilities") or {}),
            "tool_calls_ok": tool_calls_ok,
            "child_queries_ok": child_queries_ok,
            "root_trace_count": count.get("root_traces_count"),
            "total_trace_count": count.get("total_count"),
            "project_run_count": probe.get("run_count"),
            "project_state_counts": probe.get("state_counts"),
            "project_probe_matches_lock": probe.get("run_count") == expected_runs,
            "evaluation_root_count": evaluation_roots.get("root_traces_count"),
            "summary_project_exhaustive": summary.get("project_exhaustive"),
            "evaluation_reconciliation": reconciliations,
            "profile_probes": dict(raw.get("profile_probes") or {}),
            "errors": {
                name: _mcp_call_error(value)
                for name, value in calls.items()
                if not bool(dict(value or {}).get("ok"))
            },
        }
        candidates.append(candidate)
    by_id = {str(item["id"]): item for item in candidates}
    baseline = by_id.get(selected_candidates[0][0], {})
    candidate = by_id.get(selected_candidates[1][0], {})
    baseline_reconciliations = list(baseline.get("evaluation_reconciliation", []))
    baseline_reconciled = bool(baseline_reconciliations) and all(
        item["prediction_rows_reconciled"] for item in baseline_reconciliations
    )
    candidate_reconciliations = list(candidate.get("evaluation_reconciliation", []))
    candidate_reconciled = bool(candidate_reconciliations) and all(
        item["prediction_rows_reconciled"] for item in candidate_reconciliations
    )
    release_notes_lock = validate_release_notes_lock(
        release_notes or _default_release_notes_lock()
    )
    release_note_classification = _release_note_classification(
        baseline,
        candidate,
        locked_behaviors=tuple(release_notes_lock["behaviors"]),
    )
    infrastructure_conformance = _infrastructure_conformance(candidate)
    classified_behaviors = {
        str(item["release_note"]) for item in release_note_classification
    }
    if classified_behaviors != set(release_notes_lock["behaviors"]):
        raise ValueError(
            "release-note qualification does not classify every locked behavior"
        )
    unsigned = {
        "schema_version": 1,
        "kind": "mcp-release-mechanism-qualification",
        # `project` remains the source alias for backward receipt readers.
        "project": evidence_source_project(evidence_lock),
        "source_project": evidence_source_project(evidence_lock),
        "result_project": evidence_result_project(evidence_lock),
        "evidence_lock_digest": evidence_lock["evidence_lock_digest"],
        "endpoint_binding": dict(
            endpoint_binding or _qualification_endpoint_binding({})
        ),
        "release_notes_lock": release_notes_lock,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "candidate_bindings": candidate_bindings,
        "candidate_bindings_digest": stable_digest(candidate_bindings),
        "candidates": candidates,
        "findings": {
            "baseline_reads_hosted_evidence": bool(baseline.get("tool_calls_ok")),
            "candidate_reads_hosted_evidence": bool(candidate.get("tool_calls_ok")),
            "baseline_manifest_matches_lock": bool(
                baseline.get("initialized_manifest_matches_lock")
            ),
            "candidate_manifest_matches_lock": bool(
                candidate.get("initialized_manifest_matches_lock")
            ),
            "candidate_project_probe_matches_lock": bool(
                candidate.get("project_probe_matches_lock")
            ),
            "baseline_evaluation_rows_reconciled": baseline_reconciled,
            "candidate_evaluation_rows_reconciled": candidate_reconciled,
        },
        "release_note_classification": release_note_classification,
        "infrastructure_conformance": infrastructure_conformance,
        "recommendation": (
            "Use this receipt only to establish exact MCP/runtime behavior. "
            "Resolve every unavailable infrastructure gate, then run the "
            "aligned Agent blocker canary before making any package-release "
            "recommendation."
        ),
        "claim_scope": (
            "Infrastructure and MCP mechanism evidence only; no Agent task, judge, "
            "or whole-release outcome claim."
        ),
        "whole_release_claim_eligible": False,
        "receipt_digest": "",
    }
    return {
        **unsigned,
        "receipt_digest": stable_digest(unsigned),
    }


def _infrastructure_conformance(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    experiment_manifest = (
        "passed"
        if candidate.get("initialized_manifest_matches_lock") is True
        else "failed"
    )
    experiment_tools = set(candidate.get("initialized_tools") or ())
    profiles = dict(candidate.get("profile_probes") or {})
    package_default = dict(profiles.get("package_default") or {})
    read_only = dict(profiles.get("read_only") or {})
    raw_graphql = dict(profiles.get("raw_graphql") or {})
    package_default_tools = set(package_default.get("initialized_tools") or ())
    read_only_tools = set(read_only.get("initialized_tools") or ())
    raw_graphql_tools = set(raw_graphql.get("initialized_tools") or ())
    package_default_schemas = _profile_tool_schema_map(package_default)
    read_only_schemas = _profile_tool_schema_map(read_only)
    raw_graphql_schemas = _profile_tool_schema_map(raw_graphql)
    locked_schema_digest = str(candidate.get("locked_tool_schema_digest") or "")
    read_only_schema_digest = str(read_only.get("tool_schema_digest") or "")
    write_tools = {"create_wandb_report_tool", "log_analysis_to_wandb"}
    package_default_status = (
        "passed"
        if (
            package_default_tools
            and package_default_tools == experiment_tools | write_tools
            and package_default_tools & write_tools == write_tools
            and {name: package_default_schemas.get(name) for name in experiment_tools}
            == {name: read_only_schemas.get(name) for name in experiment_tools}
            and all(
                _valid_tool_input_schema(package_default_schemas.get(name))
                for name in write_tools
            )
        )
        else "failed"
        if package_default_tools
        else "unavailable"
    )
    read_only_status = (
        "passed"
        if (
            read_only_tools
            and read_only_tools == experiment_tools
            and not read_only_tools & write_tools
            and locked_schema_digest
            and read_only_schema_digest == locked_schema_digest
        )
        else "failed"
        if read_only_tools
        else "unavailable"
    )
    raw_graphql_status = (
        "passed"
        if (
            raw_graphql_tools
            and raw_graphql_tools == experiment_tools | {"query_wandb_graphql_tool"}
            and {name: raw_graphql_schemas.get(name) for name in experiment_tools}
            == {name: read_only_schemas.get(name) for name in experiment_tools}
            and _valid_tool_input_schema(
                raw_graphql_schemas.get("query_wandb_graphql_tool")
            )
        )
        else "failed"
        if raw_graphql_tools
        else "unavailable"
    )
    mutation_value = dict(
        dict(raw_graphql.get("mutation_probe") or {}).get("value") or {}
    )
    mutation_errors = mutation_value.get("errors")
    mutation_rejected = (
        isinstance(mutation_errors, list)
        and len(mutation_errors) == 1
        and isinstance(mutation_errors[0], Mapping)
        and mutation_errors[0].get("error") == "read_only_violation"
        and "mutation" in list(mutation_errors[0].get("operation_types") or ())
    )
    mutation_status = (
        "passed"
        if mutation_rejected
        else "failed"
        if raw_graphql.get("mutation_probe") is not None
        else "unavailable"
    )
    gates = [
        {
            "id": "experiment-read-only-tool-manifest",
            "status": experiment_manifest,
            "evidence": (
                "initialized experiment MCP manifest versus its exact read-only lock"
            ),
        },
        {
            "id": "default-tool-manifest",
            "status": package_default_status,
            "evidence": (
                "exact candidate runtime initialized with package-default "
                "write-tool registration"
            ),
        },
        {
            "id": "read-only-tool-manifest",
            "status": read_only_status,
            "evidence": (
                "exact candidate runtime initialized with WANDB_MCP_READ_ONLY=true"
            ),
        },
        {
            "id": "raw-graphql-opt-in-manifest",
            "status": raw_graphql_status,
            "evidence": (
                "exact candidate runtime initialized with "
                "WANDB_MCP_ENABLE_RAW_GRAPHQL=true"
            ),
        },
        {
            "id": "graphql-mutation-rejection",
            "status": mutation_status,
            "evidence": (
                "query_wandb_graphql_tool rejected a mutation before transport"
            ),
        },
        {
            "id": "weighted-admission-server-busy",
            "status": "unavailable",
            "evidence": "requires a bounded admission-concurrency probe",
        },
        {
            "id": "timeout-and-bounded-response",
            "status": "unavailable",
            "evidence": "requires explicit SDK and tool timeout probes",
        },
        {
            "id": "telemetry-attribution-and-privacy",
            "status": "unavailable",
            "evidence": "requires trace attribution and credential-redaction scan",
        },
        {
            "id": "fresh-wheel-python-3-11",
            "status": "unavailable",
            "evidence": "requires clean Python 3.11 wheel install",
        },
        {
            "id": "fresh-wheel-python-3-12",
            "status": "unavailable",
            "evidence": "requires clean Python 3.12 wheel install",
        },
        {
            "id": "sdk-latest-ci-security",
            "status": "unavailable",
            "evidence": "requires upstream CI and security receipts",
        },
        {
            "id": "installed-wheel-stdio-protocol",
            "status": "unavailable",
            "evidence": (
                "requires installed-wheel MCP sessions proving stdout contains "
                "only JSON-RPC across supported client identities"
            ),
        },
        {
            "id": "serverless-runtime-digest",
            "status": "unavailable",
            "evidence": "requires a scanned public image and runtime lock",
        },
        {
            "id": "serverless-deletion-and-zero-orphans",
            "status": "unavailable",
            "evidence": "requires deletion receipts and an orphan query",
        },
    ]
    failed = [item["id"] for item in gates if item["status"] == "failed"]
    unavailable = [item["id"] for item in gates if item["status"] == "unavailable"]
    return {
        "complete": not failed and not unavailable,
        "failed": failed,
        "unavailable": unavailable,
        "gates": gates,
        "claim_scope": (
            "Infrastructure conformance only; these gates are not Agent task scores."
        ),
    }


def _profile_tool_schema_map(
    profile: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("name") or ""): dict(item.get("input_schema") or {})
        for item in profile.get("tool_schemas") or ()
        if isinstance(item, Mapping)
        and str(item.get("name") or "")
        and isinstance(item.get("input_schema"), Mapping)
    }


def _valid_tool_input_schema(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("type") == "object"


def validate_release_notes_lock(
    raw: Mapping[str, Any],
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "repository",
        "commit",
        "path",
        "raw_url",
        "sha256",
        "bytes",
        "status",
        "behaviors",
    }
    if set(raw) != expected:
        raise ValueError("release-notes lock fields do not match V1")
    behaviors = raw.get("behaviors")
    commit = str(raw.get("commit") or "")
    allowed_source = _MCP_RELEASE_NOTES_SOURCES.get(commit)
    if (
        raw.get("schema_version") != 1
        or raw.get("repository") != "wandb/wandb-mcp-server"
        or raw.get("path") != "docs/releases/v0.4.0.md"
        or allowed_source is None
        or raw.get("sha256") != allowed_source[0]
        or raw.get("bytes") != allowed_source[1]
        or raw.get("status") != "release_candidate"
        or not isinstance(behaviors, list)
        or not behaviors
        or len(behaviors) != len(set(behaviors))
        or any(not isinstance(item, str) or not item for item in behaviors)
    ):
        raise ValueError("release-notes lock does not bind the exact 0.4 RC source")
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(
            "release-notes lock does not bind the selected candidate commit"
        )
    expected_url = (
        "https://raw.githubusercontent.com/wandb/wandb-mcp-server/"
        f"{commit}/docs/releases/v0.4.0.md"
    )
    if raw.get("raw_url") != expected_url:
        raise ValueError("release-notes lock URL does not match its exact source")
    return dict(raw)


def _default_release_notes_lock() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repository": "wandb/wandb-mcp-server",
        "commit": _MCP_RELEASE_NOTES_COMMIT,
        "path": "docs/releases/v0.4.0.md",
        "raw_url": (
            "https://raw.githubusercontent.com/wandb/wandb-mcp-server/"
            f"{_MCP_RELEASE_NOTES_COMMIT}/docs/releases/v0.4.0.md"
        ),
        "sha256": _MCP_RELEASE_NOTES_SHA256,
        "bytes": _MCP_RELEASE_NOTES_BYTES,
        "status": "release_candidate",
        "behaviors": [
            str(item["release_note"]) for item in _release_note_classification({}, {})
        ],
    }


def _release_capabilities(raw_manifest: Any) -> dict[str, bool]:
    tools = {
        str(item.get("name") or ""): item
        for item in raw_manifest or []
        if isinstance(item, Mapping)
    }
    query = tools.get("query_wandb_tool") or {}
    query_schema = query.get("input_schema") or {}
    query_properties = (
        query_schema.get("properties") if isinstance(query_schema, Mapping) else {}
    )
    query_properties = query_properties if isinstance(query_properties, Mapping) else {}
    history = tools.get("get_run_history_tool") or {}
    history_schema = history.get("input_schema") or {}
    history_properties = (
        history_schema.get("properties") if isinstance(history_schema, Mapping) else {}
    )
    history_properties = (
        history_properties if isinstance(history_properties, Mapping) else {}
    )
    names = set(tools)
    return {
        "structured_query": "resource" in query_properties,
        "exact_count_mode": "response_mode" in query_properties,
        "projected_summary_keys": "summary_keys" in query_properties,
        "projected_config_keys": "config_keys" in query_properties,
        "cursor_continuation": "cursor" in query_properties,
        # Main exposes caller-authored GraphQL through query_wandb_tool itself.
        # Looking only for the separately registered 0.4 escape-hatch tool
        # therefore misclassifies the safer default as pre-existing.
        "caller_raw_graphql_default": "query" in query_properties,
        "bounded_history_keys": "keys" in history_properties,
        "bounded_history_range": any(
            name in history_properties
            for name in ("min_step", "max_step", "min_x", "max_x")
        ),
        # Basic sampling and step ranges existed on main. The 0.4 behavior is
        # the combination of a caller-selected axis and exact target lookup.
        "history_custom_axis": "x_axis" in history_properties,
        "history_target_lookup": "target_x" in history_properties,
        "raw_graphql_registered_by_default": ("query_wandb_graphql_tool" in names),
        "write_tools_registered_by_default": bool(
            {"create_wandb_report_tool", "log_analysis_to_wandb"} & names
        ),
    }


def _release_note_classification(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    locked_behaviors: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    baseline_capabilities = dict(baseline.get("release_capabilities") or {})
    candidate_capabilities = dict(candidate.get("release_capabilities") or {})
    contracts = (
        (
            "sdk-first-structured-query",
            "structured_query",
            "agent-and-mechanism",
        ),
        ("exact-count-mode", "exact_count_mode", "agent-and-mechanism"),
        (
            "projected-summary-and-config",
            "projected_summary_keys",
            "agent-and-mechanism",
        ),
        (
            "bounded-history",
            "history_target_lookup",
            "agent-and-mechanism",
        ),
        (
            "raw-graphql-disabled-by-default",
            "caller_raw_graphql_default",
            "infrastructure",
        ),
    )
    classified: list[dict[str, str]] = []
    for release_note, capability, evidence_kind in contracts:
        baseline_value = bool(baseline_capabilities.get(capability))
        candidate_value = bool(candidate_capabilities.get(capability))
        if capability == "caller_raw_graphql_default":
            baseline_value = not baseline_value
            candidate_value = not candidate_value
        status = (
            "observed_branch_delta"
            if candidate_value and not baseline_value
            else "already_present_on_main"
            if candidate_value and baseline_value
            else "unqualified"
        )
        classified.append(
            {
                "release_note": release_note,
                "status": status,
                "evidence_kind": evidence_kind,
            }
        )
    classified.extend(
        (
            {
                "release_note": "evaluation-prediction-reconciliation",
                "status": (
                    "observed_branch_delta"
                    if candidate.get("evaluation_reconciliation")
                    and all(
                        bool(item.get("prediction_rows_reconciled"))
                        for item in candidate.get("evaluation_reconciliation", ())
                        if isinstance(item, Mapping)
                    )
                    else "mechanism_probe_required"
                ),
                "evidence_kind": "agent-and-mechanism",
            },
            {
                "release_note": "explicit-coverage-metadata",
                "status": "agent_canary_required",
                "evidence_kind": "agent-and-mechanism",
            },
            {
                "release_note": "selective-server-side-fields",
                "status": "agent_canary_required",
                "evidence_kind": "agent-and-mechanism",
            },
            {
                "release_note": "weighted-admission-server-busy",
                "status": "infrastructure_only_not_live_induced",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "shared-dedicated-local-workload-profiles",
                "status": "infrastructure_only_not_live_qualified",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "tool-and-sdk-timeout-boundaries",
                "status": "infrastructure_only_not_live_qualified",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "dedicated-internal-api-routing",
                "status": "unqualified_no_dedicated_environment",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "public-links-hide-internal-routing",
                "status": "unqualified_no_dedicated_environment",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "raw-graphql-query-only-escape-hatch",
                "status": "infrastructure_only_requires_flagged_manifest",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "read-only-write-tool-omission",
                "status": "infrastructure_only_requires_flagged_manifest",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "telemetry-attribution-and-privacy",
                "status": "mechanism_only_secret_scan",
                "evidence_kind": "mechanism",
            },
            {
                "release_note": "security-sensitive-dependency-updates",
                "status": "source_and_lock_review_required",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "ci-security-and-wheel-hardening",
                "status": "upstream_ci_only_not_runtime_qualified",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "structured-query-breaking-change",
                "status": "observed_branch_delta",
                "evidence_kind": "mechanism",
            },
            {
                "release_note": "operator-setting-defaults",
                "status": "infrastructure_only_not_live_qualified",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "managed-staging-and-helm-validation",
                "status": "unqualified_separate_release_gate",
                "evidence_kind": "infrastructure",
            },
            {
                "release_note": "stdio-protocol-safety",
                "status": "upstream_ci_only_not_runtime_qualified",
                "evidence_kind": "infrastructure",
            },
        )
    )
    if locked_behaviors is not None:
        baseline_cursor = bool(
            baseline_capabilities.get("cursor_continuation")
        )
        candidate_cursor = bool(
            candidate_capabilities.get("cursor_continuation")
        )
        additional = (
            {
                "release_note": "cursor-continuation-pagination",
                "status": (
                    "observed_branch_delta"
                    if candidate_cursor and not baseline_cursor
                    else "already_present_on_main"
                    if candidate_cursor and baseline_cursor
                    else "unqualified"
                ),
                "evidence_kind": "agent-and-mechanism",
            },
            {
                "release_note": "exact-report-name-or-display-title",
                "status": "mechanism_probe_required",
                "evidence_kind": "mechanism",
            },
            {
                "release_note": "runtime-safety-boundaries",
                "status": "infrastructure_only_not_live_qualified",
                "evidence_kind": "infrastructure",
            },
        )
        by_release_note = {
            str(item["release_note"]): item for item in (*classified, *additional)
        }
        missing = [
            behavior
            for behavior in locked_behaviors
            if behavior not in by_release_note
        ]
        if missing:
            raise ValueError(
                "release-note qualification does not classify: "
                + ", ".join(missing)
            )
        return [by_release_note[behavior] for behavior in locked_behaviors]
    return classified


def release_note_coverage_v3(
    receipt: Mapping[str, Any],
    *,
    task_ids: Sequence[str],
    dimension_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Map every locked 0.4 behavior to bounded V3 qualification evidence."""

    available_tasks = frozenset(str(item) for item in task_ids)
    available_dimensions = frozenset(str(item) for item in dimension_ids)
    outcome_dimension_suffixes = (
        "answer_correct",
        "actual_query_scope",
        "reported_project_identity",
        "bounded_evidence",
        "evidence_honesty",
    )
    dimension_prefixes = {
        dimension.rsplit(".", 1)[0]
        for dimension in available_dimensions
        if "." in dimension
    }
    complete_dimension_prefixes = sorted(
        prefix
        for prefix in dimension_prefixes
        if all(
            f"{prefix}.{suffix}" in available_dimensions
            for suffix in outcome_dimension_suffixes
        )
    )
    if len(complete_dimension_prefixes) != 1:
        raise ValueError(
            "release-note coverage requires exactly one complete deterministic "
            "outcome evaluator"
        )
    outcome_dimensions = tuple(
        f"{complete_dimension_prefixes[0]}.{suffix}"
        for suffix in outcome_dimension_suffixes
    )
    task_map = {
        "sdk-first-structured-query": (
            "maintainer-source-inventory",
            "maintainer-project-health",
            "run-inventory-projection",
        ),
        "exact-count-mode": (
            "maintainer-source-inventory",
            "maintainer-project-health",
            "run-inventory-projection",
        ),
        "projected-summary-and-config": (
            "maintainer-source-inventory",
            "maintainer-project-health",
            "run-inventory-projection",
        ),
        "cursor-continuation-pagination": (
            "filtered-failure-triage",
        ),
        "explicit-coverage-metadata": (
            "maintainer-source-inventory",
            "maintainer-project-health",
            "run-inventory-projection",
        ),
        "selective-server-side-fields": (
            "maintainer-source-inventory",
            "maintainer-project-health",
            "run-inventory-projection",
            "filtered-failure-triage",
            "selective-run-comparison",
            "missing-cost-honesty",
        ),
        "bounded-history": (
            "maintainer-run-triage",
            "maintainer-history-hotspot",
            "maintainer-project-health",
            "exact-history-target",
            "missing-cost-honesty",
        ),
        "evaluation-prediction-reconciliation": (
            "maintainer-evaluation-trace-topology",
            "maintainer-evaluation-reconciliation",
            "evaluation-summary-accuracy",
        ),
        "structured-query-breaking-change": (
            "maintainer-source-inventory",
            "maintainer-project-health",
            "run-inventory-projection",
        ),
    }
    dimension_map = {
        release_note: outcome_dimensions for release_note in task_map
    }
    infrastructure_map = {
        "raw-graphql-disabled-by-default": ("default-tool-manifest",),
        "raw-graphql-query-only-escape-hatch": (
            "raw-graphql-opt-in-manifest",
            "graphql-mutation-rejection",
        ),
        "read-only-write-tool-omission": ("read-only-tool-manifest",),
        "weighted-admission-server-busy": (
            "weighted-admission-server-busy",
        ),
        "tool-and-sdk-timeout-boundaries": (
            "timeout-and-bounded-response",
        ),
        "telemetry-attribution-and-privacy": (
            "telemetry-attribution-and-privacy",
        ),
        "security-sensitive-dependency-updates": (
            "ci-dependency-source-security",
        ),
        "ci-security-and-wheel-hardening": (
            "fresh-wheel-python-3-11",
            "fresh-wheel-python-3-12",
            "wandb-latest",
            "ci-dependency-source-security",
        ),
        "stdio-protocol-safety": ("installed-wheel-stdio-protocol",),
        "operator-setting-defaults": ("operator-setting-defaults",),
        "runtime-safety-boundaries": (
            "timeout-and-bounded-response",
            "telemetry-attribution-and-privacy",
            "installed-wheel-stdio-protocol",
        ),
    }
    not_applicable = {
        "shared-dedicated-local-workload-profiles",
        "dedicated-internal-api-routing",
        "public-links-hide-internal-routing",
        "managed-staging-and-helm-validation",
    }
    source_rows = receipt.get("release_note_classification")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("mechanism receipt has no release-note classification")
    locked_notes = receipt.get("release_notes_lock")
    locked_behaviors = (
        locked_notes.get("behaviors")
        if isinstance(locked_notes, Mapping)
        else None
    )
    if not isinstance(locked_behaviors, list) or any(
        not isinstance(item, str) or not item for item in locked_behaviors
    ):
        raise ValueError("mechanism receipt has no locked release-note behaviors")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in source_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("mechanism release-note classification is invalid")
        release_note = str(raw.get("release_note") or "")
        if not release_note or release_note in seen:
            raise ValueError(
                "mechanism release-note behaviors must be non-empty and unique"
            )
        seen.add(release_note)
        source_status = str(raw.get("status") or "")
        mapped_tasks = tuple(
            task_id
            for task_id in task_map.get(release_note, ())
            if task_id in available_tasks
        )
        gates = infrastructure_map.get(release_note, ())
        mapped_dimensions = dimension_map.get(release_note, ())
        unknown_dimensions = sorted(
            set(mapped_dimensions) - available_dimensions
        )
        if unknown_dimensions:
            raise ValueError(
                "release-note coverage references unknown scorer dimensions: "
                + ", ".join(unknown_dimensions)
            )
        if release_note in not_applicable:
            status = "not_applicable"
            rationale = (
                "This behavior belongs to managed-service or deployment "
                "qualification, not the Python package decision."
            )
        elif source_status == "already_present_on_main":
            status = "already_on_main"
            rationale = (
                "The exact locked manifests show this behavior on both arms."
            )
        elif source_status == "observed_branch_delta" and mapped_tasks:
            status = "observed_delta"
            rationale = (
                "The exact locked manifests or zero-model reconciliation "
                "probe show a branch delta; mapped tasks test its usefulness."
            )
        elif source_status == "observed_branch_delta" and gates:
            status = "infrastructure_only"
            rationale = (
                "The exact locked manifests show a branch delta, and the "
                "named infrastructure gate qualifies it independently of "
                "Agent task outcomes."
            )
        elif source_status == "mechanism_probe_required":
            status = "unqualified"
            rationale = (
                "The zero-model mechanism probe did not establish the "
                "required branch behavior."
            )
        elif source_status == "agent_canary_required" and mapped_tasks:
            status = "observed_delta"
            rationale = (
                "The behavior is mapped to a locked natural-maintainer task; "
                "the task outcome determines whether the delta is useful."
            )
        elif gates:
            status = "infrastructure_only"
            rationale = (
                "This behavior is qualified by a separate named "
                "infrastructure gate, not by an Agent task score."
            )
        else:
            status = "unqualified"
            rationale = (
                "No locked task or completed infrastructure gate currently "
                "qualifies this behavior."
            )
        result.append(
            {
                "release_note": release_note,
                "status": status,
                "task_ids": list(mapped_tasks),
                "dimensions": list(mapped_dimensions),
                "infrastructure_gates": list(gates),
                "rationale": rationale,
            }
        )
        if status == "observed_delta" and not (
            mapped_tasks and mapped_dimensions
        ):
            raise ValueError(
                "observed release-note deltas require mapped task and "
                "dimension evidence"
            )
        if status == "infrastructure_only" and not gates:
            raise ValueError(
                "infrastructure-only release-note coverage requires a "
                "named gate"
            )
    if seen != set(locked_behaviors) or len(result) != len(locked_behaviors):
        raise ValueError(
            "mechanism receipt does not classify every locked release-note "
            "behavior exactly once"
        )
    return tuple(result)


def tool_surface_coverage_v1(
    receipt: Mapping[str, Any],
    *,
    task_ids: Sequence[str],
) -> dict[str, Any]:
    """Classify every locked read-only MCP tool without overstating evidence.

    Project-scoped tools are assigned to natural maintainer tasks. Account,
    organization, and mutable external-corpus tools stay in exact zero-model
    manifest/schema conformance. A canary may deliberately defer some task
    families, but no initialized tool may disappear from the coverage receipt.
    """

    candidates = [
        item for item in receipt.get("candidates") or () if isinstance(item, Mapping)
    ]
    selected_candidates = _mcp_release_candidates_from_receipt(receipt)
    if len(candidates) != len(selected_candidates):
        raise ValueError("tool coverage requires both exact MCP candidates")
    manifests = {
        str(item.get("id") or ""): tuple(
            sorted(str(name) for name in item.get("initialized_tools") or ())
        )
        for item in candidates
    }
    if set(manifests) != {item[0] for item in selected_candidates}:
        raise ValueError("tool coverage candidate identities do not match")
    versions = {
        str(item.get("id") or ""): str(item.get("version_identity") or "")
        for item in candidates
    }
    explicit_bindings = receipt.get("candidate_bindings") is not None
    if any(
        versions.get(import_id)
        not in ({version_identity} if explicit_bindings else {"", version_identity})
        for import_id, version_identity in selected_candidates
    ):
        raise ValueError("tool coverage candidate versions do not match")
    schema_digests = {
        str(item.get("id") or ""): {
            "initialized": str(item.get("initialized_tool_schema_digest") or ""),
            "locked": str(item.get("locked_tool_schema_digest") or ""),
        }
        for item in candidates
    }
    if any(
        item.get("initialized_schema_matches_lock") is not True
        or not schema_digests[str(item.get("id") or "")]["initialized"]
        or schema_digests[str(item.get("id") or "")]["initialized"]
        != schema_digests[str(item.get("id") or "")]["locked"]
        for item in candidates
    ):
        raise ValueError(
            "tool coverage requires initialized input schemas to match "
            "their exact locks"
        )
    unique_manifests = set(manifests.values())
    if len(unique_manifests) != 1:
        raise ValueError("tool coverage requires identical exact tool surfaces")
    manifest = next(iter(unique_manifests))
    declared_task_tools = {
        tool: task_id
        for task_id, tools in _MCP_PROJECT_TASK_TOOLS.items()
        for tool in tools
    }
    declared_task_options = {
        tool: (task_id, *_MCP_PROJECT_TOOL_TASK_ALIASES.get(tool, ()))
        for tool, task_id in declared_task_tools.items()
    }
    overlap = set(declared_task_tools) & set(_MCP_ZERO_MODEL_TOOLS)
    if overlap:
        raise ValueError(
            "MCP tools cannot be both task and zero-model qualified: "
            + ", ".join(sorted(overlap))
        )
    classified = set(declared_task_tools) | set(_MCP_ZERO_MODEL_TOOLS)
    if set(manifest) != classified:
        missing = sorted(set(manifest) - classified)
        stale = sorted(classified - set(manifest))
        raise ValueError(
            "MCP tool coverage must classify the exact initialized surface"
            + (f"; unclassified: {', '.join(missing)}" if missing else "")
            + (f"; stale: {', '.join(stale)}" if stale else "")
        )
    available_tasks = {str(item) for item in task_ids}
    tools: list[dict[str, Any]] = []
    for tool in manifest:
        task_options = declared_task_options.get(tool)
        if task_options is not None:
            assigned_task = next(
                (
                    task_id
                    for task_id in task_options
                    if task_id in available_tasks
                ),
                None,
            )
            tools.append(
                {
                    "tool": tool,
                    "qualification": "agent_task",
                    "task_id": assigned_task or task_options[0],
                    "accepted_task_ids": list(task_options),
                    "status": (
                        "assigned" if assigned_task else "confirmation_required"
                    ),
                    "claim_scope": (
                        "Agent outcome and mechanism evidence only when the "
                        "locked task records a successful invocation."
                    ),
                }
            )
        else:
            tools.append(
                {
                    "tool": tool,
                    "qualification": "zero_model_conformance",
                    "status": "manifest_and_schema_only",
                    "rationale": _MCP_ZERO_MODEL_TOOLS[tool],
                    "claim_scope": (
                        "Registration and schema conformance only; no task "
                        "outcome or behavioral usefulness claim."
                    ),
                }
            )
    unsigned = {
        "schema_version": 1,
        "candidate_manifests": {
            key: list(value) for key, value in sorted(manifests.items())
        },
        "candidate_schema_digests": dict(sorted(schema_digests.items())),
        "tools": tools,
        "agent_task_tools": len(declared_task_tools),
        "zero_model_tools": len(_MCP_ZERO_MODEL_TOOLS),
        "total_tools": len(manifest),
        "taskset_comprehensive": all(
            item["status"] != "confirmation_required" for item in tools
        ),
    }
    return {
        **unsigned,
        "coverage_digest": stable_digest(unsigned),
    }


def _successful_mcp_value(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("ok") is not True:
        return {}
    value = raw.get("value")
    return dict(value) if isinstance(value, Mapping) else {}


def _mcp_call_error(raw: Any) -> str:
    if not isinstance(raw, Mapping):
        return "missing result"
    if raw.get("error"):
        return str(raw["error"])[:500]
    value = raw.get("value")
    if isinstance(value, Mapping) and value.get("error"):
        return str(value["error"])[:500]
    if isinstance(value, Mapping) and value.get("message"):
        return str(value["message"])[:500]
    return "MCP tool returned an error"


def verify_private_project_topology(
    *,
    source_project: str,
    result_project: str,
    env: Mapping[str, str],
) -> dict[str, str]:
    """Verify both W&B projects exist and are private without writing data."""

    api_key = trace_api_key(env)
    if not api_key:
        raise RuntimeError("private-project verification requires WANDB_API_KEY")
    endpoint_binding = _qualification_endpoint_binding(env)
    selected_env = {
        "WANDB_API_KEY": api_key,
        "WANDB_BASE_URL": endpoint_binding["api_base_url"],
        "WANDB_SILENT": "true",
    }
    query = """
    query FugueProjectAccess($entity: String!, $project: String!) {
      project(entityName: $entity, name: $project) {
        name
        access
      }
    }
    """
    result: dict[str, str] = {}
    with _temporary_environment(selected_env):
        from wandb.apis.internal import Api

        api = Api().api
        for project_slug in (source_project, result_project):
            entity, project = project_slug.split("/", 1)
            response = api.execute(
                query,
                {"entity": entity, "project": project},
            )
            metadata = (
                response.get("project")
                if isinstance(response, Mapping)
                else None
            )
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("name") != project
                or metadata.get("access") != "PRIVATE"
            ):
                raise RuntimeError(
                    "Fugue qualification projects must already exist with "
                    "PRIVATE access"
                )
            result[project_slug] = "PRIVATE"
    return result


def verify_hosted_source_conformance(
    *,
    evidence_lock: Path,
    env_file: Path,
    output: Path,
    source_project: str = QUALIFICATION_SOURCE_PROJECT,
    result_project: str = QUALIFICATION_RESULT_PROJECT,
) -> dict[str, Any]:
    """Verify the immutable source cohort without invoking a model or MCP.

    The source project is read through the bounded Weave Calls query API. The
    result project is identity metadata only; this verifier never initializes
    Weave, creates a Run, or publishes an Evaluation.
    """

    raw_lock = json.loads(evidence_lock.resolve().read_text(encoding="utf-8"))
    lock = validate_evidence_lock(
        raw_lock,
        expected_project=None,
        expected_source_project=source_project,
        expected_result_project=result_project,
    )
    env = load_env(env_file.resolve())
    api_key = trace_api_key(env)
    if not api_key:
        raise RuntimeError("hosted source verification requires WANDB_API_KEY")
    endpoint_binding = _qualification_endpoint_binding(env)
    trace_base_url = endpoint_binding["trace_base_url"]
    roots, children = _fetch_hosted_source_calls(
        source_project=source_project,
        trace_base_url=trace_base_url,
        api_key=api_key,
        evaluation_call_ids=[
            str(item["call_id"]) for item in lock["objects"]["evaluations"]
        ],
    )
    project_access = verify_private_project_topology(
        source_project=source_project,
        result_project=result_project,
        env=env,
    )
    receipt = build_hosted_source_conformance_receipt(
        evidence_lock=lock,
        evaluation_roots=roots,
        direct_children=children,
        project_access=project_access,
        endpoint_binding=endpoint_binding,
    )
    if api_key in json.dumps(receipt, sort_keys=True):
        raise RuntimeError("hosted source receipt contains a credential")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output.resolve(), receipt)
    return receipt


def build_hosted_source_conformance_receipt(
    *,
    evidence_lock: Mapping[str, Any],
    evaluation_roots: Sequence[Mapping[str, Any]],
    direct_children: Mapping[str, Sequence[Mapping[str, Any]]],
    project_access: Mapping[str, str] | None = None,
    endpoint_binding: Mapping[str, str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the fail-closed source snapshot receipt from public call metadata."""

    source_project = evidence_source_project(evidence_lock)
    result_project = evidence_result_project(evidence_lock)
    locked_evaluations = {
        str(item["call_id"]): dict(item)
        for item in evidence_lock["objects"]["evaluations"]
    }
    root_by_id = {
        str(item.get("id") or item.get("call_id") or ""): dict(item)
        for item in evaluation_roots
        if str(item.get("id") or item.get("call_id") or "")
    }
    blockers: list[str] = []
    resolved_access = {
        str(key): str(value)
        for key, value in (project_access or {}).items()
    }
    if resolved_access != {
        source_project: "PRIVATE",
        result_project: "PRIVATE",
    }:
        blockers.append("source_and_result_projects_must_be_private")
    if len(root_by_id) != len(evaluation_roots):
        blockers.append("duplicate_or_missing_evaluation_root_id")
    if set(root_by_id) != set(locked_evaluations):
        blockers.append("evaluation_root_set_drift")
    if set(direct_children) != set(locked_evaluations):
        blockers.append("evaluation_child_query_set_drift")

    root_rows: list[dict[str, Any]] = []
    seen_child_ids: set[str] = set()
    all_child_metadata: list[dict[str, Any]] = []
    total_predictions = 0
    total_summaries = 0
    total_children = 0
    for call_id, locked in sorted(locked_evaluations.items()):
        root = root_by_id.get(call_id, {})
        root_project = _hosted_call_project(root)
        root_operation = _hosted_operation_name(root)
        if root_project != source_project:
            blockers.append(f"{call_id}:evaluation_root_project_mismatch")
        if root_operation != "Evaluation.evaluate":
            blockers.append(f"{call_id}:evaluation_root_operation_mismatch")

        children = [dict(item) for item in direct_children.get(call_id, ())]
        child_ids = [
            str(item.get("id") or item.get("call_id") or "") for item in children
        ]
        if not all(child_ids) or len(set(child_ids)) != len(child_ids):
            blockers.append(f"{call_id}:duplicate_or_missing_child_id")
        duplicate_across_roots = seen_child_ids.intersection(child_ids)
        if duplicate_across_roots:
            blockers.append(f"{call_id}:child_reused_across_evaluation_roots")
        seen_child_ids.update(child_ids)

        operation_counts: dict[str, int] = {}
        for child in children:
            child_id = str(child.get("id") or child.get("call_id") or "")
            child_project = _hosted_call_project(child)
            parent_id = str(child.get("parent_id") or "")
            operation = _hosted_operation_name(child)
            if child_project != source_project:
                blockers.append(f"{call_id}:{child_id}:child_project_mismatch")
            if parent_id != call_id:
                blockers.append(f"{call_id}:{child_id}:child_parent_mismatch")
            if not operation:
                blockers.append(f"{call_id}:{child_id}:child_operation_missing")
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
            all_child_metadata.append(
                {
                    "id": child_id,
                    "op_name": operation,
                    "parent_id": parent_id,
                    "project_id": child_project,
                }
            )

        predictions = int(operation_counts.get("Evaluation.predict_and_score") or 0)
        summaries = int(operation_counts.get("Evaluation.summarize") or 0)
        expected_predictions = int(locked["prediction_rows"])
        unexpected_operations = sorted(
            operation
            for operation, count in operation_counts.items()
            if count
            and operation
            not in {
                "Evaluation.predict_and_score",
                "Evaluation.summarize",
            }
        )
        if predictions != expected_predictions:
            blockers.append(f"{call_id}:prediction_child_count_drift")
        if summaries != 1:
            blockers.append(f"{call_id}:summary_child_count_drift")
        if unexpected_operations:
            blockers.append(f"{call_id}:unexpected_direct_child_operation")
        if len(children) != expected_predictions + 1:
            blockers.append(f"{call_id}:direct_child_count_drift")

        total_predictions += predictions
        total_summaries += summaries
        total_children += len(children)
        root_rows.append(
            {
                "evaluation_call_id": call_id,
                "evaluation_revision": locked["revision"],
                "locked_prediction_rows": expected_predictions,
                "observed_direct_children": len(children),
                "observed_predict_and_score_children": predictions,
                "observed_summarize_children": summaries,
                "unexpected_operations": unexpected_operations,
            }
        )

    observed = {
        "evaluation_roots": len(root_by_id),
        "direct_children": total_children,
        "predict_and_score_children": total_predictions,
        "summarize_children": total_summaries,
    }
    for name in (
        "evaluation_roots",
        "direct_children",
        "predict_and_score_children",
        "summarize_children",
    ):
        if observed[name] != _SOURCE_CONFORMANCE_EXPECTATIONS[name]:
            blockers.append(f"aggregate_{name}_drift")
    blockers = sorted(set(blockers))
    source_snapshot = {
        "evaluation_roots": [
            {
                "id": call_id,
                "op_name": _hosted_operation_name(root_by_id.get(call_id, {})),
                "project_id": _hosted_call_project(root_by_id.get(call_id, {})),
            }
            for call_id in sorted(locked_evaluations)
        ],
        "direct_children": sorted(
            all_child_metadata,
            key=lambda item: (item["parent_id"], item["id"]),
        ),
    }
    unsigned = {
        "schema_version": SOURCE_CONFORMANCE_SCHEMA_VERSION,
        "kind": "mcp-release-hosted-source-conformance",
        "status": "passed" if not blockers else "failed",
        "source_project": source_project,
        "result_project": result_project,
        "project_access": dict(sorted(resolved_access.items())),
        "endpoint_binding": dict(
            endpoint_binding or _qualification_endpoint_binding({})
        ),
        "evidence_lock_digest": evidence_lock["evidence_lock_digest"],
        "created_at": created_at
        or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "query_scope": {
            "read_only": True,
            "models_invoked": 0,
            "calls_published": 0,
            "child_limit_per_root": 20,
            "fields": [
                "id",
                "op_name",
                "parent_id",
                "project_id",
            ],
        },
        "expectations": dict(_SOURCE_CONFORMANCE_EXPECTATIONS),
        "comparison_expectations": {
            "baseline": {
                "direct_children": 18,
                "includes_summarize_children": 2,
            },
            "repaired_candidate": {
                "direct_predict_and_score_children": 16,
                "excludes_non_prediction_children": True,
            },
        },
        "observed": observed,
        "evaluation_roots": root_rows,
        "source_snapshot_digest": stable_digest(source_snapshot),
        "blockers": blockers,
        "claim_scope": (
            "Hosted source shape and drift only. The 18-child legacy count and "
            "16-row repaired target are expectations for later MCP behavior; "
            "this receipt contains no Agent, model, or release outcome."
        ),
        "receipt_digest": "",
    }
    return {
        **unsigned,
        "receipt_digest": stable_digest(unsigned),
    }


def _fetch_hosted_source_calls(
    *,
    source_project: str,
    trace_base_url: str,
    api_key: str,
    evaluation_call_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=30.0, headers=headers) as client:
        roots = _query_hosted_calls(
            client,
            trace_base_url=trace_base_url,
            source_project=source_project,
            call_filter={
                "call_ids": list(evaluation_call_ids),
                "trace_roots_only": False,
            },
            limit=len(evaluation_call_ids),
        )
        children = {
            call_id: _query_hosted_calls(
                client,
                trace_base_url=trace_base_url,
                source_project=source_project,
                call_filter={
                    "parent_ids": [call_id],
                    "trace_roots_only": False,
                },
                limit=20,
            )
            for call_id in evaluation_call_ids
        }
    return roots, children


def _query_hosted_calls(
    client: httpx.Client,
    *,
    trace_base_url: str,
    source_project: str,
    call_filter: Mapping[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    response = client.post(
        f"{trace_base_url}/calls/stream_query",
        json={
            "project_id": source_project,
            "filter": dict(call_filter),
            "limit": limit,
            "include_costs": False,
            "include_feedback": False,
            "columns": [
                "id",
                "op_name",
                "parent_id",
                "project_id",
            ],
        },
    )
    if response.status_code >= 400:
        raise RuntimeError("hosted source Calls query failed")
    return _decode_hosted_call_stream(response.text)


def _decode_hosted_call_stream(value: str) -> list[dict[str, Any]]:
    text = value.strip()
    if not text:
        return []
    if text[:1] in {"[", "{"}:
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return [item for item in decoded if isinstance(item, dict)]
        if isinstance(decoded, Mapping) and isinstance(decoded.get("calls"), list):
            return [item for item in decoded["calls"] if isinstance(item, dict)]
    calls: list[dict[str, Any]] = []
    for line in text.splitlines():
        item = json.loads(line)
        if isinstance(item, dict):
            calls.append(item)
    return calls


def _hosted_call_project(value: Mapping[str, Any]) -> str:
    project = value.get("project_id")
    if project:
        return str(project)
    nested = value.get("project")
    if isinstance(nested, Mapping) and nested.get("id"):
        return str(nested["id"])
    return ""


def _hosted_operation_name(value: Mapping[str, Any]) -> str:
    operation = str(value.get("op_name") or "")
    if "/op/" in operation:
        operation = operation.split("/op/", 1)[1]
    if ":" in operation:
        operation = operation.split(":", 1)[0]
    return operation


def _source_preparation_actions() -> tuple[str, ...]:
    actions = [
        *(f"wandb-run:{item['id']}" for item in _RUNS),
        "weave-dataset:mcp-release-maintenance-cases",
        *(
            f"weave-conversation:{item['id']}:{index}"
            for item in _RUNS
            for index in range(1, 5)
        ),
        "weave-evaluation-object:maintainer-r17",
        "weave-evaluation-run:maintainer-r17",
        "weave-evaluation-object:maintainer-r18",
        "weave-evaluation-run:maintainer-r18",
    ]
    return tuple(actions)


_SOURCE_PREPARATION_ACTIONS = frozenset(_source_preparation_actions())
_EXPECTED_WEAVE_OBJECT_VERSION_COUNTS = {
    ("object", "mcp-release-maintenance-cases"): 1,
    ("object", "mcp-release-maintainer-r17"): 1,
    ("object", "mcp-release-maintainer-r18"): 1,
    ("object", "EvaluationResults"): 2,
    ("op", "fugue.qualification.maintenance_agent"): 1,
    ("op", "fugue.qualification.wandb_mcp_tool"): 1,
    ("op", "Evaluation.evaluate"): 1,
    ("op", "Evaluation.summarize"): 1,
    ("op", "Evaluation.predict_and_score"): 1,
    ("op", "fugue.qualification.evidence_alignment_scorer"): 1,
    ("op", "fugue.qualification.baseline_evidence_model"): 1,
    ("op", "fugue.qualification.candidate_evidence_model"): 1,
}


def _source_preparation_progress_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.progress.json")


def _source_preparation_lock_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.prepare.lock")


def _signed_source_progress(raw: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        **dict(raw),
        "progress_digest": "",
    }
    return {
        **unsigned,
        "progress_digest": stable_digest(unsigned),
    }


def _validate_source_progress(
    raw: Mapping[str, Any],
    *,
    source_project: str,
    result_project: str,
) -> dict[str, Any]:
    value = dict(raw)
    supplied = str(value.get("progress_digest") or "")
    unsigned = {**value, "progress_digest": ""}
    if len(supplied) != 64 or stable_digest(unsigned) != supplied:
        raise ValueError("source preparation progress digest does not match")
    if value.get("schema_version") != SOURCE_PREPARATION_PROGRESS_SCHEMA_VERSION:
        raise ValueError("source preparation progress schema is unsupported")
    if (
        value.get("source_project") != source_project
        or value.get("result_project") != result_project
        or value.get("seed_digest")
        != qualification_seed_digest(source_project=source_project)
    ):
        raise ValueError("source preparation progress identity does not match")
    completed = value.get("completed_actions")
    if (
        not isinstance(completed, list)
        or any(
            not isinstance(item, str) or item not in _SOURCE_PREPARATION_ACTIONS
            for item in completed
        )
        or len(set(completed)) != len(completed)
    ):
        raise ValueError("source preparation completed actions are invalid")
    in_flight = value.get("in_flight_action")
    if in_flight is not None and in_flight not in _SOURCE_PREPARATION_ACTIONS:
        raise ValueError("source preparation in-flight action is invalid")
    if value.get("state") not in {"preparing", "completed"}:
        raise ValueError("source preparation state is invalid")
    if not str(value.get("preparation_id") or ""):
        raise ValueError("source preparation id is missing")
    if not str(value.get("created_at") or ""):
        raise ValueError("source preparation creation time is missing")
    return value


def _load_or_create_source_progress(
    path: Path,
    *,
    source_project: str,
    result_project: str,
) -> dict[str, Any]:
    if path.is_file():
        return _validate_source_progress(
            json.loads(path.read_text(encoding="utf-8")),
            source_project=source_project,
            result_project=result_project,
        )
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    progress = _signed_source_progress(
        {
            "schema_version": SOURCE_PREPARATION_PROGRESS_SCHEMA_VERSION,
            "preparation_id": stable_digest(
                {
                    "source_project": source_project,
                    "result_project": result_project,
                    "seed_digest": qualification_seed_digest(
                        source_project=source_project
                    ),
                    "created_at": created_at,
                }
            ),
            "source_project": source_project,
            "result_project": result_project,
            "seed_digest": qualification_seed_digest(source_project=source_project),
            "created_at": created_at,
            "updated_at": created_at,
            "state": "preparing",
            "in_flight_action": None,
            "completed_actions": [],
            "last_inventory_digest": None,
            "result_lock_digest": None,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, progress)
    return progress


def _write_source_progress(
    path: Path,
    progress: Mapping[str, Any],
    **changes: Any,
) -> dict[str, Any]:
    value = _signed_source_progress(
        {
            **dict(progress),
            **changes,
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
    )
    atomic_write_json(path, value)
    return value


def _expected_source_actions(inventory: Mapping[str, Any]) -> set[str]:
    present = inventory.get("present_actions")
    if not isinstance(present, list) or any(
        not isinstance(item, str) for item in present
    ):
        raise RuntimeError("hosted source inventory actions are invalid")
    result = set(present)
    unknown = result - _SOURCE_PREPARATION_ACTIONS
    if unknown:
        raise RuntimeError(
            "hosted source inventory contains unknown actions: "
            + ", ".join(sorted(unknown))
        )
    return result


def _weave_object_versions_complete(
    object_versions: Sequence[Mapping[str, Any]],
) -> bool:
    counts: dict[tuple[str, str], int] = {}
    for item in object_versions:
        key = (
            str(item.get("kind") or ""),
            str(item.get("object_id") or ""),
        )
        counts[key] = counts.get(key, 0) + 1
    return counts == _EXPECTED_WEAVE_OBJECT_VERSION_COUNTS


def _validate_source_inventory(
    inventory: Mapping[str, Any],
    *,
    source_project: str,
) -> dict[str, Any]:
    value = dict(inventory)
    if value.get("schema_version") != 1:
        raise RuntimeError("hosted source inventory schema is unsupported")
    if value.get("source_project") != source_project or value.get(
        "seed_digest"
    ) != qualification_seed_digest(source_project=source_project):
        raise RuntimeError("hosted source inventory identity does not match")
    extras = value.get("extra_objects")
    drift = value.get("drift")
    if not isinstance(extras, list) or not isinstance(drift, list):
        raise RuntimeError("hosted source inventory findings are invalid")
    if extras:
        raise RuntimeError(
            "hosted source inventory contains extra objects: "
            + ", ".join(sorted(str(item) for item in extras))
        )
    if drift:
        raise RuntimeError(
            "hosted source inventory drifted: "
            + ", ".join(sorted(str(item) for item in drift))
        )
    present = _expected_source_actions(value)
    object_versions = value.get("weave_object_versions")
    if not isinstance(object_versions, list) or any(
        not isinstance(item, Mapping) for item in object_versions
    ):
        raise RuntimeError(
            "hosted source Weave object-version inventory is invalid"
        )
    expected_complete = (
        present == _SOURCE_PREPARATION_ACTIONS
        and _weave_object_versions_complete(object_versions)
    )
    if bool(value.get("complete")) is not expected_complete:
        raise RuntimeError("hosted source inventory completeness disagrees")
    supplied = str(value.get("inventory_digest") or "")
    unsigned = {**value, "inventory_digest": ""}
    if len(supplied) != 64 or stable_digest(unsigned) != supplied:
        raise RuntimeError("hosted source inventory digest does not match")
    return value


def _source_inventory(
    *,
    source_project: str,
    runs: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any] | None,
    conversations: Sequence[Mapping[str, Any]],
    evaluation_objects: Mapping[str, Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    object_versions: Sequence[Mapping[str, Any]] = (),
    extra_objects: Sequence[str] = (),
    drift: Sequence[str] = (),
) -> dict[str, Any]:
    present_actions = [
        *(f"wandb-run:{item['id']}" for item in runs),
        *(
            ["weave-dataset:mcp-release-maintenance-cases"]
            if dataset is not None
            else []
        ),
        *(
            f"weave-conversation:{item['run_id']}:{int(item['conversation_index'])}"
            for item in conversations
        ),
        *(f"weave-evaluation-object:{revision}" for revision in evaluation_objects),
        *(f"weave-evaluation-run:{item['revision']}" for item in evaluations),
    ]
    unsigned = {
        "schema_version": 1,
        "source_project": source_project,
        "seed_digest": qualification_seed_digest(source_project=source_project),
        "runs": [dict(item) for item in runs],
        "dataset": dict(dataset) if dataset is not None else None,
        "source_conversations": [dict(item) for item in conversations],
        "evaluation_objects": {
            key: dict(value) for key, value in sorted(evaluation_objects.items())
        },
        "evaluations": [dict(item) for item in evaluations],
        "weave_object_versions": [
            dict(item) for item in object_versions
        ],
        "present_actions": sorted(present_actions),
        "complete": (
            set(present_actions) == _SOURCE_PREPARATION_ACTIONS
            and _weave_object_versions_complete(object_versions)
        ),
        "extra_objects": sorted(str(item) for item in extra_objects),
        "drift": sorted(str(item) for item in drift),
        "inventory_digest": "",
    }
    return {
        **unsigned,
        "inventory_digest": stable_digest(unsigned),
    }


def _stable_hosted_source_inventory(
    entity: str,
    project: str,
    *,
    source_project: str,
) -> dict[str, Any]:
    first = _validate_source_inventory(
        _inventory_hosted_source(
            entity,
            project,
            source_project=source_project,
        ),
        source_project=source_project,
    )
    second = _validate_source_inventory(
        _inventory_hosted_source(
            entity,
            project,
            source_project=source_project,
        ),
        source_project=source_project,
    )
    if first["inventory_digest"] != second["inventory_digest"]:
        raise RuntimeError(
            "hosted source inventory changed during read-only verification"
        )
    return second


def _inventory_weave_receipts(
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = inventory.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RuntimeError("hosted source Dataset is not prepared")
    return {
        "dataset": dict(dataset),
        "source_conversations": [
            dict(item) for item in inventory.get("source_conversations") or ()
        ],
        "evaluations": [dict(item) for item in inventory.get("evaluations") or ()],
        "weave_object_versions": [
            dict(item)
            for item in inventory.get("weave_object_versions") or ()
        ],
    }


def _lock_matches_source_inventory(
    lock: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> bool:
    objects = lock.get("objects")
    if not isinstance(objects, Mapping):
        return False
    expected_objects = {
        "runs": [dict(item) for item in inventory.get("runs") or ()],
        **_inventory_weave_receipts(inventory),
    }
    return stable_digest(objects) == stable_digest(expected_objects) and lock.get(
        "source_inventory_digest"
    ) == inventory.get("inventory_digest")


def verify_hosted_source_drift(
    *,
    evidence_lock: Path,
    env: Mapping[str, str],
) -> EvidenceDriftCheckV1:
    """Read the complete locked source cohort twice and report safe drift state.

    This verifier never initializes a result project and never publishes an
    object. Authentication and transport failures intentionally collapse to a
    static unavailable reason so credentials and service details cannot enter
    attempt evidence.
    """

    try:
        lock = validate_evidence_lock(
            json.loads(evidence_lock.resolve().read_text(encoding="utf-8")),
            expected_project=None,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return EvidenceDriftCheckV1(
            status="unavailable",
            expected_digest="0" * 64,
            reason="the approved evidence lock could not be verified",
        )
    expected_digest = str(lock["evidence_lock_digest"])
    source_project = evidence_source_project(lock)
    result_project = evidence_result_project(lock)
    api_key = trace_api_key(env)
    if not api_key:
        return EvidenceDriftCheckV1(
            status="unavailable",
            expected_digest=expected_digest,
            reason="hosted source authentication is unavailable",
        )
    entity, project = source_project.split("/", 1)
    try:
        endpoint_binding = _qualification_endpoint_binding(env)
    except RuntimeError:
        return EvidenceDriftCheckV1(
            status="unavailable",
            expected_digest=expected_digest,
            reason="the hosted source endpoint does not match its lock",
        )
    selected_env = {
        "WANDB_API_KEY": api_key,
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project,
        "WANDB_BASE_URL": endpoint_binding["api_base_url"],
        "WANDB_MODE": "online",
        "WANDB_RESUME": "never",
        "WANDB_SILENT": "true",
        "WEAVE_ALLOW_UNSAFE_CUSTOM_OBJ_DECODE": "false",
    }
    try:
        verify_private_project_topology(
            source_project=source_project,
            result_project=result_project,
            env=env,
        )
        with _temporary_environment(selected_env):
            inventory = _stable_hosted_source_inventory(
                entity,
                project,
                source_project=source_project,
            )
    except Exception:
        return EvidenceDriftCheckV1(
            status="unavailable",
            expected_digest=expected_digest,
            reason="the hosted source inventory could not be verified",
        )
    if _lock_matches_source_inventory(lock, inventory):
        return EvidenceDriftCheckV1(
            status="matched",
            expected_digest=expected_digest,
            observed_digest=expected_digest,
        )
    observed_digest = str(inventory.get("inventory_digest") or "")
    if len(observed_digest) != 64:
        observed_digest = stable_digest(
            {
                "source_project": source_project,
                "inventory_state": "verified_but_not_locked",
            }
        )
    if observed_digest == expected_digest:
        observed_digest = stable_digest(
            {
                "source_project": source_project,
                "inventory_digest": observed_digest,
                "lock_match": False,
            }
        )
    return EvidenceDriftCheckV1(
        status="drifted",
        expected_digest=expected_digest,
        observed_digest=observed_digest,
        reason="the hosted source inventory differs from the approved lock",
    )


def prepare_hosted_project(
    *,
    project: str | None = None,
    source_project: str | None = None,
    result_project: str | None = None,
    output: Path,
    env_file: Path,
) -> dict[str, Any]:
    if project is not None:
        raise ValueError(
            "legacy single-project preparation is read-only compatibility; "
            "V3 writers require distinct source and result projects"
        )
    resolved_source_project = source_project or QUALIFICATION_SOURCE_PROJECT
    resolved_result_project = result_project or QUALIFICATION_RESULT_PROJECT
    if (
        resolved_source_project == resolved_result_project
        or resolved_source_project == QUALIFICATION_RESULT_PROJECT
        or (
            resolved_source_project,
            resolved_result_project,
        )
        != (QUALIFICATION_SOURCE_PROJECT, QUALIFICATION_RESULT_PROJECT)
    ):
        raise ValueError(
            "V3 source preparation requires the dedicated immutable source "
            "project and distinct result project"
        )
    existing_lock: dict[str, Any] | None = None
    if output.is_file():
        existing_lock = validate_evidence_lock(
            json.loads(output.read_text(encoding="utf-8")),
            expected_project=None,
            expected_source_project=resolved_source_project,
            expected_result_project=resolved_result_project,
        )
    env = load_env(env_file)
    required = ("WANDB_API_KEY",)
    missing = [name for name in required if not str(env.get(name) or "").strip()]
    if missing:
        raise RuntimeError("hosted-project preparation requires: " + ", ".join(missing))
    endpoint_binding = _qualification_endpoint_binding(env)
    entity, project_name = resolved_source_project.split("/", 1)
    selected_env = {
        "WANDB_API_KEY": str(env["WANDB_API_KEY"]),
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project_name,
        "WANDB_BASE_URL": endpoint_binding["api_base_url"],
        "WANDB_MODE": "online",
        "WANDB_RESUME": "never",
        "WANDB_SILENT": "true",
        "WEAVE_ALLOW_UNSAFE_CUSTOM_OBJ_DECODE": "false",
    }
    verify_private_project_topology(
        source_project=resolved_source_project,
        result_project=resolved_result_project,
        env=env,
    )
    progress_path = _source_preparation_progress_path(output)
    lock_path = _source_preparation_lock_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        with _temporary_environment(selected_env):
            inventory = _stable_hosted_source_inventory(
                entity,
                project_name,
                source_project=resolved_source_project,
            )
            if existing_lock is not None:
                if not inventory["complete"]:
                    raise RuntimeError(
                        "existing evidence lock no longer resolves to a complete "
                        "hosted source cohort"
                    )
                if not _lock_matches_source_inventory(existing_lock, inventory):
                    raise RuntimeError(
                        "existing evidence lock disagrees with hosted source inventory"
                    )
                return existing_lock

            progress = _load_or_create_source_progress(
                progress_path,
                source_project=resolved_source_project,
                result_project=resolved_result_project,
            )
            if (
                progress["state"] == "completed"
                and progress.get("last_inventory_digest")
                != inventory["inventory_digest"]
            ):
                raise RuntimeError(
                    "completed source preparation receipt disagrees with hosted "
                    "inventory"
                )
            present = _expected_source_actions(inventory)
            missing_completed = set(progress["completed_actions"]) - present
            if missing_completed:
                raise RuntimeError(
                    "previously completed hosted source writes are not visible; "
                    "refusing recovery: " + ", ".join(sorted(missing_completed))
                )
            in_flight = progress.get("in_flight_action")
            if in_flight is not None:
                if in_flight not in present:
                    raise RuntimeError(
                        "previous hosted source write outcome is unresolved; "
                        "refusing to retry and risk duplicate evidence"
                    )
                completed = sorted(
                    {
                        *progress["completed_actions"],
                        str(in_flight),
                    }
                )
                progress = _write_source_progress(
                    progress_path,
                    progress,
                    in_flight_action=None,
                    completed_actions=completed,
                    last_inventory_digest=inventory["inventory_digest"],
                )

            def run_action(action: str, operation: Callable[[], None]) -> None:
                nonlocal progress
                if action in _expected_source_actions(inventory):
                    return
                progress = _write_source_progress(
                    progress_path,
                    progress,
                    state="preparing",
                    in_flight_action=action,
                    last_inventory_digest=inventory["inventory_digest"],
                )
                operation()
                progress = _write_source_progress(
                    progress_path,
                    progress,
                    in_flight_action=None,
                    completed_actions=sorted(
                        {
                            *progress["completed_actions"],
                            action,
                        }
                    ),
                )

            if not inventory["complete"]:
                _materialize_hosted_source(
                    entity,
                    project_name,
                    source_project=resolved_source_project,
                    inventory=inventory,
                    run_action=run_action,
                )
                inventory = _stable_hosted_source_inventory(
                    entity,
                    project_name,
                    source_project=resolved_source_project,
                )
            if not inventory["complete"]:
                raise RuntimeError(
                    "hosted source preparation did not produce a complete cohort"
                )
        value = _evidence_lock(
            resolved_source_project,
            inventory["runs"],
            _inventory_weave_receipts(inventory),
            result_project=resolved_result_project,
            created_at=str(progress["created_at"]),
            preparation_id=str(progress["preparation_id"]),
            source_inventory_digest=str(inventory["inventory_digest"]),
        )
        atomic_write_json(output, value)
        progress = _write_source_progress(
            progress_path,
            progress,
            state="completed",
            in_flight_action=None,
            completed_actions=sorted(_SOURCE_PREPARATION_ACTIONS),
            last_inventory_digest=inventory["inventory_digest"],
            result_lock_digest=value["evidence_lock_digest"],
        )
        del progress
        return validate_evidence_lock(
            value,
            expected_project=None,
            expected_source_project=resolved_source_project,
            expected_result_project=resolved_result_project,
        )


_RUN_HISTORY_KEYS = (
    "step",
    "latency_ms",
    "deterministic_pass",
    "projected_reads",
    "broad_reads",
    "source_returned",
    "source_opened",
    "observed_cost_usd",
)


def _expected_run_config(item: Mapping[str, Any], seed_digest: str) -> dict[str, Any]:
    return {
        "fugue_seed_digest": seed_digest,
        "candidate_revision": item["candidate_revision"],
        "attempt_label": item["attempt_label"],
        "evidence_snapshot": "qualification-v1",
        "contains_sensitive_data": False,
    }


def _expected_run_history(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    latency = int(item["latency_ms"])
    rows = []
    for step, multiplier in enumerate((0.7, 0.85, 1.0), start=1):
        row: dict[str, Any] = {
            "step": step,
            "latency_ms": round(latency * multiplier, 3),
            "deterministic_pass": int(bool(item["deterministic_pass"])),
            "projected_reads": int(item["projected_reads"]),
            "broad_reads": int(item["broad_reads"]),
            "source_returned": int(item["source_returned"]),
            "source_opened": int(item["source_opened"]),
        }
        if item["observed_cost_usd"] is not None:
            row["observed_cost_usd"] = round(
                float(item["observed_cost_usd"]) * multiplier,
                6,
            )
        rows.append(row)
    return rows


def _expected_run_artifact_payload(
    item: Mapping[str, Any],
    seed_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed_digest": seed_digest,
        "run_id": item["id"],
        "attempt_label": item["attempt_label"],
        "evidence": {
            key: value
            for key, value in item.items()
            if key not in {"id", "attempt_label"}
        },
    }


def _canonical_run_history(run: Any) -> list[dict[str, Any]]:
    required_keys = [key for key in _RUN_HISTORY_KEYS if key != "observed_cost_usd"]
    cost_by_step: dict[int, float] = {}
    for raw in run.scan_history(
        keys=["step", "observed_cost_usd"],
        page_size=1000,
    ):
        if not isinstance(raw, Mapping):
            raise RuntimeError("hosted W&B Run cost history row is invalid")
        if raw.get("observed_cost_usd") is None:
            continue
        step = int(raw.get("step") or raw.get("_step") or 0)
        if step <= 0 or step in cost_by_step:
            raise RuntimeError("hosted W&B Run cost history is ambiguous")
        cost_by_step[step] = float(raw["observed_cost_usd"])
    rows = []
    for raw in run.scan_history(keys=required_keys, page_size=1000):
        if not isinstance(raw, Mapping):
            raise RuntimeError("hosted W&B Run history row is invalid")
        row = {
            key: raw[key]
            for key in required_keys
            if key in raw and raw[key] is not None
        }
        step = int(row.get("step") or raw.get("_step") or 0)
        if step in cost_by_step:
            row["observed_cost_usd"] = cost_by_step.pop(step)
        if row:
            rows.append(row)
    if cost_by_step:
        raise RuntimeError("hosted W&B Run cost history has no matching step")
    return sorted(rows, key=lambda item: int(item.get("step") or 0))


def _read_run_artifact_payload(artifact: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fugue-wandb-evidence-read-") as temp:
        entry = artifact.get_path("attempt-evidence.json")
        try:
            downloaded = entry.download(root=temp, replace=True)
        except TypeError as exc:
            if "unexpected keyword argument 'replace'" not in str(exc):
                raise
            downloaded = entry.download(root=temp)
        path = Path(downloaded.name if hasattr(downloaded, "name") else str(downloaded))
        raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("hosted W&B Run evidence artifact is invalid")
    return raw


def _inspect_wandb_run(
    run: Any,
    item: Mapping[str, Any],
    *,
    source_project: str,
    seed_digest: str,
) -> dict[str, Any]:
    run_id = str(getattr(run, "id", "") or "")
    if run_id != item["id"]:
        raise RuntimeError(f"hosted W&B Run id drifted: {item['id']}")
    if str(getattr(run, "name", "") or "") != item["attempt_label"]:
        raise RuntimeError(f"hosted W&B Run name drifted: {run_id}")
    if str(getattr(run, "state", "") or "") != "finished":
        raise RuntimeError(f"hosted W&B Run is not finished: {run_id}")
    config = dict(getattr(run, "config", {}) or {})
    expected_config = _expected_run_config(item, seed_digest)
    if {key: config.get(key) for key in expected_config} != expected_config:
        raise RuntimeError(f"hosted W&B Run config drifted: {run_id}")
    history = _canonical_run_history(run)
    expected_history = _expected_run_history(item)
    if history != expected_history:
        raise RuntimeError(f"hosted W&B Run history drifted: {run_id}")
    artifacts = [
        artifact
        for artifact in run.logged_artifacts(per_page=100)
        if str(getattr(artifact, "type", "") or "") == "fugue-qualification-evidence"
        and dict(getattr(artifact, "metadata", {}) or {}).get("seed_digest")
        == seed_digest
    ]
    if len(artifacts) != 1:
        raise RuntimeError(f"hosted W&B Run evidence artifact count drifted: {run_id}")
    artifact = artifacts[0]
    artifact_payload = _read_run_artifact_payload(artifact)
    expected_payload = _expected_run_artifact_payload(item, seed_digest)
    if artifact_payload != expected_payload:
        raise RuntimeError(
            f"hosted W&B Run evidence artifact content drifted: {run_id}"
        )
    artifact_digest = str(getattr(artifact, "digest", "") or "")
    if not artifact_digest:
        raise RuntimeError(f"hosted W&B Run artifact digest is missing: {run_id}")
    summary = dict(getattr(run, "summary", {}) or {})
    expected_summary = {
        "fugue_seed_digest": seed_digest,
        "evidence_lock_status": "prepared",
        "artifact_digest": artifact_digest,
    }
    if {key: summary.get(key) for key in expected_summary} != expected_summary:
        raise RuntimeError(f"hosted W&B Run summary drifted: {run_id}")
    artifact_receipt = {
        "name": str(getattr(artifact, "name", "") or ""),
        "version": str(getattr(artifact, "version", "") or ""),
        "digest": artifact_digest,
        "qualified_name": str(getattr(artifact, "qualified_name", "") or ""),
        "content_digest": stable_digest(artifact_payload),
    }
    receipt = {
        "id": run_id,
        "name": str(run.name),
        "url": str(run.url),
        "ref": f"wandb-run:///{source_project}/{run_id}",
        "seed_digest": seed_digest,
        "state": str(run.state),
        "config_digest": stable_digest(expected_config),
        "history_digest": stable_digest(history),
        "summary_digest": stable_digest(expected_summary),
        "artifact": artifact_receipt,
    }
    return {
        **receipt,
        "content_digest": stable_digest(receipt),
    }


def _wandb_project_runs(api: Any, source_project: str) -> list[Any]:
    try:
        return task_evidence_wandb_runs(api.runs(source_project, per_page=100))
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        message = str(exc).lower()
        if (
            status == 404
            or "project not found" in message
            or "could not find project" in message
        ):
            return []
        raise


def _inventory_wandb_runs(
    entity: str,
    project: str,
    *,
    source_project: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "install fugue[research-worker] with the W&B Sandbox extra"
        ) from exc
    del entity, project
    api = wandb.Api()
    receipts: list[dict[str, Any]] = []
    drift: list[str] = []
    seed_digest = qualification_seed_digest(source_project=source_project)
    runs = _wandb_project_runs(api, source_project)
    expected = {str(item["id"]): item for item in _RUNS}
    by_id: dict[str, Any] = {}
    extras = []
    for run in runs:
        run_id = str(getattr(run, "id", "") or "")
        if run_id not in expected:
            extras.append(f"wandb-run:{run_id or '<missing-id>'}")
            continue
        if run_id in by_id:
            extras.append(f"wandb-run-duplicate:{run_id}")
            continue
        by_id[run_id] = run
    for run_id, item in expected.items():
        existing = by_id.get(run_id)
        if existing is None:
            continue
        try:
            receipts.append(
                _inspect_wandb_run(
                    existing,
                    item,
                    source_project=source_project,
                    seed_digest=seed_digest,
                )
            )
        except RuntimeError as exc:
            drift.append(str(exc))
    return (
        sorted(receipts, key=lambda item: str(item["id"])),
        sorted(extras),
        sorted(drift),
    )


def _create_wandb_run(
    entity: str,
    project: str,
    *,
    source_project: str,
    item: Mapping[str, Any],
) -> None:
    import wandb

    seed_digest = qualification_seed_digest(source_project=source_project)
    settings = wandb.Settings(
        console="off",
        disable_git=True,
        silent=True,
    )
    run = wandb.init(
        entity=entity,
        project=project,
        id=str(item["id"]),
        name=str(item["attempt_label"]),
        group="hosted-evidence-v1",
        job_type="maintenance-evidence",
        tags=("fugue", "qualification", "mcp-release"),
        config=_expected_run_config(item, seed_digest),
        resume="never",
        reinit="create_new",
        settings=settings,
    )
    if run is None:
        raise RuntimeError(f"failed to create W&B Run {item['id']}")
    try:
        _log_run_history(run, item)
        artifact_receipt = _log_run_artifact(run, item, seed_digest)
        run.summary.update(
            {
                "fugue_seed_digest": seed_digest,
                "evidence_lock_status": "prepared",
                "artifact_digest": artifact_receipt["digest"],
            }
        )
    finally:
        run.finish()


def _log_run_history(run: Any, item: Mapping[str, Any]) -> None:
    for payload in _expected_run_history(item):
        run.log(payload, step=int(payload["step"]))


def _log_run_artifact(
    run: Any,
    item: Mapping[str, Any],
    seed_digest: str,
) -> dict[str, Any]:
    import wandb

    payload = _expected_run_artifact_payload(item, seed_digest)
    with tempfile.TemporaryDirectory(prefix="fugue-wandb-evidence-") as temp:
        path = Path(temp) / "attempt-evidence.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact = wandb.Artifact(
            name=f"qualification-evidence-{item['id']}",
            type="fugue-qualification-evidence",
            metadata={"seed_digest": seed_digest},
        )
        artifact.add_file(path.as_posix(), name="attempt-evidence.json")
        logged = run.log_artifact(
            artifact,
            aliases=("qualification-v1",),
        )
        logged.wait()
    return {
        "name": logged.name,
        "version": logged.version,
        "digest": logged.digest,
        "qualified_name": logged.qualified_name,
    }


def _plain_json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        try:
            items = sorted(value.items(), key=lambda item: str(item[0]))
        except Exception as exc:
            if type(exc).__name__ != "UnsafeDeserializationError":
                raise
            return {"_fugue_untrusted_custom_object": True}
        return {str(key): _plain_json_value(item) for key, item in items}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        try:
            return [_plain_json_value(item) for item in value]
        except Exception as exc:
            if type(exc).__name__ != "UnsafeDeserializationError":
                raise
            return [{"_fugue_untrusted_custom_object": True}]
    if hasattr(value, "model_dump"):
        return _plain_json_value(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _plain_json_value(value.to_dict())
    return str(value)


def _call_record(call: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(call, "id", "") or ""),
        "project_id": str(getattr(call, "project_id", "") or ""),
        "trace_id": str(getattr(call, "trace_id", "") or ""),
        "parent_id": str(getattr(call, "parent_id", "") or ""),
        "op_name": _hosted_operation_name(
            {"op_name": str(getattr(call, "op_name", "") or "")}
        ),
        "inputs": _plain_json_value(getattr(call, "inputs", {}) or {}),
        "output": _plain_json_value(getattr(call, "output", None)),
        "attributes": _plain_json_value(getattr(call, "attributes", {}) or {}),
        "ended_at": str(getattr(call, "ended_at", "") or ""),
        "exception": _plain_json_value(getattr(call, "exception", None)),
    }


def _weave_missing_error(exc: Exception) -> bool:
    if type(exc).__name__ in {
        "NotFoundError",
        "ObjectDeletedError",
        "ProjectNotFound",
        "RefObjectsNotFoundError",
    }:
        return True
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    message = str(exc).lower()
    return status == 404 or (
        status == 403
        and ("project not found" in message or "could not find project" in message)
    )


@contextmanager
def _source_weave_client(project: str, *, write: bool):
    from weave.trace.context.weave_client_context import (
        get_weave_client,
        set_weave_client_global,
    )
    from weave.trace.weave_init import init_weave

    previous_client = get_weave_client()
    client = None
    try:
        client = init_weave(project, ensure_project_exists=write)
        yield client
    finally:
        if client is not None and client is not previous_client:
            client.finish()
        set_weave_client_global(None)
        if previous_client is not None:
            set_weave_client_global(previous_client)


def _source_weave_project_exists(client: Any, project: str) -> bool:
    from weave.trace_server.trace_server_interface import ProjectStatsReq

    try:
        client.server.project_stats(ProjectStatsReq(project_id=project))
    except Exception as exc:
        if _weave_missing_error(exc):
            return False
        raise
    return True


def _weave_object_or_none(client: Any, uri: str) -> Any | None:
    try:
        return client.get(weave.ref(uri))
    except Exception as exc:
        if _weave_missing_error(exc):
            return None
        raise


def _raw_weave_object_or_none(
    client: Any,
    *,
    project: str,
    object_id: str,
    digest: str,
) -> Any | None:
    from weave.trace_server.trace_server_interface import ObjReadReq

    try:
        return client.server.obj_read(
            ObjReadReq(
                project_id=project,
                object_id=object_id,
                digest=digest,
                include_tags_and_aliases=True,
            )
        ).obj
    except Exception as exc:
        if _weave_missing_error(exc):
            return None
        raise


def _immutable_weave_ref(value: Any, *, kind: str) -> str:
    ref = getattr(value, "ref", None)
    if ref is None:
        raise RuntimeError(f"hosted Weave {kind} has no immutable reference")
    uri = str(ref.uri())
    if "/object/" not in uri or ":" not in uri.rsplit("/object/", 1)[1]:
        raise RuntimeError(f"hosted Weave {kind} reference is mutable")
    return uri


def _weave_dataset_rows(dataset: Any) -> list[dict[str, Any]]:
    rows: Any = getattr(dataset, "rows", None)
    seen: set[int] = set()
    while hasattr(rows, "rows") and id(rows) not in seen:
        seen.add(id(rows))
        rows = rows.rows
    try:
        materialized = list(rows)
    except TypeError as exc:
        raise RuntimeError("hosted Weave Dataset rows are not readable") from exc
    if not all(isinstance(item, Mapping) for item in materialized):
        raise RuntimeError("hosted Weave Dataset rows are invalid")
    return [dict(item) for item in materialized]


def _inspect_weave_dataset(client: Any, project: str) -> dict[str, Any] | None:
    dataset = _weave_object_or_none(
        client,
        f"weave:///{project}/object/mcp-release-maintenance-cases:qualification-v1",
    )
    if dataset is None:
        return None
    rows = _weave_dataset_rows(dataset)
    expected_rows = [dict(item) for item in _EVALUATION_ROWS]
    if rows != expected_rows:
        raise RuntimeError("hosted Weave Dataset content drifted")
    if getattr(dataset, "name", None) != "mcp-release-maintenance-cases":
        raise RuntimeError("hosted Weave Dataset name drifted")
    ref = _immutable_weave_ref(dataset, kind="Dataset")
    return {
        "name": "mcp-release-maintenance-cases",
        "ref": ref,
        "rows": len(rows),
        "content_digest": stable_digest(rows),
    }


def _expected_conversation_inputs(
    item: Mapping[str, Any],
    index: int,
    seed_digest: str,
) -> dict[str, Any]:
    return {
        "run_id": str(item["id"]),
        "attempt_label": str(item["attempt_label"]),
        "conversation_index": index,
        "source_returned": int(item["source_returned"]),
        "source_opened": int(item["source_opened"]),
        "latency_ms": int(item["latency_ms"]),
        "seed_digest": seed_digest,
    }


def _expected_conversation_output(
    item: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    return {
        "run_id": str(item["id"]),
        "attempt_label": str(item["attempt_label"]),
        "conversation_index": index,
        "evidence_status": "observed",
    }


def _expected_tool_inputs(
    item: Mapping[str, Any],
    index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "tool_name": "wandb_projected_run_read",
            "run_id": str(item["id"]),
            "public_result": {
                "latency_ms": int(item["latency_ms"]),
                "source_returned": int(item["source_returned"]),
                "source_opened": int(item["source_opened"]),
            },
        },
        {
            "tool_name": "weave_call_evidence_read",
            "run_id": str(item["id"]),
            "public_result": {
                "evidence_status": "observed",
                "conversation_index": int(index),
            },
        },
    )


def _inventory_conversations(
    calls: Sequence[Mapping[str, Any]],
    *,
    project: str,
    seed_digest: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    expected = {
        (str(item["id"]), index): (item, index)
        for item in _RUNS
        for index in range(1, 5)
    }
    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for call in calls:
        by_parent.setdefault(str(call.get("parent_id") or ""), []).append(call)
    found: dict[tuple[str, int], Mapping[str, Any]] = {}
    extras: list[str] = []
    drift: list[str] = []
    tool_ids: set[str] = set()
    for call in calls:
        if call.get("op_name") != "fugue.qualification.maintenance_agent":
            continue
        attributes = call.get("attributes")
        inputs = call.get("inputs")
        if not isinstance(attributes, Mapping) or not isinstance(inputs, Mapping):
            drift.append(f"weave-conversation:{call.get('id')}:shape")
            continue
        if attributes.get("fugue.seed_digest") != seed_digest:
            extras.append(f"weave-conversation:{call.get('id')}:other-seed")
            continue
        coordinate = (
            str(inputs.get("run_id") or ""),
            int(inputs.get("conversation_index") or 0),
        )
        if coordinate not in expected or coordinate in found:
            extras.append(
                f"weave-conversation:{coordinate[0]}:{coordinate[1]}:{call.get('id')}"
            )
            continue
        found[coordinate] = call
    receipts = []
    for coordinate, (item, index) in expected.items():
        call = found.get(coordinate)
        if call is None:
            continue
        call_id = str(call.get("id") or "")
        if (
            call.get("project_id") not in {"", project}
            or not call.get("ended_at")
            or call.get("exception") not in (None, "")
            or call.get("inputs")
            != _expected_conversation_inputs(item, index, seed_digest)
            or call.get("output") != _expected_conversation_output(item, index)
        ):
            drift.append(f"weave-conversation:{coordinate[0]}:{coordinate[1]}:content")
            continue
        tool_calls = [
            child
            for child in by_parent.get(call_id, ())
            if child.get("op_name") == "fugue.qualification.wandb_mcp_tool"
        ]
        actual_tools = sorted(
            (
                {
                    "inputs": child.get("inputs"),
                    "output": child.get("output"),
                }
                for child in tool_calls
                if isinstance(child.get("inputs"), Mapping)
            ),
            key=lambda value: str(value["inputs"].get("tool_name") or ""),
        )
        expected_inputs = sorted(
            _expected_tool_inputs(item, index),
            key=lambda value: str(value["tool_name"]),
        )
        expected_tools = [
            {
                "inputs": value,
                "output": {
                    "tool_name": value["tool_name"],
                    "run_id": value["run_id"],
                    "result": value["public_result"],
                },
            }
            for value in expected_inputs
        ]
        if (
            len(tool_calls) != 2
            or actual_tools != expected_tools
            or any(
                child.get("project_id") not in {"", project}
                or child.get("trace_id") != call.get("trace_id")
                or not child.get("ended_at")
                or child.get("exception") not in (None, "")
                for child in tool_calls
            )
        ):
            drift.append(f"weave-conversation:{coordinate[0]}:{coordinate[1]}:tools")
            continue
        ids = sorted(str(child["id"]) for child in tool_calls)
        tool_ids.update(ids)
        content = {
            "inputs": call["inputs"],
            "output": call["output"],
            "tools": actual_tools,
        }
        receipts.append(
            {
                "run_id": coordinate[0],
                "conversation_index": coordinate[1],
                "call_id": call_id,
                "ref": f"weave:///{project}/call/{call_id}",
                "tool_call_ids": ids,
                "tool_span_count": len(ids),
                "content_digest": stable_digest(content),
            }
        )
    for call in calls:
        if (
            call.get("op_name") == "fugue.qualification.wandb_mcp_tool"
            and str(call.get("id") or "") not in tool_ids
        ):
            extras.append(f"weave-tool-call:{call.get('id')}")
    return (
        sorted(
            receipts,
            key=lambda item: (
                str(item["run_id"]),
                int(item["conversation_index"]),
            ),
        ),
        sorted(extras),
        sorted(drift),
    )


def _new_evaluation(
    dataset: weave.Dataset,
    revision: str,
    *,
    seed_digest: str,
) -> weave.Evaluation:
    return weave.Evaluation(
        name=f"mcp-release-{revision}",
        evaluation_name=f"MCP release {revision}",
        description=(
            "Aligned deterministic maintenance evidence. This measures the "
            "seeded maintainer workflow, not the Fugue MCP comparison."
        ),
        dataset=dataset,
        scorers=[_evidence_alignment_scorer],
        metadata={
            "fugue_seed_digest": seed_digest,
            "revision": revision,
        },
    )


def _inspect_evaluation_objects(
    client: Any,
    project: str,
    *,
    dataset_ref: str | None,
    seed_digest: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    values: dict[str, dict[str, Any]] = {}
    drift = []
    for revision in ("maintainer-r17", "maintainer-r18"):
        object_id = f"mcp-release-{revision}"
        evaluation = _raw_weave_object_or_none(
            client,
            project=project,
            object_id=object_id,
            digest="qualification-v1",
        )
        if evaluation is None:
            continue
        value = getattr(evaluation, "val", None)
        if not isinstance(value, Mapping):
            drift.append(f"weave-evaluation-object:{revision}:shape")
            continue
        metadata = dict(value.get("metadata") or {})
        bound_ref = str(value.get("dataset") or "")
        scorers = value.get("scorers")
        expected_scorer_prefix = (
            f"weave:///{project}/op/fugue.qualification.evidence_alignment_scorer:"
        )
        if (
            value.get("name") != object_id
            or metadata.get("fugue_seed_digest") != seed_digest
            or metadata.get("revision") != revision
            or not dataset_ref
            or bound_ref != dataset_ref
            or not isinstance(scorers, list)
            or len(scorers) != 1
            or not isinstance(scorers[0], str)
            or not scorers[0].startswith(expected_scorer_prefix)
        ):
            drift.append(f"weave-evaluation-object:{revision}:content")
            continue
        object_digest = str(getattr(evaluation, "digest", "") or "")
        if not object_digest:
            drift.append(f"weave-evaluation-object:{revision}:digest")
            continue
        ref = f"weave:///{project}/object/{object_id}:{object_digest}"
        content = {
            "name": object_id,
            "metadata": metadata,
            "dataset_ref": bound_ref,
            "scorer_ref": scorers[0],
        }
        values[revision] = {
            "revision": revision,
            "ref": ref,
            "content_digest": stable_digest(content),
        }
    return values, sorted(drift)


def _inventory_weave_object_versions(
    client: Any,
    project: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    from weave.trace_server.trace_server_interface import (
        ObjectVersionFilter,
        ObjQueryReq,
    )

    expected_objects = {
        object_id: count
        for (kind, object_id), count in (
            _EXPECTED_WEAVE_OBJECT_VERSION_COUNTS.items()
        )
        if kind == "object"
    }
    expected_ops = {
        object_id: count
        for (kind, object_id), count in (
            _EXPECTED_WEAVE_OBJECT_VERSION_COUNTS.items()
        )
        if kind == "op"
    }
    rows: list[dict[str, Any]] = []
    extras: list[str] = []
    drift: list[str] = []
    for is_op, expected in (
        (False, expected_objects),
        (True, expected_ops),
    ):
        response = client.server.objs_query(
            ObjQueryReq(
                project_id=project,
                filter=ObjectVersionFilter(
                    is_op=is_op,
                    latest_only=False,
                ),
                limit=1_000,
                metadata_only=True,
                include_tags_and_aliases=True,
            )
        )
        counts: dict[str, int] = {}
        for item in response.objs:
            object_id = str(getattr(item, "object_id", "") or "")
            digest = str(getattr(item, "digest", "") or "")
            counts[object_id] = counts.get(object_id, 0) + 1
            if object_id not in expected:
                extras.append(
                    f"weave-{'op' if is_op else 'object'}:{object_id}"
                )
                continue
            if not digest:
                drift.append(
                    f"weave-{'op' if is_op else 'object'}:{object_id}:digest"
                )
                continue
            metadata = {
                "kind": "op" if is_op else "object",
                "object_id": object_id,
                "digest": digest,
                "base_object_class": str(
                    getattr(item, "base_object_class", "") or ""
                ),
                "aliases": sorted(
                    str(alias)
                    for alias in getattr(item, "aliases", None) or ()
                ),
            }
            rows.append(
                {
                    **metadata,
                    "content_digest": stable_digest(metadata),
                }
            )
        if any(
            counts.get(object_id, 0) > expected_count
            for object_id, expected_count in expected.items()
        ):
            drift.append(
                f"weave-{'op' if is_op else 'object'}-version-set"
            )
    identities = {
        (item["kind"], item["object_id"], item["digest"]) for item in rows
    }
    if len(identities) != len(rows):
        drift.append("weave-object-version-duplicate")
    return (
        sorted(
            rows,
            key=lambda item: (
                str(item["kind"]),
                str(item["object_id"]),
                str(item["digest"]),
            ),
        ),
        sorted(extras),
        sorted(drift),
    )


def _inventory_evaluation_calls(
    calls: Sequence[Mapping[str, Any]],
    *,
    project: str,
    seed_digest: str,
    evaluation_objects: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for call in calls:
        by_parent.setdefault(str(call.get("parent_id") or ""), []).append(call)
    roots: dict[str, Mapping[str, Any]] = {}
    extras: list[str] = []
    drift: list[str] = []
    for call in calls:
        if call.get("op_name") != "Evaluation.evaluate":
            continue
        attributes = call.get("attributes")
        if not isinstance(attributes, Mapping):
            extras.append(f"weave-evaluation-root:{call.get('id')}:unattributed")
            continue
        revision = str(attributes.get("fugue.evaluation_revision") or "")
        if attributes.get("fugue.seed_digest") != seed_digest or revision not in {
            "maintainer-r17",
            "maintainer-r18",
        }:
            extras.append(f"weave-evaluation-root:{call.get('id')}:other-seed")
            continue
        if revision in roots:
            extras.append(f"weave-evaluation-root:{revision}:duplicate")
            continue
        roots[revision] = call
    receipts = []
    for revision, root in roots.items():
        if revision not in evaluation_objects:
            drift.append(f"weave-evaluation-root:{revision}:object-missing")
            continue
        root_id = str(root.get("id") or "")
        children = list(by_parent.get(root_id, ()))
        operations = [str(child.get("op_name") or "") for child in children]
        prediction_count = operations.count("Evaluation.predict_and_score")
        summary_count = operations.count("Evaluation.summarize")
        expected_model_operation = (
            "fugue.qualification.baseline_evidence_model"
            if revision == "maintainer-r17"
            else "fugue.qualification.candidate_evidence_model"
        )
        descendant_ids: set[str] = set()
        nested_structure_valid = True
        for child in children:
            child_id = str(child.get("id") or "")
            if not child_id:
                nested_structure_valid = False
                continue
            descendant_ids.add(child_id)
            nested = list(by_parent.get(child_id, ()))
            if child.get("op_name") == "Evaluation.predict_and_score":
                if sorted(
                    str(item.get("op_name") or "") for item in nested
                ) != sorted(
                    (
                        expected_model_operation,
                        "fugue.qualification.evidence_alignment_scorer",
                    )
                ):
                    nested_structure_valid = False
            elif nested:
                nested_structure_valid = False
            for item in nested:
                nested_id = str(item.get("id") or "")
                if (
                    not nested_id
                    or item.get("project_id") not in {"", project}
                    or item.get("trace_id") != root.get("trace_id")
                    or not item.get("ended_at")
                    or item.get("exception") not in (None, "")
                ):
                    nested_structure_valid = False
                else:
                    descendant_ids.add(nested_id)
        if (
            root.get("project_id") not in {"", project}
            or not root.get("ended_at")
            or root.get("exception") not in (None, "")
            or prediction_count != len(_EVALUATION_ROWS)
            or summary_count != 1
            or len(children) != len(_EVALUATION_ROWS) + 1
            or len(descendant_ids) != 25
            or not nested_structure_valid
            or any(
                child.get("project_id") not in {"", project}
                or child.get("trace_id") != root.get("trace_id")
                or not child.get("ended_at")
                or child.get("exception") not in (None, "")
                for child in children
            )
        ):
            drift.append(f"weave-evaluation-root:{revision}:structure")
            continue
        structure = {
            "root_output": root.get("output"),
            "child_operations": sorted(operations),
            "child_ids": sorted(str(child.get("id") or "") for child in children),
        }
        receipts.append(
            {
                "revision": revision,
                "ref": str(evaluation_objects[revision]["ref"]),
                "call_id": root_id,
                "call_ref": f"weave:///{project}/call/{root_id}",
                "summary_digest": stable_digest(root.get("output")),
                "prediction_rows": prediction_count,
                "direct_children": len(children),
                "summarize_children": summary_count,
                "call_ids": sorted({root_id, *descendant_ids}),
                "content_digest": stable_digest(structure),
            }
        )
    return (
        sorted(receipts, key=lambda item: str(item["revision"])),
        sorted(extras),
        sorted(drift),
    )


def _inventory_weave_evidence(
    project: str,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[str],
]:
    seed_digest = qualification_seed_digest(source_project=project)
    try:
        with _source_weave_client(project, write=False) as client:
            if not _source_weave_project_exists(client, project):
                return None, [], {}, [], [], [], []
            calls = [_call_record(call) for call in client.get_calls(limit=None)]
            dataset = _inspect_weave_dataset(client, project)
            conversations, conversation_extras, conversation_drift = (
                _inventory_conversations(
                    calls,
                    project=project,
                    seed_digest=seed_digest,
                )
            )
            evaluation_objects, object_drift = _inspect_evaluation_objects(
                client,
                project,
                dataset_ref=(str(dataset["ref"]) if dataset is not None else None),
                seed_digest=seed_digest,
            )
            evaluations, evaluation_extras, evaluation_drift = (
                _inventory_evaluation_calls(
                    calls,
                    project=project,
                    seed_digest=seed_digest,
                    evaluation_objects=evaluation_objects,
                )
            )
            object_versions, object_extras, object_version_drift = (
                _inventory_weave_object_versions(client, project)
            )
            expected_call_ids = {
                *(
                    str(item.get("call_id") or "")
                    for item in conversations
                ),
                *(
                    str(call_id)
                    for item in conversations
                    for call_id in item.get("tool_call_ids") or ()
                ),
                *(
                    str(call_id)
                    for item in evaluations
                    for call_id in item.get("call_ids") or ()
                ),
            }
            observed_call_ids = [
                str(item.get("id") or "") for item in calls
            ]
            call_extras = [
                f"weave-call:{call_id or '<missing-id>'}"
                for call_id in observed_call_ids
                if not call_id or call_id not in expected_call_ids
            ]
            if len(observed_call_ids) != len(set(observed_call_ids)):
                call_extras.append("weave-call:duplicate-id")
            missing_call_ids = expected_call_ids - set(observed_call_ids)
            call_drift = [
                f"weave-call-set:missing-{len(missing_call_ids)}"
            ] if missing_call_ids else []
    except Exception as exc:
        if _weave_missing_error(exc):
            return None, [], {}, [], [], [], []
        raise
    return (
        dataset,
        conversations,
        evaluation_objects,
        evaluations,
        object_versions,
        sorted(
            {
                *conversation_extras,
                *evaluation_extras,
                *object_extras,
                *call_extras,
            }
        ),
        sorted(
            {
                *conversation_drift,
                *object_drift,
                *object_version_drift,
                *evaluation_drift,
                *call_drift,
            }
        ),
    )


def _inventory_hosted_source(
    entity: str,
    project: str,
    *,
    source_project: str,
) -> dict[str, Any]:
    runs, run_extras, run_drift = _inventory_wandb_runs(
        entity,
        project,
        source_project=source_project,
    )
    (
        dataset,
        conversations,
        evaluation_objects,
        evaluations,
        object_versions,
        weave_extras,
        weave_drift,
    ) = _inventory_weave_evidence(source_project)
    return _source_inventory(
        source_project=source_project,
        runs=runs,
        dataset=dataset,
        conversations=conversations,
        evaluation_objects=evaluation_objects,
        evaluations=evaluations,
        object_versions=object_versions,
        extra_objects=[*run_extras, *weave_extras],
        drift=[*run_drift, *weave_drift],
    )


def _seed_conversation(
    item: Mapping[str, Any],
    index: int,
    *,
    seed_digest: str,
) -> None:
    _seed_agent_conversation.call(
        run_id=str(item["id"]),
        attempt_label=str(item["attempt_label"]),
        conversation_index=index,
        source_returned=int(item["source_returned"]),
        source_opened=int(item["source_opened"]),
        latency_ms=int(item["latency_ms"]),
        seed_digest=seed_digest,
        __weave={
            "display_name": (f"qualification {item['attempt_label']} source {index}"),
            "attributes": {
                "fugue.seed_digest": seed_digest,
                "fugue.evidence_kind": "source_conversation",
            },
        },
    )


async def _run_evaluation_once(
    evaluation: weave.Evaluation,
    revision: str,
    *,
    seed_digest: str,
) -> None:
    model = (
        _baseline_evidence_model
        if revision == "maintainer-r17"
        else _candidate_evidence_model
    )
    await evaluation.evaluate.call(
        evaluation,
        model,
        __weave={
            "display_name": f"MCP release {revision} qualification",
            "attributes": {
                "fugue.seed_digest": seed_digest,
                "fugue.evaluation_revision": revision,
            },
        },
    )


def _materialize_hosted_source(
    entity: str,
    project: str,
    *,
    source_project: str,
    inventory: Mapping[str, Any],
    run_action: Callable[[str, Callable[[], None]], None],
) -> None:
    present = _expected_source_actions(inventory)
    seed_digest = qualification_seed_digest(source_project=source_project)
    for item in _RUNS:
        action = f"wandb-run:{item['id']}"
        if action not in present:
            run_action(
                action,
                lambda item=item: _create_wandb_run(
                    entity,
                    project,
                    source_project=source_project,
                    item=item,
                ),
            )
    with _source_weave_client(source_project, write=True):
        if "weave-dataset:mcp-release-maintenance-cases" not in present:
            run_action(
                "weave-dataset:mcp-release-maintenance-cases",
                lambda: weave.publish(
                    weave.Dataset(
                        name="mcp-release-maintenance-cases",
                        description=(
                            "Reviewed non-sensitive maintenance cases for "
                            "Fugue MCP release qualification."
                        ),
                        rows=[dict(item) for item in _EVALUATION_ROWS],
                    ),
                    name="mcp-release-maintenance-cases",
                    aliases=["qualification-v1"],
                ),
            )
        dataset = weave.ref(
            f"weave:///{source_project}/object/"
            "mcp-release-maintenance-cases:qualification-v1"
        ).get()
        for item in _RUNS:
            for index in range(1, 5):
                action = f"weave-conversation:{item['id']}:{index}"
                if action not in present:
                    run_action(
                        action,
                        lambda item=item, index=index: _seed_conversation(
                            item,
                            index,
                            seed_digest=seed_digest,
                        ),
                    )
        for revision in ("maintainer-r17", "maintainer-r18"):
            object_action = f"weave-evaluation-object:{revision}"
            if object_action not in present:
                run_action(
                    object_action,
                    lambda revision=revision: weave.publish(
                        _new_evaluation(
                            dataset,
                            revision,
                            seed_digest=seed_digest,
                        ),
                        name=f"mcp-release-{revision}",
                        aliases=["qualification-v1"],
                    ),
                )
            evaluation_ref = weave.ref(
                f"weave:///{source_project}/object/"
                f"mcp-release-{revision}:qualification-v1"
            )
            evaluation = _new_evaluation(
                dataset,
                revision,
                seed_digest=seed_digest,
            )
            evaluation.ref = evaluation_ref
            run_action_name = f"weave-evaluation-run:{revision}"
            if run_action_name not in present:
                run_action(
                    run_action_name,
                    lambda evaluation=evaluation, revision=revision: asyncio.run(
                        _run_evaluation_once(
                            evaluation,
                            revision,
                            seed_digest=seed_digest,
                        )
                    ),
                )


@weave.op(name="fugue.qualification.wandb_mcp_tool")
def _seed_tool_span(
    *,
    tool_name: str,
    run_id: str,
    public_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "run_id": run_id,
        "result": dict(public_result),
    }


@weave.op(name="fugue.qualification.maintenance_agent")
def _seed_agent_conversation(
    *,
    run_id: str,
    attempt_label: str,
    conversation_index: int,
    source_returned: int,
    source_opened: int,
    latency_ms: int,
    seed_digest: str,
) -> dict[str, Any]:
    question = (
        f"Inspect public maintenance evidence for {attempt_label}; source "
        f"segment {conversation_index}."
    )
    with weave.Conversation(
        conversation_id=f"{run_id}-source-{conversation_index}",
        conversation_name="Fugue MCP release qualification evidence",
        agent_name="maintenance-evidence-agent",
        agent_id="maintenance-evidence-agent-v1",
        agent_version="1",
        model="deterministic-reviewed-policy-v1",
        include_content=True,
        continue_parent_trace=True,
        attributes={
            "fugue.seed_digest": seed_digest,
            "fugue.run_id": run_id,
        },
    ) as conversation:
        with conversation.start_turn(
            user_message=question,
            system_instructions=[
                "Inspect evidence, separate observations from causes, and state "
                "when measurements are incomplete."
            ],
        ) as turn:
            with turn.tool(
                name="wandb_projected_run_read",
                arguments=json.dumps(
                    {"run_id": run_id, "fields": ["latency_ms", "source_opened"]}
                ),
                tool_call_id=f"{run_id}-projected-{conversation_index}",
            ) as tool:
                projected = _seed_tool_span(
                    tool_name="wandb_projected_run_read",
                    run_id=run_id,
                    public_result={
                        "latency_ms": latency_ms,
                        "source_returned": source_returned,
                        "source_opened": source_opened,
                    },
                )
                tool.result = projected
            with turn.tool(
                name="weave_call_evidence_read",
                arguments=json.dumps(
                    {
                        "run_id": run_id,
                        "conversation_index": conversation_index,
                    }
                ),
                tool_call_id=f"{run_id}-weave-{conversation_index}",
            ) as tool:
                traced = _seed_tool_span(
                    tool_name="weave_call_evidence_read",
                    run_id=run_id,
                    public_result={
                        "evidence_status": "observed",
                        "conversation_index": conversation_index,
                    },
                )
                tool.result = traced
            turn.record(
                messages=[
                    *turn.messages,
                    weave.Message(
                        role="assistant",
                        content=(
                            f"Observed latency {latency_ms} ms; opened "
                            f"{source_opened} of {source_returned} returned sources."
                        ),
                    ),
                ]
            )
    return {
        "run_id": run_id,
        "attempt_label": attempt_label,
        "conversation_index": conversation_index,
        "evidence_status": "observed",
    }


@weave.op(name="fugue.qualification.baseline_evidence_model")
def _baseline_evidence_model(
    *,
    case_id: str,
    question: str,
    baseline_pass: bool,
    candidate_pass: bool,
) -> dict[str, Any]:
    del question, candidate_pass
    return {
        "case_id": case_id,
        "revision": "maintainer-r17",
        "pass": baseline_pass,
    }


@weave.op(name="fugue.qualification.candidate_evidence_model")
def _candidate_evidence_model(
    *,
    case_id: str,
    question: str,
    baseline_pass: bool,
    candidate_pass: bool,
) -> dict[str, Any]:
    del question, baseline_pass
    return {
        "case_id": case_id,
        "revision": "maintainer-r18",
        "pass": candidate_pass,
    }


@weave.op(name="fugue.qualification.evidence_alignment_scorer")
def _evidence_alignment_scorer(
    *,
    output: Mapping[str, Any],
    baseline_pass: bool,
    candidate_pass: bool,
) -> dict[str, Any]:
    expected = (
        baseline_pass if output.get("revision") == "maintainer-r17" else candidate_pass
    )
    return {
        "deterministic_pass": bool(output.get("pass")),
        "aligned_with_reviewed_row": output.get("pass") is expected,
    }


def _derived_evidence_counts(
    runs: Sequence[Mapping[str, Any]],
    weave_receipts: Mapping[str, Any],
) -> dict[str, int]:
    dataset = weave_receipts.get("dataset")
    conversations = weave_receipts.get("source_conversations")
    evaluations = weave_receipts.get("evaluations")
    if (
        not isinstance(dataset, Mapping)
        or not isinstance(conversations, Sequence)
        or isinstance(conversations, str | bytes)
        or not isinstance(evaluations, Sequence)
        or isinstance(evaluations, str | bytes)
    ):
        raise ValueError("evidence receipts are incomplete")
    dataset_rows = int(dataset.get("rows") or 0)
    return {
        "runs": len(runs),
        "source_conversations": len(conversations),
        "tool_spans": sum(
            int(item.get("tool_span_count") or 2)
            for item in conversations
            if isinstance(item, Mapping)
        ),
        "dataset_rows": dataset_rows,
        "aligned_evaluation_pairs": dataset_rows,
        "evaluation_prediction_rows": sum(
            int(item.get("prediction_rows") or 0)
            for item in evaluations
            if isinstance(item, Mapping)
        ),
    }


def _evidence_lock(
    project: str,
    runs: Sequence[Mapping[str, Any]],
    weave_receipts: Mapping[str, Any],
    *,
    result_project: str | None = None,
    created_at: str | None = None,
    preparation_id: str | None = None,
    source_inventory_digest: str | None = None,
) -> dict[str, Any]:
    resolved_result_project = result_project or project
    seed_digest = qualification_seed_digest(source_project=project)
    objects = {
        "runs": [dict(item) for item in runs],
        "dataset": dict(weave_receipts["dataset"]),
        "source_conversations": [
            dict(item) for item in weave_receipts["source_conversations"]
        ],
        "evaluations": [dict(item) for item in weave_receipts["evaluations"]],
        "weave_object_versions": [
            dict(item)
            for item in weave_receipts.get("weave_object_versions") or ()
        ],
    }
    rich_source_lock = source_inventory_digest is not None
    counts = (
        _derived_evidence_counts(runs, weave_receipts)
        if rich_source_lock
        else dict(_REQUIRED_COUNTS)
    )
    unsigned = {
        "schema_version": EVIDENCE_LOCK_SCHEMA_VERSION,
        "project": project,
        **(
            {
                "source_project": project,
                "result_project": resolved_result_project,
            }
            if resolved_result_project != project
            else {}
        ),
        "seed_digest": seed_digest,
        "created_at": created_at
        or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "counts": counts,
        "objects": objects,
        **(
            {
                "preparation_id": preparation_id,
                "source_inventory_digest": source_inventory_digest,
                "source_snapshot_digest": stable_digest(objects),
            }
            if rich_source_lock
            else {}
        ),
        "facts_digest": stable_digest(
            qualification_seed(source_project=project)["facts"]
        ),
        "supported_scope": (
            "Non-sensitive hosted qualification evidence only; not customer "
            "data and not a release-wide product claim."
        ),
        "evidence_lock_digest": "",
    }
    return {
        **unsigned,
        "evidence_lock_digest": stable_digest(unsigned),
    }


@contextmanager
def _temporary_environment(values: Mapping[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, prior in previous.items():
            if prior is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior
