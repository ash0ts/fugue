from __future__ import annotations

import asyncio
import json
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import weave

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.operator import load_env
from fugue.model_plane import trace_api_key
from fugue.weave_support import WEAVE_AGENTS_BASE_URL

QUALIFICATION_RESULT_PROJECT = (
    "wandb/fugue-mcp-release-qualification-v1"
)
QUALIFICATION_SOURCE_PROJECT = "wandb/fugue-mcp-release-source-v1"
# Backward-compatible name for the original single-project contract.
QUALIFICATION_PROJECT = QUALIFICATION_RESULT_PROJECT
MCP_RELEASE_NOTES_LOCK = Path(
    "examples/comparisons/wandb-mcp-maintenance/release-notes.lock.json"
)
_MCP_RELEASE_NOTES_COMMIT = "3dd4447ef0054d4707aafc515e3f2ddfb11b17bd"
_MCP_RELEASE_NOTES_SHA256 = (
    "2e32e337dd6c98a5e4b3805b189af10c913ec1dd739a63b25b031ab35d786c99"
)
_MCP_RELEASE_NOTES_BYTES = 7838
EVIDENCE_LOCK_SCHEMA_VERSION = 1
_REQUIRED_COUNTS = {
    "runs": 6,
    "source_conversations": 24,
    "tool_spans": 48,
    "dataset_rows": 8,
    "aligned_evaluation_pairs": 8,
    "evaluation_prediction_rows": 16,
}
SOURCE_CONFORMANCE_SCHEMA_VERSION = 1
_SOURCE_CONFORMANCE_EXPECTATIONS = {
    "evaluation_roots": 2,
    "direct_children": 18,
    "predict_and_score_children": 16,
    "summarize_children": 2,
    "repaired_candidate_prediction_children": 16,
}
_MCP_RELEASE_CANDIDATES = (
    (
        "wandb-mcp-main",
        "git:53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0",
    ),
    (
        "wandb-mcp-0-4-staging",
        "git:3dd4447ef0054d4707aafc515e3f2ddfb11b17bd",
    ),
)

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
        str(item["case_id"]): bool(item["baseline_pass"])
        for item in _EVALUATION_ROWS
    }
    candidate_passes = {
        str(item["case_id"]): bool(item["candidate_pass"])
        for item in _EVALUATION_ROWS
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
            raise ValueError(
                f"evidence lock count {name} must equal {expected}"
            )


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
    source_prefix = f"weave:///{source_project}/"
    if (
        not isinstance(dataset, Mapping)
        or not str(dataset.get("ref") or "").startswith(source_prefix)
    ):
        raise ValueError("evidence lock Dataset reference is not immutable")
    for run in runs:
        if (
            not isinstance(run, Mapping)
            or not str(run.get("ref") or "").startswith(
                f"wandb-run:///{source_project}/"
            )
        ):
            raise ValueError("evidence lock W&B Run reference is not immutable")
    for conversation in conversations:
        if (
            not isinstance(conversation, Mapping)
            or not str(conversation.get("ref") or "").startswith(source_prefix)
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


def qualify_locked_mcp_revisions(
    *,
    repo_root: Path,
    evidence_lock: Path,
    env_file: Path,
    output: Path,
    source_project: str | None = None,
    result_project: str | None = None,
) -> dict[str, Any]:
    """Exercise both exact MCP locks against the immutable hosted evidence.

    This is a bounded infrastructure/mechanism qualification, not an Agent
    outcome study. It deliberately avoids the blind judge and never treats a
    successful tool RPC as proof of task correctness.
    """

    root = repo_root.resolve()
    raw_lock = json.loads(evidence_lock.resolve().read_text(encoding="utf-8"))
    resolved_source_project = (
        source_project or evidence_source_project(raw_lock)
    )
    resolved_result_project = (
        result_project or evidence_result_project(raw_lock)
    )
    lock = validate_evidence_lock(
        raw_lock,
        expected_project=None,
        expected_source_project=resolved_source_project,
        expected_result_project=resolved_result_project,
    )
    release_notes = validate_release_notes_lock(
        json.loads((root / MCP_RELEASE_NOTES_LOCK).read_text(encoding="utf-8"))
    )
    env = load_env(env_file.resolve())
    if not str(env.get("WANDB_API_KEY") or "").strip():
        raise RuntimeError("MCP qualification requires WANDB_API_KEY")
    runtime_env = {
        "WANDB_API_KEY": str(env["WANDB_API_KEY"]),
        "WANDB_BASE_URL": str(env.get("WANDB_BASE_URL") or "https://api.wandb.ai"),
    }
    observations = asyncio.run(
        _run_mcp_release_observations(root, lock, runtime_env)
    )
    receipt = _mcp_release_qualification_receipt(
        lock,
        observations,
        release_notes=release_notes,
    )
    serialized = json.dumps(receipt, sort_keys=True)
    for secret in runtime_env.values():
        if secret and secret in serialized:
            raise RuntimeError("MCP qualification receipt contains a credential")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output.resolve(), receipt)
    return receipt


async def _run_mcp_release_observations(
    repo_root: Path,
    evidence_lock: Mapping[str, Any],
    runtime_env: Mapping[str, str],
) -> list[dict[str, Any]]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise RuntimeError(
            "MCP qualification requires `uv sync --extra research-worker`"
        ) from exc
    from fugue.bench.component_imports import _managed_python_probe_command

    project = evidence_source_project(evidence_lock)
    entity, project_name = project.split("/", 1)
    evaluations = list(evidence_lock["objects"]["evaluations"])
    observations: list[dict[str, Any]] = []
    for import_id, version_identity in _MCP_RELEASE_CANDIDATES:
        path = (
            repo_root
            / ".fugue"
            / "imports"
            / "mcp"
            / "locks"
            / f"{import_id}.json"
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("id") != import_id or raw.get("version_identity") != version_identity:
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
                    _successful_mcp_value(
                        calls["count_evaluation_roots_tool"]
                    ).get("root_traces_count")
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
        if import_id == "wandb-mcp-0-4-staging":
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
                "locked_tools": sorted(
                    str(item.get("name") or "")
                    for item in raw.get("tool_manifest", [])
                    if isinstance(item, Mapping) and item.get("name")
                ),
                "release_capabilities": _release_capabilities(
                    raw.get("tool_manifest")
                ),
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
    mutation_probe: bool = False,
) -> dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from fugue.bench.component_imports import _managed_python_probe_command

    fixed_env = dict(lock["fixed_env"])
    fixed_env.update(overrides)
    command = _managed_python_probe_command(
        runtime_source,
        runtime_platform=str(lock["runtime_platform"]),
        required_env=tuple(lock["required_env"]),
        fixed_env=tuple(sorted(fixed_env.items())),
        allowed_hosts=tuple(lock["allowed_hosts"]),
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
            if mutation_probe:
                mutation = await _call_mcp_json(
                    session,
                    "query_wandb_graphql_tool",
                    {
                        "query": (
                            "mutation FugueQualificationMutation "
                            "{ __typename }"
                        )
                    },
                )
    return {
        "overrides": dict(sorted(overrides.items())),
        "initialized_tools": names,
        "tool_manifest_digest": stable_digest(names),
        "mutation_probe": mutation,
    }


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
        str(value.get("error") or "")
        if isinstance(value, Mapping)
        else ""
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
) -> dict[str, Any]:
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
                dict(child.get("metadata") or {})
                if isinstance(child, Mapping)
                else {}
            )
            op_distribution = dict(metadata.get("op_distribution") or {})
            observed_predictions = int(
                op_distribution.get("Evaluation.predict_and_score") or 0
            )
            observed_summaries = int(
                op_distribution.get("Evaluation.summarize") or 0
            )
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
            "initialized_manifest_matches_lock": (
                list(raw.get("initialized_tools") or [])
                == list(raw.get("locked_tools") or [])
            ),
            "release_capabilities": dict(
                raw.get("release_capabilities") or {}
            ),
            "tool_calls_ok": tool_calls_ok,
            "child_queries_ok": child_queries_ok,
            "root_trace_count": count.get("root_traces_count"),
            "total_trace_count": count.get("total_count"),
            "project_run_count": probe.get("run_count"),
            "project_state_counts": probe.get("state_counts"),
            "project_probe_matches_lock": probe.get("run_count") == expected_runs,
            "evaluation_root_count": evaluation_roots.get(
                "root_traces_count"
            ),
            "summary_project_exhaustive": summary.get(
                "project_exhaustive"
            ),
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
    baseline = by_id.get("wandb-mcp-main", {})
    candidate = by_id.get("wandb-mcp-0-4-staging", {})
    baseline_reconciliations = list(
        baseline.get("evaluation_reconciliation", [])
    )
    baseline_reconciled = bool(baseline_reconciliations) and all(
        item["prediction_rows_reconciled"]
        for item in baseline_reconciliations
    )
    candidate_reconciliations = list(
        candidate.get("evaluation_reconciliation", [])
    )
    candidate_reconciled = bool(candidate_reconciliations) and all(
        item["prediction_rows_reconciled"]
        for item in candidate_reconciliations
    )
    release_note_classification = _release_note_classification(
        baseline,
        candidate,
    )
    infrastructure_conformance = _infrastructure_conformance(candidate)
    release_notes_lock = validate_release_notes_lock(
        release_notes or _default_release_notes_lock()
    )
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
        "release_notes_lock": release_notes_lock,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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
    default_manifest = (
        "passed"
        if candidate.get("initialized_manifest_matches_lock") is True
        else "failed"
    )
    default_tools = set(candidate.get("initialized_tools") or ())
    profiles = dict(candidate.get("profile_probes") or {})
    read_only = dict(profiles.get("read_only") or {})
    raw_graphql = dict(profiles.get("raw_graphql") or {})
    read_only_tools = set(read_only.get("initialized_tools") or ())
    raw_graphql_tools = set(raw_graphql.get("initialized_tools") or ())
    write_tools = {"create_wandb_report_tool", "log_analysis_to_wandb"}
    read_only_status = (
        "passed"
        if (
            read_only_tools
            and read_only_tools == default_tools - write_tools
            and not read_only_tools & write_tools
        )
        else "failed"
        if read_only_tools
        else "unavailable"
    )
    raw_graphql_status = (
        "passed"
        if (
            raw_graphql_tools
            and raw_graphql_tools
            == default_tools | {"query_wandb_graphql_tool"}
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
            "id": "default-tool-manifest",
            "status": default_manifest,
            "evidence": "initialized MCP manifest versus exact lock",
        },
        {
            "id": "read-only-tool-manifest",
            "status": read_only_status,
            "evidence": (
                "exact candidate runtime initialized with "
                "WANDB_MCP_READ_ONLY=true"
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
    unavailable = [
        item["id"] for item in gates if item["status"] == "unavailable"
    ]
    return {
        "complete": not failed and not unavailable,
        "failed": failed,
        "unavailable": unavailable,
        "gates": gates,
        "claim_scope": (
            "Infrastructure conformance only; these gates are not Agent task scores."
        ),
    }


def validate_release_notes_lock(raw: Mapping[str, Any]) -> dict[str, Any]:
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
    if (
        raw.get("schema_version") != 1
        or raw.get("repository") != "wandb/wandb-mcp-server"
        or raw.get("commit") != _MCP_RELEASE_NOTES_COMMIT
        or raw.get("path") != "docs/releases/v0.4.0.md"
        or raw.get("sha256") != _MCP_RELEASE_NOTES_SHA256
        or raw.get("bytes") != _MCP_RELEASE_NOTES_BYTES
        or raw.get("status") != "release_candidate"
        or not isinstance(behaviors, list)
        or not behaviors
        or len(behaviors) != len(set(behaviors))
        or any(not isinstance(item, str) or not item for item in behaviors)
    ):
        raise ValueError("release-notes lock does not bind the exact 0.4 RC source")
    expected_url = (
        "https://raw.githubusercontent.com/wandb/wandb-mcp-server/"
        f"{_MCP_RELEASE_NOTES_COMMIT}/docs/releases/v0.4.0.md"
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
            str(item["release_note"])
            for item in _release_note_classification({}, {})
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
        query_schema.get("properties")
        if isinstance(query_schema, Mapping)
        else {}
    )
    query_properties = (
        query_properties if isinstance(query_properties, Mapping) else {}
    )
    history = tools.get("get_run_history_tool") or {}
    history_schema = history.get("input_schema") or {}
    history_properties = (
        history_schema.get("properties")
        if isinstance(history_schema, Mapping)
        else {}
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
        "bounded_history_keys": "keys" in history_properties,
        "bounded_history_range": any(
            name in history_properties
            for name in ("min_step", "max_step", "min_x", "max_x")
        ),
        "raw_graphql_registered_by_default": (
            "query_wandb_graphql_tool" in names
        ),
        "write_tools_registered_by_default": bool(
            {"create_wandb_report_tool", "log_analysis_to_wandb"} & names
        ),
    }


def _release_note_classification(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
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
            "bounded_history_range",
            "agent-and-mechanism",
        ),
        (
            "raw-graphql-disabled-by-default",
            "raw_graphql_registered_by_default",
            "infrastructure",
        ),
    )
    classified: list[dict[str, str]] = []
    for release_note, capability, evidence_kind in contracts:
        baseline_value = bool(baseline_capabilities.get(capability))
        candidate_value = bool(candidate_capabilities.get(capability))
        if capability == "raw_graphql_registered_by_default":
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
                        for item in candidate.get(
                            "evaluation_reconciliation", ()
                        )
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
        )
    )
    return classified


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
    trace_base_url = str(
        env.get("FUGUE_WEAVE_TRACE_SERVER_URL")
        or env.get("WF_TRACE_SERVER_URL")
        or WEAVE_AGENTS_BASE_URL
    ).rstrip("/")
    roots, children = _fetch_hosted_source_calls(
        source_project=source_project,
        trace_base_url=trace_base_url,
        api_key=api_key,
        evaluation_call_ids=[
            str(item["call_id"]) for item in lock["objects"]["evaluations"]
        ],
    )
    receipt = build_hosted_source_conformance_receipt(
        evidence_lock=lock,
        evaluation_roots=roots,
        direct_children=children,
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

        children = [
            dict(item) for item in direct_children.get(call_id, ())
        ]
        child_ids = [
            str(item.get("id") or item.get("call_id") or "")
            for item in children
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

        predictions = int(
            operation_counts.get("Evaluation.predict_and_score") or 0
        )
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
            return [
                item for item in decoded["calls"] if isinstance(item, dict)
            ]
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


def prepare_hosted_project(
    *,
    project: str | None = None,
    source_project: str | None = None,
    result_project: str | None = None,
    output: Path,
    env_file: Path,
) -> dict[str, Any]:
    if project and source_project and project != source_project:
        raise ValueError("legacy project and source project disagree")
    resolved_source_project = (
        project or source_project or QUALIFICATION_SOURCE_PROJECT
    )
    resolved_result_project = (
        result_project
        or (project if project is not None else QUALIFICATION_RESULT_PROJECT)
    )
    supported_routes = {
        (QUALIFICATION_PROJECT, QUALIFICATION_PROJECT),
        (QUALIFICATION_SOURCE_PROJECT, QUALIFICATION_RESULT_PROJECT),
    }
    if (resolved_source_project, resolved_result_project) not in supported_routes:
        raise ValueError(
            "qualification route must use the legacy single project or the "
            "dedicated immutable source and result projects"
        )
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        return validate_evidence_lock(
            existing,
            expected_project=None,
            expected_source_project=resolved_source_project,
            expected_result_project=resolved_result_project,
        )
    env = load_env(env_file)
    required = ("WANDB_API_KEY",)
    missing = [name for name in required if not str(env.get(name) or "").strip()]
    if missing:
        raise RuntimeError(
            "hosted-project preparation requires: " + ", ".join(missing)
        )
    entity, project_name = resolved_source_project.split("/", 1)
    selected_env = {
        "WANDB_API_KEY": str(env["WANDB_API_KEY"]),
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project_name,
        "WANDB_BASE_URL": str(
            env.get("WANDB_BASE_URL") or "https://api.wandb.ai"
        ),
        "WANDB_SILENT": "true",
    }
    with _temporary_environment(selected_env):
        run_receipts = _prepare_wandb_runs(
            entity,
            project_name,
            source_project=resolved_source_project,
        )
        weave_receipts = _prepare_weave_evidence(resolved_source_project)
    value = _evidence_lock(
        resolved_source_project,
        run_receipts,
        weave_receipts,
        result_project=resolved_result_project,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, value)
    return validate_evidence_lock(
        value,
        expected_project=None,
        expected_source_project=resolved_source_project,
        expected_result_project=resolved_result_project,
    )


def _prepare_wandb_runs(
    entity: str,
    project: str,
    *,
    source_project: str,
) -> list[dict[str, Any]]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "install fugue[research-worker] with the W&B Sandbox extra"
        ) from exc
    api = wandb.Api()
    receipts: list[dict[str, Any]] = []
    seed_digest = qualification_seed_digest(source_project=source_project)
    for item in _RUNS:
        run_path = f"{entity}/{project}/{item['id']}"
        try:
            existing = api.run(run_path)
        except (wandb.errors.CommError, ValueError):
            existing = None
        if existing is not None:
            if existing.config.get("fugue_seed_digest") != seed_digest:
                raise RuntimeError(
                    f"existing W&B Run has another seed identity: {run_path}"
                )
            receipts.append(
                {
                    "id": item["id"],
                    "name": existing.name,
                    "url": existing.url,
                    "ref": f"wandb-run:///{run_path}",
                    "seed_digest": seed_digest,
                    "state": existing.state,
                }
            )
            continue
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
            config={
                "fugue_seed_digest": seed_digest,
                "candidate_revision": item["candidate_revision"],
                "attempt_label": item["attempt_label"],
                "evidence_snapshot": "qualification-v1",
                "contains_sensitive_data": False,
            },
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
            url = run.url
        finally:
            run.finish()
        receipts.append(
            {
                "id": item["id"],
                "name": item["attempt_label"],
                "url": url,
                "ref": f"wandb-run:///{run_path}",
                "seed_digest": seed_digest,
                "state": "finished",
                "artifact": artifact_receipt,
            }
        )
    return receipts


def _log_run_history(run: Any, item: Mapping[str, Any]) -> None:
    latency = int(item["latency_ms"])
    for step, multiplier in enumerate((0.7, 0.85, 1.0), start=1):
        payload: dict[str, Any] = {
            "step": step,
            "latency_ms": round(latency * multiplier, 3),
            "deterministic_pass": int(bool(item["deterministic_pass"])),
            "projected_reads": int(item["projected_reads"]),
            "broad_reads": int(item["broad_reads"]),
            "source_returned": int(item["source_returned"]),
            "source_opened": int(item["source_opened"]),
        }
        if item["observed_cost_usd"] is not None:
            payload["observed_cost_usd"] = round(
                float(item["observed_cost_usd"]) * multiplier,
                6,
            )
        run.log(payload, step=step)


def _log_run_artifact(
    run: Any,
    item: Mapping[str, Any],
    seed_digest: str,
) -> dict[str, Any]:
    import wandb

    payload = {
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


def _prepare_weave_evidence(project: str) -> dict[str, Any]:
    client = weave.init(project)
    seed_digest = qualification_seed_digest(source_project=project)
    dataset = weave.Dataset(
        name="mcp-release-maintenance-cases",
        description=(
            "Reviewed non-sensitive maintenance cases for Fugue MCP release "
            "qualification."
        ),
        rows=[dict(item) for item in _EVALUATION_ROWS],
    )
    dataset_ref = weave.publish(
        dataset,
        name="mcp-release-maintenance-cases",
        aliases=["qualification-v1"],
    )
    conversations = _existing_source_conversations(
        client,
        project,
        seed_digest=seed_digest,
    )
    if not conversations:
        for run in _RUNS:
            for index in range(1, 5):
                _, call = _seed_agent_conversation.call(
                    run_id=str(run["id"]),
                    attempt_label=str(run["attempt_label"]),
                    conversation_index=index,
                    source_returned=int(run["source_returned"]),
                    source_opened=int(run["source_opened"]),
                    latency_ms=int(run["latency_ms"]),
                    seed_digest=seed_digest,
                    __weave={
                        "display_name": (
                            f"qualification {run['attempt_label']} source {index}"
                        ),
                        "attributes": {
                            "fugue.seed_digest": seed_digest,
                            "fugue.evidence_kind": "source_conversation",
                        },
                    },
                )
                conversations.append(
                    {
                        "run_id": run["id"],
                        "conversation_index": index,
                        "call_id": call.id,
                        "ref": f"weave:///{project}/call/{call.id}",
                    }
                )
    evaluations = asyncio.run(
        _run_evaluations(
            project,
            dataset,
            seed_digest=seed_digest,
        )
    )
    return {
        "dataset": {
            "name": "mcp-release-maintenance-cases",
            "ref": str(dataset_ref.uri()),
            "rows": len(_EVALUATION_ROWS),
            "content_digest": stable_digest(
                [dict(item) for item in _EVALUATION_ROWS]
            ),
        },
        "source_conversations": conversations,
        "evaluations": evaluations,
    }


def _existing_source_conversations(
    client: Any,
    project: str,
    *,
    seed_digest: str,
) -> list[dict[str, Any]]:
    expected = {
        (str(run["id"]), index)
        for run in _RUNS
        for index in range(1, 5)
    }
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for call in client.get_calls(limit=500):
        if "/op/fugue.qualification.maintenance_agent:" not in str(
            call.op_name
        ):
            continue
        attributes = call.attributes
        if (
            not isinstance(attributes, Mapping)
            or attributes.get("fugue.seed_digest") != seed_digest
        ):
            continue
        inputs = call.inputs
        if not isinstance(inputs, Mapping):
            continue
        coordinate = (
            str(inputs.get("run_id") or ""),
            int(inputs.get("conversation_index") or 0),
        )
        if coordinate in found:
            raise RuntimeError(
                "hosted project contains duplicate seeded source conversations"
            )
        found[coordinate] = {
            "run_id": coordinate[0],
            "conversation_index": coordinate[1],
            "call_id": call.id,
            "ref": f"weave:///{project}/call/{call.id}",
        }
    if not found:
        return []
    if set(found) != expected:
        raise RuntimeError(
            "hosted project contains a partial seeded conversation cohort"
        )
    return [found[key] for key in sorted(found)]


async def _run_evaluations(
    project: str,
    dataset: weave.Dataset,
    *,
    seed_digest: str,
) -> list[dict[str, Any]]:
    values = []
    for revision, model in (
        ("maintainer-r17", _baseline_evidence_model),
        ("maintainer-r18", _candidate_evidence_model),
    ):
        evaluation = weave.Evaluation(
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
        evaluation_ref = weave.publish(
            evaluation,
            name=f"mcp-release-{revision}",
            aliases=["qualification-v1"],
        )
        summary, call = await evaluation.evaluate.call(
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
        values.append(
            {
                "revision": revision,
                "ref": str(evaluation_ref.uri()),
                "call_id": call.id,
                "call_ref": f"weave:///{project}/call/{call.id}",
                "summary_digest": stable_digest(summary),
                "prediction_rows": len(_EVALUATION_ROWS),
            }
        )
    return values


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
                    {
                        "role": "assistant",
                        "content": (
                            f"Observed latency {latency_ms} ms; opened "
                            f"{source_opened} of {source_returned} returned sources."
                        ),
                    },
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
        baseline_pass
        if output.get("revision") == "maintainer-r17"
        else candidate_pass
    )
    return {
        "deterministic_pass": bool(output.get("pass")),
        "aligned_with_reviewed_row": output.get("pass") is expected,
    }


def _evidence_lock(
    project: str,
    runs: Sequence[Mapping[str, Any]],
    weave_receipts: Mapping[str, Any],
    *,
    result_project: str | None = None,
) -> dict[str, Any]:
    resolved_result_project = result_project or project
    seed_digest = qualification_seed_digest(source_project=project)
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
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "counts": dict(_REQUIRED_COUNTS),
        "objects": {
            "runs": [dict(item) for item in runs],
            "dataset": dict(weave_receipts["dataset"]),
            "source_conversations": [
                dict(item) for item in weave_receipts["source_conversations"]
            ],
            "evaluations": [
                dict(item) for item in weave_receipts["evaluations"]
            ],
        },
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
