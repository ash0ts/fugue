from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fugue.bench.candidates import stable_digest
from fugue.bench.reproducibility import verify_snapshot
from fugue.redaction import redact_value

if TYPE_CHECKING:
    from fugue.bench.campaign_lifecycle import PlanReceiptV1, PreparedPlanV1
    from fugue.bench.operator import RunSummary


def outcome_eligibility_failures(
    *,
    run: RunSummary,
    rows: Sequence[Mapping[str, Any]],
    plan: PlanReceiptV1,
    prepared: PreparedPlanV1,
    input_lock: Mapping[str, Any] | None,
    evaluation_lock_digest: str | None,
) -> list[str]:
    failures: list[str] = []
    if len(rows) != plan.expected_predictions:
        failures.append(
            f"observed {len(rows)} prediction rows; expected {plan.expected_predictions}"
        )
    prediction_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        failures.extend(
            _row_eligibility_failures(
                row, index=index, run_id=run.run_id, prediction_ids=prediction_ids
            )
        )
    if input_lock is None:
        failures.append("run input lock is missing")
    else:
        failures.extend(
            _snapshot_eligibility_failures(
                input_lock,
                rows,
                plan,
                prepared,
                evaluation_lock_digest=evaluation_lock_digest,
            )
        )
    if run.observability_status not in {None, "passed"}:
        failures.append(f"run observability ended as {run.observability_status}")
    failures.extend(f"evaluation failure: {item}" for item in run.evaluation_failures)
    return list(dict.fromkeys(failures))


def _row_eligibility_failures(
    row: Mapping[str, Any],
    *,
    index: int,
    run_id: str,
    prediction_ids: set[str],
) -> list[str]:
    failures: list[str] = []
    if row.get("schema_version") != 1 or row.get("prediction_schema_version") != 1:
        failures.append(f"row {index} does not use canonical prediction schema 1")
    prediction_id = str(row.get("prediction_id") or "")
    if not prediction_id:
        failures.append(f"row {index} is missing prediction identity")
    elif prediction_id in prediction_ids:
        failures.append(f"row {index} duplicates prediction identity {prediction_id}")
    prediction_ids.add(prediction_id)
    if row.get("run_id") != run_id:
        failures.append(f"row {index} does not belong to run {run_id}")
    if row.get("status") not in {"passed", "failed", "not_applicable"}:
        failures.append(f"row {index} is not terminal")
    if row.get("execution_kind") != "agent":
        return failures
    if row.get("trace_link_status") not in {"linked", "verified", "exact"}:
        failures.append(f"row {index} lacks a valid Agent link")
    roots = row.get("weave_root_span_ids")
    if (
        not isinstance(roots, list)
        or len(roots) != 1
        or not roots[0]
        or row.get("root_span_id") != roots[0]
    ):
        failures.append(f"row {index} does not reconcile to exactly one Agent root")
    conversations = row.get("weave_conversation_ids")
    if (
        not isinstance(conversations, list)
        or len(conversations) != 1
        or not conversations[0]
        or row.get("observed_conversation_id") != conversations[0]
    ):
        failures.append(
            f"row {index} does not reconcile to exactly one Agent conversation"
        )
    traces = row.get("weave_trace_ids")
    if not isinstance(traces, list) or len(traces) != 1 or not traces[0]:
        failures.append(f"row {index} does not reconcile to exactly one Agent trace")
    agent_url = row.get("agent_url") or row.get("weave_agent_url")
    if agent_url and _safe_immutable_url(agent_url) is None:
        failures.append(f"row {index} has an invalid Agent link")
    if row.get("runtime_equivalence_status") != "equivalent":
        failures.append(f"row {index} lacks equivalent runtime evidence")
    if row.get("runtime_drift") is True:
        failures.append(f"row {index} reports runtime drift")
    return failures


def _snapshot_eligibility_failures(
    snapshot: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    plan: PlanReceiptV1,
    prepared: PreparedPlanV1,
    *,
    evaluation_lock_digest: str | None,
) -> list[str]:
    if not verify_snapshot(snapshot):
        return ["run input lock digest is invalid"]
    failures: list[str] = []
    snapshot_sha = str(
        snapshot.get("snapshot_sha256") or snapshot.get("lock_sha256") or ""
    )
    for index, row in enumerate(rows, 1):
        if row.get("run_snapshot_sha256") != snapshot_sha:
            failures.append(f"row {index} does not bind the run snapshot")
    source = (snapshot.get("runtime") or {}).get("fugue_source") or {}
    comparable = ("kind", "commit", "dirty", "dirty_digest", "digest")
    if any(source.get(key) != plan.source_provenance.get(key) for key in comparable):
        failures.append("run source provenance differs from the plan receipt")
    expected_lock = str(snapshot.get("evaluation_asset_lock_sha256") or "")
    if not evaluation_lock_digest or expected_lock != evaluation_lock_digest:
        failures.append("run snapshot does not bind the exact evaluation asset lock")
    if any(
        row.get("execution_kind") == "agent"
        and (
            not expected_lock
            or row.get("evaluation_asset_lock_sha256") != expected_lock
        )
        for row in rows
    ):
        failures.append("prediction rows do not bind the evaluation asset lock")
    if not _snapshot_coordinates_match(snapshot, plan):
        failures.append("run snapshot coordinates differ from the plan receipt")
    if not _route_receipts_valid(snapshot, prepared):
        failures.append("run snapshot lacks valid model-route receipts")
    if not _runtime_locks_valid(snapshot, plan, prepared):
        failures.append("run snapshot lacks exact runtime locks")
    return failures


def _snapshot_coordinates_match(
    snapshot: Mapping[str, Any], plan: PlanReceiptV1
) -> bool:
    observed = {
        (
            item.get("candidate_id"),
            item.get("execution_fingerprint"),
            item.get("execution_kind"),
            item.get("comparison_example_id"),
            item.get("trial_index"),
            item.get("workload_id"),
            item.get("task_id"),
            bool(item.get("applicable")),
            int(item.get("planned_prediction_count") or 0),
        )
        for item in snapshot.get("planned_matrix") or []
        if isinstance(item, dict)
    }
    expected = {
        (
            item.get("candidate_id"),
            item.get("execution_fingerprint"),
            item.get("execution_kind"),
            item.get("comparison_example_id"),
            item.get("trial_index"),
            item.get("workload_id"),
            item.get("task_id"),
            bool(item.get("applicable")),
            int(item.get("expected_predictions") or 0),
        )
        for item in plan.cells
    }
    return observed == expected and len(observed) == plan.cell_count


def _route_receipts_valid(
    snapshot: Mapping[str, Any], prepared: PreparedPlanV1
) -> bool:
    expected = {
        str(item.get("candidate_id") or ""): item for item in prepared.route_locks
    }
    if len(expected) != len(prepared.route_locks):
        return False
    runtimes = snapshot.get("candidate_runtime") or {}
    if not isinstance(runtimes, dict):
        return False
    for candidate_id, lock in expected.items():
        runtime = runtimes.get(candidate_id)
        if not isinstance(runtime, dict):
            return False
        transport = runtime.get("model_transport")
        route = runtime.get("model_route")
        if not isinstance(transport, dict) or not isinstance(route, dict):
            return False
        if stable_digest(_route_identity_projection(route)) != lock.get(
            "route_configuration_sha256"
        ):
            return False
        if _json_value(transport) != lock.get("transport"):
            return False
        if runtime.get("candidate_id") not in {None, candidate_id}:
            return False
        configuration = str(runtime.get("configuration_sha256") or "")
        if configuration:
            unsigned = {
                key: value
                for key, value in runtime.items()
                if key != "configuration_sha256"
            }
            if configuration != stable_digest(unsigned):
                return False
        elif len(runtime) > 2:
            return False
    return True


def _runtime_locks_valid(
    snapshot: Mapping[str, Any],
    plan: PlanReceiptV1,
    prepared: PreparedPlanV1,
) -> bool:
    expected_pairs = {
        str(item.get("execution_fingerprint") or ""): str(
            item.get("candidate_id") or ""
        )
        for item in plan.cells
        if item.get("applicable")
    }
    locks = snapshot.get("runtime_locks") or []
    if not isinstance(locks, list):
        return False
    observed: set[str] = set()
    agent_images = {
        str(item.get("image_id") or "")
        for item in prepared.preparation.get("agent_runtimes") or []
        if item.get("image_id")
    }
    task_images = {
        str(item.get("image_id") or "")
        for item in prepared.preparation.get("task_runtimes") or []
        if item.get("image_id")
    }
    portable = prepared.preparation.get("portable_context_runtime") or {}
    context_images = {str(portable.get("image_id") or "")} - {""}
    needs_agent_runtime = any(
        item.get("applicable") and item.get("execution_kind") == "agent"
        for item in plan.cells
    )
    needs_task_runtime = any(
        item.get("applicable")
        and item.get("execution_kind") == "agent"
        and item.get("workload_id") == "harbor"
        for item in plan.cells
    )
    if needs_agent_runtime and not agent_images:
        return False
    if needs_task_runtime and not task_images:
        return False
    for item in locks:
        if not isinstance(item, dict):
            return False
        fingerprint = str(item.get("execution_fingerprint") or "")
        digest = str(item.get("configuration_sha256") or "")
        unsigned = {
            key: value for key, value in item.items() if key != "configuration_sha256"
        }
        if (
            not fingerprint
            or fingerprint in observed
            or digest != stable_digest(unsigned)
            or item.get("candidate_id") != expected_pairs.get(fingerprint)
        ):
            return False
        agent_runtime = item.get("agent_runtime") or {}
        task_runtime = item.get("task_runtime") or {}
        context_runtime = item.get("context_runtime") or {}
        if agent_images and agent_runtime.get("image_id") not in agent_images:
            return False
        if task_images and task_runtime.get("image_id") not in task_images:
            return False
        if context_images and context_runtime.get("image_id") not in context_images:
            return False
        observed.add(fingerprint)
    return set(expected_pairs) == observed and len(locks) == len(expected_pairs)


_SAFE_PREDICTION_FIELDS = (
    "schema_version",
    "prediction_schema_version",
    "prediction_id",
    "run_id",
    "candidate_id",
    "comparison_example_id",
    "trial_index",
    "execution_kind",
    "status",
    "pass",
    "reward",
    "workload_id",
    "task_name",
    "harness",
    "variant_id",
    "context_system_id",
    "context_delivery",
    "model_provider",
    "model",
    "cost_usd",
    "weave_total_cost_usd",
    "input_tokens",
    "output_tokens",
    "tool_calls",
    "turns",
    "wall_time_sec",
    "trace_link_status",
    "runtime_equivalence_status",
    "runtime_drift",
    "context_registration_status",
    "context_invoked",
    "localization_recall_at_5",
    "localization_recall_at_10",
    "localization_mrr",
    "relevant_retrieval_open_rate",
    "relevant_retrieval_change_rate",
    "off_target_change_only",
    "premature_completion",
    "prompt_injection_action_gate_allowed",
    "prompt_injection_action_gate_blocked",
    "prompt_injection_attack_encountered",
    "prompt_injection_compromised",
    "prompt_injection_false_positive_refusal",
    "prompt_injection_incorrect",
    "prompt_injection_safe_and_useful",
    "prompt_injection_safe_but_failed_or_refused",
    "prompt_injection_sensitive_action_attempted",
    "prompt_injection_task_complete",
    "source_commit",
    "run_snapshot_sha256",
    "evaluation_asset_lock_sha256",
)


def safe_prediction_row(
    row: Mapping[str, Any], secrets: Sequence[str] = ()
) -> dict[str, Any]:
    result = {
        key: _json_value(row[key]) for key in _SAFE_PREDICTION_FIELDS if key in row
    }
    return _json_value(redact_value(result, secrets=secrets))


def _safe_immutable_url(value: Any) -> str | None:
    raw = str(value or "")
    if not raw or len(raw) > 2000:
        return None
    parsed = urlparse(raw)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return None
    return raw


def safe_agent_evidence(
    row: Mapping[str, Any], secrets: Sequence[str] = ()
) -> dict[str, Any]:
    result = {
        "prediction_id": row.get("prediction_id"),
        "trace_link_status": row.get("trace_link_status"),
        "conversation_ids": [
            str(value) for value in row.get("weave_conversation_ids") or [] if value
        ],
        "root_span_ids": [
            str(value) for value in row.get("weave_root_span_ids") or [] if value
        ],
        "trace_ids": [
            str(value) for value in row.get("weave_trace_ids") or [] if value
        ],
        "agent_url": _safe_immutable_url(
            row.get("agent_url") or row.get("weave_agent_url")
        ),
    }
    return _json_value(redact_value(result, secrets=secrets))


def outcome_metrics(rows: Sequence[Mapping[str, Any]], passed: int) -> dict[str, Any]:
    total = len(rows)
    return {
        "passes": passed,
        "predictions": total,
        "pass_rate": passed / total if total else None,
        "tool_calls": sum(
            int(row.get("tool_calls") or row.get("weave_tool_count") or 0)
            for row in rows
        ),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
    }


def _route_identity_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "provider",
        "model_id",
        "display_model",
        "chat_base_url",
        "responses_base_url",
        "messages_base_url",
        "litellm_model",
        "tool_result_modalities",
    )
    return {key: _json_value(value.get(key)) for key in fields}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value
