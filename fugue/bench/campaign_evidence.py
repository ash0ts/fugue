from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlparse

from fugue.bench.candidates import (
    attempt_id as canonical_attempt_id,
)
from fugue.bench.candidates import (
    attempt_identity as canonical_attempt_identity,
)
from fugue.bench.candidates import (
    stable_digest,
)
from fugue.bench.reproducibility import verify_snapshot
from fugue.redaction import redact_value

if TYPE_CHECKING:
    from fugue.bench.campaign_lifecycle import PlanReceiptV1, PreparedPlanV1
    from fugue.bench.operator import RunSummary


def apply_campaign_run_conformance(
    rows: Sequence[dict[str, Any]],
    *,
    repo_root: Path,
    run_id: str,
    required_attempt_ids: Set[str] = frozenset(),
) -> None:
    """Attach exact run-scoped Harbor/privacy receipts to campaign rows."""

    try:
        from fugue.bench.run_conformance import (
            PRIVACY_CONTRACT_VERSION,
            read_harbor_run_conformance_receipt,
            read_hosted_evidence_privacy_receipt,
        )

        receipt = read_harbor_run_conformance_receipt(
            repo_root=repo_root,
            run_id=run_id,
        )
    except (FileNotFoundError, ValueError):
        for row in rows:
            if row.get("run_id") == run_id and _requires_harbor_conformance(
                row,
                required_attempt_ids=required_attempt_ids,
            ):
                row["harbor_conformance_status"] = "unavailable"
        return
    if receipt.get("backend") != "local_harbor_docker":
        for row in rows:
            if row.get("run_id") == run_id and _requires_harbor_conformance(
                row,
                required_attempt_ids=required_attempt_ids,
            ):
                row["harbor_conformance_status"] = "unavailable"
        return
    identity = receipt.get("execution_identity")
    local_privacy = receipt.get("local_artifact_privacy_scan")
    private_boundary = receipt.get("private_label_boundary")
    cleanup = receipt.get("docker_cleanup")
    identity = identity if isinstance(identity, Mapping) else {}
    local_privacy = local_privacy if isinstance(local_privacy, Mapping) else {}
    private_boundary = private_boundary if isinstance(private_boundary, Mapping) else {}
    cleanup = cleanup if isinstance(cleanup, Mapping) else {}
    receipt_version = int(receipt.get("schema_version") or 0)
    receipt_digest = str(receipt.get("receipt_sha256") or "")
    try:
        hosted = read_hosted_evidence_privacy_receipt(
            repo_root=repo_root,
            run_id=run_id,
        )
    except (FileNotFoundError, ValueError):
        hosted = {}
    hosted_digest = str(hosted.get("receipt_sha256") or "")
    hosted_status = str(hosted.get("status") or "unavailable")
    hosted_matches = sum(
        int(hosted.get(field) or 0)
        for field in (
            "secret_match_count",
            "private_corpus_match_count",
            "private_structure_match_count",
        )
    )
    local_matches = sum(
        int(item.get("match_count") or 0)
        for item in local_privacy.get("files_with_matches") or ()
        if isinstance(item, Mapping)
    )
    matched_containers = cleanup.get("matched_containers") or ()
    for row in rows:
        if row.get("run_id") != run_id or not _requires_harbor_conformance(
            row,
            required_attempt_ids=required_attempt_ids,
            legacy_all_agent_rows=not required_attempt_ids,
        ):
            continue
        row.update(
            {
                "harbor_environment": "local_harbor_docker",
                "harbor_conformance_status": str(
                    receipt.get("status") or "unavailable"
                ),
                "harbor_conformance_receipt_digest": receipt_digest,
                "harbor_policy_attestation_verified": (
                    identity.get("status") == "passed"
                ),
                "privacy_contract_version": receipt_version,
                "local_artifact_privacy_scan_status": str(
                    local_privacy.get("status") or "unavailable"
                ),
                "local_artifact_privacy_scan_digest": stable_digest(local_privacy),
                "local_artifact_privacy_match_count": local_matches,
                "hosted_evidence_privacy_scan_status": hosted_status,
                "hosted_evidence_privacy_scan_digest": hosted_digest or None,
                "hosted_evidence_privacy_match_count": hosted_matches,
                "private_label_boundary_verified": (
                    private_boundary.get("status") == "passed"
                ),
                "sandbox_cleanup_verified": cleanup.get("status") == "passed",
                "orphaned_sandbox": bool(matched_containers),
            }
        )
        if receipt_version != PRIVACY_CONTRACT_VERSION:
            row["local_artifact_privacy_scan_status"] = "unavailable"


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
    expected_attempts = _expected_attempts(plan)
    expected_harbor_attempts = _expected_harbor_attempts(plan)
    observed_attempts: set[str] = set()
    for index, row in enumerate(rows, 1):
        failures.extend(
            _row_eligibility_failures(
                row,
                index=index,
                run_id=run.run_id,
                prediction_ids=prediction_ids,
                expected_attempts=expected_attempts,
                expected_harbor_attempts=expected_harbor_attempts,
                observed_attempts=observed_attempts,
            )
        )
    if observed_attempts != set(expected_attempts):
        failures.append(
            "prediction rows do not reconcile to the exact planned attempt identities"
        )
    failures.extend(_project_topology_failures(rows))
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
    expected_attempts: Mapping[str, Mapping[str, Any]],
    expected_harbor_attempts: Set[str],
    observed_attempts: set[str],
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
    failures.extend(
        _attempt_identity_failures(
            row,
            index=index,
            expected_attempts=expected_attempts,
            observed_attempts=observed_attempts,
        )
    )
    if row.get("status") == "not_applicable":
        return failures
    if row.get("trace_link_status") not in {"linked", "verified", "exact"}:
        failures.append(f"row {index} lacks a valid Agent link")
    link_set = verified_trace_link_set(row)
    failures.extend(f"row {index} {item}" for item in link_set["failures"])
    conversations = row.get("weave_conversation_ids")
    if (
        not isinstance(conversations, list)
        or len(conversations) != 1
        or not conversations[0]
        or row.get("observed_conversation_id") != conversations[0]
        or row.get("conversation_correlation_verified") is not True
    ):
        failures.append(
            f"row {index} does not reconcile to exactly one Agent conversation"
        )
    traces = row.get("otel_trace_ids")
    if not isinstance(traces, list) or len(traces) != 1 or not traces[0]:
        failures.append(f"row {index} does not reconcile to exactly one Agent trace")
    agent_url = row.get("agent_url") or row.get("weave_agent_url")
    if agent_url and _safe_immutable_url(agent_url) is None:
        failures.append(f"row {index} has an invalid Agent link")
    if row.get("runtime_equivalence_status") != "equivalent":
        failures.append(f"row {index} lacks equivalent runtime evidence")
    if row.get("runtime_drift") is True:
        failures.append(f"row {index} reports runtime drift")
    failures.extend(
        f"row {index} {item}"
        for item in _harbor_conformance_failures(
            row,
            required=str(row.get("attempt_id") or "") in expected_harbor_attempts,
        )
    )
    return failures


def _attempt_identity_failures(
    row: Mapping[str, Any],
    *,
    index: int,
    expected_attempts: Mapping[str, Mapping[str, Any]],
    observed_attempts: set[str],
) -> list[str]:
    failures: list[str] = []
    attempt_id = str(row.get("attempt_id") or "")
    identity = row.get("attempt_identity")
    if not attempt_id:
        failures.append(f"row {index} is missing canonical attempt identity")
    elif attempt_id in observed_attempts:
        failures.append(f"row {index} duplicates attempt identity {attempt_id}")
    else:
        observed_attempts.add(attempt_id)
    if not isinstance(identity, Mapping):
        failures.append(f"row {index} is missing canonical attempt coordinates")
    else:
        try:
            canonical = canonical_attempt_identity(
                task_id=str(identity.get("task_id") or ""),
                arm=str(identity.get("arm") or ""),
                harness=str(identity.get("harness") or ""),
                attempt=int(identity.get("attempt") or 0),
                candidate=str(identity.get("candidate") or ""),
                runtime=str(identity.get("runtime") or ""),
            )
            recomputed = canonical_attempt_id(**canonical)
        except (TypeError, ValueError):
            failures.append(f"row {index} has invalid canonical attempt coordinates")
        else:
            if attempt_id != recomputed or dict(identity) != canonical:
                failures.append(
                    f"row {index} attempt identity does not match its coordinates"
                )
            if expected_attempts.get(attempt_id) != canonical:
                failures.append(
                    f"row {index} attempt identity was not in the immutable plan"
                )
            observed = {
                "task_id": str(row.get("task_id") or row.get("task_name") or ""),
                "arm": str(row.get("variant_id") or ""),
                "harness": str(row.get("harness") or ""),
                "attempt": int(row.get("trial_index") or 0),
                "candidate": str(row.get("candidate_id") or ""),
                "runtime": str(row.get("execution_fingerprint") or ""),
            }
            if observed != canonical:
                failures.append(
                    f"row {index} evidence coordinates differ from its attempt identity"
                )
    return failures


def _expected_attempts(plan: PlanReceiptV1) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for cell in plan.cells:
        if cell.get("execution_kind") != "agent":
            continue
        supplied_identity = cell.get("attempt_identity")
        supplied_attempt_id = str(cell.get("attempt_id") or "")
        if not isinstance(supplied_identity, Mapping) or not supplied_attempt_id:
            raise ValueError("campaign plan is missing its canonical attempt identity")
        identity = canonical_attempt_identity(
            task_id=str(cell.get("task_id") or ""),
            arm=str(cell.get("variant_id") or ""),
            harness=str(cell.get("harness") or ""),
            attempt=int(cell.get("trial_index") or 0),
            candidate=str(cell.get("candidate_id") or ""),
            runtime=str(cell.get("execution_fingerprint") or ""),
        )
        attempt_id = canonical_attempt_id(**identity)
        if dict(supplied_identity) != identity or supplied_attempt_id != attempt_id:
            raise ValueError(
                "campaign plan attempt identity differs from its coordinates"
            )
        if attempt_id in expected:
            raise ValueError("campaign plan contains duplicate attempt identity")
        expected[attempt_id] = identity
    return expected


def _expected_harbor_attempts(plan: PlanReceiptV1) -> frozenset[str]:
    return frozenset(
        str(cell.get("attempt_id") or "")
        for cell in plan.cells
        if cell.get("execution_kind") == "agent"
        and cell.get("applicable")
        and cell.get("workload_runner") == "harbor"
        and cell.get("attempt_id")
    )


def _project_topology_failures(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    failures: list[str] = []
    topologies: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, 1):
        if (
            row.get("execution_kind") != "agent"
            or row.get("status") == "not_applicable"
        ):
            continue
        trace_project = str(row.get("trace_project") or "")
        result_project = str(row.get("result_evidence_project") or "")
        source_project = str(row.get("source_evidence_project") or "")
        if not _project_slug(result_project):
            failures.append(f"row {index} lacks a declared result evidence project")
            continue
        if trace_project != result_project:
            failures.append(
                f"row {index} trace destination differs from its result project"
            )
        if source_project and not _project_slug(source_project):
            failures.append(f"row {index} has an invalid source evidence project")
        if source_project and source_project == result_project:
            failures.append(
                f"row {index} source and result evidence projects are not isolated"
            )
        receipt = row.get("trace_receipt")
        if (
            not isinstance(receipt, Mapping)
            or str(receipt.get("project_slug") or "") != result_project
        ):
            failures.append(
                f"row {index} destination receipt differs from its result project"
            )
        queried = {
            str(value) for value in row.get("mcp_queried_projects") or () if str(value)
        }
        allowed_queries = {source_project or result_project}
        unexpected = sorted(queried - allowed_queries)
        if unexpected:
            failures.append(
                f"row {index} queried unexpected project(s): {', '.join(unexpected)}"
            )
        topologies.add((source_project, result_project))
    if len(topologies) > 1:
        failures.append("campaign rows disagree on source/result project topology")
    return failures


def _project_slug(value: str) -> bool:
    return bool(
        value.count("/") == 1
        and all(part and part not in {".", ".."} for part in value.split("/"))
    )


def _harbor_conformance_failures(
    row: Mapping[str, Any],
    *,
    required: bool = False,
) -> list[str]:
    if not required and not _is_local_harbor_agent_row(row):
        return []
    checks = (
        (
            row.get("harbor_environment") == "local_harbor_docker",
            "does not identify the exact local Harbor backend",
        ),
        (
            row.get("harbor_conformance_status") == "passed",
            "lacks a passing Harbor conformance receipt",
        ),
        (
            row.get("harbor_policy_attestation_verified") is True,
            "lacks verified Harbor policy attestation",
        ),
        (
            row.get("privacy_contract_version") == 2,
            "does not use privacy contract V2",
        ),
        (
            row.get("local_artifact_privacy_scan_status") == "passed"
            and int(row.get("local_artifact_privacy_match_count") or 0) == 0,
            "lacks a clean local artifact privacy scan",
        ),
        (
            row.get("hosted_evidence_privacy_scan_status") == "passed"
            and int(row.get("hosted_evidence_privacy_match_count") or 0) == 0,
            "lacks a clean hosted evidence privacy scan",
        ),
        (
            row.get("private_label_boundary_verified") is True,
            "lacks a verified private-label boundary",
        ),
        (
            row.get("sandbox_cleanup_verified") is True,
            "lacks verified run-scoped Harbor cleanup",
        ),
        (
            row.get("orphaned_sandbox") is False,
            "does not prove zero run-scoped Harbor orphans",
        ),
    )
    failures = [message for valid, message in checks if not valid]
    assigned_integrations = {
        str(value) for value in row.get("integration_ids") or () if str(value)
    }
    invoked_integrations = {
        str(value) for value in row.get("integration_ids_invoked") or () if str(value)
    }
    if invoked_integrations - assigned_integrations:
        failures.append("reports invoked integrations outside the locked candidate")
    missing_loop_integrations = {
        value
        for value in assigned_integrations
        if value.startswith("loop-intervention-")
    } - invoked_integrations
    if missing_loop_integrations:
        failures.append(
            "lacks host-observed invocation of locked loop intervention(s): "
            + ", ".join(sorted(missing_loop_integrations))
        )
    assigned_skills = {
        str(value)
        for value in row.get("skills_assigned") or row.get("skill_ids") or ()
        if str(value)
    }
    opened_skills = {
        str(value) for value in row.get("skill_ids_opened") or () if str(value)
    }
    native_invoked_skills = {
        str(value)
        for value in (
            row.get("skill_ids_native_invoked")
            if row.get("skill_ids_native_invoked") is not None
            else row.get("skill_ids_invoked")
        )
        or ()
        if str(value)
    }
    if opened_skills - assigned_skills:
        failures.append("reports opened skills outside the locked candidate")
    if native_invoked_skills - assigned_skills:
        failures.append("reports natively invoked skills outside the locked candidate")
    missing_loop_skills = {
        value for value in assigned_skills if value.startswith("loop-intervention-")
    } - opened_skills
    if missing_loop_skills:
        failures.append(
            "lacks host-observed instruction opening for locked loop "
            "intervention(s): " + ", ".join(sorted(missing_loop_skills))
        )
    return failures


def _is_local_harbor_agent_row(row: Mapping[str, Any]) -> bool:
    """Identify local Harbor from execution evidence, never a workload label."""

    if row.get("execution_kind") != "agent":
        return False
    if row.get("harbor_environment") in {"docker", "local_harbor_docker"}:
        return True
    if any(
        row.get(field) == "local_harbor_docker"
        for field in ("execution_backend", "sandbox_backend", "backend")
    ):
        return True
    runner = row.get("runner") or row.get("workload_runner")
    return runner == "harbor" and row.get("harbor_environment") != "wandb"


def _requires_harbor_conformance(
    row: Mapping[str, Any],
    *,
    required_attempt_ids: Set[str],
    legacy_all_agent_rows: bool = False,
) -> bool:
    if row.get("execution_kind") != "agent":
        return False
    attempt_id = str(row.get("attempt_id") or "")
    if attempt_id and attempt_id in required_attempt_ids:
        return True
    if required_attempt_ids:
        return _is_local_harbor_agent_row(row)
    return legacy_all_agent_rows or _is_local_harbor_agent_row(row)


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
            str(item.get("attempt_id") or ""),
            stable_digest(item.get("attempt_identity") or {}),
            item.get("candidate_id"),
            item.get("execution_fingerprint"),
            item.get("execution_kind"),
            item.get("comparison_example_id"),
            item.get("trial_index"),
            item.get("workload_id"),
            item.get("workload_runner"),
            item.get("task_id"),
            bool(item.get("applicable")),
            int(item.get("planned_prediction_count") or 0),
        )
        for item in snapshot.get("planned_matrix") or []
        if isinstance(item, dict)
    }
    expected = {
        (
            str(item.get("attempt_id") or ""),
            stable_digest(item.get("attempt_identity") or {}),
            item.get("candidate_id"),
            item.get("execution_fingerprint"),
            item.get("execution_kind"),
            item.get("comparison_example_id"),
            item.get("trial_index"),
            item.get("workload_id"),
            item.get("workload_runner"),
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
    context_fingerprints = {
        str(item.get("execution_fingerprint") or "")
        for item in plan.cells
        if item.get("applicable")
        and str(item.get("context_system_id") or "none") != "none"
    }
    needs_agent_runtime = any(
        item.get("applicable") and item.get("execution_kind") == "agent"
        for item in plan.cells
    )
    harbor_fingerprints = {
        str(item.get("execution_fingerprint") or "")
        for item in plan.cells
        if item.get("applicable")
        and item.get("execution_kind") == "agent"
        and item.get("workload_runner") == "harbor"
    }
    if needs_agent_runtime and not agent_images:
        return False
    if harbor_fingerprints and not task_images:
        return False
    if context_fingerprints and not context_images:
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
        if (
            fingerprint in harbor_fingerprints
            and task_runtime.get("image_id") not in task_images
        ):
            return False
        if fingerprint in context_fingerprints:
            if context_runtime.get("image_id") not in context_images:
                return False
        elif context_runtime:
            return False
        observed.add(fingerprint)
    return set(expected_pairs) == observed and len(locks) == len(expected_pairs)


_SAFE_PREDICTION_FIELDS = (
    "schema_version",
    "prediction_schema_version",
    "prediction_id",
    "attempt_id",
    "attempt_identity",
    "run_id",
    "candidate_id",
    "execution_fingerprint",
    "comparison_example_id",
    "task_id",
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
    "trace_project",
    "trace_receipt",
    "source_evidence_project",
    "result_evidence_project",
    "weave_evaluation_root_call_id",
    "weave_evaluation_root_ref",
    "weave_evaluation_root_url",
    "evaluation_root_object_verified",
    "eval_predict_and_score_call_id",
    "eval_predict_and_score_ref",
    "eval_predict_and_score_url",
    "eval_predict_and_score_object_verified",
    "weave_prediction_call_id",
    "weave_prediction_ref",
    "weave_prediction_url",
    "weave_prediction_object_verified",
    "weave_agent_root_call_id",
    "weave_agent_root_ref",
    "weave_agent_root_url",
    "weave_agent_root_evidence_kind",
    "weave_agent_root_is_native_call",
    "agent_cross_transport_edge",
    "weave_agent_bridge_cross_transport_edge",
    "weave_agent_receipt_supersession",
    "weave_agent_receipt_cross_transport_edge",
    "agent_graph_verified",
    "weave_dataset_id",
    "weave_dataset_ref",
    "weave_dataset_url",
    "dataset_version_object_verified",
    "evaluation_prediction_graph_verified",
    "weave_conversation_ids",
    "planned_conversation_id",
    "observed_conversation_id",
    "conversation_correlation_verified",
    "otel_root_span_ids",
    "otel_trace_ids",
    "runtime_equivalence_status",
    "runtime_drift",
    "agent_runtime_image_id",
    "task_runtime_image_id",
    "sandbox_attestation_digest",
    "agent_runtime_completed",
    "wandb_serverless_eligible",
    "wandb_serverless_attestation_digest",
    "wandb_serverless_runtime_image",
    "wandb_serverless_sandbox_id",
    "wandb_serverless_orphans",
    "harbor_environment",
    "harbor_conformance_status",
    "harbor_conformance_receipt_digest",
    "harbor_policy_attestation_verified",
    "privacy_contract_version",
    "local_artifact_privacy_scan_status",
    "local_artifact_privacy_scan_digest",
    "local_artifact_privacy_match_count",
    "hosted_evidence_privacy_scan_status",
    "hosted_evidence_privacy_scan_digest",
    "hosted_evidence_privacy_match_count",
    "private_label_boundary_verified",
    "sandbox_cleanup_verified",
    "orphaned_sandbox",
    "context_registration_status",
    "context_invoked",
    "skill_ids",
    "skills_assigned",
    "skill_ids_opened",
    "skill_files_opened",
    "skill_ids_native_invoked",
    "skill_ids_invoked",
    "skill_provenance",
    "integration_ids",
    "integration_ids_invoked",
    "unexpected_integration_ids_invoked",
    "integration_provenance",
    "context_component_provenance",
    "mcp_tool_names",
    "mcp_tool_call_count",
    "mcp_queried_projects",
    "localization_recall_at_5",
    "localization_recall_at_10",
    "localization_mrr",
    "relevant_retrieval_returned",
    "relevant_retrieval_opened",
    "relevant_retrieval_open_rate",
    "relevant_retrieval_used",
    "relevant_retrieval_change_rate",
    "off_target_change_only",
    "premature_completion",
    "document_search_available",
    "document_search_invoked",
    "relevant_document_returned",
    "relevant_document_opened",
    "current_document_cited",
    "current_document_used",
    "artifact_schema_valid",
    "answer_facts_correct",
    "unsupported_claims_absent",
    "prompt_injection_action_gate_allowed",
    "prompt_injection_action_gate_blocked",
    "prompt_injection_attack_encountered",
    "prompt_injection_compromised",
    "prompt_injection_evidence_preserved",
    "prompt_injection_false_positive_refusal",
    "prompt_injection_incorrect",
    "prompt_injection_safe_and_useful",
    "prompt_injection_safe_but_failed_or_refused",
    "prompt_injection_sensitive_action_attempted",
    "prompt_injection_task_complete",
    "source_commit",
    "source_tree",
    "source_dirty_digest",
    "tags",
    "run_snapshot_sha256",
    "evaluation_asset_lock_sha256",
)


def safe_prediction_row(
    row: Mapping[str, Any], secrets: Sequence[str] = ()
) -> dict[str, Any]:
    result = {
        key: _json_value(row[key]) for key in _SAFE_PREDICTION_FIELDS if key in row
    }
    if row.get("execution_kind") == "agent":
        link_set = verified_trace_link_set(row)
        result["evidence_links"] = link_set["links"]
        result["evidence_link_failures"] = link_set["failures"]
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
    link_set = verified_trace_link_set(row)
    result = {
        "prediction_id": row.get("prediction_id"),
        "attempt_id": row.get("attempt_id"),
        "attempt_identity": row.get("attempt_identity"),
        "execution_fingerprint": row.get("execution_fingerprint"),
        "trace_link_status": row.get("trace_link_status"),
        "conversation_ids": [
            str(value) for value in row.get("weave_conversation_ids") or [] if value
        ],
        "otel_root_span_ids": [
            str(value) for value in row.get("otel_root_span_ids") or [] if value
        ],
        "otel_trace_ids": [
            str(value) for value in row.get("otel_trace_ids") or [] if value
        ],
        "links": link_set["links"],
        "link_failures": link_set["failures"],
    }
    return _json_value(redact_value(result, secrets=secrets))


_TRACE_LINK_SLOTS = (
    (
        "prediction_and_score",
        "evaluation_attempt",
        "eval_predict_and_score_call_id",
        "eval_predict_and_score_ref",
        "eval_predict_and_score_url",
        "eval_predict_and_score_object_verified",
    ),
    (
        "prediction",
        "prediction",
        "weave_prediction_call_id",
        "weave_prediction_ref",
        "weave_prediction_url",
        "weave_prediction_object_verified",
    ),
    (
        "evaluation_root",
        "evaluation",
        "weave_evaluation_root_call_id",
        "weave_evaluation_root_ref",
        "weave_evaluation_root_url",
        "evaluation_root_object_verified",
    ),
    (
        "agent_root",
        "agent_root",
        "weave_agent_root_call_id",
        "weave_agent_root_ref",
        "weave_agent_root_url",
        "agent_graph_verified",
    ),
)


def verified_trace_link_set(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return only authoritatively verified Weave evidence links.

    OTel trace/span identities are retained as diagnostics elsewhere. They
    never satisfy or populate a Weave link slot.
    """

    links: list[dict[str, Any]] = []
    failures: list[str] = []
    project = str(row.get("trace_project") or "")
    receipt = row.get("trace_receipt")
    app_base_url = (
        str(receipt.get("app_base_url") or "https://wandb.ai")
        if isinstance(receipt, Mapping)
        else "https://wandb.ai"
    )
    otel_ids = {
        str(value)
        for value in (
            row.get("otel_root_span_id"),
            row.get("root_span_id"),
            *(row.get("otel_root_span_ids") or ()),
            row.get("otel_trace_id"),
            row.get("trace_id"),
            *(row.get("otel_trace_ids") or ()),
        )
        if value
    }
    for slot, kind, id_field, ref_field, url_field, verified_field in _TRACE_LINK_SLOTS:
        call_id = str(row.get(id_field) or "")
        ref = str(row.get(ref_field) or "")
        url = _safe_immutable_url(row.get(url_field))
        if (
            row.get(verified_field) is not True
            or not call_id
            or call_id in otel_ids
            or not _weave_call_ref_matches(ref, project, call_id)
            or not _weave_call_url_matches(url, project, call_id, app_base_url)
        ):
            failures.append(f"does not verify the {slot} Weave link")
            continue
        evidence_kind = None
        if slot == "agent_root":
            evidence_kind = _agent_evidence_kind(row)
            evidence_failure = _agent_evidence_failure(
                row,
                evidence_kind=evidence_kind,
                call_id=call_id,
            )
            if evidence_failure:
                failures.append(evidence_failure)
                continue
        links.append(
            {
                "slot": slot,
                "system": "weave",
                "kind": kind,
                "ref": ref,
                "call_id": call_id,
                "uri": url,
                "verification_status": "verified",
                **({"evidence_kind": evidence_kind} if slot == "agent_root" else {}),
            }
        )
    dataset_ref = str(row.get("weave_dataset_ref") or row.get("weave_dataset_id") or "")
    dataset_url = _safe_immutable_url(row.get("weave_dataset_url"))
    if (
        row.get("dataset_version_object_verified") is not True
        or not _weave_object_ref_matches(dataset_ref, project)
        or not _weave_object_url_matches(
            dataset_url,
            dataset_ref,
            project,
            app_base_url,
        )
    ):
        failures.append("does not verify the dataset Weave link")
    else:
        links.append(
            {
                "slot": "dataset",
                "system": "weave",
                "kind": "dataset",
                "ref": dataset_ref,
                "uri": dataset_url,
                "verification_status": "verified",
            }
        )
    if len({str(item["slot"]) for item in links}) != 5:
        failures.append("does not contain exactly five verified Weave link slots")
    return {
        "schema_version": 1,
        "attempt_id": str(row.get("attempt_id") or ""),
        "project": project,
        "links": links,
        "failures": list(dict.fromkeys(failures)),
        "verified": not failures,
    }


def _agent_evidence_kind(row: Mapping[str, Any]) -> str:
    declared = str(row.get("weave_agent_root_evidence_kind") or "")
    if declared:
        return declared
    if row.get("weave_agent_root_is_native_call") is True:
        return "native_weave_call_v1"
    if (
        row.get("weave_agent_root_is_native_call") is False
        or row.get("agent_cross_transport_edge")
        or row.get("weave_agent_root_call_materialization_source")
    ):
        return "native_otel_cross_transport_receipt_v1"
    return "unclassified_legacy_agent_evidence_v1"


def _agent_evidence_failure(
    row: Mapping[str, Any],
    *,
    evidence_kind: str,
    call_id: str,
) -> str | None:
    if evidence_kind == "native_weave_call_v1":
        if row.get("weave_agent_root_is_native_call") is not True:
            return "does not verify native Agent Call provenance"
        return None
    if evidence_kind != "native_otel_cross_transport_receipt_v1":
        return "does not recognize the Agent evidence kind"
    if row.get("weave_agent_root_is_native_call") is not False:
        return "does not verify cross-transport receipt provenance"
    edge = row.get("agent_cross_transport_edge")
    if not isinstance(edge, Mapping):
        return "does not verify the Agent receipt cross-transport edge"
    expected_trace = str(row.get("otel_trace_id") or row.get("trace_id") or "")
    expected_span = str(row.get("otel_root_span_id") or row.get("root_span_id") or "")
    if (
        not expected_trace
        or not expected_span
        or edge.get("status") != "verified"
        or str(edge.get("source_system") or "") != "otel"
        or str(edge.get("source_trace_id") or "") != expected_trace
        or str(edge.get("source_span_id") or "") != expected_span
        or str(edge.get("receipt_system") or "") != "weave"
        or str(edge.get("receipt_call_id") or "") != call_id
    ):
        return "does not verify the Agent receipt cross-transport edge"
    return None


def _weave_call_ref_matches(ref: str, project: str, call_id: str) -> bool:
    if not _project_slug(project):
        return False
    entity, project_id = project.split("/", 1)
    return ref == f"weave:///{entity}/{project_id}/call/{call_id}"


def _weave_call_url_matches(
    url: str | None,
    project: str,
    call_id: str,
    app_base_url: str,
) -> bool:
    if url is None or not _project_slug(project):
        return False
    entity, project_id = project.split("/", 1)
    return url == (
        f"{app_base_url.rstrip('/')}/{quote(entity, safe='')}/"
        f"{quote(project_id, safe='')}/weave/calls/{quote(call_id, safe='')}"
    )


def _weave_object_ref_matches(ref: str, project: str) -> bool:
    if not _project_slug(project):
        return False
    entity, project_id = project.split("/", 1)
    prefix = f"weave:///{entity}/{project_id}/object/"
    return ref.startswith(prefix) and ":" in ref.removeprefix(prefix)


def _weave_object_url_matches(
    url: str | None,
    ref: str,
    project: str,
    app_base_url: str,
) -> bool:
    if (
        url is None
        or not _weave_object_ref_matches(ref, project)
        or not _project_slug(project)
    ):
        return False
    entity, project_id = project.split("/", 1)
    name, digest = ref.rsplit("/", 1)[-1].rsplit(":", 1)
    return url == (
        f"{app_base_url.rstrip('/')}/{quote(entity, safe='')}/"
        f"{quote(project_id, safe='')}/weave/objects/"
        f"{quote(name, safe='')}/versions/{quote(digest, safe='')}"
    )


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
