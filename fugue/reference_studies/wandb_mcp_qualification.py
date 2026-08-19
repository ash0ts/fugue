"""Readiness adapter for the W&B MCP package-release reference study.

This module is imported only when an authored binding (or the historical
qualification quartet) selects the adapter.  Fugue's generic comparison path
therefore remains usable without importing W&B-specific qualification code.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest

WANDB_MCP_READ_ONLY_TOOLS = frozenset(
    {
        "compare_artifact_versions_tool",
        "compare_runs_tool",
        "count_weave_traces_tool",
        "diagnose_run_tool",
        "get_artifact_details_tool",
        "get_run_history_tool",
        "infer_trace_schema_tool",
        "list_artifact_versions_tool",
        "probe_project_tool",
        "query_wandb_tool",
        "query_weave_traces_tool",
        "resolve_trace_roots_tool",
        "summarize_evaluation_tool",
    }
)


def qualification_input_readiness(
    spec: Any,
    *,
    repo_root: Path,
) -> tuple[dict[str, str], list[str]]:
    """Validate the exact immutable inputs selected by this reference study."""

    from fugue.bench import comparison as c

    declared = {
        "evidence_lock": spec.execution.evidence_lock,
        "source_conformance_receipt": (
            spec.execution.source_conformance_receipt
        ),
        "release_notes_lock": spec.execution.release_notes_lock,
        "mechanism_receipt": spec.execution.mechanism_receipt,
    }
    if _is_packaged_reference_study(spec):
        return _prepared_reference_source_readiness(spec, repo_root=repo_root)
    if not any(declared.values()):
        return _prepared_reference_source_readiness(spec, repo_root=repo_root)
    if spec.schema_version >= 3 and (
        spec.execution.source_evidence_project is None
        or spec.execution.source_evidence_destination is None
    ):
        return {}, [
            "V3 mechanism qualification requires a distinct immutable "
            "source evidence destination"
        ]
    if (
        spec.schema_version >= 3
        and spec.execution.source_evidence_project
        == spec.execution.evidence_project
    ):
        return {}, [
            "MCP mechanism qualification requires source and result evidence "
            "projects to be different"
        ]
    missing = [name for name, value in declared.items() if not value]
    if missing:
        return {}, [
            "comparison mechanism qualification must declare "
            + ", ".join(sorted(declared))
            + " together"
        ]

    try:
        evidence_path = c._safe_input_path(
            Path(str(declared["evidence_lock"])),
            repo_root,
            "evidence lock",
        )
        release_path = c._safe_input_path(
            Path(str(declared["release_notes_lock"])),
            repo_root,
            "release-notes lock",
        )
        conformance_path = c._safe_input_path(
            Path(str(declared["source_conformance_receipt"])),
            repo_root,
            "source conformance receipt",
        )
        receipt_path = c._safe_input_path(
            Path(str(declared["mechanism_receipt"])),
            repo_root,
            "mechanism receipt",
        )
        evidence_raw = c._load_json_object(evidence_path, "evidence lock")
        release_raw = c._load_json_object(release_path, "release-notes lock")
        conformance = c._load_digest_receipt(
            conformance_path,
            "source conformance receipt",
        )
        receipt = c._load_digest_receipt(receipt_path, "mechanism receipt")

        from fugue.reference_studies.wandb_mcp_qualification_core import (
            release_note_coverage_v3,
            tool_surface_coverage_v1,
            validate_evidence_lock,
            validate_release_notes_lock,
        )

        source_project = (
            spec.execution.source_evidence_project
            or spec.execution.evidence_project
            or ""
        )
        result_project = spec.execution.evidence_project or ""
        evidence = validate_evidence_lock(
            evidence_raw,
            expected_source_project=str(source_project),
            expected_result_project=str(result_project),
        )
        release = validate_release_notes_lock(release_raw)
        validate_release_candidate_binding(
            spec,
            release_notes=release,
            repo_root=repo_root,
        )
        prerequisite_digests = c._validate_prerequisite_result_binding(
            spec,
            repo_root=repo_root,
            source_lock_digest=str(evidence["evidence_lock_digest"]),
        )
        _validate_source_conformance_binding(
            conformance=conformance,
            evidence=evidence,
            source_project=str(source_project),
            result_project=str(result_project),
            private_labels_path=repo_root / spec.taskset.private_labels,
        )
        if receipt.get("kind") != "mcp-release-mechanism-qualification":
            raise ValueError("mechanism receipt kind does not match")
        if (
            receipt.get("project") != source_project
            or receipt.get("source_project") != source_project
        ):
            raise ValueError("mechanism receipt source project does not match")
        if receipt.get("result_project") != result_project:
            raise ValueError("mechanism receipt result project does not match")
        expected_endpoints = {
            "api_base_url": "https://api.wandb.ai",
            "trace_base_url": "https://trace.wandb.ai",
        }
        if receipt.get("endpoint_binding") != {
            **expected_endpoints,
            "endpoint_digest": stable_digest(expected_endpoints),
        }:
            raise ValueError(
                "mechanism receipt endpoint binding does not match the "
                "approved public-cloud topology"
            )
        if receipt.get("evidence_lock_digest") != evidence.get(
            "evidence_lock_digest"
        ):
            raise ValueError("mechanism receipt evidence lock does not match")
        if receipt.get("release_notes_lock") != release:
            raise ValueError("mechanism receipt release-notes lock does not match")

        receipt_candidates = {
            str(item.get("id") or ""): item
            for item in c._sequence(
                receipt.get("candidates"),
                "mechanism receipt candidates",
            )
            if isinstance(item, Mapping)
        }
        integration_ids = tuple(
            dict.fromkeys(
                str(selection["id"])
                for candidate in (spec.baseline, spec.candidate)
                for selection in candidate.integrations
            )
        )
        if len(integration_ids) == 2:
            _validate_mechanism_reconciliation_binding(
                receipt,
                evidence=evidence,
                baseline_id=integration_ids[0],
                candidate_id=integration_ids[1],
            )
        digests = {
            "evidence_lock": str(evidence["evidence_lock_digest"]),
            "evidence_lock_file": c._sha256_path(evidence_path),
            "source_conformance_receipt": str(
                conformance["receipt_digest"]
            ),
            "source_conformance_receipt_file": c._sha256_path(
                conformance_path
            ),
            "release_notes_lock": c._sha256_path(release_path),
            "mechanism_receipt": str(receipt["receipt_digest"]),
            **prerequisite_digests,
        }
        task_ids = tuple(
            str(item["id"])
            for item in c._load_public_tasks(repo_root / spec.taskset.tasks)
        )
        coverage = release_note_coverage_v3(
            receipt,
            task_ids=task_ids,
            dimension_ids=tuple(
                f"{evaluator.id}.{dimension}"
                for evaluator in spec.evaluators
                for dimension in evaluator.dimensions
            ),
        )
        digests["release_note_coverage"] = stable_digest(
            [dict(item) for item in coverage]
        )
        tool_coverage = tool_surface_coverage_v1(
            receipt,
            task_ids=task_ids,
        )
        digests["mcp_tool_surface_coverage"] = str(
            tool_coverage["coverage_digest"]
        )
        _bind_locked_release_mcp_profiles(
            repo_root=repo_root,
            integration_ids=integration_ids,
            receipt_candidates=receipt_candidates,
            digests=digests,
        )
        return dict(sorted(digests.items())), []
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"mechanism qualification inputs are not usable: {exc}"]


def bound_release_note_coverage(
    spec: Any,
    *,
    readiness: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], ...]:
    from fugue.bench import comparison as c

    if spec.schema_version < 3 or not spec.execution.mechanism_receipt:
        return ()
    if _is_packaged_reference_study(spec):
        policy = spec.decision_policy
        if policy is None:
            raise ValueError(
                "W&B MCP reference study has no governed candidate SHA"
            )
        root = (
            repo_root
            if repo_root.name == policy.candidate_sha
            and (repo_root / "source.lock.json").is_file()
            else repo_root
            / ".fugue/reference-studies/wandb-mcp"
            / policy.candidate_sha
        )
        from fugue.reference_studies.wandb_mcp import (
            read_wandb_mcp_reference_lock,
        )

        lock = read_wandb_mcp_reference_lock(root / "source.lock.json")
        current_inputs, coverage = _load_packaged_release_inputs(
            spec,
            root=root,
            source_lock=lock,
        )
        expected_inputs = c._mapping_or_empty(
            readiness.get("qualification_input_digests")
        )
        expected_coverage = expected_inputs.get("release_note_coverage")
        for name, label in (
            ("release_notes_lock", "release-notes lock"),
            ("mechanism_receipt", "mechanism receipt"),
        ):
            if current_inputs.get(name) != expected_inputs.get(name):
                raise ValueError(
                    f"{label} changed after preview; prepare and approve a "
                    "new exact preview"
                )
        if stable_digest([dict(item) for item in coverage]) != expected_coverage:
            raise ValueError(
                "release-note coverage changed after preview; prepare and "
                "approve a new exact preview"
            )
        return coverage
    receipt_path = c._safe_input_path(
        Path(spec.execution.mechanism_receipt),
        repo_root,
        "mechanism receipt",
    )
    receipt = c._load_digest_receipt(receipt_path, "mechanism receipt")
    expected_inputs = c._mapping_or_empty(
        readiness.get("qualification_input_digests")
    )
    if receipt.get("receipt_digest") != expected_inputs.get(
        "mechanism_receipt"
    ):
        raise ValueError(
            "mechanism receipt changed after preview; prepare and approve a "
            "new exact preview"
        )
    from fugue.reference_studies.wandb_mcp_qualification_core import (
        release_note_coverage_v3,
    )

    task_ids = tuple(
        str(item["id"])
        for item in c._load_public_tasks(repo_root / spec.taskset.tasks)
    )
    coverage = release_note_coverage_v3(
        receipt,
        task_ids=task_ids,
        dimension_ids=tuple(
            f"{evaluator.id}.{dimension}"
            for evaluator in spec.evaluators
            for dimension in evaluator.dimensions
        ),
    )
    if stable_digest([dict(item) for item in coverage]) != expected_inputs.get(
        "release_note_coverage"
    ):
        raise ValueError(
            "release-note coverage changed after preview; prepare and "
            "approve a new exact preview"
        )
    return coverage


def verify_source_drift(
    spec: Any,
    *,
    readiness: Mapping[str, Any],
    repo_root: Path,
    env: Mapping[str, str],
) -> Any:
    from fugue.bench import comparison as c

    if spec.schema_version < 3 or not spec.execution.evidence_lock:
        return None
    evidence_path = c._safe_input_path(
        Path(spec.execution.evidence_lock),
        repo_root,
        "evidence lock",
    )
    from fugue.reference_studies.wandb_mcp_qualification_core import (
        verify_hosted_source_drift,
    )

    check = verify_hosted_source_drift(evidence_lock=evidence_path, env=env)
    expected_digest = str(
        c._mapping_or_empty(
            readiness.get("qualification_input_digests")
        ).get("evidence_lock")
        or ""
    )
    if not expected_digest or check.expected_digest != expected_digest:
        raise ValueError(
            "source drift verification disagrees with the approved evidence "
            "lock"
        )
    return check


def _prepared_reference_source_readiness(
    spec: Any,
    *,
    repo_root: Path,
) -> tuple[dict[str, str], list[str]]:
    """Bind a locally executed reference study to its trusted source lock."""

    from fugue.bench import comparison as c

    policy = spec.decision_policy
    if policy is None or not re.fullmatch(r"[0-9a-f]{40}", policy.candidate_sha):
        return {}, [
            "W&B MCP reference study requires an exact governed candidate SHA"
        ]
    root = (
        repo_root
        if repo_root.name == policy.candidate_sha
        and (repo_root / "source.lock.json").is_file()
        else repo_root
        / ".fugue/reference-studies/wandb-mcp"
        / policy.candidate_sha
    )
    source_lock_path = root / "source.lock.json"
    receipt_path = root / "preparation.receipt.json"
    try:
        from fugue.reference_studies.wandb_mcp import (
            _SOURCE_PROJECT,
            read_wandb_mcp_reference_lock,
            read_wandb_mcp_reference_receipt,
        )
        from fugue.reference_studies.wandb_mcp_qualification_core import (
            validate_evidence_lock,
        )

        hosted_human_readable_canary = (
            spec.execution.evidence_mode == "weave_required"
        )
        if hosted_human_readable_canary:
            destination = spec.execution.evidence_destination
            if (
                spec.execution.evidence_project
                != "wandb/fugue-mcp-release-qualification-v1"
                or getattr(destination, "project_slug", None)
                != "wandb/fugue-mcp-release-qualification-v1"
            ):
                raise ValueError(
                    "the human-readable W&B MCP canary must use the locked "
                    "qualification result project"
                )
        else:
            if spec.execution.evidence_mode != "local":
                raise ValueError(
                    "packaged W&B MCP reference results must use local evidence"
                )
            if spec.execution.evidence_project is not None:
                raise ValueError(
                    "packaged W&B MCP reference results may not declare a hosted "
                    "result project"
                )
        if spec.execution.source_evidence_project != _SOURCE_PROJECT:
            raise ValueError(
                "prepared reference source project does not match the locked "
                "W&B MCP cohort"
            )
        if not spec.execution.evidence_lock:
            raise ValueError("prepared reference spec has no hosted evidence lock")
        if not spec.execution.source_conformance_receipt:
            raise ValueError(
                "prepared reference spec has no hosted source conformance receipt"
            )

        lock = read_wandb_mcp_reference_lock(source_lock_path)
        lock_value = lock.to_dict() if hasattr(lock, "to_dict") else dict(lock)
        observed_commit = str(
            lock_value.get("source_commit")
            or lock_value.get("commit")
            or lock_value.get("resolved_commit")
            or ""
        )
        if observed_commit != policy.candidate_sha:
            raise ValueError("prepared source lock candidate SHA does not match")
        receipt = read_wandb_mcp_reference_receipt(receipt_path)
        if receipt.materialization is None:
            raise ValueError(
                "prepared reference study has no runnable materialization"
            )
        for artifact in receipt.materialization.artifacts:
            path = root / artifact.path
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != artifact.byte_count
                or c._sha256_path(path) != artifact.sha256
            ):
                raise ValueError(
                    f"prepared reference artifact changed: {artifact.path}"
                )
        if hosted_human_readable_canary:
            _validate_human_readable_canary_binding(
                spec,
                root=root,
                source_lock=lock,
                receipt=receipt,
            )
        evidence_path = c._safe_input_path(
            Path(spec.execution.evidence_lock),
            repo_root,
            "prepared reference evidence lock",
        )
        conformance_path = c._safe_input_path(
            Path(spec.execution.source_conformance_receipt),
            repo_root,
            "prepared reference source conformance receipt",
        )
        evidence = validate_evidence_lock(
            c._load_json_object(
                evidence_path,
                "prepared reference evidence lock",
            ),
            expected_project=None,
            expected_source_project=_SOURCE_PROJECT,
            # Canonical results are local. The evidence-lock schema's legacy
            # result alias therefore remains the same read-only source.
            expected_result_project=_SOURCE_PROJECT,
        )
        conformance = c._load_digest_receipt(
            conformance_path,
            "prepared reference source conformance receipt",
        )
        _validate_source_conformance_binding(
            conformance=conformance,
            evidence=evidence,
            source_project=_SOURCE_PROJECT,
            result_project=_SOURCE_PROJECT,
            private_labels_path=repo_root / spec.taskset.private_labels,
        )
        release_input_digests, _ = _load_packaged_release_inputs(
            spec,
            root=root,
            source_lock=lock,
        )
        return {
            "reference_study_source_lock": lock.lock_digest,
            "reference_study_source_lock_file": c._sha256_path(
                source_lock_path
            ),
            "reference_study_candidate_source": lock.candidate_source_digest,
            "reference_study_preparation_receipt": receipt.receipt_digest,
            "reference_study_preparation_receipt_file": c._sha256_path(
                receipt_path
            ),
            "reference_study_behavior_inputs": (
                receipt.materialization.behavior_inputs_digest
            ),
            "reference_study_execution_inputs": (
                receipt.materialization.execution_inputs_digest
            ),
            "reference_study_inventory": (
                receipt.materialization.inventory_digest
            ),
            "evidence_lock": str(evidence["evidence_lock_digest"]),
            "evidence_lock_file": c._sha256_path(evidence_path),
            "source_conformance_receipt": str(
                conformance["receipt_digest"]
            ),
            "source_conformance_receipt_file": c._sha256_path(
                conformance_path
            ),
            **release_input_digests,
        }, []
    except (
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return {}, [f"W&B MCP reference study is not prepared: {exc}"]


def _validate_human_readable_canary_binding(
    spec: Any,
    *,
    root: Path,
    source_lock: Any,
    receipt: Any,
) -> None:
    """Prove that the hosted four-cell canary is the prepared artifact."""

    from fugue.bench import comparison as c
    from fugue.reference_studies.wandb_mcp import (
        _HUMAN_READABLE_TASK_IDS,
        HUMAN_READABLE_CANARY_LOCK_NAME,
        HUMAN_READABLE_COMPARISON_NAME,
        HUMAN_READABLE_PRIVATE_LABELS_NAME,
        HUMAN_READABLE_TASKS_NAME,
        _stable_digest,
    )

    expected_study_id = (
        "mcp-main-vs-0-4-"
        f"{source_lock.source_commit[:7]}-human-readable-evidence-canary-v1"
    )
    if (
        spec.id != expected_study_id
        or spec.taskset.tasks != HUMAN_READABLE_TASKS_NAME
        or spec.taskset.private_labels != HUMAN_READABLE_PRIVATE_LABELS_NAME
        or spec.execution.attempts != 1
        or tuple(spec.execution.harnesses) != ("claude-code",)
    ):
        raise ValueError(
            "hosted W&B MCP evidence canary coordinates differ from the "
            "prepared four-cell contract"
        )
    locked_spec = c.load_comparison(
        root / HUMAN_READABLE_COMPARISON_NAME,
        repo_root=root,
    )
    if locked_spec.spec_digest != spec.spec_digest:
        raise ValueError(
            "hosted W&B MCP evidence canary changed after preparation"
        )

    lock_path = root / HUMAN_READABLE_CANARY_LOCK_NAME
    lock = c._load_json_object(lock_path, "human-readable evidence canary lock")
    supplied_digest = str(lock.get("lock_digest") or "")
    unsigned = dict(lock)
    unsigned.pop("lock_digest", None)
    if _stable_digest(unsigned) != supplied_digest:
        raise ValueError("human-readable evidence canary lock digest differs")
    expected_fields = {
        "schema_version",
        "kind",
        "study_id",
        "source_lock_digest",
        "candidate_source_digest",
        "comparison",
        "tasks",
        "private_labels",
        "scorer",
        "arm_count",
        "attempts_per_coordinate",
        "logical_cell_count",
        "lock_digest",
    }
    if set(lock) != expected_fields or (
        lock.get("schema_version") != 1
        or lock.get("kind") != "wandb-mcp-human-readable-evidence-canary"
        or lock.get("study_id") != expected_study_id
        or lock.get("source_lock_digest") != source_lock.lock_digest
        or lock.get("candidate_source_digest")
        != source_lock.candidate_source_digest
        or lock.get("arm_count") != 2
        or lock.get("attempts_per_coordinate") != 1
        or lock.get("logical_cell_count") != 4
    ):
        raise ValueError("human-readable evidence canary lock is invalid")

    task_ids = tuple(
        str(item["id"])
        for item in c._load_public_tasks(root / HUMAN_READABLE_TASKS_NAME)
    )
    private_ids = tuple(
        str(item["id"])
        for item in c._load_private_labels(
            root / HUMAN_READABLE_PRIVATE_LABELS_NAME
        )
    )
    if task_ids != _HUMAN_READABLE_TASK_IDS or private_ids != task_ids:
        raise ValueError(
            "human-readable evidence canary tasks and private labels are not "
            "the aligned prepared pair"
        )
    expected_artifacts = {
        "comparison": HUMAN_READABLE_COMPARISON_NAME,
        "tasks": HUMAN_READABLE_TASKS_NAME,
        "private_labels": HUMAN_READABLE_PRIVATE_LABELS_NAME,
        "scorer": "tool_surface_scorer_v7.py",
    }
    for field, expected_path in expected_artifacts.items():
        value = lock.get(field)
        if not isinstance(value, Mapping) or value.get("path") != expected_path:
            raise ValueError(
                f"human-readable evidence canary {field} binding is invalid"
            )
        if value.get("sha256") != c._sha256_path(root / expected_path):
            raise ValueError(
                f"human-readable evidence canary {field} digest differs"
            )
    if lock["tasks"].get("task_ids") != list(task_ids) or lock[
        "private_labels"
    ].get("task_ids") != list(private_ids):
        raise ValueError(
            "human-readable evidence canary task ids differ from its files"
        )
    receipt_artifacts = {
        artifact.path: artifact.sha256
        for artifact in receipt.materialization.artifacts
    }
    for path in (HUMAN_READABLE_CANARY_LOCK_NAME, *expected_artifacts.values()):
        if receipt_artifacts.get(path) != c._sha256_path(root / path):
            raise ValueError(
                "human-readable evidence canary is not bound by the "
                f"preparation receipt: {path}"
            )


def _load_packaged_release_inputs(
    spec: Any,
    *,
    root: Path,
    source_lock: Any,
) -> tuple[dict[str, str], tuple[dict[str, Any], ...]]:
    """Validate and bind the packaged release-note and mechanism inputs."""

    from fugue.bench import comparison as c
    from fugue.reference_studies.wandb_mcp import (
        _PACKAGED_RELEASE_NOTE_COVERAGE,
        _stable_digest,
    )

    if not spec.execution.release_notes_lock:
        raise ValueError("prepared reference spec has no release-notes lock")
    if not spec.execution.mechanism_receipt:
        raise ValueError("prepared reference spec has no mechanism receipt")
    release_path = c._safe_input_path(
        Path(spec.execution.release_notes_lock),
        root,
        "prepared reference release-notes lock",
    )
    receipt_path = c._safe_input_path(
        Path(spec.execution.mechanism_receipt),
        root,
        "prepared reference mechanism receipt",
    )
    if release_path != (root / "release-notes.lock.json").resolve():
        raise ValueError("prepared release-notes lock path is not canonical")
    if receipt_path != (root / "mechanism-receipt.json").resolve():
        raise ValueError("prepared mechanism receipt path is not canonical")

    release = c._load_json_object(
        release_path,
        "prepared reference release-notes lock",
    )
    _verify_packaged_digest(
        release,
        field="lock_digest",
        label="prepared reference release-notes lock",
        digest=_stable_digest,
    )
    expected_release_fields = {
        "schema_version",
        "kind",
        "repository_url",
        "requested_ref",
        "commit",
        "tree",
        "release_notes",
        "lock_digest",
    }
    if set(release) != expected_release_fields:
        raise ValueError("prepared release-notes lock fields do not match V1")
    source_value = source_lock.to_dict()
    if (
        release.get("schema_version") != 1
        or release.get("kind") != "wandb-mcp-release-notes-lock"
        or release.get("repository_url") != source_value.get("repository_url")
        or release.get("requested_ref") != source_value.get("requested_ref")
        or release.get("commit") != source_value.get("source_commit")
        or release.get("tree") != source_value.get("source_tree")
        or release.get("release_notes") != source_value.get("release_notes")
    ):
        raise ValueError(
            "prepared release-notes lock does not match the exact source lock"
        )
    policy = spec.decision_policy
    if policy is None or release.get("commit") != policy.candidate_sha:
        raise ValueError(
            "prepared release-notes lock does not match the governed candidate"
        )

    receipt = c._load_json_object(
        receipt_path,
        "prepared reference mechanism receipt",
    )
    _verify_packaged_digest(
        receipt,
        field="receipt_digest",
        label="prepared reference mechanism receipt",
        digest=_stable_digest,
    )
    expected_receipt_fields = {
        "schema_version",
        "kind",
        "target_platform",
        "source_lock_digest",
        "release_notes_lock_digest",
        "profiles",
        "release_note_coverage",
        "runtime_dependency",
        "receipt_digest",
    }
    if set(receipt) != expected_receipt_fields:
        raise ValueError("prepared mechanism receipt fields do not match V1")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind")
        != "wandb-mcp-reference-mechanism-preparation"
        or receipt.get("source_lock_digest") != source_lock.lock_digest
        or receipt.get("release_notes_lock_digest")
        != release.get("lock_digest")
        or receipt.get("runtime_dependency") is not False
    ):
        raise ValueError(
            "prepared mechanism receipt does not match its source inputs"
        )
    _validate_packaged_mechanism_profiles(
        spec,
        repo_root=root,
        target_platform=str(receipt.get("target_platform") or ""),
        profiles=receipt.get("profiles"),
    )
    coverage = c._release_note_coverage_v3(
        [
            c._mapping(item, "prepared release-note coverage")
            for item in c._sequence(
                receipt.get("release_note_coverage"),
                "prepared release-note coverage",
            )
        ]
    )
    expected_coverage = c._release_note_coverage_v3(
        [dict(item) for item in _PACKAGED_RELEASE_NOTE_COVERAGE]
    )
    if not coverage or coverage != expected_coverage:
        raise ValueError(
            "prepared mechanism receipt release-note coverage does not match "
            "the packaged canary contract"
        )
    task_ids = {
        str(item["id"])
        for item in c._load_public_tasks(root / spec.taskset.tasks)
    }
    dimension_ids = {
        f"{evaluator.id}.{dimension}"
        for evaluator in spec.evaluators
        for dimension in evaluator.dimensions
    }
    missing_task_mappings = any(
        set(item["task_ids"]) - task_ids for item in coverage
    )
    human_readable_canary = _is_exact_human_readable_canary(
        spec,
        root=root,
        source_lock=source_lock,
    )
    if missing_task_mappings and not human_readable_canary:
        raise ValueError(
            "prepared release-note coverage references an unavailable task "
            "or scorer dimension"
        )
    if any(set(item["dimensions"]) - dimension_ids for item in coverage):
        raise ValueError(
            "prepared release-note coverage references an unavailable task "
            "or scorer dimension"
        )
    if not human_readable_canary:
        selected_coverage = [dict(item) for item in coverage]
    else:
        selected_coverage = []
        for item in coverage:
            selected_task_ids = [
                task_id for task_id in item["task_ids"] if task_id in task_ids
            ]
            if selected_task_ids:
                selected_coverage.append({**item, "task_ids": selected_task_ids})
                continue
            selected_coverage.append(
                {
                    **item,
                    "status": "not_applicable",
                    "task_ids": [],
                    "dimensions": [],
                    "infrastructure_gates": [],
                    "rationale": (
                        "This focused prepared canary does not include a task for "
                        f"{item['release_note']}. The full four-task reference "
                        "bundle remains the release-behavior canary."
                    ),
                }
            )
    coverage = c._release_note_coverage_v3(selected_coverage)
    return {
        "release_notes_lock": str(release["lock_digest"]),
        "release_notes_lock_file": c._sha256_path(release_path),
        "mechanism_receipt": str(receipt["receipt_digest"]),
        "mechanism_receipt_file": c._sha256_path(receipt_path),
        "release_note_coverage": stable_digest(
            [dict(item) for item in coverage]
        ),
    }, coverage


def _is_exact_human_readable_canary(
    spec: Any,
    *,
    root: Path,
    source_lock: Any,
) -> bool:
    """Return true only for the immutable prepared hosted canary spec."""

    from fugue.bench import comparison as c
    from fugue.reference_studies.wandb_mcp import (
        HUMAN_READABLE_COMPARISON_NAME,
        HUMAN_READABLE_PRIVATE_LABELS_NAME,
        HUMAN_READABLE_TASKS_NAME,
    )

    expected_id = (
        "mcp-main-vs-0-4-"
        f"{source_lock.source_commit[:7]}-human-readable-evidence-canary-v1"
    )
    if (
        spec.id != expected_id
        or spec.taskset.tasks != HUMAN_READABLE_TASKS_NAME
        or spec.taskset.private_labels != HUMAN_READABLE_PRIVATE_LABELS_NAME
        or spec.execution.evidence_mode != "weave_required"
        or spec.execution.evidence_project
        != "wandb/fugue-mcp-release-qualification-v1"
    ):
        return False
    try:
        locked_spec = c.load_comparison(
            root / HUMAN_READABLE_COMPARISON_NAME,
            repo_root=root,
        )
    except (FileNotFoundError, TypeError, ValueError):
        return False
    return locked_spec.spec_digest == spec.spec_digest


def _validate_packaged_mechanism_profiles(
    spec: Any,
    *,
    repo_root: Path,
    target_platform: str,
    profiles: Any,
) -> None:
    from fugue.bench import comparison as c
    from fugue.reference_studies.wandb_mcp import (
        _BASELINE_COMMIT,
        _mechanism_profile,
    )

    policy = spec.decision_policy
    if policy is None:
        raise ValueError("packaged MCP reference study has no decision policy")
    expected_profiles: list[dict[str, Any]] = []
    for role, candidate, commit in (
        ("baseline", spec.baseline, _BASELINE_COMMIT),
        ("candidate", spec.candidate, policy.candidate_sha),
    ):
        integration_ids = tuple(
            str(selection["id"]) for selection in candidate.integrations
        )
        if len(integration_ids) != 1:
            raise ValueError(
                "packaged MCP reference arms require one integration each"
            )
        integration_id = integration_ids[0]
        lock_path = (
            repo_root
            / ".fugue/imports/mcp/locks"
            / f"{integration_id}.json"
        )
        lock = c._load_json_object(
            c._safe_input_path(
                lock_path,
                repo_root,
                f"prepared MCP lock {integration_id}",
            ),
            f"prepared MCP lock {integration_id}",
        )
        if lock.get("runtime_platform") != target_platform:
            raise ValueError(
                f"prepared mechanism profile {role!r} platform does not match"
            )
        expected_profiles.append(_mechanism_profile(role, commit, lock))
    if profiles != expected_profiles:
        raise ValueError(
            "prepared mechanism receipt profiles do not match the exact MCP locks"
        )


def _verify_packaged_digest(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
    digest: Any,
) -> None:
    supplied = str(value.get(field) or "")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if not re.fullmatch(r"[0-9a-f]{64}", supplied) or digest(unsigned) != supplied:
        raise ValueError(f"{label} digest does not match")


def _is_packaged_reference_study(spec: Any) -> bool:
    binding = getattr(spec.execution, "reference_study", None)
    return bool(
        binding is not None
        and getattr(binding, "id", None) == "wandb-mcp-release"
        and getattr(binding, "version", None) == 1
    )


def _bind_locked_release_mcp_profiles(
    *,
    repo_root: Path,
    integration_ids: Sequence[str],
    receipt_candidates: Mapping[str, Any],
    digests: dict[str, str],
) -> None:
    from fugue.bench import comparison as c

    for integration_id in integration_ids:
        lock_path = (
            repo_root
            / ".fugue/imports/mcp/locks"
            / f"{integration_id}.json"
        )
        lock = c._load_json_object(
            c._safe_input_path(
                lock_path,
                repo_root,
                f"MCP lock {integration_id}",
            ),
            f"MCP lock {integration_id}",
        )
        fixed_env = c._mapping(
            lock.get("fixed_env"),
            f"MCP lock {integration_id} fixed environment",
        )
        allowed_tools = set(
            c._string_tuple(
                lock.get("allowed_tools"),
                f"MCP lock {integration_id} allowed tool",
            )
        )
        manifest_tools = {
            str(item.get("name") or "")
            for item in lock.get("tool_manifest") or ()
            if isinstance(item, Mapping) and str(item.get("name") or "")
        }
        if (
            fixed_env.get("WANDB_MCP_READ_ONLY") != "true"
            or allowed_tools != WANDB_MCP_READ_ONLY_TOOLS
            or not WANDB_MCP_READ_ONLY_TOOLS <= manifest_tools
        ):
            raise ValueError(
                f"MCP lock {integration_id!r} is not the exact read-only "
                "release qualification profile"
            )
        observed = receipt_candidates.get(integration_id)
        if not isinstance(observed, Mapping):
            raise ValueError(
                f"mechanism receipt is missing candidate {integration_id!r}"
            )
        for field_name in (
            "version_identity",
            "runtime_digest",
            "tool_manifest_digest",
        ):
            if observed.get(field_name) != lock.get(field_name):
                raise ValueError(
                    f"mechanism receipt {integration_id!r} {field_name} "
                    "does not match its MCP lock"
                )
        if observed.get("initialized_manifest_matches_lock") is not True:
            raise ValueError(
                f"mechanism receipt {integration_id!r} tool manifest was "
                "not verified"
            )
        runtime_digest = str(lock.get("runtime_digest") or "")
        manifest_digest = str(lock.get("tool_manifest_digest") or "")
        for label, digest in (
            ("runtime", runtime_digest),
            ("tool manifest", manifest_digest),
        ):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError(
                    f"MCP lock {integration_id!r} {label} digest is invalid"
                )
        digests[f"mcp_lock:{integration_id}"] = c._sha256_path(lock_path)
        digests[f"mcp_runtime:{integration_id}"] = runtime_digest.removeprefix(
            "sha256:"
        )
        digests[f"mcp_tool_manifest:{integration_id}"] = (
            manifest_digest.removeprefix("sha256:")
        )


def _validate_source_conformance_binding(
    *,
    conformance: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_project: str,
    result_project: str,
    private_labels_path: Path,
) -> None:
    from fugue.bench import comparison as c

    if (
        conformance.get("kind")
        != "mcp-release-hosted-source-conformance"
        or conformance.get("status") != "passed"
        or conformance.get("schema_version") != 2
    ):
        raise ValueError("source conformance receipt did not pass")
    if (
        conformance.get("source_project") != source_project
        or conformance.get("result_project") != result_project
    ):
        raise ValueError(
            "source conformance receipt project topology does not match"
        )
    if conformance.get("project_access") != {
        source_project: "PRIVATE",
        result_project: "PRIVATE",
    }:
        raise ValueError(
            "source conformance receipt does not prove private source and "
            "result projects"
        )
    expected_endpoints = {
        "api_base_url": "https://api.wandb.ai",
        "trace_base_url": "https://trace.wandb.ai",
    }
    expected_endpoint_binding = {
        **expected_endpoints,
        "endpoint_digest": stable_digest(expected_endpoints),
    }
    if conformance.get("endpoint_binding") != expected_endpoint_binding:
        raise ValueError(
            "source conformance receipt endpoint binding does not match "
            "the public-cloud topology"
        )
    if conformance.get("evidence_lock_digest") != evidence.get(
        "evidence_lock_digest"
    ):
        raise ValueError(
            "source conformance receipt evidence lock does not match"
        )
    if conformance.get("observed") != {
        "evaluation_roots": 2,
        "direct_children": 18,
        "predict_and_score_children": 16,
        "summarize_children": 2,
    }:
        raise ValueError(
            "source conformance receipt does not prove the locked "
            "18-child/16-prediction cohort"
        )
    locked_root_counts = {
        str(item.get("evaluation_call_id") or ""): {
            "Evaluation.predict_and_score": int(
                item.get("observed_predict_and_score_children") or 0
            ),
            "Evaluation.summarize": int(
                item.get("observed_summarize_children") or 0
            ),
        }
        for item in conformance.get("evaluation_roots") or ()
        if isinstance(item, Mapping)
        and str(item.get("evaluation_call_id") or "")
    }
    locked_root_ids = {
        str(item.get("call_id") or "")
        for item in c._mapping(
            evidence.get("objects"),
            "evidence lock objects",
        ).get("evaluations")
        or ()
        if isinstance(item, Mapping) and str(item.get("call_id") or "")
    }
    private_by_id = {
        str(item["id"]): item
        for item in c._load_private_labels(private_labels_path)
    }
    reconciliation = private_by_id.get(
        "maintainer-evaluation-reconciliation"
    )
    if reconciliation is None:
        return
    private_expected = c._mapping(
        reconciliation.get("expected"),
        "reconciliation private expected values",
    )
    fixture_root_ids = set(
        c._string_tuple(
            private_expected.get("evaluation_parent_ids") or (),
            "reconciliation Evaluation root",
        )
    )
    fixture_counts = c._mapping(
        c._mapping(
            private_expected.get("mechanism"),
            "reconciliation private mechanism",
        ).get("evaluation_parent_operation_counts"),
        "reconciliation private operation counts",
    )
    if fixture_root_ids != locked_root_ids or fixture_counts != locked_root_counts:
        raise ValueError(
            "reconciliation private truth disagrees with the approved "
            "evidence lock and source conformance receipt"
        )


def _validate_mechanism_reconciliation_binding(
    receipt: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    baseline_id: str,
    candidate_id: str,
) -> None:
    from fugue.bench import comparison as c

    findings = c._mapping(
        receipt.get("findings"),
        "mechanism receipt findings",
    )
    if (
        findings.get("baseline_evaluation_rows_reconciled") is not False
        or findings.get("candidate_evaluation_rows_reconciled") is not True
    ):
        raise ValueError(
            "mechanism receipt does not reproduce the baseline "
            "reconciliation bug and repaired candidate"
        )
    root_ids = {
        str(item.get("call_id") or "")
        for item in c._mapping(
            evidence.get("objects"),
            "evidence lock objects",
        ).get("evaluations")
        or ()
        if isinstance(item, Mapping) and str(item.get("call_id") or "")
    }
    candidates = {
        str(item.get("id") or ""): item
        for item in receipt.get("candidates") or ()
        if isinstance(item, Mapping)
    }
    for integration_id, repaired in (
        (baseline_id, False),
        (candidate_id, True),
    ):
        candidate = c._mapping(
            candidates.get(integration_id),
            f"mechanism candidate {integration_id}",
        )
        rows = [
            c._mapping(item, "mechanism reconciliation row")
            for item in candidate.get("evaluation_reconciliation") or ()
        ]
        if (
            len(rows) != 2
            or {
                str(item.get("evaluation_call_id") or "") for item in rows
            }
            != root_ids
            or any(
                int(item.get("locked_prediction_rows") or 0) != 8
                or int(
                    item.get("observed_predict_and_score_children") or 0
                )
                != 8
                or int(item.get("observed_summarize_children") or 0) != 1
                or item.get("trace_children_reconciled") is not True
                or item.get("prediction_rows_reconciled") is not repaired
                or int(item.get("tool_reported_total_predictions") or 0)
                != (8 if repaired else 9)
                for item in rows
            )
        ):
            raise ValueError(
                f"mechanism receipt {integration_id!r} does not prove the "
                "exact locked reconciliation behavior"
            )


def validate_release_candidate_binding(
    spec: Any,
    *,
    release_notes: Mapping[str, Any],
    repo_root: Path,
) -> None:
    """Bind this reference study's governed SHA to notes and MCP lock."""

    from fugue.bench import comparison as c

    policy = spec.decision_policy
    if policy is None:
        raise ValueError(
            "MCP release qualification requires a governed decision policy"
        )
    candidate_sha = policy.candidate_sha
    if release_notes.get("commit") != candidate_sha:
        raise ValueError(
            "decision-policy candidate SHA does not match the release-notes lock"
        )
    candidate_integrations = tuple(
        str(selection["id"]) for selection in spec.candidate.integrations
    )
    if len(candidate_integrations) != 1:
        raise ValueError(
            "MCP release qualification requires exactly one candidate integration"
        )
    integration_id = candidate_integrations[0]
    lock_path = (
        repo_root
        / ".fugue/imports/mcp/locks"
        / f"{integration_id}.json"
    )
    lock = c._load_json_object(
        c._safe_input_path(
            lock_path,
            repo_root,
            f"MCP release candidate lock {integration_id}",
        ),
        f"MCP release candidate lock {integration_id}",
    )
    if lock.get("version_identity") != f"git:{candidate_sha}":
        raise ValueError(
            "decision-policy candidate SHA does not match the candidate MCP lock"
        )
