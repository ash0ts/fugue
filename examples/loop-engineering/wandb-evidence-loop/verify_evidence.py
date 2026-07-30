from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fugue.bench.files import atomic_write_json
from fugue.bench.loop_failure import read_comparison_failure_lock
from fugue.bench.runtime_provenance import resolve_fugue_source_provenance
from fugue.bench.scoring import read_intervention_selection_lock

LOOP_PROJECT = "wandb/fugue-claude-loop-engineering-v1"
_PRIVATE_KEYS = {
    "expected",
    "expected_answer",
    "expected_values",
    "private_evaluation",
    "private_label",
    "private_labels",
}
_CREDENTIAL_NAMES = {
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "WANDB_API_KEY",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        values.append(value)
    return values


def _coordinate(row: Mapping[str, Any]) -> tuple[str, str, int]:
    task = str(
        row.get("comparison_example_id")
        or row.get("task_id")
        or row.get("task_name")
        or ""
    )
    harness = str(row.get("harness") or "")
    attempt = int(row.get("trial_index") or 0)
    if not task or not harness or attempt < 1:
        raise ValueError("every row requires task, harness, and positive attempt")
    return task, harness, attempt


def _tool_names(row: Mapping[str, Any]) -> set[str]:
    value = row.get("weave_tool_names") or row.get("mcp_tool_names")
    if isinstance(value, Mapping):
        return {
            str(name)
            for name, count in value.items()
            if float(count or 0) > 0
        }
    if isinstance(value, list):
        return {str(name) for name in value}
    return set()


def _skill_use_observed(row: Mapping[str, Any], skill_id: str) -> bool:
    evidence = row.get("skill_invocation_evidence")
    if not isinstance(evidence, Mapping):
        return False
    invoked = {str(value) for value in evidence.get("skills_invoked") or ()}
    return evidence.get("status") == "observed" and skill_id in invoked


def _integration_ids(row: Mapping[str, Any]) -> set[str]:
    direct = {str(value) for value in row.get("integration_ids") or ()}
    provenance = row.get("integration_provenance")
    if isinstance(provenance, list):
        direct.update(
            str(value.get("id") or "")
            for value in provenance
            if isinstance(value, Mapping)
        )
    return {value for value in direct if value}


def _source_failure(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = row.get("source_failure")
    if isinstance(direct, Mapping):
        return direct
    metadata = row.get("task_metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get("source_failure")
        if isinstance(nested, Mapping):
            return nested
    return None


def _keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _keys(nested)


def _dotenv_secrets(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    secrets: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() not in _CREDENTIAL_NAMES:
            continue
        normalized = value.strip().strip("\"'")
        if len(normalized) >= 8:
            secrets.append(normalized)
    return tuple(secrets)


def _evidence_issues(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    issues: list[str] = []
    for row in rows:
        identity = str(
            row.get("attempt_id")
            or row.get("prediction_id")
            or row.get("run_key")
            or "unknown"
        )
        if row.get("trace_project") != LOOP_PROJECT:
            issues.append(f"{identity}: evidence routed outside the loop project")
        if row.get("trace_link_status") != "linked":
            issues.append(f"{identity}: native Agent root is not linked")
        if row.get("evaluation_publication_mode") != "live":
            issues.append(f"{identity}: live Weave Evaluation row is missing")
        if row.get("harbor_environment") != "local_harbor_docker":
            issues.append(f"{identity}: local Harbor identity is missing")
        if row.get("harbor_conformance_status") != "passed":
            issues.append(f"{identity}: Harbor conformance did not pass")
        if row.get("harbor_policy_attestation_verified") is not True:
            issues.append(f"{identity}: Harbor policy attestation is unverified")
        if (
            row.get("privacy_contract_version") != 2
            or row.get("local_artifact_privacy_scan_status") != "passed"
            or row.get("hosted_evidence_privacy_scan_status") != "passed"
            or row.get("private_label_boundary_verified") is not True
        ):
            issues.append(f"{identity}: privacy evidence is incomplete")
        if (
            row.get("harbor_cleanup_verified") is not True
            or row.get("orphaned_sandbox") is not False
        ):
            issues.append(f"{identity}: run-scoped cleanup is incomplete")
        if not row.get("attempt_id") or not row.get("prediction_id"):
            issues.append(f"{identity}: attempt or prediction identity is missing")
        if not row.get("execution_fingerprint"):
            issues.append(f"{identity}: execution fingerprint is missing")
        if not row.get("run_snapshot_sha256"):
            issues.append(f"{identity}: run snapshot digest is missing")
    return issues


def verify(  # noqa: C901 - one bounded receipt reports every qualification gate
    *,
    discovery_path: Path,
    holdout_path: Path,
    selection_lock_path: Path,
    failure_lock_path: Path,
    repo_root: Path,
    env_file: Path | None = None,
) -> dict[str, Any]:
    discovery = _rows(discovery_path)
    holdout = _rows(holdout_path)
    selection = read_intervention_selection_lock(selection_lock_path)
    failure = read_comparison_failure_lock(failure_lock_path)
    blockers: list[str] = []

    if len(discovery) != 8:
        blockers.append(f"discovery requires exactly 8 rows, found {len(discovery)}")
    if len(holdout) != 8:
        blockers.append(f"holdout requires exactly 8 rows, found {len(holdout)}")

    discovery_counts = Counter(
        str(row.get("variant_id") or "") for row in discovery
    )
    expected_discovery = Counter(
        {"production": 2, "skill-only": 2, "mcp-only": 2, "combined": 2}
    )
    if discovery_counts != expected_discovery:
        blockers.append("discovery is not the complete two-task four-arm matrix")

    selected = selection.selected_variant_id
    holdout_counts = Counter(str(row.get("variant_id") or "") for row in holdout)
    if holdout_counts != Counter({"production": 4, selected: 4}):
        blockers.append("holdout is not four aligned production/winner tasks")

    for phase, rows, counts in (
        ("discovery", discovery, discovery_counts),
        ("holdout", holdout, holdout_counts),
    ):
        coordinates = {
            variant: {
                _coordinate(row)
                for row in rows
                if row.get("variant_id") == variant
            }
            for variant in counts
        }
        if coordinates and len(
            {frozenset(value) for value in coordinates.values()}
        ) != 1:
            blockers.append(f"{phase} candidates are not aligned on exact cells")

    all_rows = [*discovery, *holdout]
    attempt_ids = [str(row.get("attempt_id") or "") for row in all_rows]
    prediction_ids = [str(row.get("prediction_id") or "") for row in all_rows]
    if (
        any(not value for value in attempt_ids)
        or len(attempt_ids) != len(set(attempt_ids))
        or any(not value for value in prediction_ids)
        or len(prediction_ids) != len(set(prediction_ids))
    ):
        blockers.append("attempt/prediction identities are missing or duplicated")

    ranking_candidates = {
        str(value["variant_id"]): str(value["candidate_digest"])
        for value in selection.rankings
    }
    for row in all_rows:
        variant = str(row.get("variant_id") or "")
        if ranking_candidates.get(variant) != str(row.get("candidate_id") or ""):
            blockers.append(f"{variant or 'unknown'} differs from selection lock")
            break

    discovery_snapshots = {
        str(row.get("run_snapshot_sha256") or "") for row in discovery
    }
    if not discovery_snapshots or not discovery_snapshots <= set(
        selection.discovery_run_snapshot_sha256s
    ):
        blockers.append("discovery rows are not bound to selection snapshots")
    if {
        str(
            row.get("comparison_example_id")
            or row.get("task_id")
            or row.get("task_name")
            or ""
        )
        for row in discovery
    } != set(selection.comparison_example_ids):
        blockers.append("discovery examples differ from the selection lock")

    source = resolve_fugue_source_provenance(repo_root)
    if (
        str(source.get("commit") or "") != selection.source_commit
        or str(source.get("tree") or "") != selection.source_tree
        or bool(source.get("dirty"))
        or str(source.get("dirty_digest") or "")
        != selection.source_dirty_digest
    ):
        blockers.append("current checkout differs from the qualified clean source")
    if any(
        str(row.get("source_commit") or "") != selection.source_commit
        or str(row.get("source_tree") or "") != selection.source_tree
        or str(row.get("source_dirty_digest") or "")
        != selection.source_dirty_digest
        for row in all_rows
    ):
        blockers.append("one or more cells differ from the qualified source tree")

    failure_identity = failure["failure"]
    reproduced = [
        row
        for row in discovery
        if row.get("variant_id") == "production"
        and row.get("pass") is False
        and (source_failure := _source_failure(row)) is not None
        and source_failure.get("lock_sha256") == failure["lock_sha256"]
        and source_failure.get("task_id") == failure_identity["task_id"]
        and source_failure.get("primary_attempt_id")
        == failure_identity["primary_attempt_id"]
    ]
    if not reproduced:
        blockers.append("the locked real source failure was not reproduced")

    baseline_by_coordinate = {
        _coordinate(row): row.get("pass") is True
        for row in discovery
        if row.get("variant_id") == "production"
    }
    selected_improvements = sum(
        row.get("pass") is True
        and baseline_by_coordinate.get(_coordinate(row)) is False
        for row in discovery
        if row.get("variant_id") == selected
    )
    if selected_improvements < 1:
        blockers.append("selected arm has no paired deterministic discovery gain")

    holdout_baseline = {
        _coordinate(row): row.get("pass") is True
        for row in holdout
        if row.get("variant_id") == "production"
    }
    holdout_regressions = sum(
        row.get("pass") is not True
        and holdout_baseline.get(_coordinate(row)) is True
        for row in holdout
        if row.get("variant_id") == selected
    )
    if holdout_regressions:
        blockers.append(
            f"selected arm has {holdout_regressions} critical holdout regression(s)"
        )

    selected_rows = [
        row for row in [*discovery, *holdout] if row.get("variant_id") == selected
    ]
    if selected in {"skill-only", "combined"} and any(
        not _skill_use_observed(row, "loop-intervention-skill")
        for row in selected_rows
    ):
        blockers.append("selected Skill intervention use is unproven")
    if selected in {"mcp-only", "combined"} and any(
        "loop-intervention-mcp" not in _integration_ids(row)
        or not _tool_names(row)
        for row in selected_rows
    ):
        blockers.append("selected MCP intervention use is unproven")
    if any(not _tool_names(row) for row in all_rows):
        blockers.append("one or more cells have no observed MCP tool invocation")

    blockers.extend(_evidence_issues(all_rows))
    phase_fingerprints = {
        phase: {
            str(row.get("execution_fingerprint") or "")
            for row in rows
            if row.get("execution_fingerprint")
        }
        for phase, rows in (("discovery", discovery), ("holdout", holdout))
    }
    if any(not values for values in phase_fingerprints.values()):
        blockers.append("a phase lacks an execution fingerprint")

    serialized = json.dumps(all_rows, sort_keys=True)
    private_keys = sorted(set(_keys(all_rows)) & _PRIVATE_KEYS)
    if private_keys:
        blockers.append("private evaluation fields leaked: " + ", ".join(private_keys))
    if any(secret in serialized for secret in _dotenv_secrets(env_file)):
        blockers.append("a credential value appears in exported evidence")
    if "news" + "-research-agent" in serialized:
        blockers.append("legacy project routing appears in loop evidence")

    return {
        "schema_version": 1,
        "kind": "claude-loop-qualification-receipt",
        "qualified": not blockers,
        "discovery_rows": len(discovery),
        "holdout_rows": len(holdout),
        "selected_variant_id": selected,
        "selection_lock_sha256": selection.lock_sha256,
        "failure_lock_sha256": failure["lock_sha256"],
        "source_commit": selection.source_commit,
        "source_tree": selection.source_tree,
        "discovery_candidate_improvements": selected_improvements,
        "holdout_regressions": holdout_regressions,
        "phase_execution_fingerprints": {
            key: sorted(value) for key, value in phase_fingerprints.items()
        },
        "blockers": blockers,
        "claim_limitation": (
            "Qualification applies only to the exact failure, task locks, "
            "candidates, and local Harbor execution; it is not package-release "
            "or W&B Serverless evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the exact 8+8 Claude loop evidence boundary."
    )
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--failure-lock", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(
        discovery_path=args.discovery,
        holdout_path=args.holdout,
        selection_lock_path=args.selection_lock,
        failure_lock_path=args.failure_lock,
        repo_root=args.repo_root.resolve(),
        env_file=args.env_file,
    )
    atomic_write_json(args.output.resolve(), receipt, mode=0o600)
    print(
        json.dumps(
            {
                "qualified": receipt["qualified"],
                "discovery_rows": receipt["discovery_rows"],
                "holdout_rows": receipt["holdout_rows"],
                "selected_variant_id": receipt["selected_variant_id"],
                "blocker_count": len(receipt["blockers"]),
                "output": args.output.resolve().as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["qualified"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
