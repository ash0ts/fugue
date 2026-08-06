#!/usr/bin/env python3
"""Operator-only validation, exposure audit, and admission for sealed holdouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from holdout_support import (  # noqa: E402
    build_historical_holdout_exposure_receipt,
    build_live_holdout_audits,
    build_no_skill_diagnostic,
    build_sealed_holdout_comparison,
    fetch_task_identity_projections,
    load_sealed_holdout_manifest,
    read_historical_holdout_exposure_receipt,
    read_sealed_holdout_preparation,
    validate_sealed_holdouts_zero_model,
    write_historical_holdout_exposure_receipt,
    write_live_holdout_audits,
)

from fugue.bench.comparison import read_comparison_result  # noqa: E402
from fugue.bench.operator import load_env  # noqa: E402
from fugue.bench.study_advancement import (  # noqa: E402
    build_study_advancement_decision,
    read_holdout_exposure_audit,
    write_study_advancement_decision,
)

CAMPAIGN = Path("examples/comparisons/community-skill-selected-v1")
MANIFEST = CAMPAIGN / "sealed-holdouts.json"
LANES = (
    "superpowers-writing-plans",
    "anthropic-skill-creator",
    "vercel-react-best-practices",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    advance = commands.add_parser("advance")
    advance.add_argument("--result", type=Path, required=True)
    advance.add_argument("--audit", type=Path)
    advance.add_argument("--output", type=Path, required=True)
    history = commands.add_parser("audit-history")
    history.add_argument("--env-file", type=Path, required=True)
    history.add_argument("--operator-source", type=Path, required=True)
    history.add_argument("--reviewed-by", required=True)
    history.add_argument("--output", type=Path)
    audit = commands.add_parser("audit")
    audit.add_argument("--env-file", type=Path, required=True)
    audit.add_argument(
        "--replacement",
        action="append",
        default=[],
        metavar="EXPOSED=RESERVE",
    )
    diagnostic = commands.add_parser("diagnostic")
    diagnostic.add_argument("--lane", choices=LANES, required=True)
    diagnostic.add_argument("--decision", type=Path, required=True)
    holdout = commands.add_parser("holdout")
    holdout.add_argument("--lane", choices=LANES, required=True)
    holdout.add_argument("--decision", type=Path, required=True)
    holdout.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest_path = root / MANIFEST

    if args.command == "validate":
        receipt = read_sealed_holdout_preparation(root)
        result = {
            "preparation": receipt,
            "zero_model": validate_sealed_holdouts_zero_model(
                repo_root=root,
                preparation_receipt=receipt,
            ),
        }
    elif args.command == "advance":
        result_path = args.result if args.result.is_absolute() else root / args.result
        result_value = read_comparison_result(result_path)
        audit_value = None
        if args.audit is not None:
            audit_path = args.audit if args.audit.is_absolute() else root / args.audit
            audit_value = read_holdout_exposure_audit(audit_path)
        decision_value = build_study_advancement_decision(
            result_value,
            holdout_audit=audit_value,
        )
        output_path = args.output if args.output.is_absolute() else root / args.output
        write_study_advancement_decision(output_path, decision_value)
        result = {
            "status": decision_value.status,
            "study_id": decision_value.study_id,
            "decision_digest": decision_value.decision_digest,
            "output": str(output_path),
        }
    elif args.command == "audit-history":
        manifest = load_sealed_holdout_manifest(manifest_path)
        env_file = (
            args.env_file if args.env_file.is_absolute() else root / args.env_file
        )
        operator_source = (
            args.operator_source
            if args.operator_source.is_absolute()
            else root / args.operator_source
        )
        project_rows, endpoint, coverage = fetch_task_identity_projections(
            projects=manifest["audit"]["historical_projects"],
            env=load_env(env_file),
        )
        receipt = build_historical_holdout_exposure_receipt(
            manifest_path=manifest_path,
            operator_source=operator_source,
            project_rows=project_rows,
            project_coverage=coverage,
            reviewer_identity=args.reviewed_by,
            trace_endpoint=endpoint,
        )
        output = args.output or (
            operator_source
            / manifest["historical_exposure_receipt"]["operator_filename"]
        )
        output = output if output.is_absolute() else root / output
        write_historical_holdout_exposure_receipt(output, receipt)
        result = {
            "status": receipt["status"],
            "receipt_digest": receipt["receipt_digest"],
            "reviewer_identity_digest": receipt["reviewer_identity_digest"],
            "searched_project_count": len(receipt["searched_project_refs"]),
            "searched_call_count": receipt["searched_call_count"],
            "required_replacements": receipt["required_replacements"],
            "output": str(output),
            "outcome_data_consulted": False,
        }
    elif args.command == "audit":
        receipt = read_sealed_holdout_preparation(root)
        historical = read_historical_holdout_exposure_receipt(
            root, preparation_receipt=receipt
        )
        manifest = load_sealed_holdout_manifest(manifest_path)
        env_file = (
            args.env_file if args.env_file.is_absolute() else root / args.env_file
        )
        project_rows, endpoint, coverage = fetch_task_identity_projections(
            projects=manifest["audit"]["historical_projects"],
            env=load_env(env_file),
        )
        replacements = _replacements(args.replacement)
        audits = build_live_holdout_audits(
            manifest_path=manifest_path,
            preparation_receipt=receipt,
            historical_receipt=historical,
            project_rows=project_rows,
            project_coverage=coverage,
            replacements=replacements,
        )
        result = {
            "status": "clear" if not replacements else "replaced_exposed",
            "trace_endpoint": endpoint,
            "outcome_data_consulted": False,
            "queried_fields": manifest["audit"]["queried_fields"],
            "paths": write_live_holdout_audits(repo_root=root, audits=audits),
            "audits": {
                lane: {
                    "audit_digest": value.audit_digest,
                    "selected_task_ids": list(value.selected_task_ids),
                    "matched_task_ids": list(value.matched_task_ids),
                    "expires_at": value.expires_at,
                }
                for lane, value in audits.items()
            },
        }
    elif args.command == "diagnostic":
        comparison = root / CAMPAIGN / args.lane / "comparison.yaml"
        decision = (
            args.decision if args.decision.is_absolute() else root / args.decision
        )
        result = build_no_skill_diagnostic(
            comparison_path=comparison,
            advancement_decision_path=decision,
            repo_root=root,
        )
    else:
        comparison = root / CAMPAIGN / args.lane / "comparison.yaml"
        decision = (
            args.decision if args.decision.is_absolute() else root / args.decision
        )
        audit = args.audit if args.audit.is_absolute() else root / args.audit
        result = build_sealed_holdout_comparison(
            comparison_path=comparison,
            advancement_decision_path=decision,
            exposure_audit_path=audit,
            repo_root=root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _replacements(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError("holdout replacement must be EXPOSED=RESERVE")
        exposed, reserve = value.split("=", 1)
        if not exposed or not reserve or exposed in result:
            raise ValueError("holdout replacement identity is invalid")
        result[exposed] = reserve
    return result


if __name__ == "__main__":
    raise SystemExit(main())
