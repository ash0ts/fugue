from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fugue.bench.files import atomic_write_json
from fugue.bench.judge_calibration import (
    build_blinded_packet,
    build_final_receipt,
    build_model_output_receipt,
    load_case_set,
    validate_rubric,
    write_receipt,
)
from fugue.bench.judge_calibration_review import (
    build_adjudication_from_template,
    build_adjudication_template,
    build_human_review_template,
    build_human_submission_from_template,
    validate_adjudication,
    validate_human_submission,
)
from fugue.bench.judge_calibration_run import validate_generation_receipt


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    case_set = load_case_set(args.manifest, args.cases)
    rubric, rubric_digest = validate_rubric(args.rubric)
    packet = _read_object(args.packet, "blinded packet")
    if packet != build_blinded_packet(case_set, rubric_digest):
        raise ValueError("blinded packet disagrees with the locked cases or rubric")

    if args.action == "review-templates":
        outputs = []
        for slot, argument in (
            ("reviewer-a", args.reviewer_a_out),
            ("reviewer-b", args.reviewer_b_out),
        ):
            destination = _required_path(argument, f"--{slot}-out")
            template = build_human_review_template(
                packet=packet,
                reviewer_slot=slot,
                case_set=case_set,
                rubric_digest=rubric_digest,
            )
            selected = _private_output(root, destination)
            _write_new_or_identical(selected, template, "review template")
            outputs.append(
                {
                    "reviewer_slot": slot,
                    "path": selected.relative_to(root).as_posix(),
                    "template_digest": template["template_digest"],
                    "case_count": template["case_count"],
                }
            )
        _print_json(
            {
                "status": "awaiting_two_independent_human_reviews",
                "packet_digest": packet["packet_digest"],
                "templates": outputs,
                "authored_truth_included": False,
            }
        )
        return 0

    if args.action == "submit-review":
        template = _read_object(
            _required_path(args.review_template, "--review-template"),
            "completed review template",
        )
        submission = build_human_submission_from_template(
            template=template,
            reviewer_identity_digest=_required_text(
                args.reviewer_identity_digest, "--reviewer-identity-digest"
            ),
            packet=packet,
            case_set=case_set,
            rubric_digest=rubric_digest,
        )
        destination = _private_output(
            root, _required_path(args.submission_out, "--submission-out")
        )
        _write_new_or_identical(destination, submission, "human submission")
        _print_json(
            {
                "status": "signed_complete_human_review",
                "reviewer_slot": template["reviewer_slot"],
                "case_count": len(submission["decisions"]),
                "submission_digest": submission["submission_digest"],
                "path": destination.relative_to(root).as_posix(),
            }
        )
        return 0

    first, second = _load_reviews(args, packet, case_set)
    if args.action == "adjudication-template":
        template = build_adjudication_template(
            packet=packet,
            first=first,
            second=second,
            case_set=case_set,
            rubric_digest=rubric_digest,
        )
        destination = _private_output(
            root,
            _required_path(
                args.adjudication_template_out, "--adjudication-template-out"
            ),
        )
        _write_new_or_identical(destination, template, "adjudication template")
        _print_json(
            {
                "status": "awaiting_disagreement_adjudication",
                "disagreement_count": template["disagreement_count"],
                "template_digest": template["template_digest"],
                "path": destination.relative_to(root).as_posix(),
                "authored_truth_included": False,
            }
        )
        return 0

    if args.action == "submit-adjudication":
        template = _read_object(
            _required_path(args.adjudication_template, "--adjudication-template"),
            "completed adjudication template",
        )
        adjudication = build_adjudication_from_template(
            template=template,
            adjudicator_identity_digest=_required_text(
                args.adjudicator_identity_digest, "--adjudicator-identity-digest"
            ),
            packet=packet,
            first=first,
            second=second,
            case_set=case_set,
            rubric_digest=rubric_digest,
        )
        destination = _private_output(
            root, _required_path(args.adjudication_out, "--adjudication-out")
        )
        _write_new_or_identical(destination, adjudication, "adjudication")
        _print_json(
            {
                "status": "signed_complete_adjudication",
                "disagreement_count": len(adjudication["decisions"]),
                "adjudication_digest": adjudication["adjudication_digest"],
                "path": destination.relative_to(root).as_posix(),
            }
        )
        return 0

    generation = _read_object(
        _required_path(args.generation_receipt, "--generation-receipt"),
        "generation receipt",
    )
    validate_generation_receipt(generation)
    model_receipt, outputs = build_model_output_receipt(
        case_set,
        rubric,
        rubric_digest,
        packet,
        _required_path(args.model_outputs, "--model-outputs"),
    )
    if model_receipt != generation["model_output_receipt"]:
        raise ValueError("model outputs disagree with the locked generation receipt")
    adjudication = _read_object(
        _required_path(args.adjudication, "--adjudication"),
        "signed adjudication",
    )
    validate_adjudication(
        adjudication, first=first, second=second, case_set=case_set
    )
    receipt = build_final_receipt(
        case_set=case_set,
        rubric_digest=rubric_digest,
        model_receipt=model_receipt,
        outputs=outputs,
        first=first,
        second=second,
        adjudication=adjudication,
        evaluated_agent_model_family=args.evaluated_agent_model_family,
    )
    destination = _private_output(
        root, _required_path(args.final_receipt_out, "--final-receipt-out")
    )
    if destination.exists():
        if _read_object(destination, "existing final receipt") != receipt:
            raise ValueError("existing final receipt disagrees with recomputation")
    else:
        write_receipt(destination, receipt)
    _print_final_metrics(receipt, destination.relative_to(root).as_posix())
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create and validate two real blinded human reviews, adjudicate only "
            "their disagreements, and finalize the strict calibration receipt."
        )
    )
    parser.add_argument(
        "action",
        choices=(
            "review-templates",
            "submit-review",
            "adjudication-template",
            "submit-adjudication",
            "finalize",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    for flag in (
        "reviewer-a-out",
        "reviewer-b-out",
        "review-template",
        "submission-out",
        "first-submission",
        "second-submission",
        "adjudication-template-out",
        "adjudication-template",
        "adjudication-out",
        "adjudication",
        "model-outputs",
        "generation-receipt",
        "final-receipt-out",
    ):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--reviewer-identity-digest")
    parser.add_argument("--adjudicator-identity-digest")
    parser.add_argument("--evaluated-agent-model-family", default="anthropic")
    return parser


def _load_reviews(
    args: argparse.Namespace,
    packet: Mapping[str, Any],
    case_set: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    first = _read_object(
        _required_path(args.first_submission, "--first-submission"),
        "first human submission",
    )
    second = _read_object(
        _required_path(args.second_submission, "--second-submission"),
        "second human submission",
    )
    digest = str(packet["packet_digest"])
    validate_human_submission(first, packet_digest=digest, case_set=case_set)
    validate_human_submission(second, packet_digest=digest, case_set=case_set)
    if first["reviewer_identity_digest"] == second["reviewer_identity_digest"]:
        raise ValueError("two distinct blinded human reviewers are required")
    return first, second


def _private_output(root: Path, path: Path) -> Path:
    selected = path if path.is_absolute() else root / path
    resolved = selected.resolve()
    private_root = (root / ".fugue/private").resolve()
    if not resolved.is_relative_to(private_root) or selected.is_symlink():
        raise ValueError("review outputs must remain under .fugue/private")
    return resolved


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_new_or_identical(
    path: Path, value: Mapping[str, Any], label: str
) -> None:
    if path.exists():
        if _read_object(path, f"existing {label}") != value:
            raise ValueError(f"existing {label} disagrees with recomputation")
        return
    atomic_write_json(path, dict(value))


def _required_path(value: Path | None, flag: str) -> Path:
    if value is None:
        raise ValueError(f"{flag} is required for this action")
    return value


def _required_text(value: str | None, flag: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{flag} is required for this action")
    return value


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _print_final_metrics(receipt: Mapping[str, Any], path: str) -> None:
    overall = receipt["overall"]
    print(
        f"Calibration {str(receipt['status']).upper()} "
        f"(advisory judge; same model family: {receipt['same_model_family']})"
    )
    print(
        "Overall: "
        f"TPR={float(overall['true_positive_rate']):.1%} "
        f"TNR={float(overall['true_negative_rate']):.1%}"
    )
    for modality, metrics in sorted(receipt["modalities"].items()):
        print(
            f"{modality}: "
            f"TPR={float(metrics['true_positive_rate']):.1%} "
            f"TNR={float(metrics['true_negative_rate']):.1%}"
        )
    print(
        "Critical false passes: "
        f"{receipt['critical_false_passes']} (maximum 0)"
    )
    print(f"Receipt: {path} ({receipt['receipt_digest']})")
