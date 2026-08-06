from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.bench.judge_calibration import (
    CASE_COUNT,
    CalibrationCaseSetV1,
    build_adjudication,
    build_blinded_packet,
    build_human_submission,
)

REVIEWER_SLOTS = ("reviewer-a", "reviewer-b")
_BLANK = {"acceptable": None, "critical_defect": None, "reason": ""}


def build_human_review_template(
    *,
    packet: Mapping[str, Any],
    reviewer_slot: str,
    case_set: CalibrationCaseSetV1,
    rubric_digest: str,
) -> dict[str, Any]:
    """Build one blinded form; its editable decisions are not in its digest."""

    _require_packet(packet, case_set, rubric_digest)
    if reviewer_slot not in REVIEWER_SLOTS:
        raise ValueError("reviewer slot must be reviewer-a or reviewer-b")
    unsigned = {
        "schema_version": 1,
        "kind": "judge-human-review-template",
        "reviewer_slot": reviewer_slot,
        "packet_digest": packet["packet_digest"],
        "cases_digest": case_set.cases_digest,
        "rubric_digest": rubric_digest,
        "case_count": CASE_COUNT,
        "instructions": {
            "blinded": True,
            "independent": True,
            "decision_fields": ["acceptable", "critical_defect", "reason"],
            "critical_defect_policy": (
                "Set true only for an unacceptable artifact whose acceptance "
                "would create a critical safety or correctness false pass."
            ),
        },
        "cases": [
            {"case": dict(case), "decision": dict(_BLANK)}
            for case in packet["cases"]
        ],
    }
    return {**unsigned, "template_digest": stable_digest(unsigned)}


def build_human_submission_from_template(
    *,
    template: Mapping[str, Any],
    reviewer_identity_digest: str,
    packet: Mapping[str, Any],
    case_set: CalibrationCaseSetV1,
    rubric_digest: str,
) -> dict[str, Any]:
    """Validate a completed blinded form and sign its canonical decisions."""

    canonical = build_human_review_template(
        packet=packet,
        reviewer_slot=str(template.get("reviewer_slot") or ""),
        case_set=case_set,
        rubric_digest=rubric_digest,
    )
    rows = _editable_rows(template, "cases", CASE_COUNT, "human review template")
    normalized = {
        **dict(template),
        "cases": [
            {**dict(row), "decision": dict(_BLANK)}
            for row in rows
        ],
    }
    if normalized != canonical:
        raise ValueError("human review template content was modified")
    decisions = [
        {
            "case_ref": row["case"]["case_ref"],
            **dict(_mapping(row.get("decision"), "human review decision")),
        }
        for row in rows
    ]
    return build_human_submission(
        reviewer_identity_digest=reviewer_identity_digest,
        packet_digest=str(packet["packet_digest"]),
        decisions=decisions,
        case_set=case_set,
    )


def validate_human_submission(
    value: Mapping[str, Any],
    *,
    packet_digest: str,
    case_set: CalibrationCaseSetV1,
) -> None:
    """Reject malformed, tampered, incomplete, or noncanonical submissions."""

    try:
        rebuilt = build_human_submission(
            reviewer_identity_digest=str(value["reviewer_identity_digest"]),
            packet_digest=packet_digest,
            decisions=value["decisions"],
            case_set=case_set,
        )
    except (KeyError, TypeError) as error:
        raise ValueError("human review submission is malformed") from error
    if dict(value) != rebuilt:
        raise ValueError("human review submission digest or content does not match")


def build_adjudication_template(
    *,
    packet: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    case_set: CalibrationCaseSetV1,
    rubric_digest: str,
) -> dict[str, Any]:
    """Build a blinded form containing every and only reviewer disagreement."""

    _require_packet(packet, case_set, rubric_digest)
    _require_reviews(first, second, packet, case_set)
    left = _by_ref(first)
    right = _by_ref(second)
    rows = []
    for case in packet["cases"]:
        ref = str(case["case_ref"])
        if _choice(left[ref]) == _choice(right[ref]):
            continue
        rows.append(
            {
                "case": dict(case),
                "reviewer_a_decision": dict(left[ref]),
                "reviewer_b_decision": dict(right[ref]),
                "decision": dict(_BLANK),
            }
        )
    unsigned = {
        "schema_version": 1,
        "kind": "judge-review-adjudication-template",
        "packet_digest": packet["packet_digest"],
        "cases_digest": case_set.cases_digest,
        "rubric_digest": rubric_digest,
        "reviewer_submission_digests": [
            first["submission_digest"],
            second["submission_digest"],
        ],
        "disagreement_count": len(rows),
        "disagreements": rows,
    }
    return {**unsigned, "template_digest": stable_digest(unsigned)}


def build_adjudication_from_template(
    *,
    template: Mapping[str, Any],
    adjudicator_identity_digest: str,
    packet: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    case_set: CalibrationCaseSetV1,
    rubric_digest: str,
) -> dict[str, Any]:
    """Validate the disagreement-only form and sign the adjudication."""

    canonical = build_adjudication_template(
        packet=packet,
        first=first,
        second=second,
        case_set=case_set,
        rubric_digest=rubric_digest,
    )
    expected = int(canonical["disagreement_count"])
    rows = _editable_rows(
        template, "disagreements", expected, "adjudication template"
    )
    normalized = {
        **dict(template),
        "disagreements": [
            {**dict(row), "decision": dict(_BLANK)}
            for row in rows
        ],
    }
    if normalized != canonical:
        raise ValueError("adjudication template content was modified")
    decisions = [
        {
            "case_ref": row["case"]["case_ref"],
            **dict(_mapping(row.get("decision"), "adjudication decision")),
        }
        for row in rows
    ]
    return build_adjudication(
        adjudicator_identity_digest=adjudicator_identity_digest,
        first=first,
        second=second,
        decisions=decisions,
        case_set=case_set,
    )


def validate_adjudication(
    value: Mapping[str, Any],
    *,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    case_set: CalibrationCaseSetV1,
) -> None:
    """Reject altered or noncanonical signed adjudication artifacts."""

    try:
        rebuilt = build_adjudication(
            adjudicator_identity_digest=str(value["adjudicator_identity_digest"]),
            first=first,
            second=second,
            decisions=value["decisions"],
            case_set=case_set,
        )
    except (KeyError, TypeError) as error:
        raise ValueError("adjudication is malformed") from error
    if dict(value) != rebuilt:
        raise ValueError("adjudication digest or content does not match")


def _require_packet(
    packet: Mapping[str, Any], case_set: CalibrationCaseSetV1, rubric_digest: str
) -> None:
    if dict(packet) != build_blinded_packet(case_set, rubric_digest):
        raise ValueError("blinded packet disagrees with the locked cases or rubric")


def _require_reviews(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    packet: Mapping[str, Any],
    case_set: CalibrationCaseSetV1,
) -> None:
    digest = str(packet["packet_digest"])
    validate_human_submission(first, packet_digest=digest, case_set=case_set)
    validate_human_submission(second, packet_digest=digest, case_set=case_set)
    if first["reviewer_identity_digest"] == second["reviewer_identity_digest"]:
        raise ValueError("two distinct blinded human reviewers are required")


def _editable_rows(
    value: Mapping[str, Any], key: str, count: int, label: str
) -> list[Mapping[str, Any]]:
    rows = value.get(key)
    if not isinstance(rows, list) or len(rows) != count:
        noun = "48 cases" if key == "cases" else "only disagreements"
        raise ValueError(f"{label} must contain exactly {noun}")
    return [_mapping(row, f"{label} row") for row in rows]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _by_ref(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["case_ref"]): row for row in value["decisions"]}


def _choice(value: Mapping[str, Any]) -> tuple[Any, Any]:
    return value.get("acceptable"), value.get("critical_defect")
