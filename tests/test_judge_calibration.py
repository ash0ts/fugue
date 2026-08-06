from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.judge_calibration import (
    LABELS,
    MODALITIES,
    PASSING_LABELS,
    build_adjudication,
    build_blinded_packet,
    build_final_receipt,
    build_human_submission,
    build_model_output_receipt,
    load_case_set,
    validate_final_receipt,
    validate_model_output_receipt,
    validate_rubric,
    write_receipt,
)

EXAMPLE = (
    Path(__file__).parents[1]
    / "examples/comparisons/community-skill-selected-v1/judge"
)
REPOSITORIES = {
    "code-change": "vercel-react-best-practices",
    "implementation-plan": "superpowers-writing-plans",
    "skill-package": "anthropic-skill-creator",
}


def _digest(character: str) -> str:
    return character * 64


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_rows() -> list[dict[str, Any]]:
    rows = []
    for modality in MODALITIES:
        for index in range(16):
            acceptable = index < 8
            rows.append(
                {
                    "id": f"{modality}-{index:02d}-{'pass' if acceptable else 'fail'}",
                    "repository_id": REPOSITORIES[modality],
                    "scenario_id": f"scenario-{index:02d}",
                    "split": "calibration",
                    "public_task": {"question": f"Review artifact {index}."},
                    "response": f"Sanitized {modality} artifact {index}.",
                    "permitted_evidence": {"inspected_paths": ["src/example.py"]},
                    "authored_reference": {
                        "label": "pass" if acceptable else "fail",
                        "critical_false_pass": not acceptable and index % 2 == 0,
                    },
                    "judge_result": None,
                    "reviews": [],
                    "adjudicated_label": None,
                }
            )
    return rows


def _case_set(tmp_path: Path, rows: list[dict[str, Any]] | None = None):
    cases_path = tmp_path / "cases.jsonl"
    digest = _write_jsonl(cases_path, rows or _case_rows())
    manifest = {
        "schema_version": 1,
        "id": "cases-v1",
        "case_count": 48,
        "cases_digest": digest,
        "source_visibility": "operator-restricted",
        "repository_modalities": {
            repository: modality for modality, repository in REPOSITORIES.items()
        },
        "modalities": {
            modality: {"acceptable": 8, "defective": 8}
            for modality in MODALITIES
        },
        "review_policy": {
            "independent_blinded_reviewers": 2,
            "adjudicate_disagreements": True,
            "minimum_tpr_tnr": 0.85,
            "critical_false_passes_max": 0,
        },
        "note": "Contents are operator-restricted.",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_case_set(manifest_path, cases_path), cases_path, manifest_path


def _rubric():
    return validate_rubric(EXAMPLE / "rubric.json")


def _outputs(case_set, rubric: dict[str, Any], *, overrides=None):
    dimensions = {
        name: value["dimensions"] for name, value in rubric["modalities"].items()
    }
    override_values = overrides or {}
    rows = []
    for index, case in enumerate(case_set.cases):
        authored_pass = int(case.case_ref, 16)  # stable ordering is opaque to reviewers
        label = override_values.get(case.case_ref)
        if label is None:
            # The private fixture's source id is unavailable here; use a caller-provided map.
            label = "adequate" if index % 16 < 8 else "weak"
        rows.append(
            {
                "case_ref": case.case_ref,
                "modality": case.modality,
                "label": label,
                "dimension_labels": {
                    name: label for name in dimensions[case.modality]
                },
                "reason": f"Blinded reason {authored_pass % 997}.",
                "missing_evidence": False,
            }
        )
    return rows


def _truth(case_set):
    # Case refs sort by digest rather than authored label, so recover truth from fixtures.
    return {
        stable_digest(
            {"cases_digest": case_set.cases_digest, "source_id": row["id"]}
        ): row["authored_reference"]["label"] == "pass"
        for row in _case_rows()
    }


def _output_bundle(tmp_path: Path, case_set, *, override=None):
    rubric, rubric_digest = _rubric()
    packet = build_blinded_packet(case_set, rubric_digest)
    truth = _truth(case_set)
    labels = {
        ref: ("adequate" if acceptable else "weak") for ref, acceptable in truth.items()
    }
    labels.update(override or {})
    path = tmp_path / "outputs.jsonl"
    _write_jsonl(path, _outputs(case_set, dict(rubric), overrides=labels))
    receipt, outputs = build_model_output_receipt(
        case_set, rubric, rubric_digest, packet, path
    )
    return rubric_digest, packet, receipt, outputs, truth


def _decisions(case_set, truth, *, changes=None):
    selected = dict(truth)
    selected.update(changes or {})
    return [
        {
            "case_ref": case.case_ref,
            "acceptable": selected[case.case_ref],
            "critical_defect": not selected[case.case_ref] and case.case_ref.endswith("0"),
            "reason": "Independent blinded review.",
        }
        for case in case_set.cases
    ]


def _reviews(case_set, packet, truth, *, second_changes=None):
    first = build_human_submission(
        reviewer_identity_digest=_digest("a"),
        packet_digest=packet["packet_digest"],
        decisions=_decisions(case_set, truth),
        case_set=case_set,
    )
    second = build_human_submission(
        reviewer_identity_digest=_digest("b"),
        packet_digest=packet["packet_digest"],
        decisions=_decisions(case_set, truth, changes=second_changes),
        case_set=case_set,
    )
    return first, second


def test_locked_cases_are_balanced_and_packet_is_blinded(tmp_path: Path) -> None:
    case_set, _cases, _manifest = _case_set(tmp_path)
    _rubric_value, rubric_digest = _rubric()
    packet = build_blinded_packet(case_set, rubric_digest)
    serialized = json.dumps(packet)

    assert len(case_set.cases) == 48
    assert len({case.case_ref for case in case_set.cases}) == 48
    assert "authored_reference" not in serialized
    assert "skill_revision" not in serialized
    assert all(len(case["case_ref"]) == 64 for case in packet["cases"])
    assert packet["packet_digest"]


def test_model_output_receipt_is_strict_private_clean_and_immutable(
    tmp_path: Path,
) -> None:
    case_set, *_ = _case_set(tmp_path)
    _rubric_digest, _packet, receipt, outputs, _truth_map = _output_bundle(
        tmp_path, case_set
    )

    assert len(outputs) == 48
    assert receipt["privacy_scan"] == "passed"
    validate_model_output_receipt(receipt)
    tampered = dict(receipt)
    tampered["case_count"] = 47
    with pytest.raises(ValueError, match="48-case|digest"):
        validate_model_output_receipt(tampered)
    unknown = {**receipt, "raw_output": []}
    with pytest.raises(ValueError, match="fields do not match"):
        validate_model_output_receipt(unknown)


def test_old_generic_outputs_require_regeneration(tmp_path: Path) -> None:
    status = json.loads(
        (EXAMPLE / "legacy-model-output-status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "requires_regeneration"
    assert status["reuse_allowed"] is False
    assert status["legacy_output_artifact_sha256"] == (
        "d2b3cc88803df4980c30b60f91b24ee535ee39aae118da7093f922b9cadeea31"
    )

    case_set, *_ = _case_set(tmp_path)
    rubric, rubric_digest = _rubric()
    packet = build_blinded_packet(case_set, rubric_digest)
    legacy = [
        {
            "case_ref": case.case_ref,
            "modality": case.modality,
            "label": "adequate",
            "dimension_labels": {
                "useful_actionability": "adequate",
                "repository_grounding": "adequate",
                "reviewability": "adequate",
                "risk_calibration": "adequate",
            },
            "reason": "Legacy generic rubric.",
            "missing_evidence": False,
        }
        for case in case_set.cases
    ]
    path = tmp_path / "legacy.jsonl"
    _write_jsonl(path, legacy)
    with pytest.raises(ValueError, match="modality schema"):
        build_model_output_receipt(case_set, rubric, rubric_digest, packet, path)


def test_two_reviews_and_complete_adjudication_derive_passing_metrics(
    tmp_path: Path,
) -> None:
    case_set, *_ = _case_set(tmp_path)
    rubric_digest, packet, model_receipt, outputs, truth = _output_bundle(
        tmp_path, case_set
    )
    disputed = case_set.cases[0].case_ref
    first, second = _reviews(
        case_set, packet, truth, second_changes={disputed: not truth[disputed]}
    )
    adjudication = build_adjudication(
        adjudicator_identity_digest=_digest("c"),
        first=first,
        second=second,
        decisions=[
            row for row in _decisions(case_set, truth) if row["case_ref"] == disputed
        ],
        case_set=case_set,
    )
    receipt = build_final_receipt(
        case_set=case_set,
        rubric_digest=rubric_digest,
        model_receipt=model_receipt,
        outputs=outputs,
        first=first,
        second=second,
        adjudication=adjudication,
        evaluated_agent_model_family="anthropic",
    )

    assert receipt["status"] == "passed"
    assert receipt["same_model_family"] is True
    assert receipt["claim_role"] == "advisory"
    assert receipt["overall"]["true_positive_rate"] == 1
    assert receipt["overall"]["true_negative_rate"] == 1
    assert set(receipt["modalities"]) == set(MODALITIES)
    validate_final_receipt(receipt)
    path = tmp_path / "final.json"
    write_receipt(path, receipt)
    assert json.loads(path.read_text()) == receipt


def test_missing_adjudication_and_same_reviewer_fail_closed(tmp_path: Path) -> None:
    case_set, *_ = _case_set(tmp_path)
    _rubric_digest, packet, _receipt, _outputs_value, truth = _output_bundle(
        tmp_path, case_set
    )
    disputed = case_set.cases[0].case_ref
    first, second = _reviews(
        case_set, packet, truth, second_changes={disputed: not truth[disputed]}
    )
    with pytest.raises(ValueError, match="exact required"):
        build_adjudication(
            adjudicator_identity_digest=_digest("c"),
            first=first,
            second=second,
            decisions=[],
            case_set=case_set,
        )

    duplicate = {**second, "reviewer_identity_digest": first["reviewer_identity_digest"]}
    unsigned = dict(duplicate)
    unsigned.pop("submission_digest")
    duplicate["submission_digest"] = stable_digest(unsigned)
    with pytest.raises(ValueError, match="distinct"):
        build_adjudication(
            adjudicator_identity_digest=_digest("c"),
            first=first,
            second=duplicate,
            decisions=[],
            case_set=case_set,
        )


def test_threshold_and_critical_false_pass_fail_the_receipt(tmp_path: Path) -> None:
    case_set, *_ = _case_set(tmp_path)
    truth = _truth(case_set)
    positives = [
        ref
        for ref, acceptable in truth.items()
        if acceptable
        and next(case.modality for case in case_set.cases if case.case_ref == ref)
        == "code-change"
    ]
    negatives = [ref for ref, acceptable in truth.items() if not acceptable]
    overrides = {positives[0]: "weak", positives[1]: "weak", negatives[0]: "strong"}
    rubric_digest, packet, model_receipt, outputs, truth = _output_bundle(
        tmp_path, case_set, override=overrides
    )
    decisions = _decisions(case_set, truth)
    for row in decisions:
        if row["case_ref"] == negatives[0]:
            row["critical_defect"] = True
    first = build_human_submission(
        reviewer_identity_digest=_digest("a"), packet_digest=packet["packet_digest"],
        decisions=decisions, case_set=case_set,
    )
    second = build_human_submission(
        reviewer_identity_digest=_digest("b"), packet_digest=packet["packet_digest"],
        decisions=decisions, case_set=case_set,
    )
    adjudication = build_adjudication(
        adjudicator_identity_digest=_digest("c"), first=first, second=second,
        decisions=[], case_set=case_set,
    )
    changed_outputs = (replace(outputs[0], label="weak" if outputs[0].label in PASSING_LABELS else "adequate"), *outputs[1:])
    with pytest.raises(ValueError, match="exactly once"):
        build_final_receipt(
            case_set=case_set, rubric_digest=rubric_digest,
            model_receipt=model_receipt, outputs=changed_outputs, first=first, second=second,
            adjudication=adjudication, evaluated_agent_model_family="anthropic",
        )
    receipt = build_final_receipt(
        case_set=case_set, rubric_digest=rubric_digest,
        model_receipt=model_receipt, outputs=outputs, first=first, second=second,
        adjudication=adjudication, evaluated_agent_model_family="anthropic",
    )
    assert receipt["status"] == "failed"
    assert receipt["modalities"]["code-change"]["true_positive_rate"] == 0.75
    assert receipt["critical_false_passes"] == 1


def test_secret_private_field_digest_and_symlink_fail_closed(tmp_path: Path) -> None:
    rows = _case_rows()
    rows[0]["response"] = "anthropic_api_key=sk-ant-this-is-a-real-looking-secret"
    with pytest.raises(ValueError, match="credential-like"):
        _case_set(tmp_path, rows)

    case_set, cases_path, manifest_path = _case_set(tmp_path)
    rubric, rubric_digest = _rubric()
    packet = build_blinded_packet(case_set, rubric_digest)
    poisoned = dict(packet)
    poisoned["private_label"] = "hidden"
    with pytest.raises(ValueError, match="private field"):
        build_model_output_receipt(
            case_set, rubric, rubric_digest, poisoned, tmp_path / "missing.jsonl"
        )

    manifest = json.loads(manifest_path.read_text())
    manifest["cases_digest"] = _digest("f")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest does not match"):
        load_case_set(manifest_path, cases_path)

    symlink = tmp_path / "cases-link.jsonl"
    symlink.symlink_to(cases_path)
    manifest["cases_digest"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="regular file"):
        load_case_set(manifest_path, symlink)


def test_final_receipt_tampering_is_rejected(tmp_path: Path) -> None:
    case_set, *_ = _case_set(tmp_path)
    rubric_digest, packet, model_receipt, outputs, truth = _output_bundle(
        tmp_path, case_set
    )
    first, second = _reviews(case_set, packet, truth)
    adjudication = build_adjudication(
        adjudicator_identity_digest=_digest("c"), first=first, second=second,
        decisions=[], case_set=case_set,
    )
    receipt = build_final_receipt(
        case_set=case_set, rubric_digest=rubric_digest,
        model_receipt=model_receipt, outputs=outputs, first=first, second=second,
        adjudication=adjudication, evaluated_agent_model_family="anthropic",
    )
    receipt["status"] = "failed"
    with pytest.raises(ValueError, match="digest"):
        validate_final_receipt(receipt)
    unknown = {**receipt, "human_signoff": True}
    with pytest.raises(ValueError, match="fields do not match"):
        validate_final_receipt(unknown)


def test_checked_in_manifest_and_rubric_are_consistent() -> None:
    manifest = json.loads((EXAMPLE / "case-set-manifest.json").read_text())
    rubric, digest = validate_rubric(EXAMPLE / "rubric.json")
    assert manifest["case_count"] == 48
    assert set(manifest["modalities"]) == set(rubric["modalities"])
    assert digest == hashlib.sha256((EXAMPLE / "rubric.json").read_bytes()).hexdigest()
    assert set(rubric["passing_labels"]) == PASSING_LABELS
    assert tuple(rubric["labels"]) == LABELS
