from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.judge_calibration import (
    build_blinded_packet,
    build_model_output_receipt,
    validate_rubric,
)

_HELPERS = runpy.run_path(Path(__file__).with_name("test_judge_calibration.py"))
_case_set = _HELPERS["_case_set"]
_outputs = _HELPERS["_outputs"]
_truth = _HELPERS["_truth"]
_ENTRY = runpy.run_path(
    Path(__file__).parents[1]
    / "examples/comparisons/community-skill-selected-v1/judge/review_calibration.py"
)
_main = _ENTRY["main"]
_JUDGE = (
    Path(__file__).parents[1]
    / "examples/comparisons/community-skill-selected-v1/judge"
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _bundle(tmp_path: Path) -> dict[str, Any]:
    case_set, cases_path, manifest_path = _case_set(tmp_path)
    rubric, rubric_digest = validate_rubric(_JUDGE / "rubric.json")
    packet = build_blinded_packet(case_set, rubric_digest)
    root = tmp_path / "repo"
    private = root / ".fugue/private/calibration"
    packet_path = private / "blinded-packet.json"
    _write(packet_path, packet)
    truth = _truth(case_set)
    labels = {
        ref: ("adequate" if acceptable else "weak")
        for ref, acceptable in truth.items()
    }
    output_rows = _outputs(case_set, dict(rubric), overrides=labels)
    outputs_path = private / "model-outputs.jsonl"
    _write_jsonl(outputs_path, output_rows)
    model_receipt, _normalized = build_model_output_receipt(
        case_set, rubric, rubric_digest, packet, outputs_path
    )
    generation = {
        "schema_version": 1,
        "kind": "judge-model-output-generation-receipt",
        "preview_digest": "d" * 64,
        "approval_digest": "e" * 64,
        "model_output_receipt": model_receipt,
        "attempted_requests": 48,
        "accounted_cost_usd": 8.0,
        "observed_cost_usd": None,
        "automatic_retries": 0,
        "human_review_status": "pending_two_independent_reviews",
    }
    generation["receipt_digest"] = stable_digest(generation)
    generation_path = private / "generation-receipt.json"
    _write(generation_path, generation)
    common = [
        "--repo-root",
        str(root),
        "--manifest",
        str(manifest_path),
        "--cases",
        str(cases_path),
        "--rubric",
        str(_JUDGE / "rubric.json"),
        "--packet",
        str(packet_path),
    ]
    return {
        "case_set": case_set,
        "truth": truth,
        "root": root,
        "private": private,
        "common": common,
        "outputs": outputs_path,
        "generation": generation_path,
    }


def _complete_review(path: Path, truth: dict[str, bool], *, flip: str | None = None) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    for row in value["cases"]:
        case_ref = row["case"]["case_ref"]
        acceptable = truth[case_ref]
        if case_ref == flip:
            acceptable = not acceptable
        row["decision"] = {
            "acceptable": acceptable,
            "critical_defect": False,
            "reason": "Independent review using only the blinded artifact.",
        }
    _write(path, value)


def test_two_human_review_workflow_finalizes_and_prints_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _bundle(tmp_path)
    private, common = bundle["private"], bundle["common"]
    first_template = private / "reviewer-a.json"
    second_template = private / "reviewer-b.json"
    assert (
        _main(
            [
                "review-templates",
                *common,
                "--reviewer-a-out",
                str(first_template),
                "--reviewer-b-out",
                str(second_template),
            ]
        )
        == 0
    )
    serialized = first_template.read_text(encoding="utf-8")
    assert "authored_reference" not in serialized
    assert "private_label" not in serialized

    disputed = bundle["case_set"].cases[0].case_ref
    _complete_review(first_template, bundle["truth"])
    _complete_review(second_template, bundle["truth"], flip=disputed)
    first_submission = private / "reviewer-a-submission.json"
    second_submission = private / "reviewer-b-submission.json"
    for template, identity, destination in (
        (first_template, "a" * 64, first_submission),
        (second_template, "b" * 64, second_submission),
    ):
        assert (
            _main(
                [
                    "submit-review",
                    *common,
                    "--review-template",
                    str(template),
                    "--reviewer-identity-digest",
                    identity,
                    "--submission-out",
                    str(destination),
                ]
            )
            == 0
        )

    adjudication_template = private / "adjudication-template.json"
    assert (
        _main(
            [
                "adjudication-template",
                *common,
                "--first-submission",
                str(first_submission),
                "--second-submission",
                str(second_submission),
                "--adjudication-template-out",
                str(adjudication_template),
            ]
        )
        == 0
    )
    template = json.loads(adjudication_template.read_text(encoding="utf-8"))
    assert template["disagreement_count"] == 1
    assert [row["case"]["case_ref"] for row in template["disagreements"]] == [
        disputed
    ]
    row = template["disagreements"][0]
    row["decision"] = {
        "acceptable": bundle["truth"][disputed],
        "critical_defect": False,
        "reason": "Adjudicated from the two blinded reviews.",
    }
    _write(adjudication_template, template)

    adjudication = private / "adjudication.json"
    assert (
        _main(
            [
                "submit-adjudication",
                *common,
                "--first-submission",
                str(first_submission),
                "--second-submission",
                str(second_submission),
                "--adjudication-template",
                str(adjudication_template),
                "--adjudicator-identity-digest",
                "c" * 64,
                "--adjudication-out",
                str(adjudication),
            ]
        )
        == 0
    )

    final = private / "final-receipt.json"
    assert (
        _main(
            [
                "finalize",
                *common,
                "--first-submission",
                str(first_submission),
                "--second-submission",
                str(second_submission),
                "--adjudication",
                str(adjudication),
                "--model-outputs",
                str(bundle["outputs"]),
                "--generation-receipt",
                str(bundle["generation"]),
                "--final-receipt-out",
                str(final),
            ]
        )
        == 0
    )
    receipt = json.loads(final.read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["overall"]["true_positive_rate"] == 1
    assert receipt["overall"]["true_negative_rate"] == 1
    assert receipt["critical_false_passes"] == 0
    output = capsys.readouterr().out
    assert "Calibration PASSED" in output
    assert "Overall: TPR=100.0% TNR=100.0%" in output
    assert "Critical false passes: 0 (maximum 0)" in output

    output_rows = [
        json.loads(line)
        for line in bundle["outputs"].read_text(encoding="utf-8").splitlines()
    ]
    output_rows[0]["label"] = (
        "strong" if output_rows[0]["label"] != "strong" else "adequate"
    )
    _write_jsonl(bundle["outputs"], output_rows)
    with pytest.raises(ValueError, match="locked generation receipt"):
        _main(
            [
                "finalize",
                *common,
                "--first-submission",
                str(first_submission),
                "--second-submission",
                str(second_submission),
                "--adjudication",
                str(adjudication),
                "--model-outputs",
                str(bundle["outputs"]),
                "--generation-receipt",
                str(bundle["generation"]),
                "--final-receipt-out",
                str(private / "tampered-final.json"),
            ]
        )


def test_review_workflow_rejects_tampering_incomplete_secrets_and_same_reviewer(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    private, common = bundle["private"], bundle["common"]
    first = private / "first.json"
    second = private / "second.json"
    _main(
        [
            "review-templates",
            *common,
            "--reviewer-a-out",
            str(first),
            "--reviewer-b-out",
            str(second),
        ]
    )

    incomplete = json.loads(first.read_text(encoding="utf-8"))
    incomplete["cases"].pop()
    _write(first, incomplete)
    with pytest.raises(ValueError, match="exactly 48"):
        _main(
            [
                "submit-review",
                *common,
                "--review-template",
                str(first),
                "--reviewer-identity-digest",
                "a" * 64,
                "--submission-out",
                str(private / "incomplete-submission.json"),
            ]
        )

    first.unlink()
    _main(
        [
            "review-templates",
            *common,
            "--reviewer-a-out",
            str(first),
            "--reviewer-b-out",
            str(second),
        ]
    )
    extra = json.loads(first.read_text(encoding="utf-8"))
    extra["cases"].append(dict(extra["cases"][0]))
    _write(first, extra)
    with pytest.raises(ValueError, match="exactly 48"):
        _main(
            [
                "submit-review",
                *common,
                "--review-template",
                str(first),
                "--reviewer-identity-digest",
                "a" * 64,
                "--submission-out",
                str(private / "extra-submission.json"),
            ]
        )

    first.unlink()
    _main(
        [
            "review-templates",
            *common,
            "--reviewer-a-out",
            str(first),
            "--reviewer-b-out",
            str(second),
        ]
    )
    _complete_review(first, bundle["truth"])
    tampered = json.loads(first.read_text(encoding="utf-8"))
    tampered["cases"][0]["case"]["response"] = "Changed evidence"
    _write(first, tampered)
    with pytest.raises(ValueError, match="content was modified"):
        _main(
            [
                "submit-review",
                *common,
                "--review-template",
                str(first),
                "--reviewer-identity-digest",
                "a" * 64,
                "--submission-out",
                str(private / "tampered-submission.json"),
            ]
        )

    first.unlink()
    _main(
        [
            "review-templates",
            *common,
            "--reviewer-a-out",
            str(first),
            "--reviewer-b-out",
            str(second),
        ]
    )
    _complete_review(first, bundle["truth"])
    poisoned = json.loads(first.read_text(encoding="utf-8"))
    poisoned["cases"][0]["decision"]["reason"] = (
        "anthropic_api_key=sk-ant-this-is-a-real-looking-secret"
    )
    _write(first, poisoned)
    with pytest.raises(ValueError, match="credential-like"):
        _main(
            [
                "submit-review",
                *common,
                "--review-template",
                str(first),
                "--reviewer-identity-digest",
                "a" * 64,
                "--submission-out",
                str(private / "poisoned-submission.json"),
            ]
        )

    first.unlink()
    _main(
        [
            "review-templates",
            *common,
            "--reviewer-a-out",
            str(first),
            "--reviewer-b-out",
            str(second),
        ]
    )
    _complete_review(first, bundle["truth"])
    _complete_review(second, bundle["truth"])
    first_submission = private / "first-submission.json"
    second_submission = private / "second-submission.json"
    for template, destination in (
        (first, first_submission),
        (second, second_submission),
    ):
        _main(
            [
                "submit-review",
                *common,
                "--review-template",
                str(template),
                "--reviewer-identity-digest",
                "a" * 64,
                "--submission-out",
                str(destination),
            ]
        )
    original_first = json.loads(first_submission.read_text(encoding="utf-8"))
    tampered_submission = dict(original_first)
    tampered_submission["decisions"] = [
        dict(row) for row in original_first["decisions"]
    ]
    tampered_submission["decisions"][0]["reason"] = "Changed after signing."
    _write(first_submission, tampered_submission)
    with pytest.raises(ValueError, match="submission digest or content"):
        _main(
            [
                "adjudication-template",
                *common,
                "--first-submission",
                str(first_submission),
                "--second-submission",
                str(second_submission),
                "--adjudication-template-out",
                str(private / "tampered-adjudication-template.json"),
            ]
        )
    _write(first_submission, original_first)
    with pytest.raises(ValueError, match="distinct"):
        _main(
            [
                "adjudication-template",
                *common,
                "--first-submission",
                str(first_submission),
                "--second-submission",
                str(second_submission),
                "--adjudication-template-out",
                str(private / "adjudication-template.json"),
            ]
        )
