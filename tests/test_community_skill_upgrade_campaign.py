from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from fugue.bench.comparison import (
    COMPARISON_JUDGE_RESPONSE_MAX_CHARACTERS,
    COMPARISON_JUDGE_RESPONSE_VALIDATOR_VERSION,
    ComparisonEvaluatorV1,
    _comparison_judge_payload,
    _comparison_judge_response_schema,
    _judge_execution_calibration_issue,
    comparison_judge_public_rubric_contract,
    load_comparison,
)
from fugue.bench.evaluations import JUDGE_JSON_MAX_RESPONSE_CHARACTERS
from fugue.research.approvals import ApprovalLedger
from fugue.research.store import StudyStore

REPO_ROOT = Path(__file__).parents[1]
EXAMPLE = REPO_ROOT / "examples/comparisons/community-skill-upgrades"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_module("validate_campaign", EXAMPLE / "validate_campaign.py")
RUNNER = _load_module(
    "run_synthetic_calibration",
    EXAMPLE / "run_synthetic_calibration.py",
)


def _json(name: str) -> dict[str, object]:
    value = json.loads((EXAMPLE / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _runner_bindings() -> dict[str, object]:
    runner = CONTRACT.calibration_runner_artifact(EXAMPLE)
    validator = CONTRACT.calibration_validator_artifact(EXAMPLE)
    return {
        "runner_artifact_path": runner["path"],
        "runner_artifact_sha256": runner["sha256"],
        "validator_artifact_path": validator["path"],
        "validator_artifact_sha256": validator["sha256"],
        "response_schema_digest": CONTRACT.provider_response_schema_digest(),
        "response_request_mode": CONTRACT.CALIBRATION_RESPONSE_REQUEST_MODE,
        "response_validator_version": CONTRACT.PROVIDER_RESPONSE_VALIDATOR_VERSION,
    }


def _fixture_judge_requester(
    label_by_response: dict[str, str],
):
    def requester(prompt: str) -> tuple[dict[str, object], dict[str, int]]:
        payload = json.loads(prompt.split("\n\n", 1)[1])
        passing = label_by_response[payload["response"]] == "pass"
        score = 0.95 if passing else 0.1
        return (
            {
                "scores": {
                    dimension: score for dimension in CONTRACT.EXPECTED_DIMENSIONS
                },
                "overall_assessment": "Grounded synthetic fixture assessment.",
                "uncertainty": 0.1,
                "missing_evidence": False,
                "rationale": "The response and path evidence support this label.",
            },
            {"input_tokens": 100, "output_tokens": 20},
        )

    return requester


def test_campaign_assets_are_strict_valid_but_not_execution_eligible() -> None:
    summary = CONTRACT.validate_all(EXAMPLE)

    assert summary == {
        "schema_version": 1,
        "campaign_id": "community-skill-upgrade-campaign-v1",
        "fixtures_valid": True,
        "execution_eligible": False,
        "calibration_status": "pending_human_review",
        "calibration_cases": 48,
        "calibration_digest": (
                "f4634b7dee0898e2bfc8104961ad2f1bdd70be5feda1081214457257b9215a15"
        ),
        "calibration_cases_artifact_path": (
            "examples/comparisons/community-skill-upgrades/"
            "judge-calibration-cases.jsonl"
        ),
        "calibration_cases_artifact_sha256": (
                "a6a67f5d6815b255758017ca82f945d76a61ca418e2d9ccfd049f8327e24669e"
        ),
        "calibration_prior_runs_ledger_path": (
            "examples/comparisons/community-skill-upgrades/calibration-prior-runs.json"
        ),
        "calibration_prior_runs_ledger_sha256": (
            "06bbd801c0fd29704dfb0e1641e737e9a3e976e3e2049ac4da8f0e10b143a3de"
        ),
        "calibration_prior_runs_ledger_digest": (
            "6e6ecb7c25efa8bb81c61def334bfdddc26b74afe6dc887741f95cdd57f0fcae"
        ),
        "calibration_prior_runs": 7,
        "calibration_prior_accounted_reserve_usd": 7.094273,
        "calibration_remaining_budget_usd": 0.905727,
        "rubric_digest": (
            "6706b2a5dbbc0850b0f1b904920d14ae9df54a212b34f745540efc4d3f3ef74b"
        ),
        "agent_cells": 12,
        "campaign_ceiling_usd": 110.0,
    }
    calibration = _json("judge-calibration.json")
    assert calibration["review_status"] == "pending_human_review"
    assert calibration["reviewers_per_example"] == 0
    assert calibration["passed"] is False
    assert calibration["execution_gate"] == CONTRACT.synthetic_execution_gate(
        cases_artifact_path=summary["calibration_cases_artifact_path"],
        cases_artifact_sha256=summary["calibration_cases_artifact_sha256"],
        cases_digest=summary["calibration_digest"],
        rubric_digest=summary["rubric_digest"],
        **_runner_bindings(),
    )
    assert calibration["execution_gate"]["maximum_cost_usd"] == 0.905727
    assert calibration["execution_gate"]["prior_runs"] == 7
    assert calibration["execution_gate"]["prior_failed_requests"] == 7
    assert calibration["execution_gate"]["prior_accounted_reserve_usd"] == 7.094273
    assert calibration["execution_gate"]["campaign_allocation_usd"] == 8


def test_campaign_has_balanced_16_case_repository_strata() -> None:
    rows = CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl")
    summary = CONTRACT.validate_cases(rows)

    assert summary["examples"] == 48
    assert summary["calibration_examples"] == 36
    assert summary["holdout_examples"] == 12
    for repository_id in CONTRACT.EXPECTED_REPOSITORIES:
        repository_rows = [row for row in rows if row["repository_id"] == repository_id]
        assert len(repository_rows) == 16
        assert sum(row["split"] == "calibration" for row in repository_rows) == 12
        assert sum(row["split"] == "holdout" for row in repository_rows) == 4
        assert [row["authored_reference"]["label"] for row in repository_rows].count(
            "pass"
        ) == 8

    superpowers_holdout = [
        row
        for row in rows
        if row["repository_id"] == "superpowers-writing-plans"
        and row["split"] == "holdout"
    ]
    assert {row["id"] for row in superpowers_holdout} == {
        "superpowers-provider-task-importer-pass-01",
        "superpowers-provider-task-importer-pass-02",
        "superpowers-provider-task-importer-fail-01",
        "superpowers-provider-task-importer-fail-02",
    }
    assert all(
        row["scenario_id"] == "superpowers-provider-task-importer-plan"
        for row in superpowers_holdout
    )
    assert (
        sum(
            row["authored_reference"]["critical_false_pass"]
            for row in superpowers_holdout
        )
        == 1
    )


def test_prior_run_ledger_is_frozen_and_derives_remaining_budget() -> None:
    manifest = _json("campaign-manifest.json")
    calibration = manifest["calibration"]
    ledger = CONTRACT.frozen_prior_runs_ledger(
        root=EXAMPLE,
        calibration=calibration,
    )

    assert ledger == {
        "path": (
            "examples/comparisons/community-skill-upgrades/calibration-prior-runs.json"
        ),
        "sha256": ("06bbd801c0fd29704dfb0e1641e737e9a3e976e3e2049ac4da8f0e10b143a3de"),
        "digest": ("6e6ecb7c25efa8bb81c61def334bfdddc26b74afe6dc887741f95cdd57f0fcae"),
        "prior_runs": 7,
        "accounted_reserve_usd": 7.094273,
        "remaining_budget_usd": 0.905727,
    }
    raw = _json("calibration-prior-runs.json")
    assert sum(run["attempted_requests"] for run in raw["runs"]) == 79
    assert [run["receipt_status"] for run in raw["runs"]] == [
        "unavailable",
        "available",
        "available",
        "unavailable",
        "available",
        "available",
        "available",
    ]
    latest = raw["runs"][-1]
    assert latest["preview_digest"] == (
        "e9f15bebc211aa0f94aa2f7f305c7a801ac72bba0840957aed07d835c45fb670"
    )
    assert latest["approval_digest"] == (
        "240105d4f93939f702ecee413613b78149cc59a21af0505a791a4b64b49f8833"
    )
    assert latest["reason_code"] == "thresholds_unreachable"
    assert latest["artifact_document_digest"] == (
        "23c39ca60a8863d2db608bcdf2ffeb361acf388ee7f935a336268595d24f1ad1"
    )


def test_prior_run_ledger_byte_and_content_drift_are_rejected(tmp_path: Path) -> None:
    comparisons = tmp_path / "examples/comparisons"
    shutil.copytree(REPO_ROOT / "examples/comparisons", comparisons)
    root = comparisons / "community-skill-upgrades"
    manifest = json.loads((root / "campaign-manifest.json").read_text())
    ledger_path = root / "calibration-prior-runs.json"
    ledger_path.write_bytes(ledger_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="artifact SHA-256 drifted"):
        CONTRACT.frozen_prior_runs_ledger(
            root=root,
            calibration=manifest["calibration"],
        )


def test_frozen_cases_artifact_is_bound_to_preview_and_gate() -> None:
    manifest = _json("campaign-manifest.json")
    calibration = _json("judge-calibration.json")
    cases = EXAMPLE / str(manifest["calibration"]["cases"])

    assert CONTRACT.file_sha256(cases) == manifest["calibration"]["cases_sha256"]
    gate = calibration["execution_gate"]
    assert gate["cases_artifact_path"] == (
        "examples/comparisons/community-skill-upgrades/judge-calibration-cases.jsonl"
    )
    assert gate["cases_artifact_sha256"] == CONTRACT.file_sha256(cases)
    preview = RUNNER.calibration_preview(EXAMPLE)
    assert preview["cases_artifact_path"] == gate["cases_artifact_path"]
    assert preview["cases_artifact_sha256"] == gate["cases_artifact_sha256"]
    assert preview["preview_digest"] == gate["preview_digest"]

    changed_path = CONTRACT.calibration_preview_document(
        cases_artifact_path=f"archive/{preview['cases_artifact_path']}",
        cases_artifact_sha256=preview["cases_artifact_sha256"],
        cases_digest=preview["cases_digest"],
        rubric_digest=preview["rubric_digest"],
        **_runner_bindings(),
    )
    changed_sha = CONTRACT.calibration_preview_document(
        cases_artifact_path=preview["cases_artifact_path"],
        cases_artifact_sha256="0" * 64,
        cases_digest=preview["cases_digest"],
        rubric_digest=preview["rubric_digest"],
        **_runner_bindings(),
    )
    changed_ledger = CONTRACT.calibration_preview_document(
        cases_artifact_path=preview["cases_artifact_path"],
        cases_artifact_sha256=preview["cases_artifact_sha256"],
        cases_digest=preview["cases_digest"],
        rubric_digest=preview["rubric_digest"],
        prior_runs_ledger_digest="0" * 64,
        **_runner_bindings(),
    )
    assert changed_path["preview_digest"] != preview["preview_digest"]
    assert changed_sha["preview_digest"] != preview["preview_digest"]
    assert changed_ledger["preview_digest"] != preview["preview_digest"]


def test_case_byte_drift_is_rejected_before_a_judge_request(tmp_path: Path) -> None:
    comparisons = tmp_path / "examples/comparisons"
    shutil.copytree(REPO_ROOT / "examples/comparisons", comparisons)
    root = comparisons / "community-skill-upgrades"
    cases = root / "judge-calibration-cases.jsonl"
    cases.write_bytes(cases.read_bytes() + b"\n")
    requests: list[str] = []

    def requester(prompt: str) -> tuple[dict[str, object], dict[str, int]]:
        requests.append(prompt)
        raise AssertionError("requester must not be called after case-byte drift")

    with pytest.raises(ValueError, match="artifact SHA-256 drifted"):
        RUNNER.run_synthetic_calibration(
            root=root,
            output=tmp_path / "synthetic.json",
            requester=requester,
        )
    assert requests == []


def test_manifest_rejects_unknown_fields_and_legacy_project_routing() -> None:
    manifest = _json("campaign-manifest.json")
    unknown = copy.deepcopy(manifest)
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="unknown field"):
        CONTRACT.validate_manifest(unknown, root=EXAMPLE)

    legacy = copy.deepcopy(manifest)
    legacy["studies"][0]["evidence_project"] = "wandb/news-research-agent"
    with pytest.raises(ValueError, match="non-campaign evidence project|forbidden"):
        CONTRACT.validate_manifest(legacy, root=EXAMPLE)


def test_shared_judge_contract_is_exact_in_every_study_spec() -> None:
    manifest = _json("campaign-manifest.json")
    rubric = _json("judge-rubric.json")

    CONTRACT.validate_study_specs(root=EXAMPLE, manifest=manifest, rubric=rubric)

    drifted = copy.deepcopy(rubric)
    drifted["rubric"] = f"{drifted['rubric']} Drift."
    with pytest.raises(ValueError, match="shared judge contract drifted"):
        CONTRACT.validate_study_specs(
            root=EXAMPLE,
            manifest=manifest,
            rubric=drifted,
        )


@pytest.mark.parametrize(
    ("score", "label"),
    [
        (0.0, "unusable"),
        (0.249999, "unusable"),
        (0.25, "weak"),
        (0.499999, "weak"),
        (0.5, "adequate"),
        (0.749999, "adequate"),
        (0.75, "strong"),
        (0.899999, "strong"),
        (0.9, "exceptional"),
        (1.0, "exceptional"),
    ],
)
def test_judge_label_bands_match_study_console(score: float, label: str) -> None:
    assert CONTRACT.label_from_mean(score) == label


def test_synthetic_prompt_contains_only_blinded_public_inputs_and_rubric() -> None:
    rubric = _json("judge-rubric.json")
    row = CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl")[0]

    prompt = RUNNER._prompt(rubric, row)
    payload = json.loads(prompt.split("\n\n", 1)[1])

    assert set(payload) == {
        "public_task",
        "response",
        "permitted_evidence",
        "rubric",
    }
    serialized = json.dumps(payload, sort_keys=True)
    assert "authored_reference" not in serialized
    assert "reviews" not in serialized
    assert "adjudicated_label" not in serialized
    assert "baseline_commit" not in serialized
    assert "candidate_commit" not in serialized
    instructions = prompt.split("\n\n", 1)[0]
    exact_shape = json.dumps(
        CONTRACT.provider_response_example(),
        sort_keys=True,
        separators=(",", ":"),
    )
    assert exact_shape in instructions
    assert "exactly these top-level fields and no others" in instructions
    assert "exactly these dimensions and no others" in instructions
    assert "Never return scores_explanation" in instructions
    assert CONTRACT.provider_response_schema() == {
        "type": "object",
        "additionalProperties": False,
        "required": list(CONTRACT.PROVIDER_RESPONSE_TOP_LEVEL_FIELDS),
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "required": list(CONTRACT.EXPECTED_DIMENSIONS),
                "properties": {
                    dimension: {"type": "number"}
                    for dimension in CONTRACT.EXPECTED_DIMENSIONS
                },
            },
            "overall_assessment": {
                "type": "string",
                "description": (
                    "Brief evidence-bounded assessment; requested maximum "
                    "500 characters."
                ),
            },
            "uncertainty": {"type": "number"},
            "missing_evidence": {"type": "boolean"},
            "rationale": {
                "type": "string",
                "description": (
                    "Brief evidence-bounded rationale; requested maximum "
                    "500 characters."
                ),
            },
        },
    }


def test_synthetic_and_live_judges_share_the_response_envelope_contract() -> None:
    assert CONTRACT.PROVIDER_RESPONSE_VALIDATOR_VERSION == (
        COMPARISON_JUDGE_RESPONSE_VALIDATOR_VERSION
    )
    assert CONTRACT.PROVIDER_RESPONSE_MAX_CHARACTERS == (
        COMPARISON_JUDGE_RESPONSE_MAX_CHARACTERS
    )
    assert CONTRACT.PROVIDER_RESPONSE_MAX_CHARACTERS == (
        JUDGE_JSON_MAX_RESPONSE_CHARACTERS
    )
    assert CONTRACT.provider_response_schema() == _comparison_judge_response_schema(
        CONTRACT.EXPECTED_DIMENSIONS
    )


def test_synthetic_and_live_judges_share_the_exact_public_rubric_contract() -> None:
    rubric = _json("judge-rubric.json")
    row = CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl")[0]
    synthetic_payload = json.loads(RUNNER._prompt(rubric, row).split("\n\n", 1)[1])
    evaluator = ComparisonEvaluatorV1(
        id=str(rubric["id"]),
        type="llm_judge",
        required=False,
        profile=str(rubric["profile"]),
        rubric=str(rubric["rubric"]),
        dimensions=tuple(str(item) for item in rubric["dimensions"]),
        evidence=tuple(str(item) for item in rubric["evidence"]),
    )
    live_payload = _comparison_judge_payload(
        evaluator=evaluator,
        public_task=row["public_task"],
        row={
            "answer": row["response"],
            **{
                field: row["permitted_evidence"].get(field, [])
                for field in evaluator.evidence
            },
        },
    )
    expected = comparison_judge_public_rubric_contract(
        rubric=str(rubric["rubric"]),
        dimensions=tuple(str(item) for item in rubric["dimensions"]),
    )

    assert synthetic_payload["rubric"] == expected
    assert live_payload["rubric"] == expected
    assert set(synthetic_payload) == set(live_payload) == {
        "public_task",
        "response",
        "permitted_evidence",
        "rubric",
    }
    assert expected["passing_labels"] == ["adequate", "strong", "exceptional"]
    assert expected["passing_score"] == 0.5


@pytest.mark.parametrize(
    ("label", "missing_evidence", "expected"),
    [
        ("weak", False, "fail"),
        ("adequate", False, "pass"),
        ("strong", False, "pass"),
        ("exceptional", False, "pass"),
        ("adequate", True, "fail"),
    ],
)
def test_adequate_is_the_minimum_acceptable_synthetic_label(
    label: str,
    missing_evidence: bool,
    expected: str,
) -> None:
    assert CONTRACT._synthetic_predicted_label(
        {"overall_label": label, "missing_evidence": missing_evidence}
    ) == expected


def test_synthetic_prompt_rejects_label_description_drift_from_live_contract() -> None:
    rubric = _json("judge-rubric.json")
    rubric["labels"][0]["description"] = "A calibration-only interpretation."
    row = CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl")[0]

    with pytest.raises(ValueError, match="label bands drifted"):
        RUNNER._prompt(rubric, row)


def test_synthetic_provider_payload_rejects_unknown_explanation_field() -> None:
    payload = CONTRACT.provider_response_example()
    payload["scores_explanation"] = {
        dimension: "unsupported extra output"
        for dimension in CONTRACT.EXPECTED_DIMENSIONS
    }

    with pytest.raises(ValueError, match="scores_explanation"):
        RUNNER._validate_provider_payload(payload)


def test_synthetic_provider_payload_keeps_long_but_bounded_explanation() -> None:
    payload = CONTRACT.provider_response_example()
    payload["rationale"] = "r" * 800

    validated = RUNNER._validate_provider_payload(payload)

    assert validated["rationale"] == "r" * 800


def test_synthetic_provider_payload_rejects_explanation_above_hard_bound() -> None:
    payload = CONTRACT.provider_response_example()
    payload["rationale"] = "r" * (CONTRACT.PROVIDER_RESPONSE_MAX_CHARACTERS + 1)

    with pytest.raises(ValueError, match="rationale"):
        RUNNER._validate_provider_payload(payload)


def test_runner_and_response_schema_drift_change_the_preview() -> None:
    preview = RUNNER.calibration_preview(EXAMPLE)
    bindings = _runner_bindings()
    common = {
        "cases_artifact_path": preview["cases_artifact_path"],
        "cases_artifact_sha256": preview["cases_artifact_sha256"],
        "cases_digest": preview["cases_digest"],
        "rubric_digest": preview["rubric_digest"],
        "validator_artifact_path": bindings["validator_artifact_path"],
        "validator_artifact_sha256": bindings["validator_artifact_sha256"],
        "response_request_mode": preview["response_request_mode"],
        "response_validator_version": preview["response_validator_version"],
    }
    runner_drift = CONTRACT.calibration_preview_document(
        **common,
        runner_artifact_path=bindings["runner_artifact_path"],
        runner_artifact_sha256="0" * 64,
        response_schema_digest=bindings["response_schema_digest"],
    )
    schema_drift = CONTRACT.calibration_preview_document(
        **common,
        runner_artifact_path=bindings["runner_artifact_path"],
        runner_artifact_sha256=bindings["runner_artifact_sha256"],
        response_schema_digest="0" * 64,
    )

    assert runner_drift["preview_digest"] != preview["preview_digest"]
    assert schema_drift["preview_digest"] != preview["preview_digest"]


def test_synthetic_dry_run_is_bounded_and_preserves_human_gate() -> None:
    summary = RUNNER.dry_run_summary(EXAMPLE)
    preview = RUNNER.calibration_preview(EXAMPLE)

    assert summary["status"] == "dry_run"
    assert summary["requests"] == 48
    assert summary["automatic_retries"] == 0
    assert summary["maximum_output_tokens_per_request"] == 1200
    assert summary["budget_ceiling_usd"] == 0.905727
    assert summary["calibration_status"] == "pending_human_review"
    assert summary["human_calibration_satisfied"] is False
    assert summary["preview_digest"] == preview["preview_digest"]
    unsigned = {key: value for key, value in preview.items() if key != "preview_digest"}
    assert preview["preview_digest"] == CONTRACT.stable_digest(unsigned)
    assert preview["maximum_cost_usd"] == 0.905727
    assert preview["prior_accounted_reserve_usd"] == 7.094273
    assert preview["campaign_allocation_usd"] == 8
    assert preview["requests"] == 48


def test_synthetic_fixture_run_emits_separate_immutable_diagnostic(
    tmp_path: Path,
) -> None:
    rows = CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl")
    label_by_response = {
        row["response"]: row["authored_reference"]["label"] for row in rows
    }
    human_report_before = (EXAMPLE / "judge-calibration.json").read_bytes()
    output = tmp_path / "synthetic.json"

    def requester(prompt: str) -> tuple[dict[str, object], dict[str, int]]:
        payload = json.loads(prompt.split("\n\n", 1)[1])
        passing = label_by_response[payload["response"]] == "pass"
        score = 0.95 if passing else 0.1
        return (
            {
                "scores": {
                    dimension: score for dimension in CONTRACT.EXPECTED_DIMENSIONS
                },
                "overall_assessment": "Grounded synthetic fixture assessment.",
                "uncertainty": 0.1,
                "missing_evidence": False,
                "rationale": "The supplied response and path evidence support this label.",
            },
            {"input_tokens": 100, "output_tokens": 20},
        )

    result = RUNNER.run_synthetic_calibration(
        root=EXAMPLE,
        output=output,
        requester=requester,
    )

    assert result["status"] == "completed"
    assert result["completed_cases"] == 48
    assert result["synthetic_metrics"]["balanced_accuracy"] == 1.0
    assert result["synthetic_metrics"]["synthetic_thresholds_passed"] is True
    assert result["human_review_status"] == "pending_human_review"
    assert result["human_calibration_satisfied"] is False
    assert result["observed_cost_usd"] is None
    assert result["accounted_cost_usd"] == 0.905727
    assert result["cumulative_accounted_cost_usd"] == 8
    assert result["cases_artifact_path"] == (
        "examples/comparisons/community-skill-upgrades/judge-calibration-cases.jsonl"
    )
    assert result["cases_artifact_sha256"] == CONTRACT.file_sha256(
        EXAMPLE / "judge-calibration-cases.jsonl"
    )
    unsigned = {key: value for key, value in result.items() if key != "result_digest"}
    assert result["result_digest"] == CONTRACT.stable_digest(unsigned)
    receipt = result["receipt"]
    receipt_unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_digest"
    }
    assert receipt["receipt_digest"] == CONTRACT.stable_digest(receipt_unsigned)
    assert receipt["serialized_input_fields"] == [
        "public_task",
        "response",
        "permitted_evidence",
        "rubric",
    ]
    assert receipt["cases_artifact_path"] == result["cases_artifact_path"]
    assert receipt["cases_artifact_sha256"] == result["cases_artifact_sha256"]
    assert receipt["runner_artifact_sha256"] == result["runner_artifact_sha256"]
    assert receipt["response_schema_digest"] == result["response_schema_digest"]
    assert receipt["run_accounted_cost_usd"] == 0.905727
    assert receipt["cumulative_accounted_cost_usd"] == 8
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert (EXAMPLE / "judge-calibration.json").read_bytes() == human_report_before
    verified = CONTRACT.validate_synthetic_result(
        root=EXAMPLE,
        value=result,
        require_complete=True,
        require_approval=False,
    )
    assert verified["synthetic_metrics"] == result["synthetic_metrics"]


def test_second_request_failure_preserves_partial_and_writes_safe_receipt(
    tmp_path: Path,
) -> None:
    rows = CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl")
    label_by_response = {
        row["response"]: row["authored_reference"]["label"] for row in rows
    }
    fixture_requester = _fixture_judge_requester(label_by_response)
    calls = 0

    def requester(prompt: str) -> tuple[dict[str, object], dict[str, int]]:
        nonlocal calls
        calls += 1
        payload, usage = fixture_requester(prompt)
        if calls == 2:
            payload["scores_explanation"] = {"unexpected": "must be rejected"}
        return payload, usage

    output = tmp_path / "synthetic.json"
    with pytest.raises(ValueError, match="scores_explanation"):
        RUNNER.run_synthetic_calibration(
            root=EXAMPLE,
            output=output,
            requester=requester,
        )

    partial = json.loads(output.read_text(encoding="utf-8"))
    assert partial["status"] == "incomplete"
    assert partial["completed_cases"] == 1
    assert len(partial["results"]) == 1
    preview = RUNNER.calibration_preview(EXAMPLE)
    failure_path = RUNNER._failure_output_path(
        output,
        preview_digest=preview["preview_digest"],
        case_ordinal=2,
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    RUNNER._validate_failure_document(
        failure,
        preview=preview,
    )
    assert failure["case_id"] == rows[1]["id"]
    assert failure["case_ordinal"] == 2
    assert failure["completed_cases"] == 1
    assert failure["attempted_requests"] == 2
    assert failure["error_code"] == "strict_response_contract_rejected"
    assert CONTRACT.SHA256.fullmatch(failure["response_sha256"])
    assert failure["response_characters"] > 0
    assert failure["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert failure["raw_response_persisted"] is False
    assert failure["run_accounted_reserve_usd"] == round(
        2 * 0.905727 / 48,
        6,
    )
    assert failure["cumulative_accounted_reserve_usd"] <= 8
    serialized_failure = json.dumps(failure, sort_keys=True)
    assert "must be rejected" not in serialized_failure
    assert "scores_explanation" not in serialized_failure


def test_all_fail_calibration_stops_when_split_balanced_accuracy_is_unreachable(
    tmp_path: Path,
) -> None:
    calls = 0

    def requester(_: str) -> tuple[dict[str, object], dict[str, int]]:
        nonlocal calls
        calls += 1
        return (
            {
                "scores": {
                    dimension: 0.1 for dimension in CONTRACT.EXPECTED_DIMENSIONS
                },
                "overall_assessment": "Insufficiently grounded.",
                "uncertainty": 0.1,
                "missing_evidence": False,
                "rationale": "The response does not satisfy the rubric.",
            },
            {"input_tokens": 100, "output_tokens": 20},
        )

    output = tmp_path / "synthetic.json"
    with pytest.raises(
        RUNNER.SyntheticThresholdsUnreachable,
        match="mathematically unreachable",
    ):
        RUNNER.run_synthetic_calibration(
            root=EXAMPLE,
            output=output,
            requester=requester,
        )

    assert calls == 10
    partial = json.loads(output.read_text(encoding="utf-8"))
    assert partial["status"] == "incomplete"
    assert partial["completed_cases"] == 10
    reachability = CONTRACT.synthetic_threshold_reachability(
        CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl"),
        partial["results"],
    )
    assert reachability["reachable"] is False
    assert reachability["blockers"] == ["calibration_balanced_accuracy"]
    calibration_balanced_accuracy = next(
        gate
        for gate in reachability["gates"]
        if gate["id"] == "calibration_balanced_accuracy"
    )
    assert calibration_balanced_accuracy == {
        "id": "calibration_balanced_accuracy",
        "scope": "calibration",
        "metric": "balanced_accuracy",
        "threshold": 0.85,
        "positive_examples": 18,
        "negative_examples": 18,
        "completed_examples": 10,
        "current_true_positives": 0,
        "current_true_negatives": 4,
        "remaining_positives": 12,
        "remaining_negatives": 14,
        "maximum_true_positive_rate": 0.666667,
        "maximum_true_negative_rate": 1.0,
        "maximum_balanced_accuracy": 0.833333,
        "reachable": False,
    }
    preview = RUNNER.calibration_preview(EXAMPLE)
    termination_path = RUNNER._termination_output_path(
        output,
        preview_digest=preview["preview_digest"],
        attempted_requests=10,
        reason_code="thresholds_unreachable",
    )
    termination = json.loads(termination_path.read_text(encoding="utf-8"))
    RUNNER._validate_termination_document(
        termination,
        preview=preview,
        expected_reachability=reachability,
        expected_partial_result_digest=partial["result_digest"],
    )
    assert termination["status"] == "failed"
    assert termination["completed_cases"] == 10
    assert termination["attempted_requests"] == 10
    assert termination["partial_result_digest"] == partial["result_digest"]


def test_critical_false_pass_stops_before_the_next_request(tmp_path: Path) -> None:
    calls = 0

    def requester(_: str) -> tuple[dict[str, object], dict[str, int]]:
        nonlocal calls
        calls += 1
        score = 0.95 if calls in {1, 2, 4} else 0.1
        return (
            {
                "scores": {
                    dimension: score for dimension in CONTRACT.EXPECTED_DIMENSIONS
                },
                "overall_assessment": "Synthetic critical-policy fixture.",
                "uncertainty": 0.1,
                "missing_evidence": False,
                "rationale": "Fixture judgment for reachability testing.",
            },
            {"input_tokens": 100, "output_tokens": 20},
        )

    output = tmp_path / "synthetic.json"
    with pytest.raises(RUNNER.SyntheticThresholdsUnreachable):
        RUNNER.run_synthetic_calibration(
            root=EXAMPLE,
            output=output,
            requester=requester,
        )

    assert calls == 4
    partial = json.loads(output.read_text(encoding="utf-8"))
    reachability = CONTRACT.synthetic_threshold_reachability(
        CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl"),
        partial["results"],
    )
    assert reachability["critical_false_passes"] == 1
    assert reachability["blockers"] == ["critical_false_passes"]


def test_operator_interrupt_on_request_nine_reserves_nine_and_writes_receipt(
    tmp_path: Path,
) -> None:
    rows = CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl")
    label_by_response = {
        row["response"]: row["authored_reference"]["label"] for row in rows
    }
    fixture_requester = _fixture_judge_requester(label_by_response)
    calls = 0

    def requester(prompt: str) -> tuple[dict[str, object], dict[str, int]]:
        nonlocal calls
        calls += 1
        if calls == 9:
            raise KeyboardInterrupt
        return fixture_requester(prompt)

    output = tmp_path / "synthetic.json"
    with pytest.raises(KeyboardInterrupt):
        RUNNER.run_synthetic_calibration(
            root=EXAMPLE,
            output=output,
            requester=requester,
        )

    assert calls == 9
    partial = json.loads(output.read_text(encoding="utf-8"))
    assert partial["completed_cases"] == 8
    preview = RUNNER.calibration_preview(EXAMPLE)
    termination_path = RUNNER._termination_output_path(
        output,
        preview_digest=preview["preview_digest"],
        attempted_requests=9,
        reason_code="operator_cancelled",
    )
    termination = json.loads(termination_path.read_text(encoding="utf-8"))
    reachability = CONTRACT.synthetic_threshold_reachability(rows, partial["results"])
    RUNNER._validate_termination_document(
        termination,
        preview=preview,
        expected_reachability=reachability,
        expected_partial_result_digest=partial["result_digest"],
    )
    assert termination["status"] == "cancelled"
    assert termination["completed_cases"] == 8
    assert termination["attempted_requests"] == 9
    assert termination["current_case_id"] == rows[8]["id"]
    assert termination["current_case_ordinal"] == 9
    assert termination["response_returned"] is False
    assert termination["run_accounted_reserve_usd"] == round(
        9 * 0.905727 / 48,
        6,
    )
    assert termination["cumulative_accounted_reserve_usd"] == 7.264097

    tampered = copy.deepcopy(termination)
    tampered["run_accounted_reserve_usd"] = 0
    unsigned = {
        key: value for key, value in tampered.items() if key != "termination_digest"
    }
    tampered["termination_digest"] = CONTRACT.stable_digest(unsigned)
    with pytest.raises(ValueError, match="cost contract disagrees"):
        RUNNER._validate_termination_document(
            tampered,
            preview=preview,
            expected_reachability=reachability,
            expected_partial_result_digest=partial["result_digest"],
        )

    wrong_partial = copy.deepcopy(termination)
    wrong_partial["partial_result_digest"] = "0" * 64
    unsigned = {
        key: value
        for key, value in wrong_partial.items()
        if key != "termination_digest"
    }
    wrong_partial["termination_digest"] = CONTRACT.stable_digest(unsigned)
    with pytest.raises(ValueError, match="partial digest disagrees"):
        RUNNER._validate_termination_document(
            wrong_partial,
            preview=preview,
            expected_reachability=reachability,
            expected_partial_result_digest=partial["result_digest"],
        )


def test_interrupt_after_atomic_replace_binds_reloaded_durable_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl")
    label_by_response = {
        row["response"]: row["authored_reference"]["label"] for row in rows
    }
    output = tmp_path / "synthetic.json"
    atomic_write = RUNNER._atomic_write
    interrupted = False

    def interrupt_after_replace(path: Path, value: dict[str, object]) -> None:
        nonlocal interrupted
        atomic_write(path, value)
        if (
            not interrupted
            and path == output
            and value.get("kind") == "synthetic_gold_diagnostic"
            and value.get("completed_cases") == 1
        ):
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(RUNNER, "_atomic_write", interrupt_after_replace)
    with pytest.raises(KeyboardInterrupt):
        RUNNER.run_synthetic_calibration(
            root=EXAMPLE,
            output=output,
            requester=_fixture_judge_requester(label_by_response),
        )

    partial = json.loads(output.read_text(encoding="utf-8"))
    preview = RUNNER.calibration_preview(EXAMPLE)
    termination_path = RUNNER._termination_output_path(
        output,
        preview_digest=preview["preview_digest"],
        attempted_requests=1,
        reason_code="operator_cancelled",
    )
    termination = json.loads(termination_path.read_text(encoding="utf-8"))

    assert interrupted is True
    assert partial["completed_cases"] == 1
    assert termination["completed_cases"] == 1
    assert termination["attempted_requests"] == 1
    assert termination["partial_result_digest"] == partial["result_digest"]
    assert termination["current_case_id"] is None
    assert termination["current_case_ordinal"] is None


def test_forged_self_consistent_summary_is_rejected_against_frozen_cases(
    tmp_path: Path,
) -> None:
    rows = CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl")
    label_by_response = {
        row["response"]: row["authored_reference"]["label"] for row in rows
    }
    gate = _json("judge-calibration.json")["execution_gate"]
    output = tmp_path / "synthetic.json"
    result = RUNNER.run_synthetic_calibration(
        root=EXAMPLE,
        output=output,
        requester=_fixture_judge_requester(label_by_response),
        preview_digest=str(gate["preview_digest"]),
        approval_digest="a" * 64,
    )

    assert result["synthetic_metrics"]["synthetic_thresholds_passed"] is True
    CONTRACT.validate_synthetic_result(
        root=EXAMPLE,
        value=result,
        require_complete=True,
        require_approval=True,
    )

    forged = copy.deepcopy(result)
    critical_failure = next(
        row
        for row in rows
        if row["authored_reference"]["label"] == "fail"
        and row["authored_reference"]["critical_false_pass"]
    )
    forged_row = next(
        row for row in forged["results"] if row["case_id"] == critical_failure["id"]
    )
    forged_row["dimension_scores"] = {
        dimension: 0.95 for dimension in CONTRACT.EXPECTED_DIMENSIONS
    }
    forged_row["overall_label"] = "exceptional"
    forged_row["missing_evidence"] = False
    forged_unsigned = {
        key: value for key, value in forged.items() if key != "result_digest"
    }
    forged["result_digest"] = CONTRACT.stable_digest(forged_unsigned)

    with pytest.raises(ValueError, match="metrics disagree with frozen cases"):
        CONTRACT.validate_synthetic_result(
            root=EXAMPLE,
            value=forged,
            require_complete=True,
            require_approval=True,
        )


def test_core_gate_recomputes_frozen_cases_and_verifies_claimed_approval(
    tmp_path: Path,
) -> None:
    campaign_cases = (
        tmp_path / "examples/comparisons/community-skill-upgrades/"
        "judge-calibration-cases.jsonl"
    )
    campaign_cases.parent.mkdir(parents=True)
    shutil.copy2(EXAMPLE / "judge-calibration-cases.jsonl", campaign_cases)
    campaign_runner = campaign_cases.parent / "run_synthetic_calibration.py"
    shutil.copy2(EXAMPLE / "run_synthetic_calibration.py", campaign_runner)
    campaign_validator = campaign_cases.parent / "validate_campaign.py"
    shutil.copy2(EXAMPLE / "validate_campaign.py", campaign_validator)
    campaign_ledger = campaign_cases.parent / "calibration-prior-runs.json"
    shutil.copy2(EXAMPLE / "calibration-prior-runs.json", campaign_ledger)
    shutil.copy2(
        EXAMPLE / "calibration-prior-runs.json",
        campaign_cases.parent / "calibration-prior-runs.json",
    )
    calibration = _json("judge-calibration.json")
    (tmp_path / "calibration.json").write_text(
        json.dumps(calibration, sort_keys=True),
        encoding="utf-8",
    )
    source_spec = load_comparison(
        REPO_ROOT
        / "examples/comparisons/anthropic-skill-creator-upgrade/comparison.yaml",
        repo_root=REPO_ROOT,
    )
    judge = replace(
        next(item for item in source_spec.evaluators if item.type == "llm_judge"),
        calibration="calibration.json",
    )
    gate = calibration["execution_gate"]
    assert isinstance(gate, dict)
    ledger = ApprovalLedger(StudyStore(tmp_path).path)
    approval = ledger.approve(
        subject_kind="experiment",
        preview_digest=str(gate["preview_digest"]),
        maximum_cost_usd=float(gate["maximum_cost_usd"]),
        maximum_cells=48,
        approved_by="test-operator",
        operation_id="approve-synthetic-calibration",
    )
    ledger.claim(
        approval_digest=approval.approval_digest,
        subject_kind="experiment",
        preview_digest=str(gate["preview_digest"]),
        subject_id=f"calibration-{str(gate['preview_digest'])[:20]}",
        estimated_cells=48,
        estimated_cost_usd=float(gate["maximum_cost_usd"]),
    )
    rows = CONTRACT.read_cases(EXAMPLE / "judge-calibration-cases.jsonl")
    label_by_response = {
        row["response"]: row["authored_reference"]["label"] for row in rows
    }
    output = tmp_path / str(gate["result_path"])
    result = RUNNER.run_synthetic_calibration(
        root=EXAMPLE,
        output=output,
        requester=_fixture_judge_requester(label_by_response),
        preview_digest=str(gate["preview_digest"]),
        approval_digest=approval.approval_digest,
    )

    assert (
        _judge_execution_calibration_issue(
            judge,
            repo_root=tmp_path,
            approved_inputs=None,
        )
        is None
    )

    runner_bytes = campaign_runner.read_bytes()
    campaign_runner.write_bytes(runner_bytes + b"\n# drift\n")
    assert "runner drifted" in str(
        _judge_execution_calibration_issue(
            judge,
            repo_root=tmp_path,
            approved_inputs=None,
        )
    )
    campaign_runner.write_bytes(runner_bytes)

    drifted_calibration = copy.deepcopy(calibration)
    drifted_calibration["execution_gate"]["response_schema_digest"] = "0" * 64
    (tmp_path / "calibration.json").write_text(
        json.dumps(drifted_calibration, sort_keys=True),
        encoding="utf-8",
    )
    assert "response schema digest does not match" in str(
        _judge_execution_calibration_issue(
            judge,
            repo_root=tmp_path,
            approved_inputs=None,
        )
    )
    (tmp_path / "calibration.json").write_text(
        json.dumps(calibration, sort_keys=True),
        encoding="utf-8",
    )

    forged = copy.deepcopy(result)
    forged["results"][0]["case_id"] = "forged-case-id"
    unsigned = {key: value for key, value in forged.items() if key != "result_digest"}
    forged["result_digest"] = CONTRACT.stable_digest(unsigned)
    output.write_text(json.dumps(forged, sort_keys=True), encoding="utf-8")
    assert "rows do not match frozen cases" in str(
        _judge_execution_calibration_issue(
            judge,
            repo_root=tmp_path,
            approved_inputs=None,
        )
    )


def test_synthetic_runner_rejects_secret_values_before_persistence(
    tmp_path: Path,
) -> None:
    secret = "synthetic-secret-value"
    output = tmp_path / "synthetic.json"

    def requester(_: str) -> tuple[dict[str, object], dict[str, int]]:
        return (
            {
                "scores": {
                    dimension: 0.9 for dimension in CONTRACT.EXPECTED_DIMENSIONS
                },
                "overall_assessment": "Strong",
                "uncertainty": 0.1,
                "missing_evidence": False,
                "rationale": f"Unexpected provider text {secret}",
            },
            {"input_tokens": 100, "output_tokens": 20},
        )

    with pytest.raises(ValueError, match="configured secret value"):
        RUNNER.run_synthetic_calibration(
            root=EXAMPLE,
            output=output,
            requester=requester,
            secret_values=(secret,),
        )
    assert not output.exists()
