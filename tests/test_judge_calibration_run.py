from __future__ import annotations

import json
import runpy
import stat
from pathlib import Path
from typing import Any

import pytest

from fugue.bench.judge_calibration import build_blinded_packet, validate_rubric
from fugue.bench.judge_calibration_run import (
    _generation_state_path,
    _prompt,
    build_generation_preview,
    materialize_cases,
    run_generation,
    validate_generation_preview,
)
from fugue.bench.judge_provider_contract import (
    ProviderOutputValidationError,
    provider_response_schema,
    validate_provider_output,
)
from fugue.research.approvals import ApprovalLedger
from fugue.research.store import StudyStore

_HELPERS = runpy.run_path(Path(__file__).with_name("test_judge_calibration.py"))
_case_set = _HELPERS["_case_set"]
_outputs = _HELPERS["_outputs"]
_truth = _HELPERS["_truth"]
_RUN_CALIBRATION = runpy.run_path(
    Path(__file__).parents[1]
    / "examples/comparisons/community-skill-selected-v1/judge/run_calibration.py"
)
_calibration_env = _RUN_CALIBRATION["_calibration_env"]


def _bundle(tmp_path: Path):
    cases, *_ = _case_set(tmp_path)
    rubric, rubric_digest = validate_rubric(
        Path(__file__).parents[1]
        / "examples/comparisons/community-skill-selected-v1/judge/rubric.json"
    )
    packet = build_blinded_packet(cases, rubric_digest)
    truth = _truth(cases)
    labels = {
        ref: ("adequate" if acceptable else "weak")
        for ref, acceptable in truth.items()
    }
    outputs = _outputs(cases, dict(rubric), overrides=labels)
    preview = build_generation_preview(
        case_set=cases,
        rubric=rubric,
        rubric_digest=rubric_digest,
        packet=packet,
    )
    return cases, rubric, rubric_digest, packet, outputs, preview


def _approve(tmp_path: Path, preview: dict[str, Any], *, cost: float = 8) -> str:
    approval = ApprovalLedger(StudyStore(tmp_path).path).approve(
        subject_kind="experiment",
        preview_digest=preview["preview_digest"],
        maximum_cost_usd=cost,
        maximum_cells=48,
        approved_by="test-operator",
        operation_id="approve-judge-calibration",
    )
    return approval.approval_digest


def test_preview_is_pure_exact_and_tamper_evident(tmp_path: Path) -> None:
    _cases, _rubric, _digest, packet, _outputs_value, preview = _bundle(tmp_path)

    validate_generation_preview(preview)
    assert preview["request_count"] == 48
    assert preview["maximum_cost_usd"] == 8
    assert preview["automatic_retries"] == 0
    assert preview["human_review_automated"] is False
    assert len(preview["runner_source_sha256"]) == 64
    assert len(packet["cases"]) == 48
    assert not (tmp_path / ".fugue").exists()

    tampered = {**preview, "request_count": 49}
    with pytest.raises(ValueError, match="policy|digest"):
        validate_generation_preview(tampered)


def test_calibration_env_excludes_openai_but_keeps_required_routes(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=not-used\n"
        "ANTHROPIC_API_KEY=anthropic-test\n"
        "WANDB_API_KEY=wandb-test\n",
        encoding="utf-8",
    )

    loaded = _calibration_env(env_file)

    assert "OPENAI_API_KEY" not in loaded
    assert loaded["ANTHROPIC_API_KEY"] == "anthropic-test"
    assert loaded["WANDB_API_KEY"] == "wandb-test"


def test_operator_source_materializes_idempotently_inside_private_boundary(
    tmp_path: Path,
) -> None:
    _cases, source, manifest = _case_set(tmp_path)
    repo = tmp_path / "repo"
    destination = Path(".fugue/private/community-skill-selected-v1/cases.jsonl")

    first = materialize_cases(
        repo_root=repo,
        manifest_path=manifest,
        source_path=source,
        destination=destination,
    )
    second = materialize_cases(
        repo_root=repo,
        manifest_path=manifest,
        source_path=source,
        destination=destination,
    )
    target = repo / destination
    assert first["status"] == "materialized"
    assert second["status"] == "already_materialized"
    assert target.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert first["source_kind"] == "explicit_operator_supplied_file"
    assert "source_path" not in first

    with pytest.raises(ValueError, match="under .fugue/private"):
        materialize_cases(
            repo_root=repo,
            manifest_path=manifest,
            source_path=source,
            destination=Path("tracked/cases.jsonl"),
        )


def test_approved_generation_runs_48_once_and_stops_before_human_review(
    tmp_path: Path,
) -> None:
    cases, rubric, rubric_digest, packet, output_rows, preview = _bundle(tmp_path)
    approval = _approve(tmp_path, preview)
    calls = 0

    def request(prompt: str, schema: dict[str, Any]):
        nonlocal calls
        assert "authored_reference" not in prompt
        selected = output_rows[calls]
        assert schema == provider_response_schema(rubric, selected["modality"])
        calls += 1
        return {
            key: value for key, value in selected.items()
            if key not in {"case_ref", "modality"}
        }, {"input_tokens": 100, "output_tokens": 20}

    result = run_generation(
        repo_root=tmp_path,
        case_set=cases,
        rubric=rubric,
        rubric_digest=rubric_digest,
        packet=packet,
        approval_digest=approval,
        output_path=Path(".fugue/private/campaign/model-outputs.jsonl"),
        receipt_path=Path(".fugue/private/campaign/generation-receipt.json"),
        requester=request,
    )

    assert calls == 48
    assert result["attempted_requests"] == 48
    assert result["accounted_cost_usd"] == 8
    assert result["observed_cost_usd"] is None
    assert result["human_review_status"] == "pending_two_independent_reviews"
    assert result["model_output_receipt"]["case_count"] == 48
    output = tmp_path / ".fugue/private/campaign/model-outputs.jsonl"
    assert len(output.read_text().splitlines()) == 48
    state = json.loads(
        _generation_state_path(output, preview["preview_digest"]).read_text(
            encoding="utf-8"
        )
    )
    assert state["status"] == "completed"
    assert state["accounted_cost_usd"] == 8


def test_underfunded_approval_cannot_start_provider_calls(tmp_path: Path) -> None:
    cases, rubric, rubric_digest, packet, _outputs_value, preview = _bundle(tmp_path)
    approval = _approve(tmp_path, preview, cost=7.99)
    calls = 0

    def request(_prompt: str, _schema: dict[str, Any]):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not be called")

    with pytest.raises(Exception, match="cost limit"):
        run_generation(
            repo_root=tmp_path,
            case_set=cases,
            rubric=rubric,
            rubric_digest=rubric_digest,
            packet=packet,
            approval_digest=approval,
            output_path=Path(".fugue/private/campaign/model-outputs.jsonl"),
            receipt_path=Path(".fugue/private/campaign/generation-receipt.json"),
            requester=request,
        )
    assert calls == 0


def test_uncertain_provider_failure_is_never_replayed(tmp_path: Path) -> None:
    cases, rubric, rubric_digest, packet, output_rows, preview = _bundle(tmp_path)
    approval = _approve(tmp_path, preview)
    calls = 0

    def request(_prompt: str, _schema: dict[str, Any]):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("provider connection ended after admission")
        return {
            key: value for key, value in output_rows[calls - 1].items()
            if key not in {"case_ref", "modality"}
        }, {"input_tokens": None, "output_tokens": None}

    kwargs = {
        "repo_root": tmp_path,
        "case_set": cases,
        "rubric": rubric,
        "rubric_digest": rubric_digest,
        "packet": packet,
        "approval_digest": approval,
        "output_path": Path(".fugue/private/campaign/model-outputs.jsonl"),
        "receipt_path": Path(".fugue/private/campaign/generation-receipt.json"),
        "requester": request,
    }
    with pytest.raises(RuntimeError, match="provider connection"):
        run_generation(**kwargs)
    assert calls == 2
    with pytest.raises(RuntimeError, match="automatic replay is forbidden"):
        run_generation(**kwargs)
    assert calls == 2


def test_prompt_uses_exact_schema_without_host_or_private_identity(tmp_path: Path) -> None:
    _cases_value, rubric, _digest, packet, _rows, _preview = _bundle(tmp_path)
    selected = packet["cases"][0]
    prompt = _prompt(selected, rubric)

    assert "required_output_schema" in prompt
    assert '"additionalProperties":false' in prompt
    assert selected["case_ref"] not in prompt
    assert "authored_reference" not in prompt
    assert "source_commit" not in prompt
    assert "case_ref or modality" in prompt


def test_provider_schema_is_exact_and_never_coerces(tmp_path: Path) -> None:
    cases, rubric, _digest_value, _packet, output_rows, _preview = _bundle(tmp_path)
    case, final = cases.cases[0], output_rows[0]
    valid = {key: value for key, value in final.items() if key not in {"case_ref", "modality"}}
    schema = provider_response_schema(rubric, case.modality)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(valid)
    assert validate_provider_output(valid, rubric, case.modality) == valid
    mutations = (
        ({**valid, "case_ref": "model-owned"}, "field_set_mismatch"),
        ({key: value for key, value in valid.items() if key != "reason"}, "field_set_mismatch"),
        ({**valid, "label": "Adequate"}, "label_value_invalid"),
        ({**valid, "missing_evidence": "false"}, "missing_evidence_not_boolean"),
        ({**valid, "dimension_labels": {}}, "dimension_set_mismatch"),
        ({**valid, "reason": ""}, "reason_invalid"),
    )
    for value, code in mutations:
        with pytest.raises(ProviderOutputValidationError) as caught:
            validate_provider_output(value, rubric, case.modality)
        assert caught.value.code == code


def test_completed_malformed_provider_output_is_terminal_and_audited(
    tmp_path: Path,
) -> None:
    cases, rubric, rubric_digest, packet, _rows, preview = _bundle(tmp_path)
    approval = _approve(tmp_path, preview)
    calls = 0

    def request(_prompt_value: str, _schema: dict[str, Any]):
        nonlocal calls
        calls += 1
        return {
            "label": "adequate",
            "dimension_labels": {},
            "reason": "missing required dimensions",
            "missing_evidence": False,
            "case_ref": "provider-must-not-own-this-identity",
        }, {"input_tokens": 100, "output_tokens": 20}

    output = Path(".fugue/private/campaign/model-outputs.jsonl")
    kwargs = {
        "repo_root": tmp_path, "case_set": cases, "rubric": rubric,
        "rubric_digest": rubric_digest, "packet": packet,
        "approval_digest": approval, "output_path": output,
        "receipt_path": Path(".fugue/private/campaign/generation-receipt.json"),
        "requester": request,
    }
    with pytest.raises(ValueError, match="locked judge schema"):
        run_generation(**kwargs)
    assert calls == 1

    state_path = _generation_state_path(
        tmp_path / output, preview["preview_digest"]
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "provider_output_invalid"
    assert state["in_flight_case_ref"] is None
    assert state["attempted_requests"] == 1
    assert state["results"] == []
    assert state["terminal_failure"]["failure_code"] == "field_set_mismatch"
    assert state["terminal_failure"]["unexpected_fields"] == ["case_ref"]
    assert state["terminal_failure"]["usage"] == {
        "input_tokens": 100, "output_tokens": 20,
    }
    assert "reason" not in state["terminal_failure"]

    with pytest.raises(RuntimeError, match="new preview and approval"):
        run_generation(**kwargs)
    assert calls == 1


def test_outputs_cannot_escape_operator_private_boundary(tmp_path: Path) -> None:
    cases, rubric, rubric_digest, packet, _rows, preview = _bundle(tmp_path)
    approval = _approve(tmp_path, preview)
    with pytest.raises(ValueError, match="under .fugue/private"):
        run_generation(
            repo_root=tmp_path,
            case_set=cases,
            rubric=rubric,
            rubric_digest=rubric_digest,
            packet=packet,
            approval_digest=approval,
            output_path=Path("public/model-outputs.jsonl"),
            receipt_path=Path(".fugue/private/campaign/receipt.json"),
            requester=lambda _prompt, _schema: ({}, {}),
        )
