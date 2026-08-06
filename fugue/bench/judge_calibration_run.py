from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.bench.evaluations import JudgeResponseError
from fugue.bench.files import atomic_write_json
from fugue.bench.judge_calibration import (
    CASE_COUNT,
    CalibrationCaseSetV1,
    build_blinded_packet,
    build_model_output_receipt,
    load_case_set,
    validate_model_output_receipt,
    validate_model_output_row,
)
from fugue.bench.judge_provider_contract import (
    ProviderOutputValidationError,
    provider_response_schema,
    response_schema_digest,
    validate_provider_output,
)
from fugue.research.approvals import ApprovalLedger
from fugue.research.store import StudyStore

CALIBRATION_MAX_USD = 8.0
MAX_PROMPT_CHARACTERS = 12_000
MAX_OUTPUT_TOKENS = 1_200
Requester = Callable[[str], tuple[Mapping[str, Any], Mapping[str, Any]]]
_PROVIDER_OUTPUT_FIELDS = {
    "label", "dimension_labels", "reason", "missing_evidence",
}


def materialize_cases(
    *, repo_root: Path, manifest_path: Path, source_path: Path, destination: Path
) -> dict[str, Any]:
    """Validate then copy one explicit operator artifact into the private boundary."""
    root = repo_root.resolve()
    selected = load_case_set(manifest_path, source_path)
    target = _private_path(root, destination)
    if target.exists():
        existing = load_case_set(manifest_path, target)
        if existing != selected:
            raise ValueError("existing materialized case set disagrees with its manifest")
        status = "already_materialized"
    else:
        _atomic_bytes(target, source_path.read_bytes())
        if load_case_set(manifest_path, target) != selected:
            raise ValueError("materialized calibration cases failed post-copy validation")
        status = "materialized"
    os.chmod(target, 0o600)
    unsigned = {
        "schema_version": 1,
        "kind": "judge-calibration-case-materialization-receipt",
        "cases_digest": selected.cases_digest,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "case_count": CASE_COUNT,
        "destination": target.relative_to(root).as_posix(),
        "source_kind": "explicit_operator_supplied_file",
        "private_boundary": "verified",
        "status": status,
    }
    return {**unsigned, "receipt_digest": stable_digest(unsigned)}


def build_generation_preview(
    *,
    case_set: CalibrationCaseSetV1,
    rubric: Mapping[str, Any],
    rubric_digest: str,
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure, spend-free preview of exactly 48 blinded judge requests."""
    if packet != build_blinded_packet(case_set, rubric_digest):
        raise ValueError("calibration packet does not match the locked cases and rubric")
    prompts = [_prompt(row, rubric) for row in packet["cases"]]
    unsigned = {
        "schema_version": 1,
        "kind": "judge-model-output-generation-preview",
        "cases_digest": case_set.cases_digest,
        "rubric_digest": rubric_digest,
        "packet_digest": packet.get("packet_digest"),
        "response_schema_digest": response_schema_digest(rubric),
        "prompt_set_digest": stable_digest(prompts),
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile": rubric.get("profile"),
        "request_count": CASE_COUNT,
        "maximum_cost_usd": CALIBRATION_MAX_USD,
        "maximum_prompt_characters": MAX_PROMPT_CHARACTERS,
        "maximum_output_tokens": MAX_OUTPUT_TOKENS,
        "automatic_retries": 0,
        "human_review_automated": False,
        "claim_role": "advisory",
    }
    preview = {**unsigned, "preview_digest": stable_digest(unsigned)}
    validate_generation_preview(preview)
    return preview


def validate_generation_preview(value: Mapping[str, Any]) -> None:
    fields = {
        "schema_version", "kind", "cases_digest", "rubric_digest", "packet_digest",
        "response_schema_digest", "prompt_set_digest", "runner_source_sha256",
        "profile", "request_count",
        "maximum_cost_usd", "maximum_prompt_characters", "maximum_output_tokens",
        "automatic_retries", "human_review_automated", "claim_role", "preview_digest",
    }
    if set(value) != fields:
        raise ValueError("judge generation preview fields do not match")
    valid = (
        value["schema_version"] == 1
        and value["kind"] == "judge-model-output-generation-preview"
        and value["request_count"] == CASE_COUNT
        and value["maximum_cost_usd"] == CALIBRATION_MAX_USD
        and value["automatic_retries"] == 0
        and value["human_review_automated"] is False
        and value["claim_role"] == "advisory"
        and isinstance(value["runner_source_sha256"], str)
        and len(value["runner_source_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in value["runner_source_sha256"]
        )
    )
    if not valid:
        raise ValueError("judge generation preview policy is invalid")
    unsigned = dict(value)
    digest = unsigned.pop("preview_digest")
    if digest != stable_digest(unsigned):
        raise ValueError("judge generation preview digest does not match")


def run_generation(
    *,
    repo_root: Path,
    case_set: CalibrationCaseSetV1,
    rubric: Mapping[str, Any],
    rubric_digest: str,
    packet: Mapping[str, Any],
    approval_digest: str,
    output_path: Path,
    receipt_path: Path,
    requester: Requester,
) -> dict[str, Any]:
    """Generate only model outputs; human review remains a separate manual gate."""
    root = repo_root.resolve()
    output = _private_path(root, output_path)
    receipt = _private_path(root, receipt_path)
    preview = build_generation_preview(
        case_set=case_set, rubric=rubric, rubric_digest=rubric_digest, packet=packet
    )
    state_path = _generation_state_path(output, str(preview["preview_digest"]))
    subject_id = f"judge-calibration-{preview['preview_digest'][:20]}"
    ledger = ApprovalLedger(StudyStore(root).path)
    cases = list(packet["cases"])
    if state_path.exists():
        state = _read_state(state_path, preview, approval_digest)
        ledger.require_claimed_by(
            approval_digest=approval_digest,
            subject_kind="experiment",
            preview_digest=preview["preview_digest"],
            subject_id=subject_id,
        )
        if state["status"] == "provider_output_invalid":
            raise RuntimeError(
                "prior provider output failed the locked schema; a new preview and "
                "approval are required"
            )
        if state["in_flight_case_ref"] is not None:
            raise RuntimeError(
                "prior provider request has uncertain completion; automatic replay is forbidden"
            )
        for index, raw in enumerate(state["results"]):
            normalized = validate_model_output_row(raw, case_set, rubric)
            if normalized != raw or normalized["case_ref"] != cases[index]["case_ref"]:
                raise ValueError("durable calibration result prefix drifted")
    else:
        if output.exists() or receipt.exists():
            raise ValueError("new calibration generation refuses existing final artifacts")
        ledger.claim(
            approval_digest=approval_digest,
            subject_kind="experiment",
            preview_digest=preview["preview_digest"],
            subject_id=subject_id,
            estimated_cells=CASE_COUNT,
            estimated_cost_usd=CALIBRATION_MAX_USD,
        )
        state = _state(preview, approval_digest, [], [], 0, None, "running")
        atomic_write_json(state_path, state)
    results = [dict(row) for row in state["results"]]
    usage = [dict(row) for row in state["usage"]]
    for index in range(len(results), CASE_COUNT):
        case = cases[index]
        attempted = int(state["attempted_requests"]) + 1
        state = _state(
            preview, approval_digest, results, usage, attempted,
            case["case_ref"], "running",
        )
        atomic_write_json(state_path, state)
        try:
            raw, raw_usage = requester(_prompt(case, rubric))
        except JudgeResponseError as exc:
            state = _state(
                preview, approval_digest, results, usage, attempted, None,
                "provider_output_invalid",
                terminal_failure=_provider_response_error_failure(case, exc),
            )
            atomic_write_json(state_path, state)
            raise ValueError(
                "provider response failed strict JSON extraction; a new preview "
                "and approval are required"
            ) from exc
        safe_usage = _safe_usage(raw_usage)
        try:
            provider_output = validate_provider_output(
                raw, rubric, str(case["modality"])
            )
            normalized = validate_model_output_row(
                {
                    "case_ref": str(case["case_ref"]),
                    "modality": str(case["modality"]),
                    **provider_output,
                },
                case_set,
                rubric,
            )
        except ProviderOutputValidationError as exc:
            state = _state(
                preview, approval_digest, results, usage, attempted, None,
                "provider_output_invalid",
                terminal_failure=_provider_output_failure(
                    case=case, raw=raw, usage=safe_usage, code=exc.code,
                ),
            )
            atomic_write_json(state_path, state)
            raise ValueError(
                "provider response failed the locked judge schema; a new preview "
                "and approval are required"
            ) from exc
        results.append(normalized)
        usage.append(safe_usage)
        state = _state(preview, approval_digest, results, usage, attempted, None, "running")
        atomic_write_json(state_path, state)
    _atomic_jsonl(output, results)
    model_receipt, _outputs = build_model_output_receipt(
        case_set, rubric, rubric_digest, packet, output
    )
    generation = {
        "schema_version": 1,
        "kind": "judge-model-output-generation-receipt",
        "preview_digest": preview["preview_digest"],
        "approval_digest": approval_digest,
        "model_output_receipt": model_receipt,
        "attempted_requests": CASE_COUNT,
        "accounted_cost_usd": CALIBRATION_MAX_USD,
        "observed_cost_usd": None,
        "automatic_retries": 0,
        "human_review_status": "pending_two_independent_reviews",
    }
    generation["receipt_digest"] = stable_digest(generation)
    validate_generation_receipt(generation)
    atomic_write_json(receipt, generation)
    atomic_write_json(
        state_path,
        _state(preview, approval_digest, results, usage, CASE_COUNT, None, "completed"),
    )
    return generation

def validate_generation_receipt(value: Mapping[str, Any]) -> None:
    """Validate the immutable output of the approved model-generation stage."""

    fields = {
        "schema_version",
        "kind",
        "preview_digest",
        "approval_digest",
        "model_output_receipt",
        "attempted_requests",
        "accounted_cost_usd",
        "observed_cost_usd",
        "automatic_retries",
        "human_review_status",
        "receipt_digest",
    }
    if set(value) != fields:
        raise ValueError("judge generation receipt fields do not match")
    model_receipt = value.get("model_output_receipt")
    if not isinstance(model_receipt, Mapping):
        raise ValueError("judge generation receipt lacks a model-output receipt")
    validate_model_output_receipt(model_receipt)
    observed = value.get("observed_cost_usd")
    valid_observed = observed is None or (
        not isinstance(observed, bool)
        and isinstance(observed, int | float)
        and 0 <= observed <= CALIBRATION_MAX_USD
    )
    valid = (
        value.get("schema_version") == 1
        and value.get("kind") == "judge-model-output-generation-receipt"
        and value.get("attempted_requests") == CASE_COUNT
        and value.get("accounted_cost_usd") == CALIBRATION_MAX_USD
        and value.get("automatic_retries") == 0
        and value.get("human_review_status")
        == "pending_two_independent_reviews"
        and valid_observed
        and all(
            isinstance(value.get(key), str)
            and len(value[key]) == 64
            and all(character in "0123456789abcdef" for character in value[key])
            for key in ("preview_digest", "approval_digest")
        )
    )
    if not valid:
        raise ValueError("judge generation receipt policy is invalid")
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    if digest != stable_digest(unsigned):
        raise ValueError("judge generation receipt digest does not match")


def default_requester(*, model: str, env: Mapping[str, str]) -> Requester:
    """Create the production request function; construction itself makes no call."""
    from fugue.bench.evaluations import request_json_judge

    def request(prompt: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        return request_json_judge(
            model=model, env=env, prompt=prompt, strict_json_object=True
        )

    return request


def _prompt(case: Mapping[str, Any], rubric: Mapping[str, Any]) -> str:
    modality = str(case.get("modality") or "")
    modality_rubric = (rubric.get("modalities") or {}).get(modality)
    public_case = {
        key: value for key, value in case.items()
        if key not in {"case_ref", "modality"}
    }
    payload = {
        "case": public_case,
        "rubric": rubric.get("rubric"),
        "qualitative_labels": rubric.get("labels"),
        "passing_labels": rubric.get("passing_labels"),
        "modality_rubric": modality_rubric,
        "required_output_schema": provider_response_schema(rubric, modality),
    }
    prompt = (
        "Blindly assess one calibration artifact. Return exactly one raw JSON object "
        "with no markdown or surrounding text. Follow required_output_schema exactly: "
        "no extra keys, no omitted keys, no aliases, and no type coercion. Do not emit "
        "case_ref or modality; the host binds those trusted identities. Do not infer "
        "treatment identity, private truth, or hidden evidence. Keep reason under 2000 "
        "characters.\n\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise ValueError("calibration prompt exceeds the immutable size bound")
    return prompt

def _private_path(root: Path, path: Path) -> Path:
    selected = path if path.is_absolute() else root / path
    resolved = selected.resolve()
    private_root = (root / ".fugue/private").resolve()
    if not resolved.is_relative_to(private_root) or selected.is_symlink():
        raise ValueError("calibration outputs must be regular paths under .fugue/private")
    return resolved


def _state(
    preview: Mapping[str, Any], approval: str, results: list[dict[str, Any]],
    usage: list[dict[str, Any]], attempted: int, in_flight: Any, status: str,
    *, terminal_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 2, "kind": "judge-model-output-generation-state",
        "preview_digest": preview["preview_digest"], "approval_digest": approval,
        "attempted_requests": attempted,
        "accounted_cost_usd": round(attempted * CALIBRATION_MAX_USD / CASE_COUNT, 6),
        "in_flight_case_ref": in_flight, "results": results, "usage": usage,
        "status": status, "terminal_failure": terminal_failure,
    }
    return {**unsigned, "state_digest": stable_digest(unsigned)}


def _read_state(path: Path, preview: Mapping[str, Any], approval: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("calibration generation state must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("calibration generation state must be an object")
    unsigned = dict(value)
    digest = unsigned.pop("state_digest", None)
    if digest != stable_digest(unsigned):
        raise ValueError("calibration generation state digest does not match")
    if value.get("preview_digest") != preview["preview_digest"] or value.get("approval_digest") != approval:
        raise ValueError("calibration generation state identity drifted")
    results, usage = value.get("results"), value.get("usage")
    if not isinstance(results, list) or not isinstance(usage, list) or len(results) != len(usage):
        raise ValueError("calibration generation state rows are malformed")
    schema_version = value.get("schema_version")
    terminal_failure = value.get("terminal_failure") if schema_version == 2 else None
    attempted = len(results) + int(value.get("in_flight_case_ref") is not None)
    attempted += int(terminal_failure is not None)
    if value.get("attempted_requests") != attempted:
        raise ValueError("calibration generation state request count disagrees")
    allowed = {"running", "completed"}
    if schema_version == 2:
        allowed.add("provider_output_invalid")
    if schema_version not in {1, 2} or value.get("status") not in allowed or len(results) > CASE_COUNT:
        raise ValueError("calibration generation state status or size is invalid")
    if value.get("status") == "provider_output_invalid":
        _validate_provider_output_failure(terminal_failure)
        if value.get("in_flight_case_ref") is not None:
            raise ValueError("terminal provider failure cannot remain in flight")
    elif terminal_failure is not None:
        raise ValueError("nonterminal calibration state cannot contain a failure")
    return value


def _generation_state_path(output: Path, preview_digest: str) -> Path:
    return output.with_suffix(output.suffix + f".partial.{preview_digest[:20]}.json")


def _provider_output_failure(
    *, case: Mapping[str, Any], raw: Mapping[str, Any],
    usage: Mapping[str, Any], code: str,
) -> dict[str, Any]:
    serialized = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    keys = {str(key) for key in raw}
    dimensions = raw.get("dimension_labels")
    return {
        "case_ref": str(case["case_ref"]),
        "response_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "response_characters": len(serialized),
        "response_keys": sorted(keys),
        "missing_fields": sorted(_PROVIDER_OUTPUT_FIELDS - keys),
        "unexpected_fields": sorted(keys - _PROVIDER_OUTPUT_FIELDS),
        "dimension_keys": (
            sorted(str(key) for key in dimensions)
            if isinstance(dimensions, Mapping) else []
        ),
        "usage": dict(usage), "failure_code": code,
    }


def _provider_response_error_failure(
    case: Mapping[str, Any], error: JudgeResponseError,
) -> dict[str, Any]:
    return {
        "case_ref": str(case["case_ref"]),
        "response_sha256": error.response_sha256,
        "response_characters": error.response_characters,
        "response_keys": [], "missing_fields": sorted(_PROVIDER_OUTPUT_FIELDS),
        "unexpected_fields": [], "dimension_keys": [],
        "usage": _safe_usage(error.usage), "failure_code": error.code,
    }


def _validate_provider_output_failure(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("terminal provider failure must be an object")
    fields = {
        "case_ref", "response_sha256", "response_characters", "response_keys",
        "missing_fields", "unexpected_fields", "dimension_keys", "usage",
        "failure_code",
    }
    if set(value) != fields or not isinstance(value.get("failure_code"), str):
        raise ValueError("terminal provider failure fields do not match")
    digest = value.get("response_sha256")
    characters = value.get("response_characters")
    if (
        not isinstance(digest, str) or len(digest) != 64
        or not isinstance(characters, int) or characters < 0
    ):
        raise ValueError("terminal provider response metadata is invalid")
    _safe_usage(value.get("usage") or {})

def _safe_usage(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key in ("input_tokens", "output_tokens"):
        selected = value.get(key)
        if selected is not None and (isinstance(selected, bool) or not isinstance(selected, int) or selected < 0):
            raise ValueError("judge usage tokens must be nonnegative integers or unavailable")
        result[key] = selected
    return result


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
