from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.judge_provider_contract import response_schema_digest

SCHEMA_VERSION = 1
MODALITIES = ("code-change", "implementation-plan", "skill-package")
LABELS = ("unusable", "weak", "adequate", "strong", "exceptional")
PASSING_LABELS = frozenset({"adequate", "strong", "exceptional"})
MINIMUM_RATE = 0.85
CASE_COUNT, PER_CLASS = 48, 8
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SECRET = re.compile(
    r"(?:sk-ant-[\w-]{12,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?:api|anthropic|wandb|openai)[_-]?key\s*[:=]\s*[^\s,;}]{8,})",
    re.IGNORECASE,
)
_PRIVATE_KEYS = {"authored_reference", "expected", "gold", "private_label", "variant_id", "variant_label", "candidate_id", "skill_revision", "source_commit"}


@dataclass(frozen=True)
class CalibrationCaseV1:
    case_ref: str
    modality: str
    public_task: Mapping[str, Any]
    response: str
    permitted_evidence: Mapping[str, Any]


@dataclass(frozen=True)
class CalibrationCaseSetV1:
    cases_digest: str
    cases: tuple[CalibrationCaseV1, ...]


@dataclass(frozen=True)
class JudgeOutputV1:
    case_ref: str
    modality: str
    label: str


@dataclass(frozen=True)
class ConfusionMetricsV1:
    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int

    @property
    def tpr(self) -> float:
        return self.true_positive / (self.true_positive + self.false_negative)

    @property
    def tnr(self) -> float:
        return self.true_negative / (self.true_negative + self.false_positive)

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_positive": self.true_positive, "false_negative": self.false_negative,
            "true_negative": self.true_negative, "false_positive": self.false_positive,
            "true_positive_rate": self.tpr, "true_negative_rate": self.tnr,
        }


def load_case_set(manifest_path: Path, cases_path: Path) -> CalibrationCaseSetV1:
    manifest = _read_object(manifest_path, "case-set manifest")
    _exact(manifest, {
        "schema_version", "id", "case_count", "cases_digest", "source_visibility",
        "repository_modalities", "modalities", "review_policy", "note",
    }, "case-set manifest")
    if manifest["source_visibility"] != "operator-restricted":
        raise ValueError("calibration cases must remain operator-restricted")
    if _integer(manifest["case_count"], "case_count") != CASE_COUNT:
        raise ValueError("calibration requires exactly 48 cases")
    digest = _digest(manifest["cases_digest"], "cases_digest")
    if _sha256(cases_path) != digest:
        raise ValueError("calibration case artifact digest does not match")
    repositories = _strings(manifest["repository_modalities"], "repository modalities")
    declared = _mapping(manifest["modalities"], "modalities")
    if set(repositories.values()) != set(MODALITIES) or set(declared) != set(MODALITIES):
        raise ValueError("case manifest does not cover the exact three modalities")
    if any(value != {"acceptable": 8, "defective": 8} for value in declared.values()):
        raise ValueError("manifest must declare 8 acceptable/8 defective per modality")
    cases: list[CalibrationCaseV1] = []
    authored: list[tuple[str, bool]] = []
    for row in _jsonl(cases_path, "calibration cases"):
        _exact(row, {
            "id", "repository_id", "scenario_id", "split", "public_task", "response",
            "permitted_evidence", "authored_reference", "judge_result", "reviews",
            "adjudicated_label",
        }, "calibration case")
        if row["judge_result"] is not None or row["reviews"] != [] or row["adjudicated_label"] is not None:
            raise ValueError("locked cases may not contain judgments or reviews")
        source_id = _text(row["id"], "case id", 160)
        repository = _text(row["repository_id"], "repository id", 120)
        reference = _mapping(row["authored_reference"], "authored reference")
        _exact(reference, {"label", "critical_false_pass"}, "authored reference")
        if repository not in repositories or reference["label"] not in {"pass", "fail"} or not isinstance(reference["critical_false_pass"], bool):
            raise ValueError("case repository or authored label is unsupported")
        response = _text(row["response"], "case response", 50_000)
        _public({"public_task": row["public_task"], "response": response, "permitted_evidence": row["permitted_evidence"]}, "calibration public payload")
        modality = repositories[repository]
        cases.append(CalibrationCaseV1(
            stable_digest({"cases_digest": digest, "source_id": source_id}), modality,
            _mapping(row["public_task"], "public task"), response,
            _mapping(row["permitted_evidence"], "permitted evidence"),
        ))
        authored.append((modality, reference["label"] == "pass"))
    counts = Counter(authored)
    balanced = all(counts[(modality, acceptable)] == PER_CLASS
                   for modality in MODALITIES for acceptable in (True, False))
    if len(cases) != CASE_COUNT or len({row.case_ref for row in cases}) != CASE_COUNT or not balanced:
        raise ValueError("case artifact must be unique and balanced 8/8 per modality")
    return CalibrationCaseSetV1(digest, tuple(sorted(cases, key=lambda row: row.case_ref)))


def validate_rubric(path: Path) -> tuple[Mapping[str, Any], str]:
    rubric = _read_object(path, "judge rubric")
    if rubric.get("role") != "advisory" or set(rubric.get("passing_labels") or ()) != PASSING_LABELS or "/" not in _text(rubric.get("profile"), "rubric profile", 200):
        raise ValueError("rubric must be advisory with exact passing labels")
    if tuple(_mapping(rubric.get("labels"), "rubric labels")) != LABELS:
        raise ValueError("rubric must use the five ordered qualitative anchors")
    modalities = _mapping(rubric.get("modalities"), "rubric modalities")
    if set(modalities) != set(MODALITIES):
        raise ValueError("rubric must define the exact three modalities")
    for name, value in modalities.items():
        dimensions = _mapping(value, name).get("dimensions")
        if not isinstance(dimensions, list) or len(dimensions) != 4 or len(set(dimensions)) != 4:
            raise ValueError(f"rubric modality {name} needs four unique dimensions")
    _public(rubric, "judge rubric")
    return rubric, _sha256(path)


def build_blinded_packet(case_set: CalibrationCaseSetV1, rubric_digest: str) -> dict[str, Any]:
    rows = [{
        "case_ref": case.case_ref, "modality": case.modality,
        "public_task": dict(case.public_task), "response": case.response,
        "permitted_evidence": dict(case.permitted_evidence),
    } for case in case_set.cases]
    packet = {
        "schema_version": 1, "kind": "judge-calibration-blinded-packet",
        "cases_digest": case_set.cases_digest,
        "rubric_digest": _digest(rubric_digest, "rubric_digest"),
        "case_count": len(rows), "cases": rows,
    }
    _public(packet, "blinded packet")
    return _signed(packet, "packet_digest")


def validate_model_output_row(value: Mapping[str, Any], case_set: CalibrationCaseSetV1, rubric: Mapping[str, Any]) -> dict[str, Any]:
    _exact(value, {"case_ref", "modality", "label", "dimension_labels", "reason", "missing_evidence"}, "model output")
    ref = _digest(value["case_ref"], "case_ref")
    known = {row.case_ref: row for row in case_set.cases}
    dimensions = tuple(_mapping(_mapping(rubric["modalities"], "modalities").get(str(value["modality"])), "rubric modality")["dimensions"])
    labels = _strings(value["dimension_labels"], "dimension labels")
    valid = (ref in known and value["modality"] == known[ref].modality and set(labels) == set(dimensions) and all(row in LABELS for row in labels.values())
             and value["label"] in LABELS and isinstance(value["missing_evidence"], bool))
    if not valid:
        raise ValueError("model output does not match its case/modality schema")
    return {"case_ref": ref, "modality": value["modality"], "label": value["label"], "dimension_labels": labels, "reason": _safe_reason(value["reason"]), "missing_evidence": value["missing_evidence"]}


def build_model_output_receipt(
    case_set: CalibrationCaseSetV1, rubric: Mapping[str, Any], rubric_digest: str,
    packet: Mapping[str, Any], outputs_path: Path,
) -> tuple[dict[str, Any], tuple[JudgeOutputV1, ...]]:
    _packet(packet, case_set, rubric_digest)
    _no_secret(outputs_path.read_text(encoding="utf-8"), "model outputs")
    outputs: list[JudgeOutputV1] = []
    for raw in _jsonl(outputs_path, "model outputs"):
        row = validate_model_output_row(raw, case_set, rubric)
        outputs.append(JudgeOutputV1(row["case_ref"], row["modality"], row["label"]))
    if len(outputs) != CASE_COUNT or len({row.case_ref for row in outputs}) != CASE_COUNT:
        raise ValueError("outputs require exactly one result per case")
    profile = _text(rubric.get("profile"), "rubric profile", 200)
    receipt = _signed({
        "schema_version": 1, "kind": "judge-model-output-receipt",
        "cases_digest": case_set.cases_digest, "rubric_digest": rubric_digest,
        "packet_digest": packet["packet_digest"], "output_artifact_sha256": _sha256(outputs_path),
        "response_schema_digest": response_schema_digest(rubric), "classification_digest": stable_digest([{"case_ref": row.case_ref, "modality": row.modality, "label": row.label} for row in outputs]), "profile": profile,
        "model_family": profile.split("/", 1)[0], "case_count": len(outputs),
        "privacy_scan": "passed", "status": "validated",
    }, "receipt_digest")
    validate_model_output_receipt(receipt)
    return receipt, tuple(sorted(outputs, key=lambda row: row.case_ref))


def validate_model_output_receipt(value: Mapping[str, Any]) -> None:
    fields = {"schema_version", "kind", "cases_digest", "rubric_digest", "packet_digest",
              "output_artifact_sha256", "response_schema_digest", "classification_digest", "profile", "model_family",
              "case_count", "privacy_scan", "status", "receipt_digest"}
    _exact(value, fields, "model-output receipt")
    valid = (value["schema_version"] == SCHEMA_VERSION and value["kind"] == "judge-model-output-receipt"
             and value["privacy_scan"] == "passed" and value["status"] == "validated"
             and _integer(value["case_count"], "case_count") == CASE_COUNT and _text(value["profile"], "profile", 200).split("/", 1)[0] == _text(value["model_family"], "model family", 80))
    if not valid:
        raise ValueError("model-output receipt is not a validated 48-case artifact")
    for key in ("cases_digest", "rubric_digest", "packet_digest", "output_artifact_sha256",
                "response_schema_digest", "classification_digest"):
        _digest(value[key], key)
    _verify(value, "receipt_digest")


def build_human_submission(
    *, reviewer_identity_digest: str, packet_digest: str,
    decisions: Sequence[Mapping[str, Any]], case_set: CalibrationCaseSetV1,
) -> dict[str, Any]:
    return _signed({
        "schema_version": 1, "kind": "judge-human-review-submission",
        "reviewer_identity_digest": _digest(reviewer_identity_digest, "reviewer identity"),
        "packet_digest": _digest(packet_digest, "packet_digest"),
        "decisions": _review_rows(decisions, case_set, "human submission"),
    }, "submission_digest")


def build_adjudication(
    *, adjudicator_identity_digest: str, first: Mapping[str, Any],
    second: Mapping[str, Any], decisions: Sequence[Mapping[str, Any]],
    case_set: CalibrationCaseSetV1,
) -> dict[str, Any]:
    _reviews(first, second, case_set)
    left, right = _by_ref(first), _by_ref(second)
    disagreements = {ref for ref in left if _choice(left[ref]) != _choice(right[ref])}
    return _signed({
        "schema_version": 1, "kind": "judge-review-adjudication",
        "adjudicator_identity_digest": _digest(adjudicator_identity_digest, "adjudicator identity"),
        "reviewer_submission_digests": [first["submission_digest"], second["submission_digest"]],
        "decisions": _review_rows(decisions, case_set, "adjudication", disagreements),
    }, "adjudication_digest")


def build_final_receipt(
    *, case_set: CalibrationCaseSetV1, rubric_digest: str,
    model_receipt: Mapping[str, Any], outputs: Sequence[JudgeOutputV1],
    first: Mapping[str, Any], second: Mapping[str, Any],
    adjudication: Mapping[str, Any], evaluated_agent_model_family: str,
) -> dict[str, Any]:
    validate_model_output_receipt(model_receipt)
    _reviews(first, second, case_set)
    if (model_receipt["cases_digest"], model_receipt["rubric_digest"]) != (case_set.cases_digest, rubric_digest) or first["packet_digest"] != model_receipt["packet_digest"]:
        raise ValueError("model receipt does not bind selected cases/rubric")
    submissions = [first["submission_digest"], second["submission_digest"]]
    if adjudication.get("reviewer_submission_digests") != submissions:
        raise ValueError("adjudication does not bind both human submissions")
    _exact(adjudication, {"schema_version", "kind", "adjudicator_identity_digest", "reviewer_submission_digests", "decisions", "adjudication_digest"}, "adjudication")
    if adjudication["kind"] != "judge-review-adjudication":
        raise ValueError("adjudication kind does not match")
    _digest(adjudication["adjudicator_identity_digest"], "adjudicator identity")
    _verify(adjudication, "adjudication_digest")
    left, right, resolved = _by_ref(first), _by_ref(second), _by_ref(adjudication)
    truth: dict[str, Mapping[str, Any]] = {}
    for ref, decision in left.items():
        disagrees = _choice(decision) != _choice(right[ref])
        if disagrees != (ref in resolved):
            raise ValueError("adjudication must resolve every and only disagreement")
        truth[ref] = resolved.get(ref, decision)
    output_by_ref, case_by_ref = ({row.case_ref: row for row in outputs},
                                  {row.case_ref: row for row in case_set.cases})
    if len(outputs) != CASE_COUNT or set(output_by_ref) != set(truth) or model_receipt["classification_digest"] != stable_digest([{"case_ref": row.case_ref, "modality": row.modality, "label": row.label} for row in outputs]):
        raise ValueError("model outputs do not cover reviewed cases exactly once")
    balance = Counter((case_by_ref[ref].modality, row["acceptable"])
                      for ref, row in truth.items())
    if any(balance[(modality, acceptable)] != PER_CLASS
           for modality in MODALITIES for acceptable in (True, False)):
        raise ValueError("adjudicated truth must remain balanced 8/8 per modality")
    by_modality = {modality: _metrics(
        (ref for ref, case in case_by_ref.items() if case.modality == modality),
        truth, output_by_ref,
    ) for modality in MODALITIES}
    overall = _metrics(truth, truth, output_by_ref)
    critical_false_passes = sum(
        int(not row["acceptable"] and row["critical_defect"]
            and output_by_ref[ref].label in PASSING_LABELS)
        for ref, row in truth.items()
    )
    passed = critical_false_passes == 0 and all(
        metric.tpr >= MINIMUM_RATE and metric.tnr >= MINIMUM_RATE
        for metric in (overall, *by_modality.values()))
    family = _text(evaluated_agent_model_family, "evaluated agent model family", 80)
    receipt = _signed({
        "schema_version": 1, "kind": "judge-calibration-receipt",
        "cases_digest": case_set.cases_digest, "rubric_digest": rubric_digest,
        "model_output_receipt_digest": model_receipt["receipt_digest"],
        "reviewer_submission_digests": submissions,
        "adjudication_digest": adjudication["adjudication_digest"],
        "judge_profile": model_receipt["profile"], "judge_model_family": model_receipt["model_family"],
        "evaluated_agent_model_family": family,
        "same_model_family": model_receipt["model_family"] == family,
        "claim_role": "advisory", "review_status": "adjudicated", "case_count": CASE_COUNT,
        "overall": overall.to_dict(),
        "modalities": {name: metric.to_dict() for name, metric in by_modality.items()},
        "critical_false_passes": critical_false_passes,
        "thresholds": {"minimum_tpr_tnr": MINIMUM_RATE, "critical_false_passes_max": 0},
        "status": "passed" if passed else "failed",
    }, "receipt_digest")
    validate_final_receipt(receipt)
    return receipt


def validate_final_receipt(value: Mapping[str, Any]) -> None:
    fields = {"schema_version", "kind", "cases_digest", "rubric_digest", "model_output_receipt_digest", "reviewer_submission_digests", "adjudication_digest", "judge_profile", "judge_model_family", "evaluated_agent_model_family", "same_model_family", "claim_role", "review_status", "case_count", "overall", "modalities", "critical_false_passes", "thresholds", "status", "receipt_digest"}
    _exact(value, fields, "final calibration receipt")
    valid = (value.get("kind") == "judge-calibration-receipt"
             and value.get("claim_role") == "advisory"
             and value.get("review_status") == "adjudicated"
             and value.get("case_count") == CASE_COUNT)
    if not valid:
        raise ValueError("final calibration receipt governance is invalid")
    if value.get("same_model_family") != (value.get("judge_model_family")
                                           == value.get("evaluated_agent_model_family")):
        raise ValueError("same-model-family policy is inconsistent")
    _verify(value, "receipt_digest")


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    validate_final_receipt(receipt)
    atomic_write_json(path, dict(receipt))


def _review_rows(
    values: Sequence[Mapping[str, Any]], case_set: CalibrationCaseSetV1,
    label: str, required: set[str] | None = None,
) -> list[dict[str, Any]]:
    known, rows = {row.case_ref for row in case_set.cases}, []
    for value in values:
        _exact(value, {"case_ref", "acceptable", "critical_defect", "reason"}, label)
        ref = _digest(value["case_ref"], "case_ref")
        valid = (ref in known and isinstance(value["acceptable"], bool)
                 and isinstance(value["critical_defect"], bool)
                 and not (value["acceptable"] and value["critical_defect"]))
        if not valid:
            raise ValueError(f"{label} has an invalid case or decision")
        rows.append({"case_ref": ref, "acceptable": value["acceptable"],
                     "critical_defect": value["critical_defect"],
                     "reason": _safe_reason(value["reason"])})
    expected = known if required is None else required
    if {row["case_ref"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError(f"{label} must cover its exact required case set")
    return sorted(rows, key=lambda row: row["case_ref"])


def _reviews(first: Mapping[str, Any], second: Mapping[str, Any], cases: CalibrationCaseSetV1) -> None:
    fields = {"schema_version", "kind", "reviewer_identity_digest", "packet_digest", "decisions", "submission_digest"}
    for value in (first, second):
        _exact(value, fields, "human review submission")
        if value["kind"] != "judge-human-review-submission":
            raise ValueError("human review submission kind does not match")
        _digest(value["reviewer_identity_digest"], "reviewer identity")
        _digest(value["packet_digest"], "packet digest")
        parsed = _review_rows(value["decisions"], cases, "human submission")
        if parsed != value["decisions"]:
            raise ValueError("human review decisions are not canonical")
        _verify(value, "submission_digest")
    if first.get("reviewer_identity_digest") == second.get("reviewer_identity_digest"):
        raise ValueError("two distinct blinded human reviewers are required")
    if first.get("packet_digest") != second.get("packet_digest"):
        raise ValueError("both reviewers must use the same blinded packet")
    refs = {row.case_ref for row in cases.cases}
    if set(_by_ref(first)) != refs or set(_by_ref(second)) != refs:
        raise ValueError("both human submissions must cover all 48 cases")


def _metrics(refs: Iterable[str], truth: Mapping[str, Mapping[str, Any]],
             outputs: Mapping[str, JudgeOutputV1]) -> ConfusionMetricsV1:
    count = Counter((truth[ref]["acceptable"], outputs[ref].label in PASSING_LABELS)
                    for ref in refs)
    return ConfusionMetricsV1(count[(True, True)], count[(True, False)],
                              count[(False, False)], count[(False, True)])


def _packet(value: Mapping[str, Any], cases: CalibrationCaseSetV1, rubric: str) -> None:
    _public(value, "blinded packet")
    if (value.get("cases_digest"), value.get("rubric_digest"), value.get("case_count")) != (cases.cases_digest, rubric, CASE_COUNT):
        raise ValueError("blinded packet does not bind selected cases/rubric")
    _verify(value, "packet_digest")


def _signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {**value, key: stable_digest(value)}


def _verify(value: Mapping[str, Any], key: str) -> None:
    digest, unsigned = _digest(value.get(key), key), dict(value)
    unsigned.pop(key)
    if digest != stable_digest(unsigned):
        raise ValueError(f"{key} does not match")


def _by_ref(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = value.get("decisions")
    if not isinstance(rows, list):
        raise ValueError("review decisions must be a list")
    return {str(row["case_ref"]): _mapping(row, "review decision") for row in rows}


def _choice(value: Mapping[str, Any]) -> tuple[Any, Any]:
    return value.get("acceptable"), value.get("critical_defect")


def _safe_reason(value: Any) -> str:
    reason = _text(value, "review reason", 2_000)
    _no_secret(reason, "review reason")
    return reason


def _public(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _PRIVATE_KEYS:
                raise ValueError(f"{label} contains private field {key!r}")
            _public(child, label)
    elif isinstance(value, list | tuple):
        for child in value:
            _public(child, label)
    elif isinstance(value, str):
        _no_secret(value, label)


def _no_secret(value: str, label: str) -> None:
    if _SECRET.search(value):
        raise ValueError(f"{label} contains credential-like material")


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), label)


def _jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"{label} contains blank lines")
    return [_mapping(json.loads(line), label) for line in lines]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _strings(value: Any, label: str) -> dict[str, str]:
    selected = _mapping(value, label)
    if any(not isinstance(key, str) or not isinstance(item, str)
           for key, item in selected.items()):
        raise ValueError(f"{label} must map strings to strings")
    return dict(selected)


def _exact(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the V1 schema")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be nonempty and at most {maximum} characters")
    return value


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact must be a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()
