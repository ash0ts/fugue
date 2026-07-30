from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, replace
from typing import Any, Literal

from fugue.bench.candidates import stable_digest
from fugue.bench.library import validate_id

PROVIDER_PROTOCOL_VERSION = 1

Portability = Literal["portable", "provider_bound", "blocked"]
CellStatus = Literal["succeeded", "failed", "cancelled", "timed_out"]

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PORTABILITY = frozenset({"portable", "provider_bound", "blocked"})
_CELL_STATUS = frozenset({"succeeded", "failed", "cancelled", "timed_out"})


@dataclass(frozen=True)
class ProviderDescriptorV1:
    schema_version: int
    provider_id: str
    display_name: str
    provider_version: str
    protocol_version: int
    capabilities: tuple[str, ...]
    task_types: tuple[str, ...]
    input_types: tuple[str, ...]
    evaluator_types: tuple[str, ...]
    lifecycle_types: tuple[str, ...]
    source_provenance: dict[str, str]
    descriptor_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class CandidateBundleV1:
    schema_version: int
    provider_id: str
    candidate_ref: str
    display_name: str
    agent_config: dict[str, Any]
    model_route: dict[str, Any]
    behavior_assets: tuple[dict[str, Any], ...]
    skills: tuple[dict[str, Any], ...]
    mcp_servers: tuple[dict[str, Any], ...]
    agent_code: dict[str, Any]
    required_credentials: tuple[str, ...]
    portability: Portability
    blockers: tuple[str, ...] = ()
    bundle_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ProviderTaskV1:
    id: str
    title: str
    input: tuple[dict[str, Any], ...]
    interaction: dict[str, Any]
    stopping_policy: tuple[dict[str, Any], ...]
    lifecycle: dict[str, Any]
    credential_names: tuple[str, ...]
    integration_config: dict[str, Any]
    evaluator_ids: tuple[str, ...]
    metadata: dict[str, Any]
    portability: Portability
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ProviderScenarioV1:
    id: str
    path: tuple[str, ...]
    parent_path: tuple[str, ...]
    weight: float
    must_pass: bool
    tasks: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ProviderEvaluatorV1:
    id: str
    type: str
    implementation: dict[str, Any]
    config: dict[str, Any]
    evidence: tuple[str, ...]
    weight: float
    threshold: float
    must_pass: bool
    portability: Portability
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class SuiteBundleV1:
    schema_version: int
    provider_id: str
    suite_ref: str
    title: str
    objective: str
    attempts: int
    tasks: tuple[ProviderTaskV1, ...]
    scenarios: tuple[ProviderScenarioV1, ...]
    evaluators: tuple[ProviderEvaluatorV1, ...]
    metadata: dict[str, Any]
    bundle_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class PrivateEvaluationTaskV1:
    task_id: str
    expected: dict[str, Any]
    evaluator_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class PrivateEvaluationBundleV1:
    schema_version: int
    provider_id: str
    suite_digest: str
    tasks: tuple[PrivateEvaluationTaskV1, ...]
    metadata: dict[str, Any]
    private_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class PreparationReceiptV1:
    schema_version: int
    provider_id: str
    provider_lock_digest: str
    candidate_digest: str
    suite_digest: str
    frozen_references: tuple[dict[str, Any], ...]
    materialized_resources: tuple[dict[str, Any], ...]
    lifecycle_outputs: tuple[dict[str, Any], ...]
    runtime_artifacts: tuple[dict[str, Any], ...]
    cleanup_obligations: tuple[dict[str, Any], ...]
    prepared_at: str
    receipt_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class CellRequestV1:
    schema_version: int
    provider_id: str
    plan_digest: str
    cell_id: str
    candidate_digest: str
    suite_digest: str
    preparation_receipt_digest: str
    candidate: dict[str, Any]
    preparation: dict[str, Any]
    task: dict[str, Any]
    attempt: int
    runtime_lock_digest: str
    credential_profile_names: tuple[str, ...]
    budget: dict[str, Any]
    request_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class CellResultV1:
    schema_version: int
    provider_id: str
    cell_id: str
    request_digest: str
    status: CellStatus
    output: dict[str, Any]
    conversation: tuple[dict[str, Any], ...]
    tool_calls: tuple[dict[str, Any], ...]
    usage: dict[str, Any]
    evidence_refs: tuple[dict[str, Any], ...]
    failure: dict[str, Any] | None
    cleanup: dict[str, Any]
    result_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


def provider_descriptor_from_dict(raw: Mapping[str, Any]) -> ProviderDescriptorV1:
    _strict_fields(raw, ProviderDescriptorV1, "provider descriptor")
    value = ProviderDescriptorV1(
        schema_version=_schema(raw, "provider descriptor"),
        provider_id=validate_id(raw.get("provider_id") or "", kind="provider id"),
        display_name=_text(raw.get("display_name"), "provider display name", 200),
        provider_version=_text(
            raw.get("provider_version"), "provider version", 200
        ),
        protocol_version=_protocol(raw.get("protocol_version")),
        capabilities=_unique_text_tuple(raw.get("capabilities"), "capability"),
        task_types=_unique_text_tuple(raw.get("task_types"), "task type"),
        input_types=_unique_text_tuple(raw.get("input_types"), "input type"),
        evaluator_types=_unique_text_tuple(
            raw.get("evaluator_types"), "evaluator type"
        ),
        lifecycle_types=_unique_text_tuple(
            raw.get("lifecycle_types"), "lifecycle type", allow_empty=True
        ),
        source_provenance=_source_provenance(raw.get("source_provenance")),
        descriptor_digest=str(raw.get("descriptor_digest") or ""),
    )
    return _finish_digest(
        value, "descriptor_digest", "provider descriptor", require_supplied=False
    )


def candidate_bundle_from_dict(raw: Mapping[str, Any]) -> CandidateBundleV1:
    _strict_fields(raw, CandidateBundleV1, "candidate bundle")
    assets = tuple(
        _locked_asset(value, "behavior asset")
        for value in _sequence(
            raw.get("behavior_assets"), "behavior assets", allow_empty=True
        )
    )
    skills = tuple(
        _locked_component(value, "Skill")
        for value in _sequence(raw.get("skills"), "Skills", allow_empty=True)
    )
    mcp_servers = tuple(
        _locked_component(value, "MCP server")
        for value in _sequence(raw.get("mcp_servers"), "MCP servers", allow_empty=True)
    )
    blockers = _text_tuple(raw.get("blockers"), "candidate blocker", allow_empty=True)
    portability = _portability(raw.get("portability"), blockers)
    value = CandidateBundleV1(
        schema_version=_schema(raw, "candidate bundle"),
        provider_id=validate_id(raw.get("provider_id") or "", kind="provider id"),
        candidate_ref=_text(raw.get("candidate_ref"), "candidate ref", 500),
        display_name=_text(raw.get("display_name"), "candidate display name", 200),
        agent_config=_mapping(raw.get("agent_config"), "agent config"),
        model_route=_mapping(raw.get("model_route"), "model route"),
        behavior_assets=assets,
        skills=skills,
        mcp_servers=mcp_servers,
        agent_code=_agent_code(raw.get("agent_code")),
        required_credentials=_credential_names(raw.get("required_credentials")),
        portability=portability,
        blockers=blockers,
        bundle_digest=str(raw.get("bundle_digest") or ""),
    )
    return _finish_digest(
        value, "bundle_digest", "candidate bundle", require_supplied=False
    )


def provider_task_from_dict(raw: Mapping[str, Any]) -> ProviderTaskV1:
    _strict_fields(raw, ProviderTaskV1, "provider task")
    inputs = tuple(
        _input_part(value)
        for value in _sequence(raw.get("input"), "task input parts")
    )
    stopping = tuple(
        _stopping_condition(value)
        for value in _sequence(
            raw.get("stopping_policy"), "stopping policy", allow_empty=True
        )
    )
    evaluator_ids = _id_tuple(raw.get("evaluator_ids"), "evaluator")
    blockers = _text_tuple(raw.get("blockers"), "task blocker", allow_empty=True)
    return ProviderTaskV1(
        id=_provider_ref(raw.get("id"), "provider task id"),
        title=_text(raw.get("title"), "task title", 300),
        input=inputs,
        interaction=_interaction(raw.get("interaction")),
        stopping_policy=stopping,
        lifecycle=_lifecycle(raw.get("lifecycle")),
        credential_names=_credential_names(raw.get("credential_names")),
        integration_config=_mapping(
            raw.get("integration_config"), "integration config"
        ),
        evaluator_ids=evaluator_ids,
        metadata=_mapping(raw.get("metadata"), "task metadata"),
        portability=_portability(raw.get("portability"), blockers),
        blockers=blockers,
    )


def provider_scenario_from_dict(raw: Mapping[str, Any]) -> ProviderScenarioV1:
    _strict_fields(raw, ProviderScenarioV1, "provider scenario")
    path = _id_tuple(raw.get("path"), "scenario path segment")
    parent_path = _id_tuple(
        raw.get("parent_path"), "parent scenario path segment", allow_empty=True
    )
    scenario_id = validate_id(raw.get("id") or "", kind="provider scenario id")
    if not path or path[-1] != scenario_id:
        raise ValueError("scenario path must end with its id")
    if parent_path != path[:-1]:
        raise ValueError("scenario parent_path must equal path without its final id")
    tasks = tuple(
        _scenario_task_edge(value)
        for value in _sequence(
            raw.get("tasks"), "scenario task edges", allow_empty=True
        )
    )
    return ProviderScenarioV1(
        id=scenario_id,
        path=path,
        parent_path=parent_path,
        weight=_positive_number(raw.get("weight"), "scenario weight"),
        must_pass=_boolean(raw.get("must_pass"), "scenario must_pass"),
        tasks=tasks,
    )


def provider_evaluator_from_dict(raw: Mapping[str, Any]) -> ProviderEvaluatorV1:
    _strict_fields(raw, ProviderEvaluatorV1, "provider evaluator")
    blockers = _text_tuple(raw.get("blockers"), "evaluator blocker", allow_empty=True)
    threshold = _number(raw.get("threshold"), "evaluator threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("evaluator threshold must be between 0 and 1")
    return ProviderEvaluatorV1(
        id=validate_id(raw.get("id") or "", kind="provider evaluator id"),
        type=_text(raw.get("type"), "evaluator type", 100),
        implementation=_implementation(raw.get("implementation")),
        config=_mapping(raw.get("config"), "public evaluator config"),
        evidence=_unique_text_tuple(raw.get("evidence"), "evaluator evidence"),
        weight=_positive_number(raw.get("weight"), "evaluator weight"),
        threshold=threshold,
        must_pass=_boolean(raw.get("must_pass"), "evaluator must_pass"),
        portability=_portability(raw.get("portability"), blockers),
        blockers=blockers,
    )


def suite_bundle_from_dict(raw: Mapping[str, Any]) -> SuiteBundleV1:
    _strict_fields(raw, SuiteBundleV1, "suite bundle")
    tasks = tuple(
        provider_task_from_dict(value)
        for value in _sequence(raw.get("tasks"), "provider tasks")
    )
    scenarios = tuple(
        provider_scenario_from_dict(value)
        for value in _sequence(raw.get("scenarios"), "provider scenarios")
    )
    evaluators = tuple(
        provider_evaluator_from_dict(value)
        for value in _sequence(raw.get("evaluators"), "provider evaluators")
    )
    _unique((value.id for value in tasks), "task ids")
    _unique(("/".join(value.path) for value in scenarios), "scenario paths")
    _unique((value.id for value in evaluators), "evaluator ids")
    task_ids = {value.id for value in tasks}
    evaluator_ids = {value.id for value in evaluators}
    for task in tasks:
        missing = sorted(set(task.evaluator_ids) - evaluator_ids)
        if missing:
            raise ValueError(
                f"task {task.id!r} references unknown evaluator(s): {missing}"
            )
    for scenario in scenarios:
        missing = sorted(
            str(edge["task_id"])
            for edge in scenario.tasks
            if edge["task_id"] not in task_ids
        )
        if missing:
            raise ValueError(
                f"scenario {'/'.join(scenario.path)!r} references unknown task(s): "
                f"{missing}"
            )
    value = SuiteBundleV1(
        schema_version=_schema(raw, "suite bundle"),
        provider_id=validate_id(raw.get("provider_id") or "", kind="provider id"),
        suite_ref=_text(raw.get("suite_ref"), "suite ref", 500),
        title=_text(raw.get("title"), "suite title", 300),
        objective=_text(raw.get("objective"), "suite objective", 4000),
        attempts=_positive_int(raw.get("attempts"), "suite attempts"),
        tasks=tasks,
        scenarios=scenarios,
        evaluators=evaluators,
        metadata=_mapping(raw.get("metadata"), "suite metadata"),
        bundle_digest=str(raw.get("bundle_digest") or ""),
    )
    return _finish_digest(value, "bundle_digest", "suite bundle", require_supplied=False)


def private_evaluation_bundle_from_dict(
    raw: Mapping[str, Any],
) -> PrivateEvaluationBundleV1:
    _strict_fields(raw, PrivateEvaluationBundleV1, "private evaluation bundle")
    tasks = tuple(
        _private_task(value)
        for value in _sequence(
            raw.get("tasks"), "private evaluation tasks", allow_empty=True
        )
    )
    _unique((value.task_id for value in tasks), "private task ids")
    value = PrivateEvaluationBundleV1(
        schema_version=_schema(raw, "private evaluation bundle"),
        provider_id=validate_id(raw.get("provider_id") or "", kind="provider id"),
        suite_digest=_digest(raw.get("suite_digest"), "suite digest"),
        tasks=tasks,
        metadata=_mapping(raw.get("metadata"), "private evaluation metadata"),
        private_digest=str(raw.get("private_digest") or ""),
    )
    return _finish_digest(
        value,
        "private_digest",
        "private evaluation bundle",
        require_supplied=False,
    )


def preparation_receipt_from_dict(
    raw: Mapping[str, Any],
) -> PreparationReceiptV1:
    _strict_fields(raw, PreparationReceiptV1, "preparation receipt")
    value = PreparationReceiptV1(
        schema_version=_schema(raw, "preparation receipt"),
        provider_id=validate_id(raw.get("provider_id") or "", kind="provider id"),
        provider_lock_digest=_digest(
            raw.get("provider_lock_digest"), "provider lock digest"
        ),
        candidate_digest=_digest(raw.get("candidate_digest"), "candidate digest"),
        suite_digest=_digest(raw.get("suite_digest"), "suite digest"),
        frozen_references=_record_tuple(
            raw.get("frozen_references"), "frozen references"
        ),
        materialized_resources=_record_tuple(
            raw.get("materialized_resources"), "materialized resources"
        ),
        lifecycle_outputs=_record_tuple(
            raw.get("lifecycle_outputs"), "lifecycle outputs"
        ),
        runtime_artifacts=_record_tuple(
            raw.get("runtime_artifacts"), "runtime artifacts"
        ),
        cleanup_obligations=_record_tuple(
            raw.get("cleanup_obligations"), "cleanup obligations"
        ),
        prepared_at=_text(raw.get("prepared_at"), "preparation timestamp", 100),
        receipt_digest=str(raw.get("receipt_digest") or ""),
    )
    return _finish_digest(
        value, "receipt_digest", "preparation receipt", require_supplied=False
    )


def cell_request_from_dict(raw: Mapping[str, Any]) -> CellRequestV1:
    _strict_fields(raw, CellRequestV1, "cell request")
    candidate = candidate_bundle_from_dict(
        _mapping(raw.get("candidate"), "cell candidate")
    )
    preparation = preparation_receipt_from_dict(
        _mapping(raw.get("preparation"), "cell preparation receipt")
    )
    task = provider_task_from_dict(_mapping(raw.get("task"), "cell task"))
    value = CellRequestV1(
        schema_version=_schema(raw, "cell request"),
        provider_id=validate_id(raw.get("provider_id") or "", kind="provider id"),
        plan_digest=_digest(raw.get("plan_digest"), "plan digest"),
        cell_id=validate_id(raw.get("cell_id") or "", kind="cell id"),
        candidate_digest=_digest(raw.get("candidate_digest"), "candidate digest"),
        suite_digest=_digest(raw.get("suite_digest"), "suite digest"),
        preparation_receipt_digest=_digest(
            raw.get("preparation_receipt_digest"), "preparation receipt digest"
        ),
        candidate=candidate.to_dict(),
        preparation=preparation.to_dict(),
        task=task.to_dict(),
        attempt=_positive_int(raw.get("attempt"), "attempt"),
        runtime_lock_digest=_digest(
            raw.get("runtime_lock_digest"), "runtime lock digest"
        ),
        credential_profile_names=_unique_text_tuple(
            raw.get("credential_profile_names"),
            "credential profile",
            allow_empty=True,
        ),
        budget=_budget(raw.get("budget")),
        request_digest=str(raw.get("request_digest") or ""),
    )
    if value.provider_id != candidate.provider_id:
        raise ValueError("cell candidate provider differs from the request")
    if value.provider_id != preparation.provider_id:
        raise ValueError("cell preparation provider differs from the request")
    if value.candidate_digest != candidate.bundle_digest:
        raise ValueError("cell candidate digest does not bind the candidate")
    if value.candidate_digest != preparation.candidate_digest:
        raise ValueError("cell preparation does not bind the candidate")
    if value.suite_digest != preparation.suite_digest:
        raise ValueError("cell preparation does not bind the suite")
    if value.preparation_receipt_digest != preparation.receipt_digest:
        raise ValueError("cell preparation digest does not bind the receipt")
    return _finish_digest(
        value, "request_digest", "cell request", require_supplied=False
    )


def cell_result_from_dict(raw: Mapping[str, Any]) -> CellResultV1:
    _strict_fields(raw, CellResultV1, "cell result")
    status = str(raw.get("status") or "")
    if status not in _CELL_STATUS:
        raise ValueError(f"unsupported cell result status: {status!r}")
    failure_raw = raw.get("failure")
    failure = (
        None
        if failure_raw is None
        else _mapping(failure_raw, "cell result failure")
    )
    if status == "succeeded" and failure is not None:
        raise ValueError("a succeeded cell result cannot contain failure details")
    if status != "succeeded" and failure is None:
        raise ValueError("a non-success cell result requires failure details")
    value = CellResultV1(
        schema_version=_schema(raw, "cell result"),
        provider_id=validate_id(raw.get("provider_id") or "", kind="provider id"),
        cell_id=validate_id(raw.get("cell_id") or "", kind="cell id"),
        request_digest=_digest(raw.get("request_digest"), "cell request digest"),
        status=status,  # type: ignore[arg-type]
        output=_mapping(raw.get("output"), "cell output"),
        conversation=_record_tuple(raw.get("conversation"), "conversation"),
        tool_calls=_record_tuple(raw.get("tool_calls"), "tool calls"),
        usage=_usage(raw.get("usage")),
        evidence_refs=_record_tuple(raw.get("evidence_refs"), "evidence refs"),
        failure=failure,
        cleanup=_mapping(raw.get("cleanup"), "cleanup receipt"),
        result_digest=str(raw.get("result_digest") or ""),
    )
    return _finish_digest(
        value, "result_digest", "cell result", require_supplied=False
    )


def provider_contract_schemas() -> dict[str, dict[str, Any]]:
    """Return language-neutral strict JSON Schemas for the provider boundary."""

    common = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
    }

    def object_schema(
        name: str,
        fields: Sequence[str],
        properties: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            **common,
            "$id": f"https://fugue.local/schemas/provider/{name}-v1.json",
            "title": name,
            "type": "object",
            "required": list(fields),
            "properties": dict(properties),
        }

    def strict_object(properties: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": dict(properties),
        }

    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    text = {"type": "string", "minLength": 1}
    texts = {"type": "array", "items": text}
    free_object = {"type": "object"}
    free_records = {"type": "array", "items": free_object}
    locked_component = strict_object(
        {
            "id": text,
            "digest": digest,
            "source": text,
            "metadata": free_object,
        }
    )
    locked_asset = strict_object(
        {
            "kind": text,
            "id": text,
            "digest": digest,
            "source": text,
            "metadata": free_object,
        }
    )
    source_provenance = strict_object(
        {
            "repository": text,
            "revision": text,
            "source_digest": digest,
        }
    )
    agent_code = strict_object(
        {
            "revision": text,
            "digest": digest,
            "files": {"type": "array", "items": locked_asset},
        }
    )
    input_part = strict_object({"type": text, "payload": free_object})
    interaction = strict_object(
        {
            "type": text,
            "profile": {"type": ["string", "null"]},
            "turns": free_records,
            "directions": texts,
            "config": free_object,
        }
    )
    stopping_condition = strict_object(
        {
            "type": text,
            "limit": {"type": "number", "exclusiveMinimum": 0},
        }
    )
    lifecycle_step = strict_object({"profile": text, "args": free_object})
    lifecycle = strict_object(
        {
            "bootstrap": {"oneOf": [locked_asset, {"type": "null"}]},
            "setup": {"type": "array", "items": lifecycle_step},
            "teardown": {"type": "array", "items": lifecycle_step},
        }
    )
    provider_task = strict_object(
        {
            "id": text,
            "title": text,
            "input": {"type": "array", "items": input_part, "minItems": 1},
            "interaction": interaction,
            "stopping_policy": {
                "type": "array",
                "items": stopping_condition,
            },
            "lifecycle": lifecycle,
            "credential_names": {
                "type": "array",
                "items": text,
                "uniqueItems": True,
            },
            "integration_config": free_object,
            "evaluator_ids": {
                "type": "array",
                "items": text,
                "uniqueItems": True,
            },
            "metadata": free_object,
            "portability": {"enum": sorted(_PORTABILITY)},
            "blockers": texts,
        }
    )
    scenario_edge = strict_object(
        {
            "task_id": text,
            "weight": {"type": "number", "exclusiveMinimum": 0},
            "must_pass": {"type": "boolean"},
        }
    )
    provider_scenario = strict_object(
        {
            "id": text,
            "path": {"type": "array", "items": text, "minItems": 1},
            "parent_path": texts,
            "weight": {"type": "number", "exclusiveMinimum": 0},
            "must_pass": {"type": "boolean"},
            "tasks": {"type": "array", "items": scenario_edge},
        }
    )
    implementation = strict_object(
        {
            "kind": text,
            "id": text,
            "digest": digest,
            "runtime": {"oneOf": [locked_component, {"type": "null"}]},
        }
    )
    provider_evaluator = strict_object(
        {
            "id": text,
            "type": text,
            "implementation": implementation,
            "config": free_object,
            "evidence": texts,
            "weight": {"type": "number", "exclusiveMinimum": 0},
            "threshold": {"type": "number", "minimum": 0, "maximum": 1},
            "must_pass": {"type": "boolean"},
            "portability": {"enum": sorted(_PORTABILITY)},
            "blockers": texts,
        }
    )
    private_task = strict_object(
        {
            "task_id": text,
            "expected": free_object,
            "evaluator_config": free_object,
        }
    )
    budget = {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {
            "max_cost_usd": {"type": "number", "exclusiveMinimum": 0},
            "max_seconds": {"type": "number", "exclusiveMinimum": 0},
            "max_steps": {"type": "number", "exclusiveMinimum": 0},
        },
    }
    usage = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "input_tokens": {"type": "number", "minimum": 0},
            "output_tokens": {"type": "number", "minimum": 0},
            "cost_usd": {"type": "number", "minimum": 0},
            "latency_ms": {"type": "number", "minimum": 0},
            "tool_calls": {"type": "number", "minimum": 0},
        },
    }
    descriptor = object_schema(
        "provider-descriptor",
        tuple(ProviderDescriptorV1.__dataclass_fields__),
        {
            "schema_version": {"const": 1},
            "provider_id": text,
            "display_name": text,
            "provider_version": text,
            "protocol_version": {"const": 1},
            "capabilities": {"type": "array", "items": text, "uniqueItems": True},
            "task_types": {"type": "array", "items": text, "uniqueItems": True},
            "input_types": {"type": "array", "items": text, "uniqueItems": True},
            "evaluator_types": {
                "type": "array",
                "items": text,
                "uniqueItems": True,
            },
            "lifecycle_types": {
                "type": "array",
                "items": text,
                "uniqueItems": True,
            },
            "source_provenance": source_provenance,
            "descriptor_digest": digest,
        },
    )
    candidate = object_schema(
        "candidate-bundle",
        tuple(CandidateBundleV1.__dataclass_fields__),
        {
            "schema_version": {"const": 1},
            "provider_id": text,
            "candidate_ref": text,
            "display_name": text,
            "agent_config": free_object,
            "model_route": free_object,
            "behavior_assets": {"type": "array", "items": locked_asset},
            "skills": {"type": "array", "items": locked_component},
            "mcp_servers": {"type": "array", "items": locked_component},
            "agent_code": agent_code,
            "required_credentials": {
                "type": "array",
                "items": text,
                "uniqueItems": True,
            },
            "portability": {"enum": sorted(_PORTABILITY)},
            "blockers": {"type": "array", "items": text},
            "bundle_digest": digest,
        },
    )
    suite = object_schema(
        "suite-bundle",
        tuple(SuiteBundleV1.__dataclass_fields__),
        {
            "schema_version": {"const": 1},
            "provider_id": text,
            "suite_ref": text,
            "title": text,
            "objective": text,
            "attempts": {"type": "integer", "minimum": 1},
            "tasks": {"type": "array", "items": provider_task, "minItems": 1},
            "scenarios": {
                "type": "array",
                "items": provider_scenario,
                "minItems": 1,
            },
            "evaluators": {
                "type": "array",
                "items": provider_evaluator,
                "minItems": 1,
            },
            "metadata": free_object,
            "bundle_digest": digest,
        },
    )
    private = object_schema(
        "private-evaluation-bundle",
        tuple(PrivateEvaluationBundleV1.__dataclass_fields__),
        {
            "schema_version": {"const": 1},
            "provider_id": text,
            "suite_digest": digest,
            "tasks": {"type": "array", "items": private_task},
            "metadata": free_object,
            "private_digest": digest,
        },
    )
    preparation = object_schema(
        "preparation-receipt",
        tuple(PreparationReceiptV1.__dataclass_fields__),
        {
            "schema_version": {"const": 1},
            "provider_id": text,
            "provider_lock_digest": digest,
            "candidate_digest": digest,
            "suite_digest": digest,
            "frozen_references": free_records,
            "materialized_resources": free_records,
            "lifecycle_outputs": free_records,
            "runtime_artifacts": free_records,
            "cleanup_obligations": free_records,
            "prepared_at": text,
            "receipt_digest": digest,
        },
    )
    request = object_schema(
        "cell-request",
        tuple(CellRequestV1.__dataclass_fields__),
        {
            "schema_version": {"const": 1},
            "provider_id": text,
            "plan_digest": digest,
            "cell_id": text,
            "candidate_digest": digest,
            "suite_digest": digest,
            "preparation_receipt_digest": digest,
            "candidate": strict_object(candidate["properties"]),
            "preparation": strict_object(preparation["properties"]),
            "task": provider_task,
            "attempt": {"type": "integer", "minimum": 1},
            "runtime_lock_digest": digest,
            "credential_profile_names": {
                "type": "array",
                "items": text,
                "uniqueItems": True,
            },
            "budget": budget,
            "request_digest": digest,
        },
    )
    result = object_schema(
        "cell-result",
        tuple(CellResultV1.__dataclass_fields__),
        {
            "schema_version": {"const": 1},
            "provider_id": text,
            "cell_id": text,
            "request_digest": digest,
            "status": {"enum": sorted(_CELL_STATUS)},
            "output": free_object,
            "conversation": free_records,
            "tool_calls": free_records,
            "usage": usage,
            "evidence_refs": free_records,
            "failure": {"type": ["object", "null"]},
            "cleanup": free_object,
            "result_digest": digest,
        },
    )
    return {
        "provider-descriptor-v1": descriptor,
        "candidate-bundle-v1": candidate,
        "suite-bundle-v1": suite,
        "private-evaluation-bundle-v1": private,
        "preparation-receipt-v1": preparation,
        "cell-request-v1": request,
        "cell-result-v1": result,
    }


def _finish_digest(
    value: Any,
    field_name: str,
    label: str,
    *,
    require_supplied: bool,
) -> Any:
    raw = value.to_dict()
    supplied = str(raw.get(field_name) or "")
    unsigned = {**raw, field_name: ""}
    digest = stable_digest(unsigned)
    if require_supplied and not supplied:
        raise ValueError(f"{label} requires {field_name}")
    if supplied and supplied != digest:
        raise ValueError(f"{label} {field_name} does not match")
    return replace(value, **{field_name: digest})


def _strict_fields(raw: Mapping[str, Any], cls: type[Any], label: str) -> None:
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown field(s): {', '.join(unknown)}")
    missing = sorted(
        field_name
        for field_name, declared in cls.__dataclass_fields__.items()
        if declared.default is MISSING and declared.default_factory is MISSING
        and field_name not in raw
    )
    if missing:
        raise ValueError(f"{label} is missing field(s): {', '.join(missing)}")


def _schema(raw: Mapping[str, Any], label: str) -> int:
    value = raw.get("schema_version")
    if value != PROVIDER_PROTOCOL_VERSION:
        raise ValueError(
            f"{label} schema_version must be {PROVIDER_PROTOCOL_VERSION}"
        )
    return PROVIDER_PROTOCOL_VERSION


def _protocol(value: Any) -> int:
    if value != PROVIDER_PROTOCOL_VERSION:
        raise ValueError(
            f"provider protocol_version must be {PROVIDER_PROTOCOL_VERSION}"
        )
    return PROVIDER_PROTOCOL_VERSION


def _source_provenance(raw: Any) -> dict[str, str]:
    value = _mapping(raw, "source provenance")
    allowed = {"repository", "revision", "source_digest"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            "source provenance contains unknown field(s): " + ", ".join(unknown)
        )
    if set(value) != allowed:
        raise ValueError("source provenance requires repository, revision, source_digest")
    return {
        "repository": _text(value["repository"], "source repository", 1000),
        "revision": _text(value["revision"], "source revision", 200),
        "source_digest": _digest(value["source_digest"], "source digest"),
    }


def _locked_asset(raw: Any, label: str) -> dict[str, Any]:
    value = _mapping(raw, label)
    allowed = {"kind", "id", "digest", "source", "metadata"}
    _reject_unknown(value, allowed, label)
    return {
        "kind": _text(value.get("kind"), f"{label} kind", 100),
        "id": _text(value.get("id"), f"{label} id", 500),
        "digest": _digest(value.get("digest"), f"{label} digest"),
        "source": _text(value.get("source"), f"{label} source", 2000),
        "metadata": _mapping(value.get("metadata", {}), f"{label} metadata"),
    }


def _locked_component(raw: Any, label: str) -> dict[str, Any]:
    value = _mapping(raw, label)
    allowed = {"id", "digest", "source", "metadata"}
    _reject_unknown(value, allowed, label)
    return {
        "id": _text(value.get("id"), f"{label} id", 500),
        "digest": _digest(value.get("digest"), f"{label} digest"),
        "source": _text(value.get("source"), f"{label} source", 2000),
        "metadata": _mapping(value.get("metadata", {}), f"{label} metadata"),
    }


def _agent_code(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "agent code")
    allowed = {"revision", "digest", "files"}
    _reject_unknown(value, allowed, "agent code")
    files = tuple(
        _locked_asset(item, "agent code file")
        for item in _sequence(value.get("files"), "agent code files")
    )
    if stable_digest([item["digest"] for item in files]) != _digest(
        value.get("digest"), "agent code digest"
    ):
        raise ValueError("agent code digest must bind the ordered file digests")
    return {
        "revision": _text(value.get("revision"), "agent code revision", 200),
        "digest": str(value["digest"]),
        "files": list(files),
    }


def _input_part(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "task input part")
    allowed = {"type", "payload"}
    _reject_unknown(value, allowed, "task input part")
    return {
        "type": _text(value.get("type"), "task input part type", 100),
        "payload": _mapping(value.get("payload"), "task input part payload"),
    }


def _interaction(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "task interaction")
    allowed = {"type", "profile", "turns", "directions", "config"}
    _reject_unknown(value, allowed, "task interaction")
    interaction_type = _text(value.get("type"), "interaction type", 100)
    turns = _record_tuple(value.get("turns"), "interaction turns")
    directions = _text_tuple(
        value.get("directions"), "interaction direction", allow_empty=True
    )
    profile_raw = value.get("profile")
    profile = (
        None
        if profile_raw is None
        else _text(profile_raw, "interaction profile", 300)
    )
    if interaction_type == "scripted" and not turns:
        raise ValueError("scripted interaction requires at least one turn")
    if interaction_type == "model" and (not profile or not directions):
        raise ValueError("model interaction requires a profile and directions")
    return {
        "type": interaction_type,
        "profile": profile,
        "turns": list(turns),
        "directions": list(directions),
        "config": _mapping(value.get("config"), "interaction config"),
    }


def _stopping_condition(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "stopping condition")
    allowed = {"type", "limit"}
    _reject_unknown(value, allowed, "stopping condition")
    return {
        "type": _text(value.get("type"), "stopping condition type", 100),
        "limit": _positive_number(value.get("limit"), "stopping condition limit"),
    }


def _lifecycle(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "task lifecycle")
    allowed = {"bootstrap", "setup", "teardown"}
    _reject_unknown(value, allowed, "task lifecycle")
    bootstrap_raw = value.get("bootstrap")
    bootstrap = (
        None
        if bootstrap_raw is None
        else _locked_asset(bootstrap_raw, "bootstrap asset")
    )
    return {
        "bootstrap": bootstrap,
        "setup": list(_lifecycle_steps(value.get("setup"), "setup")),
        "teardown": list(_lifecycle_steps(value.get("teardown"), "teardown")),
    }


def _lifecycle_steps(raw: Any, label: str) -> tuple[dict[str, Any], ...]:
    result = []
    for item in _sequence(raw, f"{label} lifecycle steps", allow_empty=True):
        value = _mapping(item, f"{label} lifecycle step")
        allowed = {"profile", "args"}
        _reject_unknown(value, allowed, f"{label} lifecycle step")
        result.append(
            {
                "profile": _text(
                    value.get("profile"), f"{label} lifecycle profile", 300
                ),
                "args": _mapping(value.get("args"), f"{label} lifecycle args"),
            }
        )
    return tuple(result)


def _scenario_task_edge(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "scenario task edge")
    allowed = {"task_id", "weight", "must_pass"}
    _reject_unknown(value, allowed, "scenario task edge")
    return {
        "task_id": _provider_ref(value.get("task_id"), "scenario task id"),
        "weight": _positive_number(value.get("weight"), "task edge weight"),
        "must_pass": _boolean(value.get("must_pass"), "task edge must_pass"),
    }


def _implementation(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "evaluator implementation")
    allowed = {"kind", "id", "digest", "runtime"}
    _reject_unknown(value, allowed, "evaluator implementation")
    runtime_raw = value.get("runtime")
    return {
        "kind": _text(value.get("kind"), "evaluator implementation kind", 100),
        "id": _text(value.get("id"), "evaluator implementation id", 500),
        "digest": _digest(
            value.get("digest"), "evaluator implementation digest"
        ),
        "runtime": (
            None
            if runtime_raw is None
            else _locked_component(runtime_raw, "evaluator runtime")
        ),
    }


def _private_task(raw: Any) -> PrivateEvaluationTaskV1:
    value = _mapping(raw, "private evaluation task")
    _strict_fields(value, PrivateEvaluationTaskV1, "private evaluation task")
    return PrivateEvaluationTaskV1(
        task_id=_provider_ref(value.get("task_id"), "private task id"),
        expected=_mapping(value.get("expected"), "private expected values"),
        evaluator_config=_mapping(
            value.get("evaluator_config"), "private evaluator config"
        ),
    )


def _budget(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "cell budget")
    allowed = {"max_cost_usd", "max_seconds", "max_steps"}
    _reject_unknown(value, allowed, "cell budget")
    result: dict[str, Any] = {}
    for key in allowed:
        if key in value and value[key] is not None:
            result[key] = _positive_number(value[key], key)
    if not result:
        raise ValueError("cell budget must declare at least one bound")
    return result


def _usage(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "cell usage")
    allowed = {
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_ms",
        "tool_calls",
    }
    _reject_unknown(value, allowed, "cell usage")
    result: dict[str, Any] = {}
    for key, item in value.items():
        _non_negative_number(item, f"usage {key}")
        # JSON protocol digests bind the provider's exact JSON value. Preserve
        # an integer as an integer so a valid provider-authored digest does not
        # change merely because validation coerced ``1`` to ``1.0``.
        result[key] = item
    return result


def _portability(raw: Any, blockers: Sequence[str]) -> Portability:
    value = str(raw or "")
    if value not in _PORTABILITY:
        raise ValueError(f"unsupported portability classification: {value!r}")
    if value == "blocked" and not blockers:
        raise ValueError("blocked artifacts require at least one blocker")
    if value != "blocked" and blockers:
        raise ValueError("only blocked artifacts may declare blockers")
    return value  # type: ignore[return-value]


def _credential_names(raw: Any) -> tuple[str, ...]:
    names = _unique_text_tuple(raw, "credential name", allow_empty=True)
    invalid = [name for name in names if not _ENV_NAME.fullmatch(name)]
    if invalid:
        raise ValueError(f"invalid credential name(s): {invalid}")
    return names


def _provider_ref(raw: Any, label: str) -> str:
    value = _text(raw, label, 1000)
    if value.startswith("/") or value.endswith("/") or "//" in value:
        raise ValueError(f"{label} must be a relative canonical reference")
    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"{label} cannot contain traversal segments")
    if any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", segment) is None
        for segment in segments
    ):
        raise ValueError(f"{label} contains unsupported characters")
    return value


def _id_tuple(raw: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    values = tuple(
        validate_id(value, kind=label)
        for value in _text_tuple(raw, label, allow_empty=allow_empty)
    )
    _unique(values, f"{label}s")
    return values


def _unique_text_tuple(
    raw: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    values = _text_tuple(raw, label, allow_empty=allow_empty)
    _unique(values, f"{label}s")
    return values


def _text_tuple(
    raw: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    values = _sequence(raw, f"{label}s", allow_empty=allow_empty)
    result = tuple(_text(value, label, 4000) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{label}s must not be empty")
    return result


def _record_tuple(raw: Any, label: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        _mapping(value, label.removesuffix("s") or "record")
        for value in _sequence(raw, label, allow_empty=True)
    )


def _sequence(raw: Any, label: str, *, allow_empty: bool = False) -> list[Any]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be an array")
    if not allow_empty and not raw:
        raise ValueError(f"{label} must not be empty")
    return list(raw)


def _mapping(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    return _json_value(dict(raw))


def _reject_unknown(
    raw: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown field(s): {', '.join(unknown)}")


def _text(raw: Any, label: str, max_length: int) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty string")
    value = raw.strip()
    if len(value) > max_length:
        raise ValueError(f"{label} exceeds {max_length} characters")
    return value


def _digest(raw: Any, label: str) -> str:
    value = str(raw or "")
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _positive_int(raw: Any, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(f"{label} must be a positive integer")
    return raw


def _positive_number(raw: Any, label: str) -> int | float:
    value = _number(raw, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _non_negative_number(raw: Any, label: str) -> int | float:
    value = _number(raw, label)
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _number(raw: Any, label: str) -> int | float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = raw
    if value != value or value in {float("inf"), float("-inf")}:
        raise ValueError(f"{label} must be finite")
    return value


def _boolean(raw: Any, label: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{label} must be a boolean")
    return raw


def _unique(values: Sequence[Any] | Any, label: str) -> None:
    materialized = list(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{label} must be unique")


def _json_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("provider artifacts must contain JSON values only") from exc
