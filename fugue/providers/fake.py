from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.bench.provider_contract import (
    candidate_bundle_from_dict,
    cell_request_from_dict,
    cell_result_from_dict,
    preparation_receipt_from_dict,
    private_evaluation_bundle_from_dict,
    provider_descriptor_from_dict,
    suite_bundle_from_dict,
)

_PROVIDER_ID = "fugue-fake"


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise SystemExit(
            "usage: python -m fugue.providers.fake "
            "{describe|resolve-candidate|resolve-suite|prepare|run-cell}"
        )
    operation = arguments[0]
    request = _read_request()
    if operation == "describe":
        _reject_unknown(request, set(), "describe request")
        response = _descriptor()
    elif operation == "resolve-candidate":
        _reject_exact(request, {"candidate_ref"}, "candidate request")
        response = _candidate(str(request["candidate_ref"]))
    elif operation == "resolve-suite":
        _reject_exact(request, {"suite_ref"}, "suite request")
        response = _suite_response(str(request["suite_ref"]))
    elif operation == "prepare":
        response = _prepare(request)
    elif operation == "run-cell":
        response = _run_cell(request)
    else:
        raise SystemExit(f"unsupported fake-provider operation: {operation}")
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


def _descriptor() -> dict[str, Any]:
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return provider_descriptor_from_dict(
        {
            "schema_version": 1,
            "provider_id": _PROVIDER_ID,
            "display_name": "Fugue reference evaluation provider",
            "provider_version": "1.0.0",
            "protocol_version": 1,
            "capabilities": [
                "candidate_resolution",
                "suite_resolution",
                "trusted_preparation",
                "single_cell_execution",
                "cancellation",
            ],
            "task_types": ["single_turn", "multi_turn"],
            "input_types": ["text", "structured_context"],
            "evaluator_types": ["deterministic", "judge"],
            "lifecycle_types": ["bootstrap", "setup", "teardown"],
            "source_provenance": {
                "repository": "fugue",
                "revision": "working-tree",
                "source_digest": source_digest,
            },
            "descriptor_digest": "",
        }
    ).to_dict()


def _candidate(candidate_ref: str) -> dict[str, Any]:
    if candidate_ref not in {"reference", "candidate"}:
        raise ValueError(f"unknown fake candidate: {candidate_ref}")
    file_digest = stable_digest(
        {"path": "fugue/providers/fake.py", "candidate_ref": candidate_ref}
    )
    code_digest = stable_digest([file_digest])
    return candidate_bundle_from_dict(
        {
            "schema_version": 1,
            "provider_id": _PROVIDER_ID,
            "candidate_ref": candidate_ref,
            "display_name": f"Fake {candidate_ref}",
            "agent_config": {
                "system_prompt": "Answer with locked evidence.",
                "mode": candidate_ref,
            },
            "model_route": {
                "provider": "fake",
                "model": "deterministic-v1",
            },
            "behavior_assets": [
                {
                    "kind": "prompt",
                    "id": "system-prompt",
                    "digest": stable_digest("Answer with locked evidence."),
                    "source": "inline",
                    "metadata": {},
                }
            ],
            "skills": [],
            "mcp_servers": [],
            "agent_code": {
                "revision": "working-tree",
                "digest": code_digest,
                "files": [
                    {
                        "kind": "python",
                        "id": "fake-provider",
                        "digest": file_digest,
                        "source": "fugue/providers/fake.py",
                        "metadata": {},
                    }
                ],
            },
            "required_credentials": [],
            "portability": "portable",
            "blockers": [],
            "bundle_digest": "",
        }
    ).to_dict()


def _suite_response(suite_ref: str) -> dict[str, Any]:
    if suite_ref != "conformance":
        raise ValueError(f"unknown fake suite: {suite_ref}")
    evaluator_digest = stable_digest("fake-answer-evaluator-v1")
    suite = suite_bundle_from_dict(
        {
            "schema_version": 1,
            "provider_id": _PROVIDER_ID,
            "suite_ref": suite_ref,
            "title": "Provider conformance suite",
            "objective": "Exercise portable and provider-bound task contracts.",
            "attempts": 1,
            "tasks": [
                {
                    "id": "single-turn",
                    "title": "Single turn",
                    "input": [
                        {
                            "type": "text",
                            "payload": {"text": "Return the word evidence."},
                        }
                    ],
                    "interaction": {
                        "type": "single_turn",
                        "profile": None,
                        "turns": [],
                        "directions": [],
                        "config": {},
                    },
                    "stopping_policy": [{"type": "steps", "limit": 3}],
                    "lifecycle": {
                        "bootstrap": None,
                        "setup": [],
                        "teardown": [],
                    },
                    "credential_names": [],
                    "integration_config": {},
                    "evaluator_ids": ["answer-evidence"],
                    "metadata": {"feature_ids": ["single-turn"]},
                    "portability": "portable",
                    "blockers": [],
                },
                {
                    "id": "structured-multi-turn",
                    "title": "Structured multi-turn",
                    "input": [
                        {
                            "type": "text",
                            "payload": {"text": "Read the structured context."},
                        },
                        {
                            "type": "structured_context",
                            "payload": {"schema": "fake/v1", "value": {"count": 2}},
                        },
                    ],
                    "interaction": {
                        "type": "scripted",
                        "profile": None,
                        "turns": [
                            {"role": "user", "content": "Now return the count."}
                        ],
                        "directions": [],
                        "config": {},
                    },
                    "stopping_policy": [
                        {"type": "steps", "limit": 5},
                        {"type": "timeout", "limit": 10},
                    ],
                    "lifecycle": {
                        "bootstrap": None,
                        "setup": [
                            {"profile": "fake-setup", "args": {"value": 2}}
                        ],
                        "teardown": [
                            {"profile": "fake-teardown", "args": {"verify": True}}
                        ],
                    },
                    "credential_names": [],
                    "integration_config": {},
                    "evaluator_ids": ["answer-evidence"],
                    "metadata": {"feature_ids": ["structured-context", "multi-turn"]},
                    "portability": "provider_bound",
                    "blockers": [],
                },
            ],
            "scenarios": [
                {
                    "id": "conformance",
                    "path": ["conformance"],
                    "parent_path": [],
                    "weight": 1,
                    "must_pass": True,
                    "tasks": [
                        {
                            "task_id": "single-turn",
                            "weight": 1,
                            "must_pass": True,
                        },
                        {
                            "task_id": "structured-multi-turn",
                            "weight": 1,
                            "must_pass": True,
                        },
                    ],
                }
            ],
            "evaluators": [
                {
                    "id": "answer-evidence",
                    "type": "deterministic",
                    "implementation": {
                        "kind": "builtin",
                        "id": "answer-contains",
                        "digest": evaluator_digest,
                        "runtime": None,
                    },
                    "config": {},
                    "evidence": ["answer"],
                    "weight": 1,
                    "threshold": 1,
                    "must_pass": True,
                    "portability": "portable",
                    "blockers": [],
                }
            ],
            "metadata": {"source": "fake-provider"},
            "bundle_digest": "",
        }
    )
    private = private_evaluation_bundle_from_dict(
        {
            "schema_version": 1,
            "provider_id": _PROVIDER_ID,
            "suite_digest": suite.bundle_digest,
            "tasks": [
                {
                    "task_id": "single-turn",
                    "expected": {"contains": "evidence"},
                    "evaluator_config": {"answer-evidence": {"contains": "evidence"}},
                },
                {
                    "task_id": "structured-multi-turn",
                    "expected": {"equals": "2"},
                    "evaluator_config": {"answer-evidence": {"equals": "2"}},
                },
            ],
            "metadata": {"classification": "private-evaluation"},
            "private_digest": "",
        }
    )
    return {
        "suite": suite.to_dict(),
        "private_evaluation": private.to_dict(),
    }


def _prepare(request: dict[str, Any]) -> dict[str, Any]:
    _reject_exact(
        request,
        {"provider_lock_digest", "candidate", "suite"},
        "preparation request",
    )
    candidate = candidate_bundle_from_dict(_mapping(request["candidate"], "candidate"))
    suite = suite_bundle_from_dict(_mapping(request["suite"], "suite"))
    lock_digest = _digest(request["provider_lock_digest"], "provider lock digest")
    return preparation_receipt_from_dict(
        {
            "schema_version": 1,
            "provider_id": _PROVIDER_ID,
            "provider_lock_digest": lock_digest,
            "candidate_digest": candidate.bundle_digest,
            "suite_digest": suite.bundle_digest,
            "frozen_references": [],
            "materialized_resources": [
                {
                    "kind": "provider-task-v1",
                    "id": task.id,
                    "digest": stable_digest(task.to_dict()),
                }
                for task in suite.tasks
            ],
            "lifecycle_outputs": [],
            "runtime_artifacts": [
                {
                    "kind": "fake-runtime",
                    "digest": stable_digest("fake-runtime-v1"),
                }
            ],
            "cleanup_obligations": [
                {"kind": "fake-resource", "required_state": "deleted"}
            ],
            "prepared_at": datetime.now(UTC).isoformat(),
            "receipt_digest": "",
        }
    ).to_dict()


def _run_cell(request: dict[str, Any]) -> dict[str, Any]:
    cell = cell_request_from_dict(request)
    prompt = " ".join(
        str(part.get("payload", {}).get("text", ""))
        for part in cell.task["input"]
        if part.get("type") == "text"
    )
    answer = "2" if "structured context" in prompt.lower() else "evidence"
    return cell_result_from_dict(
        {
            "schema_version": 1,
            "provider_id": _PROVIDER_ID,
            "cell_id": cell.cell_id,
            "request_digest": cell.request_digest,
            "status": "succeeded",
            "output": {"answer": answer},
            "conversation": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            "tool_calls": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cost_usd": 0,
                "latency_ms": 1,
                "tool_calls": 0,
            },
            "evidence_refs": [
                {"kind": "mechanism", "ref": "fake://deterministic"}
            ],
            "failure": None,
            "cleanup": {"complete": True, "remaining_resources": 0},
            "result_digest": "",
        }
    ).to_dict()


def _read_request() -> dict[str, Any]:
    try:
        value = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"stdin must contain one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("stdin must contain one JSON object")
    return value


def _mapping(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    return raw


def _reject_unknown(raw: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown field(s): {', '.join(unknown)}")


def _reject_exact(raw: dict[str, Any], fields: set[str], label: str) -> None:
    _reject_unknown(raw, fields, label)
    missing = sorted(fields - set(raw))
    if missing:
        raise ValueError(f"{label} is missing field(s): {', '.join(missing)}")


def _digest(raw: Any, label: str) -> str:
    value = str(raw or "")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
