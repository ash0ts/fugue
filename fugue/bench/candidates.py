from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

CANDIDATE_IDENTITY_SCHEMA_VERSION = 1
EXECUTION_IDENTITY_SCHEMA_VERSION = 1
COMPARISON_IDENTITY_SCHEMA_VERSION = 1
ATTEMPT_IDENTITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResolvedCandidate:
    """One resolved, presentation-free candidate and its execution envelope."""

    candidate_id: str
    execution_fingerprint: str
    _definition_json: str = field(repr=False)
    _execution_definition_json: str = field(repr=False)

    @property
    def definition(self) -> dict[str, Any]:
        return json.loads(self._definition_json)

    @property
    def execution_definition(self) -> dict[str, Any]:
        return json.loads(self._execution_definition_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "definition": _canonical(self.definition),
            "execution_fingerprint": self.execution_fingerprint,
            "execution_definition": _canonical(self.execution_definition),
        }


def resolve_candidate(
    *,
    harness: str,
    harness_version: str,
    model_route: Mapping[str, Any],
    prompt_digest: str | None,
    skills: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    integrations: Sequence[Mapping[str, Any]],
    agent: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> ResolvedCandidate:
    """Resolve identity once; callers must reuse the returned representation."""

    if not isinstance(harness, str) or not harness.strip():
        raise ValueError("harness must be a non-empty string")
    if not isinstance(harness_version, str) or not harness_version.strip():
        raise ValueError("harness_version must be a non-empty string")

    definition = _canonical(
        {
            "identity_schema_version": CANDIDATE_IDENTITY_SCHEMA_VERSION,
            "harness": harness,
            "harness_version": harness_version,
            "model_route": model_route,
            "prompt_digest": prompt_digest,
            "skills": list(skills),
            "context": context,
            "integrations": list(integrations),
            "agent": agent,
        }
    )
    candidate_id = stable_digest(definition)
    execution_definition = _canonical(
        {
            "identity_schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            **dict(execution),
        }
    )
    return ResolvedCandidate(
        candidate_id=candidate_id,
        execution_fingerprint=stable_digest(execution_definition),
        _definition_json=_canonical_json(definition),
        _execution_definition_json=_canonical_json(execution_definition),
    )


def stable_digest(value: Any) -> str:
    payload = _canonical_json(value)
    return hashlib.sha256(payload.encode()).hexdigest()


def comparison_example_id(
    *, dataset_id: str, workload_id: str, logical_task_id: str
) -> str:
    """Identify one logical comparison example independently of its trial."""

    return stable_digest(
        {
            "identity_schema_version": COMPARISON_IDENTITY_SCHEMA_VERSION,
            "dataset": dataset_id.strip(),
            "workload": workload_id.strip(),
            "logical_task": logical_task_id.strip(),
        }
    )


def attempt_identity(
    *,
    task_id: str,
    arm: str,
    harness: str,
    attempt: int,
    candidate: str,
    runtime: str,
) -> dict[str, Any]:
    """Return the canonical identity shared by design, execution, and evidence."""

    if attempt < 1:
        raise ValueError("attempt must be positive")
    values = {
        "task_id": task_id.strip(),
        "arm": arm.strip(),
        "harness": harness.strip(),
        "attempt": attempt,
        "candidate": candidate.strip(),
        "runtime": runtime.strip(),
    }
    missing = [key for key, value in values.items() if value in {"", None}]
    if missing:
        raise ValueError(
            "attempt identity requires " + ", ".join(sorted(missing))
        )
    return values


def attempt_id(**coordinates: Any) -> str:
    """Hash one immutable attempt independently of a particular run id."""

    return stable_digest(
        {
            "schema_version": ATTEMPT_IDENTITY_SCHEMA_VERSION,
            **attempt_identity(**coordinates),
        }
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))
