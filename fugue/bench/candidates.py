from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

CANDIDATE_IDENTITY_SCHEMA_VERSION = 1
EXECUTION_IDENTITY_SCHEMA_VERSION = 1


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
    model_route: Mapping[str, Any],
    prompt_digest: str | None,
    skills: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    integrations: Sequence[Mapping[str, Any]],
    agent: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> ResolvedCandidate:
    """Resolve identity once; callers must reuse the returned representation."""

    definition = _canonical(
        {
            "identity_schema_version": CANDIDATE_IDENTITY_SCHEMA_VERSION,
            "harness": harness,
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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))
