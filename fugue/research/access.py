from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fugue.research.contracts import ResearchError

AGENT_SCOPES = frozenset(
    {
        "approval:request",
        "mcp:agent",
        "research:read",
        "research:write",
        "result:record",
        "study:preview",
        "study:start-approved",
        "study:watch",
        "trace:preview",
        "trace:run",
    }
)
OPERATOR_SCOPES = frozenset({"administration:read", "study:cancel"})
KNOWN_SCOPES = AGENT_SCOPES | OPERATOR_SCOPES
_CURRENT_GRANT: ContextVar[AccessGrantV1 | None] = ContextVar(
    "fugue_research_access_grant",
    default=None,
)


@dataclass(frozen=True)
class AccessGrantV1:
    schema_version: int
    token_digest: str
    subject: str
    instance_id: str
    research_ids: tuple[str, ...]
    scopes: tuple[str, ...]
    expires_at: str


class ResearchAccessAuthorizer:
    def __init__(self, grants: Iterable[AccessGrantV1]) -> None:
        self.grants = tuple(grants)
        if not self.grants:
            raise RuntimeError("at least one research access grant is required")

    @classmethod
    def for_token(
        cls,
        token: str,
        *,
        subject: str = "test-agent",
        instance_id: str = "local-test",
    ) -> ResearchAccessAuthorizer:
        if not token:
            raise RuntimeError("a research API key is required")
        return cls(
            (
                AccessGrantV1(
                    schema_version=1,
                    token_digest=token_digest(token),
                    subject=subject,
                    instance_id=instance_id,
                    research_ids=("*",),
                    scopes=tuple(sorted(AGENT_SCOPES)),
                    expires_at="9999-12-31T23:59:59+00:00",
                ),
            )
        )

    @classmethod
    def from_file(cls, path: Path) -> ResearchAccessAuthorizer:
        resolved = path.resolve(strict=True)
        if (
            path.is_symlink()
            or not resolved.is_file()
            or resolved.stat().st_size > 1_000_000
        ):
            raise RuntimeError("research access grants must be a small regular file")
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or set(raw) != {
            "schema_version",
            "instance_id",
            "grants",
        }:
            raise RuntimeError("research access grant file has an invalid shape")
        if raw["schema_version"] != 1 or not isinstance(raw["grants"], list):
            raise RuntimeError("research access grant file version must be 1")
        grants = tuple(
            _grant_from_dict(item, instance_id=str(raw["instance_id"]))
            for item in raw["grants"]
        )
        return cls(grants)

    def authorize(
        self,
        token: str,
        *,
        scope: str,
        research_id: str | None = None,
    ) -> AccessGrantV1 | None:
        supplied = token_digest(token)
        current = datetime.now(UTC)
        selected: AccessGrantV1 | None = None
        for grant in self.grants:
            matches = hmac.compare_digest(supplied, grant.token_digest)
            if not matches:
                continue
            try:
                expires = datetime.fromisoformat(grant.expires_at)
            except ValueError:
                continue
            if expires.tzinfo is None or expires <= current:
                continue
            if scope not in grant.scopes:
                continue
            if scope != "mcp:agent" and "*" not in grant.research_ids:
                if research_id is None or research_id not in grant.research_ids:
                    continue
            selected = grant
        return selected


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def access_scope(method: str, path: str) -> str:
    if path.startswith("/mcp"):
        return "mcp:agent"
    if path == "/v1/research-publications":
        return "administration:read"
    if method in {"GET", "HEAD"}:
        return (
            "study:watch"
            if "/events" in path or path.endswith("/outcome")
            else "research:read"
        )
    if path.endswith(":request-approval"):
        return "approval:request"
    if "trace-audits:preview" in path:
        return "trace:preview"
    if path.endswith("/trace-audits"):
        return "trace:run"
    if (
        "task-suites:derive-preview" in path
        or path.endswith("experiments:preview")
        or path.endswith("comparisons:preview")
    ):
        return "study:preview"
    if path.endswith("studies:preview"):
        return "study:preview"
    if path.endswith(":cancel"):
        return "study:cancel"
    if path.endswith(":start") and "/comparisons/" in path:
        return "study:start-approved"
    if path.endswith("/experiments") or (
        path.startswith("/v1/research/") and path.endswith("/studies")
    ):
        return "study:start-approved"
    if path.endswith("/updates"):
        return "result:record"
    return "research:write"


def bind_current_grant(grant: AccessGrantV1) -> Token[AccessGrantV1 | None]:
    return _CURRENT_GRANT.set(grant)


def reset_current_grant(token: Token[AccessGrantV1 | None]) -> None:
    _CURRENT_GRANT.reset(token)


def require_current_access(scope: str, research_id: str) -> None:
    grant = _CURRENT_GRANT.get()
    if grant is None:
        raise ResearchError(
            "access_context_missing",
            "the authenticated research access context is unavailable",
            category="policy",
        )
    if scope not in grant.scopes:
        raise ResearchError(
            "scope_forbidden",
            "the authenticated grant does not allow this operation",
            category="policy",
        )
    if "*" not in grant.research_ids and research_id not in grant.research_ids:
        raise ResearchError(
            "research_forbidden",
            "the authenticated grant does not allow this Research",
            category="policy",
        )


def research_id_from_path(path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    if (
        len(parts) >= 3
        and parts[0] == "v1"
        and parts[1]
        in {
            "research",
            "studies",
        }
    ):
        return parts[2].split(":", 1)[0]
    return None


def _grant_from_dict(raw: Any, *, instance_id: str) -> AccessGrantV1:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "token_digest",
        "subject",
        "instance_id",
        "research_ids",
        "scopes",
        "expires_at",
    }:
        raise RuntimeError("research access grant has an invalid shape")
    if raw["schema_version"] != 1 or raw["instance_id"] != instance_id:
        raise RuntimeError("research access grant is for another Fugue instance")
    digest = str(raw["token_digest"])
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("research access token digest is invalid")
    research_ids = _string_tuple(raw["research_ids"], "research ids")
    scopes = _string_tuple(raw["scopes"], "research scopes")
    unknown = sorted(set(scopes) - KNOWN_SCOPES)
    if unknown:
        raise RuntimeError("research access grant contains unknown scopes")
    return AccessGrantV1(
        schema_version=1,
        token_digest=digest,
        subject=str(raw["subject"]),
        instance_id=instance_id,
        research_ids=research_ids,
        scopes=scopes,
        expires_at=str(raw["expires_at"]),
    )


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise RuntimeError(f"{label} must be a non-empty string list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise RuntimeError(f"{label} must not contain duplicates")
    return result
