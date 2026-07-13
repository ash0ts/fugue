from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_TOKEN_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"']+"),
    re.compile(r"(?i)((?:api[_ -]?key|token|password|secret)\s*[:=]\s*)[^\s\"']+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return (
        normalized in {"authorization", "password", "secret", "token", "apikey"}
        or "api_key" in normalized
        or normalized.endswith(
            ("_access_token", "_refresh_token", "_auth_token", "_password", "_secret")
        )
    )


def secrets_from_env(env: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value
                for key, value in env.items()
                if sensitive_key(key) and len(value.strip()) >= 8
            },
            key=len,
            reverse=True,
        )
    )


def redact_text(value: str, secrets: Iterable[str] = ()) -> str:
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[redacted]")
    for pattern in _TOKEN_PATTERNS:
        result = pattern.sub(
            lambda match: (
                f"{match.group(1)}[redacted]" if match.lastindex else "[redacted]"
            ),
            result,
        )
    return result


def redact_value(value: Any, *, secrets: Iterable[str] = (), key: str = "") -> Any:
    if sensitive_key(key):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(name): redact_value(item, secrets=secrets, key=str(name))
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item, secrets=secrets, key=key) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, secrets=secrets, key=key) for item in value)
    if isinstance(value, str):
        return redact_text(value, secrets)
    return value
