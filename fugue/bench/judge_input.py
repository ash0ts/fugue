"""Versioned, provider-visible privacy transform for comparison judge inputs."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fugue.redaction import sensitive_key

COMPARISON_JUDGE_INPUT_SANITIZER_VERSION = 3
COMPARISON_JUDGE_INPUT_SANITIZER_CONTRACT = (
    "fugue-judge-input-sanitization-v3"
)
COMPARISON_JUDGE_INPUT_SANITIZER_TRANSFORM = (
    "high-confidence-credential-redaction-v3"
)
COMPARISON_JUDGE_INPUT_SANITIZER_IMPLEMENTATION_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()

_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(?P<prefix>"
    r"(?:authorization\s*[:=]\s*bearer\s+|"
    r"(?:api[_ -]?key|token|secret)\s*[:=]\s*)"
    r")"
    r"(?:"
    r"(?P<quote>[\"'])(?P<quoted>[^\"'\r\n]*)(?P=quote)"
    r"|"
    r"(?P<bare><[^>\r\n]{1,200}>|\[[^\]\r\n]{1,200}\]|"
    r"[^\s\r\n\\\"')\]},;]+)"
    r")"
)
_PASSWORD_ASSIGNMENT = re.compile(
    r"(?im)(?P<prefix>\b(?:password|passphrase)\s*[:=]\s*)"
    r"(?P<value>[^\r\n]+)"
)
_LINE_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?im)(?P<prefix>^\s*(?:[-*+]\s+)?"
    r"(?:api[_ -]?key|token|secret|password|passphrase)\s*[:=]\s*)"
    r"(?P<value>[^\r\n]+)$"
)
_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_EXACT_PLACEHOLDERS = frozenset(
    {
        "[redacted]",
        "<redacted>",
        "example",
        "example-placeholder",
        "placeholder",
        "dummy",
        "dummy-key",
        "fake",
        "fake-token",
        "sample",
        "testing",
        "test-key",
        "test_key",
        "not-a-real-key",
        "not_a_real_key",
        "from-env",
        "from_env",
        "from-env-file",
        "from_env_file",
        "first-key",
        "first_key",
        "second-key",
        "second_key",
        "changeme",
        "your-key",
        "your_key",
    }
)
_SAFE_CODE_EXPRESSION = re.compile(
    r"""
    (?:
        Path\(\s*(?:
            [\"'][^\"'\r\n]{0,200}[\"']
            |
            [A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*
            |
            (?:values|config|env)\[\s*[\"'][^\"'\r\n]{1,200}[\"']\s*\]
        )\s*\)(?:\.read_text\(\))?
        |
        (?:os\.)?getenv\(\s*[\"'][^\"'\r\n]{1,200}[\"']\s*\)
        |
        os\.environ(?:
            \[\s*[\"'][^\"'\r\n]{1,200}[\"']\s*\]
            |
            \.get\(\s*[\"'][^\"'\r\n]{1,200}[\"']\s*\)
        )
        |
        (?:values|config|env)(?:
            \[\s*[\"'][^\"'\r\n]{1,200}[\"']\s*\]
            |
            \.get\(\s*[\"'][^\"'\r\n]{1,200}[\"']\s*\)
        )
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def _safe_credential_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    return bool(
        normalized in _EXACT_PLACEHOLDERS
        or (
            normalized.startswith("${")
            and normalized.endswith("}")
            and len(normalized) <= 200
        )
        or (
            normalized.startswith("{{")
            and normalized.endswith("}}")
            and len(normalized) <= 200
        )
        or bool(_SAFE_CODE_EXPRESSION.fullmatch(value.strip()))
        or normalized in {"none", "null", "nil"}
    )


def _redact_text(value: str) -> str:
    def line_assignment(match: re.Match[str]) -> str:
        candidate = match.group("value").strip()
        quote = (
            candidate[0]
            if len(candidate) >= 2
            and candidate[0] in {"\"", "'"}
            and candidate[-1] == candidate[0]
            else ""
        )
        compared = candidate[1:-1] if quote else candidate
        if _safe_credential_placeholder(compared):
            return match.group(0)
        return f"{match.group('prefix')}{quote}[redacted]{quote}"

    def password_assignment(match: re.Match[str]) -> str:
        candidate = match.group("value").strip()
        if _safe_credential_placeholder(candidate):
            return match.group(0)
        return f"{match.group('prefix')}[redacted]"

    def assignment(match: re.Match[str]) -> str:
        quoted = match.group("quoted")
        candidate = quoted if quoted is not None else str(match.group("bare") or "")
        if _safe_credential_placeholder(candidate):
            return match.group(0)
        prefix = match.group("prefix")
        quote = match.group("quote") or ""
        return f"{prefix}{quote}[redacted]{quote}"

    def token(match: re.Match[str]) -> str:
        candidate = match.group(0)
        return candidate if _safe_credential_placeholder(candidate) else "[redacted]"

    def redact_line(value_line: str) -> str:
        body = value_line.rstrip("\r\n")
        ending = value_line[len(body) :]
        line_match = _LINE_CREDENTIAL_ASSIGNMENT.fullmatch(body)
        if line_match is not None:
            return line_assignment(line_match) + ending
        return _TOKEN.sub(
            token,
            _CREDENTIAL_ASSIGNMENT.sub(
                assignment,
                _PASSWORD_ASSIGNMENT.sub(password_assignment, value_line),
            ),
        )

    return "".join(redact_line(line) for line in value.splitlines(keepends=True))


def sanitize_comparison_judge_value(value: Any, *, key: str = "") -> Any:
    """Redact likely credentials without corrupting explicit test fixtures."""

    if sensitive_key(key) and isinstance(value, str):
        if _safe_credential_placeholder(value):
            return value
        return "[redacted]"
    if isinstance(value, Mapping):
        return {
            str(name): sanitize_comparison_judge_value(item, key=str(name))
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_comparison_judge_value(item, key=key) for item in value]
    if isinstance(value, tuple):
        return tuple(
            sanitize_comparison_judge_value(item, key=key) for item in value
        )
    if isinstance(value, str):
        return _redact_text(value)
    return value


def comparison_judge_input_sanitizer_conformance_cases() -> tuple[
    dict[str, Any], ...
]:
    """Return deterministic redaction and safe-placeholder compatibility cases.

    These cases are intentionally model-free. They are used to prove that a
    calibration payload remains byte-equivalent when no transform is needed,
    while the same locked implementation still redacts representative secrets
    and preserves the explicit placeholders used by task fixtures and code.
    """

    return (
        {
            "id": "authorization-bearer-redaction",
            "family": "credential_redaction",
            "input": "Authorization: Bearer compatibility-secret-value-123456",
            "expected_output": "Authorization: Bearer [redacted]",
            "expected_transformed": True,
        },
        {
            "id": "nested-sensitive-key-redaction",
            "family": "credential_redaction",
            "input": {
                "metadata": {"api_key": "compatibility-secret-value-123456"}
            },
            "expected_output": {"metadata": {"api_key": "[redacted]"}},
            "expected_transformed": True,
        },
        {
            "id": "password-assignment-redaction",
            "family": "credential_redaction",
            "input": "password = compatibility-secret-value-123456",
            "expected_output": "password = [redacted]",
            "expected_transformed": True,
        },
        {
            "id": "token-pattern-redaction",
            "family": "credential_redaction",
            "input": "Use sk-compatibilityproof123456 for this request.",
            "expected_output": "Use [redacted] for this request.",
            "expected_transformed": True,
        },
        {
            "id": "environment-placeholder-preservation",
            "family": "safe_placeholder_preservation",
            "input": {"api_key": "${WANDB_API_KEY}"},
            "expected_output": {"api_key": "${WANDB_API_KEY}"},
            "expected_transformed": False,
        },
        {
            "id": "path-read-expression-preservation",
            "family": "safe_placeholder_preservation",
            "input": {"api_key": "Path('wandb.key').read_text()"},
            "expected_output": {"api_key": "Path('wandb.key').read_text()"},
            "expected_transformed": False,
        },
        {
            "id": "config-expression-preservation",
            "family": "safe_placeholder_preservation",
            "input": {"token": "config['token']"},
            "expected_output": {"token": "config['token']"},
            "expected_transformed": False,
        },
        {
            "id": "from-env-placeholder-preservation",
            "family": "safe_placeholder_preservation",
            "input": "api_key: from-env",
            "expected_output": "api_key: from-env",
            "expected_transformed": False,
        },
    )
