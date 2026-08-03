"""Host-side deterministic scorer for the Skill Creator confirmatory Study.

The scorer parses and validates the returned package itself.  It deliberately
does not accept an Agent-authored validation receipt or a fixture-side pass
flag as evidence of correctness.
"""

import hashlib
import json
import re

_DIMENSIONS = dict.fromkeys(
    (
        "artifact_validity",
        "frontmatter_semantics",
        "compatibility_selection",
        "name_help_consistency",
        "behavior_preservation",
        "packaging",
        "instruction_quality",
        "dependency_secret_safety",
        "assigned_script_use",
    ),
    False,
)
_ALLOWED_METADATA = {
    "name",
    "description",
    "license",
    "allowed-tools",
    "metadata",
    "compatibility",
}
_DISPOSITIONS = {"created", "updated", "unchanged", "rejected", "diagnosed"}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _array(value):
    return value if isinstance(value, list) else []


def _text(value):
    return value if isinstance(value, str) else ""


def _parse_one_json(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    match = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _scalar(raw):
    value = raw.strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~"}:
        return None
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    if value.startswith("[") or value.startswith("{"):
        return [] if value.startswith("[") else {}
    return value


def _frontmatter(value):
    text = _text(value)
    if not text.startswith("---\n"):
        return {}, "", False
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, "", False
    metadata = {}
    for line in text[4:end].splitlines():
        if not line or line.startswith((" ", "\t")) or ":" not in line:
            return {}, "", False
        key, raw = line.split(":", 1)
        key = key.strip()
        if not key or key in metadata:
            return {}, "", False
        metadata[key] = _scalar(raw)
    return metadata, text[end + 5 :], True


def _canonical_validate(value):
    metadata, body, parsed = _frontmatter(value)
    if not parsed or set(metadata) - _ALLOWED_METADATA:
        return False, metadata, body
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*", name
    ):
        return False, metadata, body
    if not 1 <= len(name) <= 64:
        return False, metadata, body
    if (
        not isinstance(description, str)
        or not 1 <= len(description.strip()) <= 1024
        or "<" in description
        or ">" in description
    ):
        return False, metadata, body
    if "compatibility" in metadata:
        compatibility = metadata["compatibility"]
        if not isinstance(compatibility, str) or len(compatibility) > 500:
            return False, metadata, body
    return bool(body.strip()), metadata, body


def _contains_groups(value, groups):
    normalized = " ".join(_text(value).lower().split())
    return all(
        any(" ".join(_text(term).lower().split()) in normalized for term in group)
        for group in _array(groups)
        if isinstance(group, list) and group
    )


def _artifact_validity(result, expected, task):
    files = result.get("files")
    findings = result.get("findings")
    memo = result.get("maintainer_memo")
    return bool(
        set(result)
        == {
            "schema_version",
            "task_id",
            "disposition",
            "skill_name",
            "files",
            "findings",
            "maintainer_memo",
        }
        and result.get("schema_version") == 1
        and result.get("task_id") == task.get("id")
        and result.get("disposition") in _DISPOSITIONS
        and result.get("disposition") == expected.get("disposition")
        and isinstance(result.get("skill_name"), str)
        and isinstance(files, dict)
        and all(
            isinstance(path, str) and isinstance(content, str) and content
            for path, content in files.items()
        )
        and isinstance(findings, dict)
        and isinstance(memo, str)
        and 20 <= len(memo.strip()) <= 800
    )


def _frontmatter_semantics(files, expected):
    if expected.get("disposition") == "rejected":
        return not files
    skill_path = _text(expected.get("skill_path"))
    if not skill_path:
        return True
    valid, _metadata, _body = _canonical_validate(files.get(skill_path))
    return valid


def _compatibility_selection(files, expected):
    contract = _mapping(expected.get("compatibility"))
    policy = contract.get("policy")
    if policy == "not_applicable":
        return True
    skill_path = _text(expected.get("skill_path"))
    _valid, metadata, _body = _canonical_validate(files.get(skill_path))
    present = "compatibility" in metadata
    value = metadata.get("compatibility")
    if policy == "absent":
        return not present
    if policy == "exact":
        return present and value == contract.get("value")
    if policy == "length":
        return isinstance(value, str) and len(value) == contract.get("length")
    if policy != "required" or not isinstance(value, str) or not value:
        return False
    maximum = int(contract.get("max_length") or 500)
    return len(value) <= maximum and _contains_groups(value, contract.get("groups"))


def _name_help_consistency(result, files, expected):
    exact = _text(expected.get("skill_name"))
    if result.get("skill_name") != exact:
        return False
    if expected.get("disposition") == "rejected":
        return not files and _contains_groups(
            result.get("maintainer_memo"), expected.get("memo_groups")
        )
    required_findings = expected.get("findings")
    if isinstance(required_findings, dict):
        if result.get("findings") != required_findings:
            return False
        combined = "\n".join(files.values())
        if not _contains_groups(combined, expected.get("script_groups")):
            return False
        if any(item in combined for item in _array(expected.get("script_forbidden"))):
            return False
        return True
    skill_path = _text(expected.get("skill_path"))
    _valid, metadata, _body = _canonical_validate(files.get(skill_path))
    return metadata.get("name") == exact and 1 <= len(exact) <= 64


def _behavior_preservation(files, expected):
    for path, digest in _mapping(expected.get("exact_file_sha256")).items():
        content = files.get(path)
        if not isinstance(content, str) or hashlib.sha256(content.encode()).hexdigest() != digest:
            return False
    skill_path = _text(expected.get("skill_path"))
    if skill_path:
        _valid, metadata, body = _canonical_validate(files.get(skill_path))
        for key, value in _mapping(expected.get("preserved_metadata")).items():
            if metadata.get(key) != value:
                return False
        body_digest = expected.get("preserved_body_sha256")
        if body_digest and hashlib.sha256(body.encode()).hexdigest() != body_digest:
            return False
        if not _contains_groups(body, expected.get("preserved_body_groups")):
            return False
    for path, digest in _mapping(expected.get("preserved_file_sha256")).items():
        content = files.get(path)
        if not isinstance(content, str) or hashlib.sha256(content.encode()).hexdigest() != digest:
            return False
    return True


def _packaging(files, expected):
    allowed = set(str(value) for value in _array(expected.get("allowed_paths")))
    if set(files) != allowed:
        return False
    return all(
        path
        and not path.startswith("/")
        and ".." not in path.split("/")
        and not path.endswith("/")
        for path in files
    )


def _instruction_quality(result, files, expected):
    if expected.get("disposition") == "rejected":
        return _contains_groups(result.get("maintainer_memo"), expected.get("memo_groups"))
    if expected.get("findings") is not None:
        return _contains_groups("\n".join(files.values()), expected.get("script_groups"))
    skill_path = _text(expected.get("skill_path"))
    _valid, metadata, body = _canonical_validate(files.get(skill_path))
    description = metadata.get("description")
    return bool(
        isinstance(description, str)
        and 40 <= len(description) <= 1024
        and re.search(r"^#\s+\S", body, re.MULTILINE)
        and _contains_groups(body, expected.get("instruction_groups"))
    )


_UNSAFE = ("pip install", "uv add", "npm install", "curl ", "wget ", "git clone", "rm -rf")
_SECRET = re.compile(
    r"(?:api[_ -]?key|password|secret)\s*[:=]\s*(?!never\b|none\b|redacted\b|<|\$\{)[^\s]+",
    re.IGNORECASE,
)


def _dependency_secret_safety(result, files):
    combined = "\n".join(files.values()) + "\n" + _text(result.get("maintainer_memo"))
    for line in combined.splitlines():
        lowered = line.lower()
        for command in _UNSAFE:
            if command in lowered and not re.search(
                r"(?:do not|never|must not|without|avoid|forbidden|prohibited)[^.\n]{0,100}"
                + re.escape(command),
                lowered,
            ):
                return False
    return not _SECRET.search(combined)


def _assigned_script_use(evidence, expected):
    required = [str(value) for value in _array(expected.get("required_opened_suffixes"))]
    if not required:
        return True
    activity = json.dumps(
        {
            "opened_paths": evidence.get("opened_paths"),
            "tool_calls": evidence.get("tool_calls"),
        },
        sort_keys=True,
    ).lower()
    return all(suffix.lower() in activity for suffix in required)


def score(task, output, evidence):
    expected = _mapping(evidence.get("expected"))
    result = _parse_one_json(output)
    if result is None or not expected:
        return dict(_DIMENSIONS)
    files = _mapping(result.get("files"))
    return {
        "artifact_validity": _artifact_validity(result, expected, _mapping(task)),
        "frontmatter_semantics": _frontmatter_semantics(files, expected),
        "compatibility_selection": _compatibility_selection(files, expected),
        "name_help_consistency": _name_help_consistency(result, files, expected),
        "behavior_preservation": _behavior_preservation(files, expected),
        "packaging": _packaging(files, expected),
        "instruction_quality": _instruction_quality(result, files, expected),
        "dependency_secret_safety": _dependency_secret_safety(result, files),
        "assigned_script_use": _assigned_script_use(evidence, expected),
    }
