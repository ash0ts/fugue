"""Candidate-neutral deterministic checks for compatibility-aware Skills."""

import json
import re

_DIMENSIONS = dict.fromkeys(
    (
        "schema_validity",
        "compatibility_preservation",
        "name_rules",
        "packaging",
        "instruction_quality",
        "dependency_secret_safety",
    ),
    False,
)


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


def _frontmatter(value):
    text = _text(value)
    if not text.startswith("---\n"):
        return {}, ""
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, ""
    metadata = {}
    for line in text[4:end].splitlines():
        if not line or line.startswith((" ", "\t")) or ":" not in line:
            return {}, ""
        key, raw = line.split(":", 1)
        key = key.strip()
        if not key or key in metadata:
            return {}, ""
        metadata[key] = raw.strip().strip("'\"")
    return metadata, text[end + 5 :]


def _contains_groups(text, groups):
    normalized = " ".join(text.lower().split())
    return all(
        any(" ".join(str(term).lower().split()) in normalized for term in group)
        for group in _array(groups)
        if isinstance(group, list) and group
    )


def _schema_validity(result, expected, task):
    validation = _mapping(result.get("validation"))
    files = _mapping(result.get("files"))
    return bool(
        set(result)
        == {
            "schema_version",
            "task_id",
            "skill_name",
            "files",
            "validation",
            "maintainer_memo",
        }
        and result.get("schema_version") == 1
        and result.get("task_id") == task.get("id")
        and isinstance(result.get("skill_name"), str)
        and files
        and all(
            isinstance(path, str) and isinstance(content, str) and content
            for path, content in files.items()
        )
        and set(validation) == {"command", "passed"}
        and isinstance(validation.get("command"), str)
        and validation.get("passed") is True
        and isinstance(result.get("maintainer_memo"), str)
        and 20 <= len(result["maintainer_memo"].strip()) <= 800
        and expected
    )


def _compatibility_preservation(files, expected):
    skill_path = _text(expected.get("skill_path"))
    metadata, body = _frontmatter(files.get(skill_path))
    compatibility = _text(metadata.get("compatibility"))
    if not compatibility or len(compatibility) > 500:
        return False
    if not _contains_groups(
        compatibility,
        expected.get("compatibility_groups"),
    ):
        return False
    preserved_body = expected.get("preserved_body")
    if isinstance(preserved_body, str) and body != preserved_body:
        return False
    for path, content in _mapping(expected.get("preserved_files")).items():
        if files.get(path) != content:
            return False
    for key, value in _mapping(expected.get("preserved_metadata")).items():
        if metadata.get(key) != value:
            return False
    return True


def _name_rules(result, files, expected):
    name = _text(result.get("skill_name"))
    skill_path = _text(expected.get("skill_path"))
    metadata, _body = _frontmatter(files.get(skill_path))
    command = _text(_mapping(result.get("validation")).get("command"))
    exact = _text(expected.get("skill_name"))
    return bool(
        name == exact
        and metadata.get("name") == exact
        and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
        and 1 <= len(name) <= 64
        and f"/{name}" in command
        and name in _text(result.get("maintainer_memo"))
    )


def _packaging(files, expected):
    allowed = set(str(value) for value in _array(expected.get("allowed_paths")))
    return bool(
        allowed
        and set(files) == allowed
        and _text(expected.get("skill_path")) in files
        and all(
            not path.startswith("/")
            and ".." not in path.split("/")
            and not path.endswith((".py", ".sh", ".js", ".ts"))
            for path in files
        )
    )


def _instruction_quality(files, expected):
    metadata, body = _frontmatter(files.get(_text(expected.get("skill_path"))))
    description = _text(metadata.get("description"))
    return bool(
        40 <= len(description) <= 1024
        and _contains_groups(description, expected.get("description_groups"))
        and _contains_groups(body, expected.get("instruction_groups"))
        and re.search(r"^#\s+\S", body, re.MULTILINE)
        and re.search(r"^##\s+Workflow\b", body, re.MULTILINE | re.IGNORECASE)
        and re.search(r"^##\s+Validation\b", body, re.MULTILINE | re.IGNORECASE)
        and re.search(
            r"^##\s+Failure handling\b",
            body,
            re.MULTILINE | re.IGNORECASE,
        )
    )


def _dependency_secret_safety(result, files):
    combined = "\n".join(files.values())
    command = _text(_mapping(result.get("validation")).get("command"))
    forbidden = (
        "pip install",
        "uv add",
        "npm install",
        "curl ",
        "wget ",
        "git clone",
        "rm -rf",
        "gh pr",
        "api_key=",
        "api-key=",
        "password=",
        "secret=",
    )
    return bool(
        command.startswith("python3 scripts/quick_validate.py skills/")
        and not any(term in combined.lower() for term in forbidden)
        and not any(term in command.lower() for term in forbidden)
        and not re.search(
            r"(?:WANDB|ANTHROPIC|OPENAI)_API_KEY\s*=\s*\S+",
            combined,
            re.IGNORECASE,
        )
    )


def score(task, output, evidence):
    expected = _mapping(evidence.get("expected"))
    result = _parse_one_json(output)
    if result is None or not expected:
        return dict(_DIMENSIONS)
    files = _mapping(result.get("files"))
    return {
        "schema_validity": _schema_validity(result, expected, _mapping(task)),
        "compatibility_preservation": _compatibility_preservation(
            files,
            expected,
        ),
        "name_rules": _name_rules(result, files, expected),
        "packaging": _packaging(files, expected),
        "instruction_quality": _instruction_quality(files, expected),
        "dependency_secret_safety": _dependency_secret_safety(result, files),
    }
