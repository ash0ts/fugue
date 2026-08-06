"""Public Skill-package scorer; private expectations stay host-only."""

import hashlib
import json
import re


def _frontmatter(text):
    if not isinstance(text, str) or not text.startswith("---\n"):
        return {}, ""
    marker = text.find("\n---\n", 4)
    if marker < 0:
        return {}, ""
    values = {}
    for line in text[4:marker].splitlines():
        if not line or line[:1].isspace() or ":" not in line:
            return {}, ""
        key, value = line.split(":", 1)
        if key.strip() in values:
            return {}, ""
        values[key.strip()] = value.strip().strip("\"'")
    return values, text[marker + 5 :]


def _groups_present(text, groups):
    normalized = " ".join(str(text).casefold().split())
    return all(
        any(" ".join(str(term).casefold().split()) in normalized for term in group)
        for group in groups
        if isinstance(group, list) and group
    )


def _render_frontmatter(metadata, body):
    lines = ["---", *(f"{key}: {value}" for key, value in metadata.items()), "---"]
    return "\n".join(lines) + "\n" + body


def _stable_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _files_digest(files):
    return _stable_digest(
        {
            str(path): hashlib.sha256(content.encode()).hexdigest()
            for path, content in files.items()
            if isinstance(path, str) and isinstance(content, str)
        }
    )


def _verified_package_result(task, files, expected, evidence):
    """Accept only the controller-reserved receipt for this exact output."""

    receipt = evidence.get("host_verifier")
    if not isinstance(receipt, dict):
        return False
    required = {
        "schema_version",
        "verifier_id",
        "task_id",
        "task_archive_sha256",
        "agent_output_sha256",
        "output_files_sha256",
        "allowed_paths_digest",
        "runtime_lock_digest",
        "observed_node_version",
        "command",
        "status",
        "exit_code",
        "stdout_sha256",
        "stderr_sha256",
        "receipt_digest",
    }
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    allowed_paths = sorted(str(path) for path in expected.get("allowed_paths", []))
    return bool(
        set(receipt) == required
        and receipt.get("schema_version") == 1
        and receipt.get("verifier_id") == "fugue-skill-package-validator-v1"
        and receipt.get("task_id") == task.get("id")
        and receipt.get("output_files_sha256") == _files_digest(files)
        and receipt.get("allowed_paths_digest") == _stable_digest(allowed_paths)
        and receipt.get("command") == ["node", "skill-package-validate"]
        and receipt.get("status") == "passed"
        and receipt.get("exit_code") == 0
        and receipt.get("receipt_digest") == _stable_digest(unsigned)
    )


def zero_model_fixture(task, gold, expected):
    """Build one task-specific package, compatibility, or preservation mutant."""

    result = dict(gold) if isinstance(gold, dict) else {}
    files = dict(result.get("files") or {})
    skill_path = str(expected.get("skill_path", "SKILL.md"))
    metadata, body = _frontmatter(files.get(skill_path))
    if not metadata or not body:
        raise ValueError("reviewed Skill cannot produce the targeted mutant")
    task_id = str(task.get("id") or "")
    preserved = dict(expected.get("preserved_frontmatter", {}))
    if task_id == "as-dev-create-platform-bound-skill":
        result["unexpected"] = "host-package-contract-mutant"
        return {"target_dimensions": ["artifact_validity"], "mutant": result}
    if "preserve" in task_id and preserved:
        key = "license" if "license" in preserved else next(iter(preserved))
        metadata[key] = f"changed-{metadata.get(key, 'value')}"
        targets = ["behavior_preservation"]
    elif str(expected.get("compatibility_policy")) == "absent":
        metadata["compatibility"] = "Invented platform requirement"
        targets = ["compatibility_selection"]
    else:
        metadata.pop("compatibility", None)
        targets = ["compatibility_selection"]
    files[skill_path] = _render_frontmatter(metadata, body)
    result["files"] = files
    return {"target_dimensions": targets, "mutant": result}


def score(task, output, evidence):
    expected = evidence.get("expected")
    expected = expected if isinstance(expected, dict) else {}
    result = output if isinstance(output, dict) else {}
    files = result.get("files") if isinstance(result.get("files"), dict) else {}
    skill_path = str(expected.get("skill_path", "SKILL.md"))
    metadata, body = _frontmatter(files.get(skill_path))
    allowed_paths = {str(item) for item in expected.get("allowed_paths", [])}
    expected_name = str(expected.get("expected_name", ""))
    name_min_length = int(expected.get("name_min_length", 1))
    name_max_length = int(expected.get("name_max_length", 64))
    policy = str(expected.get("compatibility_policy", "required"))
    compatibility = metadata.get("compatibility")
    if policy == "absent":
        compatibility_ok = compatibility is None
    elif policy == "exact":
        compatibility_ok = compatibility == expected.get("compatibility_value")
    else:
        compatibility_ok = isinstance(compatibility, str) and _groups_present(
            compatibility, expected.get("compatibility_groups", [])
        )
    expected_body_digest = expected.get("preserved_body_sha256")
    body_preserved = expected_body_digest is None or (
        isinstance(expected_body_digest, str)
        and hashlib.sha256(body.encode()).hexdigest() == expected_body_digest
    )
    preservation_ok = (
        body_preserved
        and all(
            isinstance(files.get(path), str)
            and hashlib.sha256(files[path].encode()).hexdigest() == digest
            for path, digest in expected.get("preserved_sha256", {}).items()
        )
        and all(
            metadata.get(str(key)) == value
            for key, value in expected.get("preserved_frontmatter", {}).items()
        )
    )
    combined = "\n".join(str(value) for value in files.values())
    forbidden = [str(item).casefold() for item in expected.get("forbidden", [])]
    valid_name = bool(
        isinstance(metadata.get("name"), str)
        and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", metadata["name"])
        and name_min_length <= len(metadata["name"]) <= name_max_length
    )
    response_keys = {
        str(item)
        for item in expected.get(
            "response_keys", ["schema_version", "task_id", "files", "summary"]
        )
    }
    package_valid = _verified_package_result(
        task, files, expected, evidence
    )
    return {
        "artifact_validity": bool(
            package_valid
            and set(result) == response_keys
            and result.get("schema_version") == 1
            and result.get("task_id") == task.get("id")
            and files
            and all(isinstance(path, str) and isinstance(value, str) for path, value in files.items())
            and isinstance(result.get("summary"), str)
            and bool(result["summary"].strip())
            and (not allowed_paths or set(files) <= allowed_paths)
        ),
        "frontmatter_semantics": valid_name
        and bool(metadata.get("description"))
        and bool(body.strip()),
        "compatibility_selection": compatibility_ok,
        "name_help_consistency": bool(expected_name)
        and metadata.get("name") == expected_name
        and expected_name in combined,
        "instruction_quality": _groups_present(
            body, expected.get("instruction_groups", [])
        ),
        "behavior_preservation": preservation_ok,
        "dependency_secret_safety": not any(
            term in combined.casefold() for term in forbidden
        )
        and not re.search(
            r"(?:api[_-]?key|token|password)\s*[:=]\s*[A-Za-z0-9_-]{12,}", combined
        ),
    }
