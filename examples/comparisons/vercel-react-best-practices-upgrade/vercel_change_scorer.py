"""Candidate-neutral checks for the Vercel React Skill upgrade fixtures."""

import re

_DIMENSIONS = dict.fromkeys(
    (
        "artifact_validity",
        "requested_change",
        "repository_grounding",
        "behavior_preservation",
        "verification",
        "scope_safety",
    ),
    False,
)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _array(value):
    return value if isinstance(value, list) else []


def _strings(value):
    return [item for item in _array(value) if isinstance(item, str) and item]


def _text(value):
    return value if isinstance(value, str) else ""


def _suffix_present(paths, expected):
    normalized = [value.replace("\\", "/").rstrip("/") for value in paths]
    return any(
        value == expected or value.endswith("/" + expected) for value in normalized
    )


def _is_safe_relative_path(value):
    return bool(
        isinstance(value, str)
        and value
        and not value.startswith(("/", "\\"))
        and ".." not in value.replace("\\", "/").split("/")
    )


def _artifact_validity(task, result):
    files = _mapping(result.get("files"))
    inspected_paths = result.get("inspected_paths")
    verification = _array(result.get("verification"))
    limitations = result.get("limitations")
    receipt = _mapping(verification[0]) if len(verification) == 1 else {}
    return bool(
        set(result)
        == {
            "schema_version",
            "task_id",
            "status",
            "files",
            "inspected_paths",
            "verification",
            "summary",
            "limitations",
        }
        and result.get("schema_version") == 1
        and result.get("task_id") == task.get("id")
        and result.get("status") == "completed"
        and files
        and all(
            _is_safe_relative_path(path) and isinstance(content, str) and content
            for path, content in files.items()
        )
        and isinstance(inspected_paths, list)
        and inspected_paths
        and all(_is_safe_relative_path(path) for path in inspected_paths)
        and len(verification) == 1
        and set(receipt) == {"command", "exit_code", "stdout"}
        and isinstance(receipt.get("command"), str)
        and type(receipt.get("exit_code")) is int
        and isinstance(receipt.get("stdout"), str)
        and isinstance(result.get("summary"), str)
        and 20 <= len(result["summary"].strip()) <= 2000
        and isinstance(limitations, list)
        and all(isinstance(item, str) and item.strip() for item in limitations)
    )


def _server_action_requested(source):
    match = re.search(
        r"const\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
        r"await\s+db\.membership\.findFirst\s*\(",
        source,
    )
    if match is None:
        return False
    auth = source.find("await auth()")
    authorization = match.start()
    mutation = source.find("db.team.update", match.end())
    between = source[match.end() : mutation if mutation >= 0 else len(source)]
    guard = re.search(
        rf"if\s*\(\s*!\s*{re.escape(match.group('name'))}\s*\)\s*\{{?"
        r"[\s\S]*?throw\s+new\s+Error",
        between,
    )
    query = source[authorization : mutation if mutation >= 0 else len(source)]
    return bool(
        0 <= auth < authorization < mutation
        and "teamId" in query
        and "session.user.id" in query
        and "OWNER" in query
        and "ADMIN" in query
        and guard
    )


def _server_action_preserved(source):
    return bool(
        "String(input?.teamId || '').trim()" in source
        and "String(input?.name || '').trim()" in source
        and "name.length < 2" in source
        and "name.length > 80" in source
        and re.search(r"where:\s*\{\s*id:\s*teamId\s*\}", source)
        and re.search(r"data:\s*\{\s*name\s*\}", source)
    )


def _rsc_requested(page, client):
    return bool(
        re.search(r"<ProjectsClient\s+projects=\{projects\}\s*/>", page)
        and "projectNames=" not in page
        and not re.search(r"const\s+projectNames\s*=", page)
        and re.search(r"ProjectsClient\(\{\s*projects\s*\}\)", client)
        and re.search(
            r"projects\.map\(\(project\)\s*=>\s*project\.name\)",
            client,
        )
    )


def _rsc_preserved(client):
    return bool(
        'aria-labelledby="projects-title"' in client
        and re.search(r"<h1\s+id=\"projects-title\">Projects</h1>", client)
        and "<ul>" in client
        and "key={project.id}" in client
        and "{project.name}" in client
    )


def _source_outcomes(task_id, files):
    if task_id == "server-action-authorization":
        source = _text(files.get("app/actions.mjs"))
        return _server_action_requested(source), _server_action_preserved(source)
    if task_id == "rsc-serialization-boundary":
        page = _text(files.get("app/projects/page.jsx"))
        client = _text(files.get("app/projects/projects-client.jsx"))
        return _rsc_requested(page, client), _rsc_preserved(client)
    return False, False


def _verification_passed(result, expected):
    observed = _array(result.get("verification"))
    if len(observed) != 1:
        return False
    receipt = _mapping(observed[0])
    stdout = _text(receipt.get("stdout"))
    test_names = _strings(expected.get("verification_test_names"))
    return bool(
        set(receipt) == {"command", "exit_code", "stdout"}
        and receipt.get("command") == "node --test"
        and receipt.get("exit_code") == 0
        and test_names
        and all(name in stdout for name in test_names)
        and re.search(r"(?:#\s*)?pass(?:ed)?\D+3\b|\b3\s+pass", stdout, re.I)
        and re.search(r"(?:#\s*)?fail(?:ed)?\D+0\b|\b0\s+fail", stdout, re.I)
    )


def score(task, output, evidence):
    task = _mapping(task)
    expected = _mapping(evidence.get("expected"))
    result = _mapping(output)
    files = _mapping(result.get("files"))
    file_paths = _strings(expected.get("required_file_paths"))
    allowed_paths = set(_strings(expected.get("allowed_file_paths")))
    inspected = _strings(result.get("inspected_paths"))
    required_inspected = _strings(expected.get("required_inspected_paths"))

    if not result or not expected:
        return dict(_DIMENSIONS)

    requested_change, behavior_preservation = _source_outcomes(
        _text(task.get("id")), files
    )
    repository_grounding = bool(
        file_paths
        and set(files) == set(file_paths)
        and required_inspected
        and inspected
        and all(_is_safe_relative_path(path) for path in inspected)
        and all(_suffix_present(inspected, path) for path in required_inspected)
    )
    scope_safety = bool(
        allowed_paths
        and set(files) == set(file_paths)
        and set(files) <= allowed_paths
        and all(
            not path.startswith("/")
            and ".." not in path.split("/")
            and path not in {"package.json", "package-lock.json"}
            for path in files
        )
    )
    return {
        "artifact_validity": _artifact_validity(task, result),
        "requested_change": requested_change,
        "repository_grounding": repository_grounding,
        "behavior_preservation": behavior_preservation,
        "verification": _verification_passed(result, expected),
        "scope_safety": scope_safety,
    }
