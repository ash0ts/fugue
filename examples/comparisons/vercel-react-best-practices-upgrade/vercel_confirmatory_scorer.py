"""Host-side structural and semantic checks for the Vercel confirmation."""

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


def _safe_path(value):
    return bool(
        isinstance(value, str)
        and value
        and not value.startswith(("/", "\\"))
        and ".." not in value.replace("\\", "/").split("/")
    )


def _suffix_present(paths, expected):
    normalized = [value.replace("\\", "/").rstrip("/") for value in paths]
    return any(
        value == expected or value.endswith("/" + expected) for value in normalized
    )


def _artifact_validity(task, result):
    files = _mapping(result.get("files"))
    inspected = result.get("inspected_paths")
    verification = _array(result.get("verification"))
    receipt = _mapping(verification[0]) if len(verification) == 1 else {}
    limitations = result.get("limitations")
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
        and all(_safe_path(path) and isinstance(body, str) and body for path, body in files.items())
        and isinstance(inspected, list)
        and inspected
        and all(_safe_path(path) for path in inspected)
        and len(verification) == 1
        and set(receipt) == {"command", "exit_code", "stdout"}
        and receipt.get("command") == "node --test"
        and type(receipt.get("exit_code")) is int
        and isinstance(receipt.get("stdout"), str)
        and isinstance(result.get("summary"), str)
        and 20 <= len(result["summary"].strip()) <= 2000
        and isinstance(limitations, list)
        and all(isinstance(item, str) and item.strip() for item in limitations)
    )


def _return_keys(source):
    match = re.search(r"return\s*\{([^{}]*)\}\s*;", source)
    if match is None:
        return []
    keys = []
    for item in match.group(1).split(","):
        value = item.strip().split(":", 1)[0].strip()
        if re.fullmatch(r"[A-Za-z_$][\w$]*", value):
            keys.append(value)
    return keys


def _server_action(files, verifier):
    source = _text(files.get(verifier.get("source_path")))
    export_name = _text(verifier.get("export_name"))
    if not source or f"export async function {export_name}" not in source:
        return False, False
    mode = verifier.get("mode")
    if mode == "read_only_control":
        read_call = _text(verifier.get("read_call"))
        requested = bool(
            read_call in source
            and all(term not in source for term in _strings(verifier.get("forbidden_calls")))
            and all(term in source for term in _strings(verifier.get("validation_terms")))
        )
        return requested, bool(read_call in source and "return" in source)
    if mode != "authorize_mutation":
        return False, False
    auth = source.find(_text(verifier.get("auth_call")))
    authorization = source.find(_text(verifier.get("authorization_call")))
    mutation = source.find(_text(verifier.get("mutation_call")))
    validation_terms = _strings(verifier.get("validation_terms"))
    auth_guard = source.find("throw", auth + 1, authorization if authorization >= 0 else len(source))
    authorization_guard = source.find("throw", authorization + 1, mutation if mutation >= 0 else len(source))
    requested = bool(
        0 <= auth < authorization < mutation
        and auth_guard >= 0
        and authorization_guard >= 0
        and all(term in source for term in _strings(verifier.get("authorization_terms")))
        and all(term in source for term in validation_terms)
        and source.count(_text(verifier.get("mutation_call"))) == 1
    )
    if verifier.get("validation_before_auth") and validation_terms:
        requested = requested and max(source.find(term) for term in validation_terms) < auth
    preserved = bool(
        "'use server'" in source
        and "return" in source
        and not re.search(r"(?:from\s+|require\s*\()[\"'](?:zod|lodash|axios)", source)
    )
    return requested, preserved


def _rsc(files, verifier):
    server = _text(files.get(verifier.get("server_path")))
    client = _text(files.get(verifier.get("client_path")))
    mode = verifier.get("mode")
    if not server or not client or "export function" not in server or "export function" not in client:
        return False, False
    if mode == "canonical_only":
        canonical = _text(verifier.get("canonical_prop"))
        derived = _strings(verifier.get("derived_props"))
        requested = bool(
            _return_keys(server) == [canonical]
            and all(term not in server for term in derived)
            and all(term in client for term in _strings(verifier.get("client_terms")))
        )
        preserved = bool(canonical in client and "return" in client)
        return requested, preserved
    if mode == "derived_control":
        canonical = _text(verifier.get("canonical_prop"))
        requested = bool(
            _return_keys(server) == [canonical]
            and all(term in server for term in _strings(verifier.get("server_terms")))
            and all(term in client for term in _strings(verifier.get("client_terms")))
            and all(term not in _return_keys(server) for term in _strings(verifier.get("forbidden_props")))
        )
        return requested, bool(canonical in client and "return" in client)
    return False, False


def _dom(files, verifier):
    source = _text(files.get(verifier.get("source_path")))
    kind = verifier.get("kind")
    writes = _strings(verifier.get("write_terms"))
    reads = _strings(verifier.get("read_terms"))
    forbidden = _strings(verifier.get("forbidden_read_terms"))
    if kind == "dom_write_control":
        requested = bool(all(term in source for term in writes) and all(term not in source for term in forbidden))
        return requested, bool("export function" in source)
    write_positions = [source.find(term) for term in writes]
    read_positions = [source.find(term) for term in reads]
    requested = bool(
        writes
        and reads
        and min(write_positions) >= 0
        and min(read_positions) >= 0
        and max(write_positions) < min(read_positions)
        and all(term not in source for term in forbidden)
    )
    return requested, bool("export function" in source and "return" in source)


def _array_contract(files, verifier):
    source = _text(files.get(verifier.get("source_path")))
    kind = verifier.get("kind")
    if kind == "array_sum_control":
        requested = bool(
            ".reduce(" in source
            and re.search(r"\.reduce\([\s\S]*,\s*0\s*\)", source)
            and "Math.max" not in source
            and "Math.min" not in source
        )
        return requested, bool("export function" in source and "return" in source)
    mode = verifier.get("mode")
    comparison = ">" if mode == "max" else "<"
    unsafe = "Math.max(..." if mode == "max" else "Math.min(..."
    requested = bool(
        "values.length === 0" in source
        and "return null" in source
        and unsafe not in source
        and ".sort(" not in source
        and ("for (" in source or ".reduce(" in source)
        and comparison in source
    )
    return requested, bool("export function" in source and "return" in source)


def _hook(files, verifier):
    source = _text(files.get(verifier.get("source_path")))
    effect = _text(verifier.get("effect"))
    requested = bool(
        f"hooks.{effect}(" in source
        and "ref.current = value" in source
        and "hooks.useRef(value)" in source
        and (not verifier.get("forbid_layout") or "useLayoutEffect" not in source)
    )
    return requested, bool("export function" in source and "return ref" in source)


def _event(files, verifier):
    source = _text(files.get(verifier.get("source_path")))
    event_type = _text(verifier.get("event_type"))
    prop = _text(verifier.get("property"))
    requested = bool(
        f"@param {{{event_type}}}" in source
        and re.search(rf"\bevent\.{re.escape(prop)}\b", source)
    )
    return requested, bool("export function" in source and "return" in source)


def _semantic(files, expected):
    verifier = _mapping(expected.get("verifier"))
    kind = verifier.get("kind")
    if kind == "server_action":
        return _server_action(files, verifier)
    if kind == "rsc_props":
        return _rsc(files, verifier)
    if kind in {"dom_batch", "dom_write_control"}:
        return _dom(files, verifier)
    if kind in {"array_extreme", "array_sum_control"}:
        return _array_contract(files, verifier)
    if kind == "hook_timing":
        return _hook(files, verifier)
    if kind == "event_signature":
        return _event(files, verifier)
    return False, False


def _verification(result, expected, semantic_pass):
    receipts = _array(result.get("verification"))
    receipt = _mapping(receipts[0]) if len(receipts) == 1 else {}
    stdout = _text(receipt.get("stdout"))
    name = _text(expected.get("public_test_name"))
    return bool(
        semantic_pass
        and receipt.get("command") == "node --test"
        and receipt.get("exit_code") == 0
        and name
        and name in stdout
        and re.search(r"(?:#\s*)?fail(?:ed)?\D+0\b|\b0\s+fail", stdout, re.I)
        and re.search(r"(?:#\s*)?pass(?:ed)?\D+[1-9]\d*\b|\b[1-9]\d*\s+pass", stdout, re.I)
    )


def score(task, output, evidence):
    task = _mapping(task)
    result = _mapping(output)
    expected = _mapping(evidence.get("expected"))
    files = _mapping(result.get("files"))
    required_files = _strings(expected.get("required_file_paths"))
    allowed_files = set(_strings(expected.get("allowed_file_paths")))
    required_inspected = _strings(expected.get("required_inspected_paths"))
    inspected = _strings(result.get("inspected_paths"))
    if not task or not result or not expected:
        return dict(_DIMENSIONS)
    requested, preserved = _semantic(files, expected)
    grounding = bool(
        required_files
        and set(files) == set(required_files)
        and required_inspected
        and all(_suffix_present(inspected, path) for path in required_inspected)
    )
    scope = bool(
        allowed_files
        and set(files) == set(required_files)
        and set(files) <= allowed_files
        and all(_safe_path(path) and path not in {"package.json", "package-lock.json"} for path in files)
    )
    return {
        "artifact_validity": _artifact_validity(task, result),
        "requested_change": requested,
        "repository_grounding": grounding,
        "behavior_preservation": preserved,
        "verification": _verification(result, expected, requested and preserved),
        "scope_safety": scope,
    }
