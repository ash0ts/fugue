"""Public code-change structure scorer; a separate host verifier runs tests."""

import hashlib
import json
import re


def _groups_present(text, groups):
    normalized = " ".join(str(text).casefold().split())
    return all(
        any(" ".join(str(term).casefold().split()) in normalized for term in group)
        for group in groups
        if isinstance(group, list) and group
    )


def _stable_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _files_digest(files):
    return _stable_digest(
        {
            str(path): hashlib.sha256(content.encode()).hexdigest()
            for path, content in files.items()
            if isinstance(content, str)
        }
    )


def _verified_host_result(task, files, expected, evidence):
    """Accept only a controller-produced receipt bound to this exact output."""

    receipt = evidence.get("host_verifier")
    if not isinstance(receipt, dict):
        return False
    allowed = {
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
    return bool(
        set(receipt) == allowed
        and receipt.get("schema_version") == 1
        and receipt.get("verifier_id") == "fugue-node-test-v1"
        and receipt.get("task_id") == task.get("id")
        and receipt.get("task_archive_sha256")
        == expected.get("task_archive_sha256")
        and isinstance(receipt.get("agent_output_sha256"), str)
        and len(receipt["agent_output_sha256"]) == 64
        and receipt.get("output_files_sha256") == _files_digest(files)
        and receipt.get("allowed_paths_digest")
        == _stable_digest(sorted(str(path) for path in expected.get("allowed_paths", [])))
        and receipt.get("runtime_lock_digest")
        == expected.get("verifier_runtime_lock_digest")
        and receipt.get("command") == ["node", "--test"]
        and receipt.get("status") == "passed"
        and receipt.get("exit_code") == 0
        and receipt.get("receipt_digest") == _stable_digest(unsigned)
    )


def _response_contract_valid(result, expected):
    response_keys = {
        str(item)
        for item in expected.get(
            "response_keys", ["schema_version", "task_id", "files", "summary"]
        )
    }
    if set(result) != response_keys:
        return False
    if "status" not in response_keys:
        return True
    inspected = result.get("inspected_paths")
    verification = result.get("verification")
    limitations = result.get("limitations")
    return bool(
        result.get("status") == "completed"
        and isinstance(inspected, list)
        and inspected
        and all(isinstance(path, str) and path.strip() for path in inspected)
        and isinstance(verification, list)
        and len(verification) == 1
        and isinstance(verification[0], dict)
        and set(verification[0]) == {"command", "exit_code", "stdout"}
        and verification[0].get("command") == "node --test"
        and type(verification[0].get("exit_code")) is int
        and isinstance(verification[0].get("stdout"), str)
        and isinstance(limitations, list)
        and all(isinstance(item, str) for item in limitations)
    )


def zero_model_fixture(task, gold, expected):
    """Build a schema-valid mutant caught only by the pinned task verifier."""

    result = dict(gold) if isinstance(gold, dict) else {}
    files = dict(result.get("files") or {})
    required = [str(item) for item in expected.get("required_paths", [])]
    if not required or required[0] not in files:
        raise ValueError("reviewed code change cannot produce the verifier mutant")
    files[required[0]] = (
        files[required[0]]
        + "\nthrow new Error('fugue zero-model verifier mutant');\n"
    )
    result["files"] = files
    return {
        "target_dimensions": ["behavior_preservation", "verification_passed"],
        "mutant": result,
    }


def score(task, output, evidence):
    expected = evidence.get("expected")
    expected = expected if isinstance(expected, dict) else {}
    result = output if isinstance(output, dict) else {}
    files = result.get("files") if isinstance(result.get("files"), dict) else {}
    required_paths = {str(item) for item in expected.get("required_paths", [])}
    allowed_paths = {str(item) for item in expected.get("allowed_paths", [])}
    combined = "\n".join(str(value) for value in files.values())
    changed = {str(item) for item in evidence.get("changed_paths", [])}
    preserved_files_ok = all(
        isinstance(files.get(path), str)
        and hashlib.sha256(files[path].encode()).hexdigest() == digest
        for path, digest in expected.get("preserved_sha256", {}).items()
    )
    forbidden = [str(item).casefold() for item in expected.get("forbidden", [])]
    verified = _verified_host_result(task, files, expected, evidence)
    return {
        "artifact_validity": bool(
            _response_contract_valid(result, expected)
            and result.get("schema_version") == 1
            and result.get("task_id") == task.get("id")
            and files
            and all(isinstance(path, str) and isinstance(value, str) for path, value in files.items())
            and isinstance(result.get("summary"), str)
            and bool(result["summary"].strip())
        ),
        "requested_behavior": _groups_present(
            combined, expected.get("behavior_groups", [])
        ),
        "repository_grounding": required_paths <= set(files)
        and (not changed or required_paths <= changed),
        # The pinned task tests jointly exercise the requested repair and the
        # declared preserved public behavior.  Empty optional file-hash sets
        # therefore cannot make preservation pass without the output-bound
        # trusted verifier receipt.
        "behavior_preservation": verified and preserved_files_ok,
        "verification_passed": verified,
        "scope_safety": (not allowed_paths or set(files) <= allowed_paths)
        and not any(term in combined.casefold() for term in forbidden)
        and not re.search(r"(?:api[_-]?key|token|password)\s*[:=]\s*[A-Za-z0-9_-]{12,}", combined),
    }
