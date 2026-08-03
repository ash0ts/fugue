"""Candidate-neutral checks for the Vercel React Skill upgrade fixtures."""


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _array(value):
    return value if isinstance(value, list) else []


def _strings(value):
    return [item for item in _array(value) if isinstance(item, str) and item]


def _suffix_present(paths, expected):
    normalized = [value.replace("\\", "/").rstrip("/") for value in paths]
    return any(value == expected or value.endswith("/" + expected) for value in normalized)


def _verification_passed(output, commands):
    observed = _array(output.get("verification"))
    for command in commands:
        if not any(
            isinstance(item, dict)
            and item.get("command") == command
            and item.get("exit_code") == 0
            for item in observed
        ):
            return False
    return True


def score(task, output, evidence):
    del task
    expected = _mapping(evidence.get("expected"))
    result = _mapping(output)
    facts = _mapping(result.get("facts"))
    changed = _strings(evidence.get("changed_paths"))
    inspected = _strings(evidence.get("inspected_paths"))
    required_changed = _strings(expected.get("required_changed_paths"))
    allowed_changed = _strings(expected.get("allowed_changed_paths"))
    required_inspected = _strings(expected.get("required_inspected_paths"))
    required_facts = _strings(expected.get("required_facts"))
    preservation_facts = _strings(expected.get("preservation_facts"))
    commands = _strings(expected.get("verification_commands"))

    artifact_validity = bool(
        result.get("status") == "completed"
        and isinstance(result.get("summary"), str)
        and 20 <= len(result["summary"]) <= 2000
        and isinstance(result.get("limitations"), list)
        and isinstance(result.get("changed_paths"), list)
        and isinstance(result.get("inspected_paths"), list)
        and isinstance(result.get("verification"), list)
        and facts
    )
    requested_change = bool(
        required_facts and all(facts.get(name) is True for name in required_facts)
    )
    behavior_preservation = bool(
        preservation_facts
        and all(facts.get(name) is True for name in preservation_facts)
    )
    repository_grounding = bool(
        changed
        and inspected
        and required_changed
        and all(_suffix_present(changed, path) for path in required_changed)
        and all(_suffix_present(inspected, path) for path in required_inspected)
    )
    scope_safety = bool(
        changed
        and allowed_changed
        and all(
            any(
                value.replace("\\", "/").rstrip("/") == allowed
                or value.replace("\\", "/").rstrip("/").endswith("/" + allowed)
                for allowed in allowed_changed
            )
            for value in changed
        )
        and facts.get("dependencies_added") is False
    )
    skill_mechanism_used = any(
        "react-best-practices" in value.replace("\\", "/")
        for value in inspected
    )
    return {
        "artifact_validity": artifact_validity,
        "requested_change": requested_change,
        "repository_grounding": repository_grounding,
        "behavior_preservation": behavior_preservation,
        "verification": bool(commands and _verification_passed(result, commands)),
        "scope_safety": scope_safety,
        "skill_mechanism_used": skill_mechanism_used,
    }
