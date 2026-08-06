"""Public structural scorer; task-specific truth is supplied host-side."""

import re


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _groups_present(text, groups):
    normalized = " ".join(text.casefold().split())
    return all(
        any(" ".join(str(term).casefold().split()) in normalized for term in group)
        for group in groups
        if isinstance(group, list) and group
    )


def _statements(text):
    return [
        " ".join(value.casefold().split())
        for value in re.split(r"(?<=[.!?])\s+|\n+", text)
        if value.strip() and not value.lstrip().startswith("#")
    ]


def _sections(text):
    sections = []
    body = []
    for line in text.splitlines():
        if re.match(r"^#{1,4}\s+", line):
            if body:
                sections.append(" ".join("\n".join(body).casefold().split()))
                body = []
        else:
            body.append(line)
    if body:
        sections.append(" ".join("\n".join(body).casefold().split()))
    return [section for section in sections if section]


def _group_in_text(text, group):
    return any(
        " ".join(str(term).casefold().split()) in text
        for term in group
    )


def _groups_bound_to_statements(text, groups):
    statements = _statements(text)
    return bool(statements) and all(
        any(_group_in_text(statement, group) for statement in statements)
        for group in groups
        if isinstance(group, list) and group
    )


def _interface_contracts_present(text, groups):
    relevant = [group for group in groups if isinstance(group, list) and group]
    if not relevant:
        return True
    sections = _sections(text)
    minimum_bound = min(2, len(relevant))
    return _groups_present(text, relevant) and any(
        sum(_group_in_text(section, group) for group in relevant) >= minimum_bound
        for section in sections
    )


def _code_spans(text):
    return "\n".join(re.findall(r"`([^`\n]+)`", text))


def _mutate_unique_group(plan, expected, target_key):
    target_groups = expected.get(target_key, [])
    other_terms = {
        " ".join(str(term).casefold().split())
        for key in (
            "constraint_groups",
            "interface_groups",
            "decomposition_groups",
            "verification_groups",
        )
        if key != target_key
        for group in expected.get(key, [])
        if isinstance(group, list)
        for term in group
    }
    grounded = {
        " ".join(str(path).casefold().split())
        for path in expected.get("required_paths", [])
    }
    for group in target_groups:
        if not isinstance(group, list) or not group:
            continue
        terms = [str(term) for term in group]
        if any(
            " ".join(term.casefold().split()) in other_terms | grounded
            for term in terms
        ):
            continue
        mutant = plan
        changed = False
        for term in sorted(terms, key=len, reverse=True):
            mutant, count = re.subn(
                re.escape(term), "omitted reviewed requirement", mutant, flags=re.I
            )
            changed = changed or count > 0
        if changed:
            return mutant
    raise ValueError(f"reviewed plan cannot isolate {target_key}")


def zero_model_fixture(task, gold, expected):
    """Build a task-specific mutant for one named deterministic contract."""

    plan = _text(gold.get("plan")) if isinstance(gold, dict) else _text(gold)
    if not plan:
        raise ValueError("reviewed plan cannot produce a structured mutant")
    task_id = str(task.get("id") or "")
    if task_id == "sp-dev-credential-rotation":
        mutant_plan = _mutate_unique_group(plan, expected, "constraint_groups")
        targets = ["constraint_fidelity"]
    elif task_id == "sp-dev-evidence-destination":
        mutant_plan = _mutate_unique_group(plan, expected, "interface_groups")
        targets = ["interface_consistency"]
    elif task_id == "sp-dev-package-tree-qualification":
        mutant_plan = _mutate_unique_group(plan, expected, "verification_groups")
        targets = ["verification"]
    else:
        paths = [str(item) for item in expected.get("required_paths", [])]
        if not paths or paths[0] not in plan:
            raise ValueError("reviewed plan cannot produce the grounding mutant")
        mutant_plan = plan.replace(paths[0], "the relevant implementation module")
        targets = ["repository_grounding"]
    mutant = {**gold, "plan": mutant_plan} if isinstance(gold, dict) else mutant_plan
    return {
        "target_dimensions": targets,
        "mutant": mutant,
    }


def score(task, output, evidence):
    expected = evidence.get("expected")
    expected = expected if isinstance(expected, dict) else {}
    plan = _text(output.get("plan")) if isinstance(output, dict) else _text(output)
    encoded = plan.encode("utf-8")
    headings = re.findall(r"(?m)^#{1,4}\s+.+$", plan)
    required_paths = [str(item) for item in expected.get("required_paths", [])]
    forbidden = [str(item).casefold() for item in expected.get("forbidden", [])]
    inline_contracts = _code_spans(plan)
    return {
        "artifact_validity": bool(
            int(expected.get("min_characters", 300)) <= len(plan)
            and len(encoded) <= int(expected.get("max_bytes", 1_048_576))
        ),
        "constraint_fidelity": _groups_bound_to_statements(
            plan, expected.get("constraint_groups", [])
        ),
        "interface_consistency": _interface_contracts_present(
            plan, expected.get("interface_groups", [])
        ),
        "decomposition": len(headings) >= int(expected.get("minimum_headings", 2))
        and len(_sections(plan)) >= 2
        and _groups_present(plan, expected.get("decomposition_groups", [])),
        "repository_grounding": bool(required_paths)
        and all(path in inline_contracts for path in required_paths),
        "verification": bool(inline_contracts)
        and _groups_present(
            plan, expected.get("verification_groups", [])
        ),
        "scope_secret_safety": not any(term in plan.casefold() for term in forbidden)
        and not re.search(r"(?:api[_-]?key|token|password)\s*[:=]\s*[A-Za-z0-9_-]{12,}", plan),
    }
