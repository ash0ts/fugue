"""Deterministic scorer for source-grounded writing-plans upgrade attempts."""

_DIMENSIONS = dict.fromkeys(
    (
        "plan_structure",
        "constraint_fidelity",
        "repository_grounding",
        "interface_contracts",
        "reviewable_decomposition",
        "verification_quality",
        "scope_and_secret_safety",
        "historical_diff_coverage",
    ),
    False,
)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _array(value):
    return value if isinstance(value, list) else []


def _output_text(output):
    return output.strip() if isinstance(output, str) else ""


def _task_blocks(text):
    lines = text.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if line.strip().lower().startswith("### task ")
    ]
    blocks = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        blocks.append("\n".join(lines[start:end]).strip())
    return blocks


def _inline_code(text):
    values = []
    for line in text.splitlines():
        parts = line.split("`")
        values.extend(part.strip() for part in parts[1::2] if part.strip())
    return values


def _source_paths(text):
    roots = ("fugue/", "tests/", "examples/", ".github/", "compose.")
    values = []
    for token in _inline_code(text):
        candidate = token.split(":", 1)[0].strip()
        if candidate.startswith(roots):
            values.append(candidate)
    return list(dict.fromkeys(values))


def _task_titles(blocks):
    return [block.splitlines()[0].strip().lower() for block in blocks]


def _plan_structure(text, blocks):
    lower = text.lower()
    placeholders = (
        "tbd",
        "todo",
        "implement later",
        "fill in details",
        "similar to task",
        "add appropriate error handling",
    )
    return bool(
        text.startswith("# ")
        and "**goal:**" in lower
        and "**architecture:**" in lower
        and "**tech stack:**" in lower
        and blocks
        and len(text) <= 14_000
        and not any(value in lower for value in placeholders)
    )


def _constraint_fidelity(text, expected):
    lower = text.lower()
    constraints = [
        str(value).strip().lower()
        for value in _array(expected.get("constraints"))
        if str(value).strip()
    ]
    return bool(constraints and all(value in lower for value in constraints))


def _repository_grounding(text, expected):
    observed = set(_source_paths(text))
    allowed = set(str(value) for value in _array(expected.get("allowed_paths")))
    required = set(str(value) for value in _array(expected.get("required_paths")))
    if not observed or not allowed or not required or not observed <= allowed:
        return False
    covered = len(observed & required) / len(required)
    minimum = expected.get("minimum_path_fraction", 1)
    symbols = [
        str(value).lower()
        for value in _array(expected.get("required_symbols"))
    ]
    minimum_symbols = int(expected.get("minimum_symbols", len(symbols)))
    return bool(
        isinstance(minimum, (int, float))
        and covered >= float(minimum)
        and sum(symbol in text.lower() for symbol in symbols) >= minimum_symbols
    )


def _interface_contracts(blocks, expected):
    if not blocks:
        return False
    interface_lines = []
    cross_task_references = 0
    for block in blocks:
        lower = block.lower()
        if (
            "**interfaces:**" not in lower
            or "consumes:" not in lower
            or "produces:" not in lower
        ):
            return False
        selected = [
            line.strip()
            for line in block.splitlines()
            if "consumes:" in line.lower() or "produces:" in line.lower()
        ]
        if not selected or not all(
            any(token in line for token in ("(", ":", "->"))
            for line in selected
        ):
            return False
        interface_lines.extend(selected)
        cross_task_references += sum(
            "task " in line.lower() for line in selected
        )
    interface_text = "\n".join(interface_lines).lower()
    symbols = [
        str(value).lower()
        for value in _array(expected.get("required_interface_tokens"))
    ]
    minimum = int(expected.get("minimum_interface_tokens", len(symbols)))
    if sum(symbol in interface_text for symbol in symbols) < minimum:
        return False
    return len(blocks) == 1 or cross_task_references >= len(blocks) - 1


def _reviewable_decomposition(blocks, expected):
    minimum = int(expected.get("task_count_min", 1))
    maximum = int(expected.get("task_count_max", 1))
    if not minimum <= len(blocks) <= maximum:
        return False
    forbidden = {
        str(value).lower()
        for value in _array(expected.get("forbidden_standalone_titles"))
    }
    for title, block in zip(_task_titles(blocks), blocks, strict=True):
        lower = block.lower()
        title_words = {
            word.strip("-:,.()[]")
            for word in title.split()
            if word.strip("-:,.()[]")
        }
        if (
            title_words & forbidden
            or "**files:**" not in lower
            or "- [ ]" not in lower
            or "test" not in lower
            or "run" not in lower
            or "expected" not in lower
            or "commit" not in lower
        ):
            return False
    return True


def _verification_quality(text, blocks, expected):
    lower = text.lower()
    commands = [
        str(value).lower()
        for value in _array(expected.get("required_command_tokens"))
    ]
    cycles = all(
        (
            "failing test" in block.lower()
            or "verify it fails" in block.lower()
            or "expected: fail" in block.lower()
        )
        and (
            "verify it passes" in block.lower()
            or "expected: pass" in block.lower()
            or "make sure it passes" in block.lower()
        )
        for block in blocks
    )
    return bool(commands and all(value in lower for value in commands) and cycles)


def _scope_and_secret_safety(text, expected):
    lower = text.lower()
    unsafe = (
        "pip install",
        "uv add",
        "npm install",
        "curl ",
        "wget ",
        "rm -rf",
        "git push",
        "gh pr create",
    )
    allowed = set(str(value) for value in _array(expected.get("allowed_paths")))
    paths = set(_source_paths(text))
    secret_assignments = (
        "wandb_api_key=\"",
        "wandb_api_key='",
        "anthropic_api_key=\"",
        "anthropic_api_key='",
    )
    creates = [
        line
        for line in text.splitlines()
        if line.strip().lower().startswith("- create:")
        and "none" not in line.lower()
    ]
    return bool(
        not any(value in lower for value in unsafe)
        and not any(value in lower for value in secret_assignments)
        and not creates
        and paths <= allowed
    )


def _historical_diff_coverage(blocks, expected):
    task_text = "\n".join(blocks).lower()
    groups = _array(expected.get("coverage_groups"))
    covered = sum(
        any(str(term).lower() in task_text for term in _array(group))
        for group in groups
    )
    return bool(groups and covered >= int(expected.get("minimum_coverage_groups", 0)))


def score(task, output, evidence):
    expected = _mapping(evidence.get("expected"))
    text = _output_text(output)
    blocks = _task_blocks(text)
    if not expected or not text:
        return dict(_DIMENSIONS)
    return {
        "plan_structure": _plan_structure(text, blocks),
        "constraint_fidelity": _constraint_fidelity(text, expected),
        "repository_grounding": _repository_grounding(text, expected),
        "interface_contracts": _interface_contracts(blocks, expected),
        "reviewable_decomposition": _reviewable_decomposition(blocks, expected),
        "verification_quality": _verification_quality(text, blocks, expected),
        "scope_and_secret_safety": _scope_and_secret_safety(text, expected),
        "historical_diff_coverage": _historical_diff_coverage(blocks, expected),
    }
