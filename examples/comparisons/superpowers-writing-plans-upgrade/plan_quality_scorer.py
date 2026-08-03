"""Candidate-neutral deterministic checks for source-grounded plans."""

import re

_DIMENSIONS = dict.fromkeys(
    (
        "artifact_validity",
        "requirement_coverage",
        "repository_grounding",
        "dependency_contracts",
        "reviewable_decomposition",
        "verification_quality",
        "scope_and_secret_safety",
        "scenario_coverage",
    ),
    False,
)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _array(value):
    return value if isinstance(value, list) else []


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _normalized(value):
    return " ".join(str(value).lower().split())


def _contains_group(text, group):
    normalized = _normalized(text)
    observed = re.findall(r"[a-z0-9_]+", normalized)
    ignored = {"a", "an", "and", "is", "of", "or", "the", "to", "with"}
    for term in _array(group):
        phrase = _normalized(term)
        if phrase in normalized:
            return True
        required = [
            value
            for value in re.findall(r"[a-z0-9_]+", phrase)
            if value not in ignored
        ]
        if not required:
            continue
        matched = sum(
            any(
                candidate.startswith(value) or value.startswith(candidate)
                for candidate in observed
            )
            for value in required
        )
        if matched / len(required) >= 0.75:
            return True
    return False


def _unit_blocks(text):
    lines = text.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(
            r"^\s*#{2,4}\s+(?:task|milestone|phase)\b",
            line,
            flags=re.IGNORECASE,
        )
    ]
    blocks = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        blocks.append("\n".join(lines[start:end]).strip())
    return blocks


def _inline_code(line):
    parts = line.split("`")
    return [part.strip() for part in parts[1::2] if part.strip()]


def _path(value):
    candidate = value.strip().strip(".,;:()[]{}")
    if ":" in candidate and not candidate.startswith(("http:", "https:")):
        candidate = candidate.split(":", 1)[0]
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", candidate):
        return ""
    if not candidate.startswith(
        ("fugue/", "tests/", "examples/", ".github/", "compose.")
    ):
        return ""
    return candidate


def _modification_paths(text):
    action = re.compile(
        r"\b(?:modify|create|delete|rename|test(?:s|ing)?|files?)\b",
        flags=re.IGNORECASE,
    )
    values = []
    for line in text.splitlines():
        if re.search(
            r"\b(?:do not|must not|without)\s+(?:modify|change|edit|create|delete)\b",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        if not action.search(line):
            continue
        for token in _inline_code(line):
            candidate = _path(token)
            if candidate:
                values.append(candidate)
    return list(dict.fromkeys(values))


def _artifact_validity(text, blocks):
    placeholders = (
        "tbd",
        "todo later",
        "fill in details",
        "similar to above",
        "appropriate tests",
        "appropriate error handling",
    )
    lower = text.lower()
    return bool(
        500 <= len(text) <= 14_000
        and re.match(r"^\s*#\s+\S", text)
        and blocks
        and not any(value in lower for value in placeholders)
    )


def _requirement_coverage(text, expected):
    requirements = _array(expected.get("requirements"))
    if not requirements:
        return False
    groups_total = 0
    groups_covered = 0
    for requirement in requirements:
        item = _mapping(requirement)
        groups = _array(item.get("groups"))
        contradictions = _array(item.get("contradictions"))
        if not groups:
            return False
        groups_total += len(groups)
        groups_covered += sum(_contains_group(text, group) for group in groups)
        if any(_contains_group(text, group) for group in contradictions):
            return False
    return groups_covered / groups_total >= 0.85


def _repository_grounding(text, expected):
    paths = set(_modification_paths(text))
    prefixes = tuple(str(value) for value in _array(expected.get("allowed_prefixes")))
    anchors = _array(expected.get("modification_anchor_groups"))
    symbols = [
        _normalized(value) for value in _array(expected.get("required_symbols"))
    ]
    configured_minimum = int(expected.get("minimum_symbols", len(symbols)))
    minimum_symbols = min(
        configured_minimum,
        max(2, (len(symbols) + 1) // 2),
    )
    if not paths or not prefixes or not anchors:
        return False
    if any(not path.startswith(prefixes) for path in paths):
        return False
    if not all(any(str(path) in paths for path in _array(group)) for group in anchors):
        return False
    return sum(symbol in _normalized(text) for symbol in symbols) >= minimum_symbols


def _dependency_contracts(text, blocks, expected):
    lower = _normalized(text)
    terms = [
        _normalized(value) for value in _array(expected.get("dataflow_terms"))
    ]
    minimum = int(expected.get("minimum_dataflow_terms", 2))
    signature = bool(
        re.search(r"`?[A-Za-z_][A-Za-z0-9_]*\([^\n)]*\)\s*(?:->|→)", text)
        or re.search(r"\breturns?\b", lower)
    )
    dataflow = sum(term in lower for term in terms) >= minimum
    if len(blocks) < 2:
        return signature and dataflow
    cross_unit = bool(
        re.search(r"\b(?:task|milestone|phase)\s+[1-9][0-9]*\b", lower)
        or re.search(r"\b(?:consumes?|produces?|depends on|feeds|passes)\b", lower)
    )
    return signature and dataflow and cross_unit


def _reviewable_decomposition(blocks):
    if not blocks:
        return False
    action = re.compile(
        r"\b(?:add|change|update|preserve|reject|bind|propagate|write|implement|verify|map|exercise)\b",
        flags=re.IGNORECASE,
    )
    for block in blocks:
        if (
            not _modification_paths(block)
            or not action.search(block)
            or not re.search(r"\b(?:test|verify|assert|run)\b", block, re.IGNORECASE)
        ):
            return False
    return True


def _verification_quality(text, expected):
    lower = _normalized(text)
    commands = _array(expected.get("verification_command_groups"))
    scenarios = _array(expected.get("verification_scenario_groups"))
    has_commands = bool(commands) and all(
        _contains_group(lower, group) for group in commands
    )
    has_scenarios = bool(scenarios) and all(
        _contains_group(lower, group) for group in scenarios
    )
    failure_path = _contains_group(
        lower,
        ["failure injection", "simulate failure", "monkeypatch", "forced failure"],
    )
    return has_commands and has_scenarios and failure_path


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
    secret_literal = re.compile(
        r"(?:WANDB|ANTHROPIC|OPENAI)_API_KEY\s*=\s*['\"][^'$\{][^'\"]+['\"]",
        flags=re.IGNORECASE,
    )
    prefixes = tuple(str(value) for value in _array(expected.get("allowed_prefixes")))
    return bool(
        prefixes
        and not any(value in lower for value in unsafe)
        and not secret_literal.search(text)
        and all(path.startswith(prefixes) for path in _modification_paths(text))
    )


def _scenario_coverage(text, expected):
    groups = _array(expected.get("scenario_groups"))
    if not groups:
        return False
    covered = sum(_contains_group(text, group) for group in groups)
    minimum = float(expected.get("minimum_scenario_fraction", 1))
    return covered / len(groups) >= minimum


def score(task, output, evidence):
    del task
    expected = _mapping(evidence.get("expected"))
    text = _text(output)
    blocks = _unit_blocks(text)
    if not expected or not text:
        return dict(_DIMENSIONS)
    return {
        "artifact_validity": _artifact_validity(text, blocks),
        "requirement_coverage": _requirement_coverage(text, expected),
        "repository_grounding": _repository_grounding(text, expected),
        "dependency_contracts": _dependency_contracts(text, blocks, expected),
        "reviewable_decomposition": _reviewable_decomposition(blocks),
        "verification_quality": _verification_quality(text, expected),
        "scope_and_secret_safety": _scope_and_secret_safety(text, expected),
        "scenario_coverage": _scenario_coverage(text, expected),
    }
