"""Host-only deterministic checks for the writing-plans confirmatory cohort.

The scorer does not reward headings introduced by either Skill revision.  It
compares a plan with an independently authored repository oracle: exact paths
and symbols, required cross-component edges, cohesive work units, and a
scenario-level verification matrix.  The oracle is frozen in the private label
bundle and is never mounted into an Agent cell.
"""

import re

_DIMENSIONS = dict.fromkeys(
    (
        "artifact_validity",
        "global_constraint_fidelity",
        "interface_graph_consistency",
        "right_sized_decomposition",
        "repository_grounding",
        "verification_matrix",
        "scope_secret_safety",
    ),
    False,
)

_RELATION = re.compile(
    r"\b(?:calls?|consumes?|depends on|feeds?|passes?|produces?|reads?|"
    r"returns?|supplies?|uses?|writes?)\b",
    flags=re.IGNORECASE,
)
_ACTION = re.compile(
    r"\b(?:add|change|create|define|delete|implement|modify|preserve|reject|"
    r"remove|rename|update|validate|verify|write)\b",
    flags=re.IGNORECASE,
)
_VERIFY = re.compile(
    r"\b(?:assert|check|exercise|pytest|ruff|test|validate|verification|verify)\b",
    flags=re.IGNORECASE,
)
_MODIFICATION_ACTION = (
    r"(?:add(?:ed|ing|s)?|chang(?:e|ed|es|ing)|creat(?:e|ed|es|ing)|"
    r"delet(?:e|ed|es|ing)|edit(?:ed|ing|s)?|implement(?:ed|ing|s)?|"
    r"modif(?:ied|ies|y|ying)|mov(?:e|ed|es|ing)|remov(?:e|ed|es|ing)|"
    r"renam(?:e|ed|es|ing)|replac(?:e|ed|es|ing)|touch(?:ed|es|ing)?|"
    r"updat(?:e|ed|es|ing)|writ(?:e|es|ing|ten))"
)
_DIRECT_MODIFICATION = re.compile(
    rf"^\s*(?:[-*+]\s+|\d+[.)]\s+)?(?:\*\*)?"
    rf"(?:(?:files?|paths?)\s+to\s+)?{_MODIFICATION_ACTION}\b",
    flags=re.IGNORECASE,
)
_TRAILING_MODIFICATION = re.compile(
    rf"\b{_MODIFICATION_ACTION}\s+(?:only\s+|the\s+)?$",
    flags=re.IGNORECASE,
)
_PATH_LEADING_MODIFICATION = re.compile(
    rf"^\s*(?:[,;:—-]\s*)?(?:then\s+)?{_MODIFICATION_ACTION}\b",
    flags=re.IGNORECASE,
)
_WRITING_PLANS_FILE_DIRECTIVE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?:\*\*)?"
    r"(?:create|modify|test)(?:\*\*)?\s*:\s*$",
    flags=re.IGNORECASE,
)
_NEGATED_MODIFICATION = re.compile(
    rf"(?:"
    rf"\b(?:do\s+not|don't|must\s+not|never|should\s+not|without)\b"
    rf"[^.;]{{0,100}}\b{_MODIFICATION_ACTION}\b"
    rf"|\bno\b[^.;]{{0,100}}\b(?:files?|paths?)\b[^.;]{{0,100}}"
    rf"\b(?:need(?:s|ed)?|require(?:d|s)?)\b[^.;]{{0,40}}"
    rf"\b{_MODIFICATION_ACTION}\b"
    rf"|\bno\s+(?:other\s+)?(?:changes?|edits?|modifications?|updates?)\b"
    rf")",
    flags=re.IGNORECASE,
)
_REFERENCE_INTRODUCTION = re.compile(
    r"(?:\balready\s+(?:defined|implemented|present|used)\b|"
    r"\bas\s+(?:defined|implemented|seen|used)\s+in\b|"
    r"\bbased\s+on\b|\bcompare(?:d)?\s+(?:to|with)\b|"
    r"\bfor\s+reference\b|\bidiom\b|\bmirrors?\b|\bmodeled\s+on\b|"
    r"\bpattern\s+(?:from|in)\b|"
    r"\bfollow(?:ing|s)?\s+(?:the\s+)?(?:existing\s+)?"
    r"(?:implementation|pattern)\s+in\b|"
    r"\bfollow(?:ing|s)?\s*$|"
    r"\busing\s+(?:the\s+)?(?:existing\s+)?(?:implementation|pattern)\s+in\b|"
    r"\bconsistent\s+with\b|"
    r"\bsee\b|\bsource\s+inspection\b)"
    r"[^.;:]{0,100}$",
    flags=re.IGNORECASE,
)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _array(value):
    return value if isinstance(value, list) else []


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _normalized(value):
    return " ".join(str(value).casefold().split())


def _contains_group(text, group):
    normalized = _normalized(text)
    tokens = set(re.findall(r"[a-z0-9_]+", normalized))
    ignored = {"a", "an", "and", "is", "of", "or", "the", "to", "with"}
    for raw in _array(group):
        phrase = _normalized(raw)
        if phrase and phrase in normalized:
            return True
        required = [
            token for token in re.findall(r"[a-z0-9_]+", phrase) if token not in ignored
        ]
        if required and all(
            any(
                observed.startswith(token) or token.startswith(observed)
                for observed in tokens
            )
            for token in required
        ):
            return True
    return False


def _contains_contradiction(text, group):
    for line in text.splitlines():
        if not _contains_group(line, group):
            continue
        if re.search(
            r"\b(?:avoid|disallow|do not|must not|never|no|"
            r"prohibit(?:ed|s|ing)?|reject(?:ed|s|ing)?|without)\b",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        return True
    return False


def _unit_blocks(text):
    lines = text.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(
            r"^\s*#{2,4}\s+(?:change|implementation|milestone|phase|step|task|work)\b",
            line,
            flags=re.IGNORECASE,
        )
    ]
    return [
        "\n".join(
            lines[
                start : starts[position + 1]
                if position + 1 < len(starts)
                else len(lines)
            ]
        ).strip()
        for position, start in enumerate(starts)
    ]


def _inline_code_spans(text):
    return [
        (match.group(1).strip(), match.start(), match.end())
        for match in re.finditer(r"`([^`]+)`", text)
        if match.group(1).strip()
    ]


def _path(value):
    candidate = value.strip().strip(",;:()[]{}").rstrip(".")
    if ":" in candidate and not candidate.startswith(("http:", "https:")):
        candidate = candidate.split(":", 1)[0]
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", candidate):
        return ""
    roots = (".github/", ".codex/", "docs/", "fugue/", "tests/")
    return candidate if candidate.startswith(roots) else ""


def _modification_paths(text):
    values = []
    for line in text.splitlines():
        for value, start, end in _inline_code_spans(line):
            path = _path(value)
            if not path:
                continue
            prefix = line[:start]
            clause_start = max(prefix.rfind(";"), prefix.rfind(". ")) + 1
            clause_prefix = prefix[clause_start:]
            clause = line[clause_start:]
            if _NEGATED_MODIFICATION.search(clause_prefix):
                continue
            if _REFERENCE_INTRODUCTION.search(clause_prefix):
                continue
            if not (
                _DIRECT_MODIFICATION.search(clause)
                or _TRAILING_MODIFICATION.search(clause_prefix)
                or _PATH_LEADING_MODIFICATION.search(line[end:])
                or _WRITING_PLANS_FILE_DIRECTIVE.fullmatch(prefix)
            ):
                continue
            values.append(path)
    return list(dict.fromkeys(values))


def _artifact_validity(text, blocks):
    placeholders = (
        "appropriate error handling",
        "appropriate tests",
        "fill in details",
        "implement later",
        "similar to above",
        "tbd",
        "todo later",
    )
    lower = text.casefold()
    return bool(
        700 <= len(text) <= 18_000
        and re.match(r"^\s*#\s+\S", text)
        and blocks
        and not any(value in lower for value in placeholders)
    )


def _global_constraint_fidelity(text, expected):
    constraints = _array(expected.get("global_constraints"))
    if not constraints:
        return False
    covered = 0
    total = 0
    for raw in constraints:
        constraint = _mapping(raw)
        groups = _array(constraint.get("groups"))
        contradictions = _array(constraint.get("contradictions"))
        if not groups:
            return False
        total += len(groups)
        covered += sum(_contains_group(text, group) for group in groups)
        if any(_contains_contradiction(text, group) for group in contradictions):
            return False
    minimum = float(expected.get("minimum_global_fraction", 1.0))
    return total > 0 and covered / total >= minimum


def _repository_grounding(text, blocks, expected):
    paths = set(_modification_paths(text))
    allowed = {str(value) for value in _array(expected.get("allowed_paths"))}
    anchors = _array(expected.get("anchor_groups"))
    bindings = _array(expected.get("repository_bindings"))
    if not paths or not allowed or not anchors or not bindings:
        return False
    if not paths <= allowed:
        return False
    if not all(any(str(path) in paths for path in _array(group)) for group in anchors):
        return False
    for raw in bindings:
        binding = _mapping(raw)
        path = str(binding.get("path") or "")
        symbols = [_normalized(value) for value in _array(binding.get("symbols"))]
        minimum = int(binding.get("minimum_symbols", len(symbols)))
        matching = [block for block in blocks if path in _modification_paths(block)]
        if not matching or not symbols:
            return False
        if (
            max(
                sum(symbol in _normalized(block) for symbol in symbols)
                for block in matching
            )
            < minimum
        ):
            return False
    return True


def _interface_graph_consistency(blocks, expected):
    edges = _array(expected.get("interface_edges"))
    if str(expected.get("interface_mode") or "") == "not_applicable":
        prohibited = _array(expected.get("prohibited_interface_groups"))
        return bool(prohibited) and not any(
            _contains_group("\n".join(blocks), group) for group in prohibited
        )
    if not edges:
        return False
    for raw in edges:
        edge = _mapping(raw)
        producer = _array(edge.get("producer"))
        artifact = _array(edge.get("artifact"))
        consumer = _array(edge.get("consumer"))
        if not producer or not artifact or not consumer:
            return False
        if not any(
            _RELATION.search(block)
            and _contains_group(block, producer)
            and _contains_group(block, artifact)
            and _contains_group(block, consumer)
            for block in blocks
        ):
            return False
    return True


def _right_sized_decomposition(blocks, expected):
    minimum = int(expected.get("unit_count_min", 1))
    maximum = int(expected.get("unit_count_max", 6))
    contracts = _array(expected.get("unit_contracts"))
    forbidden_titles = {
        str(value).casefold()
        for value in _array(expected.get("forbidden_standalone_titles"))
    }
    if not minimum <= len(blocks) <= maximum or not contracts:
        return False
    matched_blocks = set()
    for raw in contracts:
        contract = _mapping(raw)
        anchor_paths = {str(value) for value in _array(contract.get("anchor_paths"))}
        responsibilities = _array(contract.get("responsibilities"))
        minimum_responsibilities = int(
            contract.get("minimum_responsibilities", len(responsibilities))
        )
        candidates = []
        for index, block in enumerate(blocks):
            if index in matched_blocks:
                continue
            block_paths = set(_modification_paths(block))
            covered = sum(_contains_group(block, group) for group in responsibilities)
            if block_paths & anchor_paths and covered >= minimum_responsibilities:
                candidates.append(index)
        if not candidates:
            return False
        matched_blocks.add(candidates[0])
    for block in blocks:
        title = re.sub(r"^\s*#{2,4}\s*", "", block.splitlines()[0]).casefold()
        if title.strip(" :.-0123456789") in forbidden_titles:
            return False
        if (
            not _modification_paths(block)
            or not _ACTION.search(block)
            or not _VERIFY.search(block)
        ):
            return False
    return True


def _verification_matrix(text, expected):
    rows = _array(expected.get("verification_rows"))
    if not rows:
        return False
    observed = 0
    for raw in rows:
        row = _mapping(raw)
        scenario = _array(row.get("scenario"))
        command = _array(row.get("command"))
        assertion = _array(row.get("assertion"))
        if not scenario or not command or not assertion:
            return False
        observed += bool(
            _contains_group(text, scenario)
            and _contains_group(text, command)
            and _contains_group(text, assertion)
        )
    minimum = float(expected.get("minimum_verification_fraction", 1.0))
    return observed / len(rows) >= minimum


def _scope_secret_safety(text, expected):
    lower = text.casefold()
    paths = set(_modification_paths(text))
    allowed = {str(value) for value in _array(expected.get("allowed_paths"))}
    prohibited_actions = (
        "curl ",
        "git push",
        "gh pr create",
        "npm install",
        "pip install",
        "rm -rf",
        "uv add",
        "wget ",
        *_array(expected.get("prohibited_actions")),
    )
    secret_literal = re.compile(
        r"(?:ANTHROPIC|OPENAI|WANDB)_API_KEY\s*=\s*['\"][^'$\{][^'\"]+['\"]",
        flags=re.IGNORECASE,
    )
    prohibited_paths = {
        str(value) for value in _array(expected.get("prohibited_paths"))
    }
    return bool(
        allowed
        and paths <= allowed
        and paths.isdisjoint(prohibited_paths)
        and not any(str(value).casefold() in lower for value in prohibited_actions)
        and not secret_literal.search(text)
    )


def score(task, output, evidence):
    del task
    expected = _mapping(evidence.get("expected"))
    text = _text(output)
    blocks = _unit_blocks(text)
    if not expected or not text:
        return dict(_DIMENSIONS)
    return {
        "artifact_validity": _artifact_validity(text, blocks),
        "global_constraint_fidelity": _global_constraint_fidelity(text, expected),
        "interface_graph_consistency": _interface_graph_consistency(blocks, expected),
        "right_sized_decomposition": _right_sized_decomposition(blocks, expected),
        "repository_grounding": _repository_grounding(text, blocks, expected),
        "verification_matrix": _verification_matrix(text, expected),
        "scope_secret_safety": _scope_secret_safety(text, expected),
    }
