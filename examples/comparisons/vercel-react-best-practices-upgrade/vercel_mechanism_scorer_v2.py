"""Task-specific Skill-opening evidence for the Vercel confirmatory campaign.

These dimensions are mechanism evidence only.  They cannot satisfy an outcome
or safety gate and must never be pooled into task correctness.
"""

_RULE_BY_TAG = {
    "server-action-security": "rules/server-auth-actions.md",
    "rsc-serialization": "rules/server-dedup-props.md",
    "dom-batching-control": "rules/js-batch-dom-css.md",
    "large-array-control": "rules/js-min-max-loop.md",
    "hook-timing-control": "rules/advanced-use-latest.md",
    "event-signature-control": "rules/advanced-event-handler-refs.md",
}


def _values(value):
    return value if isinstance(value, list) else []


def _opened_paths(evidence):
    return {
        value.replace("\\", "/").lower()
        for value in _values(evidence.get("opened_paths"))
        if isinstance(value, str) and value
    }


def _suffix_opened(opened_paths, expected):
    expected = expected.lower()
    return any(
        value == expected or value.endswith("/" + expected)
        for value in opened_paths
    )


def _required_rule(task):
    tags = {str(value) for value in _values(task.get("tags"))}
    matches = [path for tag, path in _RULE_BY_TAG.items() if tag in tags]
    return matches[0] if len(matches) == 1 else None


def score(task, _output, evidence):
    opened_paths = _opened_paths(evidence)
    required = _required_rule(task)
    skill_opened = any(
        _suffix_opened(opened_paths, marker)
        for marker in (
            "react-best-practices/skill.md",
            "react-best-practices/agents.md",
        )
    ) or any("/react-best-practices/rules/" in value for value in opened_paths)
    return {
        "skill_material_opened": skill_opened,
        "task_relevant_rule_opened": bool(
            required
            and any(
                value == required.lower()
                or value.endswith("/react-best-practices/" + required.lower())
                for value in opened_paths
            )
        ),
    }
