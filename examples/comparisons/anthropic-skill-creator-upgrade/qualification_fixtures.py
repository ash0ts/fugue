"""Build host-only base-fail and gold-pass scorer qualification fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from prepare_confirmatory import _initial_files

EXAMPLE = Path(__file__).resolve().parent
LABELS = EXAMPLE / "confirmatory-private-labels.jsonl"


def _terms(groups: object) -> list[str]:
    if not isinstance(groups, list):
        return []
    return [
        str(group[0])
        for group in groups
        if isinstance(group, list) and group
    ]


def _set_frontmatter_value(text: str, key: str, value: str | None) -> str:
    prefix, body = text.split("\n---\n", 1)
    lines = prefix.splitlines()
    replaced = False
    updated: list[str] = []
    for line in lines:
        if line.startswith(f"{key}:"):
            replaced = True
            if value is not None:
                updated.append(f"{key}: {value}")
        elif line.startswith("audience:"):
            continue
        else:
            updated.append(line)
    if value is not None and not replaced:
        updated.append(f"{key}: {value}")
    return "\n".join(updated) + "\n---\n" + body


def _created_skill(expected: dict[str, object]) -> str:
    name = str(expected["skill_name"])
    title = " ".join(part.title() for part in name.split("-"))
    lines = [
        "---",
        f"name: {name}",
        (
            "description: Review immutable local inputs and produce a bounded, "
            "cited maintainer result. Use when exact evidence must be checked."
        ),
    ]
    compatibility = expected.get("compatibility")
    if isinstance(compatibility, dict) and compatibility.get("policy") == "required":
        lines.append("compatibility: " + ", ".join(_terms(compatibility.get("groups"))))
    body_terms = _terms(expected.get("instruction_groups"))
    lines.extend(
        (
            "---",
            "",
            f"# {title}",
            "",
            "## Workflow",
            "",
            "Inspect immutable local inputs, cite each evidence path, and keep the result bounded.",
            "",
            "## Required behavior",
            "",
            "; ".join(body_terms) + ".",
            "",
            "If a required input is missing or unavailable, stop without guessing.",
            "",
        )
    )
    return "\n".join(lines)


def gold_output(task_id: str, expected: dict[str, object]) -> dict[str, object]:
    disposition = str(expected["disposition"])
    if disposition == "rejected":
        files: dict[str, str] = {}
        memo = (
            "Rejected the exact 65-character identifier because it exceeds the "
            "maximum 64-character package contract; no files were created."
        )
    elif expected.get("findings") is not None:
        files = {
            "scripts/init_skill.py": (
                "def help_text():\n"
                "    return 'Skill name requirements: lowercase kebab-case; Max 64 characters'\n"
            )
        }
        memo = "Corrected only the initializer help and reconciled it with the validator."
    elif disposition == "created":
        files = {str(expected["skill_path"]): _created_skill(expected)}
        memo = "Created the requested bounded single-file Skill without dependencies or secrets."
    else:
        all_inputs = _initial_files(task_id)
        allowed = {str(path) for path in expected.get("allowed_paths", [])}
        files = {
            path: content for path, content in all_inputs.items() if path in allowed
        }
        skill_path = str(expected.get("skill_path") or "")
        if skill_path:
            skill = files[skill_path]
            contract = expected.get("compatibility")
            if isinstance(contract, dict):
                policy = contract.get("policy")
                if policy == "required":
                    skill = _set_frontmatter_value(
                        skill,
                        "compatibility",
                        ", ".join(_terms(contract.get("groups"))),
                    )
                elif policy == "absent":
                    skill = _set_frontmatter_value(skill, "compatibility", None)
            skill = _set_frontmatter_value(skill, "__remove_unknown__", None)
            if expected.get("preserved_body_groups"):
                skill += (
                    "\n## Requested update\n\n"
                    + "; ".join(_terms(expected.get("instruction_groups")))
                    + ".\n"
                )
            files[skill_path] = skill
        memo = "Applied only the requested package change and preserved all locked behavior."
    return {
        "schema_version": 1,
        "task_id": task_id,
        "disposition": disposition,
        "skill_name": expected["skill_name"],
        "files": files,
        "findings": expected.get("findings", {}),
        "maintainer_memo": memo,
    }


def qualification_evidence() -> dict[str, object]:
    return {
        "opened_paths": [
            "/opt/skills/skill-creator/SKILL.md",
            "/opt/skills/skill-creator/scripts/init_skill.py",
            "/opt/skills/skill-creator/scripts/quick_validate.py",
        ],
        "tool_calls": [],
    }


def augment(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    augmented: list[dict[str, object]] = []
    for row in rows:
        task_id = str(row["id"])
        expected = dict(row["expected"])
        gold = gold_output(task_id, expected)
        base = dict(gold)
        base["schema_version"] = 0
        augmented.append(
            {
                "id": task_id,
                "expected": expected,
                "base_output": base,
                "gold_output": gold,
                "base_evidence": qualification_evidence(),
                "gold_evidence": qualification_evidence(),
            }
        )
    return augmented


def main() -> None:
    rows = [
        json.loads(line)
        for line in LABELS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in augment(rows):
        print(json.dumps(row, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
