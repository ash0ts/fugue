from __future__ import annotations

import re
from pathlib import Path

import yaml


def test_fugue_dev_skill_has_valid_minimal_frontmatter() -> None:
    path = Path(".codex/skills/fugue-dev/SKILL.md")
    content = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)

    assert match is not None
    frontmatter = yaml.safe_load(match.group(1))
    assert set(frontmatter) == {"name", "description"}
    assert frontmatter["name"] == "fugue-dev"
    assert re.fullmatch(r"[a-z0-9-]+", frontmatter["name"])
    assert 1 <= len(frontmatter["description"]) <= 1024
    assert "<" not in frontmatter["description"]
    assert ">" not in frontmatter["description"]
