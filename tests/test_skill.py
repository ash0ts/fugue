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


def test_fugue_dev_skill_preserves_release_semantics() -> None:
    content = Path(".codex/skills/fugue-dev/SKILL.md").read_text(encoding="utf-8")
    normalized = " ".join(content.split())
    required = (
        "Resolve one immutable candidate",
        "Candidate identity contains behavior only",
        "only to the execution fingerprint",
        "Require authored context delivery",
        "Nothing executes before the lock is durable",
        "Setup is the only stateful preparation boundary",
        "confirmed skill/context registration",
        "one versioned prediction row",
        "project, prediction identity, scorer version, and revision",
        "Direct diagnostics never synthesize Agent identity",
        "every applicable cell is terminal",
        "Terminal unscored cells are permitted",
        "Curator output stays inside its immutable declaration allowlist",
    )
    for invariant in required:
        assert invariant in normalized

    assert "unscored cells block" not in content.lower()
    assert "raw public MCP configuration" in content
