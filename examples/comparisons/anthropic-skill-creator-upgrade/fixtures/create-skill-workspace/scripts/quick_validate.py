"""Small offline structural check used by the locked authoring tasks."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    root = Path(sys.argv[1])
    skill = root / "SKILL.md"
    if not skill.is_file():
        return 1
    text = skill.read_text(encoding="utf-8")
    if not text.startswith("---\n") or text.count("---\n") < 2:
        return 1
    header = text.split("---\n", 2)[1]
    name = next(
        (
            line.split(":", 1)[1].strip()
            for line in header.splitlines()
            if line.startswith("name:")
        ),
        "",
    )
    compatibility = next(
        (
            line.split(":", 1)[1].strip()
            for line in header.splitlines()
            if line.startswith("compatibility:")
        ),
        "",
    )
    return 0 if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) and len(name) <= 64 and compatibility else 1


if __name__ == "__main__":
    raise SystemExit(main())
