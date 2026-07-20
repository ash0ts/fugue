from __future__ import annotations

import shutil
from pathlib import Path

SKILL_NAME = "optimize-agent-with-fugue"


def skill_directory() -> Path:
    path = (
        Path(__file__).resolve().parents[1] / "resources" / "agent-skills" / SKILL_NAME
    )
    if not (path / "SKILL.md").is_file():
        raise RuntimeError("packaged Fugue Agent Skill is unavailable")
    return path


def export_skill(destination: Path) -> Path:
    target = destination.resolve()
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty skill destination: {target}"
            )
    else:
        target.mkdir(parents=True)
    shutil.copytree(skill_directory(), target, dirs_exist_ok=True)
    return target
