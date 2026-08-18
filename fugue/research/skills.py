from __future__ import annotations

import shutil
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

SKILL_NAME = "optimize-agent-with-fugue"


def skill_directory() -> Traversable:
    resource = (
        files("fugue").joinpath("resources", "agent-skills", SKILL_NAME)
    )
    if not resource.joinpath("SKILL.md").is_file():
        raise RuntimeError("packaged Fugue Agent Skill is unavailable")
    return resource


def export_skill(destination: Path) -> Path:
    target = destination.resolve()
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty skill destination: {target}"
            )
    else:
        target.mkdir(parents=True)
    with as_file(skill_directory()) as source:
        shutil.copytree(source, target, dirs_exist_ok=True)
    return target
