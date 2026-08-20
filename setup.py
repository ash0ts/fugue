from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class FugueBuildPy(build_py):
    """Embed source identity in the built distribution, never the source tree."""

    def run(self) -> None:
        super().run()
        resources = Path(self.build_lib) / "fugue" / "resources"
        for bytecode in resources.rglob("*.pyc"):
            bytecode.unlink()
        for cache in sorted(
            resources.rglob("__pycache__"),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            if cache.is_dir() and not any(cache.iterdir()):
                cache.rmdir()
        commit = os.environ.get("FUGUE_BUILD_SOURCE_COMMIT") or _git_head()
        target = Path(self.build_lib) / "fugue" / "resources" / "build-provenance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"schema_version": 1, "source_commit": commit}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


setup(cmdclass={"build_py": FugueBuildPy})
