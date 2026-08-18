from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

SOURCE_ROOT = Path(__file__).resolve().parent


def _load_build_helper() -> ModuleType:
    """Load the in-tree helper without trusting ambient import search paths."""

    helper_path = SOURCE_ROOT / "tools" / "fugue_build.py"
    spec = importlib.util.spec_from_file_location("_fugue_build_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load Fugue build helper at {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BUILD_HELPER = _load_build_helper()
carry_source_commit = _BUILD_HELPER.carry_source_commit
resolve_source_commit = _BUILD_HELPER.resolve_source_commit
write_source_commit = _BUILD_HELPER.write_source_commit


class FugueBuildPy(build_py):
    """Embed source identity in the built distribution, never the source tree."""

    def run(self) -> None:
        commit = resolve_source_commit(SOURCE_ROOT)
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
        write_source_commit(Path(self.build_lib), commit)
        target = Path(self.build_lib) / "fugue" / "resources" / "build-provenance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"schema_version": 1, "source_commit": commit}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


class FugueSdist(sdist):
    """Carry the exact source identity into Git-free release archives."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)
        carry_source_commit(
            SOURCE_ROOT,
            Path(base_dir),
            env=os.environ,
        )


setup(cmdclass={"build_py": FugueBuildPy, "sdist": FugueSdist})
