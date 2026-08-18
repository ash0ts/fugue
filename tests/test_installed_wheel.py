from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required to build a wheel")
def test_wheel_runs_from_empty_directory_without_harbor_or_weave(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", wheel_dir.as_posix()],
        cwd=repo_root,
        env={**os.environ, "UV_CACHE_DIR": (tmp_path / "uv-cache").as_posix()},
        check=True,
        capture_output=True,
        text=True,
    )
    [wheel] = wheel_dir.glob("fugue-*.whl")
    installed = tmp_path / "installed"
    installed.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert not any(
            "__pycache__" in name or name.endswith(".pyc") for name in names
        )
        assert "fugue/resources/context-systems/none.yaml" in names
        assert "fugue/resources/runtime/claude-code/package.json" in names
        assert "fugue/resources/vendor/weave-node-sdk.tgz" in names
        assert "fugue/resources/ci/standalone-comparison.yml" in names
        assert archive.read("fugue/resources/source-commit.txt").decode() == (
            _git_head(repo_root) + "\n"
        )
        for template in (
            "prompt-change",
            "skill-change",
            "mcp-change",
            "memory-change",
            "harness-change",
        ):
            assert f"fugue/resources/templates/{template}/comparison.yaml" in names
            assert f"fugue/resources/templates/{template}/scorer.py" in names
            assert (
                "fugue/resources/templates/"
                f"{template}/configs/fugue/task-authoring/profiles.yaml"
            ) in names
        archive.extractall(installed)

    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(
        """
import importlib.abc
import sys

class OptionalDependencyBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'harbor', 'weave'}:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, OptionalDependencyBlocker())
""",
        encoding="utf-8",
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _write_fake_scorer_docker(fake_bin / "docker")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((blocker.as_posix(), installed.as_posix())),
        "PATH": os.pathsep.join((fake_bin.as_posix(), os.environ.get("PATH", ""))),
    }

    doctor = _wheel_cli(empty, env, "doctor", "--json")
    assert doctor.returncode == 0, doctor.stderr
    report = json.loads(doctor.stdout)
    assert report["ok"] is True
    assert report["distribution"]["source_commit"] == _git_head(repo_root)
    assert report["optional_features"]["weave"]["installed"] is False
    assert report["optional_features"]["local_runner"]["installed"] is False

    initialized = _wheel_cli(empty, env, "init", "comparison")
    assert initialized.returncode == 0, initialized.stderr
    comparison = "comparison/comparison.yaml"
    checked = _wheel_cli(empty, env, "check", comparison, "--json")
    assert checked.returncode in {0, 2}, checked.stderr
    assert json.loads(checked.stdout)["status"] in {
        "ready",
        "needs_review",
        "blocked",
    }
    previewed = _wheel_cli(
        empty,
        env,
        "compare",
        comparison,
        "--preview",
        "--json",
    )
    assert previewed.returncode in {0, 2}, previewed.stderr
    assert "ModuleNotFoundError" not in previewed.stderr

    package_root = _wheel_python(
        empty,
        env,
        "from fugue.bench.job_config import _installed_fugue_package_root; "
        "print(_installed_fugue_package_root())",
    )
    assert package_root.returncode == 0, package_root.stderr
    assert Path(package_root.stdout.strip()).is_relative_to(installed)

    for template in (
        "skill-change",
        "mcp-change",
        "memory-change",
        "harness-change",
    ):
        destination = f"study-{template}"
        initialized = _wheel_cli(
            empty,
            env,
            "init",
            destination,
            "--template",
            template,
        )
        assert initialized.returncode == 0, initialized.stderr
        checked = _wheel_cli(
            empty,
            env,
            "check",
            f"{destination}/comparison.yaml",
            "--json",
        )
        assert checked.returncode == 2, checked.stderr
        readiness = json.loads(checked.stdout)
        assert readiness["task_count"] == 2
        assert readiness["estimated_cells"] == 8
        assert readiness["base_failures"] == 2
        assert readiness["gold_passes"] == 2
        assert not any("not found" in item for item in readiness["blockers"])


def _wheel_cli(
    cwd: Path,
    env: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    script = (
        "from fugue.bench.cli import main; "
        f"raise SystemExit(main({list(arguments)!r}))"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _wheel_python(
    cwd: Path,
    env: dict[str, str],
    script: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_fake_scorer_docker(path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
import os
import sys

arguments = sys.argv[1:]
if arguments and arguments[0] == "info":
    print("standalone-scorer-test")
    raise SystemExit(0)
if not arguments or arguments[0] != "run":
    raise SystemExit(2)
mount = arguments[arguments.index("--mount") + 1]
source = next(
    item.removeprefix("src=")
    for item in mount.split(",")
    if item.startswith("src=")
)
os.execv(
    sys.executable,
    [sys.executable, source + "/scorer.py", source + "/input.json"],
)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
