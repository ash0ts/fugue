from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from fugue.bench.manifest import PreparedArtifact, TaskSpec
from fugue.model_plane import (
    BRIDGE_BASE_URL_HOST,
    bridge_master_key,
    resolve_model_route,
)

DEFAULT_OPENWIKI_PACKAGE = "openwiki@0.1.0"
POINTER_DIR = ".fugue-memory"
POINTER_FILE = "README.md"


class MemoryBuilder(Protocol):
    name: str

    def build(self, repo_checkout: Path, task: TaskSpec, artifact_dir: Path) -> None:
        """Build memory artifacts from a checkout pinned to task.base_commit."""


class NoneBuilder:
    name = "none"

    def build(self, repo_checkout: Path, task: TaskSpec, artifact_dir: Path) -> None:
        return None


class AgentsMdBuilder:
    name = "agentsmd"

    def build(self, repo_checkout: Path, task: TaskSpec, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        top_level = _top_level_entries(repo_checkout)
        config_files = _interesting_files(repo_checkout)
        agents = [
            "# Repository Notes",
            "",
            f"- Task: `{task.id}`",
            f"- Repository: `{task.repo_slug}`",
            f"- Base commit: `{task.base_commit or 'unknown'}`",
            "",
            "## Top-Level Map",
            "",
            *[f"- `{entry}`" for entry in top_level],
            "",
            "## Configuration And Entry Points",
            "",
            *[f"- `{entry}`" for entry in config_files],
            "",
            "Use these notes as orientation only. The benchmark task still defines",
            "the required code change and verification target.",
            "",
        ]
        (artifact_dir / "AGENTS.md").write_text("\n".join(agents))
        _write_pointer(artifact_dir, self.name)


class OpenWikiBuilder:
    name = "openwiki"

    def __init__(self, package: str | None = None):
        self.package = package or os.environ.get(
            "FUGUE_OPENWIKI_PACKAGE", DEFAULT_OPENWIKI_PACKAGE
        )

    def build(self, repo_checkout: Path, task: TaskSpec, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        wiki_dir = artifact_dir / "openwiki"
        if wiki_dir.exists():
            shutil.rmtree(wiki_dir)
        wiki_dir.mkdir(parents=True)

        command_template = os.environ.get(
            "FUGUE_OPENWIKI_CMD",
            "npx -y {package} --repo {repo} --output {out}",
        )
        command = command_template.format(
            package=self.package,
            repo=repo_checkout.as_posix(),
            out=wiki_dir.as_posix(),
            task=task.id,
        )
        env = os.environ.copy()
        route = resolve_model_route(env.get("FUGUE_MODEL"), env)
        env.setdefault(
            "OPENAI_BASE_URL",
            route.chat_base_url or f"{BRIDGE_BASE_URL_HOST}/v1",
        )
        if route.chat_base_url:
            env.setdefault("OPENAI_API_KEY", env.get(route.api_key_env, ""))
        else:
            env.setdefault("OPENAI_API_KEY", bridge_master_key(env))
        subprocess.run(command, shell=True, check=True, env=env)

        agents = artifact_dir / "AGENTS.md"
        if not agents.exists():
            agents.write_text(
                "# Repository Wiki\n\n"
                "A generated repository wiki is available in `openwiki/`. "
                "Use it for orientation before editing code.\n"
            )
        _write_pointer(artifact_dir, self.name)


class StubBuilder:
    def __init__(self, name: str):
        self.name = name

    def build(self, repo_checkout: Path, task: TaskSpec, artifact_dir: Path) -> None:
        raise NotImplementedError(f"{self.name} is not implemented for the pilot")


def builder_for(condition: str) -> MemoryBuilder:
    builders: dict[str, MemoryBuilder] = {
        "none": NoneBuilder(),
        "agentsmd": AgentsMdBuilder(),
        "openwiki": OpenWikiBuilder(),
        "semsearch": StubBuilder("semsearch"),
        "deepwiki": StubBuilder("deepwiki"),
    }
    try:
        return builders[condition]
    except KeyError as exc:
        raise ValueError(f"unknown memory condition: {condition}") from exc


def build_artifact(
    *,
    condition: str,
    task: TaskSpec,
    repo_checkout: Path,
    artifact_root: Path,
) -> PreparedArtifact:
    builder = builder_for(condition)
    artifact_dir = artifact_root / condition / task.id
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    if condition != "none":
        builder.build(repo_checkout, task, artifact_dir)
    return PreparedArtifact(
        condition=condition,
        task_id=task.id,
        path=artifact_dir,
        builder=builder.name,
        metadata={
            "task": asdict(task),
            "repo_checkout": repo_checkout.as_posix(),
            "openwiki_package": getattr(builder, "package", None),
        },
    )


def write_condition_instruction(artifact_root: Path, condition: str) -> Path | None:
    if condition == "none":
        return None
    instruction = artifact_root / condition / "INSTRUCTION.md"
    instruction.parent.mkdir(parents=True, exist_ok=True)
    instruction.write_text(
        "Additional repository memory for this trial has been injected into "
        "the repository root. Start by reading `AGENTS.md` or "
        f"`{POINTER_DIR}/{POINTER_FILE}` when present, then solve the task.\n"
    )
    return instruction


def clone_repo_at_commit(task: TaskSpec, checkout_root: Path) -> Path:
    if not task.repo or not task.base_commit:
        raise ValueError(
            f"{task.id} needs repo and base_commit to build memory artifacts"
        )
    checkout = checkout_root / task.id
    if checkout.exists():
        shutil.rmtree(checkout)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    url = task.repo if "://" in task.repo or task.repo.endswith(".git") else (
        f"https://github.com/{task.repo}.git"
    )
    subprocess.run(
        ["git", "clone", "--no-tags", "--filter=blob:none", url, checkout.as_posix()],
        check=True,
    )
    subprocess.run(["git", "checkout", task.base_commit], cwd=checkout, check=True)
    subprocess.run(
        ["git", "remote", "remove", "origin"],
        cwd=checkout,
        check=False,
    )
    subprocess.run(
        ["git", "tag", "-l"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    tags = subprocess.run(
        ["git", "tag", "-l"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if tags:
        subprocess.run(["git", "tag", "-d", *tags], cwd=checkout, check=True)
    return checkout


def _write_pointer(artifact_dir: Path, condition: str) -> None:
    pointer = artifact_dir / POINTER_DIR / POINTER_FILE
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        f"# Fugue Memory Condition: {condition}\n\n"
        "These files were generated from the benchmark task's base commit only. "
        "Treat them as repository orientation, not as the source of the patch.\n"
    )


def _top_level_entries(repo_checkout: Path) -> list[str]:
    ignored = {".git", ".tox", ".venv", "__pycache__", "node_modules"}
    entries = [
        path.name + ("/" if path.is_dir() else "")
        for path in sorted(repo_checkout.iterdir(), key=lambda p: p.name.lower())
        if path.name not in ignored and not path.name.startswith(".mypy_cache")
    ]
    return entries[:40] or ["."]


def _interesting_files(repo_checkout: Path) -> list[str]:
    names = {
        "AGENTS.md",
        "CLAUDE.md",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "go.mod",
        "Cargo.toml",
        "tox.ini",
        "pytest.ini",
    }
    found = [
        path.relative_to(repo_checkout).as_posix()
        for path in repo_checkout.rglob("*")
        if path.is_file() and path.name in names and ".git" not in path.parts
    ]
    return sorted(found)[:60] or ["No standard entry-point files detected."]
