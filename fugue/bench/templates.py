from __future__ import annotations

import os
import platform
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Protocol


class _Resource(Protocol):
    @property
    def name(self) -> str: ...

    def is_dir(self) -> bool: ...

    def is_file(self) -> bool: ...

    def iterdir(self): ...

    def read_bytes(self) -> bytes: ...


@dataclass(frozen=True)
class StandaloneTemplate:
    id: str
    title: str
    changed_dimension: str


_TEMPLATES = (
    StandaloneTemplate("prompt-change", "Prompt change", "prompt_id"),
    StandaloneTemplate("skill-change", "Skill change", "skills"),
    StandaloneTemplate("mcp-change", "MCP change", "integrations"),
    StandaloneTemplate("memory-change", "Memory change", "context"),
    StandaloneTemplate(
        "harness-change",
        "Harness configuration change",
        "agent_kwargs",
    ),
)
_TEMPLATE_BY_ID = {item.id: item for item in _TEMPLATES}
_RENAMED_ROOT_FILES = {
    "env.example.template": ".env.example",
    "fugue-study.marker": ".fugue-study.json",
    "gitignore.template": ".gitignore",
}
_SCORER_PLATFORM_MARKER = b"{{FUGUE_SCORER_PLATFORM}}"


def standalone_templates() -> tuple[StandaloneTemplate, ...]:
    """Return the stable installed-template catalogue."""

    return _TEMPLATES


def standalone_template_ids() -> tuple[str, ...]:
    return tuple(item.id for item in _TEMPLATES)


def get_standalone_template(template_id: str) -> StandaloneTemplate:
    try:
        return _TEMPLATE_BY_ID[template_id]
    except KeyError as exc:
        choices = ", ".join(standalone_template_ids())
        raise ValueError(
            f"unknown standalone template {template_id!r}; choose one of: {choices}"
        ) from exc


def scaffold_standalone_template(
    destination: Path,
    *,
    template_id: str,
    force: bool = False,
) -> Path:
    """Copy one packaged study template without following links.

    Template contents are ordinary package resources, which keeps this path
    usable from a wheel or a zip importer. The complete resource inventory is
    validated before the first destination write.
    """

    template = get_standalone_template(template_id)
    destination = destination.expanduser().absolute()
    if destination.is_symlink():
        raise ValueError(f"template destination may not be a symlink: {destination}")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(
            f"refusing to overwrite non-empty template directory: {destination}"
        )

    package_root = files("fugue").joinpath("resources", "templates", template.id)
    entries = _resource_files(package_root)
    if not entries:
        raise FileNotFoundError(f"packaged template is empty: {template.id}")

    rendered: dict[PurePosixPath, bytes] = {}
    for relative, source in entries:
        output_relative = _rendered_relative(relative)
        if output_relative in rendered:
            raise ValueError(
                f"packaged template has duplicate output path: {output_relative}"
            )
        rendered[output_relative] = source.read_bytes().replace(
            _SCORER_PLATFORM_MARKER,
            _native_scorer_platform().encode("ascii"),
        )
    workflow = files("fugue").joinpath("resources", "ci", "standalone-comparison.yml")
    if not workflow.is_file():
        raise FileNotFoundError("packaged standalone comparison workflow is missing")
    rendered[PurePosixPath(".github/workflows/fugue-comparison.yml")] = (
        workflow.read_bytes()
    )
    required = {
        PurePosixPath(".env.example"),
        PurePosixPath(".fugue-study.json"),
        PurePosixPath(".gitignore"),
        PurePosixPath("comparison.yaml"),
        PurePosixPath("private-labels.jsonl"),
        PurePosixPath("tasks.jsonl"),
    }
    missing = sorted(path.as_posix() for path in required - set(rendered))
    if missing:
        raise ValueError(
            f"packaged template {template.id!r} is incomplete: {', '.join(missing)}"
        )

    _preflight_destination(destination, rendered)
    destination.mkdir(parents=True, exist_ok=True)
    for relative, payload in sorted(
        rendered.items(), key=lambda item: item[0].as_posix()
    ):
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            target,
            payload,
            mode=0o600 if relative.name == "private-labels.jsonl" else 0o644,
        )
    return destination / "comparison.yaml"


def _native_scorer_platform() -> str:
    machine = platform.machine().lower()
    aliases = {
        "amd64": "linux/amd64",
        "x86_64": "linux/amd64",
        "aarch64": "linux/arm64",
        "arm64": "linux/arm64",
    }
    try:
        return aliases[machine]
    except KeyError as exc:
        raise RuntimeError(
            "standalone scorer runtime supports only amd64 and arm64 hosts; "
            f"observed architecture {machine or 'unknown'}"
        ) from exc


def _resource_files(
    root: _Resource,
) -> list[tuple[PurePosixPath, _Resource]]:
    if not root.is_dir():
        raise FileNotFoundError("packaged standalone template resources are missing")
    result: list[tuple[PurePosixPath, _Resource]] = []

    def visit(current: _Resource, prefix: PurePosixPath) -> None:
        for child in sorted(current.iterdir(), key=lambda item: item.name):
            # ``python -m compileall fugue`` is a required release gate and
            # can leave interpreter-specific bytecode beside template fixture
            # sources in a source checkout. Bytecode is never a study input,
            # is not portable across Python versions, and must not leak into
            # a generated study or its fixture digest.
            if child.name == "__pycache__" or child.name.endswith(".pyc"):
                continue
            _safe_resource_name(child.name)
            if _is_symlink(child):
                raise ValueError(
                    f"packaged template may not contain symlinks: {prefix / child.name}"
                )
            relative = prefix / child.name
            if child.is_dir():
                visit(child, relative)
            elif child.is_file():
                result.append((relative, child))
            else:
                raise ValueError(
                    f"packaged template entry is not a regular file: {relative}"
                )

    visit(root, PurePosixPath())
    return result


def _rendered_relative(relative: PurePosixPath) -> PurePosixPath:
    _validate_relative(relative)
    if len(relative.parts) == 1:
        return PurePosixPath(_RENAMED_ROOT_FILES.get(relative.name, relative.name))
    return relative


def _preflight_destination(
    root: Path,
    entries: dict[PurePosixPath, bytes],
) -> None:
    for relative in entries:
        _validate_relative(relative)
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(
                    f"template destination may not contain symlinks: {current}"
                )
        if current.exists() and not current.is_file():
            raise ValueError(f"template output path is not a regular file: {current}")


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    if path.is_symlink():
        raise ValueError(f"template output may not be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_relative(path: PurePosixPath) -> None:
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe packaged template path: {path}")
    for part in path.parts:
        _safe_resource_name(part)


def _safe_resource_name(name: str) -> None:
    if name in {"", ".", ".."} or "/" in name or "\\" in name or "\0" in name:
        raise ValueError(f"unsafe packaged template resource name: {name!r}")


def _is_symlink(resource: _Resource) -> bool:
    checker = getattr(resource, "is_symlink", None)
    return bool(checker()) if callable(checker) else False
