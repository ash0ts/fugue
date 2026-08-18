from __future__ import annotations

import argparse
import base64
import os
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


def restore_comparison_private_labels(
    *,
    repo_root: Path,
    comparison_path: Path,
    encoded: str,
) -> Path:
    """Restore the spec-declared host-only labels through a safe CI boundary."""

    root = repo_root.resolve()
    source_input = comparison_path
    if not source_input.is_absolute():
        source_input = root / source_input
    if source_input.is_symlink():
        raise ValueError("comparison must be a regular file inside the study root")
    source = source_input.resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("comparison must be a regular file inside the study root")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("comparison must contain one YAML mapping")
    taskset = raw.get("taskset")
    if not isinstance(taskset, dict):
        raise ValueError("comparison taskset must be a mapping")
    relative = _safe_relative_path(
        taskset.get("private_labels"),
        label="comparison private labels",
    )
    destination = source.parent.joinpath(*relative.parts)
    if not destination.resolve(strict=False).is_relative_to(root):
        raise ValueError("comparison private labels must remain inside the study root")
    _prepare_private_parent(root, destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "private labels destination already exists; refusing to overwrite it"
        )
    try:
        body = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("private labels payload is not valid base64") from exc
    if not body.strip():
        raise ValueError("private labels payload is empty")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.private-input"
    )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, destination)
        destination.chmod(0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination


def _safe_relative_path(raw: Any, *, label: str) -> PurePosixPath:
    value = str(raw or "").strip()
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a safe study-relative path")
    return path


def _prepare_private_parent(root: Path, parent: Path) -> None:
    relative = parent.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("private labels path contains a symlink")
        if current.exists():
            if not current.is_dir():
                raise ValueError("private labels parent is not a directory")
            continue
        current.mkdir(mode=0o700)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restore one comparison's protected private-label corpus",
    )
    parser.add_argument("comparison")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--encoded-env", default="FUGUE_PRIVATE_LABELS_B64")
    args = parser.parse_args(argv)
    encoded = os.environ.get(args.encoded_env, "")
    if not encoded:
        parser.error(f"missing encoded payload environment: {args.encoded_env}")
    destination = restore_comparison_private_labels(
        repo_root=Path(args.repo_root),
        comparison_path=Path(args.comparison),
        encoded=encoded,
    )
    print(destination.as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())
