from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any

ASSET_MANIFEST_NAME = "asset-manifest-v1.json"
ASSET_MANIFEST_SCHEMA_VERSION = 1
_ALGORITHM = "sha256"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_GENERATED_RESOURCE_PATHS = frozenset(
    {
        ASSET_MANIFEST_NAME,
        "build-provenance.json",
        "source-commit.txt",
    }
)


def verify_packaged_assets(
    resource_root: Traversable | None = None,
) -> dict[str, Any]:
    """Verify every static file shipped below ``fugue.resources``.

    Build provenance is intentionally excluded because it is generated after the
    source manifest is frozen. Python bytecode is also excluded because it is not
    package data and may be created beside an editable installation at import time.
    """

    root = resource_root or files("fugue").joinpath("resources")
    manifest_item = root.joinpath(ASSET_MANIFEST_NAME)
    manifest_errors: list[str] = []
    entries: dict[str, dict[str, Any]] = {}
    manifest_algorithm: str | None = None

    if not manifest_item.is_file():
        manifest_errors.append(f"missing {ASSET_MANIFEST_NAME}")
    else:
        try:
            document = _load_manifest(manifest_item.read_bytes())
            manifest_algorithm = str(document["algorithm"])
            entries = {
                str(entry["path"]): dict(entry) for entry in document["files"]
            }
        except (KeyError, OSError, TypeError, UnicodeDecodeError, ValueError) as exc:
            manifest_errors.append(str(exc))

    actual, unsafe_files, traversal_errors = _inventory_resource_files(root)
    manifest_errors.extend(traversal_errors)
    expected_paths = set(entries)
    actual_paths = set(actual)
    missing_files = sorted(expected_paths - actual_paths)
    unexpected_files = sorted(actual_paths - expected_paths)
    tampered_files: list[dict[str, Any]] = []
    verified_files = 0

    for relative in sorted(expected_paths & actual_paths):
        entry = entries[relative]
        body = actual[relative]
        observed_sha256 = hashlib.sha256(body).hexdigest()
        observed_size = len(body)
        expected_sha256 = str(entry["sha256"])
        expected_size = int(entry["size_bytes"])
        if observed_sha256 == expected_sha256 and observed_size == expected_size:
            verified_files += 1
            continue
        tampered_files.append(
            {
                "path": relative,
                "expected_sha256": expected_sha256,
                "observed_sha256": observed_sha256,
                "expected_size_bytes": expected_size,
                "observed_size_bytes": observed_size,
            }
        )

    ready = not any(
        (
            manifest_errors,
            unsafe_files,
            missing_files,
            unexpected_files,
            tampered_files,
        )
    )
    return {
        "schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
        "manifest": ASSET_MANIFEST_NAME,
        "algorithm": manifest_algorithm or _ALGORITHM,
        "ready": ready,
        "expected_files": len(expected_paths),
        "actual_static_files": len(actual_paths),
        "verified_files": verified_files,
        "missing_files": missing_files,
        "unexpected_files": unexpected_files,
        "tampered_files": tampered_files,
        "unsafe_files": sorted(unsafe_files),
        "manifest_errors": manifest_errors,
    }


def generate_packaged_asset_manifest(resource_root: Path) -> bytes:
    """Return the canonical manifest bytes for a source resource directory."""

    inventory, unsafe_files, traversal_errors = _inventory_resource_files(
        resource_root
    )
    if unsafe_files or traversal_errors:
        details = sorted([*unsafe_files, *traversal_errors])
        raise ValueError("cannot manifest unsafe resource inventory: " + "; ".join(details))
    document = {
        "algorithm": _ALGORITHM,
        "files": [
            {
                "path": relative,
                "sha256": hashlib.sha256(body).hexdigest(),
                "size_bytes": len(body),
            }
            for relative, body in sorted(inventory.items())
        ],
        "schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
    }
    return (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _load_manifest(body: bytes) -> dict[str, Any]:
    try:
        document = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {ASSET_MANIFEST_NAME}: {exc.msg}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"invalid {ASSET_MANIFEST_NAME}: root must be an object")
    if set(document) != {"algorithm", "files", "schema_version"}:
        raise ValueError(
            f"invalid {ASSET_MANIFEST_NAME}: expected algorithm, files, and "
            "schema_version"
        )
    if document["schema_version"] != ASSET_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"invalid {ASSET_MANIFEST_NAME}: unsupported schema_version"
        )
    if document["algorithm"] != _ALGORITHM:
        raise ValueError(f"invalid {ASSET_MANIFEST_NAME}: algorithm must be sha256")
    raw_entries = document["files"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"invalid {ASSET_MANIFEST_NAME}: files must be non-empty")

    seen: set[str] = set()
    previous: str | None = None
    for index, entry in enumerate(raw_entries):
        label = f"files[{index}]"
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(
                f"invalid {ASSET_MANIFEST_NAME}: {label} has an invalid shape"
            )
        relative = entry["path"]
        if not isinstance(relative, str) or not _safe_relative_path(relative):
            raise ValueError(
                f"invalid {ASSET_MANIFEST_NAME}: {label}.path is unsafe"
            )
        if _ignored_resource_path(relative):
            raise ValueError(
                f"invalid {ASSET_MANIFEST_NAME}: {label}.path is not static package data"
            )
        if relative in seen:
            raise ValueError(
                f"invalid {ASSET_MANIFEST_NAME}: duplicate path {relative!r}"
            )
        if previous is not None and relative <= previous:
            raise ValueError(
                f"invalid {ASSET_MANIFEST_NAME}: files must be sorted by path"
            )
        digest = entry["sha256"]
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise ValueError(
                f"invalid {ASSET_MANIFEST_NAME}: {label}.sha256 is invalid"
            )
        size = entry["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"invalid {ASSET_MANIFEST_NAME}: {label}.size_bytes is invalid"
            )
        seen.add(relative)
        previous = relative
    return document


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"invalid {ASSET_MANIFEST_NAME}: duplicate key {key!r}")
        value[key] = item
    return value


def _inventory_resource_files(
    root: Traversable,
) -> tuple[dict[str, bytes], list[str], list[str]]:
    inventory: dict[str, bytes] = {}
    unsafe_files: list[str] = []
    errors: list[str] = []

    def visit(directory: Traversable, parts: tuple[str, ...]) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            errors.append(f"cannot enumerate {'/'.join(parts) or '.'}: {exc}")
            return
        for child in children:
            name = child.name
            relative = "/".join((*parts, name))
            if not _safe_relative_path(relative):
                unsafe_files.append(relative)
                continue
            if isinstance(child, Path) and child.is_symlink():
                unsafe_files.append(relative)
                continue
            try:
                if child.is_dir():
                    if name == "__pycache__":
                        continue
                    visit(child, (*parts, name))
                elif child.is_file() and not _ignored_resource_path(relative):
                    inventory[relative] = child.read_bytes()
            except OSError as exc:
                errors.append(f"cannot read {relative}: {exc}")

    visit(root, ())
    return inventory, unsafe_files, errors


def _safe_relative_path(value: str) -> bool:
    if (
        not value
        or "\\" in value
        or value.startswith("/")
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return False
    path = PurePosixPath(value)
    return (
        path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
        and not path.is_absolute()
    )


def _ignored_resource_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        relative in _GENERATED_RESOURCE_PATHS
        or "__pycache__" in path.parts
        or path.suffix == ".pyc"
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate or validate Fugue's packaged-resource manifest."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "resources",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    expected = generate_packaged_asset_manifest(root)
    destination = root / ASSET_MANIFEST_NAME
    if args.write:
        destination.write_bytes(expected)
        return 0
    if not destination.is_file() or destination.read_bytes() != expected:
        print(
            f"{destination} is stale; run "
            "python -m fugue.resource_integrity --write"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
