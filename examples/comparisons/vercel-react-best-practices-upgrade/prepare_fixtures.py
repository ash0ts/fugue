"""Prepare deterministic task archives for the Vercel Skill canary."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = Path(__file__).resolve().parent
FIXTURES_ROOT = EXAMPLE_ROOT / "fixtures"
SOURCE_LOCK = EXAMPLE_ROOT / "fixture-sources.lock.json"
OUTPUT_ROOT = (
    REPO_ROOT
    / ".fugue/comparison-resources/vercel-react-best-practices-upgrade"
)
FIXTURE_IDS = ("server-action-auth", "rsc-serialization")


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_and_verify_source_lock() -> dict[str, object]:
    value = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    supplied = str(value.pop("manifest_digest", ""))
    if value.get("schema_version") != 1 or supplied != _stable_digest(value):
        raise RuntimeError("fixture source lock digest does not match")
    records = value.get("fixtures")
    if not isinstance(records, list) or not records:
        raise RuntimeError("fixture source lock is empty")
    locked_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("fixture source lock record is invalid")
        relative = str(record.get("path") or "")
        path = FIXTURES_ROOT / relative
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not path.is_file()
            or path.is_symlink()
        ):
            raise RuntimeError(f"fixture source path is unsafe: {relative}")
        if path.stat().st_size != record.get("size") or _sha256(path) != record.get(
            "sha256"
        ):
            raise RuntimeError(f"fixture source changed: {relative}")
        locked_paths.add(relative)
    actual_paths = {
        path.relative_to(FIXTURES_ROOT).as_posix()
        for path in FIXTURES_ROOT.rglob("*")
        if path.is_file()
    }
    if actual_paths != locked_paths:
        raise RuntimeError("fixture source set changed")
    return {**value, "manifest_digest": supplied}


def _build_archive(fixture_id: str, destination: Path) -> dict[str, object]:
    source = FIXTURES_ROOT / fixture_id
    if fixture_id not in FIXTURE_IDS or not source.is_dir():
        raise RuntimeError(f"unknown fixture: {fixture_id}")
    files = tuple(sorted(path for path in source.rglob("*") if path.is_file()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w", format=tarfile.USTAR_FORMAT) as archive:
        for path in files:
            data = path.read_bytes()
            relative = path.relative_to(source).as_posix()
            info = tarfile.TarInfo(f"repo/{relative}")
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            archive.addfile(info, io.BytesIO(data))
    try:
        archive_path = destination.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        archive_path = destination.as_posix()
    return {
        "id": fixture_id,
        "archive": archive_path,
        "sha256": _sha256(destination),
        "file_count": len(files),
    }


def main() -> None:
    source_lock = _load_and_verify_source_lock()
    archives = [
        _build_archive(fixture_id, OUTPUT_ROOT / f"{fixture_id}.tar")
        for fixture_id in FIXTURE_IDS
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_manifest_digest": source_lock["manifest_digest"],
        "archives": archives,
    }
    manifest["manifest_digest"] = _stable_digest(manifest)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "fixtures.lock.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
