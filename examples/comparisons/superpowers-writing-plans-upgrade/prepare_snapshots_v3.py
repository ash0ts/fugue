"""Prepare complete, immutable Fugue source trees for the V3 Skill canary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (
    REPO_ROOT
    / ".fugue/comparison-resources/superpowers-writing-plans-upgrade-v3"
)
SNAPSHOTS = {
    "credential-rotation": "c729d68f8c2e0f1b8ebc93428b1579973f81ac4a",
    "evidence-destination": "02f75cd953e5286389f5f2b6712ef95670a42a5f",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, commit in SNAPSHOTS.items():
        resolved = _git("rev-parse", f"{commit}^{{commit}}")
        if resolved != commit:
            raise RuntimeError(f"snapshot {name} did not resolve to its exact commit")
        tree = _git("rev-parse", f"{commit}^{{tree}}")
        paths = tuple(
            value
            for value in _git("ls-tree", "-r", "--name-only", commit).splitlines()
            if value
        )
        archive = OUTPUT / f"{name}-full-{commit}.tar"
        subprocess.run(
            (
                "git",
                "archive",
                "--format=tar",
                "--prefix=repo/",
                f"--output={archive}",
                commit,
            ),
            cwd=REPO_ROOT,
            check=True,
        )
        records.append(
            {
                "id": name,
                "commit": commit,
                "tree": tree,
                "scope": "complete_repository_tree",
                "archive": archive.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(archive),
                "file_count": len(paths),
                "paths_digest": hashlib.sha256(
                    "\n".join(paths).encode()
                ).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "repository": "https://github.com/ash0ts/fugue",
        "snapshots": records,
    }
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = OUTPUT / "snapshots.lock.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
