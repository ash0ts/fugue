from __future__ import annotations

import base64
import stat
from pathlib import Path

import pytest
import yaml

from fugue.bench.private_inputs import restore_comparison_private_labels


def _comparison(root: Path, private_labels: str) -> Path:
    path = root / "comparison.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "taskset": {
                    "tasks": "tasks.jsonl",
                    "private_labels": private_labels,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _encoded() -> str:
    return base64.b64encode(b'{"id":"task","expected":{}}\n').decode()


def test_restore_uses_only_the_spec_declared_path_with_mode_0600(
    tmp_path: Path,
) -> None:
    comparison = _comparison(tmp_path, ".fugue/private/labels.jsonl")

    restored = restore_comparison_private_labels(
        repo_root=tmp_path,
        comparison_path=comparison,
        encoded=_encoded(),
    )

    assert restored == tmp_path / ".fugue/private/labels.jsonl"
    assert stat.S_IMODE(restored.stat().st_mode) == 0o600
    assert restored.read_text(encoding="utf-8").startswith('{"id"')


@pytest.mark.parametrize(
    "private_labels",
    ("../outside.jsonl", "/tmp/outside.jsonl", "nested/../outside.jsonl"),
)
def test_restore_rejects_unsafe_declared_paths(
    tmp_path: Path,
    private_labels: str,
) -> None:
    comparison = _comparison(tmp_path, private_labels)

    with pytest.raises(ValueError, match="safe study-relative"):
        restore_comparison_private_labels(
            repo_root=tmp_path,
            comparison_path=comparison,
            encoded=_encoded(),
        )


def test_restore_rejects_symlink_parents_and_existing_destinations(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    comparison = _comparison(tmp_path, "linked/labels.jsonl")

    with pytest.raises(ValueError, match="symlink"):
        restore_comparison_private_labels(
            repo_root=tmp_path,
            comparison_path=comparison,
            encoded=_encoded(),
        )

    safe = _comparison(tmp_path, "labels.jsonl")
    (tmp_path / "labels.jsonl").write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        restore_comparison_private_labels(
            repo_root=tmp_path,
            comparison_path=safe,
            encoded=_encoded(),
        )


def test_restore_rejects_symlinked_comparison_and_malformed_payload(
    tmp_path: Path,
) -> None:
    real = _comparison(tmp_path, "labels.jsonl")
    linked = tmp_path / "linked-comparison.yaml"
    linked.symlink_to(real)

    with pytest.raises(ValueError, match="comparison must be a regular file"):
        restore_comparison_private_labels(
            repo_root=tmp_path,
            comparison_path=linked,
            encoded=_encoded(),
        )

    with pytest.raises(ValueError, match="not valid base64"):
        restore_comparison_private_labels(
            repo_root=tmp_path,
            comparison_path=real,
            encoded="not base64!",
        )
