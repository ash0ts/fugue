from __future__ import annotations

import json
from pathlib import Path

from fugue.doctor import doctor_report
from fugue.resource_integrity import (
    ASSET_MANIFEST_NAME,
    generate_packaged_asset_manifest,
    verify_packaged_assets,
)


def _resource_root(tmp_path: Path) -> Path:
    root = tmp_path / "resources"
    (root / "runtime" / "example").mkdir(parents=True)
    (root / "runtime" / "example" / "recipe.json").write_text(
        '{"version": 1}\n'
    )
    (root / "templates").mkdir()
    (root / "templates" / "README.md").write_text("# Example\n")
    (root / ASSET_MANIFEST_NAME).write_bytes(
        generate_packaged_asset_manifest(root)
    )
    return root


def test_checked_in_packaged_asset_manifest_is_exact() -> None:
    integrity = verify_packaged_assets()

    assert integrity["ready"] is True
    assert integrity["expected_files"] > 0
    assert integrity["verified_files"] == integrity["expected_files"]
    assert integrity["actual_static_files"] == integrity["expected_files"]
    assert integrity["missing_files"] == []
    assert integrity["tampered_files"] == []
    assert integrity["unexpected_files"] == []
    assert integrity["unsafe_files"] == []
    assert integrity["manifest_errors"] == []


def test_packaged_asset_integrity_reports_missing_and_tampered_files(
    tmp_path: Path,
) -> None:
    root = _resource_root(tmp_path)
    (root / "runtime" / "example" / "recipe.json").write_text(
        '{"version": 2}\n'
    )
    (root / "templates" / "README.md").unlink()

    integrity = verify_packaged_assets(root)

    assert integrity["ready"] is False
    assert integrity["missing_files"] == ["templates/README.md"]
    assert [item["path"] for item in integrity["tampered_files"]] == [
        "runtime/example/recipe.json"
    ]
    assert integrity["verified_files"] == 0


def test_packaged_asset_integrity_rejects_unmanifested_static_files(
    tmp_path: Path,
) -> None:
    root = _resource_root(tmp_path)
    (root / "unreviewed.txt").write_text("not in the immutable inventory\n")

    integrity = verify_packaged_assets(root)

    assert integrity["ready"] is False
    assert integrity["unexpected_files"] == ["unreviewed.txt"]
    assert integrity["verified_files"] == 2


def test_packaged_asset_integrity_rejects_manifest_path_traversal(
    tmp_path: Path,
) -> None:
    root = _resource_root(tmp_path)
    manifest = json.loads((root / ASSET_MANIFEST_NAME).read_text())
    manifest["files"][0]["path"] = "../outside.txt"
    (root / ASSET_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    integrity = verify_packaged_assets(root)

    assert integrity["ready"] is False
    assert integrity["verified_files"] == 0
    assert len(integrity["manifest_errors"]) == 1
    assert "path is unsafe" in integrity["manifest_errors"][0]


def test_packaged_asset_integrity_rejects_malformed_manifest(
    tmp_path: Path,
) -> None:
    root = _resource_root(tmp_path)
    (root / ASSET_MANIFEST_NAME).write_text('{"schema_version": 1')

    integrity = verify_packaged_assets(root)

    assert integrity["ready"] is False
    assert integrity["manifest_errors"]
    assert "invalid asset-manifest-v1.json" in integrity["manifest_errors"][0]


def test_doctor_fails_packaged_assets_on_integrity_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "fugue.doctor.verify_packaged_assets",
        lambda: {
            "schema_version": 1,
            "manifest": ASSET_MANIFEST_NAME,
            "algorithm": "sha256",
            "ready": False,
            "expected_files": 2,
            "actual_static_files": 2,
            "verified_files": 1,
            "missing_files": [],
            "unexpected_files": [],
            "tampered_files": [{"path": "runtime/example/recipe.json"}],
            "unsafe_files": [],
            "manifest_errors": [],
        },
    )

    report = doctor_report(tmp_path)

    packaged = report["readiness"]["requirements"]["packaged_assets"]
    assert report["ok"] is False
    assert packaged == {
        "ready": False,
        "detail": "verified 1/2; 1 changed",
    }
    assert report["assets"]["integrity"]["ready"] is False
