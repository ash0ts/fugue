from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from fugue.bench import runtime_provenance


class _FakeDistribution:
    def __init__(
        self,
        root: Path,
        paths: list[str] | None,
        *,
        text: dict[str, str] | None = None,
    ) -> None:
        self._root = root
        self.files = (
            None
            if paths is None
            else [PurePosixPath(path) for path in paths]
        )
        self.version = "9.8.7"
        self._text = text or {}

    def locate_file(self, path: PurePosixPath) -> Path:
        return self._root.joinpath(*path.parts)

    def read_text(self, filename: str) -> str | None:
        return self._text.get(filename)


def test_distribution_contract_covers_package_and_metadata_but_not_install_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _write_package(tmp_path)
    metadata = tmp_path / "fugue-9.8.7.dist-info"
    metadata.mkdir()
    values = {
        "METADATA": "Name: fugue\nVersion: 9.8.7\nRequires-Dist: pyyaml\n",
        "WHEEL": "Wheel-Version: 1.0\nTag: py3-none-any\n",
        "entry_points.txt": "[console_scripts]\nfugue = fugue.bench.cli:main\n",
        "top_level.txt": "fugue\n",
        "RECORD": "environment-specific-record\n",
        "direct_url.json": '{"url":"file:///private/build/path"}\n',
        "INSTALLER": "pip\n",
        "REQUESTED": "\n",
    }
    for name, value in values.items():
        (metadata / name).write_text(value, encoding="utf-8")
    license_dir = metadata / "licenses"
    license_dir.mkdir()
    (license_dir / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    declared = [
        f"{metadata.name}/{name}"
        for name in values
    ] + [f"{metadata.name}/licenses/LICENSE"]
    installed = _FakeDistribution(tmp_path, list(reversed(declared)))
    monkeypatch.setattr(runtime_provenance, "files", lambda _name: package)
    monkeypatch.setattr(
        runtime_provenance,
        "distribution",
        lambda _name: installed,
    )

    first = runtime_provenance.resolve_fugue_distribution_provenance()
    repeated = runtime_provenance.resolve_fugue_distribution_provenance()

    assert first == repeated
    assert first == {
        "schema_version": 2,
        "kind": "installed_distribution",
        "name": "fugue",
        "version": "9.8.7",
        "source_commit": "a" * 40,
        "digest_kind": "installed_distribution_contract_v1",
        "digest": first["digest"],
        "files": 8,
        "package_files": 3,
        "metadata_files": 5,
    }
    assert len(first["digest"]) == 64

    # These files describe one installation, not Fugue's executable contract.
    (metadata / "RECORD").write_text("different-record\n", encoding="utf-8")
    (metadata / "direct_url.json").write_text(
        '{"url":"file:///another/path"}\n', encoding="utf-8"
    )
    (metadata / "INSTALLER").write_text("uv\n", encoding="utf-8")
    (metadata / "REQUESTED").write_text("different\n", encoding="utf-8")
    (package / "__pycache__" / "ignored.pyc").write_bytes(b"different-bytecode")
    assert (
        runtime_provenance.resolve_fugue_distribution_provenance()["digest"]
        == first["digest"]
    )

    (metadata / "METADATA").write_text(
        "Name: fugue\nVersion: 9.8.8\nRequires-Dist: pyyaml\n",
        encoding="utf-8",
    )
    assert (
        runtime_provenance.resolve_fugue_distribution_provenance()["digest"]
        != first["digest"]
    )
    (metadata / "METADATA").write_text(values["METADATA"], encoding="utf-8")

    (metadata / "entry_points.txt").write_text(
        "[console_scripts]\nfugue = fugue.other:main\n", encoding="utf-8"
    )
    entry_point_changed = (
        runtime_provenance.resolve_fugue_distribution_provenance()["digest"]
    )
    assert entry_point_changed != first["digest"]

    (package / "resources" / "schema.json").write_text(
        '{"schema_version":2}\n', encoding="utf-8"
    )
    assert (
        runtime_provenance.resolve_fugue_distribution_provenance()["digest"]
        != entry_point_changed
    )


def test_distribution_provider_without_file_manifest_still_hashes_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _write_package(tmp_path)
    installed = _FakeDistribution(
        tmp_path,
        None,
        text={
            "METADATA": "Name: fugue\nVersion: 9.8.7\n",
            "entry_points.txt": "[console_scripts]\nfugue = fugue.bench.cli:main\n",
        },
    )
    monkeypatch.setattr(runtime_provenance, "files", lambda _name: package)
    monkeypatch.setattr(
        runtime_provenance,
        "distribution",
        lambda _name: installed,
    )

    before = runtime_provenance.resolve_fugue_distribution_provenance()
    installed._text["entry_points.txt"] = (
        "[console_scripts]\nfugue = fugue.changed:main\n"
    )
    after = runtime_provenance.resolve_fugue_distribution_provenance()

    assert before["digest_kind"] == "installed_distribution_contract_v1"
    assert before["metadata_files"] == 2
    assert before["package_files"] == 3
    assert after["digest"] != before["digest"]


def test_uninstalled_package_fallback_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _write_package(tmp_path)
    monkeypatch.setattr(runtime_provenance, "files", lambda _name: package)

    def _not_installed(_name: str) -> None:
        raise runtime_provenance.PackageNotFoundError

    monkeypatch.setattr(runtime_provenance, "distribution", _not_installed)

    value = runtime_provenance.resolve_fugue_distribution_provenance()

    assert value["kind"] == "package_content_fallback"
    assert value["digest_kind"] == "package_content_fallback_v1"
    assert value["version"] == "0+uninstalled"
    assert value["metadata_files"] == 0
    assert value["package_files"] == value["files"] == 3
    assert value["source_commit"] == "a" * 40


def _write_package(root: Path) -> Path:
    package = root / "fugue"
    resources = package / "resources"
    resources.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (resources / "schema.json").write_text(
        '{"schema_version":1}\n', encoding="utf-8"
    )
    (resources / "build-provenance.json").write_text(
        json.dumps({"schema_version": 1, "source_commit": "a" * 40}) + "\n",
        encoding="utf-8",
    )
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"bytecode")
    return package
