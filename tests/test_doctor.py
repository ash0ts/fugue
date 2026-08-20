from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path

from fugue.doctor import doctor_report


def test_doctor_is_ready_in_an_empty_workspace(tmp_path: Path) -> None:
    report = doctor_report(tmp_path)

    assert report["ok"] is True
    assert report["distribution"]["kind"] == "installed_distribution"
    assert report["workspace"]["kind"] == "unversioned"
    assert report["workspace"]["files"] == 0
    assert report["assets"]["vendor_archive"] is True
    assert "none" in report["assets"]["context_systems"]
    assert report["assets"]["schemas"] > 0
    assert report["host"]["architecture"]
    assert report["host"]["free_disk_bytes"] > 0
    assert "daemon_available" in report["host"]["docker"]
    assert report["model_route"]["credential_name"]
    assert all(
        item["available"]
        for item in report["assets"]["runtime_groups"].values()
    )


def test_doctor_does_not_require_optional_features(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def missing_version(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr("fugue.doctor.version", missing_version)
    monkeypatch.setattr("fugue.doctor.importlib.util.find_spec", lambda name: None)

    report = doctor_report(tmp_path)

    assert report["ok"] is True
    assert report["optional_features"]["weave"]["installed"] is False
    assert report["optional_features"]["local_runner"]["installed"] is False


def test_doctor_uses_supplied_environment_for_credentials(tmp_path: Path) -> None:
    report = doctor_report(
        tmp_path,
        model="anthropic/claude-sonnet-5",
        env={"ANTHROPIC_API_KEY": "present-for-test"},
    )

    assert report["model_route"]["credential_present"] is True
    assert report["credentials_present"]["ANTHROPIC_API_KEY"] is True
