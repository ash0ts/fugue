from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.bench.files import (
    as_list,
    as_mapping,
    atomic_write_json,
    docker_build_command,
    docker_compose_command,
    latest_jsonl_records,
    require_unique,
    store_consistent,
)


@pytest.mark.parametrize(
    ("help_text", "expected"),
    [
        ("Usage: docker build\n", ["docker", "build", "--pull"]),
        (
            "      --provenance string   Shorthand for --attest=type=provenance\n",
            ["docker", "build", "--provenance=false", "--pull"],
        ),
    ],
)
def test_docker_build_command_adapts_to_client_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    help_text: str,
    expected: list[str],
) -> None:
    monkeypatch.setattr(
        "fugue.bench.files.subprocess.check_output",
        lambda *args, **kwargs: help_text,
    )

    assert docker_build_command("--pull") == expected


@pytest.mark.parametrize(
    ("plugin_returncode", "legacy_path", "expected"),
    [
        (0, None, ["docker", "compose", "up", "-d"]),
        (1, "/usr/bin/docker-compose", ["docker-compose", "up", "-d"]),
    ],
)
def test_docker_compose_command_adapts_to_worker_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    plugin_returncode: int,
    legacy_path: str | None,
    expected: list[str],
) -> None:
    monkeypatch.setattr(
        "fugue.bench.files.subprocess.run",
        lambda *args, **kwargs: type("Result", (), {"returncode": plugin_returncode})(),
    )
    monkeypatch.setattr(
        "fugue.bench.files.shutil.which",
        lambda _name: legacy_path,
    )

    assert docker_compose_command("up", "-d") == expected


def test_atomic_json_is_private_complete_and_replaceable(tmp_path: Path) -> None:
    path = tmp_path / "nested/state.json"

    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2})

    assert json.loads(path.read_text()) == {"value": 2}
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(path.parent.glob(".*.tmp"))


def test_shared_schema_coercion_and_duplicate_validation() -> None:
    assert as_mapping(None) == {}
    assert as_list(None) == []
    with pytest.raises(ValueError, match="expected mapping"):
        as_mapping([])
    with pytest.raises(ValueError, match="duplicate item id.*same"):
        require_unique(["same", "same"], "item")
    values = {"same": {"value": 1}}
    store_consistent(values, "same", {"value": 1}, error="changed")
    with pytest.raises(ValueError, match="changed"):
        store_consistent(values, "same", {"value": 2}, error="changed")


def test_latest_jsonl_records_ignores_damage_and_keeps_latest(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"id":"a","status":"running"}\n'
        "not-json\n"
        '{"id":"a","status":"passed"}\n'
    )

    assert latest_jsonl_records(path, "id") == [{"id": "a", "status": "passed"}]
