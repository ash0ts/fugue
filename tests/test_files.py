from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.bench.files import (
    as_list,
    as_mapping,
    atomic_write_json,
    latest_jsonl_records,
    require_unique,
)


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


def test_latest_jsonl_records_ignores_damage_and_keeps_latest(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"id":"a","status":"running"}\n'
        "not-json\n"
        '{"id":"a","status":"passed"}\n'
    )

    assert latest_jsonl_records(path, "id") == [{"id": "a", "status": "passed"}]
