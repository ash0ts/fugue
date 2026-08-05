from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import fugue.comparison as public_comparison
from fugue.bench.tasksets import (
    SimpleTasksetBuilder,
    import_weave_dataset,
    simple_task_json_schema,
    write_taskset_schemas,
)


def test_public_comparison_api_exposes_taskset_builders() -> None:
    assert public_comparison.SimpleTasksetBuilder is SimpleTasksetBuilder
    assert callable(public_comparison.preview_comparison)


def test_builder_keeps_public_tasks_and_private_labels_separate(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    labels = tmp_path / "labels.jsonl"

    SimpleTasksetBuilder().add(
        task_id="expense-limit",
        input={"question": "What is the current limit?"},
        expected={"amount": 125},
        base_output={"amount": 100},
        gold_output={"amount": 125},
        tags=("policy",),
    ).write(tasks_path=tasks, private_labels_path=labels)

    public = json.loads(tasks.read_text())
    private = json.loads(labels.read_text())
    assert "expected" not in public
    assert private["expected"] == {"amount": 125}
    assert stat.S_IMODE(labels.stat().st_mode) == 0o600


def test_generated_schemas_reject_unknown_public_fields(tmp_path: Path) -> None:
    public, private = write_taskset_schemas(tmp_path)
    assert public.is_file()
    assert private.is_file()
    schema = simple_task_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"id", "input"}


def test_weave_dataset_import_requires_immutable_ref_and_is_idempotent(
    tmp_path: Path,
) -> None:
    ref = "weave:///entity/project/object/tasks:v0123456789"
    calls: list[str] = []

    def loader(value, env):
        del env
        calls.append(value)
        return [
            {
                "id": "task-one",
                "input": {"question": "Inspect the evidence."},
                "tags": ["evidence"],
                "partition": "holdout",
            }
        ]

    first = import_weave_dataset(
        ref,
        import_id="evidence-tasks",
        repo_root=tmp_path,
        loader=loader,
    )
    second = import_weave_dataset(
        ref,
        import_id="evidence-tasks",
        repo_root=tmp_path,
        loader=loader,
    )

    assert first == second
    assert first.row_count == 1
    assert first.tasks_path.endswith("/tasks.jsonl")
    assert calls == [ref, ref]
    imported = json.loads((tmp_path / first.tasks_path).read_text())
    assert imported["id"] == "task-one"

    with pytest.raises(ValueError, match="immutable revision"):
        import_weave_dataset(
            "weave:///entity/project/object/tasks:latest",
            import_id="mutable",
            repo_root=tmp_path,
            loader=loader,
        )


def test_weave_dataset_import_rejects_private_fields(tmp_path: Path) -> None:
    def loader(value, env):
        del value, env
        return [
            {
                "id": "leak",
                "input": {"question": "Question"},
                "expected": {"answer": "private"},
            }
        ]

    with pytest.raises(ValueError, match="unknown public task fields"):
        import_weave_dataset(
            "weave:///entity/project/object/tasks:v1",
            import_id="leaky",
            repo_root=tmp_path,
            loader=loader,
        )
