from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.library import validate_id

SIMPLE_TASKSET_SCHEMA_VERSION = 1
TASKSET_IMPORT_ROOT = Path(".fugue/imports/tasksets")
_PUBLIC_FIELDS = frozenset({"id", "input", "resources", "tags", "partition"})
_PRIVATE_FIELDS = frozenset({"id", "expected", "base_output", "gold_output"})
_PARTITIONS = frozenset({"discovery", "qualification", "holdout"})


@dataclass(frozen=True)
class SimpleTaskV1:
    id: str
    input: dict[str, Any]
    resources: tuple[dict[str, str], ...] = ()
    tags: tuple[str, ...] = ()
    partition: str = "holdout"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrivateTaskLabelV1:
    id: str
    expected: Any
    base_output: Any | None = None
    gold_output: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeaveTasksetImportV1:
    schema_version: int
    import_id: str
    dataset_ref: str
    row_count: int
    row_digests: tuple[str, ...]
    tasks_path: str
    tasks_sha256: str
    import_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimpleTasksetBuilder:
    tasks: list[SimpleTaskV1] = field(default_factory=list)
    private_labels: list[PrivateTaskLabelV1] = field(default_factory=list)

    def add(
        self,
        *,
        task_id: str,
        input: Mapping[str, Any],
        expected: Any,
        base_output: Any | None = None,
        gold_output: Any | None = None,
        resources: Sequence[Mapping[str, str]] = (),
        tags: Sequence[str] = (),
        partition: str = "holdout",
    ) -> SimpleTasksetBuilder:
        self.tasks.append(
            simple_task(
                task_id=task_id,
                input=input,
                resources=resources,
                tags=tags,
                partition=partition,
            )
        )
        self.private_labels.append(
            private_task_label(
                task_id=task_id,
                expected=expected,
                base_output=base_output,
                gold_output=gold_output,
            )
        )
        return self

    def write(
        self, *, tasks_path: Path, private_labels_path: Path
    ) -> tuple[Path, Path]:
        return write_simple_taskset(
            self.tasks,
            self.private_labels,
            tasks_path=tasks_path,
            private_labels_path=private_labels_path,
        )


def simple_task(
    *,
    task_id: str,
    input: Mapping[str, Any],
    resources: Sequence[Mapping[str, str]] = (),
    tags: Sequence[str] = (),
    partition: str = "holdout",
) -> SimpleTaskV1:
    if partition not in _PARTITIONS:
        raise ValueError(
            "task partition must be discovery, qualification, or holdout"
        )
    task_input = _json_object(input, "task input")
    task_resources = tuple(
        _resource(dict(value), index)
        for index, value in enumerate(resources, start=1)
    )
    task_tags = tuple(_short_text(value, "task tag", 100) for value in tags)
    if len(set(task_tags)) != len(task_tags):
        raise ValueError("task tags must be unique")
    return SimpleTaskV1(
        id=validate_id(task_id, kind="task id"),
        input=task_input,
        resources=task_resources,
        tags=task_tags,
        partition=partition,
    )


def private_task_label(
    *,
    task_id: str,
    expected: Any,
    base_output: Any | None = None,
    gold_output: Any | None = None,
) -> PrivateTaskLabelV1:
    _json_value(expected, "expected value")
    _json_value(base_output, "base output")
    _json_value(gold_output, "gold output")
    return PrivateTaskLabelV1(
        id=validate_id(task_id, kind="task id"),
        expected=expected,
        base_output=base_output,
        gold_output=gold_output,
    )


def write_simple_taskset(
    tasks: Sequence[SimpleTaskV1 | Mapping[str, Any]],
    private_labels: Sequence[PrivateTaskLabelV1 | Mapping[str, Any]],
    *,
    tasks_path: Path,
    private_labels_path: Path,
) -> tuple[Path, Path]:
    public = tuple(_parse_public(value) for value in tasks)
    private = tuple(_parse_private(value) for value in private_labels)
    task_ids = [value.id for value in public]
    label_ids = [value.id for value in private]
    if not task_ids:
        raise ValueError("taskset must contain at least one task")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task ids must be unique")
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("private label ids must be unique")
    if set(task_ids) != set(label_ids):
        raise ValueError("public tasks and private labels must have identical ids")
    _write_jsonl(tasks_path, (value.to_dict() for value in public), mode=0o644)
    _write_jsonl(
        private_labels_path,
        (value.to_dict() for value in private),
        mode=0o600,
    )
    return tasks_path, private_labels_path


def prepare_locked_private_labels(
    *,
    source_path: Path,
    destination_path: Path,
    expected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize a reviewed private-label bundle without exposing its values."""

    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("private-label source must be a regular file")
    expected_by_id: dict[str, str] = {}
    for item in expected:
        if set(item) != {"task_id", "label_digest"}:
            raise ValueError("private-label lock fields changed")
        task_id = validate_id(str(item["task_id"]), kind="task id")
        digest = str(item["label_digest"])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("private-label lock requires SHA-256 digests")
        if task_id in expected_by_id:
            raise ValueError("private-label lock task ids must be unique")
        expected_by_id[task_id] = digest
    rows: dict[str, dict[str, Any]] = {}
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError("private-label rows must be objects")
        parsed = _parse_private(raw).to_dict()
        if parsed["id"] in rows:
            raise ValueError("private-label source task ids must be unique")
        rows[parsed["id"]] = parsed
    if set(rows) != set(expected_by_id):
        raise ValueError("private-label source does not match its reviewed lock")
    for task_id, row in rows.items():
        if stable_digest(row) != expected_by_id[task_id]:
            raise ValueError(f"private-label content changed: {task_id}")
    ordered = [rows[task_id] for task_id in sorted(rows)]
    _write_jsonl(destination_path, ordered, mode=0o600)
    unsigned = {
        "schema_version": 1,
        "kind": "locked_private_labels_preparation",
        "task_count": len(ordered),
        "task_ids": sorted(rows),
        "label_digests": [expected_by_id[task_id] for task_id in sorted(rows)],
        "lock_digest": stable_digest(list(expected)),
    }
    return {**unsigned, "receipt_digest": stable_digest(unsigned)}


def write_taskset_schemas(destination: Path) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    public = destination / "simple-task.schema.json"
    private = destination / "private-task-label.schema.json"
    atomic_write_json(public, simple_task_json_schema())
    atomic_write_json(private, private_task_label_json_schema())
    return public, private


def simple_task_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://fugue.local/schemas/simple-task-v1.json",
        "title": "Fugue simple public task",
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "input"],
        "properties": {
            "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]*$"},
            "input": {"type": "object"},
            "resources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "target"],
                    "properties": {
                        "path": {"type": "string"},
                        "target": {
                            "type": "string",
                            "pattern": "^/workspace/resources/",
                        },
                    },
                },
            },
            "tags": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "maxLength": 100},
            },
            "partition": {"enum": sorted(_PARTITIONS)},
        },
    }


def private_task_label_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://fugue.local/schemas/private-task-label-v1.json",
        "title": "Fugue private task label",
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "expected"],
        "properties": {
            "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]*$"},
            "expected": {},
            "base_output": {},
            "gold_output": {},
        },
    }


def import_weave_dataset(
    dataset_ref: str,
    *,
    import_id: str,
    repo_root: Path,
    env: Mapping[str, str] | None = None,
    loader: Callable[[str, Mapping[str, str] | None], Iterable[Mapping[str, Any]]]
    | None = None,
) -> WeaveTasksetImportV1:
    normalized_ref = _immutable_weave_dataset_ref(dataset_ref)
    validated_id = validate_id(import_id, kind="taskset import id")
    rows = tuple((loader or _load_weave_rows)(normalized_ref, env))
    tasks = tuple(_parse_public(row) for row in rows)
    if not tasks:
        raise ValueError("Weave Dataset contains no task rows")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Weave Dataset task ids must be unique")
    text = _jsonl(task.to_dict() for task in tasks)
    digest = hashlib.sha256(text.encode()).hexdigest()
    destination = (
        repo_root.resolve() / TASKSET_IMPORT_ROOT / validated_id / digest
    )
    receipt_path = destination / "import.json"
    if receipt_path.is_file():
        raw_existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        raw_existing["row_digests"] = tuple(raw_existing["row_digests"])
        existing = WeaveTasksetImportV1(**raw_existing)
        if (
            existing.dataset_ref != normalized_ref
            or existing.tasks_sha256 != digest
        ):
            raise ValueError("taskset import destination contains conflicting data")
        return existing
    destination.mkdir(parents=True, exist_ok=False)
    tasks_path = destination / "tasks.jsonl"
    tasks_path.write_text(text, encoding="utf-8")
    tasks_path.chmod(0o644)
    unsigned = WeaveTasksetImportV1(
        schema_version=SIMPLE_TASKSET_SCHEMA_VERSION,
        import_id=validated_id,
        dataset_ref=normalized_ref,
        row_count=len(tasks),
        row_digests=tuple(stable_digest(task.to_dict()) for task in tasks),
        tasks_path=tasks_path.relative_to(repo_root.resolve()).as_posix(),
        tasks_sha256=digest,
    )
    receipt = replace(
        unsigned,
        import_digest=stable_digest(
            {
                key: value
                for key, value in unsigned.to_dict().items()
                if key != "import_digest"
            }
        ),
    )
    atomic_write_json(receipt_path, receipt.to_dict())
    return receipt


def _load_weave_rows(
    dataset_ref: str, env: Mapping[str, str] | None
) -> Iterable[Mapping[str, Any]]:
    from fugue.weave_support import initialize_weave

    project = _weave_project(dataset_ref)
    weave = initialize_weave(project, env)
    dataset = weave.ref(dataset_ref).get()
    rows = getattr(dataset, "rows", None)
    if rows is None:
        raise ValueError("Weave reference does not resolve to a Dataset")
    return tuple(dict(row) for row in rows)


def _immutable_weave_dataset_ref(value: str) -> str:
    text = value.strip()
    if not text.startswith("weave:///"):
        raise ValueError("Dataset must use an immutable weave:/// reference")
    if "/object/" not in text:
        raise ValueError("Weave reference must identify a Dataset object")
    revision = text.rsplit("/", 1)[-1]
    if ":" not in revision or revision.endswith(":latest"):
        raise ValueError("Weave Dataset reference must include an immutable revision")
    version = revision.rsplit(":", 1)[-1]
    if not version or version == "latest":
        raise ValueError("Weave Dataset reference must not use latest")
    if "?" in text or "#" in text:
        raise ValueError("Weave Dataset reference must not contain selectors")
    return text


def _weave_project(reference: str) -> str:
    parts = reference.removeprefix("weave:///").split("/")
    if len(parts) < 4 or parts[2] != "object":
        raise ValueError("invalid Weave Dataset reference")
    return f"{parts[0]}/{parts[1]}"


def _parse_public(value: SimpleTaskV1 | Mapping[str, Any]) -> SimpleTaskV1:
    if isinstance(value, SimpleTaskV1):
        return value
    unknown = sorted(set(value) - _PUBLIC_FIELDS)
    if unknown:
        raise ValueError("unknown public task fields: " + ", ".join(unknown))
    return simple_task(
        task_id=str(value.get("id") or ""),
        input=_json_object(value.get("input"), "task input"),
        resources=tuple(value.get("resources") or ()),
        tags=tuple(value.get("tags") or ()),
        partition=str(value.get("partition") or "holdout"),
    )


def _parse_private(
    value: PrivateTaskLabelV1 | Mapping[str, Any],
) -> PrivateTaskLabelV1:
    if isinstance(value, PrivateTaskLabelV1):
        return value
    unknown = sorted(set(value) - _PRIVATE_FIELDS)
    if unknown:
        raise ValueError("unknown private label fields: " + ", ".join(unknown))
    if "expected" not in value:
        raise ValueError("private task label requires expected")
    return private_task_label(
        task_id=str(value.get("id") or ""),
        expected=value["expected"],
        base_output=value.get("base_output"),
        gold_output=value.get("gold_output"),
    )


def _resource(value: dict[str, Any], index: int) -> dict[str, str]:
    unknown = sorted(set(value) - {"path", "target"})
    if unknown:
        raise ValueError(
            f"unknown task resource {index} fields: {', '.join(unknown)}"
        )
    source = PurePosixPath(str(value.get("path") or ""))
    if (
        source.is_absolute()
        or not source.parts
        or any(part in {"", ".", ".."} for part in source.parts)
    ):
        raise ValueError("task resource path must be repository-relative")
    target = PurePosixPath(str(value.get("target") or ""))
    allowed = PurePosixPath("/workspace/resources")
    if (
        not target.is_absolute()
        or any(part in {"", ".", ".."} for part in target.parts)
        or target.parts[: len(allowed.parts)] != allowed.parts
    ):
        raise ValueError("task resource target must be under /workspace/resources")
    return {"path": source.as_posix(), "target": target.as_posix()}


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    result = dict(value)
    _json_value(result, label)
    return result


def _json_value(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc


def _short_text(value: Any, label: str, maximum: int) -> str:
    text = str(value).strip()
    if not text or len(text) > maximum:
        raise ValueError(f"{label} must contain 1..{maximum} characters")
    return text


def _write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], *, mode: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_jsonl(rows), encoding="utf-8")
    path.chmod(mode)


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
