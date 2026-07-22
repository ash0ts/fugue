from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fugue.bench.manifest import BenchmarkManifest, TaskSpec

_TASK_IMAGE = (
    "python:3.13.5-slim-bookworm@"
    "sha256:4c2cf9917bd1cbacc5e9b07320025bdb7cdf2df7b0ceaccb55e9dd7e30987419"
)
_SAFE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class WBATransportTaskMaterializer:
    """Build the locked, offline Harbor tasks for the transport ablation."""

    def materialize(
        self,
        manifest: BenchmarkManifest,
        destination: Path,
        source_path: Path,
        *,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        root = (repo_root or Path.cwd()).resolve()
        rows = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != len(manifest.tasks):
            raise ValueError("WBA task source count does not match the manifest")
        selected: list[dict[str, Any]] = []
        for task in manifest.tasks:
            index = task.metadata.get("source_index")
            if not isinstance(index, int) or not 0 <= index < len(rows):
                raise ValueError(f"{task.id}: invalid source index")
            row = _task_row(rows[index], task, root)
            _write_task(destination / task.id, row, root)
            selected.append(
                {
                    "task_id": task.id,
                    "source_index": index,
                    "scenario": row["scenario"],
                    "fixture_sha256": row["fixture_sha256"],
                }
            )
        (destination / "selection.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n"
        )
        return {"tasks": len(selected), "offline": True, "scenarios": 4}


class WBATransportTaskMaterializerV2:
    """Build V2 tasks with a public schema and private pointer assertions."""

    def materialize(
        self,
        manifest: BenchmarkManifest,
        destination: Path,
        source_path: Path,
        *,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        root = (repo_root or Path.cwd()).resolve()
        rows = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != len(manifest.tasks):
            raise ValueError("WBA V2 task source count does not match the manifest")
        selected: list[dict[str, Any]] = []
        for task in manifest.tasks:
            index = task.metadata.get("source_index")
            if not isinstance(index, int) or not 0 <= index < len(rows):
                raise ValueError(f"{task.id}: invalid source index")
            row = _task_row_v2(rows[index], task, root)
            _write_task_v2(destination / task.id, row, root)
            selected.append(
                {
                    "task_id": task.id,
                    "source_index": index,
                    "scenario": row["scenario"],
                    "fixture_sha256": row["fixture_sha256"],
                    "artifact_schema_sha256": _stable_digest(row["artifact_schema"]),
                }
            )
        (destination / "selection.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n"
        )
        return {
            "tasks": len(selected),
            "offline": True,
            "scenarios": 4,
            "evaluation_contract": "public-schema-private-json-pointer-v2",
        }


def _task_row(
    value: Any,
    task: TaskSpec,
    repo_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{task.id}: source row must be an object")
    allowed = {
        "id",
        "title",
        "scenario",
        "fixture_path",
        "fixture_sha256",
        "fixture_repeat",
        "instruction",
        "expected_terms",
        "artifact_path",
        "artifact_json",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{task.id}: unknown source fields: {', '.join(unknown)}")
    if value.get("id") != task.id:
        raise ValueError(f"{task.id}: source identity drift")
    fixture_path = _safe_relative(value.get("fixture_path"), "fixture path")
    fixture = repo_root / fixture_path
    fixture_sha = str(value.get("fixture_sha256") or "")
    if (
        not fixture.is_file()
        or hashlib.sha256(fixture.read_bytes()).hexdigest() != fixture_sha
    ):
        raise ValueError(f"{task.id}: fixture content drift")
    artifact_path = _safe_relative(value.get("artifact_path"), "artifact path")
    terms = value.get("expected_terms")
    artifact = value.get("artifact_json")
    if (
        not isinstance(terms, list)
        or not terms
        or not all(isinstance(item, str) and item.strip() for item in terms)
        or not isinstance(artifact, Mapping)
    ):
        raise ValueError(f"{task.id}: expected evidence is invalid")
    title = str(value.get("title") or "").strip()
    instruction = str(value.get("instruction") or "").strip()
    scenario = str(value.get("scenario") or "").strip()
    if not title or not instruction or not scenario:
        raise ValueError(f"{task.id}: task description is incomplete")
    return {
        **dict(value),
        "fixture_path": fixture_path.as_posix(),
        "artifact_path": artifact_path.as_posix(),
        "fixture_repeat": _repeat_count(value.get("fixture_repeat")),
    }


def _write_task(root: Path, row: Mapping[str, Any], repo_root: Path) -> None:
    for name in ("environment", "solution", "tests"):
        (root / name).mkdir(parents=True, exist_ok=True)
    fixture_source = repo_root / str(row["fixture_path"])
    fixture_name = fixture_source.name
    if not _SAFE_FILE.fullmatch(fixture_name):
        raise ValueError("fixture filename is unsafe")
    resource_dir = root / "environment" / "resources"
    resource_dir.mkdir()
    (resource_dir / fixture_name).write_bytes(
        fixture_source.read_bytes() * int(row["fixture_repeat"])
    )
    (root / "environment" / "Dockerfile").write_text(
        "\n".join(
            [
                f"FROM {_TASK_IMAGE}",
                "WORKDIR /workspace",
                f"COPY resources/{fixture_name} /workspace/resources/{fixture_name}",
            ]
        )
        + "\n"
    )
    (root / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.3"',
                "",
                "[task]",
                f'name = "fugue/{row["id"]}"',
                'description = "Locked WBA transport analysis task"',
                "",
                "[agent]",
                "timeout_sec = 900.0",
                "",
                "[verifier]",
                "timeout_sec = 60.0",
                "",
                "[environment]",
                "build_timeout_sec = 600.0",
                "cpus = 2.0",
                "memory_mb = 4096",
                "storage_mb = 4096",
                "",
            ]
        )
    )
    artifact = f"/logs/artifacts/{row['artifact_path']}"
    instruction = (
        f"# {row['title']}\n\n{row['instruction']}\n\n"
        f"The locked input is `/workspace/resources/{fixture_name}`. Inspect it "
        f"with the shell before answering. Write the requested JSON artifact to "
        f"`{artifact}` and write your evidence-grounded final answer to "
        "`/logs/artifacts/fugue-answer.md`.\n"
    )
    (root / "instruction.md").write_text(instruction)

    reference_answer = "\n".join(str(item) for item in row["expected_terms"]) + "\n"
    reference_artifact = (
        json.dumps(row["artifact_json"], indent=2, sort_keys=True) + "\n"
    )
    (root / "solution" / "reference-answer.md").write_text(reference_answer)
    (root / "solution" / "reference-artifact.json").write_text(reference_artifact)
    (root / "solution" / "solve.sh").write_text(
        "#!/bin/sh\nset -eu\nmkdir -p /logs/artifacts\n"
        "cp /solution/reference-answer.md /logs/artifacts/fugue-answer.md\n"
        f"cp /solution/reference-artifact.json {artifact}\n"
    )
    verifier = _verifier_script(
        tuple(str(item) for item in row["expected_terms"]),
        str(row["artifact_path"]),
        dict(row["artifact_json"]),
    )
    (root / "tests" / "test.sh").write_text(verifier)
    for path in (root / "solution" / "solve.sh", root / "tests" / "test.sh"):
        path.chmod(0o755)


def _task_row_v2(
    value: Any,
    task: TaskSpec,
    repo_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{task.id}: source row must be an object")
    allowed = {
        "id",
        "title",
        "scenario",
        "fixture_path",
        "fixture_sha256",
        "fixture_repeat",
        "instruction",
        "artifact_path",
        "artifact_schema",
        "expected_fields",
        "reference_artifact",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{task.id}: unknown source fields: {', '.join(unknown)}")
    if value.get("id") != task.id:
        raise ValueError(f"{task.id}: source identity drift")
    fixture_path = _safe_relative(value.get("fixture_path"), "fixture path")
    fixture = repo_root / fixture_path
    fixture_sha = str(value.get("fixture_sha256") or "")
    if (
        not fixture.is_file()
        or hashlib.sha256(fixture.read_bytes()).hexdigest() != fixture_sha
    ):
        raise ValueError(f"{task.id}: fixture content drift")
    artifact_path = _safe_relative(value.get("artifact_path"), "artifact path")
    schema = value.get("artifact_schema")
    expected = value.get("expected_fields")
    reference = value.get("reference_artifact")
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ValueError(f"{task.id}: public artifact schema must describe an object")
    if not isinstance(reference, Mapping):
        raise ValueError(f"{task.id}: reference artifact must be an object")
    if not isinstance(expected, list) or not expected:
        raise ValueError(f"{task.id}: private expected fields are required")
    for assertion in expected:
        if not isinstance(assertion, Mapping):
            raise ValueError(f"{task.id}: expected field must be an object")
        if set(assertion) != {"pointer", "value", "comparison"}:
            raise ValueError(f"{task.id}: expected field contract is invalid")
        pointer = assertion.get("pointer")
        comparison = assertion.get("comparison")
        if (
            not isinstance(pointer, str)
            or not pointer.startswith("/")
            or comparison not in {"exact", "unordered"}
            or (
                comparison == "unordered"
                and not isinstance(assertion.get("value"), list)
            )
        ):
            raise ValueError(f"{task.id}: expected field assertion is invalid")
        if _json_pointer(reference, pointer, missing=_MISSING) is _MISSING:
            raise ValueError(
                f"{task.id}: reference artifact does not satisfy {pointer}"
            )
        if isinstance(assertion.get("value"), list):
            field_schema = _schema_at_pointer(schema, pointer)
            semantics = (
                field_schema.get("x-fugue-list-semantics")
                if isinstance(field_schema, Mapping)
                else None
            )
            required_semantics = "set" if comparison == "unordered" else "ordered"
            if semantics != required_semantics:
                raise ValueError(
                    f"{task.id}: {pointer} must declare {required_semantics} list semantics"
                )
    title = str(value.get("title") or "").strip()
    instruction = str(value.get("instruction") or "").strip()
    scenario = str(value.get("scenario") or "").strip()
    if not title or not instruction or not scenario:
        raise ValueError(f"{task.id}: task description is incomplete")
    return {
        **dict(value),
        "fixture_path": fixture_path.as_posix(),
        "artifact_path": artifact_path.as_posix(),
        "fixture_repeat": _repeat_count(value.get("fixture_repeat")),
    }


def _write_task_v2(root: Path, row: Mapping[str, Any], repo_root: Path) -> None:
    for name in ("environment", "solution", "tests"):
        (root / name).mkdir(parents=True, exist_ok=True)
    fixture_source = repo_root / str(row["fixture_path"])
    fixture_name = fixture_source.name
    if not _SAFE_FILE.fullmatch(fixture_name):
        raise ValueError("fixture filename is unsafe")
    resource_dir = root / "environment" / "resources"
    resource_dir.mkdir()
    (resource_dir / fixture_name).write_bytes(
        fixture_source.read_bytes() * int(row["fixture_repeat"])
    )
    (root / "environment" / "Dockerfile").write_text(
        "\n".join(
            [
                f"FROM {_TASK_IMAGE}",
                "WORKDIR /workspace",
                f"COPY resources/{fixture_name} /workspace/resources/{fixture_name}",
            ]
        )
        + "\n"
    )
    (root / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.3"',
                "",
                "[task]",
                f'name = "fugue/{row["id"]}"',
                'description = "Locked WBA transport analysis task V2"',
                "",
                "[agent]",
                "timeout_sec = 900.0",
                "",
                "[verifier]",
                "timeout_sec = 60.0",
                "",
                "[environment]",
                "build_timeout_sec = 600.0",
                "cpus = 2.0",
                "memory_mb = 4096",
                "storage_mb = 4096",
                "",
            ]
        )
    )
    artifact = f"/logs/artifacts/{row['artifact_path']}"
    public_schema = json.dumps(row["artifact_schema"], indent=2, sort_keys=True)
    instruction = (
        f"# {row['title']}\n\n{row['instruction']}\n\n"
        f"The locked input is `/workspace/resources/{fixture_name}`. Inspect it "
        f"with the shell before answering. Write a JSON artifact to `{artifact}` "
        "that validates against this public schema:\n\n"
        f"```json\n{public_schema}\n```\n\n"
        "For arrays, `x-fugue-list-semantics` states whether order matters. "
        "You may add explanatory fields unless the schema says otherwise. Write "
        "a non-empty evidence-grounded final answer to "
        "`/logs/artifacts/fugue-answer.md`.\n"
    )
    (root / "instruction.md").write_text(instruction)
    (root / "solution" / "reference-answer.md").write_text(
        "The requested findings are recorded in the validated artifact.\n"
    )
    (root / "solution" / "reference-artifact.json").write_text(
        json.dumps(row["reference_artifact"], indent=2, sort_keys=True) + "\n"
    )
    (root / "solution" / "solve.sh").write_text(
        "#!/bin/sh\nset -eu\nmkdir -p /logs/artifacts\n"
        "cp /solution/reference-answer.md /logs/artifacts/fugue-answer.md\n"
        f"cp /solution/reference-artifact.json {artifact}\n"
    )
    (root / "tests" / "test.sh").write_text(
        _verifier_script_v2(
            str(row["artifact_path"]),
            dict(row["artifact_schema"]),
            list(row["expected_fields"]),
        )
    )
    for path in (root / "solution" / "solve.sh", root / "tests" / "test.sh"):
        path.chmod(0o755)


def _verifier_script(
    expected_terms: tuple[str, ...],
    artifact_path: str,
    expected_artifact: dict[str, Any],
) -> str:
    terms_json = repr(json.dumps(expected_terms, separators=(",", ":")))
    artifact_json = repr(
        json.dumps(expected_artifact, sort_keys=True, separators=(",", ":"))
    )
    return (
        "#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\npython - <<'PY'\n"
        "import json\nfrom pathlib import Path\n"
        f"terms=json.loads({terms_json})\n"
        f"expected=json.loads({artifact_json})\n"
        "answer_path=Path('/logs/artifacts/fugue-answer.md')\n"
        f"artifact_path=Path('/logs/artifacts/{artifact_path}')\n"
        "answer=answer_path.read_text().casefold() if answer_path.is_file() else ''\n"
        "answer_ok=all(term.casefold() in answer for term in terms)\n"
        "try:\n"
        "    artifact=json.loads(artifact_path.read_text())\n"
        "except (OSError, json.JSONDecodeError):\n"
        "    artifact={}\n"
        "artifact_ok=all(artifact.get(key)==value for key,value in expected.items())\n"
        "score=float(answer_ok and artifact_ok)\n"
        "Path('/logs/verifier/reward.json').write_text(json.dumps({"
        "'reward':score,'task_pass':score,'answer_facts':float(answer_ok),"
        "'artifact_contract':float(artifact_ok)}))\n"
        "raise SystemExit(0 if score else 1)\nPY\n"
    )


def _verifier_script_v2(
    artifact_path: str,
    artifact_schema: dict[str, Any],
    expected_fields: list[Mapping[str, Any]],
) -> str:
    schema_json = repr(json.dumps(artifact_schema, separators=(",", ":")))
    expected_json = repr(json.dumps(expected_fields, separators=(",", ":")))
    return (
        "#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\npython - <<'PY'\n"
        "import json\nfrom pathlib import Path\n"
        f"schema=json.loads({schema_json})\n"
        f"expected=json.loads({expected_json})\n"
        "answer_path=Path('/logs/artifacts/fugue-answer.md')\n"
        f"artifact_path=Path('/logs/artifacts/{artifact_path}')\n"
        "answer_ok=answer_path.is_file() and bool(answer_path.read_text().strip())\n"
        "def valid(value,schema):\n"
        "    kind=schema.get('type')\n"
        "    checks={'object':lambda v:isinstance(v,dict),'array':lambda v:isinstance(v,list),"
        "'string':lambda v:isinstance(v,str),'integer':lambda v:isinstance(v,int) and not isinstance(v,bool),"
        "'number':lambda v:isinstance(v,(int,float)) and not isinstance(v,bool),'boolean':lambda v:isinstance(v,bool),"
        "'null':lambda v:v is None}\n"
        "    if kind in checks and not checks[kind](value): return False\n"
        "    if 'enum' in schema and value not in schema['enum']: return False\n"
        "    if isinstance(value,dict):\n"
        "        required=schema.get('required',[])\n"
        "        if any(key not in value for key in required): return False\n"
        "        properties=schema.get('properties',{})\n"
        "        if schema.get('additionalProperties') is False and any(key not in properties for key in value): return False\n"
        "        if any(key in value and not valid(value[key],child) for key,child in properties.items()): return False\n"
        "    if isinstance(value,list):\n"
        "        if len(value)<schema.get('minItems',0): return False\n"
        "        if 'maxItems' in schema and len(value)>schema['maxItems']: return False\n"
        "        if schema.get('uniqueItems') and len({json.dumps(item,sort_keys=True) for item in value})!=len(value): return False\n"
        "        item_schema=schema.get('items')\n"
        "        if item_schema and any(not valid(item,item_schema) for item in value): return False\n"
        "    return True\n"
        "def pointer(value,path):\n"
        "    current=value\n"
        "    for raw in path.split('/')[1:]:\n"
        "        part=raw.replace('~1','/').replace('~0','~')\n"
        "        try: current=current[int(part)] if isinstance(current,list) else current[part]\n"
        "        except (KeyError,IndexError,TypeError,ValueError): return missing\n"
        "    return current\n"
        "missing=object()\n"
        "try:\n"
        "    artifact=json.loads(artifact_path.read_text())\n"
        "except (OSError,json.JSONDecodeError):\n"
        "    artifact=missing\n"
        "schema_ok=artifact is not missing and valid(artifact,schema)\n"
        "facts_ok=schema_ok\n"
        "if facts_ok:\n"
        "    for item in expected:\n"
        "        observed=pointer(artifact,item['pointer'])\n"
        "        wanted=item['value']\n"
        "        if item['comparison']=='unordered' and observed is not missing:\n"
        "            observed=sorted(json.dumps(v,sort_keys=True) for v in observed) if isinstance(observed,list) else observed\n"
        "            wanted=sorted(json.dumps(v,sort_keys=True) for v in wanted)\n"
        "        if observed is missing or observed!=wanted: facts_ok=False; break\n"
        "score=float(answer_ok and schema_ok and facts_ok)\n"
        "Path('/logs/verifier/reward.json').write_text(json.dumps({"
        "'reward':score,'task_pass':score,'answer_present':float(answer_ok),"
        "'artifact_schema':float(schema_ok),'artifact_facts':float(facts_ok)}))\n"
        "raise SystemExit(0 if score else 1)\nPY\n"
    )


def _safe_relative(value: Any, label: str) -> Path:
    path = Path(str(value or ""))
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be repository-relative")
    return path


def _repeat_count(value: Any) -> int:
    count = 1 if value is None else value
    if not isinstance(count, int) or not 1 <= count <= 100:
        raise ValueError("fixture_repeat must be an integer between 1 and 100")
    return count


_MISSING = object()


def _json_pointer(value: Any, pointer: str, *, missing: Any) -> Any:
    current = value
    for raw in pointer.split("/")[1:]:
        part = raw.replace("~1", "/").replace("~0", "~")
        try:
            current = current[int(part)] if isinstance(current, list) else current[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return missing
    return current


def _schema_at_pointer(schema: Mapping[str, Any], pointer: str) -> Any:
    current: Any = schema
    for raw in pointer.split("/")[1:]:
        part = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping):
            return _MISSING
        if current.get("type") == "object":
            current = (current.get("properties") or {}).get(part, _MISSING)
        elif current.get("type") == "array":
            current = current.get("items", _MISSING)
        else:
            return _MISSING
        if current is _MISSING:
            return _MISSING
    return current


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
