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
