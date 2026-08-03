"""Prepare and preflight the dependency-free Vercel confirmation fixtures."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import runpy
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = EXAMPLE_ROOT / "conference_fixture_catalog.py"
SOURCE_LOCK = EXAMPLE_ROOT / "confirmatory-fixtures.lock.json"
TASKS_PATH = EXAMPLE_ROOT / "confirmatory-tasks.jsonl"
PRIVATE_PATH = EXAMPLE_ROOT / "confirmatory-private-labels.jsonl"
SCORER_PATH = EXAMPLE_ROOT / "vercel_confirmatory_scorer.py"
OUTPUT_ROOT = (
    REPO_ROOT
    / ".fugue/comparison-resources/vercel-react-best-practices-confirmatory-v1"
)
NODE_IMAGE = (
    "node@sha256:53ada149d435c38b14476cb57e4a7da73c15595aba79bd6971b547ceb6d018bf"
)


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_catalog() -> tuple[dict[str, Any], ...]:
    spec = importlib.util.spec_from_file_location("vercel_confirmatory_catalog", CATALOG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load fixture catalog")
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    spec.loader.exec_module(module)
    fixtures = tuple(module.FIXTURES)
    if len(fixtures) != 24 or len({item["id"] for item in fixtures}) != 24:
        raise RuntimeError("confirmatory catalog must contain 24 unique fixtures")
    if sum(item["split"] == "discovery" for item in fixtures) != 8:
        raise RuntimeError("confirmatory catalog must contain eight development fixtures")
    if sum(item["split"] == "holdout" for item in fixtures) != 16:
        raise RuntimeError("confirmatory catalog must contain sixteen holdouts")
    return fixtures


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_and_verify_source_lock() -> dict[str, Any]:
    value = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    supplied = str(value.pop("manifest_digest", ""))
    if value.get("schema_version") != 1 or supplied != _stable_digest(value):
        raise RuntimeError("confirmatory fixture lock digest does not match")
    records = value.get("sources")
    if not isinstance(records, list) or not records:
        raise RuntimeError("confirmatory fixture lock is empty")
    for record in records:
        relative = str(record.get("path") or "")
        path = EXAMPLE_ROOT / relative
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not path.is_file()
            or path.is_symlink()
        ):
            raise RuntimeError(f"unsafe confirmatory source path: {relative}")
        if path.stat().st_size != record.get("size") or _sha256(path) != record.get(
            "sha256"
        ):
            raise RuntimeError(f"confirmatory source changed: {relative}")
    return {**value, "manifest_digest": supplied}


def _hidden_test_source(fixture: dict[str, Any]) -> str:
    target_paths = list(fixture["target_files"])
    contract = json.dumps(fixture["verifier"], sort_keys=True)
    paths = json.dumps(target_paths)
    task_id = str(fixture["id"])
    return f"""import test from 'node:test';
import assert from 'node:assert/strict';
import {{ readFileSync }} from 'node:fs';

const contract = {contract};
const paths = {paths};
const sources = Object.fromEntries(paths.map((path) => [path, readFileSync(new URL(`../${{path}}`, import.meta.url), 'utf8')]));
const text = (path) => sources[path] || '';
const containsAll = (source, values) => (values || []).every((value) => source.includes(value));
const containsNone = (source, values) => (values || []).every((value) => !source.includes(value));

test('host-only verifier for {task_id}', () => {{
  if (contract.kind === 'server_action') {{
    const source = text(contract.source_path);
    assert.ok(source.includes(`export async function ${{contract.export_name}}`));
    if (contract.mode === 'read_only_control') {{
      assert.ok(source.includes(contract.read_call));
      assert.ok(containsNone(source, contract.forbidden_calls));
      assert.ok(containsAll(source, contract.validation_terms));
      return;
    }}
    const auth = source.indexOf(contract.auth_call);
    const authorization = source.indexOf(contract.authorization_call);
    const mutation = source.indexOf(contract.mutation_call);
    assert.ok(auth >= 0 && auth < authorization && authorization < mutation);
    assert.ok(source.indexOf('throw', auth) < authorization);
    assert.ok(source.indexOf('throw', authorization) < mutation);
    assert.ok(containsAll(source, contract.authorization_terms));
    assert.ok(containsAll(source, contract.validation_terms));
    if (contract.validation_before_auth) {{
      assert.ok(contract.validation_terms.every((term) => source.indexOf(term) < auth));
    }}
    return;
  }}
  if (contract.kind === 'rsc_props') {{
    const server = text(contract.server_path);
    const client = text(contract.client_path);
    const compact = server.split(/\\s+/).join('');
    if (contract.mode === 'canonical_only') {{
      assert.ok(compact.includes(`return{{${{contract.canonical_prop}}}};`));
      assert.ok(containsNone(server, contract.derived_props));
      assert.ok(containsAll(client, contract.client_terms));
    }} else {{
      assert.ok(compact.includes(`return{{${{contract.canonical_prop}}}};`));
      assert.ok(containsAll(server, contract.server_terms));
      assert.ok(containsAll(client, contract.client_terms));
    }}
    return;
  }}
  const source = text(contract.source_path);
  if (contract.kind === 'dom_batch') {{
    const writes = contract.write_terms.map((term) => source.indexOf(term));
    const reads = contract.read_terms.map((term) => source.indexOf(term));
    assert.ok(Math.max(...writes) < Math.min(...reads));
    assert.ok(containsNone(source, contract.forbidden_read_terms));
    return;
  }}
  if (contract.kind === 'dom_write_control') {{
    assert.ok(containsAll(source, contract.write_terms));
    assert.ok(containsNone(source, contract.forbidden_read_terms));
    return;
  }}
  if (contract.kind === 'array_extreme') {{
    assert.ok(source.includes('values.length === 0') && source.includes('return null'));
    assert.ok(source.includes('for (') || source.includes('.reduce('));
    assert.ok(!source.includes('.sort('));
    assert.ok(!source.includes(contract.mode === 'max' ? 'Math.max(...' : 'Math.min(...'));
    return;
  }}
  if (contract.kind === 'array_sum_control') {{
    assert.ok(source.split(/\\s+/).join('').includes('.reduce((total,value)=>total+value,0)'));
    return;
  }}
  if (contract.kind === 'hook_timing') {{
    assert.ok(source.includes(`hooks.${{contract.effect}}(`));
    assert.ok(source.includes('ref.current = value'));
    if (contract.forbid_layout) assert.ok(!source.includes('useLayoutEffect'));
    return;
  }}
  if (contract.kind === 'event_signature') {{
    assert.ok(source.includes(`@param {{${{contract.event_type}}}}`));
    assert.ok(source.includes(`event.${{contract.property}}`));
    return;
  }}
  assert.fail(`unknown hidden verifier: ${{contract.kind}}`);
}});
"""


def _public_files(
    fixture: dict[str, Any], *, gold: bool, include_hidden: bool = False
) -> dict[str, bytes]:
    task_id = str(fixture["id"])
    files: dict[str, bytes] = {
        "package.json": (
            json.dumps(
                {
                    "name": task_id,
                    "private": True,
                    "type": "module",
                    "scripts": {"test": "node --test"},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "README.md": (
            f"# {fixture['title']}\n\n"
            f"Reported symptom: {fixture['symptom']}\n\n"
            "Repair the behavior without adding dependencies or changing unrelated files. "
            "Run `node --test` before returning the complete changed source files.\n"
        ).encode(),
        "tests/task.test.mjs": str(fixture["public_test_source"]).encode(),
    }
    side = "gold" if gold else "base"
    for path, sources in fixture["target_files"].items():
        files[str(path)] = str(sources[side]).encode()
    if include_hidden:
        files["tests/host-only-verifier.test.mjs"] = _hidden_test_source(
            fixture
        ).encode()
    return files


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for relative, data in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _node_runtime_receipt() -> dict[str, str]:
    node = shutil.which("node")
    if node is not None:
        version = subprocess.run(
            [node, "--version"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        return {
            "mode": "host_binary",
            "identity": f"sha256:{_sha256(Path(node).resolve())}",
            "version": version,
        }
    if shutil.which("docker") is not None:
        return {"mode": "docker", "identity": NODE_IMAGE, "version": "node-22"}
    raise RuntimeError("Node.js or Docker is required for fixture preflight")


def _run_public_tests(root: Path) -> subprocess.CompletedProcess[str]:
    runtime = _node_runtime_receipt()
    if runtime["mode"] == "host_binary":
        node = shutil.which("node")
        assert node is not None
        command = [node, "--test"]
        cwd = root
    else:
        docker = shutil.which("docker")
        assert docker is not None
        command = [
            docker,
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--cpus",
            "1",
            "--memory",
            "256m",
            "--mount",
            f"type=bind,src={root.resolve()},dst=/workspace,readonly",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--workdir",
            "/workspace",
            NODE_IMAGE,
            "node",
            "--test",
        ]
        cwd = None
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={"PATH": str(Path(shutil.which("docker") or command[0]).parent)},
    )


def _validate_node_test_receipt(
    fixture: dict[str, Any],
    completed: subprocess.CompletedProcess[str],
    *,
    should_pass: bool,
) -> None:
    detail = completed.stdout + completed.stderr
    public_name = str(fixture["public_test_name"])
    hidden_name = f"host-only verifier for {fixture['id']}"
    common = bool(
        "TAP version 13" in detail
        and public_name in detail
        and hidden_name in detail
        and re.search(r"# tests\s+2\b", detail)
    )
    if should_pass:
        valid = bool(
            common
            and completed.returncode == 0
            and re.search(r"# pass\s+2\b", detail)
            and re.search(r"# fail\s+0\b", detail)
        )
        expectation = "passing"
    else:
        valid = bool(
            common
            and completed.returncode == 1
            and re.search(r"# fail\s+[1-9]\d*\b", detail)
        )
        expectation = "failing"
    if not valid:
        raise RuntimeError(
            f"{expectation} Node test receipt was not produced for "
            f"{fixture['id']} (exit {completed.returncode})\n{detail[-4000:]}"
        )


def _output(
    fixture: dict[str, Any],
    *,
    gold: bool,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    side = "gold" if gold else "base"
    return {
        "schema_version": 1,
        "task_id": fixture["id"],
        "status": "completed",
        "files": {
            path: sources[side] for path, sources in fixture["target_files"].items()
        },
        "inspected_paths": fixture["required_inspected_paths"],
        "verification": [
            {
                "command": "node --test",
                "exit_code": completed.returncode,
                "stdout": completed.stdout + completed.stderr,
            }
        ],
        "summary": (
            "Frozen gold fixture after host-side semantic and executable verification."
            if gold
            else "Frozen base fixture before the requested repair was applied."
        ),
        "limitations": ["This receipt is generated outside Agent execution."],
    }


def _preflight(fixture: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    scorer = runpy.run_path(SCORER_PATH.as_posix())["score"]
    with tempfile.TemporaryDirectory(prefix=f"fugue-vercel-{fixture['id']}-") as value:
        root = Path(value)
        base_root = root / "base"
        gold_root = root / "gold"
        _write_tree(base_root, _public_files(fixture, gold=False, include_hidden=True))
        _write_tree(gold_root, _public_files(fixture, gold=True, include_hidden=True))
        base_run = _run_public_tests(base_root)
        gold_run = _run_public_tests(gold_root)
        _validate_node_test_receipt(fixture, base_run, should_pass=False)
        _validate_node_test_receipt(fixture, gold_run, should_pass=True)
        base_scores = scorer(
            {"id": fixture["id"]},
            _output(fixture, gold=False, completed=base_run),
            {"expected": expected},
        )
        gold_scores = scorer(
            {"id": fixture["id"]},
            _output(fixture, gold=True, completed=gold_run),
            {"expected": expected},
        )
    if all(base_scores.values()):
        raise RuntimeError(f"base scorer unexpectedly passed: {fixture['id']}")
    if not all(gold_scores.values()):
        raise RuntimeError(f"gold scorer failed: {fixture['id']}: {gold_scores}")
    return {
        "task_id": fixture["id"],
        "split": fixture["split"],
        "base_public_test_exit_code": base_run.returncode,
        "gold_public_test_exit_code": gold_run.returncode,
        "base_scores": base_scores,
        "gold_scores": gold_scores,
        "base_tree_digest": _stable_digest(
            {key: hashlib.sha256(data).hexdigest() for key, data in _public_files(fixture, gold=False).items()}
        ),
        "gold_tree_digest": _stable_digest(
            {key: hashlib.sha256(data).hexdigest() for key, data in _public_files(fixture, gold=True).items()}
        ),
        "hidden_verifier_sha256": hashlib.sha256(
            _hidden_test_source(fixture).encode()
        ).hexdigest(),
    }


def _build_archive(fixture: dict[str, Any], destination: Path) -> dict[str, Any]:
    files = _public_files(fixture, gold=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w", format=tarfile.USTAR_FORMAT) as archive:
        for relative in sorted(files):
            data = files[relative]
            info = tarfile.TarInfo(f"repo/{relative}")
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            archive.addfile(info, io.BytesIO(data))
    try:
        archive_path = destination.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        archive_path = destination.as_posix()
    return {
        "id": fixture["id"],
        "archive": archive_path,
        "sha256": _sha256(destination),
        "file_count": len(files),
    }


def prepare(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    source_lock = _load_and_verify_source_lock()
    fixtures = _load_catalog()
    tasks = _load_jsonl(TASKS_PATH)
    labels = _load_jsonl(PRIVATE_PATH)
    expected_by_id = {str(item["id"]): item["expected"] for item in labels}
    fixture_ids = [str(item["id"]) for item in fixtures]
    if [str(item["id"]) for item in tasks] != fixture_ids:
        raise RuntimeError("public task order differs from the frozen fixture catalog")
    if [str(item["id"]) for item in labels] != fixture_ids:
        raise RuntimeError("private-label order differs from the frozen fixture catalog")
    receipts = [_preflight(fixture, expected_by_id[str(fixture["id"])]) for fixture in fixtures]
    archives = [
        _build_archive(fixture, output_root / f"{fixture['id']}.tar")
        for fixture in fixtures
    ]
    receipt = {
        "schema_version": 1,
        "source_lock_digest": source_lock["manifest_digest"],
        "preflight_runtime": _node_runtime_receipt(),
        "tasks": receipts,
    }
    receipt["receipt_digest"] = _stable_digest(receipt)
    manifest = {
        "schema_version": 1,
        "source_lock_digest": source_lock["manifest_digest"],
        "preflight_receipt_digest": receipt["receipt_digest"],
        "archives": archives,
    }
    manifest["manifest_digest"] = _stable_digest(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "fixtures.lock.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    print(json.dumps(prepare(), sort_keys=True))


if __name__ == "__main__":
    main()
