from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import toml
from filelock import FileLock

from fugue.bench.files import atomic_write_json
from fugue.bench.files import inspect_docker_image as _inspect_image
from fugue.bench.manifest import (
    BenchmarkManifest,
    DatasetVerifierRuntimeSpec,
    TaskSpec,
    VerifierRuntimeSpec,
)

TASK_RUNTIME_ROOT = Path(".fugue/runtime/task-images")
TASK_RUNTIME_CONTRACT_VERSION = 9

VerifierRuntime = DatasetVerifierRuntimeSpec | VerifierRuntimeSpec


def prepare_task_runtime(
    manifest: BenchmarkManifest,
    task: TaskSpec,
    *,
    repo_root: Path,
    rebuild: bool = False,
    gold_patch: str | None = None,
) -> dict[str, Any]:
    """Build a task once and publish a local dataset pinned to its image ID."""
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required to prepare task images")
    requires_gold_verification = _requires_gold_verification(manifest)
    if requires_gold_verification and not (gold_patch or "").strip():
        raise RuntimeError(
            f"task {task.id} requires a pinned gold patch for setup verification"
        )
    gold_patch_sha256 = (
        hashlib.sha256(gold_patch.encode()).hexdigest() if gold_patch else None
    )
    architecture = task_architecture(task)
    verifier_runtime = _verifier_runtime(manifest, task)
    root = _lock_root(manifest, task, repo_root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / f".prepare-{architecture}.lock", timeout=3600):
        source = _resolve_task_source(manifest, task, repo_root)
        source_digest = _tree_digest(source)
        dockerfile = source / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            raise RuntimeError(f"task {task.id} has no environment/Dockerfile")
        recipe_sha256 = _digest(
            {
                "contract_version": TASK_RUNTIME_CONTRACT_VERSION,
                "dataset": manifest.dataset.harbor_ref,
                "task_id": task.id,
                "architecture": architecture,
                "source_sha256": source_digest,
                "verifier_runtime": (
                    verifier_runtime.to_dict() if verifier_runtime else None
                ),
                "gold_patch_sha256": gold_patch_sha256,
                "trial_policy": {
                    "network_mode": "no-network",
                    "allowed_hosts": [],
                    "image_pull_policy": "never",
                },
            }
        )
        existing = read_task_runtime_lock(manifest, task, repo_root)
        if not rebuild and existing and existing.get("recipe_sha256") == recipe_sha256:
            ready, _ = task_runtime_ready(manifest, task, repo_root)
            if ready:
                return existing

        image = f"fugue-task-{_slug(task.id)}-{architecture}:{recipe_sha256[:12]}"
        with tempfile.TemporaryDirectory(prefix="fugue-task-build-") as temporary_build:
            build = Path(temporary_build) / "environment"
            shutil.copytree(source / "environment", build, symlinks=True)
            _extend_task_image(build / "Dockerfile", verifier_runtime)
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--provenance=false",
                    "--platform",
                    f"linux/{architecture}",
                    "--pull",
                    "-t",
                    image,
                    build.as_posix(),
                ],
                cwd=repo_root,
                check=True,
                timeout=3600,
            )
        inspected = _inspect_image(image)
        if inspected.get("Architecture") != architecture:
            raise RuntimeError(
                f"task {task.id} built for {inspected.get('Architecture') or 'unknown'}, "
                f"expected {architecture}"
            )
        _verify_task_image(image, verifier_runtime, architecture, repo_root)

        dataset_root = root / f"dataset-{recipe_sha256[:16]}"
        temporary = root / f".dataset-{uuid.uuid4().hex}.tmp"
        task_root = temporary / task.id
        shutil.copytree(source, task_root, symlinks=True)
        _reject_escaping_symlinks(task_root)
        verifier_script = _lock_verifier_script(task_root, task, verifier_runtime)
        verification = (
            _verify_swe_bench_outcomes(
                image,
                task,
                task_root,
                gold_patch=gold_patch or "",
                gold_patch_sha256=gold_patch_sha256 or "",
                architecture=architecture,
                repo_root=repo_root,
            )
            if requires_gold_verification
            else None
        )
        task_toml = task_root / "task.toml"
        value = toml.loads(task_toml.read_text())
        environment = value.setdefault("environment", {})
        environment.pop("allow_internet", None)
        environment["docker_image"] = str(inspected["Id"])
        environment["network_mode"] = "no-network"
        environment["allowed_hosts"] = []
        task_toml.write_text(toml.dumps(value))
        prepared_source_sha256 = _tree_digest(task_root)
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        os.replace(temporary, dataset_root)

        lock = {
            "schema_version": 1,
            "contract_version": TASK_RUNTIME_CONTRACT_VERSION,
            "dataset": manifest.dataset.harbor_ref,
            "task_id": task.id,
            "architecture": architecture,
            "source_sha256": source_digest,
            "prepared_source_sha256": prepared_source_sha256,
            "recipe_sha256": recipe_sha256,
            "image": image,
            "image_id": str(inspected["Id"]),
            "os": str(inspected.get("Os") or "linux"),
            "dataset_path": dataset_root.as_posix(),
            "verifier_runtime": (
                verifier_runtime.to_dict() if verifier_runtime else None
            ),
            "verifier_script": verifier_script,
            "verification": verification,
        }
        atomic_write_json(root / f"runtime-lock-{architecture}.json", lock)
        return lock


def read_task_runtime_lock(
    manifest: BenchmarkManifest,
    task: TaskSpec,
    repo_root: Path,
) -> dict[str, Any] | None:
    architecture = task_architecture(task)
    verifier_runtime = _verifier_runtime(manifest, task)
    path = _lock_root(manifest, task, repo_root) / f"runtime-lock-{architecture}.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    required = {
        "schema_version": 1,
        "contract_version": TASK_RUNTIME_CONTRACT_VERSION,
        "dataset": manifest.dataset.harbor_ref,
        "task_id": task.id,
        "architecture": architecture,
        "verifier_runtime": (verifier_runtime.to_dict() if verifier_runtime else None),
    }
    if not isinstance(value, dict) or any(
        value.get(key) != item for key, item in required.items()
    ):
        return None
    if _requires_gold_verification(manifest):
        verification = value.get("verification") or {}
        if not (
            verification.get("base_failed") is True
            and verification.get("gold_passed") is True
            and _is_sha256(str(verification.get("gold_patch_sha256") or ""))
        ):
            return None
    image_id = str(value.get("image_id") or "")
    dataset_path = Path(str(value.get("dataset_path") or ""))
    if not image_id.startswith("sha256:") or not dataset_path.is_dir():
        return None
    return value


def task_runtime_ready(
    manifest: BenchmarkManifest,
    task: TaskSpec,
    repo_root: Path,
) -> tuple[bool, str]:
    lock = read_task_runtime_lock(manifest, task, repo_root)
    if lock is None:
        return False, "run fugue setup --prepare to build the locked task image"
    try:
        inspected = _inspect_image(str(lock["image_id"]))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return False, f"prepared task image is unavailable: {exc}"
    if inspected.get("Id") != lock.get("image_id"):
        return False, "prepared task image drifted from its lock"
    return True, f"{lock['image_id']} is available"


def task_architecture(task: TaskSpec) -> str:
    architecture = str(task.metadata.get("architecture") or "amd64")
    if architecture not in {"amd64", "arm64"}:
        raise ValueError(
            f"task {task.id} has unsupported architecture {architecture!r}"
        )
    return architecture


def _extend_task_image(dockerfile: Path, runtime: VerifierRuntime | None) -> None:
    if runtime is None:
        return
    source = dockerfile.read_text()
    packages = " ".join(shlex.quote(package) for package in runtime.python_packages)
    interpreter = _runtime_python(runtime)
    if isinstance(runtime, DatasetVerifierRuntimeSpec):
        install = (
            "RUN /opt/miniconda3/bin/python -m venv "
            f"{shlex.quote(Path(interpreter).parent.parent.as_posix())} && "
            f"{shlex.quote(interpreter)} -m pip install --no-cache-dir {packages}"
        )
    else:
        install = (
            f"RUN {shlex.quote(interpreter)} -m pip install "
            f"--break-system-packages --no-cache-dir {packages}"
        )
    dockerfile.write_text(
        source.rstrip()
        + "\n\n# The verifier runs offline; setup owns this dependency layer.\n"
        + install
        + "\n"
    )


def _verify_task_image(
    image: str,
    runtime: VerifierRuntime | None,
    architecture: str,
    repo_root: Path,
) -> None:
    if runtime is None:
        return
    expected = {
        package.rsplit("==", 1)[0]: package.rsplit("==", 1)[1]
        for package in runtime.python_packages
    }
    script = (
        "import importlib.metadata,json,sys;"
        f"expected=json.loads({json.dumps(json.dumps(expected))});"
        "actual={name:importlib.metadata.version(name) for name in expected};"
        "sys.exit(0 if actual==expected else 1)"
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--platform",
            f"linux/{architecture}",
            image,
            _runtime_python(runtime),
            "-c",
            script,
        ],
        cwd=repo_root,
        check=True,
        timeout=120,
    )


def _verify_swe_bench_outcomes(
    image: str,
    task: TaskSpec,
    task_root: Path,
    *,
    gold_patch: str,
    gold_patch_sha256: str,
    architecture: str,
    repo_root: Path,
) -> dict[str, Any]:
    if not _is_sha256(gold_patch_sha256):
        raise RuntimeError(f"task {task.id} has an invalid gold patch digest")
    runtime_root = repo_root / ".fugue" / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="fugue-task-verify-", dir=runtime_root
    ) as temporary:
        root = Path(temporary)
        patch_path = root / "gold.patch"
        patch_path.write_text(gold_patch)
        patch_path.chmod(0o600)
        base_resolved = _run_swe_bench_verifier(
            image,
            task,
            task_root,
            architecture=architecture,
            logs=root / "base-logs",
            gold_patch_path=None,
            repo_root=repo_root,
        )
        if base_resolved:
            raise RuntimeError(
                f"task {task.id} base checkout unexpectedly passes its verifier"
            )
        gold_resolved = _run_swe_bench_verifier(
            image,
            task,
            task_root,
            architecture=architecture,
            logs=root / "gold-logs",
            gold_patch_path=patch_path,
            repo_root=repo_root,
        )
        if not gold_resolved:
            raise RuntimeError(f"task {task.id} gold patch does not pass its verifier")
    return {
        "base_failed": True,
        "gold_passed": True,
        "gold_patch_sha256": gold_patch_sha256,
    }


def _run_swe_bench_verifier(
    image: str,
    task: TaskSpec,
    task_root: Path,
    *,
    architecture: str,
    logs: Path,
    gold_patch_path: Path | None,
    repo_root: Path,
) -> bool:
    (logs / "verifier").mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--platform",
        f"linux/{architecture}",
        "--mount",
        f"type=bind,source={(task_root / 'tests').resolve()},target=/tests,readonly",
        "--mount",
        f"type=bind,source={logs.resolve()},target=/logs",
    ]
    shell = "bash /tests/test.sh"
    if gold_patch_path is not None:
        command.extend(
            (
                "--mount",
                f"type=bind,source={gold_patch_path.resolve()},target=/fugue-gold.patch,readonly",
            )
        )
        shell = (
            "git -C /testbed apply --check /fugue-gold.patch && "
            "git -C /testbed apply /fugue-gold.patch && bash /tests/test.sh"
        )
    command.extend((image, "bash", "-lc", shell))
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    report_path = logs / "verifier" / "report.json"
    try:
        report = json.loads(report_path.read_text())
        resolved = (report.get(task.id) or {}).get("resolved")
    except (OSError, json.JSONDecodeError, AttributeError):
        resolved = None
    expected_returncode = 0 if resolved is True else 1 if resolved is False else None
    if expected_returncode is None or result.returncode != expected_returncode:
        detail = (result.stderr or result.stdout or "verifier report missing").strip()
        raise RuntimeError(
            f"task {task.id} verifier did not produce a trustworthy outcome: "
            f"{detail[-1_000:]}"
        )
    return bool(resolved)


def _lock_verifier_script(
    task_root: Path,
    task: TaskSpec,
    runtime: VerifierRuntime | None,
) -> dict[str, str] | None:
    if runtime is None:
        return None
    script = task_root / "tests" / "test.sh"
    if not script.is_file():
        raise RuntimeError(f"task {task.id} has no tests/test.sh to lock")
    source = script.read_text()
    original_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if isinstance(runtime, DatasetVerifierRuntimeSpec):
        rewritten = _rewrite_dataset_verifier(source, task, runtime)
    else:
        rewritten = _rewrite_task_verifier(source, task, runtime, original_sha256)
    _reject_trial_resolvers(rewritten, task)
    script.write_text(rewritten)
    return {
        "original_sha256": original_sha256,
        "prepared_sha256": hashlib.sha256(rewritten.encode()).hexdigest(),
    }


def _rewrite_task_verifier(
    source: str,
    task: TaskSpec,
    runtime: VerifierRuntimeSpec,
    original_sha256: str,
) -> str:
    if original_sha256 != runtime.test_script_sha256:
        raise RuntimeError(
            f"task {task.id} verifier script changed: expected "
            f"{runtime.test_script_sha256}, got {original_sha256}"
        )
    lines = source.splitlines(keepends=True)
    matches: list[tuple[int, int, list[str]]] = []
    for index, line in enumerate(lines):
        if not line.lstrip().startswith(("pip install ", "pip3 install ")):
            continue
        end = index + 1
        while end < len(lines) and lines[end - 1].rstrip().endswith("\\"):
            end += 1
        command = "".join(lines[index:end]).replace("\\\n", " ")
        tokens = shlex.split(command)
        packages = [token for token in tokens[2:] if not token.startswith("-")]
        matches.append((index, end, packages))
    expected = list(runtime.python_packages)
    selected = [match for match in matches if match[2] == expected]
    if len(selected) != 1:
        raise RuntimeError(
            f"task {task.id} verifier must contain one exact locked install block"
        )
    start, end, _ = selected[0]
    lines[start:end] = [
        "# Setup locked these verifier dependencies into the task image.\n"
    ]
    return "".join(lines)


def _rewrite_dataset_verifier(
    source: str,
    task: TaskSpec,
    runtime: DatasetVerifierRuntimeSpec,
) -> str:
    if runtime.profile != "swebench-v4-offline":
        raise RuntimeError(f"unsupported dataset verifier profile: {runtime.profile}")
    # SWE-bench images install some projects with their local test extra. Setup has
    # already executed that exact local install; the trial must not repeat it.
    install_pattern = re.compile(
        r"(?m)^[ \t]*python -m pip install -e "
        r"\.(?:\[[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*\])?"
        r"(?:[ \t]+--verbose)?[ \t]*$"
    )
    parser_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)uv run parser\.py(?P<suffix>[ \t]*(?:\|.*)?)$"
    )
    rewritten, installs = install_pattern.subn(
        "        # Setup prepared the task environment before the trial.", source
    )
    if installs > 1:
        raise RuntimeError(
            f"task {task.id} verifier contains multiple local editable installs"
        )
    interpreter = shlex.quote(runtime.python_interpreter)
    rewritten, parser_runs = parser_pattern.subn(
        rf"\g<indent>{interpreter} parser.py\g<suffix>", rewritten
    )
    if parser_runs != 1:
        raise RuntimeError(
            f"task {task.id} verifier must contain one exact uv parser invocation"
        )
    return _rewrite_swebench_parser(rewritten, task)


def _rewrite_swebench_parser(source: str, task: TaskSpec) -> str:
    import_line = "from swebench.harness.test_spec.test_spec import make_test_spec"
    if source.count(import_line) != 1:
        raise RuntimeError(
            f"task {task.id} verifier must contain the pinned test-spec import"
        )
    construction = "test_spec   = make_test_spec(datum)"
    if source.count(construction) != 1:
        raise RuntimeError(
            f"task {task.id} verifier must contain the pinned test-spec construction"
        )
    source = source.replace(import_line, "from types import SimpleNamespace")
    offline_construction = """def directive(name):
    value = datum[name]
    return json.loads(value) if isinstance(value, str) else list(value)

test_spec = SimpleNamespace(
    instance_id=datum[KEY_INSTANCE_ID],
    repo=datum[\"repo\"],
    version=datum[\"version\"],
    FAIL_TO_PASS=directive(FAIL_TO_PASS),
    PASS_TO_PASS=directive(PASS_TO_PASS),
)"""
    return source.replace(construction, offline_construction)


def _reject_trial_resolvers(source: str, task: TaskSpec) -> None:
    forbidden = re.compile(
        r"(?m)^[ \t]*(?:(?:python\d*(?:\.\d+)?|/[^ \t]+/python) -m pip|"
        r"pip3?|uv (?:run|pip|sync|tool)|curl|wget|apt(?:-get)?|npm|npx|pnpm|yarn)\b"
    )
    match = forbidden.search(source)
    if match:
        raise RuntimeError(
            f"task {task.id} verifier still contains a trial-time resolver: "
            f"{match.group(0).strip()}"
        )


def _runtime_python(runtime: VerifierRuntime) -> str:
    if isinstance(runtime, DatasetVerifierRuntimeSpec):
        return runtime.python_interpreter
    return "python3"


def _verifier_runtime(
    manifest: BenchmarkManifest, task: TaskSpec
) -> VerifierRuntime | None:
    return task.verifier_runtime or manifest.dataset.verifier_runtime


def _resolve_task_source(
    manifest: BenchmarkManifest,
    task: TaskSpec,
    repo_root: Path,
) -> Path:
    if manifest.dataset.path:
        dataset = manifest.dataset.path
        dataset_path = dataset if dataset.is_absolute() else repo_root / dataset
        source = (dataset_path / task.id).resolve()
        if not source.is_dir():
            raise RuntimeError(f"local task source is missing: {source}")
        return source

    task_name = task.id
    if manifest.dataset.ref and "/" in manifest.dataset.ref and "/" not in task_name:
        task_name = f"{manifest.dataset.ref.split('/', 1)[0]}/{task_name}"
    harbor = shutil.which("harbor")
    if harbor is None:
        raise RuntimeError("harbor is required to prepare remote task images")
    first_line = Path(harbor).read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith("#!"):
        raise RuntimeError(f"cannot resolve Harbor interpreter from {harbor}")
    interpreter = first_line.removeprefix("#!").strip()
    if not Path(interpreter).is_file():
        raise RuntimeError(f"Harbor interpreter is unavailable: {interpreter}")
    payload = json.dumps(
        {
            "name": manifest.dataset.ref,
            "ref": manifest.dataset.version,
            "task_name": task_name,
        }
    )
    script = """
import asyncio, json, sys
from harbor.models.job.config import DatasetConfig
from harbor.tasks.client import TaskClient

async def resolve():
    value = json.loads(sys.argv[1])
    config = DatasetConfig(
        name=value["name"], ref=value["ref"], task_names=[value["task_name"]]
    )
    task_configs = await config.get_task_configs()
    if len(task_configs) != 1:
        raise RuntimeError(
            f"task resolution returned {len(task_configs)} entries"
        )
    result = await TaskClient().download_tasks([task_configs[0].get_task_id()])
    if len(result.results) != 1:
        raise RuntimeError("task download did not return exactly one result")
    print(result.results[0].path.resolve())

asyncio.run(resolve())
"""
    result = subprocess.run(
        [interpreter, "-c", script, payload],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if result.returncode:
        detail = (
            result.stderr or result.stdout or "Harbor task resolution failed"
        ).strip()
        raise RuntimeError(detail[-2_000:])
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Harbor task resolution returned no source path")
    source = Path(lines[-1]).resolve()
    if not source.is_dir():
        raise RuntimeError(f"Harbor task source is missing: {source}")
    return source


def _lock_root(manifest: BenchmarkManifest, task: TaskSpec, repo_root: Path) -> Path:
    dataset_key = hashlib.sha256(manifest.dataset.harbor_ref.encode()).hexdigest()[:16]
    return repo_root / TASK_RUNTIME_ROOT / dataset_key / _slug(task.id)


def _requires_gold_verification(manifest: BenchmarkManifest) -> bool:
    return manifest.dataset.ref == "swe-bench/swe-bench-verified"


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(f"L\0{relative}\0{os.readlink(path)}\0".encode())
        elif path.is_file():
            digest.update(f"F\0{relative}\0".encode())
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _reject_escaping_symlinks(root: Path) -> None:
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        target = (path.parent / os.readlink(path)).resolve()
        if not target.is_relative_to(resolved_root):
            raise RuntimeError(
                f"task contains escaping symlink: {path.relative_to(root)}"
            )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-").lower()
