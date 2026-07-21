from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import toml

from fugue.bench import task_runtime
from fugue.bench.manifest import load_manifest


def _fixture(tmp_path: Path):
    dataset = tmp_path / "dataset" / "task-one"
    (dataset / "environment").mkdir(parents=True)
    (dataset / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (dataset / "instruction.md").write_text("Do the task.\n")
    (dataset / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "task-one"\n\n'
        "[environment]\nallow_internet = true\n"
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "dataset:\n  path: dataset\n"
        "harnesses:\n  - {name: codex, agent: fugue.agents:FugueCodex}\n"
        "tasks:\n  - id: task-one\n    metadata: {architecture: arm64}\n"
    )
    return load_manifest(manifest_path), dataset


def _verifier_fixture(tmp_path: Path):
    manifest, dataset = _fixture(tmp_path)
    (dataset / "tests").mkdir()
    script = (
        "#!/bin/sh\n"
        "pip3 install --break-system-packages \\\n"
        "    pytest==8.4.1 \\\n"
        "    pytest-json-ctrf==0.3.5\n"
        "pytest -q\n"
    )
    (dataset / "tests" / "test.sh").write_text(script)
    manifest_path = tmp_path / "verifier-manifest.yaml"
    manifest_path.write_text(
        "dataset:\n  path: dataset\n"
        "harnesses:\n  - {name: codex, agent: fugue.agents:FugueCodex}\n"
        "tasks:\n  - id: task-one\n    metadata: {architecture: arm64}\n"
        "    verifier_runtime:\n"
        "      python_packages: [pytest==8.4.1, pytest-json-ctrf==0.3.5]\n"
        f"      test_script_sha256: {task_runtime.hashlib.sha256(script.encode()).hexdigest()}\n"
    )
    return load_manifest(manifest_path), dataset


def _swebench_verifier_fixture(tmp_path: Path, *, editable_target: str = "."):
    manifest, dataset = _fixture(tmp_path)
    (dataset / "tests").mkdir()
    script = (
        "#!/bin/bash\n"
        f"python -m pip install -e {editable_target}\n"
        "from swebench.harness.test_spec.test_spec import make_test_spec\n"
        "test_spec   = make_test_spec(datum)\n"
        'uv run parser.py | tee -a "$LOG_FILE"\n'
    )
    (dataset / "tests" / "test.sh").write_text(script)
    manifest_path = tmp_path / "swebench-manifest.yaml"
    manifest_path.write_text(
        "dataset:\n  path: dataset\n"
        "  verifier_runtime:\n"
        "    profile: swebench-v4-offline\n"
        "    python_interpreter: /opt/fugue-verifier/bin/python\n"
        "    python_packages: [swebench==4.0.3, datasets==2.16.1, fastcore==1.10.5]\n"
        "harnesses:\n  - {name: codex, agent: fugue.agents:FugueCodex}\n"
        "tasks:\n  - id: task-one\n    metadata: {architecture: arm64}\n"
    )
    return load_manifest(manifest_path), dataset, script


def test_gold_verification_requirement_is_manifest_scoped(tmp_path: Path) -> None:
    authored_manifest, _ = _fixture(tmp_path / "authored")
    swe_manifest = replace(
        authored_manifest,
        dataset=replace(
            authored_manifest.dataset,
            ref="swe-bench/swe-bench-verified",
        ),
    )

    assert not task_runtime.task_runtime_requires_gold_verification(authored_manifest)
    assert task_runtime.task_runtime_requires_gold_verification(swe_manifest)


def test_task_preparation_locks_image_and_disables_trial_network(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, source = _fixture(tmp_path)
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    [
                        {
                            "Id": "sha256:" + "a" * 64,
                            "Architecture": "arm64",
                            "Os": "linux",
                        }
                    ]
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(task_runtime.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(task_runtime, "_resolve_task_source", lambda *args: source)
    monkeypatch.setattr(task_runtime.subprocess, "run", run)
    monkeypatch.setattr(
        task_runtime,
        "docker_build_command",
        lambda *args: ["docker", "build", "--provenance=false", *args],
    )
    lock = task_runtime.prepare_task_runtime(
        manifest, manifest.tasks[0], repo_root=tmp_path
    )

    prepared = Path(lock["dataset_path"]) / "task-one" / "task.toml"
    config = toml.loads(prepared.read_text())
    assert config["environment"]["docker_image"] == "sha256:" + "a" * 64
    assert config["environment"]["network_mode"] == "no-network"
    assert config["environment"]["allowed_hosts"] == []
    assert "allow_internet" not in config["environment"]
    assert commands[0][0:5] == [
        "docker",
        "build",
        "--provenance=false",
        "--platform",
        "linux/arm64",
    ]


def test_task_preparation_locks_verifier_dependencies_before_the_trial(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, source = _verifier_fixture(tmp_path)
    build_dockerfile = ""
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        nonlocal build_dockerfile
        commands.append(command)
        if command[1] == "build":
            build_dockerfile = (Path(command[-1]) / "Dockerfile").read_text()
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    [
                        {
                            "Id": "sha256:" + "a" * 64,
                            "Architecture": "arm64",
                            "Os": "linux",
                        }
                    ]
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(task_runtime.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(task_runtime, "_resolve_task_source", lambda *args: source)
    monkeypatch.setattr(task_runtime.subprocess, "run", run)

    lock = task_runtime.prepare_task_runtime(
        manifest, manifest.tasks[0], repo_root=tmp_path
    )

    assert "pytest==8.4.1 pytest-json-ctrf==0.3.5" in build_dockerfile
    assert any(command[1:4] == ["run", "--rm", "--network"] for command in commands)
    prepared = Path(lock["dataset_path"]) / "task-one" / "tests" / "test.sh"
    prepared_source = prepared.read_text()
    assert "pip3 install" not in prepared_source
    assert "Setup locked these verifier dependencies" in prepared_source
    assert lock["verifier_runtime"] == {
        "python_packages": ["pytest==8.4.1", "pytest-json-ctrf==0.3.5"],
        "test_script_sha256": manifest.tasks[0].verifier_runtime.test_script_sha256,
    }


def test_swebench_verifier_is_prepared_for_offline_trial(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, source, original_script = _swebench_verifier_fixture(tmp_path)
    build_dockerfile = ""
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        nonlocal build_dockerfile
        commands.append(command)
        if command[1] == "build":
            build_dockerfile = (Path(command[-1]) / "Dockerfile").read_text()
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    [
                        {
                            "Id": "sha256:" + "a" * 64,
                            "Architecture": "arm64",
                            "Os": "linux",
                        }
                    ]
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(task_runtime.shutil, "which", lambda name: "/docker")
    monkeypatch.setattr(task_runtime, "_resolve_task_source", lambda *args: source)
    monkeypatch.setattr(task_runtime.subprocess, "run", run)

    lock = task_runtime.prepare_task_runtime(
        manifest, manifest.tasks[0], repo_root=tmp_path
    )

    interpreter = "/opt/fugue-verifier/bin/python"
    assert "/opt/miniconda3/bin/python -m venv /opt/fugue-verifier" in build_dockerfile
    assert f"{interpreter} -m pip install --no-cache-dir" in build_dockerfile
    assert "swebench==4.0.3 datasets==2.16.1 fastcore==1.10.5" in build_dockerfile
    verify = next(
        command for command in commands if command[1:4] == ["run", "--rm", "--network"]
    )
    assert interpreter in verify
    prepared = Path(lock["dataset_path"]) / "task-one" / "tests" / "test.sh"
    rewritten = prepared.read_text()
    assert "python -m pip" not in rewritten
    assert "uv run" not in rewritten
    assert "make_test_spec" not in rewritten
    assert "test_spec = SimpleNamespace(" in rewritten
    assert f'{interpreter} parser.py | tee -a "$LOG_FILE"' in rewritten
    assert lock["verifier_script"] == {
        "original_sha256": hashlib.sha256(original_script.encode()).hexdigest(),
        "prepared_sha256": hashlib.sha256(rewritten.encode()).hexdigest(),
    }


@pytest.mark.parametrize("editable_target", [".[test]", ".[test,dev] --verbose"])
def test_swebench_verifier_accepts_locked_editable_extras(
    tmp_path: Path, editable_target: str
) -> None:
    manifest, source, _script = _swebench_verifier_fixture(
        tmp_path, editable_target=editable_target
    )

    task_runtime._lock_verifier_script(
        source, manifest.tasks[0], manifest.dataset.verifier_runtime
    )

    rewritten = (source / "tests" / "test.sh").read_text()
    assert "pip install" not in rewritten
    assert "Setup prepared the task environment" in rewritten


def test_swebench_verifier_accepts_a_prepared_base_without_install(
    tmp_path: Path,
) -> None:
    manifest, source, _script = _swebench_verifier_fixture(tmp_path)
    script = source / "tests" / "test.sh"
    script.write_text(script.read_text().replace("python -m pip install -e .\n", ""))

    task_runtime._lock_verifier_script(
        source, manifest.tasks[0], manifest.dataset.verifier_runtime
    )

    rewritten = script.read_text()
    assert "pip install" not in rewritten
    assert "/opt/fugue-verifier/bin/python parser.py" in rewritten


@pytest.mark.parametrize("second_target", [".", ".[test]"])
def test_swebench_verifier_rejects_multiple_local_installs(
    tmp_path: Path, second_target: str
) -> None:
    manifest, source, _script = _swebench_verifier_fixture(tmp_path)
    script = source / "tests" / "test.sh"
    script.write_text(
        script.read_text().replace(
            "python -m pip install -e .\n",
            "python -m pip install -e .\n"
            f"python -m pip install -e {second_target}\n",
        )
    )

    with pytest.raises(RuntimeError, match="multiple local editable installs"):
        task_runtime._lock_verifier_script(
            source, manifest.tasks[0], manifest.dataset.verifier_runtime
        )


def test_swebench_verifier_rewrite_fails_closed_on_upstream_shape_change(
    tmp_path: Path,
) -> None:
    manifest, source, _script = _swebench_verifier_fixture(tmp_path)
    script = source / "tests" / "test.sh"
    script.write_text(
        script.read_text().replace("uv run parser.py", "uv run --offline parser.py")
    )

    with pytest.raises(RuntimeError, match="one exact uv parser invocation"):
        task_runtime._lock_verifier_script(
            source, manifest.tasks[0], manifest.dataset.verifier_runtime
        )


@pytest.mark.parametrize(
    "install",
    [
        "python -m pip install -e .[test] extra-package",
        "python -m pip install -e '.[test]' && curl https://example.invalid",
    ],
)
def test_swebench_verifier_rewrite_rejects_nonlocal_install(
    tmp_path: Path, install: str
) -> None:
    manifest, source, _script = _swebench_verifier_fixture(tmp_path)
    script = source / "tests" / "test.sh"
    script.write_text(
        script.read_text().replace("python -m pip install -e .", install)
    )

    with pytest.raises(RuntimeError, match="trial-time resolver"):
        task_runtime._lock_verifier_script(
            source, manifest.tasks[0], manifest.dataset.verifier_runtime
        )


def test_task_runtime_lock_rejects_dataset_drift(tmp_path: Path) -> None:
    manifest, _source = _fixture(tmp_path)
    root = task_runtime._lock_root(manifest, manifest.tasks[0], tmp_path)
    dataset = root / "dataset-deadbeef"
    dataset.mkdir(parents=True)
    path = root / "runtime-lock-arm64.json"
    value = {
        "schema_version": 1,
        "contract_version": task_runtime.TASK_RUNTIME_CONTRACT_VERSION,
        "dataset": manifest.dataset.harbor_ref,
        "task_id": "task-one",
        "architecture": "arm64",
        "image_id": "sha256:" + "a" * 64,
        "dataset_path": dataset.as_posix(),
    }
    path.write_text(json.dumps(value))
    assert task_runtime.read_task_runtime_lock(manifest, manifest.tasks[0], tmp_path)

    value["dataset"] = "different"
    path.write_text(json.dumps(value))
    assert (
        task_runtime.read_task_runtime_lock(manifest, manifest.tasks[0], tmp_path)
        is None
    )


@pytest.mark.parametrize("contract_version", [7, 8])
def test_task_runtime_lock_rejects_preintegration_contracts(
    tmp_path: Path, contract_version: int
) -> None:
    manifest, _source = _fixture(tmp_path)
    root = task_runtime._lock_root(manifest, manifest.tasks[0], tmp_path)
    dataset = root / "dataset-deadbeef"
    dataset.mkdir(parents=True)
    value = {
        "schema_version": 1,
        "contract_version": contract_version,
        "dataset": manifest.dataset.harbor_ref,
        "task_id": "task-one",
        "architecture": "arm64",
        "image_id": "sha256:" + "a" * 64,
        "dataset_path": dataset.as_posix(),
    }
    (root / "runtime-lock-arm64.json").write_text(json.dumps(value))

    assert (
        task_runtime.read_task_runtime_lock(manifest, manifest.tasks[0], tmp_path)
        is None
    )


def test_swebench_runtime_lock_requires_base_fail_gold_pass_verification(
    tmp_path: Path,
) -> None:
    manifest, _source = _fixture(tmp_path)
    manifest = replace(
        manifest,
        dataset=replace(
            manifest.dataset,
            ref="swe-bench/swe-bench-verified",
        ),
    )
    root = task_runtime._lock_root(manifest, manifest.tasks[0], tmp_path)
    dataset = root / "dataset-deadbeef"
    dataset.mkdir(parents=True)
    value = {
        "schema_version": 1,
        "contract_version": task_runtime.TASK_RUNTIME_CONTRACT_VERSION,
        "dataset": manifest.dataset.harbor_ref,
        "task_id": "task-one",
        "architecture": "arm64",
        "verifier_runtime": None,
        "image_id": "sha256:" + "a" * 64,
        "dataset_path": dataset.as_posix(),
    }
    (root / "runtime-lock-arm64.json").write_text(json.dumps(value))

    assert (
        task_runtime.read_task_runtime_lock(manifest, manifest.tasks[0], tmp_path)
        is None
    )


def test_swebench_setup_verifies_base_and_gold_without_persisting_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source, _script = _swebench_verifier_fixture(tmp_path)
    calls: list[Path | None] = []
    patch = "diff --git a/src/a.py b/src/a.py\n"
    patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()

    def verify(*args, gold_patch_path=None, **kwargs) -> bool:
        calls.append(gold_patch_path)
        if gold_patch_path is not None:
            assert gold_patch_path.read_text() == patch
        return gold_patch_path is not None

    monkeypatch.setattr(task_runtime, "_run_swe_bench_verifier", verify)
    verification = task_runtime._verify_swe_bench_outcomes(
        "task-image",
        manifest.tasks[0],
        source,
        gold_patch=patch,
        gold_patch_sha256=patch_sha256,
        architecture="arm64",
        repo_root=tmp_path,
    )

    assert calls[0] is None
    assert calls[1] is not None
    assert verification == {
        "base_failed": True,
        "gold_passed": True,
        "gold_patch_sha256": patch_sha256,
    }
    assert patch not in json.dumps(verification)
    assert not list((tmp_path / ".fugue/runtime").glob("fugue-task-verify-*"))


@pytest.mark.parametrize("base_resolved,gold_resolved", [(True, True), (False, False)])
def test_swebench_setup_verification_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_resolved: bool,
    gold_resolved: bool,
) -> None:
    manifest, source, _script = _swebench_verifier_fixture(tmp_path)

    def verify(*args, gold_patch_path=None, **kwargs) -> bool:
        return gold_resolved if gold_patch_path is not None else base_resolved

    monkeypatch.setattr(task_runtime, "_run_swe_bench_verifier", verify)
    with pytest.raises(RuntimeError, match="base checkout|gold patch"):
        task_runtime._verify_swe_bench_outcomes(
            "task-image",
            manifest.tasks[0],
            source,
            gold_patch="patch",
            gold_patch_sha256=hashlib.sha256(b"patch").hexdigest(),
            architecture="arm64",
            repo_root=tmp_path,
        )


def test_swebench_setup_container_prepares_harbor_verifier_log_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source, _script = _swebench_verifier_fixture(tmp_path)

    def run(command: list[str], **kwargs):
        logs_mount = next(
            value
            for value in command
            if value.startswith("type=bind,source=") and value.endswith("target=/logs")
        )
        mount = dict(item.split("=", 1) for item in logs_mount.split(","))
        logs = Path(mount["source"])
        assert (logs / "verifier").is_dir()
        (logs / "verifier" / "report.json").write_text(
            json.dumps({"task-one": {"resolved": False}})
        )
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(task_runtime.subprocess, "run", run)

    assert (
        task_runtime._run_swe_bench_verifier(
            "task-image",
            manifest.tasks[0],
            source,
            architecture="arm64",
            logs=tmp_path / "logs",
            gold_patch_path=None,
            repo_root=tmp_path,
        )
        is False
    )


def test_local_task_source_does_not_import_harbor(tmp_path: Path) -> None:
    manifest, source = _fixture(tmp_path)

    assert (
        task_runtime._resolve_task_source(manifest, manifest.tasks[0], tmp_path)
        == source.resolve()
    )
