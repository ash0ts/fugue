from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from fugue.bench.candidates import (
    stable_digest,
)
from fugue.bench.files import atomic_write_json
from fugue.bench.library import validate_id
from fugue.bench.provider_contract import (
    CandidateBundleV1,
    CellRequestV1,
    CellResultV1,
    PreparationReceiptV1,
    PrivateEvaluationBundleV1,
    ProviderDescriptorV1,
    SuiteBundleV1,
    candidate_bundle_from_dict,
    cell_request_from_dict,
    cell_result_from_dict,
    preparation_receipt_from_dict,
    private_evaluation_bundle_from_dict,
    provider_contract_schemas,
    provider_descriptor_from_dict,
    suite_bundle_from_dict,
)

_MAX_PROVIDER_OUTPUT_BYTES = 64 * 1024 * 1024
_SAFE_PROVIDER_ENV = (
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "UV_CACHE_DIR",
    "VIRTUAL_ENV",
)
_PROVIDER_OPERATIONS = frozenset(
    {"describe", "resolve-candidate", "resolve-suite", "prepare", "run-cell"}
)


@dataclass(frozen=True)
class ProviderLockV1:
    schema_version: int
    provider_id: str
    protocol_version: int
    command: tuple[str, ...]
    executable_path: str
    executable_sha256: str
    descriptor: dict[str, Any]
    descriptor_digest: str
    source_digest: str
    lock_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


class ProviderProcessError(RuntimeError):
    def __init__(
        self,
        operation: str,
        message: str,
        *,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.returncode = returncode
        self.stderr = stderr


class ProviderClient:
    """Strict JSON subprocess client for one locked evaluation provider."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_sec: float = 30.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.command = _validate_command(command)
        self.timeout_sec = _positive_timeout(timeout_sec)
        self.env = _provider_env(env)

    @classmethod
    def from_command_text(
        cls,
        command: str,
        *,
        timeout_sec: float = 30.0,
        env: Mapping[str, str] | None = None,
    ) -> ProviderClient:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"invalid provider command: {exc}") from exc
        return cls(argv, timeout_sec=timeout_sec, env=env)

    @classmethod
    def from_lock(
        cls,
        lock: ProviderLockV1,
        *,
        timeout_sec: float = 30.0,
        env: Mapping[str, str] | None = None,
    ) -> ProviderClient:
        client = cls(lock.command, timeout_sec=timeout_sec, env=env)
        executable = Path(client.command[0]).absolute()
        if executable.as_posix() != lock.executable_path:
            raise ValueError("provider executable path differs from the lock")
        if _sha256_path(executable) != lock.executable_sha256:
            raise ValueError("provider executable bytes differ from the lock")
        return client

    def invoke(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if operation not in _PROVIDER_OPERATIONS:
            raise ValueError(f"unsupported provider operation: {operation}")
        request = _json_value(dict(payload))
        try:
            completed = subprocess.run(
                [*self.command, operation],
                input=json.dumps(
                    request, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
                env=self.env,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderProcessError(
                operation,
                f"provider operation timed out after {self.timeout_sec:g} seconds",
            ) from exc
        except OSError as exc:
            raise ProviderProcessError(
                operation, f"provider operation could not start: {exc}"
            ) from exc
        stdout = completed.stdout
        stderr = completed.stderr[-4000:]
        if len(stdout.encode()) > _MAX_PROVIDER_OUTPUT_BYTES:
            raise ProviderProcessError(
                operation,
                "provider output exceeded the 64 MiB protocol limit",
                returncode=completed.returncode,
                stderr=stderr,
            )
        if completed.returncode:
            raise ProviderProcessError(
                operation,
                f"provider operation failed with exit code {completed.returncode}",
                returncode=completed.returncode,
                stderr=stderr,
            )
        try:
            value = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ProviderProcessError(
                operation,
                "provider output was not exactly one JSON value",
                returncode=completed.returncode,
                stderr=stderr,
            ) from exc
        if not isinstance(value, dict):
            raise ProviderProcessError(
                operation, "provider output must be a JSON object", stderr=stderr
            )
        return value

    def describe(self) -> ProviderDescriptorV1:
        return provider_descriptor_from_dict(self.invoke("describe", {}))

    def resolve_candidate(self, candidate_ref: str) -> CandidateBundleV1:
        return candidate_bundle_from_dict(
            self.invoke(
                "resolve-candidate",
                {"candidate_ref": _non_empty_text(candidate_ref, "candidate ref")},
            )
        )

    def resolve_suite(
        self, suite_ref: str
    ) -> tuple[SuiteBundleV1, PrivateEvaluationBundleV1]:
        response = self.invoke(
            "resolve-suite",
            {"suite_ref": _non_empty_text(suite_ref, "suite ref")},
        )
        _reject_unknown(response, {"suite", "private_evaluation"}, "suite response")
        if set(response) != {"suite", "private_evaluation"}:
            raise ValueError(
                "suite response requires suite and private_evaluation fields"
            )
        suite = suite_bundle_from_dict(
            _mapping(response.get("suite"), "suite response")
        )
        private = private_evaluation_bundle_from_dict(
            _mapping(
                response.get("private_evaluation"), "private evaluation response"
            )
        )
        if private.suite_digest != suite.bundle_digest:
            raise ValueError("private evaluation bundle does not bind the public suite")
        public_ids = {task.id for task in suite.tasks}
        private_ids = {task.task_id for task in private.tasks}
        if not private_ids.issubset(public_ids):
            raise ValueError("private evaluation contains tasks absent from public suite")
        return suite, private

    def prepare(
        self,
        *,
        provider_lock_digest: str,
        candidate: CandidateBundleV1,
        suite: SuiteBundleV1,
    ) -> PreparationReceiptV1:
        receipt = preparation_receipt_from_dict(
            self.invoke(
                "prepare",
                {
                    "provider_lock_digest": provider_lock_digest,
                    "candidate": candidate.to_dict(),
                    "suite": suite.to_dict(),
                },
            )
        )
        _verify_prepared_task_locks(receipt, suite)
        return receipt

    def run_cell(self, request: CellRequestV1) -> CellResultV1:
        checked_request = cell_request_from_dict(request.to_dict())
        result = cell_result_from_dict(
            self.invoke("run-cell", checked_request.to_dict())
        )
        if result.provider_id != request.provider_id:
            raise ValueError("cell result provider differs from the request")
        if result.cell_id != request.cell_id:
            raise ValueError("cell result cell id differs from the request")
        if result.request_digest != request.request_digest:
            raise ValueError("cell result does not bind the exact request")
        return result


def validate_provider(command: str, *, timeout_sec: float = 30.0) -> dict[str, Any]:
    client = ProviderClient.from_command_text(command, timeout_sec=timeout_sec)
    descriptor = client.describe()
    return {
        "valid": True,
        "command": list(client.command),
        "descriptor": descriptor.to_dict(),
    }


def lock_provider(
    command: str,
    *,
    output: Path,
    timeout_sec: float = 30.0,
) -> ProviderLockV1:
    client = ProviderClient.from_command_text(command, timeout_sec=timeout_sec)
    descriptor = client.describe()
    executable = Path(client.command[0]).absolute()
    unsigned = ProviderLockV1(
        schema_version=1,
        provider_id=descriptor.provider_id,
        protocol_version=descriptor.protocol_version,
        command=client.command,
        executable_path=executable.as_posix(),
        executable_sha256=_sha256_path(executable),
        descriptor=descriptor.to_dict(),
        descriptor_digest=descriptor.descriptor_digest,
        source_digest=descriptor.source_provenance["source_digest"],
    )
    lock = replace(
        unsigned, lock_digest=_artifact_digest(unsigned.to_dict(), "lock_digest")
    )
    atomic_write_json(output.resolve(), lock.to_dict(), mode=0o644)
    return lock


def load_provider_lock(path: Path) -> ProviderLockV1:
    raw = _load_json(path)
    _strict_dataclass(raw, ProviderLockV1, "provider lock")
    descriptor = provider_descriptor_from_dict(
        _mapping(raw.get("descriptor"), "provider lock descriptor")
    )
    command = tuple(
        _non_empty_text(value, "provider command argument")
        for value in _sequence(raw.get("command"), "provider command")
    )
    value = ProviderLockV1(
        schema_version=_schema(raw.get("schema_version"), "provider lock"),
        provider_id=validate_id(raw.get("provider_id") or "", kind="provider id"),
        protocol_version=_schema(
            raw.get("protocol_version"), "provider protocol"
        ),
        command=command,
        executable_path=_absolute_path(
            raw.get("executable_path"), "provider executable path"
        ),
        executable_sha256=_digest(
            raw.get("executable_sha256"), "provider executable digest"
        ),
        descriptor=descriptor.to_dict(),
        descriptor_digest=_digest(
            raw.get("descriptor_digest"), "provider descriptor digest"
        ),
        source_digest=_digest(raw.get("source_digest"), "provider source digest"),
        lock_digest=_digest(raw.get("lock_digest"), "provider lock digest"),
    )
    if value.provider_id != descriptor.provider_id:
        raise ValueError("provider lock id differs from its descriptor")
    if value.descriptor_digest != descriptor.descriptor_digest:
        raise ValueError("provider lock descriptor digest differs from its descriptor")
    if value.source_digest != descriptor.source_provenance["source_digest"]:
        raise ValueError("provider lock source digest differs from its descriptor")
    _verify_artifact(value.to_dict(), "lock_digest", "provider lock")
    ProviderClient.from_lock(value)
    return value


def provider_conformance(
    *,
    provider_lock: Path,
    candidate_ref: str,
    suite_ref: str,
    exercise_run_cell: bool = False,
    timeout_sec: float = 120.0,
    task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    runtime_env = {"FUGUE_PROVIDER_OFFLINE_CONFORMANCE": "1"}
    temporary_workspace: tempfile.TemporaryDirectory[str] | None = None
    if exercise_run_cell:
        temporary_workspace = tempfile.TemporaryDirectory(
            prefix="fugue-provider-conformance-"
        )
        runtime_env["FUGUE_PROVIDER_WORKSPACE"] = temporary_workspace.name
    try:
        lock = load_provider_lock(provider_lock)
        client = ProviderClient.from_lock(
            lock,
            timeout_sec=timeout_sec,
            env=runtime_env or None,
        )
        descriptor = client.describe()
        if descriptor.descriptor_digest != lock.descriptor_digest:
            raise ValueError("live provider descriptor differs from the provider lock")
        candidate = client.resolve_candidate(candidate_ref)
        suite, private = client.resolve_suite(suite_ref)
        _require_provider(candidate.provider_id, lock)
        _require_provider(suite.provider_id, lock)
        preparation = client.prepare(
            provider_lock_digest=lock.lock_digest,
            candidate=candidate,
            suite=suite,
        )
        classifications = {
            value: sum(task.portability == value for task in suite.tasks)
            for value in ("portable", "provider_bound", "blocked")
        }
        selected_ids = tuple(dict.fromkeys(task_ids or ()))
        unknown_tasks = sorted(set(selected_ids) - {task.id for task in suite.tasks})
        if unknown_tasks:
            raise ValueError(
                "unknown provider conformance task(s): " + ", ".join(unknown_tasks)
            )
        selected_tasks = tuple(
            task
            for task in suite.tasks
            if not selected_ids or task.id in selected_ids
        )
        selected_evaluator_ids = {
            evaluator_id
            for task in selected_tasks
            for evaluator_id in task.evaluator_ids
        }
        evaluator_issues = _provider_evaluator_issues(
            suite,
            preparation,
            evaluator_ids=selected_evaluator_ids,
        )
        result_rows: list[dict[str, Any]] = []
        if exercise_run_cell:
            required_credentials = {
                *candidate.required_credentials,
                *(
                    name
                    for task in selected_tasks
                    for name in task.credential_names
                ),
            }
            if required_credentials:
                raise ValueError(
                    "offline provider conformance cannot inject credentials: "
                    + ", ".join(sorted(required_credentials))
                )
            runtime_digest = stable_digest(
                {"kind": "offline-provider-conformance", "schema_version": 1}
            )
            plan_digest = stable_digest(
                {
                    "provider_lock": lock.lock_digest,
                    "candidate": candidate.bundle_digest,
                    "suite": suite.bundle_digest,
                    "runtime": runtime_digest,
                }
            )
            for index, task in enumerate(selected_tasks, start=1):
                if task.portability == "blocked":
                    continue
                unsigned = CellRequestV1(
                    schema_version=1,
                    provider_id=lock.provider_id,
                    plan_digest=plan_digest,
                    cell_id=validate_id(
                        f"conformance-{index}", kind="conformance cell id"
                    ),
                    candidate_digest=candidate.bundle_digest,
                    suite_digest=suite.bundle_digest,
                    preparation_receipt_digest=preparation.receipt_digest,
                    candidate=candidate.to_dict(),
                    preparation=preparation.to_dict(),
                    task=task.to_dict(),
                    attempt=1,
                    runtime_lock_digest=runtime_digest,
                    credential_profile_names=tuple(
                        dict.fromkeys(
                            (*candidate.required_credentials, *task.credential_names)
                        )
                    ),
                    budget={"max_cost_usd": 1.0, "max_seconds": 300},
                )
                request = cell_request_from_dict(unsigned.to_dict())
                result = client.run_cell(request)
                result_rows.append(
                    {
                        "task_id": task.id,
                        "cell_id": result.cell_id,
                        "status": result.status,
                        "request_digest": request.request_digest,
                        "result_digest": result.result_digest,
                        "failure": result.failure,
                        "usage": result.usage,
                        "evidence_refs": list(result.evidence_refs),
                        "cleanup": result.cleanup,
                    }
                )
        return {
            "schema_version": 1,
            "scope": "offline_protocol_conformance",
            "provider_id": lock.provider_id,
            "provider_lock_digest": lock.lock_digest,
            "descriptor_digest": descriptor.descriptor_digest,
            "candidate_ref": candidate.candidate_ref,
            "candidate_digest": candidate.bundle_digest,
            "suite_ref": suite.suite_ref,
            "suite_digest": suite.bundle_digest,
            "private_digest": private.private_digest,
            "preparation_receipt_digest": preparation.receipt_digest,
            "task_count": len(suite.tasks),
            "selected_task_ids": [task.id for task in selected_tasks],
            "selected_task_count": len(selected_tasks),
            "scenario_count": len(suite.scenarios),
            "evaluator_count": len(suite.evaluators),
            "attempts": suite.attempts,
            "classifications": classifications,
            "evaluator_issues": evaluator_issues,
            "run_cell_exercised": exercise_run_cell,
            "protocol_cell_results": result_rows,
            "task_outcomes_qualified": False,
            "claim_limitation": (
                "Provider conformance validates protocol artifacts and lifecycle "
                "operations only; it is not a Fugue experiment or task-success claim."
            ),
            "conformant": bool(selected_tasks)
            and bool(suite.scenarios)
            and not evaluator_issues
            and (
                not exercise_run_cell
                or bool(result_rows)
            ),
        }
    finally:
        if temporary_workspace is not None:
            temporary_workspace.cleanup()


def write_provider_schemas(destination: Path) -> tuple[Path, ...]:
    root = destination.resolve()
    paths = []
    for name, schema in provider_contract_schemas().items():
        path = root / f"{name}.schema.json"
        atomic_write_json(path, schema, mode=0o644)
        paths.append(path)
    return tuple(paths)


def scaffold_provider(
    destination: Path,
    *,
    provider_id: str,
    force: bool = False,
) -> tuple[Path, Path]:
    """Create a dependency-free Python provider starting point."""

    validated_id = validate_id(provider_id, kind="provider id")
    root = destination.resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(
            f"{root}: destination is not empty; pass --force to replace scaffold files"
        )
    root.mkdir(parents=True, exist_ok=True)
    provider_path = root / "provider.py"
    readme_path = root / "README.md"
    if not force and (provider_path.exists() or readme_path.exists()):
        raise FileExistsError("provider scaffold files already exist")
    provider_text = _PROVIDER_SCAFFOLD.replace("__PROVIDER_ID__", validated_id)
    readme_text = _PROVIDER_SCAFFOLD_README.replace(
        "__PROVIDER_ID__", validated_id
    )
    _atomic_write_text(provider_path, provider_text, mode=0o755)
    _atomic_write_text(readme_path, readme_text, mode=0o644)
    return provider_path, readme_path


def _require_provider(provider_id: str, lock: ProviderLockV1) -> None:
    if provider_id != lock.provider_id:
        raise ValueError(
            f"provider artifact {provider_id!r} differs from lock {lock.provider_id!r}"
        )


def _provider_evaluator_issues(
    suite: SuiteBundleV1,
    preparation: PreparationReceiptV1,
    *,
    evaluator_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    frozen_kinds = {
        str(value.get("kind") or "") for value in preparation.frozen_references
    }
    for evaluator in suite.evaluators:
        if evaluator_ids is not None and evaluator.id not in evaluator_ids:
            continue
        runtime = evaluator.implementation.get("runtime")
        if evaluator.type in {"custom", "post_run_verifier"} and runtime is None:
            issues.append(
                {
                    "evaluator_id": evaluator.id,
                    "code": "missing_scorer_runtime",
                    "message": (
                        f"{evaluator.type} evaluator requires a pinned isolated "
                        "scorer runtime"
                    ),
                }
            )
        if evaluator.type == "reference" and not (
            {"reference", f"evaluator:{evaluator.id}"} & frozen_kinds
        ):
            issues.append(
                {
                    "evaluator_id": evaluator.id,
                    "code": "missing_frozen_reference",
                    "message": (
                        "reference evaluator requires preparation to freeze its "
                        "private reference"
                    ),
                }
            )
    return issues


def _verify_prepared_task_locks(
    receipt: PreparationReceiptV1, suite: SuiteBundleV1
) -> None:
    locks = {
        str(value.get("id") or ""): str(value.get("digest") or "")
        for value in receipt.materialized_resources
        if value.get("kind") == "provider-task-v1"
    }
    expected = {
        task.id: stable_digest(task.to_dict())
        for task in suite.tasks
        if task.portability != "blocked"
    }
    if locks != expected:
        raise ValueError(
            "preparation receipt must bind every runnable public provider task "
            "exactly and must not materialize blocked tasks"
        )


def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    if not command:
        raise ValueError("provider command must not be empty")
    argv = tuple(_non_empty_text(value, "provider command argument") for value in command)
    if any("\x00" in value or "\n" in value or "\r" in value for value in argv):
        raise ValueError("provider command arguments cannot contain control characters")
    executable = shutil.which(argv[0])
    if executable is None:
        raise FileNotFoundError(f"provider executable not found: {argv[0]}")
    # Do not resolve interpreter symlinks: a virtual environment's Python
    # executable is intentionally a symlink whose path selects that venv.
    return (Path(executable).absolute().as_posix(), *argv[1:])


def _provider_env(extra: Mapping[str, str] | None) -> dict[str, str]:
    env = {key: os.environ[key] for key in _SAFE_PROVIDER_ENV if key in os.environ}
    if extra:
        for key, value in extra.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("provider environment must contain string pairs")
            env[key] = value
    return env


def _positive_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("provider timeout must be positive")
    return float(value)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_digest(raw: Mapping[str, Any], field_name: str) -> str:
    return stable_digest({**raw, field_name: ""})


def _verify_artifact(raw: Mapping[str, Any], field_name: str, label: str) -> None:
    if raw[field_name] != _artifact_digest(raw, field_name):
        raise ValueError(f"{label} {field_name} does not match")


def _strict_dataclass(raw: Mapping[str, Any], cls: type[Any], label: str) -> None:
    fields = set(cls.__dataclass_fields__)
    _reject_unknown(raw, fields, label)
    missing = sorted(fields - set(raw))
    if missing:
        raise ValueError(f"{label} is missing field(s): {', '.join(missing)}")


def _reject_unknown(
    raw: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown field(s): {', '.join(unknown)}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.resolve().read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return raw


def _mapping(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    return _json_value(dict(raw))


def _sequence(raw: Any, label: str) -> list[Any]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} must be a non-empty array")
    return list(raw)


def _schema(raw: Any, label: str) -> int:
    if raw != 1:
        raise ValueError(f"{label} schema version must be 1")
    return 1


def _digest(raw: Any, label: str) -> str:
    value = str(raw or "")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _absolute_path(raw: Any, label: str) -> str:
    value = Path(_non_empty_text(raw, label))
    if not value.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return value.absolute().as_posix()


def _non_empty_text(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return raw.strip()


def _json_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("provider values must be JSON serializable") from exc


def _atomic_write_text(path: Path, text: str, *, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


_PROVIDER_SCAFFOLD = '''#!/usr/bin/env python3
"""Fugue Evaluation Provider V1 scaffold."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROVIDER_ID = "__PROVIDER_ID__"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def signed(value, field):
    value[field] = digest({**value, field: ""})
    return value


def describe():
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return signed(
        {
            "schema_version": 1,
            "provider_id": PROVIDER_ID,
            "display_name": PROVIDER_ID,
            "provider_version": "0.1.0",
            "protocol_version": 1,
            "capabilities": ["candidate_resolution", "suite_resolution"],
            "task_types": ["single_turn"],
            "input_types": ["text"],
            "evaluator_types": ["deterministic"],
            "lifecycle_types": [],
            "source_provenance": {
                "repository": "replace-with-immutable-source",
                "revision": "replace-with-revision",
                "source_digest": source_digest,
            },
            "descriptor_digest": "",
        },
        "descriptor_digest",
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("expected exactly one provider operation")
    request = json.load(sys.stdin)
    if not isinstance(request, dict):
        raise SystemExit("stdin must contain one JSON object")
    operation = sys.argv[1]
    if operation == "describe":
        if request:
            raise ValueError("describe accepts no fields")
        result = describe()
    else:
        raise NotImplementedError(
            f"implement strict {operation!r} request and response handling"
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
'''

_PROVIDER_SCAFFOLD_README = """# __PROVIDER_ID__ Fugue provider

This is a language-neutral JSON subprocess boundary. The Python file is only a
starting implementation; providers may use any language.

Implement `resolve-candidate`, `resolve-suite`, `prepare`, and `run-cell`.
Reject unknown fields, keep private evaluation data separate, and ensure
`run-cell` uses only assets locked during resolution and preparation.

Validate the descriptor:

```bash
fugue provider validate --command "./provider.py"
```

Then run Fugue's conformance suite before registering the provider.
"""
