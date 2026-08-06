from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    ComparisonEvaluatorV1,
    _comparison_preparation_receipt_path,
    _evaluator,
    _evaluator_digest,
    _observed_frozen_public_source_digest,
    _public_source_lock,
    _qualification_results,
    _score_deterministic_output,
    _verified_public_source_lock,
    _verify_local_public_source_drift,
)
from fugue.bench.post_trial_verifier import (
    POST_TRIAL_VERIFIER_RESOURCE_ROOT,
    PostTrialVerifierResultV1,
    PostTrialVerifierV1,
    post_trial_verifier_lock_from_dict,
    prepare_post_trial_verifier,
    resolve_post_trial_verifier_lock,
    run_post_trial_verifier,
    verify_post_trial_verifier_receipt,
)
from fugue.bench.task_authoring import (
    ScorerRuntimeProfileV1,
    TaskAuthoringLimitsV1,
    run_isolated_evaluator,
)


def _profile() -> ScorerRuntimeProfileV1:
    return ScorerRuntimeProfileV1(
        id="node22-verifier-v1",
        title="Node verifier",
        image="node@example@sha256:" + "a" * 64,
        platform="linux/amd64",
        command=("node", "/input/verifier.cjs", "/input/input.json"),
        profile_digest="b" * 64,
    )


def _prepared(
    tmp_path: Path,
    *,
    verifier_id: str = "fugue-node-test-v1",
    command: list[str] | None = None,
) -> tuple[PostTrialVerifierV1, Any, dict[str, Any], dict[str, Any], bytes]:
    command = command or ["node", "--test"]
    source = tmp_path / "verifier.cjs"
    source.write_text("process.stdout.write('{}');\n", encoding="utf-8")
    runtime_lock_path = tmp_path / "verifier-runtime.json"
    runtime_lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": verifier_id,
                "runtime": {
                    "kind": "oci",
                    "image": _profile().image,
                    "network": "none",
                    "read_only_base": True,
                },
                "command": command,
                "verifier_source_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    verifier = PostTrialVerifierV1(
        type="node_test",
        source="verifier.cjs",
        runtime_lock="verifier-runtime.json",
        runtime="node22-verifier-v1",
        dimension="public_regression_test_success",
    )
    lock = resolve_post_trial_verifier_lock(
        verifier,
        evaluator_id="public-behavioral",
        repo_root=tmp_path,
        runtime_profile=_profile(),
        dimension_role="safety_gate",
    )
    prepare_post_trial_verifier(
        verifier,
        lock,
        repo_root=tmp_path,
        runtime_profile=_profile(),
        dimension_role="safety_gate",
    )
    archive = b"frozen task base"
    archive_digest = hashlib.sha256(archive).hexdigest()
    target = (
        tmp_path
        / POST_TRIAL_VERIFIER_RESOURCE_ROOT
        / archive_digest
        / "task-base.tar"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(archive)
    task = {
        "id": "task-one",
        # The Git-visible source is intentionally absent. Execution must consume
        # only the prepared, ignored .fugue resource selected by its digest.
        "resources": [{"path": "fixtures/task-base.tar"}],
    }
    expected = {
        "task_archive_sha256": archive_digest,
        "allowed_paths": ["app.js"],
    }
    return verifier, lock, task, expected, archive


def _runtime_receipts(
    kwargs: dict[str, Any], profile: ScorerRuntimeProfileV1
) -> tuple[dict[str, Any], dict[str, Any]]:
    files = kwargs["files"]
    input_files = [
        {
            "path": name,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in sorted(files.items())
    ]
    unsigned_input = {
        "schema_version": 1,
        "kind": "isolated_evaluator_input",
        "status": "bound",
        "files": input_files,
        "files_digest": stable_digest(input_files),
        "runtime_profile_id": profile.id,
        "runtime_profile_digest": profile.profile_digest,
        "runtime_image": profile.image,
        "runtime_platform": profile.platform,
        "writable_workdir": True,
    }
    unsigned_cleanup = {
        "schema_version": 1,
        "kind": "isolated_evaluator_cleanup",
        "status": "verified_absent",
        "container_name_sha256": "c" * 64,
        "runtime_profile_id": profile.id,
        "runtime_profile_digest": profile.profile_digest,
        "runtime_image": profile.image,
        "runtime_platform": profile.platform,
    }
    return (
        {**unsigned_input, "receipt_digest": stable_digest(unsigned_input)},
        {**unsigned_cleanup, "receipt_digest": stable_digest(unsigned_cleanup)},
    )


def _passing_runner(**kwargs: Any) -> dict[str, Any]:
    profile = kwargs["profile"]
    input_receipt, cleanup_receipt = _runtime_receipts(kwargs, profile)
    files = kwargs["files"]
    contract = json.loads(files["input.json"])
    output = json.loads(files["agent-output.json"])
    file_digests = {
        path: hashlib.sha256(content.encode()).hexdigest()
        for path, content in output["files"].items()
    }
    unsigned = {
        "schema_version": 1,
        "verifier_id": "fugue-node-test-v1",
        "task_id": contract["task_id"],
        "task_archive_sha256": contract["task_archive"]["sha256"],
        "agent_output_sha256": contract["agent_output"]["sha256"],
        "output_files_sha256": stable_digest(file_digests),
        "allowed_paths_digest": stable_digest(contract["allowed_paths"]),
        "runtime_lock_digest": contract["runtime_lock_digest"],
        "observed_node_version": "v22.0.0",
        "command": ["node", "--test"],
        "status": "passed",
        "exit_code": 0,
        "stdout_sha256": "e" * 64,
        "stderr_sha256": "f" * 64,
    }
    return {
        **unsigned,
        "receipt_digest": stable_digest(unsigned),
        "fugue_input_receipt": input_receipt,
        "fugue_runtime_receipt": cleanup_receipt,
    }


def test_public_source_lock_reconciles_frozen_inputs_and_detects_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_tasks = b'{"id":"task-one","input":{"question":"safe"}}\n'
    public_digest = hashlib.sha256(public_tasks).hexdigest()
    public_path = (
        tmp_path
        / ".fugue/runtime/comparison-inputs/tasksets"
        / f"{public_digest}.jsonl"
    )
    public_path.parent.mkdir(parents=True)
    public_path.write_bytes(public_tasks)

    compiled = b'{"id":"task-one","instruction":"safe"}\n'
    compiled_digest = hashlib.sha256(compiled).hexdigest()
    compiled_relative = ".fugue/runtime/comparisons/test/public-cases.jsonl"
    compiled_path = tmp_path / compiled_relative
    compiled_path.parent.mkdir(parents=True)
    compiled_path.write_bytes(compiled)

    resource = b"public task resource"
    resource_digest = hashlib.sha256(resource).hexdigest()
    resource_relative = (
        ".fugue/runtime/comparison-inputs/resources/"
        f"{resource_digest}/task.tar"
    )
    resource_path = tmp_path / resource_relative
    resource_path.parent.mkdir(parents=True)
    resource_path.write_bytes(resource)

    lock = _public_source_lock(
        public_tasks_sha256=public_digest,
        compiled_public_cases_sha256=compiled_digest,
        task_resources=(
            {
                "locked_relative": resource_relative,
                "sha256": resource_digest,
            },
        ),
    )

    assert _verified_public_source_lock(lock) == lock
    assert _observed_frozen_public_source_digest(
        lock,
        repo_root=tmp_path,
        compiled_public_cases_relative=compiled_relative,
    ) == lock["lock_digest"]

    spec = SimpleNamespace(spec_digest="a" * 64)
    frozen_inputs = {
        "public_source_lock": lock,
        "compiled_public_cases_relative": compiled_relative,
    }
    unsigned_receipt = {
        "schema_version": 1,
        "kind": "comparison_preparation",
        "spec_digest": spec.spec_digest,
        "frozen_inputs": frozen_inputs,
        "frozen_inputs_digest": stable_digest(frozen_inputs),
        "receipt_digest": "",
    }
    receipt_path = _comparison_preparation_receipt_path(
        spec,  # type: ignore[arg-type]
        repo_root=tmp_path,
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                **unsigned_receipt,
                "receipt_digest": stable_digest(unsigned_receipt),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "fugue.bench.comparison._resolved_public_source_lock",
        lambda _spec, *, repo_root: lock,
    )
    readiness = {
        "qualification_input_digests": {
            "public_source_lock": lock["lock_digest"]
        }
    }
    clean = _verify_local_public_source_drift(
        spec,  # type: ignore[arg-type]
        readiness=readiness,
        repo_root=tmp_path,
        env={},
    )
    assert clean.status == "matched"

    resource_path.write_bytes(b"mutated public task resource")
    assert _observed_frozen_public_source_digest(
        lock,
        repo_root=tmp_path,
        compiled_public_cases_relative=compiled_relative,
    ) != lock["lock_digest"]
    mutated = _verify_local_public_source_drift(
        spec,  # type: ignore[arg-type]
        readiness=readiness,
        repo_root=tmp_path,
        env={},
    )
    assert mutated.status == "drifted"


def test_verifier_lock_preparation_and_receipt_bind_every_identity(
    tmp_path: Path,
) -> None:
    verifier, lock, task, expected, archive = _prepared(tmp_path)
    output = {"files": {"app.js": "export const safe = true;"}}

    result = run_post_trial_verifier(
        verifier,
        lock,
        evaluator_id="public-behavioral",
        task=task,
        output=output,
        expected=expected,
        evidence={"attempt_id": "d" * 64, "task_id": "task-one"},
        repo_root=tmp_path,
        runtime_profile=_profile(),
        runner=_passing_runner,
    )

    assert result.passed is True
    receipt = result.receipt
    assert receipt["logical_attempt_id"] == "d" * 64
    assert receipt["task_id"] == "task-one"
    assert receipt["evaluator_id"] == "public-behavioral"
    assert receipt["normalized_output_digest"] == stable_digest(output)
    assert receipt["task_archive_sha256"] == hashlib.sha256(archive).hexdigest()
    assert receipt["verifier_source_sha256"] == lock.source_sha256
    assert receipt["runtime_profile_digest"] == _profile().profile_digest
    assert receipt["dimension_role"] == "safety_gate"
    assert result.scorer_evidence() == {
        "host_verifier": receipt["verifier_result"]
    }
    assert "task.tar" not in json.dumps(result.scorer_evidence())
    assert post_trial_verifier_lock_from_dict(lock.to_dict()) == lock


def test_verifier_receipt_identity_and_command_come_from_the_pinned_lock(
    tmp_path: Path,
) -> None:
    verifier_id = "fugue-skill-package-validator-v1"
    command = ["node", "skill-package-validate"]
    verifier, lock, task, expected, _archive = _prepared(
        tmp_path, verifier_id=verifier_id, command=command
    )

    def runner(**kwargs: Any) -> dict[str, Any]:
        payload = _passing_runner(**kwargs)
        unsigned = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "receipt_digest",
                "fugue_input_receipt",
                "fugue_runtime_receipt",
            }
        }
        unsigned["verifier_id"] = verifier_id
        unsigned["command"] = command
        return {
            **unsigned,
            "receipt_digest": stable_digest(unsigned),
            "fugue_input_receipt": payload["fugue_input_receipt"],
            "fugue_runtime_receipt": payload["fugue_runtime_receipt"],
        }

    result = run_post_trial_verifier(
        verifier,
        lock,
        evaluator_id="public-behavioral",
        task=task,
        output={"files": {"app.js": "export const safe = true;"}},
        expected=expected,
        evidence={"attempt_id": "d" * 64, "task_id": "task-one"},
        repo_root=tmp_path,
        runtime_profile=_profile(),
        runner=runner,
    )

    assert result.passed is True
    assert result.receipt["verifier_result"]["verifier_id"] == verifier_id
    assert result.receipt["verifier_result"]["command"] == command


def test_verifier_uses_the_public_attachment_lock_without_private_digest_duplication(
    tmp_path: Path,
) -> None:
    verifier, lock, _task, expected, archive = _prepared(tmp_path)
    archive_digest = hashlib.sha256(archive).hexdigest()
    task = {
        "id": "task-one",
        "attachments": [
            {
                "locked_relative": (
                    ".fugue/runtime/comparison-inputs/resources/"
                    f"{archive_digest}/task-base.tar"
                ),
                "sha256": archive_digest,
            }
        ],
    }
    public_only_expected = {"allowed_paths": expected["allowed_paths"]}

    result = run_post_trial_verifier(
        verifier,
        lock,
        evaluator_id="public-behavioral",
        task=task,
        output={"files": {"app.js": "export const safe = true;"}},
        expected=public_only_expected,
        evidence={"attempt_id": "d" * 64, "task_id": "task-one"},
        repo_root=tmp_path,
        runtime_profile=_profile(),
        runner=_passing_runner,
    )

    assert result.passed is True
    assert result.receipt["task_archive_sha256"] == archive_digest


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("source", "digest changed"),
        ("runtime", "declaration or runtime drifted"),
        ("base", "exact frozen base archive is unavailable"),
    ],
)
def test_verifier_fails_closed_on_source_runtime_or_base_drift(
    tmp_path: Path,
    kind: str,
    message: str,
) -> None:
    verifier, lock, task, expected, _archive = _prepared(tmp_path)
    profile = _profile()
    if kind == "source":
        prepared = prepare_post_trial_verifier(
            verifier,
            lock,
            repo_root=tmp_path,
            runtime_profile=profile,
            dimension_role="safety_gate",
        )
        prepared.chmod(0o600)
        prepared.write_text("changed", encoding="utf-8")
    elif kind == "runtime":
        profile = replace(profile, profile_digest="e" * 64)
    else:
        archive_path = next(
            (tmp_path / POST_TRIAL_VERIFIER_RESOURCE_ROOT).rglob("task-base.tar")
        )
        archive_path.write_bytes(b"changed archive")

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        run_post_trial_verifier(
            verifier,
            lock,
            evaluator_id="public-behavioral",
            task=task,
            output={"files": {}},
            expected=expected,
            evidence={"attempt_id": "d" * 64, "task_id": "task-one"},
            repo_root=tmp_path,
            runtime_profile=profile,
            runner=_passing_runner,
        )


def test_verifier_receipt_tamper_and_missing_attempt_fail_closed(
    tmp_path: Path,
) -> None:
    verifier, lock, task, expected, _archive = _prepared(tmp_path)
    output = {"files": {}}
    with pytest.raises(ValueError, match="logical attempt identity"):
        run_post_trial_verifier(
            verifier,
            lock,
            evaluator_id="public-behavioral",
            task=task,
            output=output,
            expected=expected,
            evidence={"task_id": "task-one"},
            repo_root=tmp_path,
            runtime_profile=_profile(),
            runner=_passing_runner,
        )

    result = run_post_trial_verifier(
        verifier,
        lock,
        evaluator_id="public-behavioral",
        task=task,
        output=output,
        expected=expected,
        evidence={"attempt_id": "d" * 64, "task_id": "task-one"},
        repo_root=tmp_path,
        runtime_profile=_profile(),
        runner=_passing_runner,
    )
    tampered = {**result.receipt, "task_id": "task-two"}
    with pytest.raises(ValueError, match="identity does not match"):
        verify_post_trial_verifier_receipt(
            tampered,
            lock=lock,
            task_id="task-one",
            logical_attempt_id="d" * 64,
            normalized_output_digest=stable_digest(output),
            agent_output_sha256=result.receipt["agent_output_sha256"],
            task_archive_sha256=expected["task_archive_sha256"],
        )


def test_receipted_inline_runtime_has_no_network_credentials_or_writable_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr("fugue.bench.task_authoring.shutil.which", lambda _: "/docker")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if command[1:3] == ["container", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[1:3] == ["rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"score":1,"reason":"passed","details":{}}',
            stderr="",
        )

    monkeypatch.setattr("fugue.bench.task_authoring.subprocess.run", fake_run)
    payload = run_isolated_evaluator(
        files={
            "verifier.cjs": b"process.stdout.write('{}');",
            "input.json": b"{}",
        },
        profile=_profile(),
        limits=TaskAuthoringLimitsV1(1, 1, 1, 1, 1, 1, 0, 0, 30, 256, 1.0, 64_000),
        writable_workdir=True,
        output_kind="object",
    )

    run_command, run_kwargs = calls[0]
    assert "--network" in run_command and "none" in run_command
    assert "--read-only" in run_command
    assert any(value.endswith(",readonly") for value in run_command)
    assert run_kwargs["env"].keys() == {"PATH"}
    assert "ANTHROPIC_API_KEY" not in run_kwargs["env"]
    assert payload["fugue_input_receipt"]["status"] == "bound"
    assert payload["fugue_runtime_receipt"]["status"] == "verified_absent"
    assert calls[-1][0][1:3] == ["container", "ls"]


def test_comparison_evaluator_binds_verifier_source_runtime_and_dimension() -> None:
    root = Path.cwd()
    evaluator = _evaluator(
        {
            "id": "public-behavioral",
            "type": "deterministic",
            "required": True,
            "scorer": (
                "examples/comparisons/community-skill-selected-v1/"
                "vercel-react-best-practices/scorer.py"
            ),
            "runtime": "python312-sandbox-v1",
            "verifier": {
                "type": "node_test",
                "source": (
                    "examples/comparisons/community-skill-selected-v1/"
                    "vercel-react-best-practices/host_node_verifier.cjs"
                ),
                "runtime_lock": (
                    "examples/comparisons/community-skill-selected-v1/"
                    "vercel-react-best-practices/host-verifier.lock.json"
                ),
                "runtime": "node22-verifier-v1",
                "dimension": "verification_passed",
            },
            "dimensions": ["verification_passed", "requested_behavior"],
            "dimension_roles": {
                "verification_passed": "safety_gate",
                "requested_behavior": "outcome",
            },
        }
    )

    digest = _evaluator_digest(evaluator, root)
    changed = replace(
        evaluator,
        verifier=replace(evaluator.verifier, dimension="requested_behavior"),
    )

    assert digest != _evaluator_digest(changed, root)
    assert evaluator.verifier is not None
    assert evaluator.to_dict()["verifier"]["runtime"] == "node22-verifier-v1"


def test_comparison_runs_verifier_before_public_scorer_and_retains_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verifier, lock, task, expected, _archive = _prepared(tmp_path)
    evaluator = ComparisonEvaluatorV1(
        id="public-behavioral",
        type="deterministic",
        required=True,
        scorer="unused.py",
        runtime="python312-sandbox-v1",
        verifier=verifier,
        dimensions=("public_regression_test_success",),
        dimension_roles={"public_regression_test_success": "safety_gate"},
    )
    raw_result = {"status": "passed", "receipt_digest": "a" * 64}
    outer_receipt = {"receipt_digest": "b" * 64}

    def fake_verifier(*args: Any, **kwargs: Any) -> PostTrialVerifierResultV1:
        assert kwargs["evidence"]["attempt_id"] == "d" * 64
        return PostTrialVerifierResultV1(
            passed=True,
            reason="passed",
            receipt={**outer_receipt, "verifier_result": raw_result},
        )

    def fake_scorer(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["evidence"] == {"host_verifier": raw_result}
        return {
            "score": 1.0,
            "reason": "verified",
            "details": {"public_regression_test_success": True},
        }

    monkeypatch.setattr(
        "fugue.bench.comparison.run_post_trial_verifier", fake_verifier
    )
    monkeypatch.setattr("fugue.bench.comparison._run_custom_scorer", fake_scorer)

    passed, scores, receipts = _score_deterministic_output(
        task=task,
        output={"files": {"app.js": "export const safe = true;"}},
        expected=expected,
        evidence={"attempt_id": "d" * 64, "task_id": "task-one"},
        evaluators=(evaluator,),
        repo_root=Path.cwd(),
        post_trial_verifier_locks={evaluator.id: lock},
        execute_post_trial_verifiers=True,
    )

    assert passed is True
    assert scores == {"public-behavioral.public_regression_test_success": True}
    assert receipts == {evaluator.id: {**outer_receipt, "verifier_result": raw_result}}


def test_readiness_qualifies_known_good_and_mutant_with_same_host_verifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verifier, lock, task, expected, _archive = _prepared(tmp_path)
    evaluator = ComparisonEvaluatorV1(
        id="public-behavioral",
        type="deterministic",
        required=True,
        scorer="unused.py",
        runtime="python312-sandbox-v1",
        verifier=verifier,
        dimensions=("public_regression_test_success",),
        dimension_roles={"public_regression_test_success": "safety_gate"},
    )
    prepared_modes: list[bool] = []

    def fake_verifier(*args: Any, **kwargs: Any) -> PostTrialVerifierResultV1:
        prepared_modes.append(kwargs["prepared"])
        assert kwargs["evidence"]["task_id"] == task["id"]
        assert len(kwargs["evidence"]["attempt_id"]) == 64
        passed = bool(kwargs["output"].get("good"))
        result = {"status": "passed" if passed else "failed"}
        return PostTrialVerifierResultV1(
            passed=passed,
            reason="qualified",
            receipt={"receipt_digest": "a" * 64, "verifier_result": result},
        )

    def fake_scorer(*args: Any, **kwargs: Any) -> dict[str, Any]:
        passed = kwargs["evidence"]["host_verifier"]["status"] == "passed"
        return {
            "score": float(passed),
            "reason": "qualified",
            "details": {"public_regression_test_success": passed},
        }

    monkeypatch.setattr(
        "fugue.bench.comparison._post_trial_verifier_lock",
        lambda *_args: lock,
    )
    monkeypatch.setattr(
        "fugue.bench.comparison.run_post_trial_verifier", fake_verifier
    )
    monkeypatch.setattr("fugue.bench.comparison._run_custom_scorer", fake_scorer)

    base_failures, gold_passes, blockers, warnings = _qualification_results(
        [task],
        [
            {
                "id": task["id"],
                "expected": expected,
                "base_output": {"good": False},
                "gold_output": {"good": True},
            }
        ],
        [evaluator],
        repo_root=Path.cwd(),
    )

    assert (base_failures, gold_passes) == (1, 1)
    assert blockers == []
    assert warnings == []
    assert prepared_modes == [False, False]
