from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from fugue.bench.candidates import stable_digest
from fugue.bench.provider_contract import (
    CellRequestV1,
    candidate_bundle_from_dict,
    cell_request_from_dict,
    cell_result_from_dict,
    provider_contract_schemas,
    provider_descriptor_from_dict,
    suite_bundle_from_dict,
)
from fugue.bench.providers import (
    ProviderClient,
    ProviderProcessError,
    load_provider_lock,
    lock_provider,
    provider_conformance,
    scaffold_provider,
)


def _fake_command() -> str:
    return f"{sys.executable} -m fugue.providers.fake"


def test_fake_provider_crosses_full_offline_contract(tmp_path: Path) -> None:
    lock_path = tmp_path / "provider.lock.json"
    lock = lock_provider(_fake_command(), output=lock_path)
    loaded = load_provider_lock(lock_path)
    assert loaded == lock

    client = ProviderClient.from_lock(lock)
    suite, private = client.resolve_suite("conformance")
    assert len(suite.tasks) == 2
    assert len(suite.scenarios) == 1
    assert "expected" not in json.dumps(suite.to_dict())
    assert private.tasks[0].expected
    report = provider_conformance(
        provider_lock=lock_path,
        candidate_ref="reference",
        suite_ref="conformance",
    )
    assert report["conformant"] is True
    assert report["evaluator_issues"] == []

def test_fake_provider_prepares_and_runs_one_locked_cell(tmp_path: Path) -> None:
    lock_path = tmp_path / "provider.lock.json"
    lock = lock_provider(_fake_command(), output=lock_path)
    client = ProviderClient.from_lock(lock)
    candidate = client.resolve_candidate("reference")
    suite, _private = client.resolve_suite("conformance")
    preparation = client.prepare(
        provider_lock_digest=lock.lock_digest,
        candidate=candidate,
        suite=suite,
    )
    assert {
        value["id"]: value["digest"]
        for value in preparation.materialized_resources
        if value["kind"] == "provider-task-v1"
    } == {
        task.id: stable_digest(task.to_dict()) for task in suite.tasks
    }
    unsigned = CellRequestV1(
        schema_version=1,
        provider_id=lock.provider_id,
        plan_digest=stable_digest("plan"),
        cell_id="conformance-cell",
        candidate_digest=candidate.bundle_digest,
        suite_digest=suite.bundle_digest,
        preparation_receipt_digest=preparation.receipt_digest,
        candidate=candidate.to_dict(),
        preparation=preparation.to_dict(),
        task=suite.tasks[0].to_dict(),
        attempt=1,
        runtime_lock_digest=stable_digest("runtime"),
        credential_profile_names=("test-profile",),
        budget={"max_cost_usd": 0.01, "max_seconds": 10},
    )
    request = cell_request_from_dict(unsigned.to_dict())
    result = client.run_cell(request)
    assert result.status == "succeeded"
    assert result.output == {"answer": "evidence"}
    assert result.request_digest == request.request_digest
    assert len(result.conversation) == 2


def test_provider_conformance_reports_cell_evidence_and_cleanup(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "provider.lock.json"
    lock_provider(_fake_command(), output=lock_path)

    report = provider_conformance(
        provider_lock=lock_path,
        candidate_ref="reference",
        suite_ref="conformance",
        exercise_run_cell=True,
    )

    assert report["conformant"] is True
    assert report["scope"] == "offline_protocol_conformance"
    assert len(report["protocol_cell_results"]) == 2
    assert all(
        row["failure"] is None for row in report["protocol_cell_results"]
    )
    assert all(
        row["usage"]["input_tokens"] == 1
        for row in report["protocol_cell_results"]
    )
    assert all(row["evidence_refs"] for row in report["protocol_cell_results"])
    assert all(
        row["cleanup"] == {"complete": True, "remaining_resources": 0}
        for row in report["protocol_cell_results"]
    )
    assert "evaluation_results" not in report
    assert report["task_outcomes_qualified"] is False
    assert "not a Fugue experiment" in report["claim_limitation"]


def test_provider_conformance_can_select_one_exact_task(tmp_path: Path) -> None:
    lock_path = tmp_path / "provider.lock.json"
    lock_provider(_fake_command(), output=lock_path)

    report = provider_conformance(
        provider_lock=lock_path,
        candidate_ref="reference",
        suite_ref="conformance",
        exercise_run_cell=True,
        task_ids=["structured-multi-turn"],
    )

    assert report["selected_task_ids"] == ["structured-multi-turn"]
    assert report["selected_task_count"] == 1
    assert [row["task_id"] for row in report["protocol_cell_results"]] == [
        "structured-multi-turn"
    ]


def test_provider_conformance_rejects_unknown_selected_task(tmp_path: Path) -> None:
    lock_path = tmp_path / "provider.lock.json"
    lock_provider(_fake_command(), output=lock_path)

    with pytest.raises(ValueError, match="unknown provider conformance task"):
        provider_conformance(
            provider_lock=lock_path,
            candidate_ref="reference",
            suite_ref="conformance",
            task_ids=["missing-task"],
        )


def test_cell_result_digest_preserves_integer_usage_values() -> None:
    raw = {
        "schema_version": 1,
        "provider_id": "reference",
        "cell_id": "cell-1",
        "request_digest": stable_digest("request"),
        "status": "succeeded",
        "output": {"answer": "evidence"},
        "conversation": [],
        "tool_calls": [],
        "usage": {"tool_calls": 1, "latency_ms": 10.5},
        "evidence_refs": [],
        "failure": None,
        "cleanup": {"complete": True, "remaining_resources": 0},
        "result_digest": "",
    }
    raw["result_digest"] = stable_digest(raw)

    result = cell_result_from_dict(raw)

    assert result.usage == {"tool_calls": 1, "latency_ms": 10.5}
    assert result.result_digest == raw["result_digest"]


def test_candidate_identity_changes_for_behavior_not_execution() -> None:
    client = ProviderClient.from_command_text(_fake_command())
    reference = client.resolve_candidate("reference")
    candidate = client.resolve_candidate("candidate")
    assert reference.bundle_digest != candidate.bundle_digest
    assert reference.agent_code["digest"] != candidate.agent_code["digest"]


def test_provider_artifacts_fail_closed_on_unknown_fields() -> None:
    descriptor = ProviderClient.from_command_text(_fake_command()).describe().to_dict()
    descriptor["surprise"] = True
    with pytest.raises(ValueError, match="unknown field"):
        provider_descriptor_from_dict(descriptor)

    candidate = (
        ProviderClient.from_command_text(_fake_command())
        .resolve_candidate("reference")
        .to_dict()
    )
    candidate["agent_code"]["files"][0]["surprise"] = True
    candidate["bundle_digest"] = ""
    with pytest.raises(ValueError, match="unknown field"):
        candidate_bundle_from_dict(candidate)


def test_provider_digest_rejects_drift() -> None:
    descriptor = ProviderClient.from_command_text(_fake_command()).describe().to_dict()
    descriptor["display_name"] = "drifted"
    with pytest.raises(ValueError, match="does not match"):
        provider_descriptor_from_dict(descriptor)


def test_provider_digest_preserves_integer_number_identity() -> None:
    client = ProviderClient.from_command_text(_fake_command())
    suite, _private = client.resolve_suite("conformance")
    raw = suite.to_dict()
    raw["tasks"][0]["stopping_policy"][0]["limit"] = 3
    raw["bundle_digest"] = stable_digest(
        {**raw, "bundle_digest": ""}
    )

    reparsed = suite_bundle_from_dict(raw)

    assert reparsed.tasks[0].stopping_policy[0]["limit"] == 3
    assert isinstance(reparsed.tasks[0].stopping_policy[0]["limit"], int)
    assert reparsed.bundle_digest == raw["bundle_digest"]


def test_provider_timeout_is_structured() -> None:
    client = ProviderClient(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_sec=0.01,
    )
    with pytest.raises(ProviderProcessError, match="timed out") as captured:
        client.invoke("describe", {})
    assert captured.value.operation == "describe"


def test_provider_malformed_output_is_structured() -> None:
    client = ProviderClient([sys.executable, "-c", "print('not-json')"])
    with pytest.raises(ProviderProcessError, match="exactly one JSON value") as captured:
        client.invoke("describe", {})
    assert captured.value.operation == "describe"


def test_provider_command_does_not_use_a_shell(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    client = ProviderClient(
        [
            sys.executable,
            "-c",
            "import json; print(json.dumps({'value': __import__('sys').argv[1]}))",
            f";touch {marker}",
        ]
    )
    assert client.invoke("describe", {})["value"] == f";touch {marker}"
    assert not marker.exists()


def test_provider_lock_rejects_executable_drift(tmp_path: Path) -> None:
    provider_path, _readme = scaffold_provider(
        tmp_path / "provider", provider_id="drift-provider"
    )
    lock_path = tmp_path / "provider.lock.json"
    lock_provider(str(provider_path), output=lock_path)
    provider_path.write_text(provider_path.read_text() + "\n# drift\n")
    with pytest.raises(ValueError, match="executable bytes differ"):
        load_provider_lock(lock_path)


def test_provider_schemas_are_strict_and_cover_every_public_artifact() -> None:
    schemas = provider_contract_schemas()
    assert set(schemas) == {
        "provider-descriptor-v1",
        "candidate-bundle-v1",
        "suite-bundle-v1",
        "private-evaluation-bundle-v1",
        "preparation-receipt-v1",
        "cell-request-v1",
        "cell-result-v1",
    }
    assert all(schema["additionalProperties"] is False for schema in schemas.values())
    client = ProviderClient.from_command_text(_fake_command())
    descriptor = client.describe()
    candidate = client.resolve_candidate("reference")
    suite, private = client.resolve_suite("conformance")
    preparation = client.prepare(
        provider_lock_digest=stable_digest("provider-lock"),
        candidate=candidate,
        suite=suite,
    )
    request = cell_request_from_dict(
        CellRequestV1(
            schema_version=1,
            provider_id=descriptor.provider_id,
            plan_digest=stable_digest("plan"),
            cell_id="schema-cell",
            candidate_digest=candidate.bundle_digest,
            suite_digest=suite.bundle_digest,
            preparation_receipt_digest=preparation.receipt_digest,
            candidate=candidate.to_dict(),
            preparation=preparation.to_dict(),
            task=suite.tasks[0].to_dict(),
            attempt=1,
            runtime_lock_digest=stable_digest("runtime"),
            credential_profile_names=(),
            budget={"max_seconds": 10},
        ).to_dict()
    )
    result = client.run_cell(request)
    artifacts = {
        "provider-descriptor-v1": descriptor.to_dict(),
        "candidate-bundle-v1": candidate.to_dict(),
        "suite-bundle-v1": suite.to_dict(),
        "private-evaluation-bundle-v1": private.to_dict(),
        "preparation-receipt-v1": preparation.to_dict(),
        "cell-request-v1": request.to_dict(),
        "cell-result-v1": result.to_dict(),
    }
    for name, artifact in artifacts.items():
        Draft202012Validator(schemas[name]).validate(artifact)


def test_provider_schema_files_are_current_cross_language_goldens() -> None:
    schemas = provider_contract_schemas()
    root = Path("schemas/fugue/providers")
    assert {
        path.name for path in root.glob("*.schema.json")
    } == {f"{name}.schema.json" for name in schemas}
    for name, schema in schemas.items():
        assert json.loads((root / f"{name}.schema.json").read_text()) == schema


def test_provider_scaffold_is_dependency_free_and_validates(
    tmp_path: Path,
) -> None:
    provider_path, readme_path = scaffold_provider(
        tmp_path / "provider", provider_id="third-party"
    )
    assert provider_path.stat().st_mode & 0o111
    assert "third-party" in readme_path.read_text()
    descriptor = ProviderClient([str(provider_path)]).describe()
    assert descriptor.provider_id == "third-party"
