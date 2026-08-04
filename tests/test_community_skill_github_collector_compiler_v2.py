from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "examples" / "comparisons" / "community-skill-upgrades"
COMPILER_PATH = COMMON / "compile_github_collector_plan_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "compile_github_collector_plan_v2", COMPILER_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PLANS = {
    "superpowers": ROOT
    / "examples/comparisons/superpowers-writing-plans-upgrade/conference-query-plan-v2.json",
    "anthropic": ROOT
    / "examples/comparisons/anthropic-skill-creator-upgrade/conference-github-candidate-query-plan-v2.json",
    "vercel": ROOT
    / "examples/comparisons/vercel-react-best-practices-upgrade/conference-query-plan-v2.json",
}
PROTOCOLS = {
    "superpowers": ROOT
    / "examples/comparisons/superpowers-writing-plans-upgrade/conference-sampling-frame-protocol-v2.json",
    "anthropic": ROOT
    / "examples/comparisons/anthropic-skill-creator-upgrade/conference-sampling-frame-protocol-v2.json",
    "vercel": ROOT
    / "examples/comparisons/vercel-react-best-practices-upgrade/conference-sampling-frame-protocol-v2.json",
}


def _artifact_digest(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_digest", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_protocol_bound_to_source(
    tmp_path: Path, lane: str, source_path: Path
) -> Path:
    protocol = json.loads(PROTOCOLS[lane].read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if lane in {"superpowers", "anthropic"}:
        protocol["query_plan"]["sha256"] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        protocol["query_plan"]["required_status"] = source["status"]
    else:
        source["artifact_digest"] = _artifact_digest(source)
        source_path.write_text(json.dumps(source), encoding="utf-8")
        protocol["query_plan"]["artifact_digest"] = source["artifact_digest"]
        protocol["artifact_digest"] = _artifact_digest(protocol)
    path = tmp_path / f"{lane}-bound-protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    (
        "lane",
        "expected_lane_id",
        "expected_queries",
        "expected_cutoff",
        "endpoint",
        "granularity",
    ),
    [
        (
            "superpowers",
            "superpowers-writing-plans-upgrade",
            34,
            "2026-07-31T23:59:59Z",
            "search/issues",
            "second",
        ),
        (
            "anthropic",
            "anthropic-skill-creator-upgrade",
            7,
            "2026-02-06T14:59:12Z",
            "search/repositories",
            "second",
        ),
        (
            "vercel",
            "vercel-react-best-practices-upgrade",
            27,
            "2026-01-17T23:59:59Z",
            "search/issues",
            "day",
        ),
    ],
)
def test_lane_plan_compiles_to_one_strict_collector_contract(
    lane: str,
    expected_lane_id: str,
    expected_queries: int,
    expected_cutoff: str,
    endpoint: str,
    granularity: str,
) -> None:
    path = PLANS[lane]
    protocol_path = PROTOCOLS[lane]
    compiled: dict[str, Any] = MODULE.compile_plan(lane, path, protocol_path)

    assert compiled["candidate_discovery_only"] is True
    assert compiled["lane_id"] == expected_lane_id
    assert compiled["acquisition_repetitions"] == 2
    assert compiled["selection_source_cutoff_utc"] == expected_cutoff
    assert (
        compiled["source_query_plan_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert (
        compiled["source_sampling_protocol_sha256"]
        == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    )
    assert compiled["sampling_protocol_id"] == MODULE._PROTOCOL_IDS[lane]
    assert compiled["max_response_bytes"] == 16 * 1024 * 1024
    assert len(compiled["queries"]) == expected_queries
    assert {query["endpoint"] for query in compiled["queries"]} == {endpoint}
    assert all(query["max_results_per_shard"] <= 900 for query in compiled["queries"])
    assert all(
        f"{query['date_qualifier']}:" not in query["query"]
        for query in compiled["queries"]
    )
    assert all(
        query["query"].split().count("is:public") == 1 for query in compiled["queries"]
    )
    assert {query["shard_granularity"] for query in compiled["queries"]} == {
        granularity
    }
    assert {query["unsplittable_overflow_policy"] for query in compiled["queries"]} == {
        "fail_closed_new_protocol_required"
    }
    assert compiled["public_visibility"] == {
        "required_query_qualifier": "is:public",
        "independent_repository_lookup": True,
        "persist_search_body_before_visibility_verification": False,
        "persist_repository_response_body": False,
    }
    assert compiled["credential_profile"] == {
        "profile_id": f"{expected_lane_id}-github-public-read-v2",
        "token_env_name": "GITHUB_TOKEN",
        "credential_required": True,
        "credential_value_serialized": False,
    }
    implementation = dict(compiled["collector_implementation"])
    lock_digest = implementation.pop("lock_sha256")
    assert lock_digest == MODULE.COLLECTOR.stable_digest(implementation)
    assert len(implementation["collector_sha256"]) == 64
    assert len(implementation["compiler_sha256"]) == 64
    assert len(implementation["source_commit"]) == 40
    assert len(implementation["source_tree"]) == 40
    assert MODULE.COLLECTOR.validate_query_plan(compiled) == compiled


def test_superpowers_and_skill_creator_have_explicitly_different_time_estimands() -> (
    None
):
    superpowers = MODULE.compile_plan(
        "superpowers", PLANS["superpowers"], PROTOCOLS["superpowers"]
    )
    anthropic = MODULE.compile_plan(
        "anthropic", PLANS["anthropic"], PROTOCOLS["anthropic"]
    )

    assert "post-treatment" in superpowers["temporal_scope"]
    assert "pre-treatment" in anthropic["temporal_scope"]


def test_compiler_refuses_unknown_lanes_and_ambiguous_temporal_ownership(
    tmp_path: Path,
) -> None:
    with pytest.raises(MODULE.CollectorPlanCompileError, match="unsupported lane"):
        MODULE.compile_plan("unknown", PLANS["superpowers"], PROTOCOLS["superpowers"])

    broken = (
        PLANS["superpowers"]
        .read_text()
        .replace("merged:{start}..{end}", "merged:2026-01-01..2026-01-02", 1)
    )
    path = tmp_path / "broken.json"
    path.write_text(broken)
    protocol = _write_protocol_bound_to_source(tmp_path, "superpowers", path)
    with pytest.raises(MODULE.CollectorPlanCompileError, match="compiler-owned"):
        MODULE.compile_plan("superpowers", path, protocol)


@pytest.mark.parametrize(
    ("lane", "path_parts", "replacement", "message"),
    [
        (
            "superpowers",
            ("observation_window", "end_inclusive_utc"),
            "2026-08-01T00:00:00Z",
            "locked lane protocol",
        ),
        (
            "anthropic",
            ("readiness", "authorizes_network_preparation"),
            True,
            "locked lane protocol",
        ),
        (
            "vercel",
            ("queries", 0, "query"),
            '"server action" authorization in:title,body is:pr is:merged '
            "merged:2023-01-01..2026-01-18 -repo:vercel-labs/agent-skills",
            "locked treatment preperiod",
        ),
    ],
)
def test_compiler_rejects_lane_semantic_boundary_drift(
    tmp_path: Path,
    lane: str,
    path_parts: tuple[str | int, ...],
    replacement: object,
    message: str,
) -> None:
    source = json.loads(PLANS[lane].read_text())
    target: Any = source
    for part in path_parts[:-1]:
        target = target[part]
    target[path_parts[-1]] = replacement
    path = tmp_path / f"{lane}-drift.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    protocol = _write_protocol_bound_to_source(tmp_path, lane, path)

    with pytest.raises(MODULE.CollectorPlanCompileError, match=message):
        MODULE.compile_plan(lane, path, protocol)


def test_compiler_wraps_strict_collector_validation_errors(tmp_path: Path) -> None:
    source = json.loads(PLANS["superpowers"].read_text())
    source["atomic_queries"][0]["id"] = "Unsafe Query ID"
    path = tmp_path / "invalid-collector-plan.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    protocol = _write_protocol_bound_to_source(tmp_path, "superpowers", path)

    with pytest.raises(
        MODULE.CollectorPlanCompileError, match="compiled collector plan is invalid"
    ):
        MODULE.compile_plan("superpowers", path, protocol)


def test_compiler_rejects_duplicate_json_and_symlink_ancestors(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        PLANS["superpowers"]
        .read_text()
        .replace(
            '"schema_version": 2',
            '"schema_version": 2, "schema_version": 2',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(MODULE.CollectorPlanCompileError, match="duplicate JSON key"):
        MODULE.compile_plan("superpowers", duplicate, PROTOCOLS["superpowers"])

    real = tmp_path / "real"
    real.mkdir()
    source_link = tmp_path / "source-link"
    source_link.symlink_to(PLANS["superpowers"])
    with pytest.raises(MODULE.CollectorPlanCompileError, match="symlink ancestor"):
        MODULE.compile_plan("superpowers", source_link, PROTOCOLS["superpowers"])

    protocol_link = tmp_path / "protocol-link"
    protocol_link.symlink_to(PROTOCOLS["superpowers"])
    with pytest.raises(MODULE.CollectorPlanCompileError, match="symlink ancestor"):
        MODULE.compile_plan("superpowers", PLANS["superpowers"], protocol_link)

    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real, target_is_directory=True)
    compiled = MODULE.compile_plan(
        "superpowers", PLANS["superpowers"], PROTOCOLS["superpowers"]
    )
    with pytest.raises(MODULE.CollectorPlanCompileError, match="symlink ancestor"):
        MODULE._write_compiled_plan(parent_link / "plan.json", compiled)


def test_compiler_cli_writes_owner_only_and_returns_stable_blocked_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "compiled" / "plan.json"
    assert (
        MODULE.main(
            [
                "--lane",
                "anthropic",
                str(PLANS["anthropic"]),
                "--protocol",
                str(PROTOCOLS["anthropic"]),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(capsys.readouterr().out)["status"] == "compiled"

    source = json.loads(PLANS["anthropic"].read_text())
    source["readiness"]["authorizes_preview"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(source), encoding="utf-8")
    invalid_protocol = _write_protocol_bound_to_source(tmp_path, "anthropic", invalid)
    blocked_output = tmp_path / "blocked.json"
    assert (
        MODULE.main(
            [
                "--lane",
                "anthropic",
                str(invalid),
                "--protocol",
                str(invalid_protocol),
                "--output",
                str(blocked_output),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert error["status"] == "blocked"
    assert "authorizes_preview" in error["error"]
    assert not blocked_output.exists()


@pytest.mark.parametrize(
    ("path_parts", "replacement", "message"),
    [
        (
            ("protocol_id",),
            "another-lane-sampling-protocol-v2",
            "protocol_id",
        ),
        (("status",), "sampled", "sampling protocol status"),
        (
            ("raw_public_data_privacy", "zero_finding_is_not_safety_proof"),
            False,
            "raw_public_data_privacy",
        ),
        (("query_plan", "sha256"), "0" * 64, "query_plan.sha256"),
    ],
)
def test_compiler_rejects_sampling_protocol_drift(
    tmp_path: Path,
    path_parts: tuple[str, ...],
    replacement: object,
    message: str,
) -> None:
    protocol = json.loads(PROTOCOLS["superpowers"].read_text(encoding="utf-8"))
    target: Any = protocol
    for part in path_parts[:-1]:
        target = target[part]
    target[path_parts[-1]] = replacement
    path = tmp_path / "drifted-protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(MODULE.CollectorPlanCompileError, match=message):
        MODULE.compile_plan("superpowers", PLANS["superpowers"], path)


def test_vercel_compiler_recomputes_self_digests_and_exact_query_reference(
    tmp_path: Path,
) -> None:
    source = json.loads(PLANS["vercel"].read_text(encoding="utf-8"))
    source["queries"][0]["query"] = source["queries"][0]["query"].replace(
        '"server action"', '"unrelated phrase"'
    )
    stale_source = tmp_path / "stale-source.json"
    stale_source.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(
        MODULE.CollectorPlanCompileError,
        match="source query plan.artifact_digest disagrees",
    ):
        MODULE.compile_plan("vercel", stale_source, PROTOCOLS["vercel"])

    source["artifact_digest"] = _artifact_digest(source)
    stale_source.write_text(json.dumps(source), encoding="utf-8")
    protocol = json.loads(PROTOCOLS["vercel"].read_text(encoding="utf-8"))
    protocol["query_plan"]["artifact_digest"] = source["artifact_digest"]
    protocol["query_plan"]["path"] = "another-query-plan.json"
    protocol["artifact_digest"] = _artifact_digest(protocol)
    wrong_path = tmp_path / "wrong-path-protocol.json"
    wrong_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(MODULE.CollectorPlanCompileError, match="query_plan.path"):
        MODULE.compile_plan("vercel", stale_source, wrong_path)

    protocol["query_plan"]["path"] = MODULE._QUERY_REFERENCE_PATHS["vercel"]
    stale_protocol = tmp_path / "stale-protocol.json"
    stale_protocol.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(
        MODULE.CollectorPlanCompileError,
        match="sampling protocol.artifact_digest disagrees",
    ):
        MODULE.compile_plan("vercel", stale_source, stale_protocol)


def test_compiled_plan_write_is_atomic_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = MODULE.compile_plan(
        "superpowers", PLANS["superpowers"], PROTOCOLS["superpowers"]
    )
    output = tmp_path / "compiled" / "plan.json"
    original_validate = MODULE.COLLECTOR.validate_query_plan

    def fail_reload(_value: object) -> dict[str, Any]:
        raise MODULE.CollectorPlanCompileError("forced reload failure")

    monkeypatch.setattr(MODULE.COLLECTOR, "validate_query_plan", fail_reload)
    with pytest.raises(MODULE.CollectorPlanCompileError, match="forced reload failure"):
        MODULE._write_compiled_plan(output, compiled)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []

    monkeypatch.setattr(MODULE.COLLECTOR, "validate_query_plan", original_validate)
    MODULE._write_compiled_plan(output, compiled)
    assert json.loads(output.read_text(encoding="utf-8")) == compiled
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_compiled_plan_recovers_after_post_link_parent_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = MODULE.compile_plan(
        "superpowers", PLANS["superpowers"], PROTOCOLS["superpowers"]
    )
    output = tmp_path / "post-link" / "plan.json"
    original_fsync_directory = MODULE._fsync_directory
    calls = 0

    def fail_after_link(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected post-link parent fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(MODULE, "_fsync_directory", fail_after_link)
    with pytest.raises(MODULE.CollectorPublicationRecoveryRequired):
        MODULE._write_compiled_plan(output, compiled)
    assert output.is_file()
    assert MODULE._compiled_publication_marker(output).is_file()

    monkeypatch.setattr(MODULE, "_fsync_directory", original_fsync_directory)
    MODULE._write_compiled_plan(output, compiled)
    assert json.loads(output.read_text(encoding="utf-8")) == compiled
    assert not MODULE._compiled_publication_marker(output).exists()
    assert not list(output.parent.glob(".plan.json.staging-*"))
