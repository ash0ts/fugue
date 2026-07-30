from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from fugue.bench import mcp_release_qualification
from fugue.bench.comparison import check_comparison, load_comparison
from fugue.bench.mcp_release_qualification import (
    QUALIFICATION_PROJECT,
    QUALIFICATION_RESULT_PROJECT,
    QUALIFICATION_SOURCE_PROJECT,
    _evidence_lock,
    _mcp_release_qualification_receipt,
    build_hosted_source_conformance_receipt,
    evidence_result_project,
    evidence_source_project,
    qualification_seed,
    qualification_seed_digest,
    validate_evidence_lock,
    validate_release_notes_lock,
)
from fugue.research.comparisons import ComparisonRegistry

EXAMPLE = Path("examples/comparisons/wandb-mcp-maintenance")


def test_research_registry_exposes_only_canonical_mcp_v3_studies() -> None:
    registry = ComparisonRegistry.from_file(Path.cwd())
    mcp_entries = tuple(
        entry
        for entry in registry.catalog()
        if entry.path.startswith(
            "examples/comparisons/wandb-mcp-maintenance/"
        )
    )

    assert {
        (entry.id, entry.path)
        for entry in mcp_entries
    } == {
        (
            "mcp-main-vs-0-4-natural-maintainer-canary-v3",
            "examples/comparisons/wandb-mcp-maintenance/"
            "natural-maintainer-canary-local-v3.yaml",
        ),
        (
            "mcp-main-vs-0-4-natural-maintainer-confirmation-v3",
            "examples/comparisons/wandb-mcp-maintenance/"
            "natural-maintainer-confirmation-local-v3.yaml",
        ),
    }


def test_release_notes_lock_binds_exact_rc_source_and_all_classifications() -> None:
    release_notes = validate_release_notes_lock(
        json.loads((EXAMPLE / "release-notes.lock.json").read_text())
    )
    receipt = _mcp_release_qualification_receipt(_lock(), [])

    assert release_notes["commit"] == (
        "3dd4447ef0054d4707aafc515e3f2ddfb11b17bd"
    )
    assert release_notes["sha256"] == (
        "2e32e337dd6c98a5e4b3805b189af10c913ec1dd739a63b25b031ab35d786c99"
    )
    assert {
        item["release_note"] for item in receipt["release_note_classification"]
    } == set(release_notes["behaviors"])


def _lock(
    project: str = QUALIFICATION_PROJECT,
    *,
    result_project: str | None = None,
) -> dict:
    return _evidence_lock(
        project,
        [
            {
                "id": f"run-{index}",
                "ref": f"wandb-run:///{project}/run-{index}",
            }
            for index in range(6)
        ],
        {
            "dataset": {
                "name": "cases",
                "ref": f"weave:///{project}/object/cases:v1",
                "rows": 8,
            },
            "source_conversations": [
                {
                    "call_id": f"call-{index}",
                    "ref": f"weave:///{project}/call/call-{index}",
                }
                for index in range(24)
            ],
            "evaluations": [
                {
                    "revision": revision,
                    "ref": f"weave:///{project}/object/evaluation-{revision}:v1",
                    "call_ref": f"weave:///{project}/call/evaluation-{revision}",
                    "call_id": f"evaluation-{revision}",
                    "prediction_rows": 8,
                }
                for revision in ("maintainer-r17", "maintainer-r18")
            ],
        },
        result_project=result_project,
    )


def _source_call_snapshot(
    lock: dict,
    *,
    cross_project_child: bool = False,
) -> tuple[list[dict], dict[str, list[dict]]]:
    project = evidence_source_project(lock)
    roots = []
    children = {}
    for evaluation in lock["objects"]["evaluations"]:
        call_id = evaluation["call_id"]
        roots.append(
            {
                "id": call_id,
                "project_id": project,
                "op_name": (
                    f"weave:///{project}/op/Evaluation.evaluate:root-digest"
                ),
            }
        )
        rows = [
            {
                "id": f"{call_id}-prediction-{index}",
                "project_id": (
                    QUALIFICATION_PROJECT
                    if cross_project_child and index == 0
                    else project
                ),
                "parent_id": call_id,
                "op_name": (
                    f"weave:///{project}/op/"
                    "Evaluation.predict_and_score:prediction-digest"
                ),
            }
            for index in range(8)
        ]
        rows.append(
            {
                "id": f"{call_id}-summary",
                "project_id": project,
                "parent_id": call_id,
                "op_name": (
                    f"weave:///{project}/op/"
                    "Evaluation.summarize:summary-digest"
                ),
            }
        )
        children[call_id] = rows
    return roots, children


def test_seed_has_real_nonzero_evidence_and_actionable_anomaly() -> None:
    seed = qualification_seed()
    facts = seed["facts"]

    assert seed["project"] == QUALIFICATION_PROJECT
    assert facts["run_count"] == 6
    assert facts["source_conversation_count"] == 24
    assert facts["evaluation_prediction_rows"] == 16
    assert facts["latency_anomaly"] == {
        "attempt_label": "broad-history-scan-claude",
        "latency_ms": 4200,
        "cohort_median_ms": 920,
        "ratio": 4.5652,
    }
    assert facts["cost_coverage"] == {
        "attempts": 6,
        "attempts_with_observed_cost": 5,
        "total_observed_usd": 0.96,
        "complete": False,
    }
    assert facts["regressions"] == ["partial-evidence"]
    assert len(qualification_seed_digest()) == 64


def test_split_source_and_result_lock_preserves_legacy_seed_behavior() -> None:
    legacy = _lock()
    split = _lock(
        QUALIFICATION_SOURCE_PROJECT,
        result_project=QUALIFICATION_RESULT_PROJECT,
    )

    assert evidence_source_project(legacy) == QUALIFICATION_PROJECT
    assert evidence_result_project(legacy) == QUALIFICATION_PROJECT
    assert "source_project" not in legacy
    assert qualification_seed()["project"] == QUALIFICATION_PROJECT
    assert qualification_seed_digest() == legacy["seed_digest"]

    assert evidence_source_project(split) == QUALIFICATION_SOURCE_PROJECT
    assert evidence_result_project(split) == QUALIFICATION_RESULT_PROJECT
    assert split["project"] == QUALIFICATION_SOURCE_PROJECT
    assert split["seed_digest"] == qualification_seed_digest(
        source_project=QUALIFICATION_SOURCE_PROJECT
    )
    assert (
        validate_evidence_lock(
            split,
            expected_project=None,
            expected_source_project=QUALIFICATION_SOURCE_PROJECT,
            expected_result_project=QUALIFICATION_RESULT_PROJECT,
        )
        == split
    )


def test_source_conformance_receipt_binds_exact_18_to_16_shape() -> None:
    lock = _lock(
        QUALIFICATION_SOURCE_PROJECT,
        result_project=QUALIFICATION_RESULT_PROJECT,
    )
    roots, children = _source_call_snapshot(lock)

    receipt = build_hosted_source_conformance_receipt(
        evidence_lock=lock,
        evaluation_roots=roots,
        direct_children=children,
        created_at="2026-07-30T00:00:00Z",
    )

    assert receipt["status"] == "passed"
    assert receipt["source_project"] == QUALIFICATION_SOURCE_PROJECT
    assert receipt["result_project"] == QUALIFICATION_RESULT_PROJECT
    assert receipt["observed"] == {
        "evaluation_roots": 2,
        "direct_children": 18,
        "predict_and_score_children": 16,
        "summarize_children": 2,
    }
    assert receipt["expectations"]["direct_children"] == 18
    assert (
        receipt["expectations"]["repaired_candidate_prediction_children"]
        == 16
    )
    assert receipt["query_scope"]["models_invoked"] == 0
    assert receipt["query_scope"]["calls_published"] == 0
    assert receipt["blockers"] == []
    assert len(receipt["source_snapshot_digest"]) == 64
    assert len(receipt["receipt_digest"]) == 64


def test_source_conformance_rejects_cross_project_or_drifted_children() -> None:
    lock = _lock(
        QUALIFICATION_SOURCE_PROJECT,
        result_project=QUALIFICATION_RESULT_PROJECT,
    )
    roots, children = _source_call_snapshot(
        lock,
        cross_project_child=True,
    )
    first_root = next(iter(children))
    children[first_root].pop()

    receipt = build_hosted_source_conformance_receipt(
        evidence_lock=lock,
        evaluation_roots=roots,
        direct_children=children,
        created_at="2026-07-30T00:00:00Z",
    )

    assert receipt["status"] == "failed"
    assert "aggregate_direct_children_drift" in receipt["blockers"]
    assert "aggregate_summarize_children_drift" in receipt["blockers"]
    assert any(
        item.endswith(":child_project_mismatch")
        for item in receipt["blockers"]
    )


def test_zero_model_verifier_reads_only_source_and_redacts_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock(
        QUALIFICATION_SOURCE_PROJECT,
        result_project=QUALIFICATION_RESULT_PROJECT,
    )
    evidence_lock = tmp_path / "evidence.lock.json"
    evidence_lock.write_text(json.dumps(lock), encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=secret-verifier-key\n", encoding="utf-8")
    output = tmp_path / "source-conformance.json"
    roots, children = _source_call_snapshot(lock)
    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return roots, children

    def reject_weave_init(*args, **kwargs):
        raise AssertionError("source verifier must not initialize Weave")

    monkeypatch.setattr(
        mcp_release_qualification,
        "_fetch_hosted_source_calls",
        fake_fetch,
    )
    monkeypatch.setattr(
        mcp_release_qualification.weave,
        "init",
        reject_weave_init,
    )

    receipt = mcp_release_qualification.verify_hosted_source_conformance(
        evidence_lock=evidence_lock,
        env_file=env_file,
        output=output,
    )

    assert receipt["status"] == "passed"
    assert captured["source_project"] == QUALIFICATION_SOURCE_PROJECT
    assert captured["api_key"] == "secret-verifier-key"
    assert "secret-verifier-key" not in output.read_text(encoding="utf-8")


def test_evidence_lock_requires_exact_counts_and_immutable_refs() -> None:
    lock = _lock()

    assert validate_evidence_lock(lock) == lock

    wrong_count = _lock()
    wrong_count["counts"]["runs"] = 0
    wrong_count["evidence_lock_digest"] = ""
    from fugue.bench.candidates import stable_digest

    wrong_count["evidence_lock_digest"] = stable_digest(wrong_count)
    with pytest.raises(ValueError, match="runs must equal 6"):
        validate_evidence_lock(wrong_count)

    mutable_ref = _lock()
    mutable_ref["objects"]["dataset"]["ref"] = "wandb/latest"
    mutable_ref["evidence_lock_digest"] = ""
    mutable_ref["evidence_lock_digest"] = stable_digest(mutable_ref)
    with pytest.raises(ValueError, match="Dataset reference is not immutable"):
        validate_evidence_lock(mutable_ref)


def test_existing_lock_must_validate_before_idempotent_reuse(
    tmp_path: Path,
) -> None:
    from fugue.bench.mcp_release_qualification import prepare_hosted_project

    output = tmp_path / "evidence.lock.json"
    output.write_text('{"schema_version": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="digest does not match"):
        prepare_hosted_project(
            project=QUALIFICATION_PROJECT,
            output=output,
            env_file=tmp_path / "missing.env",
        )


def test_checked_in_hosted_lock_is_exact_and_contains_no_credentials() -> None:
    lock = json.loads((EXAMPLE / "evidence.lock.json").read_text(encoding="utf-8"))

    assert validate_evidence_lock(lock) == lock
    serialized = json.dumps(lock, sort_keys=True)
    assert "WANDB_API_KEY" not in serialized
    assert "ANTHROPIC_API_KEY" not in serialized
    assert "sk-ant-" not in serialized
    assert all(
        item["ref"].startswith("weave:///")
        for item in lock["objects"]["source_conversations"]
    )


@pytest.mark.parametrize(
    ("filename", "tasks", "cells", "cap", "candidate_passes"),
    [
        ("natural-maintainer-canary-local-v3.yaml", 2, 8, 10, 4),
        ("natural-maintainer-confirmation-local-v3.yaml", 4, 16, 20, 8),
    ],
)
def test_v3_natural_maintainer_specs_are_exact_source_isolated_studies(
    filename: str,
    tasks: int,
    cells: int,
    cap: int,
    candidate_passes: int,
) -> None:
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / filename, repo_root=root)
    readiness = check_comparison(spec, repo_root=root)

    assert spec.schema_version == 3
    assert spec.baseline.integrations == ({"id": "wandb-mcp-main"},)
    assert spec.candidate.integrations == (
        {"id": "wandb-mcp-0-4-staging"},
    )
    assert spec.execution.source_evidence_project == QUALIFICATION_SOURCE_PROJECT
    assert spec.execution.evidence_project == QUALIFICATION_RESULT_PROJECT
    assert (
        spec.execution.source_evidence_destination.project
        == "fugue-mcp-release-source-v1"
    )
    assert (
        spec.execution.evidence_destination.project
        == "fugue-mcp-release-qualification-v1"
    )
    assert spec.execution.environment == {"type": "docker"}
    assert spec.execution.harnesses == ("claude-code",)
    assert spec.execution.attempts == 2
    assert spec.execution.concurrency == 1
    assert spec.execution.evidence_checkpoint_cells == 1
    assert spec.execution.preparation_required is True
    assert spec.execution.max_cost_usd == cap
    assert readiness.task_count == tasks
    assert readiness.estimated_cells == cells
    assert readiness.estimated_cost_usd == cap
    assert readiness.base_failures == tasks
    assert readiness.gold_passes == tasks
    assert spec.decision_policy is not None
    assert spec.decision_policy.candidate_sha == (
        "3dd4447ef0054d4707aafc515e3f2ddfb11b17bd"
    )
    gates = {gate.source: gate.target for gate in spec.decision_policy.gates}
    assert gates["matrix.rows"] == cells
    assert gates["task.candidate_passed"] == candidate_passes
    assert gates["infrastructure.gate.final-staging-head"] is True
    assert (
        gates["infrastructure.gate.human-maintainer-actionability-review"]
        is True
    )
    assert gates["infrastructure.gate.fresh-wheel-python-3-11"] is True
    assert gates["infrastructure.gate.fresh-wheel-python-3-12"] is True
    assert any(
        "infrastructure receipt is not usable" in blocker
        for blocker in readiness.blockers
    )


@pytest.mark.parametrize(
    ("task_filename", "private_filename", "expected_count"),
    [
        (
            "natural-maintainer-canary-tasks.jsonl",
            "natural-maintainer-canary-private.jsonl",
            2,
        ),
        (
            "natural-maintainer-confirmation-tasks.jsonl",
            "natural-maintainer-confirmation-private.jsonl",
            4,
        ),
    ],
)
def test_v3_natural_maintainer_tasks_keep_truth_host_only(
    task_filename: str,
    private_filename: str,
    expected_count: int,
) -> None:
    public_tasks = [
        json.loads(line)
        for line in (EXAMPLE / task_filename).read_text().splitlines()
        if line.strip()
    ]
    private_labels = [
        json.loads(line)
        for line in (EXAMPLE / private_filename).read_text().splitlines()
        if line.strip()
    ]

    assert len(public_tasks) == len(private_labels) == expected_count
    assert {item["id"] for item in public_tasks} == {
        item["id"] for item in private_labels
    }
    assert all("expected" not in item for item in public_tasks)
    assert all(
        {
            "id",
            "expected",
            "base_output",
            "base_evidence",
            "gold_output",
            "gold_evidence",
        }
        == set(item)
        for item in private_labels
    )
    public = json.dumps(public_tasks, sort_keys=True)
    assert "base_output" not in public
    assert "gold_output" not in public
    assert "evaluation_root_count\": 2" not in public
    assert "largest_latency_ms\": 4200" not in public


def _natural_maintainer_score():
    return runpy.run_path(
        (EXAMPLE / "natural_maintainer_scorer.py").as_posix()
    )["score"]


@pytest.mark.parametrize(
    "private_filename",
    [
        "natural-maintainer-canary-private.jsonl",
        "natural-maintainer-confirmation-private.jsonl",
    ],
)
def test_natural_maintainer_scorer_qualifies_gold_and_rejects_known_bad(
    private_filename: str,
) -> None:
    score = _natural_maintainer_score()
    labels = [
        json.loads(line)
        for line in (EXAMPLE / private_filename).read_text().splitlines()
        if line.strip()
    ]
    for label in labels:
        task = {"id": label["id"]}
        base = score(
            task,
            label["base_output"],
            {**label["base_evidence"], "expected": label["expected"]},
        )
        gold = score(
            task,
            label["gold_output"],
            {**label["gold_evidence"], "expected": label["expected"]},
        )
        assert not all(base.values()), label["id"]
        assert all(gold.values()), label["id"]


def test_natural_maintainer_scorer_requires_one_json_and_actual_source_scope() -> None:
    score = _natural_maintainer_score()
    label = json.loads(
        (EXAMPLE / "natural-maintainer-canary-private.jsonl")
        .read_text()
        .splitlines()[0]
    )
    task = {"id": label["id"]}
    evidence = {**label["gold_evidence"], "expected": label["expected"]}
    raw = json.dumps(label["gold_output"])

    assert all(score(task, raw, evidence).values())
    assert all(score(task, f"```json\n{raw}\n```", evidence).values())
    prose = score(task, f"Result:\n```json\n{raw}\n```", evidence)
    assert prose["answer_correct"] is False
    assert prose["locked_source_scope"] is True
    duplicate = score(
        task,
        f"```json\n{raw}\n```\n```json\n{raw}\n```",
        evidence,
    )
    assert duplicate["answer_correct"] is False

    cross_project = {
        **evidence,
        "mcp_queried_projects": [
            QUALIFICATION_SOURCE_PROJECT,
            QUALIFICATION_RESULT_PROJECT,
        ],
    }
    result = score(task, label["gold_output"], cross_project)
    assert result["answer_correct"] is True
    assert result["locked_source_scope"] is False


def test_natural_maintainer_scorer_normalizes_raw_mcp_arguments() -> None:
    score = _natural_maintainer_score()
    label = json.loads(
        (EXAMPLE / "natural-maintainer-confirmation-private.jsonl")
        .read_text()
        .splitlines()[0]
    )
    evidence = {
        "expected": label["expected"],
        "mcp_tool_calls": [
            {
                "tool": "query_wandb_tool",
                "arguments": {
                    "entity_name": "wandb",
                    "project_name": "fugue-mcp-release-source-v1",
                    "resource": "runs",
                    "response_mode": "count",
                },
                "terminal_status": "succeeded",
                "total_count": 6,
            },
            {
                "tool": "query_wandb_tool",
                "arguments": {
                    "entity_name": "wandb",
                    "project_name": "fugue-mcp-release-source-v1",
                    "resource": "runs",
                    "response_mode": "items",
                    "columns": ["id"],
                    "config_keys": ["attempt_label"],
                    "summary_keys": ["latency_ms"],
                    "limit": 50,
                },
                "terminal_status": "succeeded",
                "returned_count": 6,
                "has_more": False,
                "project_exhaustive": True,
                "truncation_applied": False,
            },
            {
                "tool": "summarize_evaluation_tool",
                "arguments": {
                    "entity_name": "wandb",
                    "project_name": "fugue-mcp-release-source-v1",
                    "max_evals": 25,
                },
                "terminal_status": "succeeded",
                "total_count": 2,
                "prediction_count": 16,
            },
        ],
    }

    assert all(
        score(
            {"id": label["id"]},
            label["gold_output"],
            evidence,
        ).values()
    )


def test_live_mcp_receipt_separates_reachability_from_row_reconciliation() -> None:
    lock = _lock()
    evaluations = lock["objects"]["evaluations"]
    call_ids = [item["call_id"] for item in evaluations]

    def observation(import_id: str, *, works: bool) -> dict:
        if not works:
            failed = {
                "ok": False,
                "protocol_error": True,
                "value": {"message": "relogin required"},
            }
            return {
                "id": import_id,
                "version_identity": "git:" + "1" * 40,
                "runtime_digest": "sha256:" + "2" * 64,
                "tool_manifest_digest": "sha256:" + "3" * 64,
                "server": {"name": "mcp", "version": "old"},
                "initialized_tools": ["query_wandb_tool"],
                "locked_tools": ["query_wandb_tool"],
                "release_capabilities": {
                    "structured_query": False,
                    "exact_count_mode": False,
                    "projected_summary_keys": False,
                    "bounded_history_range": False,
                    "raw_graphql_registered_by_default": False,
                },
                "calls": {
                    "count_weave_traces_tool": failed,
                    "probe_project_tool": failed,
                    "summarize_evaluation_tool": failed,
                },
                "evaluation_child_ops": {call_id: failed for call_id in call_ids},
            }
        return {
            "id": import_id,
            "version_identity": "git:" + "4" * 40,
            "runtime_digest": "sha256:" + "5" * 64,
            "tool_manifest_digest": "sha256:" + "6" * 64,
            "server": {"name": "mcp", "version": "new"},
            "initialized_tools": [
                "create_wandb_report_tool",
                "get_run_history_tool",
                "log_analysis_to_wandb",
                "query_wandb_tool",
            ],
            "locked_tools": [
                "create_wandb_report_tool",
                "get_run_history_tool",
                "log_analysis_to_wandb",
                "query_wandb_tool",
            ],
            "release_capabilities": {
                "structured_query": True,
                "exact_count_mode": True,
                "projected_summary_keys": True,
                "bounded_history_range": True,
                "raw_graphql_registered_by_default": False,
            },
            "calls": {
                "count_weave_traces_tool": {
                    "ok": True,
                    "value": {"root_traces_count": 26, "total_count": 26},
                },
                    "probe_project_tool": {
                        "ok": True,
                        "value": {"run_count": 6, "state_counts": {"finished": 6}},
                    },
                    "count_evaluation_roots_tool": {
                        "ok": True,
                        "value": {
                            "root_traces_count": 2,
                            "total_count": 2,
                        },
                    },
                "summarize_evaluation_tool": {
                    "ok": True,
                    "value": {
                        "evaluations": [
                            {"eval_id": call_id, "total_predictions": 9}
                            for call_id in call_ids
                        ]
                    },
                },
            },
            "evaluation_child_ops": {
                call_id: {
                    "ok": True,
                    "value": {
                        "metadata": {
                            "op_distribution": {
                                "Evaluation.predict_and_score": 8,
                                "Evaluation.summarize": 1,
                            }
                        }
                    },
                }
                for call_id in call_ids
            },
            "profile_probes": {
                "read_only": {
                    "overrides": {"WANDB_MCP_READ_ONLY": "true"},
                    "initialized_tools": [
                        "get_run_history_tool",
                        "query_wandb_tool",
                    ],
                    "tool_manifest_digest": "a" * 64,
                    "mutation_probe": None,
                },
                "raw_graphql": {
                    "overrides": {
                        "WANDB_MCP_ENABLE_RAW_GRAPHQL": "true"
                    },
                    "initialized_tools": [
                        "create_wandb_report_tool",
                        "get_run_history_tool",
                        "log_analysis_to_wandb",
                        "query_wandb_graphql_tool",
                        "query_wandb_tool",
                    ],
                    "tool_manifest_digest": "b" * 64,
                    "mutation_probe": {
                        "ok": True,
                        "value": {
                            "errors": [
                                {
                                    "error": "read_only_violation",
                                    "operation_types": ["mutation"],
                                }
                            ]
                        },
                    },
                },
            },
        }

    receipt = _mcp_release_qualification_receipt(
        lock,
        [
            observation("wandb-mcp-main", works=False),
            observation("wandb-mcp-0-4-staging", works=True),
        ],
    )

    assert receipt["findings"] == {
        "baseline_reads_hosted_evidence": False,
        "candidate_reads_hosted_evidence": True,
        "baseline_manifest_matches_lock": True,
        "candidate_manifest_matches_lock": True,
        "candidate_project_probe_matches_lock": True,
        "baseline_evaluation_rows_reconciled": False,
        "candidate_evaluation_rows_reconciled": False,
    }
    candidate = receipt["candidates"][1]
    assert all(
        item["trace_children_reconciled"] is True
        and item["prediction_rows_reconciled"] is False
        for item in candidate["evaluation_reconciliation"]
    )
    assert receipt["whole_release_claim_eligible"] is False
    conformance = receipt["infrastructure_conformance"]
    assert conformance["complete"] is False
    assert conformance["failed"] == []
    assert "read-only-tool-manifest" not in conformance["unavailable"]
    assert "raw-graphql-opt-in-manifest" not in conformance["unavailable"]
    assert "graphql-mutation-rejection" not in conformance["unavailable"]
    assert next(
        item
        for item in conformance["gates"]
        if item["id"] == "default-tool-manifest"
    )["status"] == "passed"
    assert {
        item["id"]
        for item in conformance["gates"]
        if item["status"] == "passed"
    } >= {
        "default-tool-manifest",
        "read-only-tool-manifest",
        "raw-graphql-opt-in-manifest",
        "graphql-mutation-rejection",
    }
    assert {
        item["status"] for item in receipt["release_note_classification"]
    } >= {"observed_branch_delta", "infrastructure_only_not_live_induced"}
    assert len(receipt["receipt_digest"]) == 64
