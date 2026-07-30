from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from fugue.bench import mcp_release_qualification
from fugue.bench.candidates import stable_digest
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
        if entry.path.startswith("examples/comparisons/wandb-mcp-maintenance/")
    )

    assert {(entry.id, entry.path) for entry in mcp_entries} == {
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

    assert release_notes["commit"] == ("3dd4447ef0054d4707aafc515e3f2ddfb11b17bd")
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


def _verified_source_inventory(
    *,
    extra_objects: tuple[str, ...] = (),
    drift: tuple[str, ...] = (),
    actions: set[str] | None = None,
    mutation: str = "",
) -> dict:
    selected = (
        set(mcp_release_qualification._SOURCE_PREPARATION_ACTIONS)
        if actions is None
        else set(actions)
    )
    seed = qualification_seed(source_project=QUALIFICATION_SOURCE_PROJECT)
    run_by_id = {str(item["id"]): item for item in seed["runs"]}
    runs = []
    for run_id, item in sorted(run_by_id.items()):
        if f"wandb-run:{run_id}" not in selected:
            continue
        receipt = {
            "id": run_id,
            "name": item["attempt_label"],
            "url": f"https://wandb.ai/{QUALIFICATION_SOURCE_PROJECT}/runs/{run_id}",
            "ref": f"wandb-run:///{QUALIFICATION_SOURCE_PROJECT}/{run_id}",
            "seed_digest": qualification_seed_digest(
                source_project=QUALIFICATION_SOURCE_PROJECT
            ),
            "state": "finished",
            "config_digest": stable_digest({"run": run_id, "kind": "config"}),
            "history_digest": stable_digest({"run": run_id, "kind": "history"}),
            "summary_digest": stable_digest({"run": run_id, "kind": "summary"}),
            "artifact": {
                "name": f"qualification-evidence-{run_id}:v0",
                "version": "v0",
                "digest": f"artifact-{run_id}",
                "qualified_name": (
                    f"{QUALIFICATION_SOURCE_PROJECT}/qualification-evidence-{run_id}:v0"
                ),
                "content_digest": stable_digest({"run": run_id, "kind": "artifact"}),
            },
        }
        receipt["content_digest"] = stable_digest({**receipt, "mutation": mutation})
        runs.append(receipt)
    dataset = None
    if "weave-dataset:mcp-release-maintenance-cases" in selected:
        dataset = {
            "name": "mcp-release-maintenance-cases",
            "ref": (
                f"weave:///{QUALIFICATION_SOURCE_PROJECT}/object/"
                "mcp-release-maintenance-cases:dataset-digest"
            ),
            "rows": 8,
            "content_digest": stable_digest({"dataset": "cases", "mutation": mutation}),
        }
    conversations = []
    for run_id in sorted(run_by_id):
        for index in range(1, 5):
            action = f"weave-conversation:{run_id}:{index}"
            if action not in selected:
                continue
            call_id = f"{run_id}-conversation-{index}"
            conversations.append(
                {
                    "run_id": run_id,
                    "conversation_index": index,
                    "call_id": call_id,
                    "ref": (f"weave:///{QUALIFICATION_SOURCE_PROJECT}/call/{call_id}"),
                    "tool_call_ids": [
                        f"{call_id}-tool-1",
                        f"{call_id}-tool-2",
                    ],
                    "tool_span_count": 2,
                    "content_digest": stable_digest(
                        {
                            "conversation": call_id,
                            "mutation": mutation,
                        }
                    ),
                }
            )
    evaluation_objects = {}
    evaluations = []
    for revision in ("maintainer-r17", "maintainer-r18"):
        object_action = f"weave-evaluation-object:{revision}"
        evaluation_ref = (
            f"weave:///{QUALIFICATION_SOURCE_PROJECT}/object/"
            f"mcp-release-{revision}:evaluation-digest"
        )
        if object_action in selected:
            evaluation_objects[revision] = {
                "revision": revision,
                "ref": evaluation_ref,
                "content_digest": stable_digest(
                    {"evaluation": revision, "mutation": mutation}
                ),
            }
        if f"weave-evaluation-run:{revision}" in selected:
            call_id = f"evaluation-{revision}"
            evaluations.append(
                {
                    "revision": revision,
                    "ref": evaluation_ref,
                    "call_id": call_id,
                    "call_ref": (
                        f"weave:///{QUALIFICATION_SOURCE_PROJECT}/call/{call_id}"
                    ),
                    "summary_digest": stable_digest({"summary": revision}),
                    "prediction_rows": 8,
                    "direct_children": 9,
                    "summarize_children": 1,
                    "content_digest": stable_digest(
                        {
                            "evaluation_call": revision,
                            "mutation": mutation,
                        }
                    ),
                }
            )
    return mcp_release_qualification._source_inventory(
        source_project=QUALIFICATION_SOURCE_PROJECT,
        runs=runs,
        dataset=dataset,
        conversations=conversations,
        evaluation_objects=evaluation_objects,
        evaluations=evaluations,
        extra_objects=extra_objects,
        drift=drift,
    )


def _preparation_env(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text("WANDB_API_KEY=unit-test-key\n", encoding="utf-8")
    return path


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
                "op_name": (f"weave:///{project}/op/Evaluation.evaluate:root-digest"),
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
                    f"weave:///{project}/op/Evaluation.summarize:summary-digest"
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
    assert receipt["expectations"]["repaired_candidate_prediction_children"] == 16
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
    assert any(item.endswith(":child_project_mismatch") for item in receipt["blockers"])


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
            source_project=QUALIFICATION_SOURCE_PROJECT,
            result_project=QUALIFICATION_RESULT_PROJECT,
            output=output,
            env_file=tmp_path / "missing.env",
        )


def test_v3_preparation_rejects_legacy_or_same_project_writers(
    tmp_path: Path,
) -> None:
    prepare = mcp_release_qualification.prepare_hosted_project

    with pytest.raises(ValueError, match="legacy single-project"):
        prepare(
            project=QUALIFICATION_RESULT_PROJECT,
            output=tmp_path / "legacy.json",
            env_file=tmp_path / "missing.env",
        )
    with pytest.raises(ValueError, match="distinct result project"):
        prepare(
            source_project=QUALIFICATION_SOURCE_PROJECT,
            result_project=QUALIFICATION_SOURCE_PROJECT,
            output=tmp_path / "same.json",
            env_file=tmp_path / "missing.env",
        )
    with pytest.raises(ValueError, match="dedicated immutable source"):
        prepare(
            source_project=QUALIFICATION_RESULT_PROJECT,
            result_project=QUALIFICATION_RESULT_PROJECT,
            output=tmp_path / "result.json",
            env_file=tmp_path / "missing.env",
        )


def test_fresh_then_missing_lock_rerun_reuses_remote_source_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"inventory": _verified_source_inventory(actions=set())}
    writes = []
    inventory_calls = []

    def inventory(entity, project, *, source_project):
        inventory_calls.append((entity, project, source_project))
        return state["inventory"]

    def materialize(
        entity,
        project,
        *,
        source_project,
        inventory,
        run_action,
    ):
        del inventory, run_action
        writes.append((entity, project, source_project))
        state["inventory"] = _verified_source_inventory()

    monkeypatch.setattr(
        mcp_release_qualification,
        "_inventory_hosted_source",
        inventory,
    )
    monkeypatch.setattr(
        mcp_release_qualification,
        "_materialize_hosted_source",
        materialize,
    )
    output = tmp_path / "evidence.lock.json"
    kwargs = {
        "source_project": QUALIFICATION_SOURCE_PROJECT,
        "result_project": QUALIFICATION_RESULT_PROJECT,
        "output": output,
        "env_file": _preparation_env(tmp_path),
    }

    first = mcp_release_qualification.prepare_hosted_project(**kwargs)
    output.unlink()
    second = mcp_release_qualification.prepare_hosted_project(**kwargs)

    assert first == second
    assert writes == [
        (
            "wandb",
            "fugue-mcp-release-source-v1",
            QUALIFICATION_SOURCE_PROJECT,
        )
    ]
    assert inventory_calls
    assert all(
        item
        == (
            "wandb",
            "fugue-mcp-release-source-v1",
            QUALIFICATION_SOURCE_PROJECT,
        )
        for item in inventory_calls
    )
    assert first["source_project"] == QUALIFICATION_SOURCE_PROJECT
    assert first["result_project"] == QUALIFICATION_RESULT_PROJECT
    assert first["counts"] == {
        "runs": 6,
        "source_conversations": 24,
        "tool_spans": 48,
        "dataset_rows": 8,
        "aligned_evaluation_pairs": 8,
        "evaluation_prediction_rows": 16,
    }
    assert "unit-test-key" not in output.read_text(encoding="utf-8")
    assert "unit-test-key" not in (
        tmp_path / "evidence.lock.json.progress.json"
    ).read_text(encoding="utf-8")


def test_existing_lock_reuse_rejects_remote_content_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"inventory": _verified_source_inventory()}
    monkeypatch.setattr(
        mcp_release_qualification,
        "_inventory_hosted_source",
        lambda *args, **kwargs: state["inventory"],
    )
    output = tmp_path / "evidence.lock.json"
    kwargs = {
        "source_project": QUALIFICATION_SOURCE_PROJECT,
        "result_project": QUALIFICATION_RESULT_PROJECT,
        "output": output,
        "env_file": _preparation_env(tmp_path),
    }
    mcp_release_qualification.prepare_hosted_project(**kwargs)
    state["inventory"] = _verified_source_inventory(mutation="drifted")

    with pytest.raises(RuntimeError, match="disagrees with hosted source"):
        mcp_release_qualification.prepare_hosted_project(**kwargs)


def test_source_preparation_recovers_visible_in_flight_write_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_action = "wandb-run:maint-r18-01"
    state = {"inventory": _verified_source_inventory(actions=set())}
    attempts = []

    monkeypatch.setattr(
        mcp_release_qualification,
        "_inventory_hosted_source",
        lambda *args, **kwargs: state["inventory"],
    )

    def interrupted_materialize(
        entity,
        project,
        *,
        source_project,
        inventory,
        run_action,
    ):
        del entity, project, source_project, inventory

        def write_then_interrupt():
            attempts.append(first_action)
            state["inventory"] = _verified_source_inventory(actions={first_action})
            raise RuntimeError("simulated interruption")

        run_action(first_action, write_then_interrupt)

    monkeypatch.setattr(
        mcp_release_qualification,
        "_materialize_hosted_source",
        interrupted_materialize,
    )
    output = tmp_path / "evidence.lock.json"
    kwargs = {
        "source_project": QUALIFICATION_SOURCE_PROJECT,
        "result_project": QUALIFICATION_RESULT_PROJECT,
        "output": output,
        "env_file": _preparation_env(tmp_path),
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        mcp_release_qualification.prepare_hosted_project(**kwargs)

    def finish_materialize(
        entity,
        project,
        *,
        source_project,
        inventory,
        run_action,
    ):
        del entity, project, source_project, inventory, run_action
        state["inventory"] = _verified_source_inventory()

    monkeypatch.setattr(
        mcp_release_qualification,
        "_materialize_hosted_source",
        finish_materialize,
    )
    result = mcp_release_qualification.prepare_hosted_project(**kwargs)

    assert result["counts"]["runs"] == 6
    assert attempts == [first_action]


def test_source_preparation_refuses_unresolved_in_flight_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = _verified_source_inventory(actions=set())
    monkeypatch.setattr(
        mcp_release_qualification,
        "_inventory_hosted_source",
        lambda *args, **kwargs: empty,
    )

    def interrupted_materialize(
        entity,
        project,
        *,
        source_project,
        inventory,
        run_action,
    ):
        del entity, project, source_project, inventory

        def unresolved():
            raise RuntimeError("transport outcome unknown")

        run_action("wandb-run:maint-r18-01", unresolved)

    monkeypatch.setattr(
        mcp_release_qualification,
        "_materialize_hosted_source",
        interrupted_materialize,
    )
    kwargs = {
        "source_project": QUALIFICATION_SOURCE_PROJECT,
        "result_project": QUALIFICATION_RESULT_PROJECT,
        "output": tmp_path / "evidence.lock.json",
        "env_file": _preparation_env(tmp_path),
    }
    with pytest.raises(RuntimeError, match="transport outcome unknown"):
        mcp_release_qualification.prepare_hosted_project(**kwargs)
    with pytest.raises(RuntimeError, match="outcome is unresolved"):
        mcp_release_qualification.prepare_hosted_project(**kwargs)


def test_source_preflight_rejects_extra_objects_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _verified_source_inventory(
        actions=set(),
        extra_objects=("weave-evaluation-root:unexpected",),
    )
    writes = []
    monkeypatch.setattr(
        mcp_release_qualification,
        "_inventory_hosted_source",
        lambda *args, **kwargs: inventory,
    )
    monkeypatch.setattr(
        mcp_release_qualification,
        "_materialize_hosted_source",
        lambda *args, **kwargs: writes.append("write"),
    )

    with pytest.raises(RuntimeError, match="contains extra objects"):
        mcp_release_qualification.prepare_hosted_project(
            source_project=QUALIFICATION_SOURCE_PROJECT,
            result_project=QUALIFICATION_RESULT_PROJECT,
            output=tmp_path / "evidence.lock.json",
            env_file=_preparation_env(tmp_path),
        )
    assert writes == []


def test_weave_inventory_requires_exact_conversation_tool_children() -> None:
    seed_digest = qualification_seed_digest(source_project=QUALIFICATION_SOURCE_PROJECT)
    calls = []
    for item in qualification_seed(source_project=QUALIFICATION_SOURCE_PROJECT)["runs"]:
        for index in range(1, 5):
            root_id = f"{item['id']}-conversation-{index}"
            trace_id = f"trace-{root_id}"
            calls.append(
                {
                    "id": root_id,
                    "project_id": QUALIFICATION_SOURCE_PROJECT,
                    "trace_id": trace_id,
                    "parent_id": "",
                    "op_name": "fugue.qualification.maintenance_agent",
                    "inputs": (
                        mcp_release_qualification._expected_conversation_inputs(
                            item,
                            index,
                            seed_digest,
                        )
                    ),
                    "output": (
                        mcp_release_qualification._expected_conversation_output(
                            item,
                            index,
                        )
                    ),
                    "attributes": {"fugue.seed_digest": seed_digest},
                    "ended_at": "2026-07-30T00:00:00Z",
                    "exception": None,
                }
            )
            for tool_index, tool_inputs in enumerate(
                mcp_release_qualification._expected_tool_inputs(item, index),
                start=1,
            ):
                calls.append(
                    {
                        "id": f"{root_id}-tool-{tool_index}",
                        "project_id": QUALIFICATION_SOURCE_PROJECT,
                        "trace_id": trace_id,
                        "parent_id": root_id,
                        "op_name": "fugue.qualification.wandb_mcp_tool",
                        "inputs": tool_inputs,
                        "output": {
                            "tool_name": tool_inputs["tool_name"],
                            "run_id": tool_inputs["run_id"],
                            "result": tool_inputs["public_result"],
                        },
                        "attributes": {},
                        "ended_at": "2026-07-30T00:00:00Z",
                        "exception": None,
                    }
                )

    receipts, extras, drift = mcp_release_qualification._inventory_conversations(
        calls,
        project=QUALIFICATION_SOURCE_PROJECT,
        seed_digest=seed_digest,
    )
    assert len(receipts) == 24
    assert sum(item["tool_span_count"] for item in receipts) == 48
    assert extras == drift == []

    calls[-1]["output"] = {"drifted": True}
    _, _, drift = mcp_release_qualification._inventory_conversations(
        calls,
        project=QUALIFICATION_SOURCE_PROJECT,
        seed_digest=seed_digest,
    )
    assert any(item.endswith(":tools") for item in drift)


def test_weave_inventory_rejects_duplicate_evaluation_roots() -> None:
    seed_digest = qualification_seed_digest(source_project=QUALIFICATION_SOURCE_PROJECT)
    objects = {
        revision: {
            "revision": revision,
            "ref": (
                f"weave:///{QUALIFICATION_SOURCE_PROJECT}/object/"
                f"mcp-release-{revision}:digest"
            ),
            "content_digest": stable_digest({"revision": revision}),
        }
        for revision in ("maintainer-r17", "maintainer-r18")
    }
    calls = []
    for revision in objects:
        root_id = f"evaluation-{revision}"
        trace_id = f"trace-{root_id}"
        calls.append(
            {
                "id": root_id,
                "project_id": QUALIFICATION_SOURCE_PROJECT,
                "trace_id": trace_id,
                "parent_id": "",
                "op_name": "Evaluation.evaluate",
                "inputs": {},
                "output": {"revision": revision},
                "attributes": {
                    "fugue.seed_digest": seed_digest,
                    "fugue.evaluation_revision": revision,
                },
                "ended_at": "2026-07-30T00:00:00Z",
                "exception": None,
            }
        )
        for index in range(8):
            calls.append(
                {
                    "id": f"{root_id}-prediction-{index}",
                    "project_id": QUALIFICATION_SOURCE_PROJECT,
                    "trace_id": trace_id,
                    "parent_id": root_id,
                    "op_name": "Evaluation.predict_and_score",
                    "inputs": {"case": index},
                    "output": {"case": index},
                    "attributes": {},
                    "ended_at": "2026-07-30T00:00:00Z",
                    "exception": None,
                }
            )
        calls.append(
            {
                "id": f"{root_id}-summary",
                "project_id": QUALIFICATION_SOURCE_PROJECT,
                "trace_id": trace_id,
                "parent_id": root_id,
                "op_name": "Evaluation.summarize",
                "inputs": {},
                "output": {},
                "attributes": {},
                "ended_at": "2026-07-30T00:00:00Z",
                "exception": None,
            }
        )

    receipts, extras, drift = mcp_release_qualification._inventory_evaluation_calls(
        calls,
        project=QUALIFICATION_SOURCE_PROJECT,
        seed_digest=seed_digest,
        evaluation_objects=objects,
    )
    assert len(receipts) == 2
    assert extras == drift == []

    calls.append({**calls[0], "id": "duplicate-evaluation-root"})
    _, extras, _ = mcp_release_qualification._inventory_evaluation_calls(
        calls,
        project=QUALIFICATION_SOURCE_PROJECT,
        seed_digest=seed_digest,
        evaluation_objects=objects,
    )
    assert "weave-evaluation-root:maintainer-r17:duplicate" in extras


def test_wandb_inventory_hashes_full_history_and_exact_artifact() -> None:
    item = qualification_seed(source_project=QUALIFICATION_SOURCE_PROJECT)["runs"][0]
    seed_digest = qualification_seed_digest(source_project=QUALIFICATION_SOURCE_PROJECT)

    class Entry:
        def download(self, *, root, replace):
            assert replace is True
            path = Path(root) / "attempt-evidence.json"
            path.write_text(
                json.dumps(
                    mcp_release_qualification._expected_run_artifact_payload(
                        item,
                        seed_digest,
                    )
                ),
                encoding="utf-8",
            )
            return path.as_posix()

    class Artifact:
        type = "fugue-qualification-evidence"
        metadata = {"seed_digest": seed_digest}
        name = f"qualification-evidence-{item['id']}:v0"
        version = "v0"
        digest = "artifact-logical-digest"
        qualified_name = (
            f"{QUALIFICATION_SOURCE_PROJECT}/qualification-evidence-{item['id']}:v0"
        )

        def get_path(self, name):
            assert name == "attempt-evidence.json"
            return Entry()

    class Run:
        id = item["id"]
        name = item["attempt_label"]
        state = "finished"
        url = f"https://wandb.ai/{QUALIFICATION_SOURCE_PROJECT}/runs/{item['id']}"
        config = mcp_release_qualification._expected_run_config(
            item,
            seed_digest,
        )
        summary = {
            "fugue_seed_digest": seed_digest,
            "evidence_lock_status": "prepared",
            "artifact_digest": Artifact.digest,
        }
        rows = mcp_release_qualification._expected_run_history(item)

        def scan_history(self, *, keys, page_size):
            assert keys
            assert page_size == 1000
            return iter(self.rows)

        def logged_artifacts(self, *, per_page):
            assert per_page == 100
            return [Artifact()]

    receipt = mcp_release_qualification._inspect_wandb_run(
        Run(),
        item,
        source_project=QUALIFICATION_SOURCE_PROJECT,
        seed_digest=seed_digest,
    )
    assert receipt["history_digest"] == stable_digest(Run.rows)
    assert receipt["artifact"]["content_digest"] == stable_digest(
        mcp_release_qualification._expected_run_artifact_payload(
            item,
            seed_digest,
        )
    )

    Run.rows = [*Run.rows, {"step": 4, "latency_ms": 1}]
    with pytest.raises(RuntimeError, match="history drifted"):
        mcp_release_qualification._inspect_wandb_run(
            Run(),
            item,
            source_project=QUALIFICATION_SOURCE_PROJECT,
            seed_digest=seed_digest,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = runpy.run_path(
        (EXAMPLE / "natural_maintainer_scorer.py").as_posix()
    )["score"]

    def fake_inline_runner(*, source, evidence, reference, profile, limits):
        assert "if __name__ == \"__main__\":" in source
        assert profile.id == "python312-sandbox-v1"
        details = scorer(
            reference["task"],
            reference["output"],
            {**evidence, "expected": reference["expected"]},
        )
        return {
            "score": 1.0 if all(details.values()) else 0.0,
            "reason": "offline qualification fixture",
            "details": details,
        }

    monkeypatch.setattr(
        "fugue.bench.task_authoring.run_inline_scorer",
        fake_inline_runner,
    )
    root = Path.cwd()
    spec = load_comparison(EXAMPLE / filename, repo_root=root)
    readiness = check_comparison(spec, repo_root=root)

    assert spec.schema_version == 3
    assert spec.baseline.integrations == ({"id": "wandb-mcp-main"},)
    assert spec.candidate.integrations == ({"id": "wandb-mcp-0-4-staging"},)
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
    assert gates["infrastructure.gate.human-maintainer-actionability-review"] is True
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
    assert 'evaluation_root_count": 2' not in public
    assert 'largest_latency_ms": 4200' not in public


def _natural_maintainer_score():
    return runpy.run_path((EXAMPLE / "natural_maintainer_scorer.py").as_posix())[
        "score"
    ]


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
                    "overrides": {"WANDB_MCP_ENABLE_RAW_GRAPHQL": "true"},
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
    assert (
        next(
            item
            for item in conformance["gates"]
            if item["id"] == "default-tool-manifest"
        )["status"]
        == "passed"
    )
    assert {
        item["id"] for item in conformance["gates"] if item["status"] == "passed"
    } >= {
        "default-tool-manifest",
        "read-only-tool-manifest",
        "raw-graphql-opt-in-manifest",
        "graphql-mutation-rejection",
    }
    assert {item["status"] for item in receipt["release_note_classification"]} >= {
        "observed_branch_delta",
        "infrastructure_only_not_live_induced",
    }
    assert len(receipt["receipt_digest"]) == 64
