from __future__ import annotations

import json
import runpy
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

import fugue.bench.comparison as comparison_module
from fugue.bench import mcp_release_qualification
from fugue.bench.analysis_contracts import (
    EvidenceDriftCheckV1,
    EvidenceTopologyV1,
    LockDescriptorV1,
)
from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    ComparisonResultV3,
    analyze_comparison_rows,
    check_comparison,
    comparison_from_dict,
    load_comparison,
)
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
    release_note_coverage_v3,
    validate_evidence_lock,
    validate_release_notes_lock,
)
from fugue.model_plane import EvidenceDestinationV1
from fugue.research.comparisons import ComparisonRegistry

EXAMPLE = Path("examples/comparisons/wandb-mcp-maintenance")


@pytest.fixture(autouse=True)
def _private_qualification_projects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mcp_release_qualification,
        "verify_private_project_topology",
        lambda **_kwargs: {
            QUALIFICATION_SOURCE_PROJECT: "PRIVATE",
            QUALIFICATION_RESULT_PROJECT: "PRIVATE",
        },
    )


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
    }


def test_release_notes_lock_binds_exact_rc_source_and_all_classifications() -> None:
    release_notes = validate_release_notes_lock(
        json.loads((EXAMPLE / "release-notes.lock.json").read_text())
    )
    receipt = _mcp_release_qualification_receipt(_lock(), [])

    assert release_notes["commit"] == ("29cc1b5b5cf4061afa1faa712021fa1b68ad0bf7")
    assert release_notes["sha256"] == (
        "257110b2542caec6bf30c24cfb59c1a3ee1074b69ad68497b9978ba130243e12"
    )
    assert {
        item["release_note"] for item in receipt["release_note_classification"]
    } == set(release_notes["behaviors"])


def test_release_decision_candidate_sha_binds_release_notes_and_mcp_lock(
    tmp_path: Path,
) -> None:
    spec = load_comparison(
        EXAMPLE / "natural-maintainer-canary-local-v3.yaml",
        repo_root=Path.cwd(),
    )
    assert spec.decision_policy is not None
    candidate_sha = spec.decision_policy.candidate_sha
    lock_dir = tmp_path / ".fugue/imports/mcp/locks"
    lock_dir.mkdir(parents=True)
    (lock_dir / "wandb-mcp-0-4-staging.json").write_text(
        json.dumps({"version_identity": f"git:{candidate_sha}"}),
        encoding="utf-8",
    )

    comparison_module._validate_release_candidate_binding(
        spec,
        release_notes={"commit": candidate_sha},
        repo_root=tmp_path,
    )

    drifted = replace(
        spec,
        decision_policy=replace(
            spec.decision_policy,
            candidate_sha="0" * 40,
        ),
    )
    with pytest.raises(ValueError, match="release-notes lock"):
        comparison_module._validate_release_candidate_binding(
            drifted,
            release_notes={"commit": candidate_sha},
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="candidate MCP lock"):
        comparison_module._validate_release_candidate_binding(
            drifted,
            release_notes={"commit": "0" * 40},
            repo_root=tmp_path,
        )


def test_release_note_infrastructure_coverage_names_exact_unique_gates() -> None:
    receipt = _mcp_release_qualification_receipt(_lock(), [])
    coverage = release_note_coverage_v3(
        receipt,
        task_ids=(
            "maintainer-source-inventory",
            "maintainer-evaluation-reconciliation",
            "maintainer-project-health",
            "maintainer-history-hotspot",
        ),
        dimension_ids=(
            "natural-maintainer.answer_correct",
            "natural-maintainer.actual_query_scope",
            "natural-maintainer.reported_project_identity",
            "natural-maintainer.bounded_evidence",
            "natural-maintainer.evidence_honesty",
            "natural-maintainer.release_mechanism_used",
        ),
    )

    infrastructure_rows = [
        item for item in coverage if item["status"] == "infrastructure_only"
    ]
    assert infrastructure_rows
    assert all(item["infrastructure_gates"] for item in infrastructure_rows)
    assert all(
        len(item["infrastructure_gates"]) == len(set(item["infrastructure_gates"]))
        for item in infrastructure_rows
    )


def test_weave_missing_project_is_distinct_from_forbidden_access() -> None:
    request = httpx.Request("POST", "https://trace.wandb.ai/project/stats")
    missing = httpx.HTTPStatusError(
        "403 Forbidden: Project not found",
        request=request,
        response=httpx.Response(403, request=request),
    )
    forbidden = httpx.HTTPStatusError(
        "403 Forbidden: access denied",
        request=request,
        response=httpx.Response(403, request=request),
    )

    assert mcp_release_qualification._weave_missing_error(missing) is True
    assert mcp_release_qualification._weave_missing_error(forbidden) is False


def test_weave_inventory_preflights_missing_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://trace.wandb.ai/project/stats")

    class Server:
        def project_stats(self, request_value):
            assert request_value.project_id == QUALIFICATION_SOURCE_PROJECT
            raise httpx.HTTPStatusError(
                "403 Forbidden: Project not found",
                request=request,
                response=httpx.Response(403, request=request),
            )

    class Client:
        server = Server()

        def get_calls(self, **kwargs):
            raise AssertionError("missing project must not query Calls")

    @contextmanager
    def source_client(project: str, *, write: bool):
        assert project == QUALIFICATION_SOURCE_PROJECT
        assert write is False
        yield Client()

    monkeypatch.setattr(
        mcp_release_qualification,
        "_source_weave_client",
        source_client,
    )

    assert mcp_release_qualification._inventory_weave_evidence(
        QUALIFICATION_SOURCE_PROJECT
    ) == (None, [], {}, [], [], [], [])


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
    object_version_count: int | None = None,
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
                    "call_ids": [
                        call_id,
                        *(f"{call_id}-child-{index}" for index in range(1, 26)),
                    ],
                    "content_digest": stable_digest(
                        {
                            "evaluation_call": revision,
                            "mutation": mutation,
                        }
                    ),
                }
            )
    object_ids = (
        "mcp-release-maintenance-cases",
        "mcp-release-maintainer-r17",
        "mcp-release-maintainer-r18",
        "EvaluationResults",
        "EvaluationResults",
    )
    op_ids = (
        "fugue.qualification.maintenance_agent",
        "fugue.qualification.wandb_mcp_tool",
        "Evaluation.evaluate",
        "Evaluation.summarize",
        "Evaluation.predict_and_score",
        "fugue.qualification.evidence_alignment_scorer",
        "fugue.qualification.baseline_evidence_model",
        "fugue.qualification.candidate_evidence_model",
    )
    object_versions = []
    for index, (kind, object_id) in enumerate(
        (
            *(("object", item) for item in object_ids),
            *(("op", item) for item in op_ids),
        ),
        start=1,
    ):
        metadata = {
            "kind": kind,
            "object_id": object_id,
            "digest": f"object-digest-{index}",
            "base_object_class": "",
            "aliases": [],
        }
        object_versions.append(
            {
                **metadata,
                "content_digest": stable_digest(metadata),
            }
        )
    if object_version_count is not None:
        object_versions = object_versions[:object_version_count]
    return mcp_release_qualification._source_inventory(
        source_project=QUALIFICATION_SOURCE_PROJECT,
        runs=runs,
        dataset=dataset,
        conversations=conversations,
        evaluation_objects=evaluation_objects,
        evaluations=evaluations,
        object_versions=object_versions,
        extra_objects=extra_objects,
        drift=drift,
    )


@pytest.mark.parametrize(
    ("object_rows", "op_rows"),
    (
        ((), ()),
        (
            (
                SimpleNamespace(
                    object_id="mcp-release-maintenance-cases",
                    digest="dataset-digest",
                    base_object_class="Dataset",
                    aliases=("qualification-v1",),
                ),
            ),
            (
                SimpleNamespace(
                    object_id="fugue.qualification.maintenance_agent",
                    digest="agent-op-digest",
                    base_object_class="Op",
                    aliases=(),
                ),
            ),
        ),
    ),
)
def test_missing_weave_object_versions_are_preparation_incomplete_not_drift(
    object_rows: tuple[SimpleNamespace, ...],
    op_rows: tuple[SimpleNamespace, ...],
) -> None:
    class Server:
        def objs_query(self, request):
            return SimpleNamespace(
                objs=op_rows if request.filter.is_op else object_rows
            )

    rows, extras, drift = mcp_release_qualification._inventory_weave_object_versions(
        SimpleNamespace(server=Server()),
        QUALIFICATION_SOURCE_PROJECT,
    )

    assert len(rows) == len(object_rows) + len(op_rows)
    assert extras == []
    assert drift == []


def test_source_inventory_is_complete_only_with_all_exact_object_versions() -> None:
    complete = mcp_release_qualification._validate_source_inventory(
        _verified_source_inventory(),
        source_project=QUALIFICATION_SOURCE_PROJECT,
    )
    missing_one = mcp_release_qualification._validate_source_inventory(
        _verified_source_inventory(object_version_count=12),
        source_project=QUALIFICATION_SOURCE_PROJECT,
    )
    empty = mcp_release_qualification._validate_source_inventory(
        _verified_source_inventory(actions=set(), object_version_count=0),
        source_project=QUALIFICATION_SOURCE_PROJECT,
    )

    assert complete["complete"] is True
    assert len(complete["weave_object_versions"]) == 13
    assert missing_one["complete"] is False
    assert missing_one["drift"] == []
    assert empty["complete"] is False
    assert empty["drift"] == []


def _rich_source_lock() -> dict:
    inventory = _verified_source_inventory()
    return _evidence_lock(
        QUALIFICATION_SOURCE_PROJECT,
        inventory["runs"],
        mcp_release_qualification._inventory_weave_receipts(inventory),
        result_project=QUALIFICATION_RESULT_PROJECT,
        preparation_id="test-preparation",
        source_inventory_digest=inventory["inventory_digest"],
    )


def _resign_source_lock(lock: dict) -> None:
    lock["source_snapshot_digest"] = stable_digest(lock["objects"])
    lock["evidence_lock_digest"] = ""
    lock["evidence_lock_digest"] = stable_digest(lock)


def test_evidence_lock_rejects_duplicate_weave_object_version_identity() -> None:
    lock = _rich_source_lock()
    versions = lock["objects"]["weave_object_versions"]
    versions[-1] = dict(versions[0])
    _resign_source_lock(lock)

    with pytest.raises(
        ValueError,
        match="must bind every source Weave object version",
    ):
        validate_evidence_lock(
            lock,
            expected_project=None,
            expected_source_project=QUALIFICATION_SOURCE_PROJECT,
            expected_result_project=QUALIFICATION_RESULT_PROJECT,
        )


def test_evidence_lock_rejects_wrong_weave_object_version_name() -> None:
    lock = _rich_source_lock()
    lock["objects"]["weave_object_versions"][0]["object_id"] = (
        "unexpected-source-object"
    )
    _resign_source_lock(lock)

    with pytest.raises(
        ValueError,
        match="must bind every source Weave object version",
    ):
        validate_evidence_lock(
            lock,
            expected_project=None,
            expected_source_project=QUALIFICATION_SOURCE_PROJECT,
            expected_result_project=QUALIFICATION_RESULT_PROJECT,
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
        project_access={
            QUALIFICATION_SOURCE_PROJECT: "PRIVATE",
            QUALIFICATION_RESULT_PROJECT: "PRIVATE",
        },
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
        project_access={
            QUALIFICATION_SOURCE_PROJECT: "PRIVATE",
            QUALIFICATION_RESULT_PROJECT: "PRIVATE",
        },
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
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
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
        mcp_release_qualification,
        "verify_private_project_topology",
        lambda **_kwargs: {
            QUALIFICATION_SOURCE_PROJECT: "PRIVATE",
            QUALIFICATION_RESULT_PROJECT: "PRIVATE",
        },
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


def test_mechanism_receipt_keeps_public_endpoint_but_redacts_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    lock = _lock(
        QUALIFICATION_SOURCE_PROJECT,
        result_project=QUALIFICATION_RESULT_PROJECT,
    )
    evidence_lock = tmp_path / "evidence.lock.json"
    evidence_lock.write_text(json.dumps(lock), encoding="utf-8")
    env_file = tmp_path / ".env"
    api_key = "secret-mechanism-key"
    env_file.write_text(f"WANDB_API_KEY={api_key}\n", encoding="utf-8")
    output = tmp_path / "mechanism.json"
    captured = {}

    async def fake_observations(repo_root, evidence, runtime_env):
        captured["repo_root"] = repo_root
        captured["evidence"] = evidence
        captured["runtime_env"] = dict(runtime_env)
        return []

    monkeypatch.setattr(
        mcp_release_qualification,
        "_run_mcp_release_observations",
        fake_observations,
    )

    receipt = mcp_release_qualification.qualify_locked_mcp_revisions(
        repo_root=Path.cwd(),
        evidence_lock=evidence_lock,
        env_file=env_file,
        output=output,
    )

    assert captured["runtime_env"] == {
        "WANDB_API_KEY": api_key,
        "WANDB_BASE_URL": "https://api.wandb.ai",
    }
    assert receipt["endpoint_binding"]["api_base_url"] == "https://api.wandb.ai"
    serialized = output.read_text(encoding="utf-8")
    assert "https://api.wandb.ai" in serialized
    assert api_key not in serialized


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
            "fugue-mcp-release-source-v2",
            QUALIFICATION_SOURCE_PROJECT,
        )
    ]
    assert inventory_calls
    assert all(
        item
        == (
            "wandb",
            "fugue-mcp-release-source-v2",
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
            prediction_id = f"{root_id}-prediction-{index}"
            calls.append(
                {
                    "id": prediction_id,
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
            for suffix, operation in (
                (
                    "model",
                    (
                        "fugue.qualification.baseline_evidence_model"
                        if revision == "maintainer-r17"
                        else "fugue.qualification.candidate_evidence_model"
                    ),
                ),
                (
                    "scorer",
                    "fugue.qualification.evidence_alignment_scorer",
                ),
            ):
                calls.append(
                    {
                        "id": f"{prediction_id}-{suffix}",
                        "project_id": QUALIFICATION_SOURCE_PROJECT,
                        "trace_id": trace_id,
                        "parent_id": prediction_id,
                        "op_name": operation,
                        "inputs": {},
                        "output": {},
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


def test_wandb_artifact_read_supports_latest_download_signature() -> None:
    class Entry:
        def download(self, *, root):
            path = Path(root) / "attempt-evidence.json"
            path.write_text('{"status": "prepared"}', encoding="utf-8")
            return path.as_posix()

    class Artifact:
        def get_path(self, name):
            assert name == "attempt-evidence.json"
            return Entry()

    assert mcp_release_qualification._read_run_artifact_payload(Artifact()) == {
        "status": "prepared"
    }


def test_wandb_history_preserves_intentionally_missing_cost() -> None:
    class Run:
        def scan_history(self, *, keys, page_size):
            assert page_size == 1000
            if keys == ["step", "observed_cost_usd"]:
                return iter(())
            return iter(
                (
                    {
                        "step": 3,
                        "latency_ms": 4200,
                        "broad_reads": 4,
                        "projected_reads": 0,
                        "deterministic_pass": 0,
                        "source_returned": 4,
                        "source_opened": 1,
                    },
                )
            )

    assert mcp_release_qualification._canonical_run_history(Run()) == [
        {
            "step": 3,
            "latency_ms": 4200,
            "broad_reads": 4,
            "projected_reads": 0,
            "deterministic_pass": 0,
            "source_returned": 4,
            "source_opened": 1,
        }
    ]


def test_hosted_call_inventory_redacts_code_bearing_custom_objects() -> None:
    from weave.trace.serialization.custom_objs import UnsafeDeserializationError

    class UnsafeMapping(dict):
        def items(self):
            raise UnsafeDeserializationError("hosted op must not be decoded")

    assert mcp_release_qualification._plain_json_value(UnsafeMapping()) == {
        "_fugue_untrusted_custom_object": True
    }


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


def _prerequisite_v3_result(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    comparison_id: str,
) -> ComparisonResultV3:
    source_project = QUALIFICATION_SOURCE_PROJECT
    result_project = QUALIFICATION_RESULT_PROJECT
    source_digest = "f" * 64
    source_destination = EvidenceDestinationV1(
        entity="wandb",
        project="fugue-mcp-release-source-v2",
        api_base_url="https://api.wandb.ai",
        trace_base_url="https://trace.wandb.ai",
        app_base_url="https://wandb.ai",
    )
    result_destination = EvidenceDestinationV1(
        entity="wandb",
        project="fugue-mcp-release-qualification-v1",
        api_base_url="https://api.wandb.ai",
        trace_base_url="https://trace.wandb.ai",
        app_base_url="https://wandb.ai",
    )
    matched = EvidenceDriftCheckV1(
        status="matched",
        expected_digest=source_digest,
        observed_digest=source_digest,
    )
    topology = EvidenceTopologyV1(
        source_destination=source_destination,
        result_destination=result_destination,
        source_lock_digest=source_digest,
        pre_run_drift=matched,
        post_run_drift=matched,
        execution_identity="e" * 64,
    )
    integration_ids = {
        "baseline": "wandb-mcp-main",
        "candidate": "wandb-mcp-0-4-staging",
    }
    integration_locks = {
        "wandb-mcp-main": {
            "version_identity": ("git:53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0"),
            "runtime_digest": (
                "sha256:"
                "d7b861a16b6a23007e3fda15318aa2c7a635b65aa7de5b7d8fe1aaf9a7fcb339"
            ),
        },
        "wandb-mcp-0-4-staging": {
            "version_identity": ("git:29cc1b5b5cf4061afa1faa712021fa1b68ad0bf7"),
            "runtime_digest": (
                "sha256:"
                "288ef25a1308b51a0d3a4e994a839b88ac498e1bd7a9839f0ac1a96bc91475fd"
            ),
        },
    }
    lock_paths: dict[str, Path] = {}
    for integration_id, value in integration_locks.items():
        lock_path = tmp_path / f"{integration_id}-cohort-lock.json"
        lock_path.write_text(
            json.dumps(value, sort_keys=True),
            encoding="utf-8",
        )
        lock_paths[integration_id] = lock_path
    original_safe_input = comparison_module._safe_input_path

    def cohort_lock_input(path: Path, repo_root: Path, label: str) -> Path:
        for prefix in (
            "baseline cohort integration lock ",
            "candidate cohort integration lock ",
        ):
            if label.startswith(prefix):
                return lock_paths[label.removeprefix(prefix)]
        return original_safe_input(path, repo_root, label)

    monkeypatch.setattr(
        comparison_module,
        "_safe_input_path",
        cohort_lock_input,
    )
    rows: list[dict[str, object]] = []
    for task_id in (
        "maintainer-evaluation-reconciliation",
        "maintainer-project-health",
    ):
        for trial_index in (1, 2):
            for variant, passed in (
                ("baseline", False),
                ("candidate", True),
            ):
                integration_id = integration_ids[variant]
                lock = integration_locks[integration_id]
                call_prefix = f"{variant}-{task_id}-{trial_index}"
                call_ref_prefix = f"weave:///{result_project}/call/{call_prefix}"

                rows.append(
                    {
                        "variant_id": variant,
                        "task_id": task_id,
                        "harness": "claude-code",
                        "trial_index": trial_index,
                        "candidate_digest": stable_digest(
                            {"variant": variant, "candidate": integration_id}
                        ),
                        "execution_fingerprint": (
                            ("a" if variant == "baseline" else "b") * 64
                        ),
                        "trace_project": result_project,
                        "trace_receipt": result_destination.to_dict(),
                        "queried_projects": [source_project],
                        "pass": passed,
                        "status": "passed",
                        "comparison_evaluation_status": "completed",
                        "comparison_required_evaluation_complete": True,
                        "comparison_deterministic_scores": {
                            "natural-maintainer.answer_correct": passed,
                        },
                        "comparison_deterministic_criticality": {
                            "natural-maintainer.answer_correct": True,
                        },
                        "comparison_dimension_roles": {
                            "natural-maintainer.answer_correct": "outcome",
                        },
                        "integration_provenance": [
                            {
                                "kind": "mcp",
                                "id": integration_id,
                                "version_identity": lock["version_identity"],
                                "runtime_digest": lock["runtime_digest"],
                                "lock_digest": "sha256:" + stable_digest(lock),
                            }
                        ],
                        "weave_evaluation_root_call_id": (f"{call_prefix}-evaluation"),
                        "weave_evaluation_root_ref": (f"{call_ref_prefix}-evaluation"),
                        "evaluation_root_object_verified": True,
                        "evaluation_root_dataset_relationship_verified": True,
                        "evaluation_root_prediction_relationship_verified": True,
                        "weave_dataset_id": (
                            f"weave:///{result_project}/object/"
                            "maintainer-canary-dataset:v1"
                        ),
                        "dataset_version_object_verified": True,
                        "eval_predict_and_score_call_id": (
                            f"{call_prefix}-predict-and-score"
                        ),
                        "eval_predict_and_score_ref": (
                            f"{call_ref_prefix}-predict-and-score"
                        ),
                        "eval_predict_and_score_object_verified": True,
                        "prediction_call_id": f"{call_prefix}-prediction",
                        "weave_prediction_ref": f"{call_ref_prefix}-prediction",
                        "weave_prediction_object_verified": True,
                        "prediction_child_relationship_verified": True,
                        "evaluation_prediction_graph_verified": True,
                        "native_agent_root_call_id": f"{call_prefix}-agent",
                        "weave_agent_root_ref": f"{call_ref_prefix}-agent",
                        "trace_link_status": "linked",
                        "agent_graph_verified": True,
                        "infrastructure_conformance_complete": True,
                        "privacy_contract_version": 2,
                        "local_artifact_privacy_scan_status": "passed",
                        "local_artifact_privacy_scan_digest": "1" * 64,
                        "local_artifact_privacy_match_count": 0,
                        "hosted_evidence_privacy_scan_status": "passed",
                        "hosted_evidence_privacy_scan_digest": "2" * 64,
                        "hosted_evidence_privacy_match_count": 0,
                        "private_label_boundary_verified": True,
                        "sandbox_cleanup_verified": True,
                        "sandbox_deleted": True,
                        "orphaned_sandbox": False,
                        "harbor_environment": "local_harbor_docker",
                        "harbor_conformance_status": "passed",
                        "harbor_conformance_receipt_digest": "f" * 64,
                        "harbor_policy_attestation_verified": True,
                        "source_pre_run_drift": matched.to_dict(),
                        "source_post_run_drift": matched.to_dict(),
                    }
                )
    canary_spec = load_comparison(
        EXAMPLE / "natural-maintainer-canary-local-v3.yaml",
        repo_root=Path.cwd(),
    )
    cohort_lineage = comparison_module._comparison_cohort_lineage(
        canary_spec,
        repo_root=Path.cwd(),
        source_lock_digest=source_digest,
    )
    execution_lock = {
        "cohort_lineage": cohort_lineage,
        "lock_digest": "0" * 64,
        "expected_cell_count": len(rows),
        "evidence_project": result_project,
        "evidence_destination": result_destination.to_dict(),
        "source_evidence_project": source_project,
        "source_evidence_destination": source_destination.to_dict(),
        "source_lock_digest": source_digest,
        "evidence_topology_identity": "e" * 64,
    }
    monkeypatch.setattr(
        comparison_module,
        "_resolve_approved_comparison_execution_lock",
        lambda _rows, *, supplied: execution_lock,
    )
    monkeypatch.setattr(
        comparison_module,
        "_validate_approved_comparison_rows",
        lambda _rows, *, source, approved: None,
    )
    monkeypatch.setattr(
        comparison_module,
        "_scorer_revisions_v3",
        lambda _execution_lock: (
            LockDescriptorV1(
                id="natural-maintainer",
                label="Natural maintainer",
                digest=str(cohort_lineage["scorer_digests"]["natural-maintainer"]),
                details={"kind": "scorer"},
            ),
        ),
    )
    result = analyze_comparison_rows(
        comparison_id=comparison_id,
        preview_digest="c" * 64,
        rows=rows,
        source="canary-test-run",
        expected_evidence_project=result_project,
        expected_source_evidence_project=source_project,
        result_schema_version=3,
        evidence_topology=topology,
    )
    assert isinstance(result, ComparisonResultV3)
    assert result.behavioral_summary.status == "improved"
    assert result.integrity["status"] == "reconciled"
    assert all(item.status == "valid" for item in result.task_validity)
    return result


def _use_prerequisite_paths(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    result_path: Path,
    attestation_path: Path,
) -> None:
    inputs = {
        "evidence lock": tmp_path / "evidence-lock.json",
        "release-notes lock": tmp_path / "release-notes-lock.json",
        "source conformance receipt": tmp_path / "source-conformance.json",
        "mechanism receipt": tmp_path / "mechanism-receipt.json",
    }
    inputs["evidence lock"].write_text(
        json.dumps({"evidence_lock_digest": "f" * 64}),
        encoding="utf-8",
    )
    inputs["release-notes lock"].write_text(
        json.dumps({"commit": ("29cc1b5b5cf4061afa1faa712021fa1b68ad0bf7")}),
        encoding="utf-8",
    )
    for label in ("source conformance receipt", "mechanism receipt"):
        receipt = {
            "schema_version": 1,
            "kind": label.replace(" ", "-"),
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = stable_digest(receipt)
        inputs[label].write_text(
            json.dumps(receipt, sort_keys=True),
            encoding="utf-8",
        )
    integration_locks = {
        "wandb-mcp-main": {
            "version_identity": ("git:53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0"),
            "runtime_digest": (
                "sha256:"
                "d7b861a16b6a23007e3fda15318aa2c7a635b65aa7de5b7d8fe1aaf9a7fcb339"
            ),
        },
        "wandb-mcp-0-4-staging": {
            "version_identity": ("git:29cc1b5b5cf4061afa1faa712021fa1b68ad0bf7"),
            "runtime_digest": (
                "sha256:"
                "288ef25a1308b51a0d3a4e994a839b88ac498e1bd7a9839f0ac1a96bc91475fd"
            ),
        },
    }
    lock_paths = {}
    for integration_id, value in integration_locks.items():
        lock_path = tmp_path / f"{integration_id}.json"
        lock_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        lock_paths[integration_id] = lock_path
    original = comparison_module._safe_input_path

    def selected(path: Path, repo_root: Path, label: str) -> Path:
        if label == "follow-up prerequisite result":
            return path.resolve()
        if label == "prerequisite result":
            if path.is_file():
                return path.resolve()
            return result_path
        if label == "prerequisite attestation":
            if path.is_file():
                return path.resolve()
            return attestation_path
        if label in inputs:
            return inputs[label]
        for prefix in (
            "prerequisite MCP lock ",
            "MCP release candidate lock ",
            "baseline cohort integration lock ",
            "candidate cohort integration lock ",
        ):
            if label.startswith(prefix):
                return lock_paths[label.removeprefix(prefix)]
        return original(path, repo_root, label)

    monkeypatch.setattr(comparison_module, "_safe_input_path", selected)
    monkeypatch.setattr(
        mcp_release_qualification,
        "validate_evidence_lock",
        lambda raw, **_kwargs: raw,
    )
    monkeypatch.setattr(
        mcp_release_qualification,
        "validate_release_notes_lock",
        lambda raw: raw,
    )


def _write_prerequisite_attestation(
    result: ComparisonResultV3,
    path: Path,
) -> None:
    canary_spec = load_comparison(
        EXAMPLE / "natural-maintainer-canary-local-v3.yaml",
        repo_root=Path.cwd(),
    )
    attestation = {
        "schema_version": 1,
        "kind": "comparison-prerequisite-attestation",
        "comparison_id": result.comparison_id,
        "qualification_digest": result.qualification_digest,
        "result_digest": result.result_digest,
        "source_lock_digest": result.evidence_topology.source_lock_digest,
        "runtime_locks_digest": stable_digest(
            [item.to_dict() for item in result.runtime_locks]
        ),
        "scorer_revisions_digest": stable_digest(
            [item.to_dict() for item in result.scorer_revisions]
        ),
        "cohort_lineage_digest": result.cohort_lineage["lineage_digest"],
        "taskset_digest": comparison_module._sha256_path(
            Path.cwd() / canary_spec.taskset.tasks
        ),
        "private_labels_digest": comparison_module._sha256_path(
            Path.cwd() / canary_spec.taskset.private_labels
        ),
        "review_status": "accepted_valid_non_regressing_useful",
        "reviewed_by": "test-release-owner",
        "reviewed_at": "2026-07-30T00:00:00Z",
        "receipt_digest": "",
    }
    attestation["receipt_digest"] = stable_digest(attestation)
    path.write_text(
        json.dumps(attestation, sort_keys=True),
        encoding="utf-8",
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
    scorer = runpy.run_path((EXAMPLE / "natural_maintainer_scorer.py").as_posix())[
        "score"
    ]

    def fake_inline_runner(*, source, evidence, reference, profile, limits):
        assert 'if __name__ == "__main__":' in source
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
        == "fugue-mcp-release-source-v2"
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
        "29cc1b5b5cf4061afa1faa712021fa1b68ad0bf7"
    )
    gates = {gate.source: gate.target for gate in spec.decision_policy.gates}
    assert gates["matrix.terminal_rows"] == cells
    assert gates["task.candidate_passed"] == candidate_passes
    assert gates["infrastructure.gate.final-staging-head"] is True
    assert "infrastructure.gate.human-maintainer-actionability-review" not in gates
    assert gates["infrastructure.gate.fresh-wheel-python-3-11"] is True
    assert gates["infrastructure.gate.fresh-wheel-python-3-12"] is True
    assert gates["infrastructure.gate.installed-wheel-stdio-protocol"] is True
    assert not any(
        "infrastructure receipt is not usable" in blocker
        for blocker in readiness.blockers
    )
    assert any(
        "package-release gate does not block this local behavioral study" in warning
        and "infrastructure receipt is not usable" in warning
        for warning in readiness.warnings
    )


def test_v3_natural_maintainer_spec_requires_a_locked_role_per_dimension() -> None:
    path = EXAMPLE / "natural-maintainer-canary-local-v3.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["evaluators"][0]["dimension_roles"]["answer_correct"]

    with pytest.raises(ValueError, match="typed role for every dimension"):
        comparison_from_dict(raw, repo_root=Path.cwd(), source=path)


def test_natural_maintainer_confirmation_requires_canary_result_and_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_comparison(
        EXAMPLE / "natural-maintainer-confirmation-local-v3.yaml",
        repo_root=Path.cwd(),
    )
    assert spec.execution.prerequisite_comparison_id == (
        "mcp-main-vs-0-4-natural-maintainer-canary-v3"
    )
    assert spec.execution.prerequisite_result == (
        ".fugue/qualification/mcp-natural-maintainer-canary-v3/result.json"
    )
    assert spec.execution.prerequisite_attestation == (
        ".fugue/qualification/mcp-natural-maintainer-canary-v3/"
        "prerequisite-attestation.json"
    )
    _use_prerequisite_paths(
        monkeypatch,
        tmp_path=tmp_path,
        result_path=tmp_path / "missing-result.json",
        attestation_path=tmp_path / "missing-attestation.json",
    )

    readiness = check_comparison(spec, repo_root=Path.cwd())

    assert readiness.status == "blocked"
    assert any("missing-result.json" in blocker for blocker in readiness.blockers), (
        readiness.blockers
    )


def test_authorize_followup_promotes_exact_canary_for_confirmation_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _prerequisite_v3_result(
        monkeypatch,
        tmp_path=tmp_path,
        comparison_id="mcp-main-vs-0-4-natural-maintainer-canary-v3",
    )
    source_result = tmp_path / "source-result.json"
    source_result.write_text(
        json.dumps(result.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    canonical_result = tmp_path / "canonical" / "result.json"
    canonical_attestation = tmp_path / "canonical" / "attestation.json"
    _use_prerequisite_paths(
        monkeypatch,
        tmp_path=tmp_path,
        result_path=canonical_result,
        attestation_path=canonical_attestation,
    )
    requested_outputs: dict[str, Path] = {}

    def selected_output(path: Path, _repo_root: Path, label: str) -> Path:
        requested_outputs[label] = path
        return {
            "canonical prerequisite result": canonical_result,
            "canonical prerequisite attestation": canonical_attestation,
        }[label]

    monkeypatch.setattr(
        comparison_module,
        "_safe_repository_output_path",
        selected_output,
    )
    receipt, promoted_result, promoted_attestation = (
        comparison_module.authorize_comparison_followup(
            result_path=source_result,
            followup_spec_path=(
                EXAMPLE / "natural-maintainer-confirmation-local-v3.yaml"
            ),
            reviewed_by="test-release-owner",
            reviewed_at="2026-07-30T00:00:00Z",
            repo_root=Path.cwd(),
        )
    )

    assert requested_outputs == {
        "canonical prerequisite result": Path(
            ".fugue/qualification/mcp-natural-maintainer-canary-v3/result.json"
        ),
        "canonical prerequisite attestation": Path(
            ".fugue/qualification/mcp-natural-maintainer-canary-v3/"
            "prerequisite-attestation.json"
        ),
    }
    assert promoted_result == canonical_result
    assert promoted_attestation == canonical_attestation
    assert json.loads(canonical_result.read_text()) == json.loads(
        source_result.read_text()
    )
    attestation = json.loads(canonical_attestation.read_text())
    assert attestation["qualification_digest"] == result.qualification_digest
    assert attestation["review_status"] == ("accepted_valid_non_regressing_useful")
    assert receipt["prerequisite_result"] == result.qualification_digest

    confirmation = load_comparison(
        EXAMPLE / "natural-maintainer-confirmation-local-v3.yaml",
        repo_root=Path.cwd(),
    )
    accepted = comparison_module._validate_prerequisite_result_binding(
        confirmation,
        repo_root=Path.cwd(),
        source_lock_digest=result.evidence_topology.source_lock_digest,
    )
    assert accepted["prerequisite_result"] == result.qualification_digest
    assert accepted["prerequisite_attestation"] == (attestation["receipt_digest"])


def test_natural_maintainer_confirmation_rejects_nonmatching_v3_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _prerequisite_v3_result(
        monkeypatch,
        tmp_path=tmp_path,
        comparison_id="different-natural-maintainer-canary-v3",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text("{}", encoding="utf-8")
    _use_prerequisite_paths(
        monkeypatch,
        tmp_path=tmp_path,
        result_path=result_path,
        attestation_path=attestation_path,
    )
    spec = load_comparison(
        EXAMPLE / "natural-maintainer-confirmation-local-v3.yaml",
        repo_root=Path.cwd(),
    )

    readiness = check_comparison(spec, repo_root=Path.cwd())

    assert readiness.status == "blocked"
    assert any(
        "prerequisite comparison identity does not match" in blocker
        for blocker in readiness.blockers
    ), readiness.blockers


def test_natural_maintainer_confirmation_rejects_nonmatching_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _prerequisite_v3_result(
        monkeypatch,
        tmp_path=tmp_path,
        comparison_id="mcp-main-vs-0-4-natural-maintainer-canary-v3",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    attestation = {
        "schema_version": 1,
        "kind": "comparison-prerequisite-attestation",
        "comparison_id": result.comparison_id,
        "qualification_digest": result.qualification_digest,
        "result_digest": "0" * 64,
        "source_lock_digest": result.evidence_topology.source_lock_digest,
        "runtime_locks_digest": stable_digest(
            [item.to_dict() for item in result.runtime_locks]
        ),
        "scorer_revisions_digest": stable_digest(
            [item.to_dict() for item in result.scorer_revisions]
        ),
        "review_status": "accepted_valid_non_regressing_useful",
        "reviewed_by": "test-release-owner",
        "reviewed_at": "2026-07-30T00:00:00Z",
        "receipt_digest": "",
    }
    attestation["receipt_digest"] = stable_digest(attestation)
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(
        json.dumps(attestation, sort_keys=True),
        encoding="utf-8",
    )
    _use_prerequisite_paths(
        monkeypatch,
        tmp_path=tmp_path,
        result_path=result_path,
        attestation_path=attestation_path,
    )
    spec = load_comparison(
        EXAMPLE / "natural-maintainer-confirmation-local-v3.yaml",
        repo_root=Path.cwd(),
    )

    readiness = check_comparison(spec, repo_root=Path.cwd())

    assert readiness.status == "blocked"
    assert any(
        "prerequisite attestation does not sign the exact useful canary result"
        in blocker
        for blocker in readiness.blockers
    ), readiness.blockers


def test_natural_maintainer_confirmation_rejects_common_runtime_lineage_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _prerequisite_v3_result(
        monkeypatch,
        tmp_path=tmp_path,
        comparison_id="mcp-main-vs-0-4-natural-maintainer-canary-v3",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    attestation_path = tmp_path / "attestation.json"
    _write_prerequisite_attestation(result, attestation_path)
    _use_prerequisite_paths(
        monkeypatch,
        tmp_path=tmp_path,
        result_path=result_path,
        attestation_path=attestation_path,
    )
    spec = load_comparison(
        EXAMPLE / "natural-maintainer-confirmation-local-v3.yaml",
        repo_root=Path.cwd(),
    )
    drifted = replace(
        spec,
        execution=replace(
            spec.execution,
            model="anthropic/claude-sonnet-drifted",
        ),
    )

    readiness = check_comparison(drifted, repo_root=Path.cwd())

    assert any(
        "execution lineage does not match the confirmation cohort" in blocker
        for blocker in readiness.blockers
    ), readiness.blockers


def test_natural_maintainer_confirmation_binds_exact_canary_task_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _prerequisite_v3_result(
        monkeypatch,
        tmp_path=tmp_path,
        comparison_id="mcp-main-vs-0-4-natural-maintainer-canary-v3",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    attestation_path = tmp_path / "attestation.json"
    _write_prerequisite_attestation(result, attestation_path)
    _use_prerequisite_paths(
        monkeypatch,
        tmp_path=tmp_path,
        result_path=result_path,
        attestation_path=attestation_path,
    )
    canary_tasks = (
        Path.cwd() / "examples/comparisons/wandb-mcp-maintenance/"
        "natural-maintainer-canary-tasks.jsonl"
    )
    original_sha256_path = comparison_module._sha256_path

    def drift_task_digest(path: Path) -> str:
        if path.resolve() == canary_tasks.resolve():
            return "0" * 64
        return original_sha256_path(path)

    monkeypatch.setattr(
        comparison_module,
        "_sha256_path",
        drift_task_digest,
    )
    spec = load_comparison(
        EXAMPLE / "natural-maintainer-confirmation-local-v3.yaml",
        repo_root=Path.cwd(),
    )

    readiness = check_comparison(spec, repo_root=Path.cwd())

    assert any(
        "execution lineage does not match the confirmation cohort" in blocker
        for blocker in readiness.blockers
    ), readiness.blockers


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
    assert "019fb261-e3b0-7303-b683-99af42e85bb8" not in public
    assert "019fb261-efc7-7838-ba01-4f1e32d64878" not in public


@pytest.mark.parametrize(
    "task_filename",
    [
        "natural-maintainer-canary-tasks.jsonl",
        "natural-maintainer-confirmation-tasks.jsonl",
    ],
)
def test_v3_natural_maintainer_tasks_publish_the_complete_output_contract(
    task_filename: str,
) -> None:
    public_tasks = [
        json.loads(line)
        for line in (EXAMPLE / task_filename).read_text().splitlines()
        if line.strip()
    ]

    for task in public_tasks:
        prompt = task["input"]["question"]
        assert "locked, read-only source cohort" in prompt
        assert "fields and types" in prompt
        assert "at most 1200 characters" in prompt
        assert "immutable source project" not in prompt
    inventory = (
        next(
            task for task in public_tasks if task["id"] == "maintainer-source-inventory"
        )
        if any(task["id"] == "maintainer-source-inventory" for task in public_tasks)
        else None
    )
    if inventory is not None:
        prompt = inventory["input"]["question"]
        assert "summary_reported_predictions" in prompt
        assert "direct_prediction_rows" not in prompt
        assert "`summary-only`, `incomplete`, or `conflicted`" in prompt


@pytest.mark.parametrize(
    "private_filename",
    [
        "natural-maintainer-canary-private.jsonl",
        "natural-maintainer-confirmation-private.jsonl",
    ],
)
def test_natural_maintainer_private_reconciliation_locks_exact_roots_and_ops(
    private_filename: str,
) -> None:
    roots = {
        "019fb261-e3b0-7303-b683-99af42e85bb8",
        "019fb261-efc7-7838-ba01-4f1e32d64878",
    }
    labels = [
        json.loads(line)
        for line in (EXAMPLE / private_filename).read_text().splitlines()
        if line.strip()
    ]
    label = next(
        item for item in labels if item["id"] == "maintainer-evaluation-reconciliation"
    )
    expected = label["expected"]

    assert set(expected["evaluation_parent_ids"]) == roots
    expected_counts = expected["mechanism"]["evaluation_parent_operation_counts"]
    assert set(expected_counts) == roots
    assert all(
        counts
        == {
            "Evaluation.predict_and_score": 8,
            "Evaluation.summarize": 1,
        }
        for counts in expected_counts.values()
    )
    for evidence_key in ("base_evidence", "gold_evidence"):
        child_calls = [
            call
            for call in label[evidence_key]["mcp_tool_calls"]
            if call["tool"] == "query_weave_traces_tool"
        ]
        assert {call["parent_filter_ids"][0] for call in child_calls} == roots
        assert all(
            call["operation_counts"]
            == {
                "Evaluation.predict_and_score": 8,
                "Evaluation.summarize": 1,
            }
            for call in child_calls
        )


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


@pytest.mark.parametrize(
    ("private_filename", "task_id", "factual_field", "wrong_value"),
    [
        (
            "natural-maintainer-canary-private.jsonl",
            "maintainer-evaluation-reconciliation",
            "evaluation_root_count",
            999,
        ),
        (
            "natural-maintainer-canary-private.jsonl",
            "maintainer-project-health",
            "largest_latency_ms",
            999,
        ),
        (
            "natural-maintainer-confirmation-private.jsonl",
            "maintainer-source-inventory",
            "run_count",
            999,
        ),
        (
            "natural-maintainer-confirmation-private.jsonl",
            "maintainer-history-hotspot",
            "latency_ms",
            999,
        ),
    ],
)
def test_natural_maintainer_scorer_dimensions_are_independent(
    private_filename: str,
    task_id: str,
    factual_field: str,
    wrong_value: object,
) -> None:
    score = _natural_maintainer_score()
    label = next(
        json.loads(line)
        for line in (EXAMPLE / private_filename).read_text().splitlines()
        if json.loads(line)["id"] == task_id
    )
    task = {"id": task_id}
    evidence = {**label["gold_evidence"], "expected": label["expected"]}

    wrong_fact = {**label["gold_output"], factual_field: wrong_value}
    factual_result = score(task, wrong_fact, evidence)
    assert factual_result["answer_correct"] is False
    assert factual_result["actual_query_scope"] is True
    assert factual_result["reported_project_identity"] is True
    assert factual_result["bounded_evidence"] is True
    assert factual_result["evidence_honesty"] is True
    assert factual_result["release_mechanism_used"] is True

    wrong_identity = {
        **label["gold_output"],
        "source_project": "wandb/not-the-locked-source",
    }
    identity_result = score(task, wrong_identity, evidence)
    assert identity_result["answer_correct"] is True
    assert identity_result["actual_query_scope"] is True
    assert identity_result["reported_project_identity"] is False
    assert identity_result["bounded_evidence"] is True
    assert identity_result["evidence_honesty"] is True
    assert identity_result["release_mechanism_used"] is True

    unbounded = {**label["gold_output"], "bounded": False}
    bounded_result = score(task, unbounded, evidence)
    assert bounded_result["answer_correct"] is True
    assert bounded_result["actual_query_scope"] is True
    assert bounded_result["reported_project_identity"] is True
    assert bounded_result["bounded_evidence"] is False
    assert bounded_result["evidence_honesty"] is True
    assert bounded_result["release_mechanism_used"] is True

    dishonest = {**label["gold_output"], "evidence_status": "incomplete"}
    honesty_result = score(task, dishonest, evidence)
    assert honesty_result["answer_correct"] is True
    assert honesty_result["actual_query_scope"] is True
    assert honesty_result["reported_project_identity"] is True
    assert honesty_result["bounded_evidence"] is True
    assert honesty_result["evidence_honesty"] is False
    assert honesty_result["release_mechanism_used"] is True


@pytest.mark.parametrize(
    ("private_filename", "task_id", "tool"),
    [
        (
            "natural-maintainer-canary-private.jsonl",
            "maintainer-evaluation-reconciliation",
            "query_weave_traces_tool",
        ),
        (
            "natural-maintainer-canary-private.jsonl",
            "maintainer-project-health",
            "query_wandb_tool",
        ),
        (
            "natural-maintainer-confirmation-private.jsonl",
            "maintainer-source-inventory",
            "query_wandb_tool",
        ),
        (
            "natural-maintainer-confirmation-private.jsonl",
            "maintainer-history-hotspot",
            "get_run_history_tool",
        ),
    ],
)
def test_natural_maintainer_boundedness_rejects_extra_projected_fields(
    private_filename: str,
    task_id: str,
    tool: str,
) -> None:
    score = _natural_maintainer_score()
    label = next(
        json.loads(line)
        for line in (EXAMPLE / private_filename).read_text().splitlines()
        if json.loads(line)["id"] == task_id
    )
    evidence = json.loads(json.dumps(label["gold_evidence"]))
    call = next(
        item
        for item in evidence["mcp_tool_calls"]
        if item["tool"] == tool
        and (tool != "query_wandb_tool" or item.get("response_mode") == "items")
    )
    fields_key = "keys" if tool == "get_run_history_tool" else "projected_fields"
    call[fields_key].append("unrequested_private_field")

    result = score(
        {"id": task_id},
        label["gold_output"],
        {**evidence, "expected": label["expected"]},
    )

    assert result["actual_query_scope"] is True
    assert result["reported_project_identity"] is True
    assert result["bounded_evidence"] is False
    assert result["release_mechanism_used"] is False


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
    assert prose["actual_query_scope"] is True
    assert prose["reported_project_identity"] is False
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
    assert result["actual_query_scope"] is False
    assert result["reported_project_identity"] is True


def test_natural_maintainer_reconciliation_rejects_arbitrary_evaluation_roots() -> None:
    score = _natural_maintainer_score()
    label = json.loads(
        (EXAMPLE / "natural-maintainer-canary-private.jsonl")
        .read_text()
        .splitlines()[0]
    )
    evidence = json.loads(json.dumps(label["gold_evidence"]))
    child_calls = [
        call
        for call in evidence["mcp_tool_calls"]
        if call["tool"] == "query_weave_traces_tool"
    ]
    child_calls[0]["parent_filter_ids"] = ["unrelated-evaluation-root-a"]
    child_calls[1]["parent_filter_ids"] = ["unrelated-evaluation-root-b"]

    result = score(
        {"id": label["id"]},
        label["gold_output"],
        {**evidence, "expected": label["expected"]},
    )

    assert result["actual_query_scope"] is True
    assert result["reported_project_identity"] is True
    assert result["answer_correct"] is False
    assert result["bounded_evidence"] is False
    assert result["evidence_honesty"] is False
    assert result["release_mechanism_used"] is False


def test_natural_maintainer_scope_requires_project_on_every_successful_call() -> None:
    score = _natural_maintainer_score()
    label = json.loads(
        (EXAMPLE / "natural-maintainer-canary-private.jsonl")
        .read_text()
        .splitlines()[0]
    )
    evidence = json.loads(json.dumps(label["gold_evidence"]))
    evidence["mcp_tool_calls"][1].pop("queried_projects")

    result = score(
        {"id": label["id"]},
        label["gold_output"],
        {**evidence, "expected": label["expected"]},
    )

    assert result["answer_correct"] is True
    assert result["actual_query_scope"] is False
    assert result["reported_project_identity"] is True


def test_natural_maintainer_reconciliation_rejects_wrong_operation_histogram() -> None:
    score = _natural_maintainer_score()
    label = json.loads(
        (EXAMPLE / "natural-maintainer-canary-private.jsonl")
        .read_text()
        .splitlines()[0]
    )
    evidence = json.loads(json.dumps(label["gold_evidence"]))
    child_call = next(
        call
        for call in evidence["mcp_tool_calls"]
        if call["tool"] == "query_weave_traces_tool"
    )
    child_call["operation_counts"] = {
        "Evaluation.predict_and_score": 9,
        "Evaluation.summarize": 0,
    }

    result = score(
        {"id": label["id"]},
        label["gold_output"],
        {**evidence, "expected": label["expected"]},
    )

    assert result["actual_query_scope"] is True
    assert result["reported_project_identity"] is True
    assert result["answer_correct"] is False
    assert result["bounded_evidence"] is True
    assert result["evidence_honesty"] is False
    assert result["release_mechanism_used"] is True


def test_natural_maintainer_reconciliation_requires_two_locked_roots() -> None:
    score = _natural_maintainer_score()
    label = json.loads(
        (EXAMPLE / "natural-maintainer-canary-private.jsonl")
        .read_text()
        .splitlines()[0]
    )
    evidence = json.loads(json.dumps(label["gold_evidence"]))
    summary = next(
        call
        for call in evidence["mcp_tool_calls"]
        if call["tool"] == "summarize_evaluation_tool"
    )
    summary["total_count"] = 3

    result = score(
        {"id": label["id"]},
        label["gold_output"],
        {**evidence, "expected": label["expected"]},
    )

    assert result["actual_query_scope"] is True
    assert result["reported_project_identity"] is True
    assert result["answer_correct"] is False
    assert result["bounded_evidence"] is False
    assert result["evidence_honesty"] is False
    assert result["release_mechanism_used"] is True


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
                    "project_name": "fugue-mcp-release-source-v2",
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
                    "project_name": "fugue-mcp-release-source-v2",
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
                    "project_name": "fugue-mcp-release-source-v2",
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
