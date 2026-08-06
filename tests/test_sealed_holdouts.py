from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import _followup_authorization_readiness, load_comparison
from fugue.bench.study_advancement import (
    HoldoutExposureAuditV1,
    StudyAdvancementDecisionV1,
    write_holdout_exposure_audit,
    write_study_advancement_decision,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/comparisons/community-skill-selected-v1"
sys.path.insert(0, str(EXAMPLE))

from holdout_support import (  # noqa: E402
    _QUERY_FIELDS,
    _safe_identity_projection,
    allocate_campaign_branch,
    build_live_holdout_audits,
    build_no_skill_diagnostic,
    build_sealed_holdout_comparison,
    fetch_task_identity_projections,
    load_sealed_holdout_manifest,
    validate_campaign_allocation_receipt,
    validate_historical_holdout_exposure_receipt,
    validate_holdout_authorization,
)

MANIFEST = (
    ROOT / "examples/comparisons/community-skill-selected-v1/sealed-holdouts.json"
)


def _preparation(manifest: dict) -> dict:
    lanes = []
    for lane in manifest["lanes"]:
        items = []
        for selection in lane["selections"]:
            for role, task_prefix, paired_prefix, digest_prefix in (
                ("selected", "", "reserve_", ""),
                ("reserve", "reserve_", "", "reserve_"),
            ):
                task_id = selection[f"{task_prefix}task_id"]
                prompt_fingerprint = stable_digest({"prompt": task_id})
                input_fingerprint = stable_digest({"question": task_id})
                resource_fingerprint = selection[f"{digest_prefix}resource_sha256"]
                items.append(
                    {
                        "lane_id": lane["id"],
                        "task_id": task_id,
                        "role": role,
                        "behavior_family": selection[
                            f"{task_prefix}behavior_family"
                        ],
                        "paired_task_id": selection[f"{paired_prefix}task_id"],
                        "source_task_digest": selection[
                            f"{digest_prefix}task_sha256"
                        ],
                        "private_label_digest": selection[
                            f"{digest_prefix}private_label_sha256"
                        ],
                        "resource_digest": selection[
                            f"{digest_prefix}resource_sha256"
                        ],
                        "prepared_task_digest": stable_digest({"task_id": task_id}),
                        "task_sha256": selection[f"{digest_prefix}task_sha256"],
                        "prompt_fingerprint": prompt_fingerprint,
                        "input_fingerprint": input_fingerprint,
                        "resource_fingerprint": resource_fingerprint,
                        "content_fingerprint": stable_digest(
                            {
                                "prompt_fingerprint": prompt_fingerprint,
                                "input_fingerprint": input_fingerprint,
                                "resource_fingerprint": resource_fingerprint,
                            }
                        ),
                    }
                )
        selected_ids = {item["task_id"] for item in items if item["role"] == "selected"}
        reserve_ids = {item["task_id"] for item in items if item["role"] == "reserve"}

        def suite_digest(ids: set[str], prepared: list[dict]) -> str:
            selected = [item for item in prepared if item["task_id"] in ids]
            return stable_digest(
                {"items": sorted(selected, key=lambda item: item["task_id"])}
            )

        lanes.append(
            {
                "lane_id": lane["id"],
                "study_id": lane["study_id"],
                "holdout_suite_digest": suite_digest(selected_ids, items),
                "tasks_path": f"private/{lane['id']}/tasks.jsonl",
                "tasks_sha256": "1" * 64,
                "private_labels_path": f"private/{lane['id']}/labels.jsonl",
                "private_labels_sha256": "2" * 64,
                "reserve_tasks_path": f"private/{lane['id']}/reserve-tasks.jsonl",
                "reserve_tasks_sha256": "3" * 64,
                "reserve_private_labels_path": f"private/{lane['id']}/reserve-labels.jsonl",
                "reserve_private_labels_sha256": "4" * 64,
                "reserve_pool_digest": suite_digest(reserve_ids, items),
                "items": items,
            }
        )
    unsigned = {
        "schema_version": 1,
        "kind": "sealed_holdout_preparation_receipt",
        "manifest_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_digest": stable_digest(manifest),
        "historical_exposure_receipt_digest": "5" * 64,
        "pool_fingerprint_digest": stable_digest(_pool(lanes)),
        "lanes": lanes,
        "selected_task_count": 12,
        "reserve_task_count": 12,
        "private_content_in_receipt": False,
    }
    return {**unsigned, "receipt_digest": stable_digest(unsigned)}


def _pool(lanes: list[dict]) -> list[dict]:
    keys = (
        "lane_id",
        "task_id",
        "role",
        "behavior_family",
        "task_sha256",
        "prompt_fingerprint",
        "input_fingerprint",
        "resource_fingerprint",
        "content_fingerprint",
    )
    values = [{key: item[key] for key in keys} for lane in lanes for item in lane["items"]]
    return sorted(values, key=lambda item: (item["lane_id"], item["task_id"]))


def _coverage(manifest: dict, rows: dict[str, list[dict]]) -> dict[str, dict]:
    return {
        project: {
            "project_ref": project,
            "project_status": "present",
            "returned_call_count": len(rows[project]),
            "query_limit": 10_000,
            "projection_digest": stable_digest(
                sorted(rows[project], key=lambda item: json.dumps(item, sort_keys=True))
            ),
            "truncated": False,
            "complete": True,
        }
        for project in manifest["audit"]["historical_projects"]
    }


def _historical(manifest: dict, preparation: dict) -> dict:
    pool = _pool(preparation["lanes"])
    projects = manifest["audit"]["historical_projects"]
    coverage = _coverage(manifest, {project: [] for project in projects})
    unsigned = {
        "schema_version": 1,
        "kind": "historical_holdout_exposure_receipt",
        "manifest_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_digest": stable_digest(manifest),
        "audit_policy_digest": stable_digest(manifest["audit"]),
        "pool_fingerprint_digest": stable_digest(pool),
        "pool_fingerprints": pool,
        "searched_project_refs": projects,
        "searched_call_count": 0,
        "queried_fields": list(_QUERY_FIELDS),
        "project_coverage": list(coverage.values()),
        "project_coverage_complete": True,
        "projection_digest": stable_digest(
            {
                item["project_ref"]: item["projection_digest"]
                for item in coverage.values()
            }
        ),
        "trace_endpoint_digest": stable_digest({"trace_endpoint": "https://trace.wandb.ai"}),
        "matches": [],
        "required_replacements": [],
        "outcome_data_consulted": False,
        "reviewer_identity_digest": "6" * 64,
        "reviewed_at": "2026-08-06T12:00:00+00:00",
        "status": "reviewed_clear",
        "private_content_in_receipt": False,
    }
    receipt = {**unsigned, "receipt_digest": stable_digest(unsigned)}
    preparation["historical_exposure_receipt_digest"] = receipt["receipt_digest"]
    preparation.pop("receipt_digest", None)
    preparation["receipt_digest"] = stable_digest(preparation)
    return receipt


def test_public_manifest_contains_identities_not_sealed_content() -> None:
    manifest = load_sealed_holdout_manifest(MANIFEST)
    selections = [item for lane in manifest["lanes"] for item in lane["selections"]]
    assert len(selections) == 12
    assert {item["behavior_family"] for item in selections} >= {
        "evidence-destination-reactivation",
        "compatibility-product-version",
        "server-action-authorization",
    }
    assert all(
        item["reserve_behavior_family"] == item["behavior_family"]
        and item["reserve_task_id"] != item["task_id"]
        for item in selections
    )
    assert len({item["reserve_task_id"] for item in selections}) == 12
    serialized = MANIFEST.read_text(encoding="utf-8")
    for private_key in ("question", "expected", "base_output", "gold_output"):
        assert f'"{private_key}"' not in serialized


def test_manifest_rejects_cross_family_or_unbound_reserve(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["lanes"][0]["selections"][0]["reserve_behavior_family"] = "other"
    path = tmp_path / "sealed-holdouts.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="exact selected behavior family"):
        load_sealed_holdout_manifest(path)


def test_live_audit_binds_historical_review_and_safe_fingerprints() -> None:
    manifest = load_sealed_holdout_manifest(MANIFEST)
    preparation = _preparation(manifest)
    historical = _historical(manifest, preparation)
    project_rows = {
        project: [
            {
                "id": f"call-{index}",
                "op_name": "predict",
                "task_ids": [],
                "prompt_fingerprints": [],
                "input_fingerprints": [],
                "resource_fingerprints": [],
            }
        ]
        for index, project in enumerate(manifest["audit"]["historical_projects"])
    }
    audits = build_live_holdout_audits(
        manifest_path=MANIFEST,
        preparation_receipt=preparation,
        historical_receipt=historical,
        project_rows=project_rows,
        project_coverage=_coverage(manifest, project_rows),
        now=__import__("datetime").datetime.fromisoformat("2026-08-06T12:00:00+00:00"),
    )
    assert set(audits) == {
        "superpowers-writing-plans",
        "anthropic-skill-creator",
        "vercel-react-best-practices",
    }
    assert all(audit.outcome_data_consulted is False for audit in audits.values())
    assert all(
        audit.queried_fields == tuple(sorted(_QUERY_FIELDS))
        for audit in audits.values()
    )
    assert all(
        audit.historical_exposure_receipt_digest == historical["receipt_digest"]
        and audit.pool_fingerprint_digest == preparation["pool_fingerprint_digest"]
        for audit in audits.values()
    )


def test_exposed_holdout_accepts_only_its_same_family_preregistered_reserve() -> None:
    manifest = load_sealed_holdout_manifest(MANIFEST)
    preparation = _preparation(manifest)
    historical = _historical(manifest, preparation)
    projects = manifest["audit"]["historical_projects"]
    project_rows = {project: [] for project in projects}
    exposed = manifest["lanes"][0]["selections"][0]
    project_rows[projects[0]] = [
        {
            "id": "call-a",
            "op_name": "predict",
            "task_ids": [exposed["task_id"]],
            "prompt_fingerprints": [],
            "input_fingerprints": [],
            "resource_fingerprints": [],
        }
    ]
    with pytest.raises(ValueError, match="every exposed holdout"):
        build_live_holdout_audits(
            manifest_path=MANIFEST,
            preparation_receipt=preparation,
            historical_receipt=historical,
            project_rows=project_rows,
            project_coverage=_coverage(manifest, project_rows),
        )
    with pytest.raises(ValueError, match="not preregistered"):
        build_live_holdout_audits(
            manifest_path=MANIFEST,
            preparation_receipt=preparation,
            historical_receipt=historical,
            project_rows=project_rows,
            project_coverage=_coverage(manifest, project_rows),
            replacements={exposed["task_id"]: "another-family-reserve"},
        )
    audits = build_live_holdout_audits(
        manifest_path=MANIFEST,
        preparation_receipt=preparation,
        historical_receipt=historical,
        project_rows=project_rows,
        project_coverage=_coverage(manifest, project_rows),
        replacements={exposed["task_id"]: exposed["reserve_task_id"]},
    )
    audit = audits["superpowers-writing-plans"]
    assert audit.status == "replaced_exposed"
    assert audit.replacements[0]["behavior_family"] == exposed["behavior_family"]


def test_live_fetch_requests_no_outputs_scores_or_costs(monkeypatch) -> None:
    requests = []

    class Response:
        status_code = 200
        text = json.dumps(
            {
                "id": "call-a",
                "op_name": "predict",
                "attributes": {"fugue": {"task_id": "task-a"}},
            }
        )

        def raise_for_status(self) -> None:
            return None

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, url, *, json):
            requests.append((url, json))
            return Response()

    monkeypatch.setattr("holdout_support.httpx.Client", Client)
    rows, endpoint, coverage = fetch_task_identity_projections(
        projects=("wandb/project-a",),
        env={"WANDB_API_KEY": "test-key"},
        project_exists=lambda _project: True,
    )
    assert endpoint == "https://trace.wandb.ai"
    assert requests[0][1]["columns"] == list(_QUERY_FIELDS)
    assert requests[0][1]["include_costs"] is False
    assert requests[0][1]["include_feedback"] is False
    assert requests[0][1]["limit"] == 10_001
    assert rows == {
        "wandb/project-a": [
            {
                "id": "call-a",
                "op_name": "predict",
                "task_ids": ["task-a"],
                "prompt_fingerprints": [],
                "input_fingerprints": [],
                "resource_fingerprints": [],
            }
        ]
    }
    assert coverage["wandb/project-a"]["complete"] is True
    assert coverage["wandb/project-a"]["project_status"] == "present"


def test_live_fetch_records_authoritatively_absent_projects(monkeypatch) -> None:
    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            raise AssertionError("absent projects must not query the trace endpoint")

    monkeypatch.setattr("holdout_support.httpx.Client", Client)
    rows, _endpoint, coverage = fetch_task_identity_projections(
        projects=("wandb/missing",),
        env={"WANDB_API_KEY": "test-key"},
        project_exists=lambda _project: False,
    )
    assert rows == {"wandb/missing": []}
    assert coverage["wandb/missing"]["project_status"] == "absent"


def test_safe_projection_rejects_outcome_fields() -> None:
    with pytest.raises(ValueError, match="outcome field"):
        _safe_identity_projection(
            {
                "id": "call-a",
                "op_name": "predict",
                "inputs": {"task_id": "task-a"},
                "output": {"score": 1},
            }
        )


def test_same_prompt_under_another_id_requires_the_same_family_reserve() -> None:
    manifest = load_sealed_holdout_manifest(MANIFEST)
    preparation = _preparation(manifest)
    historical = _historical(manifest, preparation)
    projects = manifest["audit"]["historical_projects"]
    project_rows = {project: [] for project in projects}
    exposed = manifest["lanes"][0]["selections"][0]
    pool_item = next(
        item
        for item in _pool(preparation["lanes"])
        if item["task_id"] == exposed["task_id"]
    )
    project_rows[projects[0]] = [
        {
            "id": "renamed-call",
            "op_name": "predict",
            "task_ids": ["different-task-id"],
            "prompt_fingerprints": [pool_item["prompt_fingerprint"]],
            "input_fingerprints": [],
            "resource_fingerprints": [],
        }
    ]
    with pytest.raises(ValueError, match="every exposed holdout"):
        build_live_holdout_audits(
            manifest_path=MANIFEST,
            preparation_receipt=preparation,
            historical_receipt=historical,
            project_rows=project_rows,
            project_coverage=_coverage(manifest, project_rows),
        )
    audits = build_live_holdout_audits(
        manifest_path=MANIFEST,
        preparation_receipt=preparation,
        historical_receipt=historical,
        project_rows=project_rows,
        project_coverage=_coverage(manifest, project_rows),
        replacements={exposed["task_id"]: exposed["reserve_task_id"]},
    )
    assert audits["superpowers-writing-plans"].status == "replaced_exposed"


def test_historical_receipt_rejects_tamper_missing_project_and_truncation() -> None:
    manifest = load_sealed_holdout_manifest(MANIFEST)
    preparation = _preparation(manifest)
    historical = _historical(manifest, preparation)
    pool = _pool(preparation["lanes"])
    tampered = {**historical, "searched_call_count": 1}
    with pytest.raises(ValueError, match="digest does not match"):
        validate_historical_holdout_exposure_receipt(
            tampered, manifest=manifest, pool_fingerprints=pool
        )

    projects = manifest["audit"]["historical_projects"]
    rows = {project: [] for project in projects}
    coverage = _coverage(manifest, rows)
    coverage.pop(projects[-1])
    with pytest.raises(ValueError, match="every historical project"):
        build_live_holdout_audits(
            manifest_path=MANIFEST,
            preparation_receipt=preparation,
            historical_receipt=historical,
            project_rows=rows,
            project_coverage=coverage,
        )
    coverage = _coverage(manifest, rows)
    coverage[projects[0]]["truncated"] = True
    coverage[projects[0]]["complete"] = False
    with pytest.raises(ValueError, match="incomplete or truncated"):
        build_live_holdout_audits(
            manifest_path=MANIFEST,
            preparation_receipt=preparation,
            historical_receipt=historical,
            project_rows=rows,
            project_coverage=coverage,
        )


def test_no_skill_diagnostic_replaces_holdout_allocation(tmp_path: Path) -> None:
    source_root = ROOT / "examples/comparisons/community-skill-selected-v1"
    lane = "superpowers-writing-plans"
    lane_root = tmp_path / "examples/comparisons/community-skill-selected-v1" / lane
    lane_root.mkdir(parents=True)
    raw = yaml.safe_load((source_root / lane / "comparison.yaml").read_text())
    tasks = [
        {"id": task_id, "input": {"question": task_id}, "partition": "qualification"}
        for task_id in (
            "sp-dev-credential-rotation",
            "sp-dev-single-file-validation-fix",
            "sp-dev-evidence-destination",
            "sp-dev-package-tree-qualification",
        )
    ]
    labels = [
        {
            "id": item["id"],
            "expected": {},
            "base_output": "bad",
            "gold_output": "good",
        }
        for item in tasks
    ]
    (lane_root / "tasks.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in tasks)
    )
    private = (
        tmp_path
        / ".fugue/private/community-skill-selected-v1/superpowers-writing-plans"
    )
    private.mkdir(parents=True)
    (private / "private-labels.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in labels)
    )
    raw["taskset"]["private_labels"] = (
        "../../../../.fugue/private/community-skill-selected-v1/"
        "superpowers-writing-plans/private-labels.jsonl"
    )
    raw["evaluators"][0]["scorer"] = "scorer.py"
    (lane_root / "scorer.py").write_text(
        "def score(task, output, evidence): return {'ok': True}\n"
    )
    (lane_root / "comparison.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    decision = StudyAdvancementDecisionV1(
        schema_version=1,
        kind="study_advancement_decision",
        study_id=raw["id"],
        development_result_digest="a" * 64,
        development_qualification_digest="b" * 64,
        preview_digest="c" * 64,
        status="run_no_skill_diagnostic",
        repeated_improvements=(),
        repeated_regressions=(),
        critical_blockers=(),
        mechanism_gate="passed",
        holdout_suite_digest=None,
        holdout_exposure_audit_digest=None,
        next_action="Run the diagnostic.",
    )
    decision_path = tmp_path / "decision.json"
    write_study_advancement_decision(decision_path, decision)
    lock = build_no_skill_diagnostic(
        comparison_path=lane_root / "comparison.yaml",
        advancement_decision_path=decision_path,
        repo_root=tmp_path,
    )
    assert lock["allocation_action"] == "replaces_holdout"
    assert lock["logical_cells"] == 4
    assert lock["holdout_logical_cells_replaced"] == 16
    assert lock["holdout_logical_cells_admitted"] == 0


def test_advancing_lane_builds_exact_digest_bound_sixteen_cell_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = "superpowers-writing-plans"
    campaign = tmp_path / "examples/comparisons/community-skill-selected-v1"
    lane_root = campaign / lane
    lane_root.mkdir(parents=True)
    raw = yaml.safe_load((MANIFEST.parent / lane / "comparison.yaml").read_text())
    raw["evaluators"] = [raw["evaluators"][0]]
    (lane_root / "comparison.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    (lane_root / "scorer.py").write_text(
        (MANIFEST.parent / lane / "scorer.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    task_ids = (
        "sp-holdout-doc-copy-only",
        "sp-holdout-exactly-once-recovery",
        "sp-holdout-project-reactivation-a-b-a",
        "sp-holdout-v1-read-v2-write",
    )
    private_root = (
        tmp_path / ".fugue/private/community-skill-selected-v1/sealed-holdouts"
    )
    private_lane = private_root / lane
    private_lane.mkdir(parents=True)
    resource_root = private_lane / "resources"
    resource_root.mkdir()
    tasks = []
    for task_id in task_ids:
        resource = resource_root / f"{task_id}.tar"
        resource.write_bytes(f"resource:{task_id}".encode())
        tasks.append(
            {
                "id": task_id,
                "input": {"question": task_id},
                "partition": "holdout",
                "resources": [
                    {
                        "path": resource.relative_to(tmp_path).as_posix(),
                        "target": "/workspace/resources/fugue-source.tar",
                    }
                ],
            }
        )
    labels = [
        {
            "id": task_id,
            "expected": {},
            "base_output": "bad",
            "gold_output": "good",
        }
        for task_id in task_ids
    ]
    tasks_path = private_lane / "tasks.jsonl"
    labels_path = private_lane / "private-labels.jsonl"
    reserve_tasks_path = private_lane / "reserve-tasks.jsonl"
    reserve_labels_path = private_lane / "reserve-private-labels.jsonl"
    tasks_path.write_text("".join(json.dumps(item) + "\n" for item in tasks))
    labels_path.write_text("".join(json.dumps(item) + "\n" for item in labels))
    reserve_tasks_path.write_text("")
    reserve_labels_path.write_text("")
    public_lane = load_sealed_holdout_manifest(MANIFEST)["lanes"][0]
    selection_by_id = {item["task_id"]: item for item in public_lane["selections"]}
    items = []
    for task, label in zip(tasks, labels, strict=True):
        selection = selection_by_id[task["id"]]
        resource_path = tmp_path / task["resources"][0]["path"]
        items.append(
            {
                "task_id": task["id"],
                "role": "selected",
                "behavior_family": selection["behavior_family"],
                "paired_task_id": selection["reserve_task_id"],
                "source_task_digest": stable_digest({"source": task["id"]}),
                "private_label_digest": stable_digest(label),
                "resource_digest": sha256(resource_path.read_bytes()).hexdigest(),
                "prepared_task_digest": stable_digest(task),
            }
        )
    suite_digest = stable_digest({"items": items})
    preparation = {
        "receipt_digest": "b" * 64,
        "historical_exposure_receipt_digest": "e" * 64,
        "pool_fingerprint_digest": "f" * 64,
        "lanes": [
            {
                "lane_id": lane,
                "holdout_suite_digest": suite_digest,
                "tasks_path": tasks_path.relative_to(tmp_path).as_posix(),
                "private_labels_path": labels_path.relative_to(tmp_path).as_posix(),
                "reserve_tasks_path": reserve_tasks_path.relative_to(
                    tmp_path
                ).as_posix(),
                "reserve_private_labels_path": reserve_labels_path.relative_to(
                    tmp_path
                ).as_posix(),
                "items": items,
            }
        ],
    }
    monkeypatch.setattr(
        "holdout_support.read_sealed_holdout_preparation",
        lambda _root: preparation,
    )
    zero_results = [
        {
            "task_id": task_id,
            "gold_status": "all_dimensions_passed",
            "mutant_status": "target_dimensions_failed_only",
        }
        for task_id in task_ids
    ]
    reserve_receipt = {
        "preparation_receipt_digest": preparation["receipt_digest"],
        "task_count": 12,
        "results": [],
        "receipt_digest": "d" * 64,
    }
    (private_root / "reserve-preparation-receipt.json").write_text(
        json.dumps(reserve_receipt)
    )
    (private_root / "zero-model-receipt.json").write_text(
        json.dumps(
            {
                "preparation_receipt_digest": preparation["receipt_digest"],
                "task_count": 12,
                "reserve_preparation_receipt_digest": reserve_receipt[
                    "receipt_digest"
                ],
                "results": zero_results,
                "receipt_digest": "c" * 64,
            }
        )
    )
    audit = HoldoutExposureAuditV1(
        schema_version=1,
        kind="holdout_exposure_audit",
        study_id=raw["id"],
        holdout_suite_digest=suite_digest,
        selected_task_ids=task_ids,
        searched_project_refs=("wandb/project",),
        searched_call_count=0,
        queried_fields=tuple(sorted(_QUERY_FIELDS)),
        projection_digest="d" * 64,
        prior_evidence_digest="e" * 64,
        historical_exposure_receipt_digest="e" * 64,
        pool_fingerprint_digest="f" * 64,
        project_coverage_digest="a" * 64,
        matched_task_ids=(),
        replacements=(),
        outcome_data_consulted=False,
        status="clear",
        audited_at="2026-08-06T12:00:00+00:00",
        expires_at="2099-08-06T13:00:00+00:00",
    )
    audit_path = tmp_path / "audit.json"
    write_holdout_exposure_audit(audit_path, audit)
    decision = StudyAdvancementDecisionV1(
        schema_version=1,
        kind="study_advancement_decision",
        study_id=raw["id"],
        development_result_digest="1" * 64,
        development_qualification_digest="2" * 64,
        preview_digest="3" * 64,
        status="advance_holdout",
        repeated_improvements=("task:dimension",),
        repeated_regressions=(),
        critical_blockers=(),
        mechanism_gate="passed",
        holdout_suite_digest=suite_digest,
        holdout_exposure_audit_digest=audit.audit_digest,
        next_action="Run holdout.",
    )
    decision_path = tmp_path / "decision.json"
    write_study_advancement_decision(decision_path, decision)

    receipt = build_sealed_holdout_comparison(
        comparison_path=lane_root / "comparison.yaml",
        advancement_decision_path=decision_path,
        exposure_audit_path=audit_path,
        repo_root=tmp_path,
    )
    spec = load_comparison(tmp_path / receipt["spec_path"], repo_root=tmp_path)
    assert spec.execution.attempts == 2
    assert spec.execution.schedule["maximum_physical_executions"] == 17
    assert spec.execution.schedule["infrastructure_retry_limit"] == 1
    assert spec.execution.schedule["stages"][0]["task_ids"] == list(task_ids)
    authorization = validate_holdout_authorization(
        tmp_path / receipt["authorization_path"],
        comparison_id=spec.id,
        spec_digest=spec.spec_digest,
        tasks_path=tmp_path / spec.taskset.tasks,
        private_labels_path=tmp_path / spec.taskset.private_labels,
        attempts=spec.execution.attempts,
        repo_root=tmp_path,
    )
    assert authorization["advancement_decision_digest"] == decision.decision_digest
    assert authorization["holdout_exposure_audit_digest"] == audit.audit_digest
    assert authorization["logical_cells"] == 16
    digests, blockers = _followup_authorization_readiness(spec, repo_root=tmp_path)
    assert not blockers
    assert digests["holdout_authorization"] == authorization["receipt_digest"]
    assert digests["campaign_allocation_receipt"] == receipt["allocation_digest"]

    first_resource = tmp_path / tasks[0]["resources"][0]["path"]
    first_resource.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="selected item digests changed"):
        validate_holdout_authorization(
            tmp_path / receipt["authorization_path"],
            comparison_id=spec.id,
            spec_digest=spec.spec_digest,
            tasks_path=tmp_path / spec.taskset.tasks,
            private_labels_path=tmp_path / spec.taskset.private_labels,
            attempts=spec.execution.attempts,
            repo_root=tmp_path,
        )


def test_campaign_allocation_refuses_both_branches_and_stays_bounded(
    tmp_path: Path,
) -> None:
    lanes = (
        "superpowers-writing-plans",
        "anthropic-skill-creator",
        "vercel-react-best-practices",
    )

    def allocate(index: int, lane: str) -> dict:
        return allocate_campaign_branch(
            repo_root=tmp_path,
            lane=lane,
            branch="holdout",
            source_study_id=f"source-{index}",
            followup_study_id=f"holdout-{index}",
            followup_spec_digest=str(index + 1) * 64,
            logical_cells=16,
            receipt_path=tmp_path / f"allocation-{index}.json",
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(lambda item: allocate(*item), enumerate(lanes)))
    ledger = json.loads(
        (
            tmp_path / ".fugue/private/community-skill-selected-v1/"
            "campaign-allocation-ledger.json"
        ).read_text()
    )
    assert ledger["logical_cells"] == 96
    assert ledger["infrastructure_replacement_allowance"] == 4
    assert ledger["maximum_physical_executions"] == 100
    assert len(
        {
            ledger["lanes"][lane]["followup"]["allocation_digest"]
            for lane in lanes
        }
    ) == 3
    for index, receipt in enumerate(receipts):
        validate_campaign_allocation_receipt(
            receipt,
            comparison_id=f"holdout-{index}",
            spec_digest=str(index + 1) * 64,
            logical_cells=16,
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        allocate_campaign_branch(
            repo_root=tmp_path,
            lane="superpowers-writing-plans",
            branch="no_skill_diagnostic",
            source_study_id="source-0",
            followup_study_id="diagnostic-0",
            followup_spec_digest="f" * 64,
            logical_cells=4,
            receipt_path=tmp_path / "diagnostic-allocation.json",
        )


def test_allocation_receipt_rejects_a_stale_lane_state(tmp_path: Path) -> None:
    receipt = allocate_campaign_branch(
        repo_root=tmp_path,
        lane="superpowers-writing-plans",
        branch="holdout",
        source_study_id="source",
        followup_study_id="holdout",
        followup_spec_digest="a" * 64,
        logical_cells=16,
        receipt_path=tmp_path / "allocation.json",
    )
    ledger_path = (
        tmp_path
        / ".fugue/private/community-skill-selected-v1/"
        "campaign-allocation-ledger.json"
    )
    ledger = json.loads(ledger_path.read_text())
    lane = ledger["lanes"]["superpowers-writing-plans"]
    followup = lane["followup"]
    followup["source_study_id"] = "another-source"
    unsigned_followup = {
        key: value for key, value in followup.items() if key != "allocation_digest"
    }
    followup["allocation_digest"] = stable_digest(unsigned_followup)
    unsigned_ledger = {
        key: value for key, value in ledger.items() if key != "ledger_digest"
    }
    ledger["ledger_digest"] = stable_digest(unsigned_ledger)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ValueError, match="stale"):
        validate_campaign_allocation_receipt(
            receipt,
            comparison_id="holdout",
            spec_digest="a" * 64,
            logical_cells=16,
            repo_root=tmp_path,
        )
