from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fugue.bench.campaign_lifecycle import _auxiliary_model_preflight_checks
from fugue.bench.campaigns import (
    CampaignError,
    CampaignService,
    admission_receipt_from_dict,
    build_experiment_proposal,
    campaign_catalog_snapshot_from_dict,
    campaign_error_from_dict,
    campaign_event_from_dict,
    campaign_spec_from_dict,
    campaign_status_from_dict,
    experiment_proposal_from_dict,
    get_campaign,
    outcome_packet_from_dict,
    plan_receipt_from_dict,
    prepared_plan_from_dict,
)
from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.operator import (
    AgentRuntimePreparation,
    CellSummary,
    ExperimentRequest,
    ExportSummary,
    OperatorService,
    RunSummary,
    SetupPreparation,
    TaskRuntimePreparation,
)
from fugue.bench.runtime_provenance import resolve_fugue_source_provenance
from fugue.bench.task_authoring import (
    scoring_revision_from_dict,
    task_profile_catalog_from_dict,
    task_suite_draft_from_dict,
)
from fugue.model_plane import (
    model_route_identity,
    resolve_harness_model_route,
    resolve_model_route,
)
from fugue.preflight import PreflightCheck


def _campaign_repo(tmp_path: Path) -> None:
    (tmp_path / "configs/fugue/experiments").mkdir(parents=True)
    (tmp_path / "configs/fugue/context-systems").mkdir(parents=True)
    (tmp_path / "configs/fugue/campaigns").mkdir(parents=True)
    (tmp_path / "configs/fugue/task-authoring").mkdir(parents=True)
    (tmp_path / "datasets").mkdir()
    (tmp_path / "configs/fugue/context-systems/none.yaml").write_text(
        """
id: none
title: No added context
description: Control
provider: fugue.bench.context:EmptyContextProvider
version: "1"
capabilities: [prepare, retrieve, bind, ingest, sequence, serve]
deliveries: [portable]
serve_deliveries: [portable]
license: Fugue
"""
    )
    (tmp_path / "datasets/demo.yaml").write_text(
        """
dataset: {ref: demo/tasks, version: v1}
model: openai/gpt-5
k: 1
n_concurrent: 1
jobs_dir: jobs/demo
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - id: task-one
    repository: {type: git, url: https://github.com/test/repo, commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}
"""
    )
    (tmp_path / "configs/fugue/experiments/demo.yaml").write_text(
        """
id: demo
title: Demo
manifest: datasets/demo.yaml
model: openai/gpt-5
harnesses: [codex]
variants:
  - {id: baseline, label: Baseline, context: {system_id: none, delivery: portable}}
n_attempts: 1
n_concurrent: 1
jobs_dir: jobs/demo
trace_content: full
"""
    )
    (tmp_path / "configs/fugue/campaigns/demo.yaml").write_text(
        """
schema_version: 1
id: demo
revision: v1
title: Demo campaign
objective: Exercise the governed campaign lifecycle.
allowed:
  experiments: [demo]
  models: [openai/gpt-5]
  harnesses: [codex]
  workloads: [harbor]
  variants: [baseline]
  context_systems: [none]
  analyses: []
  trace_content: [full]
stages:
  - id: qualification
    predecessors: []
    max_proposals: 1
    max_cells: 1
    require_eligible_parent: false
  - id: primary
    predecessors: [qualification]
    max_proposals: 1
    max_cells: 1
    require_eligible_parent: true
limits:
  total_cost_usd: 100
  initial_cell_reserve_usd: 2
  safety_margin: 1.5
  max_cells_per_proposal: 1
  max_total_cells: 2
  max_attempts_per_cell: 1
  max_concurrent: 1
  max_active_runs: 1
task_authoring:
  enabled_stages: [qualification]
  allowed_partitions: [qualification]
  allowed_environment_profiles: [artifact-v1]
  allowed_resource_profiles: []
  allowed_interactor_profiles: []
  allowed_judge_profiles: []
  allowed_scorer_runtimes: []
  allowed_prompt_parts: [text]
  adaptive_discovery: false
  limits:
    max_tasks: 1
    max_scenarios: 1
    max_prompt_bytes: 4096
    max_authored_asset_bytes: 4096
    max_user_turns: 1
    max_agent_turns: 1
    max_interactor_calls: 0
    max_judge_calls: 0
    scorer_timeout_sec: 10
    scorer_memory_mb: 128
    scorer_cpus: 0.5
    scorer_output_bytes: 4096
evidence_scope: traces
require_clean_source: false
"""
    )
    (tmp_path / "configs/fugue/task-authoring/profiles.yaml").write_text(
        """
schema_version: 1
environments:
  - id: artifact-v1
    title: Locked artifact workspace
    kind: artifact
    base_image: python:3.12.10-slim-bookworm
    supported_harnesses: [codex]
    capabilities: [text, artifact]
    allowed_integration_ids: []
    cpus: 1
    memory_mb: 1024
    storage_mb: 2048
resources: []
interactors: []
judges: []
scorer_runtimes: []
"""
    )
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=model-secret\n"
        "WANDB_API_KEY=trace-secret\n"
        "WANDB_ENTITY=team\n"
        "WANDB_PROJECT=fugue-experiments\n"
    )


class FakeCampaignOperator(OperatorService):
    def __init__(self, repo_root: Path) -> None:
        super().__init__(repo_root, repo_root / ".env")
        self.launched: dict[str, Any] = {}
        self.valid_evidence = True
        self.missing_input_lock = False
        self.missing_rows = False
        self.duplicate_rows = False
        self.invalid_agent_url = False
        self.route_drift = False
        self.runtime_drift = False
        self.extra_runtime_lock = False
        self.evaluation_drift = False

    def prepare(
        self,
        request: ExperimentRequest,
        *,
        experiment: Any = None,
        rebuild: bool = False,
    ) -> SetupPreparation:
        del request, experiment, rebuild
        return SetupPreparation(
            context=(),
            agent_runtimes=(
                AgentRuntimePreparation(
                    harness="codex",
                    architecture="arm64",
                    status="ready",
                    image="agent:test",
                    image_id="sha256:agent",
                    recipe_sha256="a" * 64,
                ),
            ),
            task_runtimes=(
                TaskRuntimePreparation(
                    task_id="task-one",
                    architecture="arm64",
                    status="ready",
                    image="task:test",
                    image_id="sha256:task",
                    recipe_sha256="b" * 64,
                    verification_required=True,
                    verification={"base_failed": True, "gold_passed": True},
                ),
            ),
        )

    def preflight(
        self,
        request: ExperimentRequest,
        *,
        live: bool = True,
        experiment: Any = None,
    ) -> tuple[PreflightCheck, ...]:
        del request, live, experiment
        return (PreflightCheck("synthetic control plane", True, "ready"),)

    def launch(
        self,
        request: ExperimentRequest,
        *,
        experiment: Any = None,
        run_id: str | None = None,
    ) -> RunSummary:
        assert run_id is not None
        if run_id in self.launched:
            raise AssertionError("campaign launched the same run twice")
        plan = self.resolve_run_plan(request, run_id=run_id, experiment=experiment)
        evaluation_lock_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "predictions": {},
            "lock_sha256": "",
        }
        evaluation_lock = stable_digest(evaluation_lock_payload)
        evaluation_lock_payload["lock_sha256"] = evaluation_lock
        planned_matrix = [
            {
                "cell_id": cell.id,
                "candidate_id": cell.candidate_id,
                "execution_fingerprint": cell.execution_fingerprint,
                "execution_kind": cell.execution_kind,
                "comparison_example_id": cell.comparison_example_id,
                "trial_index": cell.trial_index,
                "workload_id": cell.workload_id,
                "task_id": cell.task_id,
                "applicable": cell.applicable,
                "skip_reason": cell.skip_reason,
                "planned_prediction_count": 1,
            }
            for cell in plan.cells
        ]
        candidate_runtime: dict[str, dict[str, Any]] = {}
        runtime_locks: list[dict[str, Any]] = []
        for cell in plan.cells:
            route = resolve_model_route(cell.model, self.env)
            runtime = {
                "candidate_id": cell.candidate_id,
                "harness": cell.harness,
                "model": cell.model,
                "model_route": model_route_identity(route),
                "model_transport": resolve_harness_model_route(route, cell.harness),
            }
            candidate_runtime[cell.candidate_id] = {
                **runtime,
                "configuration_sha256": stable_digest(runtime),
            }
            if self.route_drift:
                changed_runtime = dict(candidate_runtime[cell.candidate_id])
                changed_route = dict(changed_runtime["model_route"])
                changed_route["model_id"] = "different-model"
                changed_runtime["model_route"] = changed_route
                changed_runtime["configuration_sha256"] = stable_digest(
                    {
                        key: value
                        for key, value in changed_runtime.items()
                        if key != "configuration_sha256"
                    }
                )
                candidate_runtime[cell.candidate_id] = changed_runtime
            lock = {
                "execution_fingerprint": cell.execution_fingerprint,
                "candidate_id": cell.candidate_id,
                "context_runtime": None,
                "agent_runtime": {"image_id": "sha256:agent"},
                "task_runtime": {"image_id": "sha256:task"},
            }
            runtime_locks.append({**lock, "configuration_sha256": stable_digest(lock)})
        if self.runtime_drift and runtime_locks:
            changed_lock = dict(runtime_locks[0])
            changed_lock["agent_runtime"] = {"image_id": "sha256:different"}
            changed_lock["configuration_sha256"] = stable_digest(
                {
                    key: value
                    for key, value in changed_lock.items()
                    if key != "configuration_sha256"
                }
            )
            runtime_locks[0] = changed_lock
        if self.extra_runtime_lock and runtime_locks:
            extra_lock = dict(runtime_locks[0])
            extra_lock["execution_fingerprint"] = "f" * 64
            extra_lock["configuration_sha256"] = stable_digest(
                {
                    key: value
                    for key, value in extra_lock.items()
                    if key != "configuration_sha256"
                }
            )
            runtime_locks.append(extra_lock)
        snapshot = {
            "schema_version": 1,
            "run_id": run_id,
            "runtime": {
                "fugue_source": resolve_fugue_source_provenance(self.repo_root)
            },
            "candidate_runtime": candidate_runtime,
            "planned_matrix": planned_matrix,
            "runtime_locks": runtime_locks,
            "evaluation_asset_lock_sha256": (
                "d" * 64 if self.evaluation_drift else evaluation_lock
            ),
            "snapshot_sha256": "",
            "lock_sha256": "",
        }
        digest = stable_digest(snapshot)
        snapshot["snapshot_sha256"] = digest
        snapshot["lock_sha256"] = digest
        run_dir = self.repo_root / ".fugue/runtime" / run_id
        if not self.missing_input_lock:
            atomic_write_json(run_dir / "input-lock.json", snapshot)
        atomic_write_json(run_dir / "evaluation-assets.json", evaluation_lock_payload)
        atomic_write_json(
            run_dir / "run.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "run_name": request.run_name,
                "experiment_id": request.experiment_id,
                "status": "passed",
                "observability_status": "passed",
            },
        )
        self.launched[run_id] = {
            "request": request,
            "plan": plan,
            "snapshot": snapshot,
            "evaluation_lock": evaluation_lock,
        }
        return self.run_summary(run_id)

    def run_summary(self, run_id: str) -> RunSummary:
        if run_id not in self.launched:
            raise FileNotFoundError(run_id)
        value = self.launched[run_id]
        plan = value["plan"]
        cells = tuple(
            CellSummary(
                cell_id=cell.id,
                candidate_id=cell.candidate_id,
                status="passed",
                harness=cell.harness,
                variant_id=cell.variant_id,
                context_system_id=cell.context_system_id,
                workload_id=cell.workload_id,
                task_id=cell.task_id,
                benchmark_outcome="passed",
                reward=1.0,
            )
            for cell in plan.cells
        )
        return RunSummary(
            run_id=run_id,
            run_name=str(value["request"].run_name),
            experiment_id=value["request"].experiment_id,
            status="passed",
            created_at="2026-07-20T00:00:00+00:00",
            cells=cells,
            passed=len(cells),
            failed=0,
            cancelled=0,
            interrupted=0,
            pending=0,
            not_applicable=0,
            candidates=(),
            log_path=self.repo_root / ".fugue/runtime" / run_id / "combined.log",
            observability_status="passed",
        )

    def export_run(
        self,
        run_id: str,
        *,
        out: Path | None = None,
        fetch_weave: bool = False,
        to_weave: bool = False,
        republish: bool = False,
        republish_reason: str | None = None,
    ) -> ExportSummary:
        del fetch_weave, to_weave, republish, republish_reason
        value = self.launched[run_id]
        snapshot = value["snapshot"]
        rows = []
        for index, cell in enumerate(value["plan"].cells, 1):
            root = f"root-{index}"
            conversation = f"conversation-{index}"
            rows.append(
                {
                    "schema_version": 1,
                    "prediction_schema_version": 1,
                    "record_type": "trial",
                    "prediction_id": f"prediction-{index}",
                    "run_id": run_id,
                    "candidate_id": cell.candidate_id,
                    "comparison_example_id": cell.comparison_example_id,
                    "trial_index": cell.trial_index,
                    "execution_kind": "agent",
                    "status": "passed",
                    "pass": True,
                    "reward": 1.0,
                    "workload_id": cell.workload_id,
                    "task_name": cell.task_id,
                    "harness": cell.harness,
                    "variant_id": cell.variant_id,
                    "context_system_id": cell.context_system_id,
                    "model_provider": cell.model_provider,
                    "model": cell.model,
                    "trace_link_status": "linked",
                    "root_span_id": root,
                    "weave_root_span_ids": [root] if self.valid_evidence else [],
                    "observed_conversation_id": conversation,
                    "weave_conversation_ids": [conversation],
                    "weave_trace_ids": [f"trace-{index}"],
                    "runtime_equivalence_status": "equivalent",
                    "runtime_drift": False,
                    "run_snapshot_sha256": snapshot["snapshot_sha256"],
                    "evaluation_asset_lock_sha256": value["evaluation_lock"],
                    "cost_usd": 1.0,
                    "tool_calls": 2,
                    "prompt": "must never be projected",
                    "raw_conversation": "must never be projected",
                    "command": ["curl", "https://example.invalid"],
                    "environment": {"OPENAI_API_KEY": "model-secret"},
                    "expected_evidence_paths": ["private/expected.py"],
                    "gold_paths": ["private/gold.py"],
                }
            )
            if self.invalid_agent_url:
                rows[-1]["agent_url"] = "http://user:secret@example.invalid/?token=x"
        if self.missing_rows:
            rows = rows[:-1]
        if self.duplicate_rows and rows:
            rows.append(dict(rows[0]))
        destination = out or self.repo_root / "reports" / f"{run_id}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        return ExportSummary(path=destination, rows=len(rows))


def _service(tmp_path: Path) -> CampaignService:
    _campaign_repo(tmp_path)
    operator = FakeCampaignOperator(tmp_path)
    return CampaignService(tmp_path, operator=operator)


def _proposal(
    service: CampaignService,
    *,
    proposal_id: str = "qualification-1",
    stage_id: str = "qualification",
    parent_outcome_id: str | None = None,
) -> Any:
    catalog = service.catalog("demo")
    return build_experiment_proposal(
        proposal_id=proposal_id,
        campaign_id="demo",
        catalog_digest=catalog.catalog_digest,
        stage_id=stage_id,
        research_question="Does the registered Agent configuration complete the task?",
        hypothesis="The registered baseline produces one reconciled outcome.",
        fixed_dimensions=("model", "task", "runtime"),
        varied_dimensions=("registered treatment",),
        measured_dimensions=("repair outcome", "Agent evidence"),
        experiment_id="demo",
        model="openai/gpt-5",
        n_attempts=1,
        n_concurrent=1,
        harnesses=("codex",),
        context_systems=("none",),
        variants=("baseline",),
        n_tasks=1,
        trace_content="full",
        parent_outcome_id=parent_outcome_id,
        decision_rationale=(
            "The eligible qualification outcome supports the primary stage."
            if parent_outcome_id
            else ""
        ),
    )


def test_campaign_contracts_are_strict_and_digest_verified(tmp_path: Path) -> None:
    _campaign_repo(tmp_path)
    campaign = get_campaign("demo", tmp_path)
    assert campaign.schema_version == 1
    assert len(campaign.campaign_digest) == 64

    raw = campaign.to_dict()
    raw["unexpected"] = True
    with pytest.raises(ValueError, match="unknown campaign field"):
        campaign_spec_from_dict(raw)

    service = CampaignService(tmp_path, operator=FakeCampaignOperator(tmp_path))
    proposal = _proposal(service)
    tampered = proposal.to_dict()
    tampered["model"] = "openai/other"
    with pytest.raises(ValueError, match="proposal_digest"):
        experiment_proposal_from_dict(tampered)


def test_catalog_and_preview_are_pure_and_hide_execution_details(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    assert not (tmp_path / ".fugue").exists()

    catalog = service.catalog("demo")
    proposal = _proposal(service)
    first = service.preview(proposal)
    second = service.preview(proposal)

    assert catalog.catalog_digest == service.catalog("demo").catalog_digest
    assert campaign_catalog_snapshot_from_dict(catalog.to_dict()) == catalog
    assert first == second
    assert plan_receipt_from_dict(first.to_dict()) == first
    assert first.cell_count == 1
    assert first.expected_predictions == 1
    serialized = json.dumps(first.to_dict(), sort_keys=True)
    assert "command" not in serialized
    assert "jobs_dir" not in serialized
    assert "expected_evidence_paths" not in serialized
    assert not (tmp_path / ".fugue").exists()


def test_task_authoring_catalog_registers_the_virtual_harbor_workload(
    tmp_path: Path,
) -> None:
    _campaign_repo(tmp_path)
    (tmp_path / "configs/fugue/experiments/demo.yaml").write_text(
        """
id: demo
title: Demo
model: openai/gpt-5
harnesses: [codex]
workloads:
  - id: registered-baseline
    runner: harbor
    manifest: datasets/demo.yaml
    systems: [none]
    variants: [baseline]
variants:
  - {id: baseline, label: Baseline, context: {system_id: none, delivery: portable}}
n_attempts: 1
n_concurrent: 1
jobs_dir: jobs/demo
trace_content: full
"""
    )

    catalog = CampaignService(
        tmp_path, operator=FakeCampaignOperator(tmp_path)
    ).catalog("demo")

    assert catalog.task_authoring is not None


def test_proposal_rejects_unregistered_and_over_limit_components(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    catalog = service.catalog("demo")
    proposal = build_experiment_proposal(
        proposal_id="unsafe",
        campaign_id="demo",
        catalog_digest=catalog.catalog_digest,
        stage_id="qualification",
        research_question="Can an unregistered harness run?",
        hypothesis="It should be rejected before planning.",
        fixed_dimensions=("task",),
        varied_dimensions=("harness",),
        measured_dimensions=("outcome",),
        experiment_id="demo",
        model="openai/gpt-5",
        n_attempts=1,
        n_concurrent=2,
        harnesses=("hermes",),
        context_systems=("none",),
        variants=("baseline",),
    )

    with pytest.raises(CampaignError, match="does not allow harness") as exc_info:
        service.preview(proposal)
    assert exc_info.value.code == "component_not_allowed"


def test_full_campaign_lifecycle_is_idempotent_and_reconciled(tmp_path: Path) -> None:
    service = _service(tmp_path)
    proposal = _proposal(service)
    plan = service.preview(proposal)

    prepared = service.prepare(plan, "prepare-1")
    assert prepared_plan_from_dict(prepared.to_dict()) == prepared
    assert service.prepare(plan, "prepare-1") == prepared
    admission = service.admit(prepared, "admit-1")
    assert admission_receipt_from_dict(admission.to_dict()) == admission
    assert service.admit(prepared, "admit-1") == admission
    assert admission.reserved_cost_usd == 2.0

    launched = service.launch(admission, "launch-1")
    repeated = service.launch(admission, "launch-1")
    assert launched.active_runs == repeated.active_runs
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    assert len(operator.launched) == 1
    [run_id] = operator.launched

    outcome = service.finalize(run_id, "finalize-1")
    assert outcome_packet_from_dict(outcome.to_dict()) == outcome
    assert service.finalize(run_id, "finalize-1") == outcome
    assert outcome.eligible
    assert outcome.passed == 1
    assert outcome.accounted_cost_usd == 1.0
    assert outcome.row_refs[0]["prediction_id"] == "prediction-1"
    assert outcome.evidence_refs[0]["conversation_ids"] == ["conversation-1"]
    serialized = json.dumps(outcome.to_dict(), sort_keys=True)
    assert "expected_evidence_paths" not in serialized
    assert "gold" not in serialized.lower()
    assert "model-secret" not in serialized

    status = service.status("demo")
    assert campaign_status_from_dict(status.to_dict()) == status
    assert status.state == "evidence_ready"
    assert status.accounted_cost_usd == 1.0
    assert status.reserved_cost_usd == 0.0
    events = service.events("demo")
    assert [event.sequence_number for event in events] == list(range(1, 6))
    assert campaign_event_from_dict(events[0].to_dict()) == events[0]


def test_authored_task_suite_uses_the_campaign_lifecycle_and_replays_scoring(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    catalog = service.catalog("demo")
    draft = task_suite_draft_from_dict(
        {
            "schema_version": 1,
            "id": "authored-qualification",
            "title": "Authored qualification",
            "objective": "Exercise the governed task boundary.",
            "stage_id": "qualification",
            "tasks": [
                {
                    "id": "task-one",
                    "title": "Explain the evidence",
                    "prompt": [{"type": "text", "text": "Explain the evidence."}],
                    "environment": {"profile_id": "artifact-v1"},
                    "interaction": {
                        "type": "single_turn",
                        "max_user_turns": 1,
                        "max_agent_turns": 1,
                        "timeout_sec": 300,
                    },
                    "criteria_set_id": "deterministic",
                    "tags": ["qualification"],
                    "partition": "qualification",
                }
            ],
            "scenarios": [
                {
                    "id": "evidence",
                    "title": "Evidence",
                    "tasks": [{"task_id": "task-one", "weight": 1, "must_pass": True}],
                }
            ],
            "criteria_sets": [
                {
                    "id": "deterministic",
                    "title": "Deterministic outcome",
                    "pass_threshold": 1,
                    "criteria": [
                        {
                            "id": "benchmark",
                            "description": "The benchmark passes.",
                            "evaluator": {"type": "benchmark_outcome", "config": {}},
                            "evidence": ["benchmark"],
                            "weight": 1,
                            "threshold": 1,
                            "required": True,
                        }
                    ],
                }
            ],
        }
    )
    preview = service.preview_task_suite("demo", catalog.catalog_digest, draft)
    assert preview.eligible
    lock = service.lock_task_suite(preview, "lock-authored-suite")
    assert service.lock_task_suite(preview, "lock-authored-suite") == lock

    proposal = build_experiment_proposal(
        proposal_id="authored-qualification",
        campaign_id="demo",
        catalog_digest=catalog.catalog_digest,
        stage_id="qualification",
        research_question="Can the Agent explain the evidence?",
        hypothesis="The locked task produces a reconciled result.",
        fixed_dimensions=("model", "task", "runtime"),
        varied_dimensions=("harness",),
        measured_dimensions=("benchmark", "criteria"),
        experiment_id="demo",
        model="openai/gpt-5",
        n_attempts=1,
        n_concurrent=1,
        workloads=("harbor",),
        harnesses=("codex",),
        context_systems=("none",),
        variants=("baseline",),
        n_tasks=1,
        task_suite_digest=lock.suite_digest,
    )
    plan = service.preview(proposal)
    assert plan.component_digests["task_suite"] == lock.suite_digest
    prepared = service.prepare(plan, "prepare-authored-suite")
    admission = service.admit(prepared, "admit-authored-suite")
    service.launch(admission, "launch-authored-suite")
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    [run_id] = operator.launched
    service.finalize(run_id, "finalize-authored-suite")

    revision = scoring_revision_from_dict(
        {
            "schema_version": 1,
            "id": "deterministic-v1",
            "evidence_view": "answer",
        }
    )
    evaluation = service.score_task_suite(
        run_id,
        lock.suite_digest,
        revision,
        "score-authored-suite",
    )
    assert evaluation.passed == 1
    assert (
        service.score_task_suite(
            run_id,
            lock.suite_digest,
            revision,
            "score-authored-suite",
        )
        == evaluation
    )
    analysis = service.analyze_task_study(
        run_id,
        "task-study-v1",
        "analyze-authored-suite",
        evaluation_digest=evaluation.evaluation_digest,
    )
    assert analysis.evaluation_digest == evaluation.evaluation_digest
    assert analysis.task_results[0]["criteria_passes"] == 1


def test_authored_auxiliary_model_routes_fail_preflight_without_keys() -> None:
    profiles = task_profile_catalog_from_dict(
        {
            "schema_version": 1,
            "environments": [
                {
                    "id": "artifact-v1",
                    "title": "Artifact workspace",
                    "kind": "artifact",
                    "base_image": "python:3.12.10-slim-bookworm",
                    "supported_harnesses": ["codex"],
                    "capabilities": ["text", "artifact"],
                    "cpus": 1,
                    "memory_mb": 1024,
                    "storage_mb": 2048,
                }
            ],
            "resources": [],
            "interactors": [
                {
                    "id": "interactor-v1",
                    "title": "Model interactor",
                    "kind": "model",
                    "model": "openai/gpt-5",
                    "directions": ["Ask one bounded follow-up."],
                    "supported_harnesses": ["codex"],
                    "reserve_cost_usd": 0.5,
                }
            ],
            "judges": [
                {
                    "id": "judge-v1",
                    "title": "Blind judge",
                    "model": "wandb/zai-org/GLM-5.2",
                    "prompt": "Judge only the supplied evidence.",
                    "evidence": ["answer"],
                    "blind_fields": [
                        "harness",
                        "model",
                        "variant_id",
                        "context_system_id",
                        "candidate_id",
                        "treatment",
                    ],
                    "reserve_cost_usd": 0.5,
                }
            ],
            "scorer_runtimes": [],
        },
        source_sha256="a" * 64,
    )
    interactor = profiles.interactor("interactor-v1")
    judge = profiles.judge("judge-v1")
    components = {
        "interactor:interactor-v1": interactor.profile_digest,
        "judge:judge-v1": judge.profile_digest,
    }

    missing = _auxiliary_model_preflight_checks(components, profiles, {})
    assert [(item.name, item.ok) for item in missing] == [
        ("task interactor model", False),
        ("task judge model", False),
    ]
    assert "OPENAI_API_KEY" in missing[0].detail
    assert "WANDB_API_KEY" in missing[1].detail

    ready = _auxiliary_model_preflight_checks(
        components,
        profiles,
        {"OPENAI_API_KEY": "present", "WANDB_API_KEY": "present"},
    )
    assert all(item.ok for item in ready)


def test_every_campaign_artifact_rejects_unknown_fields_and_versions(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    campaign = get_campaign("demo", tmp_path)
    catalog = service.catalog("demo")
    proposal = _proposal(service)
    plan = service.preview(proposal)
    prepared = service.prepare(plan, "prepare-strict-artifacts")
    admission = service.admit(prepared, "admit-strict-artifacts")
    status = service.launch(admission, "launch-strict-artifacts")
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    [run_id] = operator.launched
    outcome = service.finalize(run_id, "finalize-strict-artifacts")
    event = service.events("demo")[0]

    artifacts = (
        (campaign_spec_from_dict, campaign),
        (campaign_catalog_snapshot_from_dict, catalog),
        (experiment_proposal_from_dict, proposal),
        (plan_receipt_from_dict, plan),
        (prepared_plan_from_dict, prepared),
        (admission_receipt_from_dict, admission),
        (outcome_packet_from_dict, outcome),
        (campaign_event_from_dict, event),
        (campaign_status_from_dict, status),
    )
    for parser, artifact in artifacts:
        unknown = artifact.to_dict()
        unknown["unexpected"] = True
        with pytest.raises(ValueError, match="unknown"):
            parser(unknown)

        unsupported = artifact.to_dict()
        unsupported["schema_version"] = 2
        with pytest.raises(ValueError, match="schema_version 1"):
            parser(unsupported)


def test_concurrent_duplicate_launch_creates_one_run(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.preview(_proposal(service))
    prepared = service.prepare(plan, "prepare-concurrent-launch")
    admission = service.admit(prepared, "admit-concurrent-launch")

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = tuple(
            executor.map(
                lambda _: service.launch(admission, "launch-concurrent"), range(2)
            )
        )

    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    assert len(operator.launched) == 1
    assert statuses[0].runs == statuses[1].runs


def test_stage_progression_requires_eligible_parent_and_records_rationale(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    qualification = service.preview(_proposal(service))
    prepared = service.prepare(qualification, "prepare-qualification")
    admission = service.admit(prepared, "admit-qualification")
    service.launch(admission, "launch-qualification")
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    [run_id] = operator.launched
    outcome = service.finalize(run_id, "finalize-qualification")

    primary = _proposal(
        service,
        proposal_id="primary-1",
        stage_id="primary",
        parent_outcome_id=outcome.outcome_id,
    )
    primary_plan = service.preview(primary)
    primary_prepared = service.prepare(primary_plan, "prepare-primary")
    primary_admission = service.admit(primary_prepared, "admit-primary")

    assert primary_admission.parent_outcome_id == outcome.outcome_id
    assert primary_admission.reserved_cell_cost_usd == 2.0


def test_operation_id_conflicts_fail_closed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first = service.preview(_proposal(service))
    service.prepare(first, "shared-operation")
    changed = service.preview(_proposal(service, proposal_id="different-qualification"))

    with pytest.raises(CampaignError) as exc_info:
        service.prepare(changed, "shared-operation")
    assert exc_info.value.code == "operation_conflict"


def test_budget_admission_and_policy_drift_fail_closed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.preview(_proposal(service))
    prepared = service.prepare(plan, "prepare-budget")
    policy_path = tmp_path / "configs/fugue/campaigns/demo.yaml"
    policy_path.write_text(
        policy_path.read_text().replace(
            "Exercise the governed campaign lifecycle.",
            "Changed after preparation.",
        )
    )

    with pytest.raises(CampaignError) as drift:
        service.admit(prepared, "admit-drifted-policy")
    assert drift.value.code == "policy_drift"

    clean = _service(tmp_path / "budget")
    policy_path = tmp_path / "budget/configs/fugue/campaigns/demo.yaml"
    policy_path.write_text(
        policy_path.read_text().replace("total_cost_usd: 100", "total_cost_usd: 1")
    )
    budget_plan = clean.preview(_proposal(clean))
    budget_prepared = clean.prepare(budget_plan, "prepare-over-budget")
    with pytest.raises(CampaignError) as exceeded:
        clean.admit(budget_prepared, "admit-over-budget")
    assert exceeded.value.code == "budget_exceeded"


def test_policy_revision_is_allowed_only_before_first_admission(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first_plan = service.preview(_proposal(service))
    first_prepared = service.prepare(first_plan, "prepare-first-policy")
    policy_path = tmp_path / "configs/fugue/campaigns/demo.yaml"
    policy_path.write_text(
        policy_path.read_text().replace(
            "Exercise the governed campaign lifecycle.",
            "Exercise a revised governed campaign lifecycle.",
        )
    )

    revised_plan = service.preview(_proposal(service, proposal_id="revised-policy"))
    revised_prepared = service.prepare(revised_plan, "prepare-revised-policy")
    with pytest.raises(CampaignError) as stale:
        service.admit(first_prepared, "admit-stale-policy")
    assert stale.value.code == "policy_drift"

    service.admit(revised_prepared, "admit-revised-policy")
    policy_path.write_text(
        policy_path.read_text().replace(
            "Exercise a revised governed campaign lifecycle.",
            "Attempt a second campaign revision.",
        )
    )
    post_admission = service.preview(
        _proposal(service, proposal_id="post-admission-policy")
    )
    with pytest.raises(CampaignError) as immutable:
        service.prepare(post_admission, "prepare-post-admission-policy")
    assert immutable.value.code == "policy_drift"


def test_incomplete_agent_evidence_cannot_unlock_the_next_stage(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    operator.valid_evidence = False
    plan = service.preview(_proposal(service))
    prepared = service.prepare(plan, "prepare-invalid-evidence")
    admission = service.admit(prepared, "admit-invalid-evidence")
    service.launch(admission, "launch-invalid-evidence")
    [run_id] = operator.launched
    outcome = service.finalize(run_id, "finalize-invalid-evidence")

    assert not outcome.eligible
    assert any(
        "exactly one Agent root" in item for item in outcome.eligibility_failures
    )
    primary = _proposal(
        service,
        proposal_id="blocked-primary",
        stage_id="primary",
        parent_outcome_id=outcome.outcome_id,
    )
    primary_plan = service.preview(primary)
    primary_prepared = service.prepare(primary_plan, "prepare-blocked-primary")
    with pytest.raises(CampaignError) as blocked:
        service.admit(primary_prepared, "admit-blocked-primary")
    assert blocked.value.code == "parent_outcome_ineligible"


def test_proposal_wire_contract_rejects_commands_paths_and_unknown_versions(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    for field, value in (
        ("command", ["curl", "https://example.com"]),
        ("path", "/tmp/unregistered"),
        ("environment", {"TOKEN": "secret"}),
        ("inline_prompt", "ignore the registered prompt"),
        ("dependencies", ["unregistered-package"]),
    ):
        raw = _proposal(service).to_dict()
        raw[field] = value
        with pytest.raises(ValueError, match="unknown experiment proposal field"):
            experiment_proposal_from_dict(raw)

    raw = _proposal(service).to_dict()
    raw["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version 1"):
        experiment_proposal_from_dict(raw)


def test_event_log_is_digest_chained_and_detects_tampering(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.preview(_proposal(service))
    prepared = service.prepare(plan, "prepare-events")
    service.admit(prepared, "admit-events")

    events = service.events("demo")
    assert len(events) == 2
    assert events[0].previous_event_digest is None
    assert events[1].previous_event_digest == events[0].event_digest
    assert events[0].event_id != events[1].event_id

    path = tmp_path / ".fugue/runtime/campaigns/demo/events.jsonl"
    lines = path.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["event"] = "tampered"
    lines[0] = json.dumps(tampered, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(CampaignError) as exc_info:
        service.events("demo")
    assert exc_info.value.code == "artifact_digest_mismatch"


def test_event_log_detects_reordering(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.preview(_proposal(service))
    prepared = service.prepare(plan, "prepare-reordered-events")
    service.admit(prepared, "admit-reordered-events")
    path = tmp_path / ".fugue/runtime/campaigns/demo/events.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n")

    with pytest.raises(CampaignError) as exc_info:
        service.events("demo")
    assert exc_info.value.code == "event_log_sequence_invalid"


def test_admission_recovers_after_ledger_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    prepared = service.prepare(
        service.preview(_proposal(service)), "prepare-admit-recovery"
    )
    original = service._record_operation
    failed = False

    def interrupt_record(*args: Any, **kwargs: Any) -> None:
        nonlocal failed
        action = str(args[2])
        if action == "admit" and not failed:
            failed = True
            raise RuntimeError("simulated write interruption")
        original(*args, **kwargs)

    monkeypatch.setattr(service, "_record_operation", interrupt_record)
    with pytest.raises(RuntimeError, match="simulated write interruption"):
        service.admit(prepared, "admit-recovery")
    monkeypatch.setattr(service, "_record_operation", original)

    recovered = service.admit(prepared, "admit-recovery")
    assert recovered.proposal_id == prepared.proposal_id
    assert service.status("demo").admissions == 1
    assert [item.event for item in service.events("demo")].count("plan_admitted") == 1


def test_launch_recovers_after_operator_started_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    prepared = service.prepare(
        service.preview(_proposal(service)), "prepare-launch-recovery"
    )
    admission = service.admit(prepared, "admit-launch-recovery")
    original = service._write_operation
    failed = False

    def interrupt_completed_launch(
        campaign_id: str,
        operation_id: str,
        value: dict[str, Any],
    ) -> None:
        nonlocal failed
        if (
            value.get("action") == "launch"
            and value.get("status") == "completed"
            and not failed
        ):
            failed = True
            raise RuntimeError("simulated launch journal interruption")
        original(campaign_id, operation_id, value)

    monkeypatch.setattr(service, "_write_operation", interrupt_completed_launch)
    with pytest.raises(RuntimeError, match="simulated launch journal interruption"):
        service.launch(admission, "launch-recovery")
    monkeypatch.setattr(service, "_write_operation", original)

    recovered = service.launch(admission, "launch-recovery")
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    assert len(operator.launched) == 1
    assert recovered.runs[0]["status"] == "passed"
    assert [item.event for item in service.events("demo")].count("run_started") == 1


def test_concurrent_distinct_finalizations_converge(tmp_path: Path) -> None:
    service = _service(tmp_path)
    prepared = service.prepare(
        service.preview(_proposal(service)), "prepare-finalize-convergence"
    )
    admission = service.admit(prepared, "admit-finalize-convergence")
    service.launch(admission, "launch-finalize-convergence")
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    [run_id] = operator.launched

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            executor.map(
                lambda operation_id: service.finalize(run_id, operation_id),
                ("finalize-a", "finalize-b"),
            )
        )

    assert outcomes[0] == outcomes[1]
    outcome_files = list(
        (tmp_path / ".fugue/runtime/campaigns/demo/outcomes").glob("*.json")
    )
    assert len(outcome_files) == 1


def test_finalization_recovers_after_outcome_and_ledger_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    prepared = service.prepare(
        service.preview(_proposal(service)), "prepare-finalize-recovery"
    )
    admission = service.admit(prepared, "admit-finalize-recovery")
    service.launch(admission, "launch-finalize-recovery")
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    [run_id] = operator.launched
    original = service._record_operation
    interrupted = False

    def interrupt_finalize(*args: Any, **kwargs: Any) -> None:
        nonlocal interrupted
        if str(args[2]) == "finalize" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated finalization journal interruption")
        original(*args, **kwargs)

    monkeypatch.setattr(service, "_record_operation", interrupt_finalize)
    with pytest.raises(RuntimeError, match="finalization journal interruption"):
        service.finalize(run_id, "finalize-recovery")
    monkeypatch.setattr(service, "_record_operation", original)

    recovered = service.finalize(run_id, "finalize-recovery")
    assert recovered.run_id == run_id
    assert recovered.eligible
    assert [item.event for item in service.events("demo")].count(
        "evidence_finalized"
    ) == 1


def test_cancellation_recovers_without_repeating_supervisor_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    prepared = service.prepare(
        service.preview(_proposal(service)), "prepare-cancel-recovery"
    )
    admission = service.admit(prepared, "admit-cancel-recovery")
    service.launch(admission, "launch-cancel-recovery")
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    [run_id] = operator.launched

    calls = 0

    def cancel_once(selected_run_id: str) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        assert selected_run_id == run_id
        return SimpleNamespace(status="cancelled")

    monkeypatch.setattr(operator.supervisor, "cancel", cancel_once)
    original = service._write_ledger
    interrupted = False

    def interrupt_ledger(campaign_id: str, ledger: dict[str, Any]) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated cancellation ledger interruption")
        original(campaign_id, ledger)

    monkeypatch.setattr(service, "_write_ledger", interrupt_ledger)
    with pytest.raises(RuntimeError, match="cancellation ledger interruption"):
        service.cancel(run_id, "cancel-recovery", "operator request")
    monkeypatch.setattr(service, "_write_ledger", original)

    service.cancel(run_id, "cancel-recovery", "operator request")
    assert calls == 1
    ledger = json.loads(
        (tmp_path / ".fugue/runtime/campaigns/demo/ledger.json").read_text()
    )
    assert ledger["admissions"][0]["status"] == "cancelled_unreconciled"
    assert [item.event for item in service.events("demo")].count("run_cancelled") == 1


@pytest.mark.parametrize(
    ("flag", "failure"),
    (
        ("missing_input_lock", "run input lock is missing"),
        ("route_drift", "model-route receipts"),
        ("runtime_drift", "exact runtime locks"),
        ("extra_runtime_lock", "exact runtime locks"),
        ("evaluation_drift", "exact evaluation asset lock"),
        ("invalid_agent_url", "invalid Agent link"),
        ("missing_rows", "observed 0 prediction rows"),
        ("duplicate_rows", "duplicates prediction identity"),
    ),
)
def test_invalid_or_partial_evidence_is_preserved_but_ineligible(
    tmp_path: Path, flag: str, failure: str
) -> None:
    root = tmp_path / flag
    service = _service(root)
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    setattr(operator, flag, True)
    prepared = service.prepare(service.preview(_proposal(service)), f"prepare-{flag}")
    admission = service.admit(prepared, f"admit-{flag}")
    service.launch(admission, f"launch-{flag}")
    [run_id] = operator.launched
    outcome = service.finalize(run_id, f"finalize-{flag}")

    assert not outcome.eligible
    assert any(failure in item for item in outcome.eligibility_failures)
    assert outcome.accounted_cost_usd >= 0
    assert service.status("demo").reserved_cost_usd == admission.reserved_cost_usd


def test_public_outcome_projection_excludes_privileged_content(tmp_path: Path) -> None:
    service = _service(tmp_path)
    prepared = service.prepare(
        service.preview(_proposal(service)), "prepare-safe-projection"
    )
    admission = service.admit(prepared, "admit-safe-projection")
    service.launch(admission, "launch-safe-projection")
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    [run_id] = operator.launched
    outcome = service.finalize(run_id, "finalize-safe-projection")

    serialized = json.dumps(outcome.to_dict(), sort_keys=True)
    for forbidden in (
        "must never be projected",
        "model-secret",
        "trace-secret",
        "private/expected.py",
        "private/gold.py",
        "raw_conversation",
        '"command"',
        '"environment"',
    ):
        assert forbidden not in serialized


def test_campaign_errors_have_a_strict_sanitized_wire_contract() -> None:
    error = CampaignError(
        "stable_failure",
        "a safe failure",
        category="evidence",
        retryable=True,
        details={"exception_type": "ValueError"},
    )
    assert campaign_error_from_dict(error.to_dict()).to_dict() == error.to_dict()

    unsafe = error.to_dict()
    unsafe["safe_to_repeat"] = "yes"
    unsafe["error_digest"] = stable_digest({**unsafe, "error_digest": ""})
    with pytest.raises(ValueError, match="must be a boolean"):
        campaign_error_from_dict(unsafe)
