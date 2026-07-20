from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fugue.bench.campaigns import (
    CampaignError,
    CampaignService,
    admission_receipt_from_dict,
    build_experiment_proposal,
    campaign_catalog_snapshot_from_dict,
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
    CellSummary,
    ExperimentRequest,
    ExportSummary,
    OperatorService,
    RunSummary,
    SetupPreparation,
)
from fugue.bench.runtime_provenance import resolve_fugue_source_provenance
from fugue.model_plane import resolve_harness_model_route, resolve_model_route
from fugue.preflight import PreflightCheck


def _campaign_repo(tmp_path: Path) -> None:
    (tmp_path / "configs/fugue/experiments").mkdir(parents=True)
    (tmp_path / "configs/fugue/context-systems").mkdir(parents=True)
    (tmp_path / "configs/fugue/campaigns").mkdir(parents=True)
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
evidence_scope: traces
require_clean_source: false
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

    def prepare(
        self,
        request: ExperimentRequest,
        *,
        experiment: Any = None,
        rebuild: bool = False,
    ) -> SetupPreparation:
        del request, experiment, rebuild
        return SetupPreparation(context=(), agent_runtimes=())

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
        del experiment
        assert run_id is not None
        if run_id in self.launched:
            raise AssertionError("campaign launched the same run twice")
        plan = self.resolve_run_plan(request, run_id=run_id)
        evaluation_lock = "e" * 64
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
            candidate_runtime[cell.candidate_id] = {
                "model_route": {
                    "provider": route.provider,
                    "model_id": route.model_id,
                },
                "model_transport": resolve_harness_model_route(route, cell.harness),
            }
            lock = {
                "execution_fingerprint": cell.execution_fingerprint,
                "candidate_id": cell.candidate_id,
                "context_runtime": None,
                "agent_runtime": {"image_id": "sha256:agent"},
                "task_runtime": {"image_id": "sha256:task"},
            }
            runtime_locks.append(
                {**lock, "configuration_sha256": stable_digest(lock)}
            )
        snapshot = {
            "schema_version": 1,
            "run_id": run_id,
            "runtime": {
                "fugue_source": resolve_fugue_source_provenance(self.repo_root)
            },
            "candidate_runtime": candidate_runtime,
            "planned_matrix": planned_matrix,
            "runtime_locks": runtime_locks,
            "evaluation_asset_lock_sha256": evaluation_lock,
            "snapshot_sha256": "",
            "lock_sha256": "",
        }
        digest = stable_digest(snapshot)
        snapshot["snapshot_sha256"] = digest
        snapshot["lock_sha256"] = digest
        run_dir = self.repo_root / ".fugue/runtime" / run_id
        atomic_write_json(run_dir / "input-lock.json", snapshot)
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
                }
            )
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


def test_catalog_and_preview_are_pure_and_hide_execution_details(tmp_path: Path) -> None:
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


def test_proposal_rejects_unregistered_and_over_limit_components(tmp_path: Path) -> None:
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
    changed = service.preview(
        _proposal(service, proposal_id="different-qualification")
    )

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
    policy_path.write_text(policy_path.read_text().replace("total_cost_usd: 100", "total_cost_usd: 1"))
    budget_plan = clean.preview(_proposal(clean))
    budget_prepared = clean.prepare(budget_plan, "prepare-over-budget")
    with pytest.raises(CampaignError) as exceeded:
        clean.admit(budget_prepared, "admit-over-budget")
    assert exceeded.value.code == "budget_exceeded"


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
    assert any("exactly one Agent root" in item for item in outcome.eligibility_failures)
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
    raw = _proposal(service).to_dict()
    raw["command"] = ["curl", "https://example.com"]
    with pytest.raises(ValueError, match="unknown experiment proposal field"):
        experiment_proposal_from_dict(raw)

    raw = _proposal(service).to_dict()
    raw["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version 1"):
        experiment_proposal_from_dict(raw)
