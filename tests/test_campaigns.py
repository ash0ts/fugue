from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fugue.bench.campaign_lifecycle as campaign_lifecycle
import fugue.bench.loop_failure as loop_failure
import fugue.bench.run_conformance as run_conformance
from fugue.bench.ai import _intervention_lock_inputs
from fugue.bench.campaign_evidence import _runtime_locks_valid
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
from fugue.bench.export import PublishedEvaluation
from fugue.bench.files import atomic_write_json
from fugue.bench.intervention_provenance import (
    build_intervention_component_lock,
    write_intervention_component_lock,
)
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
from fugue.bench.scoring import (
    build_intervention_selection_lock,
    write_intervention_selection_lock,
)
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


def _selection_campaign_repo(tmp_path: Path) -> None:
    _campaign_repo(tmp_path)
    (tmp_path / "configs/fugue/experiments/demo.yaml").write_text(
        """
id: demo
title: Selection-locked campaign
manifest: datasets/demo.yaml
model: openai/gpt-5
harnesses: [codex]
workloads:
  - id: holdout
    runner: harbor
    manifest: datasets/demo.yaml
    systems: [none]
    variants: [production, candidate]
default_preset: discovery
presets:
  discovery:
    workloads: [holdout]
    n_tasks: 1
    n_attempts: 1
  holdout:
    workloads: [holdout]
    n_tasks: 1
    n_attempts: 1
    selection_lock_required: true
    selection_lock_kind: intervention
variants:
  - {id: production, label: Production, context: {system_id: none, delivery: portable}}
  - {id: candidate, label: Candidate, context: {system_id: none, delivery: portable}}
n_attempts: 1
n_concurrent: 1
jobs_dir: jobs/demo
trace_content: full
"""
    )
    campaign_path = tmp_path / "configs/fugue/campaigns/demo.yaml"
    campaign_path.write_text(
        """
schema_version: 1
id: demo
revision: v1
title: Selection-locked campaign
objective: Freeze holdout tasks before discovery and confirm one selected intervention.
allowed:
  experiments: [demo]
  models: [openai/gpt-5]
  harnesses: [codex]
  workloads: [holdout]
  variants: [production, candidate]
  context_systems: [none]
  analyses: []
  trace_content: [full]
stages:
  - id: discovery
    predecessors: []
    max_proposals: 1
    max_cells: 2
    require_eligible_parent: false
  - id: holdout
    predecessors: [discovery]
    max_proposals: 1
    max_cells: 2
    require_eligible_parent: false
limits:
  total_cost_usd: 100
  initial_cell_reserve_usd: 2
  safety_margin: 1.5
  max_cells_per_proposal: 2
  max_total_cells: 4
  max_attempts_per_cell: 1
  max_concurrent: 1
  max_active_runs: 1
task_authoring:
  enabled_stages: [discovery, holdout]
  allowed_partitions: [discovery, holdout]
  allowed_environment_profiles: [artifact-v1]
  allowed_resource_profiles: []
  allowed_interactor_profiles: []
  allowed_judge_profiles: []
  allowed_scorer_runtimes: []
  allowed_prompt_parts: [text]
  adaptive_discovery: true
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
    (tmp_path / "source-marker.txt").write_text("locked source\n")
    (tmp_path / ".gitignore").write_text(".fugue/\njobs/\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fugue Tests",
            "-c",
            "user.email=fugue-tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
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
        self.snapshot_identity_drift = False
        self.strip_harbor_markers = False
        self.evaluation_drift = False
        self.bridge_ready = True
        self.bridge_starts = 0

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
        if not self.bridge_ready:
            return (PreflightCheck("bridge health", False, "bridge is unavailable"),)
        return (PreflightCheck("synthetic control plane", True, "ready"),)

    def start_bridge(
        self,
        request: ExperimentRequest,
        *,
        experiment: Any = None,
    ) -> None:
        del request, experiment
        self.bridge_starts += 1
        self.bridge_ready = True

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
                "attempt_id": cell.attempt_id,
                "attempt_identity": cell.attempt_identity,
                "candidate_id": cell.candidate_id,
                "execution_fingerprint": cell.execution_fingerprint,
                "execution_kind": cell.execution_kind,
                "comparison_example_id": cell.comparison_example_id,
                "trial_index": cell.trial_index,
                "workload_id": cell.workload_id,
                "workload_runner": "harbor",
                "task_id": cell.task_id,
                "applicable": cell.applicable,
                "skip_reason": cell.skip_reason,
                "planned_prediction_count": 1,
            }
            for cell in plan.cells
        ]
        if self.snapshot_identity_drift and planned_matrix:
            planned_matrix[0]["attempt_id"] = "0" * 64
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
            evaluations=(
                PublishedEvaluation(
                    candidate_id=cells[0].candidate_id,
                    name="Demo evaluation",
                    examples=1,
                    url="https://wandb.ai/team/fugue-experiments/r/call/eval-1",
                    evaluation_ref="weave:///team/fugue-experiments/object/eval:v1",
                    dataset_ref=(
                        "weave:///team/fugue-experiments/object/dataset:v1"
                    ),
                    model_ref="weave:///team/fugue-experiments/object/model:v1",
                    agent_predictions=1,
                    linked_agent_predictions=1,
                    direct_predictions=0,
                    publication_id="a" * 64,
                ),
            ),
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
            root = f"{index:016x}"
            conversation = f"conversation-{index}"
            trace_id = f"{index:032x}"
            project = "team/fugue-experiments"
            app = "https://wandb.ai/team/fugue-experiments"
            evaluation_call = f"evaluation-{index}"
            attempt_call = f"attempt-{index}"
            prediction_call = f"prediction-call-{index}"
            agent_call = f"agent-{index}"

            def call_ref(call_id: str) -> str:
                return f"weave:///team/fugue-experiments/call/{call_id}"

            def call_url(call_id: str, base: str = app) -> str:
                return f"{base}/weave/calls/{call_id}"

            rows.append(
                {
                    "schema_version": 1,
                    "prediction_schema_version": 1,
                    "record_type": "trial",
                    "prediction_id": f"prediction-{index}",
                    "attempt_id": cell.attempt_id,
                    "attempt_identity": cell.attempt_identity,
                    "run_id": run_id,
                    "candidate_id": cell.candidate_id,
                    "execution_fingerprint": cell.execution_fingerprint,
                    "comparison_example_id": cell.comparison_example_id,
                    "task_id": cell.task_id,
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
                    "trace_project": project,
                    "result_evidence_project": project,
                    "trace_receipt": {
                        "project_slug": project,
                        "app_base_url": "https://wandb.ai",
                    },
                    "trace_link_status": "linked",
                    "root_span_id": root,
                    "otel_root_span_ids": [root],
                    "otel_trace_ids": [trace_id],
                    "observed_conversation_id": conversation,
                    "weave_conversation_ids": [conversation],
                    "conversation_correlation_verified": self.valid_evidence,
                    "weave_evaluation_root_call_id": evaluation_call,
                    "weave_evaluation_root_ref": call_ref(evaluation_call),
                    "weave_evaluation_root_url": call_url(evaluation_call),
                    "evaluation_root_object_verified": self.valid_evidence,
                    "eval_predict_and_score_call_id": attempt_call,
                    "eval_predict_and_score_ref": call_ref(attempt_call),
                    "eval_predict_and_score_url": call_url(attempt_call),
                    "eval_predict_and_score_object_verified": self.valid_evidence,
                    "weave_prediction_call_id": prediction_call,
                    "weave_prediction_ref": call_ref(prediction_call),
                    "weave_prediction_url": call_url(prediction_call),
                    "weave_prediction_object_verified": self.valid_evidence,
                    "weave_agent_root_call_id": agent_call,
                    "weave_agent_root_ref": call_ref(agent_call),
                    "weave_agent_root_url": call_url(agent_call),
                    "agent_graph_verified": self.valid_evidence,
                    "weave_dataset_ref": (
                        "weave:///team/fugue-experiments/object/dataset:v1"
                    ),
                    "weave_dataset_url": (
                        f"{app}/weave/objects/dataset/versions/v1"
                    ),
                    "dataset_version_object_verified": self.valid_evidence,
                    "evaluation_prediction_graph_verified": self.valid_evidence,
                    "runtime_equivalence_status": "equivalent",
                    "runtime_drift": False,
                    "harbor_environment": "local_harbor_docker",
                    "harbor_conformance_status": "passed",
                    "harbor_policy_attestation_verified": True,
                    "privacy_contract_version": 2,
                    "local_artifact_privacy_scan_status": "passed",
                    "local_artifact_privacy_match_count": 0,
                    "hosted_evidence_privacy_scan_status": "passed",
                    "hosted_evidence_privacy_match_count": 0,
                    "private_label_boundary_verified": True,
                    "sandbox_cleanup_verified": True,
                    "orphaned_sandbox": False,
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
            if self.strip_harbor_markers:
                for field in (
                    "harbor_environment",
                    "harbor_conformance_status",
                    "harbor_conformance_receipt_digest",
                    "harbor_policy_attestation_verified",
                    "privacy_contract_version",
                    "local_artifact_privacy_scan_status",
                    "local_artifact_privacy_scan_digest",
                    "local_artifact_privacy_match_count",
                    "hosted_evidence_privacy_scan_status",
                    "hosted_evidence_privacy_scan_digest",
                    "hosted_evidence_privacy_match_count",
                    "private_label_boundary_verified",
                    "sandbox_cleanup_verified",
                    "orphaned_sandbox",
                ):
                    rows[-1].pop(field, None)
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
    service = CampaignService(tmp_path, operator=operator)

    real_privacy_check = service._apply_hosted_privacy_receipt

    def apply_hosted_privacy_receipt(
        *,
        proposal: Any,
        plan: Any,
        run_id: str,
        rows: list[dict[str, Any]],
    ) -> None:
        # The fake operator's ordinary rows already contain exact V2 evidence.
        # Authored-suite tests and direct privacy tests still exercise the real
        # hosted receipt boundary.
        if run_id in operator.launched and proposal.task_suite_digest is None:
            return
        real_privacy_check(
            proposal=proposal,
            plan=plan,
            run_id=run_id,
            rows=rows,
        )

    service._apply_hosted_privacy_receipt = apply_hosted_privacy_receipt  # type: ignore[method-assign]
    return service


def _proposal(
    service: CampaignService,
    *,
    proposal_id: str = "qualification-1",
    stage_id: str = "qualification",
    n_attempts: int = 1,
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
        n_attempts=n_attempts,
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


def test_campaign_proposal_binds_and_propagates_intervention_prefreeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    catalog = service.catalog("demo")
    inputs = {
        "failure_lock_sha256": "a" * 64,
        "discovery_suite_sha256": "b" * 64,
        "holdout_suite_sha256": "c" * 64,
        "failure_locked_at": "2026-07-30T10:00:00Z",
        "suites_frozen_at": "2026-07-30T10:05:00Z",
    }
    proposal = build_experiment_proposal(
        proposal_id="loop-discovery",
        campaign_id="demo",
        catalog_digest=catalog.catalog_digest,
        stage_id="qualification",
        research_question="Which locked intervention repairs the failure?",
        hypothesis="One predeclared arm improves the deterministic outcome.",
        fixed_dimensions=("model", "task", "runtime"),
        varied_dimensions=("skill", "mcp"),
        measured_dimensions=("task outcome", "mechanism use"),
        experiment_id="demo",
        model="openai/gpt-5",
        n_attempts=1,
        n_concurrent=1,
        task_suite_digest="b" * 64,
        intervention_lock_inputs=inputs,
        failure_lock=".fugue/failures/repeated-failure.json",
        intervention_component_locks=(
            ".fugue/interventions/skill.lock.json",
        ),
    )

    parsed = experiment_proposal_from_dict(proposal.to_dict())
    assert parsed.intervention_lock_inputs == {
        **inputs,
        "failure_locked_at": "2026-07-30T10:00:00+00:00",
        "suites_frozen_at": "2026-07-30T10:05:00+00:00",
    }
    assert parsed.failure_lock == ".fugue/failures/repeated-failure.json"
    monkeypatch.setattr(
        campaign_lifecycle,
        "read_task_suite_lock",
        lambda *_args, **_kwargs: SimpleNamespace(task_count=2),
    )
    request = service._request(parsed)
    rows = [
        {
            "tags": [
                *request.tags,
                "task-suite:" + parsed.task_suite_digest,
            ]
        }
    ]
    assert _intervention_lock_inputs(rows) == parsed.intervention_lock_inputs
    assert (
        "intervention-component-lock:"
        ".fugue/interventions/skill.lock.json"
    ) in request.tags

    with pytest.raises(ValueError, match="must match the governed phase lock"):
        build_experiment_proposal(
            proposal_id="loop-drifted-suite",
            campaign_id="demo",
            catalog_digest=catalog.catalog_digest,
            stage_id="qualification",
            research_question="Can a drifted suite be used?",
            hypothesis="It must fail before planning.",
            fixed_dimensions=("model",),
            varied_dimensions=("skill",),
            measured_dimensions=("task outcome",),
            experiment_id="demo",
            model="openai/gpt-5",
            n_attempts=1,
            n_concurrent=1,
            task_suite_digest="d" * 64,
            intervention_lock_inputs=inputs,
        )


def test_campaign_hosted_privacy_stays_unavailable_without_private_truth(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    proposal = _proposal(service)
    rows = [
        {
            "run_id": "run-without-authored-suite",
            "workload_id": "harbor",
            "execution_kind": "agent",
            "privacy_contract_version": 2,
            "local_artifact_privacy_scan_status": "passed",
            "hosted_evidence_privacy_scan_status": "passed",
        }
    ]

    service._apply_hosted_privacy_receipt(
        proposal=proposal,
        plan=service.preview(proposal),
        run_id="run-without-authored-suite",
        rows=rows,
    )

    receipt = run_conformance.read_hosted_evidence_privacy_receipt(
        repo_root=tmp_path,
        run_id="run-without-authored-suite",
    )
    assert receipt["status"] == "unavailable"
    assert rows[0]["local_artifact_privacy_scan_status"] == "unavailable"
    assert rows[0]["hosted_evidence_privacy_scan_status"] == "unavailable"


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


def test_campaign_preview_accepts_complete_expanded_agent_attempts(
    tmp_path: Path,
) -> None:
    _campaign_repo(tmp_path)
    campaign_path = tmp_path / "configs/fugue/campaigns/demo.yaml"
    campaign_path.write_text(
        campaign_path.read_text()
        .replace("max_cells: 1", "max_cells: 2")
        .replace("max_cells_per_proposal: 1", "max_cells_per_proposal: 2")
        .replace("max_total_cells: 2", "max_total_cells: 4")
        .replace("max_attempts_per_cell: 1", "max_attempts_per_cell: 2")
    )
    service = CampaignService(tmp_path, operator=FakeCampaignOperator(tmp_path))

    preview = service.preview(_proposal(service, n_attempts=2))

    assert preview.cell_count == 2
    assert {cell["trial_index"] for cell in preview.cells} == {1, 2}


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


def test_retrieval_to_action_agent_example_resolves_exact_serial_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).parents[1]
    source = campaign_lifecycle.resolve_fugue_source_provenance(repo_root)
    monkeypatch.setattr(
        campaign_lifecycle,
        "resolve_fugue_source_provenance",
        lambda _: {**source, "dirty": False},
    )
    service = CampaignService(repo_root)
    catalog = service.catalog("retrieval-to-action-v1")
    experiment = next(
        item for item in catalog.experiments if item["id"] == "retrieval-to-action"
    )
    assert {item["id"]: item["label"] for item in experiment["variants"]} == {
        "baseline": "Standard instructions, no repository search",
        "memory-only": "Repository search only",
        "policy-only": "Inspect and verify only",
        "memory-policy": "Repository search plus inspect and verify",
    }
    proposal = build_experiment_proposal(
        proposal_id="retrieval-to-action-qualification-001",
        campaign_id="retrieval-to-action-v1",
        catalog_digest=catalog.catalog_digest,
        stage_id="qualification",
        research_question="Does repository search actually help?",
        hypothesis="Search may help when Agents inspect and verify results.",
        fixed_dimensions=("model", "task", "prompt base", "runtime", "attempt"),
        varied_dimensions=("repository search", "inspect and verify", "harness"),
        measured_dimensions=("repair pass", "evidence use", "latency", "cost"),
        experiment_id="retrieval-to-action",
        model="wandb/zai-org/GLM-5.2",
        n_attempts=1,
        n_concurrent=1,
        workloads=("canary",),
        harnesses=("codex", "claude-code"),
        context_systems=("none", "rag-dense"),
        variants=("baseline", "memory-only", "policy-only", "memory-policy"),
        n_tasks=1,
        trace_content="full",
    )

    plan = service.preview(proposal)

    assert plan.cell_count == 8
    assert plan.applicable_cells == 8
    assert plan.expected_predictions == 8
    assert plan.max_concurrent == 1
    assert {(item["harness"], item["variant_id"]) for item in plan.cells} == {
        (harness, variant)
        for harness in ("codex", "claude-code")
        for variant in ("baseline", "memory-only", "policy-only", "memory-policy")
    }


def test_full_campaign_lifecycle_is_idempotent_and_reconciled(tmp_path: Path) -> None:
    service = _service(tmp_path)
    proposal = _proposal(service)
    plan = service.preview(proposal)
    assert plan.cells[0]["attempt_id"]
    assert plan.cells[0]["attempt_identity"] == {
        "task_id": plan.cells[0]["task_id"],
        "arm": plan.cells[0]["variant_id"],
        "harness": plan.cells[0]["harness"],
        "attempt": plan.cells[0]["trial_index"],
        "candidate": plan.cells[0]["candidate_id"],
        "runtime": plan.cells[0]["execution_fingerprint"],
    }

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
    assert len(outcome.row_refs[0]["attempt_id"]) == 64
    assert outcome.row_refs[0]["attempt_identity"]["task_id"] == "task-one"
    assert outcome.row_refs[0]["execution_fingerprint"] == plan.cells[0][
        "execution_fingerprint"
    ]
    assert [
        link["slot"] for link in outcome.row_refs[0]["evidence_links"]
    ] == [
        "prediction_and_score",
        "prediction",
        "evaluation_root",
        "agent_root",
        "dataset",
    ]
    assert outcome.evidence_refs[0]["conversation_ids"] == ["conversation-1"]
    assert outcome.evidence_refs[0]["attempt_id"] == outcome.row_refs[0][
        "attempt_id"
    ]
    assert outcome.evidence_refs[0]["otel_root_span_ids"] == ["0000000000000001"]
    assert all(
        link["call_id"] != "0000000000000001"
        for link in outcome.evidence_refs[0]["links"]
        if link.get("call_id")
    )
    assert outcome.evaluation_runs[0]["evaluation_ref"].endswith("/object/eval:v1")
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


def test_campaign_preparation_bootstraps_a_required_bridge_once(tmp_path: Path) -> None:
    service = _service(tmp_path)
    operator = service.operator
    assert isinstance(operator, FakeCampaignOperator)
    operator.bridge_ready = False
    plan = service.preview(_proposal(service))

    prepared = service.prepare(plan, "prepare-bridge")

    assert operator.bridge_starts == 1
    assert all(item["ok"] for item in prepared.preflight)
    assert service.prepare(plan, "prepare-bridge") == prepared
    assert operator.bridge_starts == 1


def test_authored_task_suite_uses_the_campaign_lifecycle_and_replays_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        run_conformance,
        "_fetch_hosted_evidence_snapshot",
        lambda **_kwargs: run_conformance._HostedEvidenceSnapshot(
            status="passed",
            payloads=(),
            required_call_count=4,
            observed_required_call_count=4,
            descendant_call_count=1,
            agent_span_count=1,
            required_agent_conversation_count=1,
            observed_agent_conversation_count=1,
            required_dataset_count=1,
            observed_dataset_count=1,
            query_error_count=0,
        ),
    )
    outcome = service.finalize(run_id, "finalize-authored-suite")
    hosted_privacy = run_conformance.read_hosted_evidence_privacy_receipt(
        repo_root=tmp_path,
        run_id=run_id,
    )
    [exported_row] = [
        json.loads(line)
        for line in (tmp_path / outcome.export_path).read_text().splitlines()
        if line.strip()
    ]
    assert hosted_privacy["status"] == "passed"
    assert hosted_privacy["private_label_record_count"] == 1
    assert exported_row["hosted_evidence_privacy_scan_status"] == "passed"
    assert (
        exported_row["hosted_evidence_privacy_scan_digest"]
        == hosted_privacy["receipt_sha256"]
    )

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


def test_authored_holdout_preserves_preset_and_binds_intervention_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _selection_campaign_repo(tmp_path)
    service = CampaignService(
        tmp_path,
        tmp_path / ".env",
        operator=FakeCampaignOperator(tmp_path),
    )
    catalog = service.catalog("demo")
    discovery = service.operator.rendered_jobs(
        ExperimentRequest(experiment_id="demo", preset="discovery"),
        run_id="discovery",
        write_configs=False,
    )
    candidate_by_variant = {
        job.variant_id: job.candidate_id for job in discovery
    }
    def suite_draft(
        phase: str,
        *,
        suffix: str = "",
        stage: str | None = None,
        partition: str | None = None,
    ) -> Any:
        identity = f"{phase}{suffix}"
        return task_suite_draft_from_dict(
            {
                "schema_version": 1,
                "id": f"locked-{identity}",
                "title": f"Locked {identity}",
                "objective": f"Exercise the independently frozen {phase} task.",
                "stage_id": stage or phase,
                "tasks": [
                    {
                        "id": f"{identity}-one",
                        "title": "Verify the evidence",
                        "prompt": [
                            {"type": "text", "text": "Verify the supplied evidence."}
                        ],
                        "environment": {"profile_id": "artifact-v1"},
                        "interaction": {
                            "type": "single_turn",
                            "max_user_turns": 1,
                            "max_agent_turns": 1,
                            "timeout_sec": 300,
                        },
                        "criteria_set_id": "deterministic",
                        "tags": [phase],
                        "partition": partition or phase,
                    }
                ],
                "scenarios": [
                    {
                        "id": identity,
                        "title": phase.title(),
                        "tasks": [
                            {
                                "task_id": f"{identity}-one",
                                "weight": 1,
                                "must_pass": True,
                            }
                        ],
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
                                "evaluator": {
                                    "type": "benchmark_outcome",
                                    "config": {},
                                },
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

    draft = suite_draft("holdout")
    task_preview = service.preview_task_suite(
        "demo", catalog.catalog_digest, draft
    )
    task_lock = service.lock_task_suite(task_preview, "lock-holdout")
    discovery_preview = service.preview_task_suite(
        "demo",
        catalog.catalog_digest,
        suite_draft("discovery"),
    )
    discovery_lock = service.lock_task_suite(discovery_preview, "lock-discovery")
    discovery_event = next(
        event
        for event in service.events("demo")
        if event.event == "task_suite_locked"
        and event.artifact_digest == discovery_lock.suite_digest
    )
    suites_frozen_at = discovery_event.recorded_at
    intervention_inputs = {
        "failure_lock_sha256": "f" * 64,
        "discovery_suite_sha256": discovery_lock.suite_digest,
        "holdout_suite_sha256": task_lock.suite_digest,
        "failure_locked_at": "2020-07-30T10:00:00Z",
        "suites_frozen_at": suites_frozen_at,
    }
    failure_path = tmp_path / ".fugue/failures/repeated-failure.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text('{"reviewed":true}\n', encoding="utf-8")
    def read_failure_lock(path: Path) -> dict[str, str]:
        if path != failure_path:
            raise FileNotFoundError(path)
        return {
            "lock_sha256": "f" * 64,
            "locked_at": "2020-07-30T10:00:00+00:00",
        }

    monkeypatch.setattr(
        loop_failure,
        "read_comparison_failure_lock",
        read_failure_lock,
    )
    relative_failure_path = failure_path.relative_to(tmp_path)
    selected_component = build_intervention_component_lock(
        kind="skill",
        component_id="bounded-evidence",
        lock_digest="3" * 64,
        repository="https://github.com/wandb/fugue",
        source_commit="1" * 40,
        source_tree="2" * 40,
    )
    component_path = write_intervention_component_lock(
        tmp_path / ".fugue/interventions/bounded-evidence.json",
        selected_component,
    ).relative_to(tmp_path)

    def discovery_proposal_for(
        proposal_id: str,
        discovery_digest: str,
        holdout_digest: str,
        frozen_at: str,
    ) -> Any:
        return build_experiment_proposal(
            proposal_id=proposal_id,
            campaign_id="demo",
            catalog_digest=catalog.catalog_digest,
            stage_id="discovery",
            research_question="Which intervention repairs the failure?",
            hypothesis="The candidate improves the deterministic outcome.",
            fixed_dimensions=("model", "task", "runtime"),
            varied_dimensions=("variant",),
            measured_dimensions=("benchmark",),
            experiment_id="demo",
            preset_id="discovery",
            model="openai/gpt-5",
            n_attempts=1,
            n_tasks=1,
            n_concurrent=1,
            task_suite_digest=discovery_digest,
            intervention_lock_inputs={
                **intervention_inputs,
                "discovery_suite_sha256": discovery_digest,
                "holdout_suite_sha256": holdout_digest,
                "suites_frozen_at": frozen_at,
            },
            failure_lock=relative_failure_path,
            intervention_component_locks=(component_path,),
        )

    discovery_proposal = discovery_proposal_for(
        "locked-discovery",
        discovery_lock.suite_digest,
        task_lock.suite_digest,
        suites_frozen_at,
    )
    discovery_plan = service.preview(discovery_proposal)
    assert discovery_plan.cell_count == 2
    component_prefix = "intervention_component:skill:bounded-evidence"
    assert (
        discovery_plan.component_digests[f"{component_prefix}:canonical"]
        == selected_component.component_digest
    )
    assert len(discovery_plan.component_digests[f"{component_prefix}:file"]) == 64
    assert (
        discovery_plan.component_digests["intervention_failure_lock:canonical"]
        == "f" * 64
    )
    assert len(
        discovery_plan.component_digests["intervention_failure_lock:file"]
    ) == 64
    monkeypatch.setattr(
        loop_failure,
        "read_comparison_failure_lock",
        lambda _path: {
            "lock_sha256": "f" * 64,
            "locked_at": "2020-07-30T10:01:00+00:00",
        },
    )
    with pytest.raises(CampaignError, match="time differs"):
        service.preview(discovery_proposal)
    monkeypatch.setattr(
        loop_failure,
        "read_comparison_failure_lock",
        read_failure_lock,
    )
    original_component = (tmp_path / component_path).read_bytes()
    (tmp_path / component_path).write_bytes(original_component + b"\n")
    with pytest.raises(CampaignError, match="current resolved plan differs"):
        service.prepare(discovery_plan, "prepare-drifted-component-file")
    (tmp_path / component_path).write_bytes(original_component)
    with pytest.raises(CampaignError, match="canonical discovery lock event"):
        service.preview(
            discovery_proposal_for(
                "drifted-freeze-time",
                discovery_lock.suite_digest,
                task_lock.suite_digest,
                "2026-07-30T10:05:00Z",
            )
        )
    with pytest.raises(CampaignError, match="real TaskSuiteLockV1"):
        service.preview(
            discovery_proposal_for(
                "fabricated-holdout",
                discovery_lock.suite_digest,
                "e" * 64,
                suites_frozen_at,
            )
        )

    wrong_partition_preview = service.preview_task_suite(
        "demo",
        catalog.catalog_digest,
        suite_draft(
            "discovery",
            suffix="-wrong-partition",
            partition="holdout",
        ),
    )
    wrong_partition = service.lock_task_suite(
        wrong_partition_preview,
        "lock-wrong-partition",
    )
    wrong_partition_event = next(
        event
        for event in service.events("demo")
        if event.artifact_digest == wrong_partition.suite_digest
    )
    with pytest.raises(CampaignError, match="only the 'discovery' partition"):
        service.preview(
            discovery_proposal_for(
                "wrong-discovery-partition",
                wrong_partition.suite_digest,
                task_lock.suite_digest,
                wrong_partition_event.recorded_at,
            )
        )

    reversed_discovery_preview = service.preview_task_suite(
        "demo",
        catalog.catalog_digest,
        suite_draft("discovery", suffix="-reversed"),
    )
    reversed_discovery = service.lock_task_suite(
        reversed_discovery_preview,
        "lock-reversed-discovery",
    )
    reversed_holdout_preview = service.preview_task_suite(
        "demo",
        catalog.catalog_digest,
        suite_draft("holdout", suffix="-reversed"),
    )
    reversed_holdout = service.lock_task_suite(
        reversed_holdout_preview,
        "lock-reversed-holdout",
    )
    reversed_holdout_event = next(
        event
        for event in service.events("demo")
        if event.artifact_digest == reversed_holdout.suite_digest
    )
    with pytest.raises(CampaignError, match="frozen before the discovery"):
        service.preview(
            discovery_proposal_for(
                "late-holdout",
                reversed_discovery.suite_digest,
                reversed_holdout.suite_digest,
                reversed_holdout_event.recorded_at,
            )
        )

    source = resolve_fugue_source_provenance(tmp_path)
    selection = build_intervention_selection_lock(
        experiment_id="demo",
        source_commit=str(source["commit"]),
        source_tree=str(source["tree"]),
        source_dirty_digest="",
        failure_lock_sha256="f" * 64,
        discovery_suite_sha256=discovery_lock.suite_digest,
        holdout_suite_sha256=task_lock.suite_digest,
        analysis_snapshot_sha256="a" * 64,
        discovery_run_snapshot_sha256s=("b" * 64,),
        comparison_example_ids=("c" * 64,),
        discovery_variant_ids=("production", "candidate"),
        baseline_variant_id="production",
        selected_variant_id="candidate",
        selected_components=(
            selected_component,
        ),
        rankings=tuple(
            {
                "variant_id": variant_id,
                "candidate_digest": candidate_by_variant[variant_id],
            }
            for variant_id in ("production", "candidate")
        ),
        decision="recommend",
        rationale="candidate passed the preregistered discovery gate",
        failure_locked_at="2020-07-30T10:00:00Z",
        suites_frozen_at=suites_frozen_at,
        discovery_completed_at="2099-07-30T11:00:00Z",
        selection_locked_at="2099-07-30T11:05:00Z",
    )
    selection_path = write_intervention_selection_lock(
        tmp_path / ".fugue/selection.json", selection
    )

    proposal_args = {
        "proposal_id": "locked-holdout",
        "campaign_id": "demo",
        "catalog_digest": catalog.catalog_digest,
        "stage_id": "holdout",
        "research_question": "Does the frozen candidate hold on new tasks?",
        "hypothesis": "The candidate preserves the discovery improvement.",
        "fixed_dimensions": ("model", "task", "runtime"),
        "varied_dimensions": ("variant",),
        "measured_dimensions": ("benchmark",),
        "experiment_id": "demo",
        "preset_id": "holdout",
        "model": "openai/gpt-5",
        "n_attempts": 1,
        "n_tasks": 1,
        "n_concurrent": 1,
        "task_suite_digest": task_lock.suite_digest,
        "intervention_lock_inputs": intervention_inputs,
        "failure_lock": relative_failure_path,
    }
    without_lock = build_experiment_proposal(**proposal_args)
    with pytest.raises(CampaignError, match="requires an immutable selection lock"):
        service.preview(without_lock)

    proposal = build_experiment_proposal(
        **proposal_args,
        selection_lock=".fugue/selection.json",
        intervention_component_locks=(component_path,),
    )
    plan = service.preview(proposal)
    assert plan.cell_count == 2
    assert {cell["variant_id"] for cell in plan.cells} == {
        "production",
        "candidate",
    }
    assert {cell["workload_id"] for cell in plan.cells} == {"holdout"}
    assert plan.request["preset"] == "holdout"
    assert plan.request["selection_lock"] == ".fugue/selection.json"
    assert len(plan.component_digests["selection_lock"]) == 64

    alternate_failure_path = (
        tmp_path / ".fugue/failures/different-repeated-failure.json"
    )
    alternate_failure_path.write_text('{"reviewed":true}\n', encoding="utf-8")

    def read_alternate_failure_lock(path: Path) -> dict[str, str]:
        if path == alternate_failure_path:
            return {
                "lock_sha256": "e" * 64,
                "locked_at": "2020-07-30T10:00:00+00:00",
            }
        return read_failure_lock(path)

    monkeypatch.setattr(
        loop_failure,
        "read_comparison_failure_lock",
        read_alternate_failure_lock,
    )
    wrong_failure_proposal = build_experiment_proposal(
        **{
            **proposal_args,
            "proposal_id": "wrong-failure-holdout",
            "intervention_lock_inputs": {
                **intervention_inputs,
                "failure_lock_sha256": "e" * 64,
            },
            "failure_lock": alternate_failure_path.relative_to(tmp_path),
        },
        selection_lock=".fugue/selection.json",
        intervention_component_locks=(component_path,),
    )
    with pytest.raises(CampaignError, match="different locked failure"):
        service.preview(wrong_failure_proposal)
    monkeypatch.setattr(
        loop_failure,
        "read_comparison_failure_lock",
        read_failure_lock,
    )

    source_marker = tmp_path / "source-marker.txt"
    original_source_marker = source_marker.read_text()
    source_marker.write_text(original_source_marker + "drift\n")
    with pytest.raises(CampaignError, match="catalog digest"):
        service.preview(proposal)
    source_marker.write_text(original_source_marker)

    selection_path.write_text(selection_path.read_text() + "\n")
    with pytest.raises(CampaignError, match="current resolved plan differs"):
        service.prepare(plan, "prepare-drifted-selection")


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
    assert "FUGUE_WANDB_INFERENCE_API_KEY" in missing[1].detail

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


def test_operator_cost_limit_fails_before_admission_is_recorded(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.preview(_proposal(service))
    prepared = service.prepare(plan, "prepare-approval-limit")

    with pytest.raises(CampaignError) as rejected:
        service.admit(
            prepared,
            "admit-approval-limit",
            maximum_cost_usd=1.0,
        )

    assert rejected.value.code == "approval_cost_limit"
    assert service.status("demo").reserved_cost_usd == 0.0
    assert not (
        tmp_path / ".fugue/runtime/campaigns/demo/operations/admit-approval-limit.json"
    ).exists()


def test_reservation_estimate_is_pure_and_includes_paid_calls(tmp_path: Path) -> None:
    service = _service(tmp_path)

    estimate = service.estimate_reservation(
        "demo",
        cell_count=2,
        additional_paid_calls=3,
    )

    assert estimate == 10.0
    assert not (tmp_path / ".fugue").exists()


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
        "verified Weave link" in item for item in outcome.eligibility_failures
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
        ("snapshot_identity_drift", "snapshot coordinates"),
        ("evaluation_drift", "exact evaluation asset lock"),
        ("invalid_agent_url", "invalid Agent link"),
        ("missing_rows", "observed 0 prediction rows"),
        ("duplicate_rows", "duplicates prediction identity"),
        ("strip_harbor_markers", "does not identify the exact local Harbor backend"),
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


def test_runtime_lock_validation_scopes_context_runtime_to_context_cells() -> None:
    plan = SimpleNamespace(
        cells=(
            {
                "applicable": True,
                "candidate_id": "candidate-control",
                "context_system_id": "none",
                "execution_fingerprint": "control-fingerprint",
                "execution_kind": "agent",
                "workload_id": "maintenance-canary",
                "workload_runner": "harbor",
            },
            {
                "applicable": True,
                "candidate_id": "candidate-search",
                "context_system_id": "rag-dense",
                "execution_fingerprint": "search-fingerprint",
                "execution_kind": "agent",
                "workload_id": "maintenance-canary",
                "workload_runner": "harbor",
            },
        )
    )
    prepared = SimpleNamespace(
        preparation={
            "agent_runtimes": [{"image_id": "sha256:agent"}],
            "task_runtimes": [{"image_id": "sha256:task"}],
            "portable_context_runtime": {"image_id": "sha256:context"},
        }
    )

    def lock(
        fingerprint: str,
        candidate_id: str,
        *,
        context_runtime: dict[str, str] | None,
    ) -> dict[str, Any]:
        value = {
            "execution_fingerprint": fingerprint,
            "candidate_id": candidate_id,
            "agent_runtime": {"image_id": "sha256:agent"},
            "task_runtime": {"image_id": "sha256:task"},
            "context_runtime": context_runtime,
        }
        return {**value, "configuration_sha256": stable_digest(value)}

    snapshot = {
        "runtime_locks": [
            lock(
                "control-fingerprint",
                "candidate-control",
                context_runtime=None,
            ),
            lock(
                "search-fingerprint",
                "candidate-search",
                context_runtime={"image_id": "sha256:context"},
            ),
        ]
    }

    assert _runtime_locks_valid(snapshot, plan, prepared)

    missing_task_runtime = SimpleNamespace(
        preparation={
            **prepared.preparation,
            "task_runtimes": [],
        }
    )
    assert not _runtime_locks_valid(snapshot, plan, missing_task_runtime)

    invalid = json.loads(json.dumps(snapshot))
    invalid["runtime_locks"][0] = lock(
        "control-fingerprint",
        "candidate-control",
        context_runtime={"image_id": "sha256:context"},
    )
    assert not _runtime_locks_valid(invalid, plan, prepared)


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
