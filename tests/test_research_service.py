from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fugue.research.client import FugueResearchClient
from fugue.research.contracts import ResearchError, build_experiment_draft
from fugue.research.service import ResearchService
from fugue.research.store import StudyStore

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


@dataclass
class Artifact:
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class FakeCampaignService:
    def __init__(self) -> None:
        self.launches = 0
        self.status_checks = 0
        self.plan = Artifact(
            {
                "schema_version": 1,
                "campaign_id": "campaign-1",
                "proposal_id": "proposal-1",
                "cell_count": 2,
                "plan_digest": _A,
            }
        )

    def catalog(self, _: str) -> Any:
        return SimpleNamespace(
            catalog_digest=_A,
            policy_digest=_B,
            experiments=(
                {
                    "id": "kimi-harness-baseline",
                    "harnesses": ["codex", "claude-code"],
                    "variants": [
                        {
                            "id": "base",
                            "context_system_id": "none",
                            "enabled": True,
                        }
                    ],
                    "workloads": [],
                    "presets": [],
                },
            ),
            models=({"id": "kimi-k2.7-code"},),
            harnesses=("codex", "claude-code"),
            context_systems=({"id": "none"},),
            analyses=(),
        )

    def preview_task_suite(self, _: str, __: str, draft: Any) -> Artifact:
        return Artifact(
            {
                "schema_version": 1,
                "campaign_id": "campaign-1",
                "catalog_digest": _A,
                "policy_digest": _B,
                "draft": draft.to_dict(),
                "task_count": 1,
                "scenario_count": 1,
                "prompt_bytes": 20,
                "authored_asset_bytes": 0,
                "estimated_calls": {"agent": 2},
                "capability_matrix": [
                    {
                        "task_id": "task-one",
                        "harness": "codex",
                        "applicable": True,
                    },
                    {
                        "task_id": "task-one",
                        "harness": "claude-code",
                        "applicable": True,
                    },
                ],
                "component_digests": {},
                "eligible": True,
                "failures": (),
                "preview_digest": _C,
            }
        )

    def preview(self, _: Any) -> Artifact:
        return self.plan

    def validate_proposal(self, _: Any) -> None:
        return None

    def prepare(self, _: Any, operation_id: str) -> Artifact:
        return Artifact(
            {
                "schema_version": 1,
                "operation_id": operation_id,
                "prepared_plan_digest": _B,
            }
        )

    def admit(self, _: Any, operation_id: str) -> Artifact:
        return Artifact(
            {
                "schema_version": 1,
                "operation_id": operation_id,
                "admission_digest": _C,
            }
        )

    def launch(self, _: Any, operation_id: str) -> Artifact:
        self.launches += 1
        return Artifact(
            {
                "operation_id": operation_id,
                "runs": [
                    {
                        "proposal_id": "proposal-1",
                        "run_id": "run-1",
                        "status": "running",
                    }
                ],
            }
        )

    def status(self, _: str) -> Artifact:
        self.status_checks += 1
        return Artifact(
            {
                "runs": [
                    {
                        "proposal_id": "proposal-1",
                        "run_id": "run-1",
                        "status": "passed",
                    }
                ]
            }
        )

    def finalize(self, _: str, __: str) -> Artifact:
        return Artifact(
            {
                "outcome_id": "outcome-1",
                "outcome_digest": _D,
                "run_snapshot_sha256": _A,
                "eligible": True,
            }
        )


def _service(tmp_path: Path) -> tuple[ResearchService, FakeCampaignService]:
    fake = FakeCampaignService()
    service = ResearchService(
        tmp_path,
        campaign_service=fake,  # type: ignore[arg-type]
        store=StudyStore(tmp_path),
    )
    service.store.create_study(
        study_id="study-1",
        title="Loop components",
        campaign_id="campaign-1",
        question="Which components matter?",
        operation_id="create-study",
    )
    return service, fake


def _draft() -> Any:
    return build_experiment_draft(
        study_id="study-1",
        campaign_id="campaign-1",
        proposal_id="proposal-1",
        stage_id="discovery",
        question="Does the harness matter?",
        hypothesis="Harnesses may resolve different tasks.",
        fixed_dimensions=["model", "tasks"],
        varied_dimensions=["harness"],
        measured_dimensions=["task resolution"],
        experiment_id="kimi-harness-baseline",
        model="kimi-k2.7-code",
        n_attempts=1,
        n_concurrent=1,
        harnesses=["codex", "claude-code"],
    )


def _task_suite() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "suite-one",
        "title": "Qualification suite",
        "objective": "Measure a grounded answer.",
        "stage_id": "discovery",
        "tasks": [
            {
                "id": "task-one",
                "title": "Explain the contract",
                "prompt": [{"type": "text", "text": "Explain the contract."}],
                "environment": {"profile_id": "artifact-v1"},
                "interaction": {
                    "type": "single_turn",
                    "max_user_turns": 1,
                    "max_agent_turns": 1,
                    "timeout_sec": 300,
                },
                "criteria_set_id": "grounded",
                "tags": ["explanation"],
                "partition": "discovery",
            }
        ],
        "scenarios": [
            {
                "id": "explanation",
                "title": "Explanation",
                "tasks": [{"task_id": "task-one", "weight": 1, "must_pass": True}],
            }
        ],
        "criteria_sets": [
            {
                "id": "grounded",
                "title": "Grounded",
                "pass_threshold": 1,
                "criteria": [
                    {
                        "id": "benchmark",
                        "description": "Verifier passes.",
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


def test_preview_is_pure_and_start_is_explicit_boundary(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    before = service.store.get_study("study-1")
    preview = service.preview_experiment("study-1", _draft())
    after = service.store.get_study("study-1")
    assert preview.estimated_cells == 2
    assert before == after
    assert service.store.list_experiments("study-1") == ()
    record = service.start_experiment(preview, idempotency_key="start-1")
    assert record.state == "queued"


def test_inline_task_preview_counts_selected_coordinates_without_locking(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)
    values = _draft().to_dict()
    values.pop("draft_digest")
    values["task_suite_draft"] = _task_suite()
    values["n_attempts"] = 2
    preview = service.preview_experiment(
        "study-1",
        build_experiment_draft(
            **{k: v for k, v in values.items() if k != "schema_version"}
        ),
    )
    assert preview.estimated_cells == 4
    assert preview.estimated_calls == {"agent": 4}
    assert not (tmp_path / ".fugue/runtime").exists()


def test_inline_task_preview_budgets_only_selected_harnesses(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    values = _draft().to_dict()
    values.pop("draft_digest")
    values["task_suite_draft"] = _task_suite()
    values["harnesses"] = ["codex"]
    preview = service.preview_experiment(
        "study-1",
        build_experiment_draft(
            **{k: v for k, v in values.items() if k != "schema_version"}
        ),
    )
    assert preview.estimated_cells == 1
    assert preview.estimated_calls == {"agent": 1}


def test_worker_completes_canonical_lifecycle_without_duplicate_launch(
    tmp_path: Path,
) -> None:
    service, fake = _service(tmp_path)
    preview = service.preview_experiment("study-1", _draft())
    service.start_experiment(preview, idempotency_key="start-1")
    first = service.run_once("worker-1")
    assert first and first.state == "running"
    final = service.run_once("worker-1")
    assert final and final.state == "completed"
    assert fake.launches == 1
    assert final.outcome and final.outcome["outcome_digest"] == _D
    study = service.store.get_study("study-1")
    assert study.experiments[-1].run_id == "run-1"
    assert {item.kind for item in study.run_refs} == {"run", "outcome"}


def test_python_client_preserves_same_artifacts(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    client = FugueResearchClient(service)
    study = client.studies.get("study-1")
    preview = study.experiments.preview(_draft())
    handle = study.experiments.start(preview, idempotency_key="start-client")
    service.run_once("worker-client")
    service.run_once("worker-client")
    result = handle.result()
    assert result["outcome"]["outcome_digest"] == _D
    updated = study.record(
        "Experiment completed.",
        runs=result["run_refs"],
        expected_revision=study.revision,
        idempotency_key="record-client",
    )
    assert updated.notes[-1].text == "Experiment completed."


def test_outer_loop_resumes_records_result_and_previews_child(tmp_path: Path) -> None:
    service, campaign = _service(tmp_path)
    client = FugueResearchClient(service)
    study = client.studies.get("study-1")
    assert study.context().brief["question"] == "Which components matter?"

    preview = study.experiments.preview(_draft())
    started = study.experiments.start(preview, idempotency_key="start-lineage")
    running = service.run_once("worker-before-restart")
    assert running and running.state == "running"
    cursor = started.events()[-1].sequence

    restarted = ResearchService(
        tmp_path,
        campaign_service=campaign,  # type: ignore[arg-type]
        store=StudyStore(tmp_path),
    )
    reconnected = FugueResearchClient(restarted)
    handle = reconnected.experiment(started.id)
    assert handle.events(after=cursor) == ()
    completed = restarted.run_once("worker-after-restart")
    assert completed and completed.state == "completed"
    assert campaign.launches == 1

    outcome = handle.result()
    source = {
        "kind": "outcome",
        "ref": outcome["outcome"]["outcome_id"],
        "digest": outcome["outcome"]["outcome_digest"],
    }
    resumed_study = reconnected.studies.get("study-1")
    updated = resumed_study.record(
        "The admitted experiment completed after reconnecting.",
        runs=outcome["run_refs"],
        results=[
            {
                "id": "result-reconnected",
                "statement": "The locked comparison completed with reconciled evidence.",
                "kind": "qualification",
                "outcome": "completed",
                "population": "two planned cells",
                "conditions": {"attempts": 1},
                "sample_size": 2,
                "aggregation": "planned cells",
                "exclusions": ["not a harness ranking"],
                "sources": [source],
            }
        ],
        expected_revision=resumed_study.revision,
        idempotency_key="record-reconnected-result",
    )
    assert updated.results[-1].sources[0].digest == _D

    child_values = _draft().to_dict()
    child_values.pop("draft_digest")
    child_values["proposal_id"] = "proposal-child"
    child_values["question"] = "Does the observed branch replicate?"
    child_values["parent_experiment_ids"] = [started.id]
    child = resumed_study.experiments.preview(
        build_experiment_draft(
            **{
                key: value
                for key, value in child_values.items()
                if key != "schema_version"
            }
        )
    )
    assert child.draft["parent_experiment_ids"] == [started.id]


def test_prelaunch_cancellation_is_idempotent_and_input_bound(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    preview = service.preview_experiment("study-1", _draft())
    record = service.start_experiment(preview, idempotency_key="start-cancel")
    cancelled = service.cancel_experiment(
        record.id, idempotency_key="cancel-1", reason="operator request"
    )
    repeated = service.cancel_experiment(
        record.id, idempotency_key="cancel-1", reason="operator request"
    )
    assert cancelled == repeated
    assert cancelled.state == "cancelled"
    with pytest.raises(ResearchError, match="different input"):
        service.cancel_experiment(
            record.id, idempotency_key="cancel-1", reason="changed reason"
        )
