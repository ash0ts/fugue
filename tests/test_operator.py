from __future__ import annotations

import json
from pathlib import Path

from fugue.bench.execution import write_run_manifest
from fugue.bench.operator import ExperimentRequest, OperatorService, as_json


def make_operator_repo(tmp_path: Path) -> OperatorService:
    (tmp_path / "configs/fugue/experiments").mkdir(parents=True)
    (tmp_path / "configs/fugue/context-systems").mkdir(parents=True)
    (tmp_path / "configs/fugue/prompts").mkdir(parents=True)
    (tmp_path / "configs/fugue/skills/demo-skill").mkdir(parents=True)
    (tmp_path / "datasets").mkdir()
    (tmp_path / "configs/fugue/context-systems/none.yaml").write_text(
        """
id: none
title: No added context
description: Control
provider: fugue.bench.context:EmptyContextProvider
version: "1"
capabilities: [prepare, retrieve, bind, ingest, sequence, serve]
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
  - {id: task-one, repo: test/repo, base_commit: abc123}
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
  - {id: baseline, label: Baseline, context: {system_id: none}}
n_attempts: 1
n_concurrent: 1
jobs_dir: jobs/demo
trace_content: full
"""
    )
    (tmp_path / "configs/fugue/prompts/demo-prompt.md").write_text(
        "# Demo prompt\n\nInspect the repository before editing.\n"
    )
    (tmp_path / "configs/fugue/skills/demo-skill/SKILL.md").write_text(
        "# Demo skill\n\nUse focused repository search.\n"
    )
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=model-secret\n"
        "WANDB_API_KEY=trace-secret\n"
        "WANDB_ENTITY=team\n"
        "WANDB_PROJECT=fugue-experiments\n"
    )
    return OperatorService(tmp_path, tmp_path / ".env")


def test_operator_status_masks_secrets_and_links_to_agents(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    status = service.status(ExperimentRequest(experiment_id="demo"))
    payload = as_json(status)

    assert status.model_key_present is True
    assert status.trace_key_present is True
    assert status.links.agents == (
        "https://wandb.ai/team/fugue-experiments/weave/agents"
    )
    assert "model-secret" not in payload
    assert "trace-secret" not in payload
    assert "catalog_records" not in payload


def test_operator_preview_is_side_effect_free(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    preview = service.preview(ExperimentRequest(experiment_id="demo"))

    assert preview.cells == 1
    assert preview.estimated_trials == 1
    assert preview.harnesses == ("codex",)
    assert len(preview.matrix_cells) == 1
    assert preview.matrix_cells[0].task_id == "task-one"
    assert preview.matrix_cells[0].trial_count == 1
    assert not (tmp_path / ".fugue").exists()


def test_request_for_experiment_keeps_inherited_scale_out_of_overrides(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    experiment = service.experiment("demo")

    request = service.request_for_experiment(experiment)

    assert request.harnesses == ("codex",)
    assert request.variants == ("baseline",)
    assert request.n_attempts is None
    assert request.n_tasks is None
    assert request.n_concurrent is None


def test_start_bridge_loads_the_requested_experiment(
    tmp_path: Path, monkeypatch
) -> None:
    service = make_operator_repo(tmp_path)
    captured: dict[str, object] = {}

    def fake_bridge_up(target, **kwargs):
        captured.update({"target": target, **kwargs})
        return object()

    monkeypatch.setattr("fugue.bench.operator.bridge_up", fake_bridge_up)

    service.start_bridge(ExperimentRequest(experiment_id="demo"))

    assert captured["target"] == "openai/gpt-5"
    assert captured["builder_model"] is None
    assert captured["judge_model"] is None


def test_ephemeral_experiment_launch_persists_runtime_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    service = make_operator_repo(tmp_path)

    def start_detached(**values):
        write_run_manifest(
            tmp_path,
            values["run_id"],
            {
                "status": "starting",
                "run_name": values["run_name"],
                "experiment_id": values["experiment_id"],
                "combined_log": str(
                    tmp_path
                    / ".fugue/runtime"
                    / values["run_id"]
                    / "combined.log"
                ),
            },
        )
        return service.supervisor.get(values["run_id"], recover=False)

    monkeypatch.setattr(service.supervisor, "start_detached", start_detached)
    experiment = service.experiment("demo")
    run = service.launch(
        ExperimentRequest(experiment_id="demo"),
        experiment=experiment,
    )

    snapshot = tmp_path / ".fugue/runtime" / run.run_id / "experiment.yaml"
    assert snapshot.is_file()
    assert "id: demo" in snapshot.read_text()


def test_run_links_use_the_project_recorded_at_launch(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    write_run_manifest(
        tmp_path,
        "run-original-project",
        {
            "status": "passed",
            "run_name": "Original project",
            "experiment_id": "demo",
            "trace_project": "other-team/original-project",
        },
    )

    links = service.run_links("run-original-project")

    assert links.agents == (
        "https://wandb.ai/other-team/original-project/weave/agents"
    )


def test_operator_results_prefers_enriched_normalized_exports(tmp_path: Path) -> None:
    service = make_operator_repo(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "demo.jsonl").write_text(
        json.dumps(
            {
                "record_type": "trial",
                "run_id": "run-1",
                "run_key": "run-1:task:codex:trial-1",
                "harness": "codex",
                "experiment_id": "demo",
                "variant_id": "baseline",
                "context_system_id": "none",
                "model": "openai/gpt-5",
                "pass": True,
                "reward": 0.8,
                "wall_time_sec": 4.0,
                "cost_usd": 0.02,
                "n_input_tokens": 100,
                "n_output_tokens": 20,
                "weave_agent_name": "codex",
                "weave_conversation_ids": ["conversation-1"],
                "weave_turn_count": 1,
                "weave_tool_call_count": 3,
            }
        )
        + "\n"
    )

    result = service.results()

    assert result.total == 1
    assert result.pass_rate == 1.0
    assert result.average_reward == 0.8
    assert result.average_wall_time_sec == 4.0
    assert result.tool_calls == 3
    assert result.turns == 1
    assert result.agent_traces[0].conversation_ids == ("conversation-1",)
