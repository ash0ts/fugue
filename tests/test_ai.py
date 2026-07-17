from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import pytest
import yaml
from test_operator import make_operator_repo

from fugue.assistant import AssistantAgent, AssistantModelClient
from fugue.bench.ai import (
    AnalysisResult,
    AnalysisScope,
    AnalysisSnapshot,
    AnalysisSpec,
    ExperimentAnalyst,
    ExperimentComposer,
    _aggregate,
    _write_analysis,
    get_analysis,
    save_analysis,
)
from fugue.bench.catalog import ExperimentCatalog
from fugue.bench.scoring import SelectionPolicy, select_candidate_configuration


def _client_factory(transport: httpx.MockTransport):
    def factory(model: str, env):
        return AssistantModelClient(model, env, transport=transport)

    return factory


def _tool_response(name: str, arguments: dict, call_id: str = "call-1") -> dict:
    return {
        "id": "response-1",
        "output": [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments),
            }
        ],
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }


def test_analysis_aggregates_retrieval_to_action_funnel_without_fake_usage() -> None:
    rows = [
        {
            "row_id": "row-1",
            "variant_id": "memory-policy",
            "pass": True,
            "context_invoked": True,
            "retrieval_recall_at_5": 0.5,
            "retrieval_recall_at_10": 1.0,
            "retrieval_mrr": 0.5,
            "context_result_open_rate": 0.5,
            "context_result_change_rate": 0.25,
            "relevant_retrieval_opened": True,
            "relevant_retrieval_changed": False,
            "off_target_change_only": False,
            "premature_completion": False,
            "n_input_tokens": None,
            "n_output_tokens": None,
        },
        {
            "row_id": "row-2",
            "variant_id": "memory-policy",
            "pass": False,
            "context_invoked": False,
            "retrieval_recall_at_5": 0.0,
            "retrieval_recall_at_10": 0.5,
            "retrieval_mrr": 0.0,
            "context_result_open_rate": None,
            "context_result_change_rate": None,
            "relevant_retrieval_opened": False,
            "relevant_retrieval_changed": False,
            "off_target_change_only": True,
            "premature_completion": True,
            "n_input_tokens": None,
            "n_output_tokens": None,
        },
    ]
    spec = AnalysisSpec(
        id="retrieval-funnel",
        title="Retrieval funnel",
        question="Did retrieval become action?",
        group_by=("variant_id",),
    )

    aggregates, _ = _aggregate(rows, spec)
    aggregate = aggregates[0]

    assert aggregate["context_invocation_rate"] == 0.5
    assert aggregate["retrieval_recall_at_5"] == 0.25
    assert aggregate["retrieval_recall_at_10"] == 0.75
    assert aggregate["context_result_open_rate"] == 0.5
    assert aggregate["relevant_retrieval_open_rate"] == 0.5
    assert aggregate["off_target_change_only_rate"] == 0.5
    assert aggregate["premature_completion_rate"] == 0.5
    assert aggregate["total_tokens"] is None


def test_composer_repairs_invalid_references_and_preview_stays_side_effect_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FUGUE_DISABLE_WEAVE", "1")
    service = make_operator_repo(tmp_path)
    base = service.experiment("demo").to_dict()
    attempts = 0
    session_ids: list[str] = []
    real_run = AssistantAgent.run

    async def record_session(agent, messages):
        session_ids.append(agent.session_id)
        return await real_run(agent, messages)

    monkeypatch.setattr("fugue.bench.ai.AssistantAgent.run", record_session)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        experiment = dict(base)
        experiment["id"] = "ai-demo"
        experiment["title"] = "AI demo"
        if attempts == 1:
            experiment["variants"] = [
                {
                    "id": "broken",
                    "label": "Broken",
                    "prompt_id": "does-not-exist",
                    "context": {"system_id": "none"},
                }
            ]
        return httpx.Response(
            200,
            json=_tool_response(
                "submit_experiment",
                {
                    "experiment": experiment,
                    "assets": [],
                    "rationale": "Keep the existing controlled demo.",
                    "assumptions": [],
                    "warnings": [],
                },
                f"call-{attempts}",
            ),
        )

    composer = ExperimentComposer(
        service,
        client_factory=_client_factory(httpx.MockTransport(handler)),
    )
    draft = asyncio.run(composer.compose("Make a small demo", base_experiment="demo"))

    assert attempts == 2
    assert len(set(session_ids)) == 1
    assert draft.experiment.id == "ai-demo"
    assert draft.preview.cells == 1
    assert not list((tmp_path / ".fugue/runtime").glob("*/run.json"))
    saved = composer.save(draft, experiment_id="accepted-ai-demo")
    assert saved.id == "accepted-ai-demo"


def test_composer_repairs_generated_evaluation_and_saves_only_after_acceptance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FUGUE_DISABLE_WEAVE", "1")
    service = make_operator_repo(tmp_path)
    base = service.experiment("demo").to_dict()
    base.update(
        {
            "id": "generated-demo",
            "title": "Generated demo",
            "judge_model": "openai/gpt-5-mini",
            "evaluation_generation": {
                "suite_id": "generated-suite",
                "workload_id": "capabilities",
                "size": 8,
                "sources": [
                    {"kind": "seed", "text": "The demo skill requires focused search."}
                ],
            },
            "workloads": [{"id": "capabilities", "runner": "harbor"}],
            "variants": [
                {
                    "id": "baseline",
                    "label": "Baseline",
                    "context": {"system_id": "none", "delivery": "portable"},
                },
                {
                    "id": "with-skill",
                    "label": "With skill",
                    "skills": ["demo-skill"],
                    "context": {"system_id": "none", "delivery": "portable"},
                },
            ],
        }
    )
    experiment_attempts = 0
    evaluation_attempts = 0

    def cases(count: int) -> list[dict]:
        strata = ["easy", "boundary", "failure", "integration"]
        return [
            {
                "id": f"generated-{index + 1:02d}",
                "instruction": f"Use focused search for scenario {index + 1}.",
                "family": "skill",
                "source_refs": ["seed:1"],
                "expected": {"facts": ["focused search"]},
                "tags": [strata[index % 4]],
            }
            for index in range(count)
        ]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal experiment_attempts, evaluation_attempts
        request_body = json.loads(request.content)
        assert request_body["max_output_tokens"] == 32_768
        tools = {tool["name"]: tool for tool in request_body["tools"]}
        if "submit_experiment" in tools:
            experiment_attempts += 1
            assert (
                "evaluation"
                not in tools["submit_experiment"]["parameters"]["properties"]
            )
            return httpx.Response(
                200,
                json=_tool_response(
                    "submit_experiment",
                    {
                        "experiment": base,
                        "assets": [],
                        "rationale": "Compare the skill against a true baseline.",
                        "assumptions": [],
                        "warnings": [],
                    },
                    f"experiment-{experiment_attempts}",
                ),
            )
        evaluation_attempts += 1
        submit_tool = tools["submit_evaluation"]
        case_schema = submit_tool["parameters"]["properties"]["cases"]["items"]
        assert case_schema["properties"]["instruction"]["maxLength"] == 2_000
        assert case_schema["additionalProperties"] is False
        return httpx.Response(
            200,
            json=_tool_response(
                "submit_evaluation",
                {
                    "suite_id": "generated-suite",
                    "cases": cases(7 if evaluation_attempts == 1 else 8),
                    "rubric": {
                        "dimensions": [
                            {
                                "id": "task_completion",
                                "criterion": "The task is complete.",
                            },
                            {
                                "id": "correctness",
                                "criterion": "Expected facts are correct.",
                            },
                            {
                                "id": "groundedness",
                                "criterion": "Claims use the supplied source.",
                            },
                        ]
                    },
                },
                f"evaluation-{evaluation_attempts}",
            ),
        )

    composer = ExperimentComposer(
        service,
        client_factory=_client_factory(httpx.MockTransport(handler)),
    )
    draft = asyncio.run(
        composer.compose("Generate the missing evaluation", base_experiment="demo")
    )

    assert experiment_attempts == 1
    assert evaluation_attempts == 2
    assert draft.evaluation is not None
    assert len(draft.evaluation.cases) == 8
    assert len(draft.assets) == 3
    assert draft.preview.cells == 16
    rendered = service.rendered_jobs(
        service.request_for_experiment(draft.experiment),
        run_id="preview",
        write_configs=False,
        experiment=draft.experiment,
        asset_overlay=draft.evaluation.overlay,
    )
    for task_id in {job.task_id for job in rendered}:
        task_jobs = [job for job in rendered if job.task_id == task_id]
        assert {job.variant_id for job in task_jobs} == {"baseline", "with-skill"}
        assert len({job.comparison_example_id for job in task_jobs}) == 1
    assert not (tmp_path / "configs/fugue/evaluations/generated-suite").exists()
    assert not (tmp_path / ".fugue").exists()

    composer.save(draft, experiment_id="accepted-generated-demo")

    suite = tmp_path / "configs/fugue/evaluations/generated-suite"
    assert {path.name for path in suite.iterdir()} == {
        "cases.jsonl",
        "manifest.yaml",
        "rubric.yaml",
    }


def test_composer_catalog_exposes_evidence_backed_agent_presets(tmp_path: Path):
    service = make_operator_repo(tmp_path)
    composer = ExperimentComposer(service)

    catalog = composer._catalog_summary(service.experiment("demo"))

    assert [item["id"] for item in catalog["agent_presets"]] == ["demo-maintainer"]
    assert catalog["agent_presets"][0]["evidence"]["metrics"] == {"pass_rate": 1.0}


def test_catalog_deduplicates_rows_and_blocks_secret_paths(tmp_path: Path) -> None:
    make_operator_repo(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    row = {
        "record_type": "trial",
        "run_id": "run-1",
        "run_key": "run-1:task:codex:1",
        "experiment_id": "demo",
        "workload_id": "harbor",
        "task_name": "task-one",
        "harness": "codex",
        "variant_id": "baseline",
        "candidate_id": "candidate-a",
        "tags": ["self-eval", "campaign:test"],
        "context_system_id": "none",
        "model_provider": "openai",
        "model": "openai/gpt-5",
        "pass": True,
        "reward": 1.0,
        "wall_time_sec": 2.0,
    }
    (reports / "one.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    catalog = ExperimentCatalog(tmp_path)
    status = catalog.refresh()

    assert "/catalog/v1/" in status.path
    assert status.experiments == 1
    assert status.records == 1
    assert catalog.facets()["intervention_type"] == {"baseline": 1}
    assert catalog.facets()["candidate_id"] == {"candidate-a": 1}
    assert catalog.facets()["tag"] == {"campaign:test": 1, "self-eval": 1}
    assert len(catalog.records(filters={"tag": "self-eval"})) == 1
    assert len(catalog.records(filters={"candidate_id": "candidate-a"})) == 1
    (tmp_path / "reports/.env").write_text("OPENAI_API_KEY=secret\n")
    try:
        catalog.read_artifact("reports/.env")
    except ValueError as exc:
        assert "secret policy" in str(exc)
    else:
        raise AssertionError("secret-like artifact should be blocked")


def test_catalog_connection_is_closed_after_context(tmp_path: Path) -> None:
    catalog = ExperimentCatalog(tmp_path)
    catalog.path.parent.mkdir(parents=True)

    with catalog._connect() as connection:
        connection.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_confirmed_self_eval_analysis_writes_review_only_promotion_bundle(
    tmp_path: Path,
) -> None:
    make_operator_repo(tmp_path)
    experiment_path = tmp_path / "configs/fugue/experiments/demo.yaml"
    experiment_path.write_text(
        experiment_path.read_text()
        + "\ntags: [self-eval, role:maintainer, suite:demo-v1]\n"
    )
    rows = []
    for candidate, harness, cost in (
        ("candidate-a", "codex", 0.5),
        ("candidate-b", "openclaw", 0.25),
    ):
        for index in (1, 2):
            rows.append(
                {
                    "row_id": f"{candidate}-{index}",
                    "record_type": "trial",
                    "run_id": "run-holdout",
                    "experiment_id": "demo",
                    "workload_id": "harbor",
                    "task_name": f"task-{index}",
                    "harness": harness,
                    "variant_id": "baseline",
                    "context_system_id": "none",
                    "candidate_id": candidate,
                    "comparison_example_id": f"example-{index}",
                    "trial_index": 1,
                    "model": "openai/gpt-5",
                    "pass": True,
                    "cost_usd": cost,
                    "wall_time_sec": 2.0,
                }
            )
    policy = SelectionPolicy(bootstrap_samples=200)
    selection = select_candidate_configuration(rows, policy, seed="snapshot")
    snapshot = AnalysisSnapshot(
        id="snapshot-demo",
        digest="a" * 64,
        created_at="2026-07-14T00:00:00+00:00",
        catalog_revision="revision",
        row_ids=tuple(row["row_id"] for row in rows),
        rows=tuple(rows),
    )
    spec = AnalysisSpec(
        id="demo-selection",
        title="Demo selection",
        question="Which candidate?",
        filters={"experiment_id": "demo", "tag": "phase:holdout"},
        selection=policy,
    )
    result = AnalysisResult(
        spec=spec,
        scope=AnalysisScope(
            experiments=("demo",),
            runs=("run-holdout",),
            rows=4,
            tasks=("task-1", "task-2"),
            models=("openai/gpt-5",),
            variants=("baseline",),
            sources=("local",),
            missing_metrics=(),
            warnings=(),
        ),
        snapshot=snapshot,
        evidence=(),
        aggregates=(),
        selection=selection,
        report="# Demo\n",
        report_dir=tmp_path / "reports/analyses/demo-selection/run-1",
        model="openai/gpt-5",
        provider="openai",
        session_id="session-1",
        input_tokens=1,
        output_tokens=1,
    )

    _write_analysis(result, tmp_path)

    promotion = tmp_path / "reports/self-eval/snapshot-demo/promotion.json"
    preset = tmp_path / "reports/self-eval/snapshot-demo/candidate-preset.yaml"
    assert promotion.is_file()
    assert json.loads(promotion.read_text())["selected_candidate_id"] == "candidate-b"
    assert preset.is_file()
    preset_data = yaml.safe_load(preset.read_text())
    assert preset_data["candidate"]["skills"] == []
    assert preset_data["evidence"]["suite_id"] == "demo-v1"
    assert "skill_ids" not in preset_data
    assert not (
        tmp_path / "configs/fugue/agent-presets/fugue-maintainer-recommended.yaml"
    ).exists()


def test_analyst_snapshots_scope_and_requires_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FUGUE_DISABLE_WEAVE", "1")
    service = make_operator_repo(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "demo.jsonl").write_text(
        json.dumps(
            {
                "record_type": "trial",
                "run_id": "run-1",
                "run_key": "run-1:task:codex:1",
                "experiment_id": "demo",
                "workload_id": "harbor",
                "task_name": "task-one",
                "harness": "codex",
                "variant_id": "baseline",
                "context_system_id": "none",
                "model_provider": "openai",
                "model": "openai/gpt-5",
                "pass": True,
                "reward": 0.8,
                "wall_time_sec": 4.0,
                "n_input_tokens": 100,
                "n_output_tokens": 20,
            }
        )
        + "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        names = {item["name"] for item in body.get("tools", [])}
        if "submit_analysis_plan" in names:
            return httpx.Response(
                200,
                json=_tool_response(
                    "submit_analysis_plan",
                    {
                        "id": "demo-analysis",
                        "title": "Demo analysis",
                        "filters": {"experiment_id": "demo"},
                        "group_by": ["experiment_id", "harness", "variant_id"],
                        "metrics": ["pass_rate", "reward", "wall_time_sec"],
                        "source": "hybrid",
                        "include_artifacts": False,
                    },
                ),
            )
        return httpx.Response(
            200,
            json=_tool_response(
                "submit_analysis_report",
                {
                    "claims": [
                        {
                            "text": "The scoped baseline passed its one trial.",
                            "evidence_ids": ["E001"],
                        }
                    ],
                    "conclusion": "Collect more trials before generalizing.",
                },
            ),
        )

    analyst = ExperimentAnalyst(
        service,
        client_factory=_client_factory(httpx.MockTransport(handler)),
    )
    weave_queries: list[list[str]] = []

    def fetch(run_keys, **kwargs):
        weave_queries.append(list(run_keys))
        return {
            "run-1:task:codex:1": {
                "weave_span_count": 2,
                "weave_conversation_ids": ["conversation-1"],
            }
        }

    monkeypatch.setattr("fugue.bench.ai.fetch_weave_summaries", fetch)
    spec = asyncio.run(
        analyst.plan(
            "How did the demo perform?",
            filters={"experiment_id": "demo"},
            source="hybrid",
        )
    )
    preview = analyst.prepare(spec)

    assert weave_queries == []
    assert preview.scope.rows == 1
    assert not (tmp_path / "reports/analyses").exists()

    result = asyncio.run(analyst.execute(preview))

    assert weave_queries == [["run-1:task:codex:1"]]
    assert result.scope.rows == 1
    assert result.aggregates[0]["pass_rate"] == 1.0
    assert "paired_pass_rate_delta" not in result.aggregates[0]
    assert "[E001]" in result.report
    assert (result.report_dir / "scope.json").is_file()
    path = save_analysis(result.spec, tmp_path)
    assert get_analysis(result.spec.id, tmp_path) == result.spec
    assert path.is_file()
