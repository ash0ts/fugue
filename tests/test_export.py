import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench import export
from fugue.bench.execution import CellOutcome, PlannedCell
from fugue.bench.export import (
    GeneratedEvaluationCoordinator,
    LiveEvaluationCoordinator,
    _fetch_agents_spans,
    _fetch_calls_spans,
    _summarize_spans,
    _weave_safe_row,
    export_rows,
    judge_qa_rows,
    publish_to_weave,
    write_jsonl,
)


def test_export_joins_harbor_result_and_fugue_meta(tmp_path: Path) -> None:
    jobs = Path(__file__).parent / "fixtures" / "export" / "jobs"

    rows = export_rows([jobs])

    assert len(rows) == 1
    row = rows[0]
    assert row["run_key"] == "bridge-check__abc123"
    assert row["harness"] == "hermes"
    assert row["experiment_id"] == "fixture-exp-id"
    assert row["variant_id"] == "baseline"
    assert row["prompt_id"] == "smoke-prompt"
    assert row["prompt_hashes"] == {"smoke-prompt": "prompt123"}
    assert row["skill_ids"] == ["repo-skill"]
    assert row["workload_id"] == "coding"
    assert row["preset_id"] == "smoke"
    assert row["context_system_id"] == "rag-bm25"
    assert row["context_version"] == "1"
    assert row["context_cache_keys"] == {"bridge-check": "cache123"}
    assert row["context_assigned"] is True
    assert row["context_available"] is True
    assert row["context_invoked"] is False
    assert row["context_query_count"] == 0
    assert row["agent_config_hash"] == "abc123"
    assert row["run_name"] == "fixture-exp"
    assert row["tags"] == ["fugue", "run:fixture-exp", "harness:hermes"]
    assert row["model_provider"] == "wandb"
    assert row["trace_project"] == "test/fugue"
    assert row["weave_agent_name"] == "hermes-agent"
    assert row["weave_conversation_ids"] == ["session-1"]
    assert row["native_session_ids"] == ["session-1"]
    assert row["reward"] == 1.0
    assert row["pass"] is True
    assert row["wall_time_sec"] == 5.0

    out = tmp_path / "pilot.jsonl"
    write_jsonl(rows, out)
    assert "bridge-check__abc123" in out.read_text()


def test_weave_payload_redacts_secrets_and_keeps_full_hits_local() -> None:
    row = {
        "query": "q" * 2_000,
        "api_key": "secret",
        "n_input_tokens": 123,
        "hits": [
            {
                "path": "src/app.py",
                "score": 0.9,
                "text": "repository source that remains local",
            }
        ],
        "trial_dir": "/private/jobs/trial",
    }

    safe = _weave_safe_row(row)

    assert len(safe["query"]) == 1_000
    assert safe["api_key"] == "[redacted]"
    assert safe["n_input_tokens"] == 123
    assert safe["hits"] == [{"path": "src/app.py", "score": 0.9}]
    assert "trial_dir" not in safe
    assert row["hits"][0]["text"].startswith("repository")


def test_qa_judge_uses_local_reference_and_records_separate_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / ".fugue" / "cache" / "datasets" / "qa" / "revision"
    dataset.mkdir(parents=True)
    (dataset / "selection.json").write_text(
        json.dumps([{"task_id": "qa-001", "source_index": 0}])
    )
    (dataset / "_source.jsonl").write_text(
        json.dumps({"answer": "Reference answer"}) + "\n"
    )
    trial = tmp_path / "jobs" / "trial"
    artifacts = trial / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "fugue-answer.md").write_text("Candidate answer")
    rows = [
        {
            "record_type": "trial",
            "workload_id": "qa",
            "task_name": "fugue/qa-001",
            "trial_dir": trial.as_posix(),
            "evidence_paths": ["src/app.py"],
        }
    ]

    def fake_request(client, route, api_key, **kwargs):
        assert kwargs["reference"] == "Reference answer"
        assert kwargs["answer"] == "Candidate answer"
        assert kwargs["evidence_paths"] == ["src/app.py"]
        return (
            {
                "correctness": 0.8,
                "completeness": 0.7,
                "groundedness": 0.9,
                "overall": 0.8,
                "reasoning": "Grounded but incomplete.",
            },
            {"input_tokens": 100, "output_tokens": 20},
        )

    monkeypatch.setattr(export, "_judge_request", fake_request)
    judge_qa_rows(
        rows,
        model="openai/gpt-5-mini",
        env={"OPENAI_API_KEY": "test-only"},
        repo_root=tmp_path,
    )

    assert rows[0]["judge_correctness"] == 0.8
    assert rows[0]["judge_groundedness"] == 0.9
    assert rows[0]["judge_input_tokens"] == 100
    assert rows[0]["judge_model"] == "openai/gpt-5-mini"
    safe = _weave_safe_row(rows[0])
    assert "judge_reasoning" not in safe


def test_weave_publication_uses_current_signature_and_local_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    loggers = []
    monkeypatch.setenv("WANDB_BASE_URL", "https://api.wandb.test")

    class FakeDataset:
        def __init__(self, *, name, rows) -> None:
            self.name = name
            self.rows = rows

    class FakeLogger:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)
            self.examples = []
            self.summary = None
            self.failed = None
            self.ui_url = "https://wandb.test/evaluations/eval-1"
            loggers.append(self)

        def log_example(self, inputs, output, scores) -> None:
            self.examples.append((inputs, output, scores))

        def log_summary(self) -> None:
            self.summary = True

        def fail(self, exception) -> None:
            self.failed = exception

    fake_weave = SimpleNamespace(
        init=lambda project: calls.append(
            ("init", project, __import__("os").environ.get("WANDB_BASE_URL"))
        ),
        Dataset=FakeDataset,
        EvaluationLogger=FakeLogger,
    )
    monkeypatch.setitem(sys.modules, "weave", fake_weave)
    project = f"entity/project-{tmp_path.name}"
    rows = [
        {
            "record_type": "trial",
            "experiment_id": "memory-ab",
            "run_id": "run-1",
            "run_name": "memory-smoke",
            "weave_agent_name": "codex",
            "task_name": "task",
            "harness": "codex",
            "variant_id": "rag-bm25",
            "context_system_id": "rag-bm25",
            "workload_id": "coding",
            "trial_index": 1,
            "comparison_example_id": "example-1",
            "candidate_id": "candidate-1",
            "model_provider": "wandb",
            "model": "wandb/test-model",
            "reward": 1.0,
            "pass": True,
        },
        {"record_type": "cell", "task_name": "task"},
        {"record_type": "preparation", "task_name": "task"},
    ]

    env = {
        "WANDB_API_KEY": "test-only",
        "WANDB_BASE_URL": "https://api.wandb.ai",
    }
    first = publish_to_weave(rows, project, ledger_root=tmp_path, env=env)
    second = publish_to_weave(rows, project, ledger_root=tmp_path)
    third = publish_to_weave(
        rows, project, ledger_root=tmp_path, republish=True
    )

    assert (first.published, first.skipped, first.failures) == (1, 0, ())
    assert (second.published, second.skipped) == (0, 1)
    assert (third.published, third.skipped) == (1, 0)
    assert first.evaluations[0].url == "https://wandb.test/evaluations/eval-1"
    assert len(loggers) == 2
    for logger in loggers:
        assert logger.name.startswith("memory-ab | coding |")
        assert logger.model["name"] == "codex__rag-bm25__test-model"
        assert logger.model["candidate_id"] == "candidate-1"
        assert logger.eval_attributes == {
            "fugue.evaluation_scope_id": logger.eval_attributes[
                "fugue.evaluation_scope_id"
            ],
            "fugue.experiment_id": "memory-ab",
            "fugue.workload_id": "coding",
            "fugue.record_type": "trial",
        }
        assert logger.dataset.rows == [
            {
                "comparison_example_id": "example-1",
                "workload_id": "coding",
                "task_id": "task",
            }
        ]
        inputs, output, scores = logger.examples[0]
        assert inputs == logger.dataset.rows[0]
        assert "harness" not in inputs
        assert "variant_id" not in inputs
        assert output["status"] == "passed"
        assert output["trace_link_status"] == "post_hoc_unlinked"
        assert scores == {"reward": 1.0, "passed": True}
        assert logger.summary is True
        assert logger.failed is None
    assert list((tmp_path / "v3").glob("**/*.json"))
    assert ("init", project, "https://api.wandb.ai") in calls


def test_weave_publication_keeps_direct_outcomes_and_skips_admin_rows(
    tmp_path: Path, monkeypatch
) -> None:
    logged = []
    summaries = []

    class FakeLogger:
        def __init__(self, **kwargs) -> None:
            pass

        def log_example(self, inputs, output, scores) -> None:
            logged.append(inputs["task_id"])

        def log_summary(self) -> None:
            summaries.append(True)

        def fail(self, exception) -> None:
            raise AssertionError(exception)

    monkeypatch.setitem(
        sys.modules,
        "weave",
        SimpleNamespace(init=lambda project: None, EvaluationLogger=FakeLogger),
    )
    rows = [
        {"record_type": "trial", "task_name": "trial"},
        {"record_type": "retrieval", "task_name": "query", "mrr": 1.0},
        {"record_type": "episode", "task_name": "episode"},
        {"record_type": "cell", "task_name": "cell"},
        {"record_type": "preparation", "task_name": "build"},
    ]

    published = publish_to_weave(
        rows,
        f"entity/project-{tmp_path.name}",
        ledger_root=tmp_path,
        env={"WANDB_API_KEY": "test-only"},
    )

    assert published.published == 3
    assert set(logged) == {"trial", "query", "episode"}
    assert summaries == [True, True, True]


def test_weave_publication_shares_dataset_across_candidates(
    tmp_path: Path, monkeypatch
) -> None:
    datasets = []
    loggers = []

    class FakeDataset:
        def __init__(self, *, name, rows) -> None:
            self.name = name
            self.rows = rows
            datasets.append(self)

    class FakeLogger:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)
            self.ui_url = None
            loggers.append(self)

        def log_example(self, inputs, output, scores) -> None:
            pass

        def log_summary(self) -> None:
            pass

        def fail(self, exception) -> None:
            raise AssertionError(exception)

    monkeypatch.setitem(
        sys.modules,
        "weave",
        SimpleNamespace(
            init=lambda project: None,
            Dataset=FakeDataset,
            EvaluationLogger=FakeLogger,
        ),
    )
    common = {
        "record_type": "trial",
        "experiment_id": "memory-ab",
        "run_id": "run-1",
        "workload_id": "coding",
        "task_name": "task-a",
        "trial_index": 1,
        "comparison_example_id": "example-a",
        "model": "wandb/test-model",
        "model_provider": "wandb",
    }
    rows = [
        {
            **common,
            "candidate_id": "candidate-none",
            "harness": "codex",
            "variant_id": "none",
            "context_system_id": "none",
            "pass": False,
        },
        {
            **common,
            "candidate_id": "candidate-rag",
            "harness": "codex",
            "variant_id": "rag-bm25",
            "context_system_id": "rag-bm25",
            "pass": True,
        },
    ]

    result = publish_to_weave(
        rows,
        f"entity/project-{tmp_path.name}",
        ledger_root=tmp_path,
        env={"WANDB_API_KEY": "test-only"},
    )

    assert result.published == 2
    assert len(datasets) == 1
    assert loggers[0].dataset is loggers[1].dataset
    assert datasets[0].rows == [
        {
            "comparison_example_id": "example-a",
            "workload_id": "coding",
            "task_id": "task-a",
        }
    ]
    assert loggers[0].name == loggers[1].name
    assert loggers[0].eval_attributes == loggers[1].eval_attributes
    assert loggers[0].scorers == loggers[1].scorers
    assert {logger.model["name"] for logger in loggers} == {
        "codex__none__test-model",
        "codex__rag-bm25__test-model",
    }
    assert {logger.model["variant_id"] for logger in loggers} == {
        "none",
        "rag-bm25",
    }


def test_weave_publication_groups_repeated_trials_under_one_example(
    tmp_path: Path, monkeypatch
) -> None:
    loggers = []

    class FakeDataset:
        def __init__(self, *, name, rows) -> None:
            self.name = name
            self.rows = rows

    class FakeLogger:
        ui_url = None

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)
            self.examples = []
            loggers.append(self)

        def log_example(self, inputs, output, scores) -> None:
            self.examples.append(inputs)

        def log_summary(self) -> None:
            pass

        def fail(self, exception) -> None:
            raise AssertionError(exception)

    monkeypatch.setitem(
        sys.modules,
        "weave",
        SimpleNamespace(
            init=lambda project: None,
            Dataset=FakeDataset,
            EvaluationLogger=FakeLogger,
        ),
    )
    common = {
        "record_type": "trial",
        "experiment_id": "memory-ab",
        "run_id": "run-1",
        "workload_id": "coding",
        "task_name": "task-a",
        "comparison_example_id": "example-a",
        "candidate_id": "candidate-a",
        "harness": "codex",
        "variant_id": "none",
        "context_system_id": "none",
    }

    result = publish_to_weave(
        [
            {**common, "trial_index": 1},
            {**common, "trial_index": 2},
        ],
        f"entity/project-{tmp_path.name}",
        ledger_root=tmp_path,
        env={"WANDB_API_KEY": "test-only"},
    )

    assert result.published == 1
    assert len(loggers[0].dataset.rows) == 1
    assert len(loggers[0].examples) == 2
    assert loggers[0].examples[0] == loggers[0].examples[1]


def test_live_evaluation_links_native_root_and_finalizes_cleanly(tmp_path: Path) -> None:
    loggers = []
    predictions = []

    class FakeDataset:
        def __init__(self, *, name, rows) -> None:
            self.name = name
            self.rows = rows

    class FakePrediction:
        def __init__(self, call_id: str) -> None:
            self.predict_and_score_call = SimpleNamespace(
                id=call_id,
                project_id="entity/project",
                summary=None,
            )
            self.output = None
            self.scores = {}
            self.finished = False

        def __enter__(self):
            return self

        def log_score(self, name, value) -> None:
            self.scores[name] = value

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.finished = True

    class FakeLogger:
        ui_url = "https://wandb.test/evaluations/live"

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)
            self._pseudo_evaluation = SimpleNamespace(
                ref=SimpleNamespace(uri="weave:///entity/project/object/eval:shared")
            )
            self.summarized = False
            loggers.append(self)

        def log_prediction(self, inputs):
            prediction = FakePrediction(f"predict-{len(predictions) + 1}")
            predictions.append(prediction)
            return prediction

        def log_summary(self) -> None:
            self.summarized = True

        def fail(self, exception) -> None:
            raise AssertionError(exception)

    fake_weave = SimpleNamespace(Dataset=FakeDataset, EvaluationLogger=FakeLogger)
    cell = PlannedCell(
        id="cell-a",
        run_id="run-a",
        run_name="memory-smoke",
        workload_id="coding",
        task_id="task-a",
        harness="codex",
        context_system_id="rag-bm25",
        variant_id="rag-bm25",
        model_provider="wandb",
        model="wandb/test-model",
        trial_index=1,
        comparison_example_id="example-a",
        candidate_id="candidate-a",
        config_path=Path("config.json"),
        result_path=Path("jobs/missing/result.json"),
        command=("harbor", "run"),
        env={
            "WANDB_API_KEY": "secret",
            "WANDB_ENTITY": "entity",
            "WANDB_PROJECT": "project",
            "FUGUE_EXPERIMENT_ID": "memory-ab",
            "FUGUE_DATASET": "fixture/tasks@1",
            "FUGUE_TRACE_CONTENT": "full",
        },
        n_attempts=1,
    )

    def summaries(**kwargs):
        call_id = predictions[0].predict_and_score_call.id
        return {
            next(iter(kwargs["run_keys"])): {
                "weave_agent_names": ["codex"],
                "weave_conversation_ids": ["native-conversation"],
                "weave_trace_ids": ["a" * 32],
                "weave_root_span_ids": ["b" * 16],
                "weave_root_spans": [
                    {
                        "conversation_id": "native-conversation",
                        "agent_name": "codex",
                        "trace_id": "a" * 32,
                        "span_id": "b" * 16,
                        "run_key": (
                            "run-a:coding:trial:task-a:codex:rag-bm25:"
                            "rag-bm25:t001"
                        ),
                        "harness": "codex",
                        "task_id": "task-a",
                        "eval_predict_and_score_call_id": call_id,
                    }
                ],
            }
        }

    coordinator = LiveEvaluationCoordinator(
        [cell],
        repo_root=tmp_path,
        project="entity/project",
        env=cell.env,
        weave_module=fake_weave,
        summary_fetcher=summaries,
        trace_timeout_sec=0,
    )
    overlay = coordinator.begin_cell(cell)
    assert overlay == {
        "FUGUE_WEAVE_EVAL_PREDICT_AND_SCORE_CALL_ID": "predict-1",
        "FUGUE_WEAVE_EVAL_PROJECT_ID": "entity/project",
        "FUGUE_WEAVE_EVAL_NAME": loggers[0].name,
        "FUGUE_EVALUATION_SCOPE_ID": loggers[0].eval_attributes[
            "fugue.evaluation_scope_id"
        ],
    }

    coordinator.finish_cell(cell, CellOutcome(cell.id, "passed", returncode=0))
    publication = coordinator.finalize()

    assert publication.published == 1
    assert publication.failures == ()
    assert publication.evaluations[0].linked_predictions == 1
    assert predictions[0].finished is True
    assert predictions[0].output["observed_conversation_id"] == "native-conversation"
    assert predictions[0].output["trace_link_status"] == "linked"
    live_row = json.loads(
        (tmp_path / ".fugue/runtime/run-a/evaluation-results.jsonl").read_text()
    )
    assert live_row["evaluation_prediction_latency_sec"] >= 0
    assert predictions[0].predict_and_score_call.summary == {
        "weave": {
            "genai_span_ref": [{"trace_id": "a" * 32, "span_id": "b" * 16}]
        }
    }
    assert loggers[0].summarized is True
    statuses = [
        json.loads(line)["status"]
        for line in (tmp_path / ".fugue/runtime/run-a/evaluations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert statuses == ["pending", "prediction_open", "trace_linked", "finalized"]


def test_live_evaluation_rows_recover_prediction_latency(tmp_path: Path) -> None:
    (tmp_path / "evaluation-results.jsonl").write_text(
        json.dumps({"cell_id": "cell-a", "run_key": "run-key"}) + "\n"
    )
    (tmp_path / "evaluations.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "cell_id": "cell-a",
                        "status": "prediction_open",
                        "recorded_at": "2026-07-14T22:00:00+00:00",
                    }
                ),
                json.dumps(
                    {
                        "cell_id": "cell-a",
                        "status": "finalized",
                        "recorded_at": "2026-07-14T22:02:03.5+00:00",
                    }
                ),
            ]
        )
        + "\n"
    )

    rows = export._live_evaluation_rows(tmp_path)

    assert rows[0]["evaluation_prediction_latency_sec"] == 123.5
    assert rows[0]["evaluation_publication_mode"] == "live"


def test_completed_evaluation_preserves_planned_dataset_identity(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "jobs" / "job" / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "swe-bench/task-a",
                "trial_name": "trial",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )
    (trial_dir / "agent" / "fugue-meta.json").write_text(
        json.dumps({"candidate_id": "candidate-a", "trial_index": 1})
    )
    cell = PlannedCell(
        id="cell-a",
        run_id="run-a",
        run_name="run-a",
        workload_id="coding",
        task_id="task-a",
        harness="hermes",
        context_system_id="none",
        variant_id="none",
        model_provider="wandb",
        model="wandb/test-model",
        trial_index=1,
        comparison_example_id="example-a",
        candidate_id="candidate-a",
        config_path=tmp_path / "config.json",
        result_path=tmp_path / "jobs" / "job" / "result.json",
        command=("harbor", "run"),
        env={
            "FUGUE_DATASET": "fixture/tasks@1",
            "FUGUE_REPOSITORY": "org/repo",
            "FUGUE_BASE_COMMIT": "abc123",
            "FUGUE_EXPECTED_EVIDENCE_PATHS": json.dumps(
                {"task-a": ["src/expected.py"]}
            ),
        },
        n_attempts=1,
    )
    planned = export._planned_evaluation_row(cell)

    row = export._completed_evaluation_row(
        cell, CellOutcome(cell.id, "passed", returncode=0), planned
    )

    assert row["task_name"] == "task-a"
    assert row["dataset"] == "fixture/tasks@1"
    assert row["comparison_example_id"] == "example-a"
    assert row["expected_evidence_paths"] == ["src/expected.py"]
    assert (
        export._publication_candidates([row])[0]["evaluation_scope_id"]
        == export._publication_candidates([planned])[0]["evaluation_scope_id"]
    )


def test_generated_evaluation_scope_is_shared_and_rubric_sensitive() -> None:
    case = {
        "id": "case-a",
        "instruction": "Answer from the supplied capability source.",
        "source_refs": [{"id": "seed:1", "sha256": "a" * 64}],
        "expected": {"facts": ["grounded fact"]},
        "scorer_dimensions": ["task_completion", "correctness"],
    }
    rubric = {
        "id": "suite-a",
        "dimensions": [
            {
                "id": "task_completion",
                "criterion": "Complete the task.",
                "threshold": 0.7,
            },
            {
                "id": "correctness",
                "criterion": "Include the grounded fact.",
                "threshold": 0.7,
            },
        ],
    }
    cell = PlannedCell(
        id="cell-a",
        run_id="run-a",
        run_name="run-a",
        workload_id="capabilities",
        task_id="case-a",
        harness="codex",
        context_system_id="none",
        variant_id="baseline",
        model_provider="openai",
        model="openai/gpt-5",
        trial_index=1,
        comparison_example_id="shared-example",
        candidate_id="candidate-a",
        config_path=Path("config.json"),
        result_path=Path("jobs/missing/result.json"),
        command=("harbor", "run"),
        env={"FUGUE_DATASET": "generated/suite"},
        n_attempts=1,
        evaluation_case=case,
        evaluation_rubrics=(rubric,),
        scorer_hashes={"rubric.yaml": "b" * 64},
        scorer_refs=("rubric.yaml",),
    )
    baseline = export._planned_evaluation_row(cell)
    treatment = export._planned_evaluation_row(
        replace(
            cell,
            id="cell-b",
            candidate_id="candidate-b",
            variant_id="with-skill",
            trial_index=2,
        )
    )

    candidates = export._publication_candidates([baseline, treatment])

    assert len(candidates) == 2
    assert {value["evaluation_scope_id"] for value in candidates} == {
        candidates[0]["evaluation_scope_id"]
    }
    inputs = export._evaluation_inputs(baseline)
    assert inputs["evaluation_case"] == case
    assert inputs["evaluation_rubrics"] == [rubric]
    assert "candidate_id" not in inputs
    assert "variant_id" not in inputs
    assert "trial_index" not in inputs
    assert "evaluation_correctness" in candidates[0]["scorers"]
    assert "evaluation_overall" not in candidates[0]["scorers"]

    changed = json.loads(json.dumps(treatment))
    changed["evaluation_rubrics"][0]["dimensions"][1]["criterion"] = (
        "Use a changed correctness definition."
    )
    changed_scope = export._publication_candidates([changed])[0][
        "evaluation_scope_id"
    ]
    assert changed_scope != candidates[0]["evaluation_scope_id"]


def test_local_generated_evaluation_runs_scoring_without_changing_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = {
        "id": "case-a",
        "instruction": "Answer from the source.",
        "source_refs": [{"id": "seed:1", "sha256": "a" * 64}],
        "expected": {"facts": ["grounded fact"]},
        "scorer_dimensions": ["task_completion", "correctness"],
    }
    rubric = {
        "id": "suite-a",
        "dimensions": [
            {"id": "task_completion", "criterion": "complete", "threshold": 0.7},
            {"id": "correctness", "criterion": "correct", "threshold": 0.7},
        ],
    }
    cell = PlannedCell(
        id="cell-a",
        run_id="run-a",
        run_name="run-a",
        workload_id="capabilities",
        task_id="case-a",
        harness="codex",
        context_system_id="none",
        variant_id="baseline",
        model_provider="openai",
        model="openai/gpt-5",
        trial_index=1,
        comparison_example_id="example-a",
        candidate_id="candidate-a",
        config_path=tmp_path / "config.json",
        result_path=tmp_path / "jobs" / "missing" / "result.json",
        command=("harbor", "run"),
        env={
            "FUGUE_DATASET": "generated/suite-a",
            "FUGUE_JUDGE_MODEL": "openai/gpt-5-mini",
        },
        n_attempts=1,
        evaluation_case=case,
        evaluation_rubrics=(rubric,),
        scorer_hashes={"rubric.yaml": "b" * 64},
        scorer_refs=("rubric.yaml",),
    )
    calls = []

    def score(row, **kwargs):
        calls.append(kwargs)
        row["evaluation_task_completion"] = 1
        row["evaluation_correctness"] = 0.9

    monkeypatch.setattr(export, "apply_generated_evaluation", score)
    coordinator = GeneratedEvaluationCoordinator(
        [cell], repo_root=tmp_path, env={"PRIVATE_TOKEN": "secret-value"}
    )

    coordinator.finish_cell(cell, CellOutcome(cell.id, "passed", returncode=0))

    result = json.loads(
        (tmp_path / ".fugue/runtime/run-a/evaluation-results.jsonl").read_text()
    )
    assert len(calls) == 1
    assert calls[0]["judge_model"] == "openai/gpt-5-mini"
    assert calls[0]["case"] == case
    assert calls[0]["rubrics"] == (rubric,)
    assert result["evaluation_publication_mode"] == "local"
    assert result["evaluation_task_completion"] == 1
    assert result["evaluation_correctness"] == 0.9
    assert "evaluation_overall" not in result
    assert "secret-value" not in json.dumps(result)


def test_weave_publication_never_republishes_finalized_live_predictions(
    tmp_path: Path, monkeypatch
) -> None:
    class UnexpectedLogger:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("finalized live predictions must not be republished")

    monkeypatch.setattr(
        export,
        "initialize_weave",
        lambda project, env: SimpleNamespace(EvaluationLogger=UnexpectedLogger),
    )
    result = publish_to_weave(
        [
            {
                "record_type": "trial",
                "evaluation_publication_mode": "live",
                "experiment_id": "memory-ab",
                "run_id": "run-a",
                "workload_id": "coding",
                "dataset": "fixture/tasks@1",
                "task_name": "task-a",
                "comparison_example_id": "example-a",
                "candidate_id": "candidate-a",
                "harness": "codex",
                "variant_id": "none",
                "context_system_id": "none",
                "trial_index": 1,
            }
        ],
        "entity/project",
        ledger_root=tmp_path,
        env={"WANDB_API_KEY": "test-only"},
    )

    assert result.published == 0
    assert result.skipped == 1


def test_weave_publication_fails_transactionally(tmp_path: Path, monkeypatch) -> None:
    failed = []

    class FakeLogger:
        ui_url = None

        def __init__(self, **kwargs) -> None:
            pass

        def log_example(self, inputs, output, scores) -> None:
            pass

        def log_summary(self) -> None:
            raise RuntimeError("summary failed")

        def fail(self, exception) -> None:
            failed.append(exception)

    monkeypatch.setitem(
        sys.modules,
        "weave",
        SimpleNamespace(init=lambda project: None, EvaluationLogger=FakeLogger),
    )
    result = publish_to_weave(
        [{"record_type": "trial", "task_name": "task-a"}],
        f"entity/project-{tmp_path.name}",
        ledger_root=tmp_path,
        env={"WANDB_API_KEY": "test-only"},
    )

    assert result.published == 0
    assert result.failures and "summary failed" in result.failures[0]
    assert isinstance(failed[0], RuntimeError)
    assert not list((tmp_path / "v3").glob("**/*.json"))


def test_weave_publication_rejects_duplicate_candidate_examples(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "weave",
        SimpleNamespace(init=lambda project: None, EvaluationLogger=object),
    )
    row = {
        "record_type": "trial",
        "candidate_id": "candidate-a",
        "comparison_example_id": "example-a",
    }
    with pytest.raises(ValueError, match="duplicate evaluation trial"):
        publish_to_weave(
            [row, dict(row)],
            f"entity/project-{tmp_path.name}",
            ledger_root=tmp_path,
            env={"WANDB_API_KEY": "test-only"},
        )


def test_calls_query_uses_current_shape_and_decodes_ndjson() -> None:
    requests = []

    class Response:
        status_code = 200
        text = '{"id":"root"}\n{"id":"chat"}\n'

    class Client:
        def post(self, url, *, json):
            requests.append((url, json))
            return Response()

    spans = _fetch_calls_spans(
        Client(),
        "https://trace.wandb.ai",
        "team/fugue-experiments",
        "run-key-1",
    )

    assert spans == [{"id": "root"}, {"id": "chat"}]
    url, payload = requests[0]
    assert url == "https://trace.wandb.ai/calls/stream_query"
    assert payload["filter"] == {"trace_roots_only": False}
    assert "op_name" not in payload["filter"]
    assert payload["query"]["$expr"]["$eq"][1] == {"$literal": "run-key-1"}


def test_calls_query_does_not_hide_transport_errors() -> None:
    class Response:
        status_code = 503
        text = "unavailable"

    class Client:
        @staticmethod
        def post(url, *, json):
            return Response()

    with pytest.raises(RuntimeError, match="HTTP 503"):
        _fetch_calls_spans(
            Client(),
            "https://trace.wandb.ai",
            "team/fugue-experiments",
            "run-key-1",
        )


def test_agent_span_query_uses_conversation_identity() -> None:
    requests = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"spans": [{"id": "turn-1"}]}

    class Client:
        def post(self, url, *, json):
            requests.append((url, json))
            return Response()

    spans = _fetch_agents_spans(
        Client(),
        "https://trace.wandb.ai",
        "team/fugue-experiments",
        ["conversation-1", "conversation-1"],
    )

    assert spans == [{"id": "turn-1"}]
    assert requests == [
        (
            "https://trace.wandb.ai/agents/spans/query",
            {
                "project_id": "team/fugue-experiments",
                "query": {
                    "$expr": {
                        "$eq": [
                            {"$getField": "conversation_id"},
                            {"$literal": "conversation-1"},
                        ]
                    }
                },
                "include_details": True,
                "include_costs": True,
                "limit": 10_000,
            },
        )
    ]


def test_agent_span_query_does_not_hide_transport_errors() -> None:
    class Response:
        status_code = 404

    class Client:
        @staticmethod
        def post(url, *, json):
            return Response()

    with pytest.raises(RuntimeError, match="HTTP 404"):
        _fetch_agents_spans(
            Client(),
            "https://trace.wandb.ai",
            "team/fugue-experiments",
            ["conversation-1"],
        )


def test_agent_span_summary_counts_logical_hierarchy_once() -> None:
    spans = [
        {
            "id": "turn",
            "trace_id": "trace-1",
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "hermes-agent",
                "gen_ai.conversation.id": "conversation-1",
                "gen_ai.usage.input_tokens": 12,
                "gen_ai.usage.output_tokens": 3,
            },
        },
        {
            "id": "chat",
            "parent_id": "turn",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.conversation.id": "conversation-1",
                "gen_ai.usage.input_tokens": 12,
                "gen_ai.usage.output_tokens": 3,
            },
        },
        {
            "id": "tool",
            "parent_id": "chat",
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.conversation.id": "conversation-1",
            },
        },
        {"id": "tool", "attributes": {"gen_ai.operation.name": "execute_tool"}},
    ]

    summary = _summarize_spans(spans)

    assert summary["weave_span_count"] == 3
    assert summary["weave_turn_count"] == 1
    assert summary["weave_llm_call_count"] == 1
    assert summary["weave_tool_call_count"] == 1
    assert summary["weave_agent_names"] == ["hermes-agent"]
    assert summary["weave_conversation_ids"] == ["conversation-1"]
    assert summary["weave_root_span_ids"] == ["turn"]
    assert summary["weave_input_tokens"] == 12
    assert summary["weave_output_tokens"] == 3
    assert summary["weave_usage_source"] == "chat_sum"


def test_observed_identity_accepts_benchmark_task_namespace() -> None:
    row = {
        "weave_agent_name": "hermes-agent",
        "run_key": "run-key",
        "task_name": "swe-bench/astropy__astropy-12907",
        "weave_root_spans": [
            {
                "agent_name": "hermes-agent",
                "run_key": "run-key",
                "task_id": "astropy__astropy-12907",
                "conversation_id": "native-session",
                "trace_id": "a" * 32,
                "span_id": "b" * 16,
            }
        ],
    }

    export._apply_observed_identity(row)

    assert row["trace_link_status"] == "observed"
    assert row["observed_conversation_id"] == "native-session"


def test_observed_identity_preserves_verified_live_link() -> None:
    row = {
        "trace_link_status": "linked",
        "weave_agent_name": "codex",
        "run_key": "run-key",
        "task_name": "task-a",
        "weave_root_spans": [
            {
                "agent_name": "codex",
                "run_key": "run-key",
                "task_id": "task-a",
                "conversation_id": "native-session",
                "trace_id": "a" * 32,
                "span_id": "b" * 16,
            }
        ],
    }

    export._apply_observed_identity(row)

    assert row["trace_link_status"] == "linked"


def test_agent_span_summary_preserves_unavailable_usage() -> None:
    summary = _summarize_spans(
        [
            {
                "id": "turn",
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": "openclaw",
                },
            },
            {
                "id": "chat",
                "parent_id": "turn",
                "operation_name": "chat",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": 0.0,
            },
        ]
    )

    assert summary["weave_input_tokens"] is None
    assert summary["weave_output_tokens"] is None
    assert summary["weave_total_cost_usd"] is None
    assert summary["weave_usage_status"] == "unavailable"


def test_not_applicable_cell_does_not_report_a_missing_trace() -> None:
    row = {
        "record_type": "cell",
        "status": "not_applicable",
        "applicable": False,
        "weave_root_spans": [],
        "trace_link_status": "missing",
        "trace_link_error": "no matching native invoke_agent root",
    }

    export._apply_observed_identity(row)

    assert row["trace_link_status"] == "not_applicable"
    assert row["trace_link_error"] is None
    assert row["weave_observability_status"] == "not_applicable"
    assert row["weave_usage_status"] == "not_applicable"


def test_native_chat_response_fills_full_trace_output_but_metadata_only_hashes() -> None:
    summary = _summarize_spans(
        [
            {
                "id": "chat",
                "operation_name": "chat",
                "ended_at": "2026-07-14T12:00:00+00:00",
                "output_messages": [
                    {"role": "assistant", "content": "final native response"}
                ],
            }
        ]
    )
    full = {"trace_content": "full", "agent_response_bytes": 0}
    metadata = {"trace_content": "metadata", "agent_response_bytes": 0}

    export._apply_trace_summary(full, dict(summary))
    export._apply_trace_summary(metadata, dict(summary))

    assert full["agent_response"] == "final native response"
    assert full["agent_response_bytes"] == 21
    assert len(full["agent_response_sha256"]) == 64
    assert "agent_response" not in metadata
    assert metadata["agent_response_bytes"] == 21
    assert metadata["agent_response_sha256"] == full["agent_response_sha256"]


def test_agent_span_summary_does_not_turn_missing_trace_errors_into_zero() -> None:
    summary = _summarize_spans([])

    assert summary["weave_observability_status"] == "unavailable"
    assert summary["weave_span_count"] == 0
    assert "weave_terminal_error_count" not in summary
    assert "weave_model_error_count" not in summary
    assert "weave_tool_error_count" not in summary
    assert summary["weave_usage_source"] == "unavailable"


def test_agent_span_summary_separates_error_categories() -> None:
    summary = _summarize_spans(
        [
            {
                "id": "turn",
                "status": "error",
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
            },
            {
                "id": "chat",
                "parent_id": "turn",
                "status": "error",
                "attributes": {"gen_ai.operation.name": "chat"},
            },
            {
                "id": "tool",
                "parent_id": "turn",
                "status": "error",
                "attributes": {"gen_ai.operation.name": "execute_tool"},
            },
        ]
    )

    assert summary["weave_terminal_error_count"] == 1
    assert summary["weave_model_error_count"] == 1
    assert summary["weave_tool_error_count"] == 1


def test_agent_span_summary_preserves_measured_zero_usage() -> None:
    summary = _summarize_spans(
        [
            {
                "id": "turn",
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
            },
            {
                "id": "chat",
                "parent_id": "turn",
                "operation_name": "chat",
                "attributes": {
                    "gen_ai.usage.input_tokens": 0,
                    "gen_ai.usage.output_tokens": 0,
                    "gen_ai.usage.total_cost_usd": 0.0,
                },
            },
        ]
    )

    assert summary["weave_input_tokens"] == 0
    assert summary["weave_output_tokens"] == 0
    assert summary["weave_total_cost_usd"] == 0.0
    assert summary["weave_usage_status"] == "available"
    assert summary["weave_cost_status"] == "available"


def test_evaluation_scores_do_not_replace_explicitly_unavailable_usage() -> None:
    scores = export._evaluation_scores(
        {
            "weave_usage_status": "unavailable",
            "weave_input_tokens": None,
            "weave_output_tokens": None,
            "n_input_tokens": 0,
            "n_output_tokens": 0,
            "cost_usd": 0.0,
        }
    )

    assert "input_tokens" not in scores
    assert "output_tokens" not in scores
    assert "total_cost_usd" not in scores


def test_agent_span_summary_verifies_flat_fugue_attributes() -> None:
    fugue_attributes = {
        "fugue.run_key": "run-key",
        "fugue.run_id": "run-id",
        "fugue.experiment_id": "memory-ab",
        "fugue.workload_id": "coding",
        "fugue.harness": "codex",
        "fugue.variant_id": "rag-bm25",
        "fugue.context_system_id": "rag-bm25",
        "fugue.task_id": "task-a",
        "fugue.trial_index": "1",
        "fugue.comparison_example_id": "example-a",
        "fugue.candidate_id": "candidate-a",
        "fugue.model_provider": "wandb",
        "fugue.model": "wandb/test-model",
    }
    summary = _summarize_spans(
        [
            {
                "span_id": "turn",
                "operation_name": "invoke_agent",
                "custom_attrs_string": fugue_attributes,
            },
            {
                "span_id": "chat",
                "parent_span_id": "turn",
                "operation_name": "chat",
                "custom_attrs_string": fugue_attributes,
            },
        ]
    )

    assert summary["weave_attribute_status"] == "complete"
    assert summary["weave_missing_attributes"] == []
    assert summary["weave_fugue_attributes"] == fugue_attributes


def test_agent_span_summary_supports_agents_api_rows() -> None:
    spans = [
        {
            "span_id": "turn",
            "trace_id": "trace-1",
            "operation_name": "invoke_agent",
            "agent_name": "codex",
            "conversation_id": "conversation-1",
            "status_code": "OK",
        },
        {
            "span_id": "chat",
            "parent_span_id": "turn",
            "operation_name": "chat",
            "conversation_id": "conversation-1",
            "input_tokens": 12,
            "output_tokens": 3,
            "status_code": "OK",
        },
        {
            "span_id": "tool",
            "parent_span_id": "chat",
            "operation_name": "execute_tool",
            "conversation_id": "conversation-1",
            "status_code": "ERROR",
        },
    ]

    summary = _summarize_spans(spans)

    assert summary["weave_span_count"] == 3
    assert summary["weave_turn_count"] == 1
    assert summary["weave_llm_call_count"] == 1
    assert summary["weave_tool_call_count"] == 1
    assert summary["weave_error_count"] == 1
    assert summary["weave_agent_names"] == ["codex"]
    assert summary["weave_conversation_ids"] == ["conversation-1"]
    assert summary["weave_input_tokens"] == 12
    assert summary["weave_output_tokens"] == 3


def test_weave_enrichment_marks_expected_agent_identity(monkeypatch) -> None:
    jobs = Path(__file__).parent / "fixtures" / "export" / "jobs"

    monkeypatch.setattr(
        export,
        "fetch_weave_summaries",
        lambda **kwargs: {
            "bridge-check__abc123": {
                "weave_span_count": 1,
                "weave_agent_names": ["hermes-agent"],
            }
        },
    )
    matched = export_rows([jobs], fetch_weave=True, env={"WANDB_API_KEY": "x"})
    assert matched[0]["weave_agent_name_match"] is True

    monkeypatch.setattr(
        export,
        "fetch_weave_summaries",
        lambda **kwargs: {
            "bridge-check__abc123": {
                "weave_span_count": 1,
                "weave_agent_names": ["wrong-agent"],
            }
        },
    )
    mismatched = export_rows([jobs], fetch_weave=True, env={"WANDB_API_KEY": "x"})
    assert mismatched[0]["weave_agent_name_match"] is False

    monkeypatch.setattr(
        export,
        "fetch_weave_summaries",
        lambda **kwargs: {"bridge-check__abc123": {"weave_span_count": 0}},
    )
    missing = export_rows([jobs], fetch_weave=True, env={"WANDB_API_KEY": "x"})
    assert missing[0]["weave_agent_name_match"] is None
