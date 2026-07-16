import json
import sys
import threading
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.agent_tracing import agent_conversation_id
from fugue.bench import export, operator
from fugue.bench.candidates import CANDIDATE_IDENTITY_SCHEMA_VERSION
from fugue.bench.execution import CellOutcome, PlannedCell, write_run_manifest
from fugue.bench.export import (
    GeneratedEvaluationCoordinator,
    LiveEvaluationCoordinator,
    PublicationResult,
    PublishedEvaluation,
    _fetch_agents_spans,
    _fetch_calls_spans,
    _summarize_spans,
    _weave_safe_row,
    export_rows,
    judge_qa_rows,
    publish_to_weave,
    write_jsonl,
)
from fugue.bench.operator import OperatorService


def _write_export_fixture(tmp_path: Path) -> Path:
    trial = tmp_path / "jobs" / "pilot" / "bridge-check__abc123"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "fugue/smoke-bridge-check",
                "trial_name": "bridge-check__abc123",
                "config": {"agent": {"model_name": "wandb/zai-org/GLM-5.2"}},
                "agent_info": {"name": "wandb-hermes"},
                "agent_result": {
                    "n_input_tokens": 10,
                    "n_cache_tokens": 0,
                    "n_output_tokens": 5,
                    "cost_usd": 0.01,
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
                "exception_info": None,
                "started_at": "2026-07-08T22:19:56.798954Z",
                "finished_at": "2026-07-08T22:20:01.798954Z",
            }
        )
    )
    (trial / "agent" / "fugue-meta.json").write_text(
        json.dumps(
            {
                "run_key": "bridge-check__abc123",
                "job_name": "pilot",
                "harness": "hermes",
                "experiment_id": "fixture-exp-id",
                "run_name": "fixture-exp",
                "run_group": "fixture-exp",
                "variant_id": "baseline",
                "prompt_id": "smoke-prompt",
                "prompt_hashes": {"smoke-prompt": "prompt123"},
                "skill_ids": ["repo-skill"],
                "workload_id": "coding",
                "preset_id": "smoke",
                "context_system_id": "rag-bm25",
                "context_version": "1",
                "context_config_hash": "context123",
                "context_cache_keys": {"bridge-check": "cache123"},
                "context_artifact": {"context_system_id": "rag-bm25"},
                "expected_artifact_paths": ["/logs/artifacts/fugue-answer.md"],
                "artifact_normalization": [
                    {
                        "status": "recovered",
                        "source": "/workspace/logs/artifacts/fugue-answer.md",
                        "target": "/logs/artifacts/fugue-answer.md",
                    }
                ],
                "agent_config_hash": "abc123",
                "evaluation_scope_id": "scope-123",
                "tags": ["fugue", "run:fixture-exp", "harness:hermes"],
                "model_provider": "wandb",
                "model": "wandb/zai-org/GLM-5.2",
                "trace_project": "test/fugue",
                "weave_agent_name": "hermes-agent",
                "weave_conversation_ids": ["session-1"],
                "native_session_ids": ["session-1"],
                "trace_content": "full",
            }
        )
    )
    return tmp_path / "jobs"


def test_export_reads_each_source_path_once(tmp_path: Path) -> None:
    run_dir = tmp_path / ".fugue" / "runtime" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "context-results.jsonl").write_text(
        json.dumps(
            {
                "record_type": "retrieval",
                "run_key": "run-a:retrieval:probe-a",
                "task_name": "probe-a",
            }
        )
        + "\n"
    )

    rows = export_rows([run_dir, run_dir.resolve()])

    assert len(rows) == 1
    assert rows[0]["run_key"] == "run-a:retrieval:probe-a"


def test_export_joins_harbor_result_and_fugue_meta(tmp_path: Path) -> None:
    jobs = _write_export_fixture(tmp_path)

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
    assert row["expected_artifact_paths"] == ["/logs/artifacts/fugue-answer.md"]
    assert row["artifact_normalization"][0]["status"] == "recovered"
    assert row["context_assigned"] is True
    assert row["context_available"] is True
    assert row["context_invoked"] is False
    assert row["context_query_count"] == 0
    assert row["agent_config_hash"] == "abc123"
    assert row["evaluation_scope_id"] == "scope-123"
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
    assert row["local_usage_status"] == "available"
    assert row["n_input_tokens"] == 10
    assert row["n_cache_tokens"] == 0
    assert row["n_output_tokens"] == 5
    assert row["cost_usd"] == 0.01

    out = tmp_path / "pilot.jsonl"
    write_jsonl(rows, out)
    assert "bridge-check__abc123" in out.read_text()


def test_export_marks_unattributed_harbor_zero_usage_unavailable(
    tmp_path: Path,
) -> None:
    jobs = _write_export_fixture(tmp_path)
    result_path = next(jobs.rglob("result.json"))
    result = json.loads(result_path.read_text())
    result["agent_result"] = {
        "n_input_tokens": 0,
        "n_cache_tokens": 0,
        "n_output_tokens": 0,
        "cost_usd": 0.0,
    }
    result_path.write_text(json.dumps(result))

    [row] = export_rows([jobs])

    assert row["local_usage_status"] == "unavailable"
    assert row["n_input_tokens"] is None
    assert row["n_cache_tokens"] is None
    assert row["n_output_tokens"] is None
    assert row["cost_usd"] is None
    scores = export._evaluation_scores(row)
    assert "input_tokens" not in scores
    assert "output_tokens" not in scores
    assert "total_cost_usd" not in scores


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


def test_jsonl_export_redacts_secret_keys_and_values(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"

    write_jsonl(
        [
            {
                "api_key": "named-secret",
                "error": "provider rejected opaque-live-secret-value",
            }
        ],
        output,
        env={"WANDB_API_KEY": "opaque-live-secret-value"},
    )

    payload = json.loads(output.read_text())
    assert payload["api_key"] == "[redacted]"
    assert payload["error"] == "provider rejected [redacted]"


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
        rows,
        project,
        ledger_root=tmp_path,
        republish=True,
        republish_reason="verify revised publication",
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
    markers = [
        path
        for path in (tmp_path / "v1").glob("**/*.json")
        if path.parent.name != "predictions"
    ]
    assert len(markers) == 2
    assert len(list((tmp_path / "v1").glob("**/predictions/*.json"))) == 1
    marker_values = [json.loads(path.read_text()) for path in markers]
    assert sorted(value["revision"] for value in marker_values) == [1, 2]
    assert sum(value["active"] is True for value in marker_values) == 1
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
        {
            "record_type": "trial",
            "task_name": "trial",
            "run_id": "run-trial",
            "candidate_id": "candidate-trial",
            "comparison_example_id": "example-trial",
            "trial_index": 1,
        },
        {
            "record_type": "retrieval",
            "task_name": "query",
            "mrr": 1.0,
            "run_id": "run-a",
            "candidate_id": "candidate-a",
            "execution_fingerprint": "fingerprint-a",
            "execution_kind": "provider_diagnostic",
            "trial_index": 1,
            "workload_id": "retrieval-dataset",
            "comparison_example_id": "example-query",
        },
        {
            "record_type": "episode",
            "task_name": "episode",
            "sequence_id": "sequence-a",
        },
        {
            "record_type": "cell",
            "task_name": "cell",
            "run_id": "run-a",
            "candidate_id": "candidate-a",
            "execution_fingerprint": "fingerprint-a",
            "execution_kind": "provider_diagnostic",
            "trial_index": 1,
            "workload_id": "retrieval",
            "status": "passed",
        },
        {"record_type": "preparation", "task_name": "build"},
    ]

    published = publish_to_weave(
        rows,
        f"entity/project-{tmp_path.name}",
        ledger_root=tmp_path,
        env={"WANDB_API_KEY": "test-only"},
    )

    assert published.published == 2
    assert set(logged) == {"trial", "query"}
    assert summaries == [True, True]


def test_weave_publication_counts_one_prediction_per_sequence_cell(
    tmp_path: Path, monkeypatch
) -> None:
    declared_scorers = []
    logged_scores = []

    class FakeLogger:
        ui_url = "https://wandb.test/evaluations/direct"

        def __init__(self, **kwargs) -> None:
            declared_scorers.extend(kwargs["scorers"])

        def log_example(self, inputs, output, scores) -> None:
            logged_scores.append(scores)

        def log_summary(self) -> None:
            pass

        def fail(self, exception) -> None:
            raise AssertionError(exception)

    monkeypatch.setattr(
        export,
        "initialize_weave",
        lambda project, env: SimpleNamespace(EvaluationLogger=FakeLogger),
    )
    common = {
        "experiment_id": "memory-ab",
        "run_id": "run-a",
        "workload_id": "continuity",
        "dataset": "repository-continuity",
        "task_name": "maintainer-preferences",
        "candidate_id": "candidate-direct",
        "harness": "sequence",
        "variant_id": "markdown-log",
        "context_system_id": "markdown-log",
        "execution_kind": "provider_diagnostic",
        "execution_fingerprint": "fingerprint-a",
        "trial_index": 1,
    }
    rows = (
        [
            {
                **common,
                "record_type": "episode",
                "sequence_id": "maintainer-preferences",
                "comparison_example_id": f"episode-{index}",
                "write_latency_ms": 2.0,
                "storage_bytes": 100 + index,
            }
            for index in range(2)
        ]
        + [
            {
                **common,
                "record_type": "retrieval",
                "sequence_id": "maintainer-preferences",
                "comparison_example_id": f"probe-{index}",
                "mrr": float(index),
                "query_latency_ms": 3.0,
            }
            for index in range(2)
        ]
        + [
            {
                **common,
                "record_type": "cell",
                "workload_id": "continuity",
                "comparison_example_id": "sequence-cell",
                "status": "passed",
            }
        ]
    )

    result = publish_to_weave(
        rows,
        "entity/project",
        ledger_root=tmp_path,
        env={"WANDB_API_KEY": "test-only"},
    )

    assert result.published == 1
    assert sum(item.examples for item in result.evaluations) == 1
    assert sum(item.direct_predictions for item in result.evaluations) == 1
    assert sum(item.agent_predictions for item in result.evaluations) == 0
    assert logged_scores == [
        {
            "context_queries": 2,
            "context_query_latency_ms": 6.0,
            "context_storage_bytes": 101,
            "context_write_latency_ms": 4.0,
            "episodes": 2,
            "mrr": 0.5,
        }
    ]
    assert set(logged_scores[0]) <= set(declared_scorers)


def test_direct_evaluation_projection_requires_a_completed_cell() -> None:
    def cell(*, status: str, fingerprint: str) -> dict[str, object]:
        return {
            "record_type": "cell",
            "run_id": "run-a",
            "candidate_id": "candidate-a",
            "execution_fingerprint": fingerprint,
            "execution_kind": "provider_diagnostic",
            "trial_index": 1,
            "status": status,
            "workload_id": "retrieval",
        }

    def measurement(*, fingerprint: str, query: str) -> dict[str, object]:
        return {
            "record_type": "retrieval",
            "run_id": "run-a",
            "candidate_id": "candidate-a",
            "execution_fingerprint": fingerprint,
            "execution_kind": "provider_diagnostic",
            "trial_index": 1,
            "workload_id": "retrieval-dataset",
            "comparison_example_id": query,
        }

    rows = [
        cell(status="failed", fingerprint="failed"),
        measurement(fingerprint="failed", query="failed-query"),
        cell(status="passed", fingerprint="passed"),
        measurement(fingerprint="passed", query="published-query"),
        cell(status="passed", fingerprint="empty"),
    ]

    projected = export._evaluation_rows(rows)

    assert [row["comparison_example_id"] for row in projected] == ["published-query"]
    assert projected[0]["dataset"] == "retrieval-dataset"
    assert projected[0]["workload_id"] == "retrieval"

    normalized = export.normalize_prediction_rows(rows)
    assert len(normalized) == 1
    assert normalized[0]["record_type"] == "trial"
    assert normalized[0]["source_record_type"] == "retrieval"
    assert normalized[0]["prediction_schema_version"] == 1
    assert normalized[0]["prediction_id"]
    assert normalized[0]["execution_kind"] == "provider_diagnostic"


def test_prediction_identity_ignores_scores_but_rejects_duplicate_execution() -> None:
    row = {
        "record_type": "trial",
        "run_id": "run-a",
        "candidate_id": "candidate-a",
        "comparison_example_id": "example-a",
        "trial_index": 1,
        "execution_kind": "agent",
        "reward": 1.0,
    }

    first = export.normalize_prediction_rows([row])[0]
    changed = export.normalize_prediction_rows([{**row, "reward": 0.0}])[0]

    assert first["prediction_id"] == changed["prediction_id"]
    with pytest.raises(ValueError, match="duplicate evaluation trial"):
        export.normalize_prediction_rows([row, dict(row)])


def test_export_persists_direct_evaluations_without_replacing_live_agent_runs(
    tmp_path: Path, monkeypatch
) -> None:
    live = PublishedEvaluation(
        candidate_id="candidate-agent",
        name="memory | coding | agent-scope",
        examples=1,
        url="https://wandb.test/evaluations/live",
        agent_predictions=1,
        linked_agent_predictions=1,
        linking_failures=("agent link reason remains visible",),
    )
    direct = PublishedEvaluation(
        candidate_id="candidate-direct",
        name="memory | continuity | direct-scope",
        examples=4,
        url="https://wandb.test/evaluations/direct-v1",
        evaluation_ref="weave:///direct-v1",
        direct_predictions=4,
    )
    updated_direct = replace(
        direct,
        url="https://wandb.test/evaluations/direct-v2",
        evaluation_ref="weave:///direct-v2",
    )
    write_run_manifest(
        tmp_path,
        "run-a",
        {
            "status": "passed",
            "run_name": "memory",
            "experiment_id": "memory-ab",
            "trace_project": "entity/project",
            "jobs_dirs": [],
            "evaluation_runs": [asdict(live)],
        },
    )
    publications = iter(
        (
            PublicationResult(1, 0, (direct,)),
            PublicationResult(0, 1, (direct,)),
            PublicationResult(1, 0, (updated_direct,)),
        )
    )
    monkeypatch.setattr(operator, "export_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        operator,
        "publish_to_weave",
        lambda *args, **kwargs: next(publications),
    )
    service = OperatorService(tmp_path)
    output = tmp_path / "reports" / "run-a.jsonl"

    service.export_run("run-a", out=output, to_weave=True)
    service.export_run("run-a", out=output, to_weave=True)
    service.export_run(
        "run-a",
        out=output,
        to_weave=True,
        republish=True,
        republish_reason="correct direct evaluation scope",
    )

    evaluations = service.run_summary("run-a").evaluations
    assert len(evaluations) == 2
    assert evaluations[0] == live
    assert evaluations[1] == updated_direct
    assert evaluations[1].direct_predictions == evaluations[1].examples == 4


def test_run_export_reads_only_the_exact_planned_job_roots(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "run-scoped"
    selected = tmp_path / "jobs" / "demo" / "selected-job"
    unrelated = tmp_path / "jobs" / "demo" / "older-job"
    selected.mkdir(parents=True)
    unrelated.mkdir(parents=True)
    write_run_manifest(
        tmp_path,
        run_id,
        {
            "status": "passed",
            "run_name": "scoped export",
            "experiment_id": "demo",
            "jobs_dirs": ["jobs/demo"],
            "job_paths": ["jobs/demo/selected-job"],
        },
    )
    observed: list[Path] = []

    def fake_export_rows(paths, **kwargs):
        observed.extend(paths)
        return []

    monkeypatch.setattr(operator, "export_rows", fake_export_rows)

    OperatorService(tmp_path).export_run(run_id)

    assert observed == [selected, tmp_path / ".fugue" / "runtime" / run_id]
    assert unrelated not in observed


def test_export_recovers_direct_evaluation_after_marker_only_crash(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeLogger:
        ui_url = "https://wandb.test/evaluations/recovered"

        def __init__(self, **kwargs) -> None:
            pass

        def log_example(self, inputs, output, scores) -> None:
            pass

        def log_summary(self) -> None:
            pass

        def fail(self, exception) -> None:
            raise AssertionError(exception)

    monkeypatch.setattr(
        export,
        "initialize_weave",
        lambda project, env: SimpleNamespace(EvaluationLogger=FakeLogger),
    )
    rows = [
        {
            "record_type": "retrieval",
            "experiment_id": "memory-ab",
            "run_id": "run-a",
            "workload_id": "retrieval",
            "dataset": "repository-retrieval",
            "task_name": "probe-a",
            "comparison_example_id": "example-a",
            "candidate_id": "candidate-direct",
            "harness": "direct",
            "variant_id": "rag-bm25",
            "context_system_id": "rag-bm25",
            "execution_kind": "provider_diagnostic",
            "execution_fingerprint": "fingerprint-a",
            "trial_index": 1,
        },
        {
            "record_type": "cell",
            "run_id": "run-a",
            "candidate_id": "candidate-direct",
            "execution_kind": "provider_diagnostic",
            "execution_fingerprint": "fingerprint-a",
            "trial_index": 1,
            "workload_id": "retrieval",
            "status": "passed",
        },
    ]
    ledger = tmp_path / ".fugue" / "runtime" / "publications"
    first = publish_to_weave(
        rows,
        "entity/project",
        ledger_root=ledger,
        env={"WANDB_API_KEY": "test-only"},
    )
    assert first.published == 1
    write_run_manifest(
        tmp_path,
        "run-a",
        {
            "status": "passed",
            "run_name": "memory",
            "experiment_id": "memory-ab",
            "trace_project": "entity/project",
            "jobs_dirs": [],
            "evaluation_runs": [],
        },
    )
    monkeypatch.setattr(operator, "export_rows", lambda *args, **kwargs: rows)

    recovered = OperatorService(tmp_path).export_run(
        "run-a",
        out=tmp_path / "reports" / "run-a.jsonl",
        to_weave=True,
    )

    assert (recovered.published, recovered.skipped) == (0, 1)
    assert len(recovered.evaluations) == 1
    assert recovered.evaluations[0].url == FakeLogger.ui_url
    assert recovered.evaluations[0].direct_predictions == 1
    run_evaluations = OperatorService(tmp_path).run_summary("run-a").evaluations
    assert run_evaluations == recovered.evaluations


@pytest.mark.parametrize(
    ("field", "tampered", "message"),
    (
        ("candidate_id", "candidate-other", "candidate_id does not match"),
        ("evaluation_scope_id", "scope-other", "evaluation_scope_id does not match"),
        ("publication_mode", "live", "publication_mode does not match"),
        ("examples", -1, "examples must be a nonnegative integer"),
        (
            "linked_agent_predictions",
            2,
            "linked_agent_predictions cannot exceed agent_predictions",
        ),
        ("direct_predictions", 2, "prediction counts cannot exceed examples"),
    ),
)
def test_publication_marker_rejects_tampered_identity_and_counts(
    tmp_path: Path, field: str, tampered: object, message: str
) -> None:
    marker = tmp_path / "publication.json"
    value = {
        "project": "entity/project",
        "publication_id": "publication-a",
        "candidate_id": "candidate-a",
        "evaluation_scope_id": "scope-a",
        "publication_mode": "post_hoc",
        "name": "memory | retrieval | scope-a",
        "examples": 1,
        "agent_predictions": 1,
        "linked_agent_predictions": 1,
        "direct_predictions": 0,
        "linking_failures": [],
    }
    value[field] = tampered
    marker.write_text(json.dumps(value))

    with pytest.raises(ValueError, match=message):
        export._published_evaluation_from_marker(
            marker,
            project="entity/project",
            publication_id="publication-a",
            candidate_id="candidate-a",
            evaluation_scope_id="scope-a",
            publication_mode="post_hoc",
        )


def test_export_persists_publication_failures_without_clobbering_agent_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    live = PublishedEvaluation(
        candidate_id="candidate-agent",
        name="memory | coding | agent-scope",
        examples=1,
        url="https://wandb.test/evaluations/live",
        agent_predictions=1,
        linked_agent_predictions=1,
    )
    write_run_manifest(
        tmp_path,
        "run-a",
        {
            "status": "passed",
            "run_name": "memory",
            "experiment_id": "memory-ab",
            "trace_project": "entity/project",
            "jobs_dirs": [],
            "evaluation_runs": [asdict(live)],
            "evaluation_failures": ["live Agent publication warning"],
        },
    )
    failure = "candidate-direct: RuntimeError: direct publication failed"
    monkeypatch.setattr(operator, "export_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        operator,
        "publish_to_weave",
        lambda *args, **kwargs: PublicationResult(0, 0, failures=(failure,)),
    )
    service = OperatorService(tmp_path)

    service.export_run("run-a", to_weave=True)
    service.export_run("run-a", to_weave=True)

    run = service.run_summary("run-a")
    assert run.evaluations == (live,)
    assert run.evaluation_failures == (
        "live Agent publication warning",
        failure,
    )


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


def test_live_evaluation_links_native_root_and_finalizes_cleanly(
    tmp_path: Path,
) -> None:
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
        execution_fingerprint="execution-a",
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
        evaluation_case={
            "id": "task-a",
            "scorer_dimensions": ["task_completion", "artifact_quality"],
            "expected": {"artifacts": []},
        },
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
                            "run-a:coding:trial:task-a:codex:rag-bm25:rag-bm25:t001"
                        ),
                        "harness": "codex",
                        "task_id": "task-a",
                        "candidate_id": "candidate-a",
                        "comparison_example_id": "example-a",
                        "trial_index": 1,
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
    assert publication.evaluations[0].agent_predictions == 1
    assert publication.evaluations[0].linked_agent_predictions == 1
    assert publication.evaluations[0].direct_predictions == 0
    assert predictions[0].finished is True
    assert predictions[0].output["observed_conversation_id"] == "native-conversation"
    assert predictions[0].output["trace_link_status"] == "linked"
    live_row = json.loads(
        (tmp_path / ".fugue/runtime/run-a/evaluation-results.jsonl").read_text()
    )
    assert live_row["evaluation_prediction_latency_sec"] >= 0
    assert live_row["evaluation_judge_status"] == "not_requested"
    assert live_row["adapter_outcome"]["rubric_evaluation"]["state"] == (
        "not_requested"
    )
    assert predictions[0].predict_and_score_call.summary == {
        "weave": {"genai_span_ref": [{"trace_id": "a" * 32, "span_id": "b" * 16}]}
    }
    assert loggers[0].summarized is True
    statuses = [
        json.loads(line)["status"]
        for line in (tmp_path / ".fugue/runtime/run-a/evaluations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert statuses == ["pending", "prediction_open", "trace_linked", "finalized"]


def test_live_cancellation_closes_open_prediction_once_without_trace_polling(
    tmp_path: Path,
) -> None:
    predictions = []
    loggers = []

    class FakeDataset:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakePrediction:
        def __init__(self, call_id: str) -> None:
            self.predict_and_score_call = SimpleNamespace(
                id=call_id,
                project_id="entity/project",
                summary=None,
            )
            self.output = None
            self.exit_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.exit_count += 1

    class FakeLogger:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)
            self.failed = None
            loggers.append(self)

        def log_prediction(self, inputs):
            prediction = FakePrediction(f"call-{len(predictions) + 1}")
            predictions.append(prediction)
            return prediction

        def fail(self, exception) -> None:
            self.failed = exception

    def cell(name: str) -> PlannedCell:
        return PlannedCell(
            id=f"cell-{name}",
            run_id="run-cancel",
            run_name="cancel",
            workload_id="coding",
            task_id=f"task-{name}",
            harness="codex",
            context_system_id="none",
            variant_id="none",
            model_provider="wandb",
            model="wandb/test-model",
            trial_index=1,
            comparison_example_id=f"example-{name}",
            candidate_id=f"candidate-{name}",
            execution_fingerprint=f"execution-{name}",
            config_path=Path(f"{name}.json"),
            result_path=Path("jobs") / name / "result.json",
            command=("harbor", "run"),
            env={"WANDB_API_KEY": "test-only"},
            n_attempts=1,
        )

    cells = [cell("active"), cell("queued")]
    coordinator = LiveEvaluationCoordinator(
        cells,
        repo_root=tmp_path,
        project="entity/project",
        env=cells[0].env,
        weave_module=SimpleNamespace(
            Dataset=FakeDataset,
            EvaluationLogger=FakeLogger,
        ),
        summary_fetcher=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cancelled predictions must not poll Weave")
        ),
    )

    coordinator.begin_cell(cells[0])
    coordinator.finish_cell(
        cells[0],
        CellOutcome(cells[0].id, "cancelled", error="operator cancellation"),
    )
    publication = coordinator.finalize(cancelled=True)

    assert publication.failures == ()
    assert predictions[0].exit_count == 1
    assert predictions[0].output["status"] == "cancelled"
    assert predictions[0].output["trace_link_status"] == "cancelled"
    assert predictions[0].output["trace_link_reason"] == "operator cancellation"
    assert all(logger.failed is not None for logger in loggers)
    statuses = [
        json.loads(line)["status"]
        for line in (tmp_path / ".fugue/runtime/run-cancel/evaluations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert statuses.count("prediction_open") == 1
    assert statuses.count("cancelled") == 2
    assert "failed" not in statuses
    assert "finalized" not in statuses


def test_live_cancellation_during_trace_fetch_closes_prediction_once(
    tmp_path: Path,
) -> None:
    cancellation = threading.Event()
    fetch_started = threading.Event()
    release_fetch = threading.Event()

    class FakePrediction:
        def __init__(self) -> None:
            self.predict_and_score_call = SimpleNamespace(
                id="call-a", project_id="entity/project", summary=None
            )
            self.output = None
            self.exit_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.exit_count += 1

    prediction = FakePrediction()

    class FakeLogger:
        def __init__(self, **kwargs) -> None:
            self.failed = None

        def log_prediction(self, inputs):
            return prediction

        def fail(self, exception) -> None:
            self.failed = exception

    cell = PlannedCell(
        id="cell-a",
        run_id="run-poll-cancel",
        run_name="cancel during polling",
        workload_id="coding",
        task_id="task-a",
        harness="codex",
        context_system_id="none",
        variant_id="none",
        model_provider="wandb",
        model="wandb/test-model",
        trial_index=1,
        comparison_example_id="example-a",
        candidate_id="candidate-a",
        execution_fingerprint="execution-a",
        config_path=Path("config.json"),
        result_path=Path("jobs/missing/result.json"),
        command=("harbor", "run"),
        env={"WANDB_API_KEY": "test-only"},
        n_attempts=1,
    )

    def summaries(**kwargs):
        fetch_started.set()
        assert release_fetch.wait(2)
        return {}

    coordinator = LiveEvaluationCoordinator(
        [cell],
        repo_root=tmp_path,
        project="entity/project",
        env=cell.env,
        weave_module=SimpleNamespace(EvaluationLogger=FakeLogger),
        summary_fetcher=summaries,
        trace_timeout_sec=45,
        cancellation_event=cancellation,
    )
    coordinator.begin_cell(cell)
    worker = threading.Thread(
        target=coordinator.finish_cell,
        args=(cell, CellOutcome(cell.id, "passed", returncode=0)),
    )
    worker.start()
    assert fetch_started.wait(2)

    cancellation.set()
    release_fetch.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert prediction.exit_count == 1
    assert prediction.output["status"] == "cancelled"
    assert prediction.output["trace_link_status"] == "cancelled"
    statuses = [
        json.loads(line)["status"]
        for line in (tmp_path / ".fugue/runtime/run-poll-cancel/evaluations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert statuses == ["pending", "prediction_open", "cancelled"]


def test_pre_agent_setup_failure_skips_trace_poll_and_reports_observability_failure(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "jobs/job/trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "swe-bench/task-a",
                "trial_name": "trial-a",
                "agent_execution": None,
                "exception_info": {
                    "exception_type": "RuntimeError",
                    "exception_message": "environment setup failed",
                },
            }
        )
    )

    class FakePrediction:
        def __init__(self) -> None:
            self.predict_and_score_call = SimpleNamespace(
                id="call-a", project_id="entity/project", summary=None
            )
            self.output = None

        def __enter__(self):
            return self

        def log_score(self, name, value) -> None:
            pass

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

    prediction = FakePrediction()

    class FakeLogger:
        ui_url = None

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def log_prediction(self, inputs):
            return prediction

        def log_summary(self) -> None:
            pass

        def fail(self, exception) -> None:
            raise AssertionError(exception)

    cell = PlannedCell(
        id="cell-a",
        run_id="run-pre-agent",
        run_name="pre-agent",
        workload_id="coding",
        task_id="task-a",
        harness="codex",
        context_system_id="none",
        variant_id="none",
        model_provider="wandb",
        model="wandb/test-model",
        trial_index=1,
        comparison_example_id="example-a",
        candidate_id="candidate-a",
        execution_fingerprint="execution-a",
        config_path=tmp_path / "config.json",
        result_path=tmp_path / "jobs/job/result.json",
        command=("harbor", "run"),
        env={"WANDB_API_KEY": "test-only"},
        n_attempts=1,
    )
    coordinator = LiveEvaluationCoordinator(
        [cell],
        repo_root=tmp_path,
        project="entity/project",
        env=cell.env,
        weave_module=SimpleNamespace(EvaluationLogger=FakeLogger),
        summary_fetcher=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("pre-agent failures must not poll Weave")
        ),
    )

    coordinator.begin_cell(cell)
    coordinator.finish_cell(
        cell,
        CellOutcome(cell.id, "failed", returncode=1, error="trial failed"),
    )
    publication = coordinator.finalize()

    assert len(publication.failures) == 1
    assert publication.evaluations[0].agent_predictions == 1
    assert publication.evaluations[0].linked_agent_predictions == 0
    assert publication.evaluations[0].linking_failures == (
        "cell-a: Agent execution did not start; no invoke_agent root was emitted",
    )
    assert prediction.output["trace_link_status"] == "not_started"
    assert prediction.output["trace_link_error"] == (
        "Agent execution did not start; no invoke_agent root was emitted"
    )


def test_direct_diagnostic_does_not_open_or_synthesize_agent_prediction(
    tmp_path: Path,
) -> None:
    class FailIfConstructed:
        def __init__(self, **kwargs) -> None:
            raise AssertionError(kwargs)

    cell = PlannedCell(
        id="cell-direct",
        run_id="run-a",
        run_name="memory-smoke",
        workload_id="retrieval",
        task_id="dataset-a",
        harness="direct",
        context_system_id="rag-bm25",
        variant_id="rag-bm25",
        model_provider="wandb",
        model="wandb/test-model",
        trial_index=1,
        comparison_example_id="example-a",
        candidate_id="candidate-a",
        execution_fingerprint="execution-a",
        config_path=Path("dataset.yaml"),
        result_path=Path("jobs/missing/result.json"),
        command=("python", "-m", "fugue.bench.cli"),
        env={"WANDB_API_KEY": "secret", "FUGUE_DATASET": "dataset-a"},
        n_attempts=1,
        execution_kind="provider_diagnostic",
    )
    fake_weave = SimpleNamespace(EvaluationLogger=FailIfConstructed)

    coordinator = LiveEvaluationCoordinator(
        [cell],
        repo_root=tmp_path,
        project="entity/project",
        env=cell.env,
        weave_module=fake_weave,
    )
    planned = export._planned_evaluation_row(cell)
    export._apply_observed_identity(planned)

    assert coordinator.begin_cell(cell) is None
    assert coordinator.finalize().published == 0
    assert planned["execution_kind"] == "provider_diagnostic"
    assert planned["trace_link_status"] == "not_applicable"
    assert "weave_agent_name" not in planned
    assert "planned_conversation_id" not in planned
    assert "weave_conversation_id" not in planned


@pytest.mark.parametrize("harness", ["hermes", "openclaw"])
def test_planned_agent_uses_adapter_conversation_identity(harness: str) -> None:
    cell = PlannedCell(
        id=f"cell-{harness}",
        run_id="run-a",
        run_name="memory-smoke",
        workload_id="coding",
        task_id="task-a",
        harness=harness,
        context_system_id="none",
        variant_id="none",
        model_provider="wandb",
        model="wandb/test-model",
        trial_index=1,
        comparison_example_id="example-a",
        candidate_id=f"candidate-{harness}",
        execution_fingerprint=f"execution-{harness}",
        config_path=Path("config.json"),
        result_path=Path("jobs/result.json"),
        command=("harbor", "run"),
        env={"FUGUE_DATASET": "dataset-a"},
        n_attempts=1,
    )

    planned = export._planned_evaluation_row(cell)
    expected = agent_conversation_id(harness, planned["run_key"])

    assert planned["planned_conversation_id"] == expected
    assert planned["weave_conversation_id"] == expected


def test_agent_hierarchy_uses_one_resolved_conversation_identity() -> None:
    resolved = agent_conversation_id("openclaw", "run-a:task-a:openclaw:t001")
    summary = _summarize_spans(
        [
            {
                "id": "root",
                "trace_id": "trace-a",
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.conversation.id": resolved,
                },
            },
            {
                "id": "chat",
                "trace_id": "trace-a",
                "parent_id": "root",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.conversation.id": resolved,
                },
            },
            {
                "id": "tool",
                "trace_id": "trace-a",
                "parent_id": "chat",
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.conversation.id": resolved,
                },
                "output": {
                    "_meta": {
                        "fugue_gateway_call_id": "gateway-a",
                        "fugue_context_system_id": "gitnexus",
                        "fugue_gitnexus_vector": {
                            "vector_search_attempted": True,
                            "vector_search_succeeded": True,
                            "semantic_result_count": 4,
                            "bm25_result_count": 2,
                            "model_digest": "sha256:model",
                            "query_latency_ms": 12.5,
                        },
                    }
                },
            },
        ]
    )

    assert summary["weave_conversation_ids"] == [resolved]
    assert summary["weave_turn_count"] == 1
    assert summary["weave_llm_call_count"] == 1
    assert summary["weave_tool_call_count"] == 1
    assert summary["weave_gateway_tool_call_count"] == 1
    assert summary["weave_gateway_call_ids"] == ["gateway-a"]
    assert summary["gitnexus_vector_search_attempted"] is True
    assert summary["gitnexus_vector_search_succeeded"] is True
    assert summary["gitnexus_semantic_result_count"] == 4
    assert summary["gitnexus_bm25_result_count"] == 2
    assert summary["gitnexus_vector_model_digests"] == ["sha256:model"]
    assert summary["gitnexus_vector_query_latency_ms"] == 12.5

    row = {"context_assigned": True}
    export._apply_trace_summary(row, dict(summary))
    assert row["context_invoked"] is True
    assert row["context_invocation_evidence"]["source"] == (
        "mcp_gateway_result_metadata"
    )


def test_agent_hierarchy_decodes_gateway_metadata_from_remote_tool_result() -> None:
    result = json.dumps(
        {
            "Ok": {
                "content": [{"type": "text", "text": "result"}],
                "_meta": {
                    "fugue_gateway_call_id": "gateway-remote",
                    "fugue_gitnexus_vector": {
                        "vector_search_attempted": True,
                        "vector_search_succeeded": True,
                        "semantic_result_count": 3,
                        "bm25_result_count": 0,
                        "model_digest": "sha256:remote-model",
                        "query_latency_ms": 22.5,
                    },
                },
            }
        }
    )
    raw_span = {
        "attributes": {
            "gen_ai": {
                "operation": {"name": "execute_tool"},
                "tool": {"call": {"result": result}},
            }
        }
    }
    summary = _summarize_spans(
        [
            {
                "span_id": "tool",
                "operation_name": "execute_tool",
                "raw_span_dump": json.dumps(raw_span),
            }
        ]
    )

    assert summary["weave_gateway_call_ids"] == ["gateway-remote"]
    assert summary["gitnexus_vector_search_attempted"] is True
    assert summary["gitnexus_vector_search_succeeded"] is True
    assert summary["gitnexus_semantic_result_count"] == 3
    assert summary["gitnexus_bm25_result_count"] == 0
    assert summary["gitnexus_vector_model_digests"] == ["sha256:remote-model"]
    assert summary["gitnexus_vector_query_latency_ms"] == 22.5


def test_gateway_event_log_is_identity_checked_and_preserves_vector_evidence(
    tmp_path: Path,
) -> None:
    event_log = (
        tmp_path
        / ".fugue/runtime/run-a/gateway-evidence/job-a/context-gateway.jsonl"
    )
    event_log.parent.mkdir(parents=True)
    identity = {
        "fugue_run_id": "run-a",
        "fugue_candidate_id": "candidate-a",
        "fugue_comparison_example_id": "example-a",
        "fugue_trial_index": "1",
        "fugue_execution_fingerprint": "execution-a",
        "fugue_context_system_id": "gitnexus",
    }
    events = [
        {"event": "gateway_ready", **identity},
        {
            "event": "tool_end",
            "gateway_call_id": "gateway-a",
            "duration_ms": 18.5,
            "is_error": False,
            "vector": {
                "vector_search_attempted": True,
                "vector_search_succeeded": True,
                "semantic_result_count": 4,
                "bm25_result_count": 0,
                "model_digest": "sha256:model",
                "query_latency_ms": 12.5,
            },
            **identity,
        },
        {
            "event": "tool_end",
            "gateway_call_id": "wrong-cell",
            **{**identity, "fugue_candidate_id": "candidate-b"},
        },
    ]
    event_log.write_text("".join(f"{json.dumps(event)}\n" for event in events))

    summary = export._context_event_summary(
        tmp_path / "jobs/job-a/trial-a",
        gateway_event_path=event_log.as_posix(),
        expected_identity={
            "run_id": "run-a",
            "candidate_id": "candidate-a",
            "comparison_example_id": "example-a",
            "trial_index": 1,
            "execution_fingerprint": "execution-a",
            "context_system_id": "gitnexus",
        },
    )

    assert summary["context_gateway_event_log_status"] == "available"
    assert summary["context_gateway_tool_call_count"] == 1
    assert summary["context_gateway_call_ids"] == ["gateway-a"]
    assert summary["context_gateway_identity_mismatch_count"] == 1
    assert summary["gitnexus_vector_search_attempted"] is True
    assert summary["gitnexus_vector_search_succeeded"] is True
    assert summary["gitnexus_semantic_result_count"] == 4
    assert summary["gitnexus_bm25_result_count"] == 0
    assert summary["gitnexus_vector_model_digests"] == ["sha256:model"]
    assert summary["gitnexus_vector_query_latency_ms"] == 12.5

    row = {"context_assigned": True, **summary}
    export._apply_trace_summary(
        row,
        {
            "weave_gateway_tool_call_count": 0,
            "weave_gateway_call_ids": [],
            "gitnexus_vector_search_attempted": False,
            "gitnexus_vector_search_succeeded": False,
            "gitnexus_semantic_result_count": 0,
            "gitnexus_bm25_result_count": 0,
            "gitnexus_vector_model_digests": [],
            "gitnexus_vector_query_latency_ms": 0.0,
        },
    )
    assert row["context_invoked"] is True
    assert row["context_invocation_evidence"] == {
        "status": "observed",
        "source": "mcp_gateway_event_log",
        "tool_calls": 1,
        "gateway_call_ids": ["gateway-a"],
    }
    assert row["gitnexus_vector_search_succeeded"] is True


def test_gateway_event_log_rejects_paths_outside_runtime(tmp_path: Path) -> None:
    event_log = tmp_path / "context-gateway.jsonl"
    event_log.write_text('{"event":"tool_end","gateway_call_id":"a"}\n')

    summary = export._context_event_summary(
        tmp_path / "trial",
        gateway_event_path=event_log.as_posix(),
    )

    assert summary["context_gateway_event_log_status"] == "rejected"
    assert summary["context_gateway_tool_call_count"] == 0


def test_agent_hierarchy_ignores_auxiliary_span_conversation_identity() -> None:
    trace_id = "a" * 32
    root_span_id = "b" * 16
    summary = _summarize_spans(
        [
            {
                "id": root_span_id,
                "trace_id": trace_id,
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.conversation.id": "native-conversation",
                    "weave.eval.predict_and_score_call_id": "prediction-1",
                },
            },
            {
                "id": "chat",
                "trace_id": trace_id,
                "parent_id": root_span_id,
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.conversation.id": "native-conversation",
                },
            },
            {
                "id": "terminal-helper",
                "trace_id": trace_id,
                "parent_id": "chat",
                "attributes": {
                    "gen_ai.operation.name": "tool.terminal",
                    "gen_ai.conversation.id": "planned-conversation",
                },
            },
        ]
    )

    assert summary["weave_conversation_ids"] == ["native-conversation"]
    row = {"trace_id": trace_id, "root_span_id": root_span_id, **summary}
    assert export._verified_evaluation_root(row, "prediction-1") is not None


def test_live_link_rejects_split_native_conversation_identity() -> None:
    row = {
        "trace_id": "a" * 32,
        "root_span_id": "b" * 16,
        "weave_conversation_ids": ["native-root", "split-tool"],
        "weave_root_spans": [
            {
                "conversation_id": "native-root",
                "trace_id": "a" * 32,
                "span_id": "b" * 16,
                "eval_predict_and_score_call_id": "prediction-1",
            }
        ],
    }

    root = export._verified_evaluation_root(row, "prediction-1")

    assert root is None
    assert row["trace_link_status"] == "identity_mismatch"
    assert row["trace_link_error"] == (
        "native trace operations do not share the root conversation identity"
    )


def test_live_link_reports_pre_agent_failure_without_disappearing_root() -> None:
    row = {
        "trace_id": "",
        "root_span_id": "",
        "weave_conversation_ids": [],
        "weave_root_spans": [],
    }

    root = export._verified_evaluation_root(row, "prediction-1")

    assert root is None
    assert row["trace_link_status"] == "missing"
    assert row["trace_link_error"] == (
        "no matching invoke_agent root reached Weave before the link deadline"
    )


def test_current_identity_schema_requires_canonical_candidate_id() -> None:
    with pytest.raises(ValueError, match="missing candidate_id"):
        export._publication_candidates(
            [
                {
                    "identity_schema_version": CANDIDATE_IDENTITY_SCHEMA_VERSION,
                    "record_type": "retrieval",
                    "comparison_example_id": "example-a",
                    "trial_index": 1,
                }
            ]
        )


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


def test_live_evaluation_merge_preserves_fresh_local_measurements() -> None:
    row = {
        "run_key": "run-key",
        "evaluation_scope_id": "scope",
        "context_query_count": 1,
        "evidence_paths": ["src/current.py"],
        "local_error_events": [{"id": "current"}],
    }
    live = {
        "run_key": "run-key",
        "context_query_count": 0,
        "evidence_paths": ["dev/null"],
        "local_error_events": [],
        "trace_id": "trace",
        "evaluation_prediction_latency_sec": 12.0,
    }

    export._merge_live_evaluation_row(row, live)

    assert row["evaluation_scope_id"] == "scope"
    assert row["context_query_count"] == 1
    assert row["evidence_paths"] == ["src/current.py"]
    assert row["local_error_events"] == [{"id": "current"}]
    assert row["trace_id"] == "trace"
    assert row["evaluation_prediction_latency_sec"] == 12.0


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
        execution_fingerprint="execution-a",
        config_path=tmp_path / "config.json",
        result_path=tmp_path / "jobs" / "job" / "result.json",
        command=("harbor", "run"),
        env={
            "FUGUE_DATASET": "fixture/tasks@1",
            "FUGUE_REPOSITORY": "org/repo",
            "FUGUE_BASE_COMMIT": "abc123",
        },
        n_attempts=1,
        expected_evidence_paths=("src/expected.py",),
        evaluation_asset_lock_sha256="e" * 64,
    )
    planned = export._planned_evaluation_row(cell)

    row = export._completed_evaluation_row(
        cell, CellOutcome(cell.id, "passed", returncode=0), planned
    )
    row["citation_correctness"] = 0.0
    row["evidence_recall"] = 0.0

    assert row["task_name"] == "task-a"
    assert row["dataset"] == "fixture/tasks@1"
    assert row["comparison_example_id"] == "example-a"
    assert "expected_evidence_paths" not in row
    assert row["evaluation_asset_lock_sha256"] == "e" * 64
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
        execution_fingerprint="execution-a",
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
    changed_scope = export._publication_candidates([changed])[0]["evaluation_scope_id"]
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
        execution_fingerprint="execution-a",
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


def test_completed_evaluation_recovers_setup_failure_and_fingerprint(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "jobs" / "job" / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "agent" / "runtime-fingerprint-pre_install.json").write_text(
        json.dumps({"stage": "pre_install", "comparable_digest": "runtime-a"})
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "swe-bench/task-a",
                "trial_name": "trial",
                "exception_info": {
                    "exception_type": "NonZeroAgentExitCodeError",
                    "exception_message": "installer exited with code 1",
                    "exception_traceback": "trial.py in _setup_agent\nhermes.py in install",
                },
            }
        )
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
        execution_fingerprint="execution-a",
        config_path=tmp_path / "config.json",
        result_path=tmp_path / "jobs" / "job" / "result.json",
        command=("harbor", "run"),
        env={"FUGUE_EXPERIMENT_ID": "experiment-a"},
        n_attempts=1,
    )
    planned = export._planned_evaluation_row(cell)

    row = export._completed_evaluation_row(
        cell,
        CellOutcome(cell.id, "failed", returncode=1, error="trial failed"),
        planned,
    )
    export._merge_error_events(row)

    assert row["run_id"] == "run-a"
    assert row["candidate_id"] == "candidate-a"
    assert row["harness"] == "hermes"
    assert row["runtime_fingerprints"]["pre_install"]["comparable_digest"] == (
        "runtime-a"
    )
    assert row["harness_adapter_error_count"] == 1
    assert row["error_events"][0]["terminal"] is True
    assert row["adapter_outcome"]["execution"]["state"] == "failed"
    assert row["adapter_outcome"]["deterministic_verification"]["state"] == ("unscored")
    assert row["adapter_outcome"]["exploratory_tools"]["state"] == "clean"


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
        [
            {
                "record_type": "trial",
                "task_name": "task-a",
                "run_id": "run-a",
                "candidate_id": "candidate-a",
                "comparison_example_id": "example-a",
                "trial_index": 1,
            }
        ],
        f"entity/project-{tmp_path.name}",
        ledger_root=tmp_path,
        env={"WANDB_API_KEY": "test-only"},
    )

    assert result.published == 0
    assert result.failures and "summary failed" in result.failures[0]
    assert isinstance(failed[0], RuntimeError)
    assert not list((tmp_path / "v1").glob("**/*.json"))


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
        "run_id": "run-a",
        "candidate_id": "candidate-a",
        "comparison_example_id": "example-a",
        "trial_index": 1,
    }
    with pytest.raises(ValueError, match="duplicate evaluation trial"):
        publish_to_weave(
            [row, dict(row)],
            f"entity/project-{tmp_path.name}",
            ledger_root=tmp_path,
            env={"WANDB_API_KEY": "test-only"},
        )


def test_publication_ledger_rejects_prediction_overlap_across_evaluations(
    tmp_path: Path, monkeypatch
) -> None:
    published: list[str] = []

    class FakeLogger:
        ui_url = "https://wandb.invalid/evaluation"

        def __init__(self, **kwargs) -> None:
            self._pseudo_evaluation = None
            self.model = None

        def log_example(self, inputs, output, scores) -> None:
            published.append(inputs["comparison_example_id"])

        def log_summary(self) -> None:
            pass

        def fail(self, exception) -> None:
            raise AssertionError(exception)

    monkeypatch.setattr(
        export,
        "initialize_weave",
        lambda project, env: SimpleNamespace(EvaluationLogger=FakeLogger),
    )
    common = {
        "schema_version": 1,
        "prediction_schema_version": 1,
        "record_type": "trial",
        "run_id": "run-a",
        "candidate_id": "candidate-a",
        "workload_id": "retrieval",
        "dataset": "fixture",
        "execution_kind": "provider_diagnostic",
        "trial_index": 1,
        "status": "passed",
    }
    first_row = {
        **common,
        "prediction_id": "prediction-a",
        "comparison_example_id": "example-a",
        "task_name": "task-a",
    }
    second_row = {
        **common,
        "prediction_id": "prediction-b",
        "comparison_example_id": "example-b",
        "task_name": "task-b",
    }

    first = publish_to_weave(
        [first_row],
        "entity/project",
        ledger_root=tmp_path,
        env={"WANDB_API_KEY": "test-only"},
    )
    overlapping = publish_to_weave(
        [first_row, second_row],
        "entity/project",
        ledger_root=tmp_path,
        env={"WANDB_API_KEY": "test-only"},
    )

    assert first.published == 1
    assert overlapping.published == 0
    assert "already published" in overlapping.failures[0]
    assert published == ["example-a"]


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
        "trace_link_error": (
            "no matching invoke_agent root reached Weave before the link deadline"
        ),
    }

    export._apply_observed_identity(row)

    assert row["trace_link_status"] == "not_applicable"
    assert row["trace_link_error"] is None
    assert row["weave_observability_status"] == "not_applicable"
    assert row["weave_usage_status"] == "not_applicable"


def test_native_chat_response_fills_full_trace_output_but_metadata_only_hashes() -> (
    None
):
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


def test_error_provenance_distinguishes_agent_runtime_and_adapter_failures() -> None:
    cases = [
        ("'content' must be a string, got dict", "write_file", "agent"),
        ("tool_result_error", "exec", "agent"),
        ("ModuleNotFoundError: No module named 'erfa'", "Bash", "benchmark_runtime"),
        (
            "web_search is disabled: no provider configured",
            "web_search",
            "harness_adapter",
        ),
        (
            "unknown variant `namespace`, expected `function`",
            "",
            "provider",
        ),
    ]

    for message, tool_name, origin in cases:
        event = export._classify_error(
            message,
            tool_name=tool_name,
            operation="execute_tool",
            source="test",
        )
        assert event["origin"] == origin
        assert event["recoverable"] is True


def test_terminal_harness_install_failure_is_owned_by_adapter() -> None:
    event = export._terminal_exception_event(
        {
            "exception_type": "NonZeroAgentExitCodeError",
            "exception_message": "installer exited with code 1",
            "exception_traceback": (
                "harbor/trial/trial.py in _setup_agent\n"
                "harbor/agents/installed/hermes.py in install"
            ),
        }
    )

    assert event is not None
    assert event["origin"] == "harness_adapter"
    assert event["kind"] == "integration_failure"
    assert event["terminal"] is True
    assert event["recoverable"] is False


def test_error_events_merge_native_and_weave_occurrences_without_double_counting() -> (
    None
):
    row: dict[str, object] = {
        "weave_error_events": [
            export._classify_error(
                "command failed with exit code 1",
                tool_name="bash",
                operation="execute_tool",
                source="weave_span",
                event_key=f"span-{index}",
            )
            for index in range(2)
        ],
        "local_error_events": [
            export._classify_error(
                "tool reported failure",
                tool_name="bash",
                operation="execute_tool",
                source="local_trajectory",
                event_key=f"call-{index}",
            )
            for index in range(2)
        ],
    }

    export._merge_error_events(row)

    assert len(row["error_events"]) == 2
    assert row["agent_error_count"] == 2
    assert row["recoverable_error_count"] == 2


def test_failed_context_registration_is_not_reported_as_available(
    tmp_path: Path,
) -> None:
    jobs = _write_export_fixture(tmp_path)
    meta_path = next(jobs.rglob("fugue-meta.json"))
    meta = json.loads(meta_path.read_text())
    meta["context_registration"] = {
        "status": "failed",
        "transport": "portable",
        "error": "probe unavailable",
    }
    meta_path.write_text(json.dumps(meta))

    [row] = export_rows([jobs])

    assert row["context_assigned"] is True
    assert row["context_registered"] is False
    assert row["context_available"] is False


def test_trajectory_errors_and_evidence_are_collected_without_agent_artifact(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "tool_calls": [
                            {
                                "tool_call_id": "read",
                                "function_name": "read_file",
                                "arguments": {
                                    "path": "/testbed/src/app.py",
                                    "command": (
                                        "cat /dev/null /tmp/ignore.txt "
                                        "/testbed/src/other.py"
                                    ),
                                },
                            },
                            {
                                "tool_call_id": "write",
                                "function_name": "write_file",
                                "arguments": {"path": "src/app.py"},
                            },
                        ],
                        "observation": {
                            "results": [
                                {
                                    "source_call_id": "write",
                                    "content": "'content' must be a string, got dict",
                                    "extra": {"tool_result_is_error": True},
                                }
                            ]
                        },
                    }
                ]
            }
        )
    )

    activity = export._trajectory_activity(trial)

    assert activity["inspected_paths"] == ["src/app.py", "src/other.py"]
    assert activity["changed_paths"] == ["src/app.py"]
    assert activity["error_events"][0]["kind"] == "invalid_tool_arguments"


def test_runtime_equivalence_is_computed_within_comparison_cohort() -> None:
    rows = [
        {
            "record_type": "trial",
            "run_id": "run",
            "comparison_example_id": "example",
            "trial_index": 1,
            "model": "wandb/model",
            "runtime_fingerprints": {"pre_install": {"comparable_digest": digest}},
        }
        for digest in ("same", "same")
    ]
    export._apply_runtime_equivalence(rows)
    assert all(row["runtime_equivalent"] is True for row in rows)

    rows[1]["runtime_fingerprints"]["pre_install"]["comparable_digest"] = "other"
    export._apply_runtime_equivalence(rows)
    assert all(row["runtime_equivalence_status"] == "mismatch" for row in rows)


def test_prepared_runtime_equivalence_and_in_trial_drift_are_separate() -> None:
    rows = [
        {
            "record_type": "trial",
            "run_id": "run",
            "comparison_example_id": "example",
            "trial_index": 1,
            "model": "wandb/model",
            "runtime_fingerprints": {
                "pre_execution": {"comparable_digest": "prepared"},
                "post_execution": {"comparable_digest": post},
            },
        }
        for post in ("prepared", "drifted")
    ]

    export._apply_runtime_equivalence(rows)

    assert all(row["runtime_equivalent"] is True for row in rows)
    assert rows[0]["runtime_drift"] is False
    assert rows[1]["runtime_drift"] is True


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
        "fugue.context_delivery": "portable",
        "fugue.context_registration_status": "registered",
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


def test_weave_enrichment_marks_expected_agent_identity(
    monkeypatch, tmp_path: Path
) -> None:
    jobs = _write_export_fixture(tmp_path)

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
