import json
import sys
import threading
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import httpx
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
    _merge_planned_cell_identities,
    _summarize_spans,
    _synthesize_missing_trial_rows,
    compile_export,
    export_rows,
    publish_to_weave,
    write_jsonl,
)
from fugue.bench.operator import OperatorService
from fugue.mcp_evidence import safe_graphql_event_arguments


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
                "model_transport": {
                    "harness": "hermes",
                    "wire_protocol": "chat_completions",
                    "endpoint_kind": "provider_direct",
                    "upstream_host": "api.inference.wandb.ai",
                    "bridge_required": False,
                },
                "trace_project": "test/fugue",
                "agent_runtime": {
                    "image": "fugue-agent:locked",
                    "image_id": "sha256:" + "a" * 64,
                },
                "task_runtime": {
                    "image": "fugue-task:locked",
                    "image_id": "sha256:" + "b" * 64,
                },
                "sandbox_attestation": {
                    "schema_version": 1,
                    "attestation_digest": "c" * 64,
                },
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
    assert row["model_transport"]["wire_protocol"] == "chat_completions"
    assert row["model_transport"]["endpoint_kind"] == "provider_direct"
    assert row["trace_project"] == "test/fugue"
    assert row["agent_runtime_image_id"] == "sha256:" + "a" * 64
    assert row["task_runtime_image_id"] == "sha256:" + "b" * 64
    assert row["sandbox_attestation_digest"] == "c" * 64
    assert row["sandbox_attestation"]["schema_version"] == 1
    assert "privacy_scan_complete" not in row
    assert "sandbox_cleanup_verified" not in row
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
            init=lambda project, **kwargs: calls.append(
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
            "fugue.variant_id": "rag-bm25",
            "fugue.candidate_id": "candidate-1",
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
        SimpleNamespace(
            init=lambda project, **kwargs: None,
            EvaluationLogger=FakeLogger,
        ),
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
        project="entity/project",
        url="https://wandb.test/evaluations/live",
        agent_predictions=1,
        linked_agent_predictions=1,
        linking_failures=("agent link reason remains visible",),
    )
    direct = PublishedEvaluation(
        candidate_id="candidate-direct",
        name="memory | continuity | direct-scope",
        examples=4,
        project="entity/project",
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
    monkeypatch.setattr(
        operator,
        "compile_export",
        lambda *args, **kwargs: SimpleNamespace(
            predictions=(), measurements=(), publication=next(publications)
        ),
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

    def fake_compile_export(paths, **kwargs):
        observed.extend(paths)
        return SimpleNamespace(predictions=(), measurements=(), publication=None)

    monkeypatch.setattr(operator, "compile_export", fake_compile_export)

    OperatorService(tmp_path).export_run(run_id)

    assert observed == [selected, tmp_path / ".fugue" / "runtime" / run_id]
    assert unrelated not in observed


def test_compile_export_normalizes_before_publication(
    tmp_path: Path, monkeypatch
) -> None:
    raw = {
        "record_type": "retrieval",
        "run_id": "run-a",
        "candidate_id": "candidate-a",
        "comparison_example_id": "example-a",
        "trial_index": 1,
        "execution_kind": "provider_diagnostic",
        "execution_fingerprint": "fingerprint-a",
    }
    cell = {
        **raw,
        "record_type": "cell",
        "status": "passed",
    }
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(export, "export_rows", lambda *args, **kwargs: [raw, cell])

    def publish(rows, *args, **kwargs):
        observed.extend(rows)
        return PublicationResult(published=1, skipped=0)

    monkeypatch.setattr(export, "publish_to_weave", publish)

    bundle = compile_export([tmp_path], publish=True)

    assert len(bundle.predictions) == len(bundle.measurements) == 1
    assert observed == list(bundle.predictions)
    assert observed[0]["prediction_id"]


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
    monkeypatch.setattr(export, "export_rows", lambda *args, **kwargs: rows)

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
        project="entity/project",
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
    monkeypatch.setattr(
        operator,
        "compile_export",
        lambda *args, **kwargs: SimpleNamespace(
            predictions=(),
            measurements=(),
            publication=PublicationResult(0, 0, failures=(failure,)),
        ),
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
            init=lambda project, **kwargs: None,
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
    assert {logger.name for logger in loggers} == {
        "memory-ab | coding | codex | none",
        "memory-ab | coding | codex | rag-bm25",
    }
    assert len(
        {
            logger.eval_attributes["fugue.evaluation_scope_id"]
            for logger in loggers
        }
    ) == 1
    assert {
        logger.eval_attributes["fugue.candidate_id"] for logger in loggers
    } == {"candidate-none", "candidate-rag"}
    assert loggers[0].scorers == loggers[1].scorers
    assert {logger.model["name"] for logger in loggers} == {
        "codex__none__test-model",
        "codex__rag-bm25__test-model",
    }
    assert {logger.model["variant_id"] for logger in loggers} == {
        "none",
        "rag-bm25",
    }


def test_weave_predefined_scorer_names_match_string_scorer_normalization() -> None:
    assert export._weave_predefined_scorer_names(
        [
            "comparison.deterministic.answer_present",
            "comparison_deterministic_expected_values",
            "3rd-party.score",
        ]
    ) == [
        "comparisondeterministicanswer_present",
        "comparison_deterministic_expected_values",
        "C3rdpartyscore",
    ]


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
            init=lambda project, **kwargs: None,
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
            self.ref = SimpleNamespace(
                uri="weave:///entity/project/object/tasks:dataset-v1"
            )

    class FakePrediction:
        def __init__(self, call_id: str, evaluate_call) -> None:
            self.evaluate_call = evaluate_call
            self.predict_and_score_call = SimpleNamespace(
                id=call_id,
                project_id="entity/project",
                parent_id=evaluate_call.id,
                trace_id=evaluate_call.trace_id,
                summary=None,
                ref=SimpleNamespace(
                    uri=f"weave:///entity/project/call/{call_id}"
                ),
                ui_url=f"https://wandb.test/calls/{call_id}",
            )
            self.predict_call = SimpleNamespace(
                id=f"{call_id}-model",
                project_id="entity/project",
                parent_id=call_id,
                trace_id=evaluate_call.trace_id,
                ref=SimpleNamespace(
                    uri=f"weave:///entity/project/call/{call_id}-model"
                ),
                ui_url=f"https://wandb.test/calls/{call_id}-model",
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
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)
            self.ui_url = None
            self._pseudo_evaluation = SimpleNamespace(
                dataset=self.dataset,
                ref=SimpleNamespace(
                    uri="weave:///entity/project/object/eval:shared"
                ),
            )
            self._evaluate_call = SimpleNamespace(
                id="evaluation-root-1",
                project_id="entity/project",
                parent_id=None,
                trace_id="a" * 32,
                inputs={"self": self._pseudo_evaluation},
                ref=SimpleNamespace(
                    uri="weave:///entity/project/call/evaluation-root-1"
                ),
            )
            self.summarized = False
            loggers.append(self)

        def log_prediction(self, inputs):
            prediction = FakePrediction(
                f"predict-{len(predictions) + 1}",
                self._evaluate_call,
            )
            predictions.append(prediction)
            return prediction

        def log_summary(self) -> None:
            self.summarized = True
            self.ui_url = "https://wandb.test/evaluations/live"

        def fail(self, exception) -> None:
            raise AssertionError(exception)

    client = _OutOfOrderWeaveClient("entity/project")
    fake_weave = SimpleNamespace(
        Dataset=FakeDataset,
        EvaluationLogger=FakeLogger,
        get_client=lambda: client,
    )
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
        bridge_ids = [
            call_id_value
            for call_id_value, call in client.remote.items()
            if call.get("parent_id") == f"{call_id}-model"
        ]
        bridge_id = bridge_ids[0] if bridge_ids else ""
        agent_ids = [value for value in client.remote if value not in bridge_ids]
        agent_call_id = agent_ids[0] if agent_ids else ""
        root = {
            "conversation_id": "native-conversation",
            "agent_name": "codex",
            "trace_id": "a" * 32,
            "span_id": "b" * 16,
            "otel_parent_span_id": str(
                (client.remote.get(bridge_id, {}).get("attributes") or {}).get(
                    "fugue.agent_bridge_otel_parent_span_id"
                )
                or ""
            ),
            "run_key": ("run-a:coding:trial:task-a:codex:rag-bm25:rag-bm25:t001"),
            "harness": "codex",
            "task_id": "task-a",
            "candidate_id": "candidate-a",
            "attempt_id": cell.attempt_id,
            "execution_fingerprint": "execution-a",
            "comparison_example_id": "example-a",
            "trial_index": 1,
            "eval_predict_and_score_call_id": call_id,
        }
        if agent_call_id:
            root.update(
                {
                    "weave_call_id": agent_call_id,
                    "weave_call_parent_id": bridge_id,
                    "weave_call_project_id": "entity/project",
                    "weave_call_trace_id": "a" * 32,
                    "weave_call_ref": (f"weave:///entity/project/call/{agent_call_id}"),
                    "weave_call_url": (
                        f"https://wandb.ai/entity/project/weave/calls/{agent_call_id}"
                    ),
                }
            )
        graph = [
            {
                "call_id": "evaluation-root-1",
                "project_id": "entity/project",
                "trace_id": "a" * 32,
                "terminal": True,
            },
            {
                "call_id": call_id,
                "parent_id": "evaluation-root-1",
                "project_id": "entity/project",
                "trace_id": "a" * 32,
                "terminal": True,
            },
            {
                "call_id": f"{call_id}-model",
                "parent_id": call_id,
                "project_id": "entity/project",
                "trace_id": "a" * 32,
                "terminal": True,
            },
            *(dict(value) for value in client.remote.values()),
        ]
        return {
            next(iter(kwargs["run_keys"])): {
                "weave_agent_names": ["codex"],
                "weave_conversation_ids": ["native-conversation"],
                "weave_trace_ids": ["a" * 32],
                "weave_root_span_ids": ["b" * 16],
                "weave_root_spans": [root],
                "weave_authoritative_call_graph": graph,
                "weave_authoritative_missing_call_ids": [],
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
        host_evaluator=lambda row: row.update(
            {
                "comparison_evaluation_status": "scored",
                "comparison_required_evaluation_complete": True,
                "comparison_deterministic_scores": {
                    "fact-correct": True,
                    "citation-quality": 0.75,
                },
            }
        ),
        host_scorer_names=(
            "comparison.deterministic.fact-correct",
            "comparison.deterministic.citation-quality",
        ),
    )
    overlay = coordinator.begin_cell(cell)
    assert overlay is not None
    traceparent = overlay.pop("FUGUE_WEAVE_TRACEPARENT")
    assert traceparent.startswith(f"00-{'a' * 32}-")
    assert traceparent.endswith("-01")
    assert overlay == {
        "FUGUE_ATTEMPT_ID": cell.attempt_id,
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
    assert live_row["eval_predict_and_score_call_id"] == "predict-1"
    assert live_row["eval_predict_and_score_ref"] == (
        "weave:///entity/project/call/predict-1"
    )
    assert live_row["eval_predict_and_score_url"] == (
        "https://wandb.ai/entity/project/weave/calls/predict-1"
    )
    assert live_row["weave_prediction_call_id"] == "predict-1-model"
    assert live_row["weave_prediction_ref"] == (
        "weave:///entity/project/call/predict-1-model"
    )
    assert live_row["weave_prediction_url"] == (
        "https://wandb.ai/entity/project/weave/calls/predict-1-model"
    )
    assert live_row["weave_evaluation_id"] == (
        "weave:///entity/project/object/eval:shared"
    )
    assert live_row["weave_evaluation_url"] == (
        "https://wandb.ai/entity/project/weave/calls/evaluation-root-1"
    )
    assert live_row["weave_dataset_id"] == (
        "weave:///entity/project/object/tasks:dataset-v1"
    )
    assert live_row["weave_dataset_url"] == (
        "https://wandb.ai/entity/project/weave/objects/tasks/versions/dataset-v1"
    )
    assert live_row["otel_trace_id"] == "a" * 32
    assert live_row["otel_root_span_id"] == "b" * 16
    agent_call_id = live_row["weave_agent_root_call_id"]
    assert len(agent_call_id) == 36
    assert str(export.uuid.UUID(agent_call_id)) == agent_call_id
    assert live_row["weave_agent_root_ref"] == (
        f"weave:///entity/project/call/{agent_call_id}"
    )
    assert live_row["weave_agent_root_url"] == (
        f"https://wandb.ai/entity/project/weave/calls/{agent_call_id}"
    )
    assert live_row["weave_agent_root_call_id"] != live_row["otel_root_span_id"]
    assert live_row["evaluation_root_object_verified"] is True
    assert live_row["dataset_version_object_verified"] is True
    assert live_row["eval_predict_and_score_object_verified"] is True
    assert live_row["weave_prediction_object_verified"] is True
    assert live_row["evaluation_prediction_graph_verified"] is True
    assert live_row["agent_graph_verified"] is True
    assert live_row["evaluation_judge_status"] == "not_requested"
    assert predictions[0].scores["comparison.deterministic.fact-correct"] is True
    assert (
        predictions[0].scores["comparison.deterministic.citation-quality"]
        == 0.75
    )
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


@pytest.mark.parametrize(
    "drift",
    (
        "evaluation_input",
        "dataset_owner",
        "predict_and_score_parent",
        "prediction_parent",
        "prediction_project",
        "prediction_trace",
    ),
)
def test_live_evaluation_graph_rejects_navigation_only_or_wrong_ancestry(
    drift: str,
) -> None:
    project = "entity/project"
    dataset = SimpleNamespace(
        ref=SimpleNamespace(
            uri="weave:///entity/project/object/tasks:dataset-v1"
        )
    )
    evaluation = SimpleNamespace(
        dataset=dataset,
        ref=SimpleNamespace(uri="weave:///entity/project/object/eval:v1"),
    )
    evaluation_call = SimpleNamespace(
        id="evaluation-root",
        project_id=project,
        parent_id=None,
        trace_id="weave-trace",
        inputs={"self": evaluation},
        ref=SimpleNamespace(
            uri="weave:///entity/project/call/evaluation-root"
        ),
    )
    predict_and_score = SimpleNamespace(
        id="predict-and-score",
        project_id=project,
        parent_id=evaluation_call.id,
        trace_id=evaluation_call.trace_id,
        ref=SimpleNamespace(
            uri="weave:///entity/project/call/predict-and-score"
        ),
        ui_url="https://wandb.test/calls/predict-and-score",
    )
    predict = SimpleNamespace(
        id="prediction",
        project_id=project,
        parent_id=predict_and_score.id,
        trace_id=evaluation_call.trace_id,
        ref=SimpleNamespace(uri="weave:///entity/project/call/prediction"),
        ui_url="https://wandb.test/calls/prediction",
    )
    logger = SimpleNamespace(
        _pseudo_evaluation=evaluation,
        _evaluate_call=evaluation_call,
        ui_url="https://wandb.test/calls/evaluation-root",
    )
    prediction = SimpleNamespace(
        evaluate_call=evaluation_call,
        predict_and_score_call=predict_and_score,
        predict_call=predict,
    )
    if drift == "evaluation_input":
        evaluation_call.inputs = {"self": SimpleNamespace(dataset=dataset)}
    elif drift == "dataset_owner":
        evaluation.dataset = SimpleNamespace(
            ref=SimpleNamespace(
                uri="weave:///entity/project/object/other-dataset:v1"
            )
        )
    elif drift == "predict_and_score_parent":
        predict_and_score.parent_id = "other-evaluation"
    elif drift == "prediction_parent":
        predict.parent_id = "other-prediction"
    elif drift == "prediction_project":
        predict.project_id = "entity/other"
    else:
        predict.trace_id = "other-trace"

    row: dict[str, object] = {}
    export._apply_call_evidence(
        row,
        prefix="eval_predict_and_score",
        call=predict_and_score,
        project=project,
    )
    export._apply_call_evidence(
        row,
        prefix="weave_prediction",
        call=predict,
        project=project,
    )
    export._verify_live_evaluation_graph(
        row,
        logger=logger,
        dataset=dataset,
        prediction=prediction,
        project=project,
    )

    assert row["evaluation_prediction_graph_verified"] is False
    assert row["evaluation_prediction_graph_error"]


def test_claude_live_evaluation_opens_real_otel_parent_bridge() -> None:
    project = "entity/project"
    trace_id = "a" * 32
    prediction_call = SimpleNamespace(
        id="prediction-call",
        project_id=project,
        parent_id="predict-and-score",
        trace_id=trace_id,
    )
    created: list[dict[str, object]] = []
    finished: list[tuple[object, object]] = []

    class FakeClient:
        def create_call(
            self,
            op,
            inputs,
            parent=None,
            attributes=None,
            display_name=None,
            *,
            use_stack=True,
            _call_id_override=None,
        ):
            created.append(
                {
                    "op": op,
                    "inputs": inputs,
                    "parent": parent,
                    "attributes": attributes,
                    "display_name": display_name,
                    "use_stack": use_stack,
                    "id": _call_id_override,
                }
            )
            return SimpleNamespace(
                id=_call_id_override,
                parent_id=parent.id,
                project_id=project,
                trace_id=parent.trace_id,
                ref=SimpleNamespace(
                    uri=f"weave:///entity/project/call/{_call_id_override}"
                ),
                ui_url=f"https://wandb.test/calls/{_call_id_override}",
            )

        def finish_call(self, call, output=None):
            finished.append((call, output))

    client = FakeClient()
    coordinator = object.__new__(LiveEvaluationCoordinator)
    coordinator.project = project
    coordinator.weave = SimpleNamespace(get_client=lambda: client)
    cell = PlannedCell(
        id="cell-bridge",
        run_id="run-bridge",
        run_name="bridge",
        workload_id="harbor",
        task_id="task-bridge",
        harness="claude-code",
        context_system_id="none",
        variant_id="candidate",
        model_provider="anthropic",
        model="anthropic/claude-sonnet-5",
        trial_index=1,
        comparison_example_id="example-bridge",
        candidate_id="candidate-bridge",
        execution_fingerprint="runtime-bridge",
        config_path=Path("config.yaml"),
        result_path=Path("jobs/result.json"),
        command=("harbor", "run"),
        env={},
        n_attempts=1,
    )
    row: dict[str, object] = {}
    bridge, bridge_client, traceparent = coordinator._open_agent_bridge(
        cell=cell,
        row=row,
        prediction=SimpleNamespace(predict_call=prediction_call),
    )

    assert bridge_client is client
    assert created[0]["parent"] is prediction_call
    assert created[0]["use_stack"] is False
    assert len(str(created[0]["id"])) == 36
    assert str(export.uuid.UUID(str(created[0]["id"]))) == created[0]["id"]
    bridge_parent_span = str(row["weave_agent_bridge_otel_span_id"])
    assert traceparent == f"00-{trace_id}-{bridge_parent_span}-01"
    assert bridge.id != bridge_parent_span
    assert row["weave_agent_bridge_parent_id"] == prediction_call.id
    assert row["weave_agent_bridge_trace_id"] == trace_id
    assert row["weave_agent_bridge_object_verified"] is True

    active = export._LivePrediction(
        session=SimpleNamespace(),
        prediction=SimpleNamespace(),
        bridge_call=bridge,
        bridge_client=client,
        row=row,
        opened_monotonic=0.0,
    )
    terminal_row: dict[str, object] = {}
    coordinator._finish_agent_bridge(
        active,
        status="linked",
        terminal_row=terminal_row,
    )
    coordinator._finish_agent_bridge(
        active,
        status="linked",
        terminal_row=terminal_row,
    )
    assert finished == [(bridge, {"status": "linked"})]
    assert row["weave_agent_bridge_close_status"] == "linked"
    assert row["weave_agent_bridge_closed_verified"] is True
    assert terminal_row["weave_agent_bridge_close_status"] == "linked"
    assert terminal_row["weave_agent_bridge_closed_verified"] is True

    class FailingClient:
        @staticmethod
        def finish_call(call, output=None):
            raise RuntimeError("transport unavailable")

    failed_row: dict[str, object] = {}
    failed_active = export._LivePrediction(
        session=SimpleNamespace(),
        prediction=SimpleNamespace(),
        bridge_call=bridge,
        bridge_client=FailingClient(),
        row=failed_row,
        opened_monotonic=0.0,
    )
    coordinator._finish_agent_bridge(failed_active, status="cancelled")
    assert failed_row["weave_agent_bridge_closed_verified"] is False
    assert failed_row["weave_agent_bridge_close_error"] == "RuntimeError"


class _OutOfOrderWeaveClient:
    """Model Weave's async start/end queues with hostile flush ordering."""

    def __init__(self, project: str, *, drop_ends: bool = False) -> None:
        self.project_id = project
        self.drop_ends = drop_ends
        self.pending: list[tuple[str, object, object]] = []
        self.remote: dict[str, dict[str, object]] = {}
        self.events: list[str] = []

    def create_call(
        self,
        op,
        inputs,
        parent=None,
        attributes=None,
        display_name=None,
        *,
        use_stack=True,
        _call_id_override=None,
    ):
        call = SimpleNamespace(
            id=_call_id_override,
            parent_id=parent.id,
            project_id=self.project_id,
            trace_id=parent.trace_id,
            op_name=str(op),
            attributes=attributes or {},
            inputs=inputs,
        )
        self.pending.append(("start", call, None))
        self.events.append(f"start-scheduled:{call.id}")
        return call

    def finish_call(self, call, output=None):
        self.pending.append(("end", call, output))
        self.events.append(f"end-scheduled:{call.id}")

    def flush(self) -> None:
        pending, self.pending = self.pending, []
        # If start and end are flushed together, process the end first. That
        # reproduces the SDK race which dropped the V5 terminal update.
        for kind, call, output in reversed(pending):
            if kind == "start":
                self.remote[call.id] = {
                    "call_id": call.id,
                    "parent_id": call.parent_id,
                    "project_id": call.project_id,
                    "trace_id": call.trace_id,
                    "terminal": False,
                    "attributes": dict(call.attributes),
                }
                self.events.append(f"start-published:{call.id}")
            elif self.drop_ends or call.id not in self.remote:
                self.events.append(f"end-dropped:{call.id}")
            else:
                self.remote[call.id]["terminal"] = True
                self.remote[call.id]["output"] = output
                self.events.append(f"end-published:{call.id}")


def _out_of_order_summary_fetcher(client: _OutOfOrderWeaveClient):
    def fetcher(
        *,
        run_keys,
        conversation_ids_by_run,
        call_ids_by_run,
        project,
        timeout_sec,
        env,
    ):
        del conversation_ids_by_run, timeout_sec, env
        assert project == client.project_id
        result = {}
        for run_key in run_keys:
            call_ids = call_ids_by_run.get(run_key, [])
            result[run_key] = {
                "weave_authoritative_call_graph": [
                    dict(client.remote[call_id])
                    for call_id in call_ids
                    if call_id in client.remote
                ]
            }
        return result

    return fetcher


def test_weave_summary_exposes_an_open_call_as_nonterminal() -> None:
    project = "entity/project"
    call_id = "open-call"
    summary = _summarize_spans(
        [
            {
                "id": call_id,
                "project_id": project,
                "trace_id": "a" * 32,
                "op_name": "fugue.agent_execution_bridge",
                "_fugue_evidence_source": export._WEAVE_CALL_SOURCE,
                "_fugue_query_project": project,
            }
        ],
        project=project,
        required_call_ids=[call_id],
    )

    assert summary["weave_authoritative_call_graph"] == [
        {
            "call_id": call_id,
            "project_id": project,
            "trace_id": "a" * 32,
            "op_name": "fugue.agent_execution_bridge",
            "terminal": False,
        }
    ]
    assert summary["weave_authoritative_missing_call_ids"] == []


def test_claude_call_lifecycle_flushes_start_before_end_across_a_b_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_a = "entity/project-a"
    project_b = "entity/project-b"
    client_a = _OutOfOrderWeaveClient(project_a)
    client_b = _OutOfOrderWeaveClient(project_b)
    clients = {project_a: client_a, project_b: client_b}
    active = {"client": client_b}
    activations: list[str] = []
    weave_module = SimpleNamespace(get_client=lambda: active["client"])

    def activate(project, env):
        del env
        activations.append(project)
        active["client"] = clients[project]
        return weave_module

    monkeypatch.setattr(export, "initialize_weave", activate)
    coordinator = object.__new__(LiveEvaluationCoordinator)
    coordinator.project = project_a
    coordinator.env = {}
    coordinator.weave = weave_module
    coordinator._weave_requires_reactivation = True
    coordinator._evidence_destination = {"project_slug": project_a}
    coordinator._summary_fetcher = _out_of_order_summary_fetcher(client_a)
    coordinator.trace_timeout_sec = 0
    coordinator._cancellation_event = threading.Event()
    coordinator._agent_bridge_op = "fugue.agent_execution_bridge"
    coordinator._native_agent_root_op = "fugue.claude_code_native_agent_root"
    trace_id = "a" * 32
    prediction_call = SimpleNamespace(
        id="prediction-call",
        project_id=project_a,
        parent_id="predict-and-score",
        trace_id=trace_id,
    )
    cell = PlannedCell(
        id="cell-a",
        run_id="run-a",
        run_name="comparison-a",
        workload_id="harbor",
        task_id="task-a",
        harness="claude-code",
        context_system_id="none",
        variant_id="candidate",
        model_provider="anthropic",
        model="anthropic/claude-sonnet-5",
        trial_index=1,
        comparison_example_id="example-a",
        candidate_id="candidate-a",
        execution_fingerprint="runtime-a",
        config_path=Path("config.yaml"),
        result_path=Path("jobs/result.json"),
        command=("harbor", "run"),
        env={},
        n_attempts=1,
    )
    row = {
        "run_id": "run-a",
        "cell_id": "cell-a",
        "run_key": "run-a:harbor:trial:task-a:claude-code:none:candidate:t001",
        "harness": "claude-code",
        "task_id": "task-a",
        "candidate_id": "candidate-a",
        "attempt_id": "b" * 64,
        "execution_fingerprint": "e" * 64,
        "comparison_example_id": "example-a",
        "trial_index": 1,
        "eval_predict_and_score_call_id": "predict-and-score",
        "trace_project": project_a,
        "status": "passed",
        "agent_response_sha256": "f" * 64,
    }

    bridge, bridge_client, _ = coordinator._open_agent_bridge(
        cell=cell,
        row=row,
        prediction=SimpleNamespace(predict_call=prediction_call),
    )
    assert row["weave_agent_bridge_start_verified"] is True
    activate(project_b, {})
    active_prediction = export._LivePrediction(
        session=SimpleNamespace(),
        prediction=SimpleNamespace(),
        bridge_call=bridge,
        bridge_client=bridge_client,
        row=row,
        opened_monotonic=0.0,
    )
    root = {
        "conversation_id": "native-session",
        "trace_id": trace_id,
        "span_id": "c" * 16,
    }

    coordinator._materialize_native_agent_call(
        active=active_prediction,
        row=row,
        root=root,
    )
    activate(project_b, {})
    coordinator._finish_agent_bridge(
        active_prediction,
        status="agent_completed",
        terminal_row=row,
    )

    native_id = str(row["weave_agent_root_call_id"])
    assert activations == [
        project_a,
        project_b,
        project_a,
        project_b,
        project_a,
    ]
    assert client_a.remote[native_id]["terminal"] is True
    assert client_a.remote[bridge.id]["terminal"] is True
    assert row["weave_agent_root_call_start_verified"] is True
    assert row["weave_agent_root_call_terminal_verified"] is True
    assert row["weave_agent_bridge_closed_verified"] is True
    assert f"start-published:{native_id}" in client_a.events
    assert client_a.events.index(f"start-published:{native_id}") < (
        client_a.events.index(f"end-published:{native_id}")
    )
    assert f"end-dropped:{native_id}" not in client_a.events
    assert client_b.remote == {}


def test_claude_native_call_rejects_a_missing_remote_terminal_end() -> None:
    project = "entity/project"
    client = _OutOfOrderWeaveClient(project, drop_ends=True)
    bridge = SimpleNamespace(
        id="bridge123456789",
        project_id=project,
        parent_id="prediction",
        trace_id="a" * 32,
    )
    coordinator = object.__new__(LiveEvaluationCoordinator)
    coordinator.project = project
    coordinator.weave = SimpleNamespace(get_client=lambda: client)
    coordinator._weave_requires_reactivation = False
    coordinator._summary_fetcher = _out_of_order_summary_fetcher(client)
    coordinator.trace_timeout_sec = 0
    coordinator.env = {}
    coordinator._cancellation_event = threading.Event()
    coordinator._native_agent_root_op = "fugue.claude_code_native_agent_root"
    row = {
        "run_id": "run-a",
        "cell_id": "cell-a",
        "run_key": "run-a:harbor:trial:task-a:claude-code:none:candidate:t001",
        "harness": "claude-code",
        "task_id": "task-a",
        "candidate_id": "candidate-a",
        "attempt_id": "b" * 64,
        "execution_fingerprint": "e" * 64,
        "comparison_example_id": "example-a",
        "trial_index": 1,
        "eval_predict_and_score_call_id": "predict-and-score",
        "trace_project": project,
        "status": "passed",
    }
    active_prediction = export._LivePrediction(
        session=SimpleNamespace(),
        prediction=SimpleNamespace(),
        bridge_call=bridge,
        bridge_client=client,
        row=row,
        opened_monotonic=0.0,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent evidence receipt end was not durably published",
    ):
        coordinator._materialize_native_agent_call(
            active=active_prediction,
            row=row,
            root={
                "conversation_id": "native-session",
                "trace_id": "a" * 32,
                "span_id": "c" * 16,
            },
        )

    native_id = str(row["weave_agent_root_call_id"])
    assert client.remote[native_id]["terminal"] is False
    assert row["weave_agent_root_call_start_verified"] is True
    assert row["weave_agent_root_call_terminal_verified"] is False
    assert "weave_agent_root_call_object_created" not in row
    assert f"end-dropped:{native_id}" in client.events


def test_claude_live_evaluation_normalizes_weave_uuid_traceparent() -> None:
    project = "entity/project"
    weave_trace_id = "019fb117-8955-7662-8225-67228a32b976"
    prediction_call = SimpleNamespace(
        id="prediction-call",
        project_id=project,
        parent_id="predict-and-score",
        trace_id=weave_trace_id,
    )

    class FakeClient:
        def create_call(
            self,
            op,
            inputs,
            parent=None,
            attributes=None,
            display_name=None,
            *,
            use_stack=True,
            _call_id_override=None,
        ):
            return SimpleNamespace(
                id=_call_id_override,
                parent_id=parent.id,
                project_id=project,
                trace_id=parent.trace_id,
                ref=SimpleNamespace(
                    uri=f"weave:///{project}/call/{_call_id_override}"
                ),
            )

        @staticmethod
        def finish_call(call, output=None):
            return None

    coordinator = object.__new__(LiveEvaluationCoordinator)
    coordinator.project = project
    coordinator.weave = SimpleNamespace(get_client=FakeClient)
    cell = PlannedCell(
        id="cell-uuid-bridge",
        run_id="run-uuid-bridge",
        run_name="uuid-bridge",
        workload_id="harbor",
        task_id="task-uuid-bridge",
        harness="claude-code",
        context_system_id="none",
        variant_id="candidate",
        model_provider="anthropic",
        model="anthropic/claude-sonnet-5",
        trial_index=1,
        comparison_example_id="example-uuid-bridge",
        candidate_id="candidate-uuid-bridge",
        execution_fingerprint="runtime-uuid-bridge",
        config_path=Path("config.yaml"),
        result_path=Path("jobs/result.json"),
        command=("harbor", "run"),
        env={},
        n_attempts=1,
    )
    row: dict[str, object] = {}

    bridge, _, traceparent = coordinator._open_agent_bridge(
        cell=cell,
        row=row,
        prediction=SimpleNamespace(predict_call=prediction_call),
    )

    compact = weave_trace_id.replace("-", "")
    bridge_parent_span = str(row["weave_agent_bridge_otel_span_id"])
    assert traceparent == f"00-{compact}-{bridge_parent_span}-01"
    assert bridge.id != bridge_parent_span
    assert len(bridge_parent_span) == 16
    assert row["weave_agent_bridge_cross_transport_edge"] == {
        "schema_version": 1,
        "status": "verified",
        "weave_call_id": bridge.id,
        "otel_trace_id": compact,
        "otel_span_id": bridge_parent_span,
    }
    assert row["weave_agent_bridge_trace_id"] == weave_trace_id
    assert row["weave_agent_bridge_otel_trace_id"] == compact


def test_claude_materializes_real_call_from_verified_native_otel_root() -> None:
    project = "entity/project"
    weave_trace_id = "019fb117-8955-7662-8225-67228a32b976"
    bridge = SimpleNamespace(
        id="bridge123456789",
        project_id=project,
        parent_id="prediction",
        trace_id=weave_trace_id,
    )
    created: list[dict[str, object]] = []
    finished: list[tuple[object, object]] = []

    class FakeClient:
        def create_call(
            self,
            op,
            inputs,
            parent=None,
            attributes=None,
            display_name=None,
            *,
            use_stack=True,
            _call_id_override=None,
        ):
            created.append(
                {
                    "op": op,
                    "inputs": inputs,
                    "parent": parent,
                    "attributes": attributes,
                    "display_name": display_name,
                    "use_stack": use_stack,
                    "id": _call_id_override,
                }
            )
            return SimpleNamespace(
                id=_call_id_override,
                parent_id=parent.id,
                project_id=project,
                trace_id=parent.trace_id,
            )

        def finish_call(self, call, output=None):
            finished.append((call, output))

    coordinator = object.__new__(LiveEvaluationCoordinator)
    coordinator.project = project
    row = {
        "run_id": "run-a",
        "cell_id": "cell-a",
        "run_key": "run-a:harbor:trial:task-a:claude-code:none:candidate:t001",
        "harness": "claude-code",
        "task_id": "task-a",
        "candidate_id": "candidate-a",
        "attempt_id": "a" * 64,
        "execution_fingerprint": "e" * 64,
        "comparison_example_id": "example-a",
        "trial_index": 1,
        "eval_predict_and_score_call_id": "predict-and-score",
        "trace_project": project,
        "status": "passed",
        "agent_response_sha256": "f" * 64,
    }
    root = {
        "conversation_id": "native-session",
        "trace_id": weave_trace_id.replace("-", ""),
        "span_id": "b" * 16,
    }
    active = export._LivePrediction(
        session=SimpleNamespace(),
        prediction=SimpleNamespace(),
        bridge_call=bridge,
        bridge_client=FakeClient(),
        row=row,
        opened_monotonic=0.0,
    )

    coordinator._materialize_native_agent_call(
        active=active,
        row=row,
        root=root,
    )

    assert created[0]["parent"] is bridge
    assert created[0]["use_stack"] is False
    assert "gen_ai.operation.name" not in created[0]["attributes"]
    assert created[0]["attributes"]["fugue.evidence.kind"] == (
        "native_agent_root_cross_transport_receipt"
    )
    assert created[0]["attributes"]["fugue.attempt_id"] == "a" * 64
    assert len(str(created[0]["id"])) == 36
    assert str(export.uuid.UUID(str(created[0]["id"]))) == created[0]["id"]
    assert created[0]["id"] != root["span_id"]
    assert finished[0][0].id == created[0]["id"]
    assert row["weave_agent_root_call_id"] == created[0]["id"]
    assert row["weave_agent_root_call_materialization_source"] == (
        "verified_native_otel_root_v1"
    )
    assert row["weave_agent_root_evidence_kind"] == (
        "native_otel_cross_transport_receipt_v1"
    )
    assert row["weave_agent_root_is_native_call"] is False
    assert row["agent_cross_transport_edge"]["status"] == "verified"
    assert row["weave_agent_root_call_otel_span_id"] == "b" * 16


def test_verified_native_claude_root_requires_bridge_otel_parent() -> None:
    compact_trace = "a" * 32
    row = {
        "harness": "claude-code",
        "attempt_id": "a" * 64,
        "execution_fingerprint": "e" * 64,
        "planned_conversation_id": "planned",
        "conversation_correlation_verified": True,
        "weave_conversation_ids": ["native"],
        "trace_id": compact_trace,
        "root_span_id": "b" * 16,
        "weave_agent_bridge_call_id": "bridge",
        "weave_agent_bridge_otel_span_id": "c" * 16,
        "weave_agent_bridge_otel_trace_id": compact_trace,
        "weave_root_spans": [
            {
                "conversation_id": "native",
                "trace_id": compact_trace,
                "span_id": "b" * 16,
                "otel_parent_span_id": "other-bridge",
                "attempt_id": "a" * 64,
                "execution_fingerprint": "e" * 64,
                "eval_predict_and_score_call_id": "predict-and-score",
            }
        ],
    }

    assert export._verified_native_otel_root(row, "predict-and-score") is None
    assert row["trace_link_status"] == "otel_ancestry_mismatch"
    assert "Evaluation bridge" in row["trace_link_error"]


def test_verified_native_root_rejects_missing_bridge_otel_parent_identity() -> None:
    compact_trace = "a" * 32
    row = {
        "harness": "claude-code",
        "attempt_id": "a" * 64,
        "execution_fingerprint": "e" * 64,
        "planned_conversation_id": "planned",
        "conversation_correlation_verified": True,
        "weave_conversation_ids": ["native"],
        "trace_id": compact_trace,
        "root_span_id": "b" * 16,
        "weave_agent_bridge_call_id": "bridge",
        "weave_agent_bridge_otel_trace_id": compact_trace,
        "weave_root_spans": [
            {
                "conversation_id": "native",
                "trace_id": compact_trace,
                "span_id": "b" * 16,
                "otel_parent_span_id": "c" * 16,
                "attempt_id": "a" * 64,
                "execution_fingerprint": "e" * 64,
                "eval_predict_and_score_call_id": "predict-and-score",
            }
        ],
    }

    assert export._verified_native_otel_root(row, "predict-and-score") is None
    assert row["trace_link_status"] == "otel_ancestry_mismatch"
    assert "Evaluation bridge" in row["trace_link_error"]


@pytest.mark.parametrize("harness", ["hermes", "openclaw", "claude-code", "codex"])
def test_agent_trace_is_remotely_verified_before_evaluation_output(
    monkeypatch: pytest.MonkeyPatch,
    harness: str,
) -> None:
    events: list[str] = []
    native_root = {
        "conversation_id": "native-session",
        "trace_id": "a" * 32,
        "span_id": "b" * 16,
        "weave_call_id": "native-call",
    }

    class FakeClient:
        @staticmethod
        def finish_call(call, output=None):
            events.append("bridge-finished")

    coordinator = object.__new__(LiveEvaluationCoordinator)
    coordinator.project = "entity/project"
    coordinator._wait_for_trace = lambda row, **kwargs: {}
    coordinator._append_event = lambda *args, **kwargs: None
    coordinator._record_verified_agent_root = (
        lambda **kwargs: (
            events.append("verified"),
            kwargs["row"].update(
                {
                    "trace_link_status": "linked",
                    "trace_link_error": None,
                }
            ),
        )
    )
    monkeypatch.setattr(
        export,
        "_verified_native_otel_root",
        lambda row, call_id: native_root,
    )
    monkeypatch.setattr(
        export,
        "_verified_evaluation_root",
        lambda row, call_id: native_root,
    )
    monkeypatch.setattr(
        export,
        "_verify_authoritative_agent_graph",
        lambda row: row.update(
            {"weave_authoritative_call_graph_verified": True}
        ),
    )
    cell = SimpleNamespace(
        harness=harness,
        id="cell-a",
        candidate_id="candidate-a",
    )
    row = {
        "agent_execution_status": "completed",
        "execution_kind": "agent",
        "harness": harness,
    }
    active = export._LivePrediction(
        session=SimpleNamespace(),
        prediction=SimpleNamespace(
            predict_and_score_call=SimpleNamespace(summary={})
        ),
        bridge_call=SimpleNamespace(
            id="bridge",
            project_id="entity/project",
        ),
        bridge_client=FakeClient(),
        row=row,
        opened_monotonic=0.0,
    )

    coordinator._prepare_terminal_agent_trace(
        active=active,
        cell=cell,
        row=row,
        predict_and_score_call_id="predict-and-score",
    )

    assert events == ["bridge-finished", "verified"]
    assert export._evaluation_output(row)["trace_link_status"] == "linked"
    assert "remote_verification_pending" not in json.dumps(row)


def test_live_evaluation_marks_exact_prediction_ops_eager() -> None:
    def predict_and_score() -> None:
        return None

    def predict() -> None:
        return None

    predict_and_score.eager_call_start = False
    predict.eager_call_start = False
    logger_type = type("EvaluationLogger", (), {})
    logger_type.__module__ = "weave.evaluation.eval_imperative"
    logger = logger_type()
    logger._pseudo_evaluation = SimpleNamespace(
        predict_and_score=predict_and_score
    )
    logger._context_predict_method = predict

    assert export._enable_eager_evaluation_starts(logger) is True
    assert predict_and_score.eager_call_start is True
    assert predict.eager_call_start is True


def test_real_live_evaluation_fails_closed_without_eager_ops() -> None:
    logger_type = type("EvaluationLogger", (), {})
    logger_type.__module__ = "weave.evaluation.eval_imperative"
    logger = logger_type()
    logger._pseudo_evaluation = SimpleNamespace()
    logger._context_predict_method = None

    assert export._enable_eager_evaluation_starts(logger) is False


def test_claude_begin_failure_closes_entered_prediction(
    tmp_path: Path,
) -> None:
    class FakeDataset:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakePrediction:
        def __init__(self) -> None:
            self.predict_and_score_call = SimpleNamespace(
                id="predict-and-score",
                project_id="entity/project",
                summary=None,
            )
            self.output = None
            self.exit_args = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.exit_args.append((exc_type, exc, traceback))

    prediction = FakePrediction()

    class FakeLogger:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def log_prediction(self, inputs):
            return prediction

    cell = PlannedCell(
        id="cell-claude-start-failure",
        run_id="run-claude-start-failure",
        run_name="claude start failure",
        workload_id="coding",
        task_id="task-a",
        harness="claude-code",
        context_system_id="none",
        variant_id="none",
        model_provider="anthropic",
        model="anthropic/claude-sonnet-5",
        trial_index=1,
        comparison_example_id="example-a",
        candidate_id="candidate-a",
        execution_fingerprint="execution-a",
        config_path=Path("config.yaml"),
        result_path=Path("jobs/missing/result.json"),
        command=("harbor", "run"),
        env={"WANDB_API_KEY": "test-only"},
        n_attempts=1,
    )
    coordinator = LiveEvaluationCoordinator(
        [cell],
        repo_root=tmp_path,
        project="entity/project",
        env=cell.env,
        weave_module=SimpleNamespace(
            Dataset=FakeDataset,
            EvaluationLogger=FakeLogger,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Evaluation graph is unresolved before Agent execution",
    ):
        coordinator.begin_cell(cell)

    assert len(prediction.exit_args) == 1
    assert prediction.exit_args[0][0] is RuntimeError
    assert coordinator._predictions == {}
    statuses = [
        json.loads(line)["status"]
        for line in (
            tmp_path
            / ".fugue/runtime/run-claude-start-failure/evaluations.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert statuses == ["pending", "prediction_start_failed"]


def _stub_live_agent_bridge(
    monkeypatch: pytest.MonkeyPatch,
    coordinator: LiveEvaluationCoordinator,
) -> None:
    def verified_graph(row, **_kwargs) -> None:
        row["evaluation_prediction_graph_verified"] = True

    monkeypatch.setattr(export, "_verify_live_evaluation_graph", verified_graph)

    def open_bridge(*, row, **_kwargs):
        bridge = SimpleNamespace(
            id="b" * 16,
            project_id=coordinator.project,
            trace_id="a" * 32,
        )
        row.update(
            {
                "weave_agent_bridge_call_id": bridge.id,
                "weave_agent_bridge_object_verified": True,
            }
        )
        return bridge, SimpleNamespace(), f"00-{'a' * 32}-{bridge.id}-01"

    coordinator._open_agent_bridge = open_bridge

    def finish_bridge(active, **_kwargs) -> None:
        active.bridge_finished = True
        active.row["weave_agent_bridge_closed_verified"] = True

    coordinator._finish_agent_bridge = finish_bridge


def test_live_cancellation_closes_open_prediction_once_without_trace_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    _stub_live_agent_bridge(monkeypatch, coordinator)

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
    monkeypatch: pytest.MonkeyPatch,
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
    _stub_live_agent_bridge(monkeypatch, coordinator)
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
    monkeypatch: pytest.MonkeyPatch,
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
    _stub_live_agent_bridge(monkeypatch, coordinator)

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
    assert planned["applicable"] is True
    assert planned["skip_reason"] is None
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
        tmp_path / ".fugue/runtime/run-a/gateway-evidence/job-a/context-gateway.jsonl"
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


def test_mcp_proxy_events_export_exact_tool_and_project_scope(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    event_log = trial_dir / "artifacts" / "fugue-context-events.jsonl"
    event_log.parent.mkdir(parents=True)
    event_log.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "event": "mcp_tool_request",
                    "layer": "proxy",
                    "server": "wandb-mcp-0-4",
                    "tool": "query_wandb_tool",
                    "request_id": "query-1",
                    "arguments": {
                        "entity": "wandb",
                        "project": "fugue-mcp-release-qualification-v1",
                        "resource": "run",
                        "response_mode": "items",
                        "target_x": 3,
                        "x_axis": "_step",
                        "max_evals": 20,
                        "run_id": "maint-r18-06",
                        "keys": ["_step", "latency_ms"],
                            "samples": 5,
                            "parent_filter_digest": "a" * 64,
                            "parent_filter_count": 1,
                            "columns": [
                            "id",
                            "config.attempt_label",
                            "summary.latency_ms",
                        ],
                            "filters": {
                                "limit": 6,
                                "op_name_contains": "predict_and_score",
                            "private_query": "must-not-be-exported",
                        },
                    },
                },
                {
                    "event": "mcp_tool_request",
                    "layer": "proxy",
                    "server": "wandb-mcp-0-4",
                    "tool": "summarize_evaluation_tool",
                    "request_id": "summary-1",
                    "arguments": {
                        "scope": {
                            "project_ref": (
                                "wandb/fugue-mcp-release-qualification-v1"
                            )
                        }
                    },
                },
                {
                    "event": "mcp_tool_response",
                    "layer": "upstream",
                    "server": "wandb-mcp-0-4",
                    "tool": "query_wandb_tool",
                    "request_id": "query-1",
                    "latency_ms": 12,
                    "terminal_status": "succeeded",
                    "successful": True,
                    "returned_count": 6,
                    "total_count": 6,
                    "has_more": False,
                    "project_exhaustive": True,
                    "truncation_applied": False,
                    "coverage_status": "project-exhaustive",
                },
                {
                    "event": "mcp_tool_response",
                    "layer": "upstream",
                    "server": "wandb-mcp-0-4",
                    "tool": "summarize_evaluation_tool",
                    "request_id": "summary-1",
                    "latency_ms": 8,
                    "terminal_status": "succeeded",
                    "successful": True,
                    "prediction_count": 16,
                },
            )
        )
        + "\n"
    )

    summary = export._context_event_summary(trial_dir)

    assert summary["mcp_tool_names"] == [
        "query_wandb_tool",
        "summarize_evaluation_tool",
    ]
    assert summary["mcp_tool_call_count"] == 2
    assert summary["mcp_tool_error_count"] == 0
    assert summary["integration_ids_invoked"] == ["wandb-mcp-0-4"]
    assert summary["mcp_queried_projects"] == [
        "wandb/fugue-mcp-release-qualification-v1"
    ]
    calls = summary["mcp_tool_calls"]
    assert [call["tool"] for call in calls] == [
        "query_wandb_tool",
        "summarize_evaluation_tool",
    ]
    assert calls[0]["queried_project"] == (
        "wandb/fugue-mcp-release-qualification-v1"
    )
    assert calls[0]["resource"] == "run"
    assert calls[0]["response_mode"] == "items"
    assert calls[0]["target_x"] == 3
    assert calls[0]["x_axis"] == "_step"
    assert calls[0]["max_evals"] == 20
    assert calls[0]["run_id"] == "maint-r18-06"
    assert calls[0]["keys"] == ["_step", "latency_ms"]
    assert calls[0]["samples"] == 5
    assert calls[0]["projection"] == [
        "_step",
        "attempt_label",
        "config.attempt_label",
        "id",
        "latency_ms",
        "summary.latency_ms",
    ]
    assert calls[0]["limit"] == 6
    assert calls[0]["parent_filter_count"] == 1
    assert calls[0]["op_name_filter"] == {
        "op_name_contains": ["predict_and_score"]
    }
    assert calls[0]["terminal_status"] == "succeeded"
    assert calls[0]["successful"] is True
    assert calls[0]["response_metadata_verified"] is True
    assert calls[0]["returned_count"] == 6
    assert calls[0]["total_count"] == 6
    assert calls[0]["has_more"] is False
    assert calls[0]["project_exhaustive"] is True
    assert calls[0]["truncation_applied"] is False
    assert calls[0]["coverage_status"] == "project-exhaustive"
    assert calls[1]["prediction_count"] == 16
    assert calls[0]["parent_filter_digest"] == "a" * 64
    assert "must-not-be-exported" not in json.dumps(
        summary["mcp_tool_calls"], sort_keys=True
    )
    assert export._mcp_queried_projects(
        [
            {
                "event": "mcp_tool_request",
                "tool": "query_wandb_entity_projects",
                "arguments": {"entity": "other-team"},
            },
            {
                "event": "mcp_tool_request",
                "tool": "list_entities_tool",
                "arguments": {},
            },
        ]
    ) == ["*/*", "other-team/*"]


def test_mcp_artifact_refs_resolve_only_their_qualified_project() -> None:
    project = "wandb/fugue-mcp-release-source-v2"
    events = [
        {
            "tool": "get_artifact_details_tool",
            "arguments": {
                "artifact_name": (f"{project}/qualification-evidence-maint-r18-01:v0")
            },
        },
        {
            "tool": "compare_artifact_versions_tool",
            "arguments": {
                "artifact_name_a": (
                    f"{project}/qualification-evidence-maint-r18-01:v0"
                ),
                "artifact_name_b": (
                    f"{project}/qualification-evidence-maint-r18-02:v0"
                ),
            },
        },
    ]

    assert export._mcp_queried_projects(events) == [project]
    assert export._qualified_artifact_project("unqualified:v0") is None
    assert export._qualified_artifact_project("../project/name:v0") is None


def test_mcp_artifact_digest_survives_proxy_to_export_without_private_content(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    event_log = trial_dir / "artifacts" / "fugue-context-events.jsonl"
    event_log.parent.mkdir(parents=True)
    project = "wandb/fugue-mcp-release-source-v2"
    artifact = f"{project}/qualification-evidence-maint-r18-02:v0"
    event_log.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "event": "mcp_tool_request",
                    "layer": "proxy",
                    "server": "staging",
                    "tool": "get_artifact_details_tool",
                    "request_id": "artifact-b",
                    "arguments": {
                        "artifact_name": artifact,
                        "include_files": False,
                    },
                },
                {
                    "event": "mcp_tool_response",
                    "layer": "upstream",
                    "server": "staging",
                    "tool": "get_artifact_details_tool",
                    "request_id": "artifact-b",
                    "terminal_status": "succeeded",
                    "successful": True,
                    "artifact_digest": "sha256:" + "b" * 64,
                },
            )
        )
        + "\n"
    )

    summary = export._context_event_summary(trial_dir)
    call = summary["mcp_tool_calls"][0]

    assert call["artifact_name"] == artifact
    assert call["include_files"] is False
    assert call["artifact_digest"] == "sha256:" + "b" * 64
    assert summary["mcp_queried_projects"] == [project]


def test_mcp_proxy_evidence_normalizes_raw_graphql_without_query_values(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    event_log = trial_dir / "artifacts" / "fugue-context-events.jsonl"
    event_log.parent.mkdir(parents=True)
    query = """
query PrivateInventory($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runCount
    runs(first: 50) {
      edges {
        node {
          privateCustomerAlias: name
          state
          config
          summaryMetrics
        }
      }
    }
  }
}
"""
    event_log.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "event": "mcp_tool_request",
                    "layer": "proxy",
                    "server": "main",
                    "tool": "query_wandb_tool",
                    "request_id": "graphql-1",
                    "arguments": {
                        "query": query,
                        "variables": {
                            "entity": "wandb",
                            "project": "release-project",
                            "private_filter": "customer-secret",
                        },
                        "max_items": 50,
                    },
                },
                {
                    "event": "mcp_tool_response",
                    "layer": "upstream",
                    "server": "main",
                    "tool": "query_wandb_tool",
                    "request_id": "graphql-1",
                    "terminal_status": "succeeded",
                    "successful": True,
                    "returned_count": 6,
                    "total_count": 6,
                    "has_more": False,
                    "project_exhaustive": True,
                    "truncation_applied": False,
                    "coverage_status": "project-exhaustive",
                },
            )
        )
        + "\n"
    )

    summary = export._context_event_summary(trial_dir)

    assert summary["mcp_tool_call_count"] == 1
    assert summary["mcp_tool_error_count"] == 0
    [call] = summary["mcp_tool_calls"]
    assert call["raw_graphql"] is True
    assert call["graphql_operation_type"] == "query"
    assert call["resource"] == "runs"
    assert call["response_modes"] == ["count", "items"]
    assert call["projected_fields"] == [
        "config.*",
        "id",
        "state",
        "summary.*",
    ]
    assert call["broad_projection"] is True
    assert call["graphql_projection_resolved"] is True
    assert call["graphql_requested_limit"] == 50
    assert call["graphql_limit_resolved"] is True
    assert call["graphql_scope_resolved"] is True
    assert call["effective_limit"] == 50
    serialized = json.dumps(call, sort_keys=True)
    assert "PrivateInventory" not in serialized
    assert "privateCustomerAlias" not in serialized
    assert "customer-secret" not in serialized


def test_mcp_proxy_evidence_exports_fail_closed_graphql_shape(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    event_log = trial_dir / "artifacts" / "fugue-context-events.jsonl"
    event_log.parent.mkdir(parents=True)
    arguments = safe_graphql_event_arguments(
        {
            "query": """
query Inventory($pageSize: Int!) {
  project(entityName: "other-team", name: "private-project") {
    runs(first: $pageSize) {
      edges { node { id ...BroadRun } }
    }
  }
}
fragment BroadRun on Run { config summaryMetrics }
""",
            "variables": {
                "pageSize": 6,
                "private_filter": "customer-secret",
            },
        }
    )
    event_log.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "event": "mcp_tool_request",
                    "layer": "proxy",
                    "server": "main",
                    "tool": "query_wandb_tool",
                    "request_id": "graphql-unresolved",
                    "arguments": arguments,
                },
                {
                    "event": "mcp_tool_response",
                    "layer": "upstream",
                    "server": "main",
                    "tool": "query_wandb_tool",
                    "request_id": "graphql-unresolved",
                    "terminal_status": "succeeded",
                    "successful": True,
                },
            )
        )
        + "\n"
    )

    summary = export._context_event_summary(trial_dir)

    assert summary["mcp_queried_projects"] == ["*/*"]
    [call] = summary["mcp_tool_calls"]
    assert call["queried_projects"] == ["*/*"]
    assert call["graphql_scope_resolved"] is False
    assert call["graphql_projection_resolved"] is False
    assert call["broad_projection"] is True
    assert call["graphql_limit_resolved"] is True
    assert call["graphql_requested_limit"] == 6
    serialized = json.dumps(call, sort_keys=True)
    assert "other-team" not in serialized
    assert "private-project" not in serialized
    assert "BroadRun" not in serialized
    assert "customer-secret" not in serialized


def test_mcp_proxy_evidence_keeps_errors_but_counts_only_successes(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    event_log = trial_dir / "artifacts" / "fugue-context-events.jsonl"
    event_log.parent.mkdir(parents=True)
    event_log.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "event": "mcp_tool_request",
                    "layer": "proxy",
                    "server": "wandb",
                    "tool": "get_run_history_tool",
                    "request_id": "history-1",
                    "arguments": {
                        "entity_name": "wandb",
                        "project_name": "release-project",
                        "run_id": "run-a",
                    },
                },
                {
                    "event": "mcp_tool_response",
                    "layer": "upstream",
                    "server": "wandb",
                    "tool": "get_run_history_tool",
                    "request_id": "history-1",
                    "terminal_status": "structured_error",
                    "successful": False,
                    "structured_error_code": "storage_error",
                },
                {
                    "event": "mcp_tool_request",
                    "layer": "proxy",
                    "server": "wandb",
                    "tool": "query_wandb_tool",
                    "request_id": "run-1",
                    "arguments": {
                        "entity_name": "wandb",
                        "project_name": "release-project",
                        "resource": "run",
                        "run_id": "run-a",
                    },
                },
                {
                    "event": "mcp_tool_response",
                    "layer": "upstream",
                    "server": "wandb",
                    "tool": "query_wandb_tool",
                    "request_id": "run-1",
                    "terminal_status": "succeeded",
                    "successful": True,
                },
            )
        )
        + "\n"
    )

    summary = export._context_event_summary(trial_dir)

    assert summary["mcp_tool_call_count"] == 1
    assert summary["mcp_tool_error_count"] == 1
    assert summary["mcp_tool_names"] == ["query_wandb_tool"]
    calls = summary["mcp_tool_calls"]
    assert len(calls) == 2
    failed = next(call for call in calls if call["tool"] == "get_run_history_tool")
    assert failed["terminal_status"] == "structured_error"
    assert failed["successful"] is False
    assert failed["structured_error_code"] == "storage_error"
    assert failed["response_metadata_verified"] is True


def test_mcp_proxy_evidence_rejects_untrusted_error_codes() -> None:
    normalized = export._normalized_mcp_response(
        [
            {
                "tool": "query_wandb_tool",
                "terminal_status": "structured_error",
                "successful": False,
                "structured_error_code": "sk-ant-api03-PRIVATE-TOKEN",
            }
        ],
        tool="query_wandb_tool",
    )

    assert normalized["structured_error_code"] == "tool_error"
    assert "PRIVATE-TOKEN" not in json.dumps(normalized, sort_keys=True)


def test_retrieval_to_action_funnel_preserves_rank_without_exporting_gold(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    events_path = trial_dir / "artifacts" / "fugue-context-events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "event": "retrieve",
                    "layer": "portable_client",
                    "metrics": {"result_count": 3},
                    "hits": [
                        {"path": "docs/distractor.rst", "score": 0.9},
                        {"path": "src/relevant.py", "score": 0.8},
                        {"path": "src/other.py", "score": 0.7},
                    ],
                },
                {
                    "event": "retrieve",
                    "layer": "portable_client",
                    "metrics": {"result_count": 2},
                    "hits": [
                        {"path": "src/relevant.py", "score": 0.95},
                        {"path": "tests/test_relevant.py", "score": 0.6},
                    ],
                },
            )
        )
        + "\n"
    )

    summary = export._context_event_summary(trial_dir)
    activity = export._retrieval_to_action_activity(
        summary["context_result_paths"],
        ["/testbed/src/relevant.py", "tests/test_relevant.py"],
        ["/workspace/repo/src/other.py"],
    )
    row = {
        **summary,
        **activity,
        "inspected_paths": ["src/relevant.py", "tests/test_relevant.py"],
        "changed_paths": ["src/other.py"],
        "evidence_paths": ["src/relevant.py", "src/other.py"],
        "agent_evidence_paths": ["tests/test_relevant.py"],
        "agent_execution_status": "started",
        "pass": False,
    }
    export._apply_host_evidence_scores(
        row,
        ("src/relevant.py", "tests/test_relevant.py"),
        "a" * 64,
    )

    assert summary["context_result_paths"] == [
        "docs/distractor.rst",
        "src/relevant.py",
        "src/other.py",
        "tests/test_relevant.py",
    ]
    assert summary["context_result_path_count"] == 4
    assert summary["context_result_count"] == 5
    assert activity["context_result_opened_paths"] == [
        "src/relevant.py",
        "tests/test_relevant.py",
    ]
    assert activity["context_result_changed_paths"] == ["src/other.py"]
    assert activity["context_result_open_rate"] == 0.5
    assert activity["context_result_change_rate"] == 0.25
    assert row["retrieval_recall_at_5"] == 1.0
    assert row["retrieval_recall_at_10"] == 1.0
    assert row["retrieval_mrr"] == 0.5
    assert row["relevant_retrieval_observed"] is True
    assert row["relevant_retrieval_returned"] is True
    assert row["relevant_retrieval_opened"] is True
    assert row["relevant_retrieval_used"] is True
    assert row["relevant_retrieval_changed"] is False
    assert row["off_target_change_only"] is True
    assert row["premature_completion"] is False

    hidden_gold_row = {
        "context_result_paths": ["src/other.py"],
        "inspected_paths": ["src/other.py"],
        "changed_paths": [],
        "evidence_paths": ["src/other.py"],
        "agent_execution_status": "started",
        "pass": False,
    }
    export._apply_host_evidence_scores(
        hidden_gold_row,
        ("private/expected-solution.py",),
        "b" * 64,
    )
    serialized = json.dumps(hidden_gold_row, sort_keys=True)
    assert "private/expected-solution.py" not in serialized
    assert hidden_gold_row["retrieval_recall_at_5"] == 0.0
    assert hidden_gold_row["relevant_retrieval_opened"] is False
    assert hidden_gold_row["premature_completion"] is True

    claimed_without_opening = {
        "context_result_paths": ["src/relevant.py"],
        "inspected_paths": [],
        "changed_paths": [],
        "agent_evidence_paths": ["src/relevant.py"],
        "agent_execution_status": "started",
        "pass": False,
    }
    export._apply_host_evidence_scores(
        claimed_without_opening,
        ("src/relevant.py",),
        "c" * 64,
    )
    assert claimed_without_opening["relevant_retrieval_returned"] is True
    assert claimed_without_opening["relevant_retrieval_opened"] is False
    assert claimed_without_opening["relevant_retrieval_used"] is False


def test_trial_row_separates_runtime_completion_from_task_outcome(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    result_path = trial_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "maintenance-task",
                "started_at": "2026-07-30T10:00:00+00:00",
                "finished_at": "2026-07-30T10:01:00+00:00",
                "agent_execution": {},
                "verifier_result": {"rewards": {"reward": 0.0}},
            }
        ),
        encoding="utf-8",
    )

    row = export._row_from_trial(result_path)

    assert row["agent_runtime_completed"] is True
    assert row["pass"] is False

    failed = json.loads(result_path.read_text(encoding="utf-8"))
    failed["exception_info"] = {"exception_type": "AgentRuntimeError"}
    result_path.write_text(json.dumps(failed), encoding="utf-8")
    assert export._row_from_trial(result_path)["agent_runtime_completed"] is False


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
                    "fugue.run_key": "run-key",
                    "fugue.harness": "codex",
                    "fugue.task_id": "task-a",
                    "fugue.candidate_id": "candidate-a",
                    "fugue.attempt_id": "a" * 64,
                    "fugue.execution_fingerprint": "e" * 64,
                    "fugue.comparison_example_id": "example-a",
                    "fugue.trial_index": 1,
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
            {
                "_fugue_evidence_source": export._WEAVE_CALL_SOURCE,
                "id": "agent-call",
                "parent_id": "prediction-1",
                "project_id": "entity/project",
                "trace_id": "evaluation-trace",
                "op_name": "invoke_agent",
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.conversation.id": "native-conversation",
                    "fugue.run_key": "run-key",
                    "fugue.harness": "codex",
                    "fugue.task_id": "task-a",
                    "fugue.candidate_id": "candidate-a",
                    "fugue.attempt_id": "a" * 64,
                    "fugue.execution_fingerprint": "e" * 64,
                    "fugue.comparison_example_id": "example-a",
                    "fugue.trial_index": 1,
                    "weave.eval.predict_and_score_call_id": "prediction-1",
                },
            },
        ],
        project="entity/project",
    )

    assert summary["weave_conversation_ids"] == ["native-conversation"]
    row = {
        "attempt_id": "a" * 64,
        "execution_fingerprint": "e" * 64,
        "trace_project": "entity/project",
        "eval_predict_and_score_trace_id": "evaluation-trace",
        "trace_id": trace_id,
        "root_span_id": root_span_id,
        **summary,
    }
    assert export._verified_evaluation_root(row, "prediction-1") is not None


def test_agent_evidence_keeps_otel_span_and_weave_call_identities_separate() -> None:
    attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "codex",
        "gen_ai.conversation.id": "native-conversation",
        "fugue.run_key": "run-key",
        "fugue.harness": "codex",
        "fugue.task_id": "task-a",
        "fugue.candidate_id": "candidate-a",
        "fugue.attempt_id": "a" * 64,
        "fugue.execution_fingerprint": "e" * 64,
        "fugue.comparison_example_id": "example-a",
        "fugue.trial_index": 1,
        "weave.eval.predict_and_score_call_id": "prediction-and-score-1",
    }
    otel_root = {
        "_fugue_evidence_source": export._WEAVE_AGENT_SPAN_SOURCE,
        "span_id": "b" * 16,
        "trace_id": "a" * 32,
        "attributes": attributes,
    }
    weave_call = {
        "_fugue_evidence_source": export._WEAVE_CALL_SOURCE,
        "id": "019fabcd-agent-root",
        "parent_id": "prediction-and-score-1",
        "attributes": attributes,
    }

    summary = _summarize_spans(
        [otel_root, weave_call],
        project="entity/project",
    )

    [root] = summary["weave_root_spans"]
    assert summary["otel_trace_ids"] == ["a" * 32]
    assert summary["otel_root_span_ids"] == ["b" * 16]
    assert root["trace_id"] == "a" * 32
    assert root["span_id"] == "b" * 16
    assert root["weave_call_id"] == "019fabcd-agent-root"
    assert root["weave_call_ref"] == (
        "weave:///entity/project/call/019fabcd-agent-root"
    )
    assert root["weave_call_url"] == (
        "https://wandb.ai/entity/project/weave/calls/019fabcd-agent-root"
    )
    assert summary["weave_call_id"] == "019fabcd-agent-root"
    assert summary["weave_call_id"] != root["span_id"]

    otel_only = _summarize_spans([otel_root], project="entity/project")
    assert otel_only["weave_call_id"] is None
    assert "weave_call_id" not in otel_only["weave_root_spans"][0]


def test_agent_evidence_joins_cross_transport_receipt_without_call_impersonation() -> (
    None
):
    attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "hermes",
        "gen_ai.conversation.id": "native-conversation",
        "fugue.run_key": "run-key",
        "fugue.harness": "hermes",
        "fugue.task_id": "task-a",
        "fugue.candidate_id": "candidate-a",
        "fugue.attempt_id": "a" * 64,
        "fugue.execution_fingerprint": "e" * 64,
        "fugue.comparison_example_id": "example-a",
        "fugue.trial_index": 1,
        "weave.eval.predict_and_score_call_id": "prediction-and-score-1",
    }
    otel_root = {
        "_fugue_evidence_source": export._WEAVE_AGENT_SPAN_SOURCE,
        "span_id": "b" * 16,
        "trace_id": "c" * 32,
        "attributes": attributes,
    }
    receipt_id = "019fabcd-agent-evidence-receipt"
    receipt = {
        "_fugue_evidence_source": export._WEAVE_CALL_SOURCE,
        "id": receipt_id,
        "parent_id": "bridge-call",
        "project_id": "entity/project",
        "trace_id": "d" * 32,
        "op_name": "fugue.native_agent_evidence_receipt",
        "attributes": {
            **{
                key: value
                for key, value in attributes.items()
                if key != "gen_ai.operation.name"
            },
            "fugue.evidence.kind": ("native_agent_root_cross_transport_receipt"),
            "fugue.evidence.cross_transport_edge_verified": True,
            "fugue.native_agent_root_receipt": True,
            "fugue.native_otel_trace_id": "c" * 32,
            "fugue.native_otel_span_id": "b" * 16,
        },
    }

    summary = _summarize_spans(
        [otel_root, receipt],
        project="entity/project",
        required_call_ids=[receipt_id],
    )

    [root] = summary["weave_root_spans"]
    assert root["weave_call_id"] == receipt_id
    assert root["weave_call_evidence_kind"] == (
        "native_otel_cross_transport_receipt_v1"
    )
    assert root["weave_call_is_native"] is False
    row = {"trace_project": "entity/project"}
    export._apply_verified_agent_evidence(row, root, project="entity/project")
    assert row["weave_agent_root_call_id"] == receipt_id
    assert row["weave_agent_root_is_native_call"] is False
    assert row["agent_cross_transport_edge"] == {
        "schema_version": 1,
        "status": "verified",
        "source_system": "otel",
        "source_trace_id": "c" * 32,
        "source_span_id": "b" * 16,
        "receipt_system": "weave",
        "receipt_call_id": receipt_id,
    }


def test_late_native_agent_call_explicitly_supersedes_receipt() -> None:
    row = {
        "trace_project": "entity/project",
        "weave_agent_root_call_id": "receipt-call",
        "weave_agent_root_evidence_kind": ("native_otel_cross_transport_receipt_v1"),
        "weave_agent_root_is_native_call": False,
        "agent_cross_transport_edge": {"status": "verified"},
    }
    root = {
        "trace_id": "a" * 32,
        "span_id": "b" * 16,
        "weave_call_id": "native-call",
        "weave_call_evidence_kind": "native_weave_call_v1",
        "weave_call_is_native": True,
    }

    export._apply_verified_agent_evidence(row, root, project="entity/project")

    assert row["weave_agent_root_call_id"] == "native-call"
    assert row["weave_agent_root_evidence_kind"] == "native_weave_call_v1"
    assert row["weave_agent_root_is_native_call"] is True
    assert "agent_cross_transport_edge" not in row
    assert row["weave_agent_receipt_cross_transport_edge"] == {"status": "verified"}
    assert row["weave_agent_receipt_supersession"] == {
        "schema_version": 1,
        "status": "superseded_by_native_call",
        "receipt_call_id": "receipt-call",
        "native_call_id": "native-call",
    }


def test_live_link_rejects_split_native_conversation_identity() -> None:
    row = {
        "attempt_id": "a" * 64,
        "execution_fingerprint": "e" * 64,
        "trace_id": "a" * 32,
        "root_span_id": "b" * 16,
        "weave_conversation_ids": ["native-root", "split-tool"],
        "weave_root_spans": [
            {
                "conversation_id": "native-root",
                "attempt_id": "a" * 64,
                "execution_fingerprint": "e" * 64,
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


def test_conversation_correlation_keeps_distinct_planned_and_native_ids() -> None:
    row = {
        "planned_conversation_id": "planned-fugue-id",
        "native_session_ids": ["native-claude-session"],
        "attempt_id": "a" * 64,
        "execution_fingerprint": "e" * 64,
    }
    root = {
        "conversation_id": "native-claude-session",
        "attempt_id": "a" * 64,
        "execution_fingerprint": "e" * 64,
    }

    export._apply_conversation_correlation(row, root)

    assert row["conversation_correlation_verified"] is True
    assert row["conversation_ids_match"] is False
    assert row["conversation_correlation"]["status"] == "verified"


def _verified_checkpoint_links(destination: Mapping[str, object]) -> dict[str, object]:
    project = str(destination["project_slug"])
    app_base = str(destination["app_base_url"])
    call_ids = {
        "eval_predict_and_score": "predict-and-score-call",
        "weave_prediction": "prediction-call",
        "weave_evaluation_root": "evaluation-call",
        "weave_agent_root": "agent-call",
    }
    values: dict[str, object] = {
        "trace_project": project,
        "weave_agent_root_evidence_kind": "native_weave_call_v1",
        "weave_agent_root_is_native_call": True,
    }
    for prefix, call_id in call_ids.items():
        values[f"{prefix}_call_id"] = call_id
        values[f"{prefix}_ref"] = f"weave:///{project}/call/{call_id}"
        values[f"{prefix}_url"] = f"{app_base}/{project}/weave/calls/{call_id}"
    values.update(
        {
            "weave_dataset_ref": f"weave:///{project}/object/tasks:v1",
            "weave_dataset_url": (
                f"{app_base}/{project}/weave/objects/tasks/versions/v1"
            ),
        }
    )
    return values


def test_first_cell_evidence_checkpoint_requires_real_graph_and_host_scorer() -> None:
    destination = export.trace_destination_identity(
        {
            "FUGUE_WEAVE_PROJECT": (
                "wandb/fugue-mcp-release-qualification-v1"
            )
        }
    )
    valid = {
        **_verified_checkpoint_links(destination),
        "trace_receipt": destination,
        "evaluation_root_object_verified": True,
        "dataset_version_object_verified": True,
        "eval_predict_and_score_object_verified": True,
        "weave_prediction_object_verified": True,
        "evaluation_prediction_graph_verified": True,
        "agent_graph_verified": True,
        "conversation_correlation_verified": True,
        "trace_link_status": "linked",
        "weave_agent_root_call_id": "agent-call",
        "otel_root_span_id": "b" * 16,
        "host_evaluator_status": "passed",
        "local_cell_conformance": {
            "status": "passed",
            "docker_cleanup": {"status": "passed"},
            "local_artifact_privacy_scan": {"status": "passed"},
            "private_label_boundary": {"status": "passed"},
        },
        "negative_routing_receipt": {"status": "passed"},
    }

    assert (
        export._live_evidence_checkpoint_failures(
            valid,
            expected_destination=destination,
            host_evaluator_required=True,
        )
        == []
    )

    invalid = {**valid, "host_evaluator_status": "failed"}
    assert "host evaluator did not complete successfully" in (
        export._live_evidence_checkpoint_failures(
            invalid,
            expected_destination=destination,
            host_evaluator_required=True,
        )
    )
    advisory_judge_unavailable = {
        **valid,
        "comparison_judge_checkpoint_status": "advisory_unavailable",
        "comparison_judge_checkpoint_unavailable": [
            "maintainer-actionability"
        ],
    }
    assert (
        export._live_evidence_checkpoint_failures(
            advisory_judge_unavailable,
            expected_destination=destination,
            host_evaluator_required=True,
        )
        == []
    )
    required_judge_unavailable = {
        **valid,
        "comparison_judge_checkpoint_status": "failed",
        "comparison_judge_checkpoint_unavailable": [
            "maintainer-actionability"
        ],
        "comparison_judge_checkpoint_required_unavailable": [
            "maintainer-actionability"
        ],
    }
    assert (
        "configured first-cell judge did not score: maintainer-actionability"
        in export._live_evidence_checkpoint_failures(
            required_judge_unavailable,
            expected_destination=destination,
            host_evaluator_required=True,
        )
    )


def test_evidence_checkpoint_covers_every_planned_cell_and_cancels_after_late_failure(
) -> None:
    destination = export.trace_destination_identity(
        {"FUGUE_WEAVE_PROJECT": "wandb/qualification"}
    )
    coordinator = object.__new__(LiveEvaluationCoordinator)
    coordinator._evidence_checkpoint_cells = 3
    coordinator._evidence_checkpoint_terminal = 0
    coordinator._evidence_destination = destination
    coordinator._host_evaluator = None
    coordinator._cancellation_event = threading.Event()
    coordinator._checkpoint_conformance = lambda cell: {
        "status": "passed",
        "docker_cleanup": {"status": "passed"},
        "local_artifact_privacy_scan": {"status": "passed"},
        "private_label_boundary": {"status": "passed"},
    }
    coordinator._negative_routing_receipt = lambda row: {"status": "passed"}
    events: list[tuple[str, dict[str, object]]] = []
    coordinator._append_event = (
        lambda status, **values: events.append((status, values))
    )
    cell = SimpleNamespace(id="cell-a")
    valid = {
        **_verified_checkpoint_links(destination),
        "cell_id": "cell-a",
        "candidate_id": "candidate-a",
        "trace_receipt": destination,
        "evaluation_root_object_verified": True,
        "dataset_version_object_verified": True,
        "eval_predict_and_score_object_verified": True,
        "weave_prediction_object_verified": True,
        "evaluation_prediction_graph_verified": True,
        "agent_graph_verified": True,
        "conversation_correlation_verified": True,
        "trace_link_status": "linked",
        "weave_agent_root_call_id": "agent-call",
        "otel_root_span_id": "b" * 16,
    }
    rows = [
        {**valid, "cell_id": "cell-a"},
        {**valid, "cell_id": "cell-b"},
        {
            **valid,
            "cell_id": "cell-c",
            "trace_link_status": "failed",
            "weave_agent_root_call_id": None,
        },
    ]

    for row in rows:
        coordinator._apply_evidence_checkpoint(cell, row)

    assert [row["evidence_checkpoint_status"] for row in rows] == [
        "passed",
        "passed",
        "failed",
    ]
    assert coordinator._evidence_checkpoint_terminal == 3
    assert coordinator._cancellation_event.is_set()
    assert [status for status, _ in events] == [
        "evidence_checkpoint_passed",
        "evidence_checkpoint_passed",
        "evidence_checkpoint_failed",
    ]
    after_limit = {**valid, "cell_id": "cell-d"}
    coordinator._apply_evidence_checkpoint(cell, after_limit)
    assert "evidence_checkpoint_status" not in after_limit


def test_evaluation_evidence_accepts_canonical_object_refs_not_python_identity() -> None:
    class Ref:
        def __init__(self, uri: str) -> None:
            self._uri = uri

        def uri(self) -> str:
            return self._uri

    class Persisted:
        def __init__(self, uri: str) -> None:
            self.ref = Ref(uri)

    project = "wandb/fugue-mcp-release-qualification-v1"
    evaluation_uri = f"weave:///{project}/object/evaluation:digest"
    dataset_uri = f"weave:///{project}/object/dataset:digest"
    dataset = Persisted(dataset_uri)
    evaluation = Persisted(evaluation_uri)
    evaluation.dataset = Persisted(dataset_uri)
    evaluation_call = SimpleNamespace(
        id="evaluation-call",
        project_id=project,
        inputs={"self": Ref(evaluation_uri)},
        ref=Ref(f"weave:///{project}/call/evaluation-call"),
    )
    logger = SimpleNamespace(
        _pseudo_evaluation=evaluation,
        _evaluate_call=evaluation_call,
        ui_url=f"https://wandb.ai/{project}/r/call/evaluation-call",
    )
    row: dict[str, object] = {}

    export._apply_evaluation_evidence(
        row,
        logger=logger,
        dataset=dataset,
        project=project,
    )

    assert row["evaluation_root_object_verified"] is True
    assert row["evaluation_root_dataset_relationship_verified"] is True
    assert row["dataset_version_object_verified"] is True


def test_dataset_navigation_uses_the_locked_application_origin() -> None:
    project = "team/private-project"
    ref = f"weave:///{project}/object/tasks:dataset-v1"

    assert export._object_url(
        SimpleNamespace(ui_url="https://stale.example/not-the-locked-origin"),
        ref,
        app_base_url="https://wandb.internal.example/",
    ) == (
        "https://wandb.internal.example/team/private-project/"
        "weave/objects/tasks/versions/dataset-v1"
    )


def test_native_agent_call_requires_authoritative_evaluation_ancestry() -> None:
    row = {
        "attempt_id": "a" * 64,
        "execution_fingerprint": "e" * 64,
        "trace_project": "entity/project",
        "trace_id": "a" * 32,
        "root_span_id": "b" * 16,
        "eval_predict_and_score_trace_id": "weave-trace",
        "weave_prediction_call_id": "prediction",
        "weave_conversation_ids": ["native-session"],
        "weave_root_spans": [
            {
                "conversation_id": "native-session",
                "attempt_id": "a" * 64,
                "execution_fingerprint": "e" * 64,
                "trace_id": "a" * 32,
                "span_id": "b" * 16,
                "eval_predict_and_score_call_id": "predict-and-score",
                "weave_call_id": "agent-root",
                "weave_call_parent_id": "unrelated-call",
                "weave_call_project_id": "entity/project",
                "weave_call_trace_id": "weave-trace",
            }
        ],
    }

    root = export._verified_evaluation_root(row, "predict-and-score")

    assert root is None
    assert row["trace_link_status"] == "ancestry_mismatch"
    assert row["trace_link_error"] == (
        "Agent evidence Call is not a child of the exact evaluation prediction"
    )


def test_claude_native_agent_call_requires_verified_bridge_ancestry() -> None:
    bridge_id = "c" * 16
    evaluation_trace_id = "d" * 32
    agent_call_id = "agent-root-call"
    row = {
        "harness": "claude-code",
        "attempt_id": "a" * 64,
        "execution_fingerprint": "e" * 64,
        "trace_project": "entity/project",
        "trace_id": evaluation_trace_id,
        "root_span_id": "b" * 16,
        "weave_evaluation_root_call_id": "evaluation",
        "eval_predict_and_score_call_id": "predict-and-score",
        "eval_predict_and_score_trace_id": evaluation_trace_id,
        "weave_prediction_call_id": "prediction",
        "weave_agent_bridge_call_id": bridge_id,
        "weave_agent_bridge_otel_span_id": "f" * 16,
        "weave_agent_bridge_parent_id": "prediction",
        "weave_agent_bridge_trace_id": evaluation_trace_id,
        "weave_agent_bridge_otel_trace_id": evaluation_trace_id,
        "weave_agent_bridge_object_verified": True,
        "weave_agent_bridge_closed_verified": True,
        "weave_agent_root_call_id": agent_call_id,
        "weave_authoritative_call_graph": [
            {
                "call_id": "evaluation",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": "predict-and-score",
                "parent_id": "evaluation",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": "prediction",
                "parent_id": "predict-and-score",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": bridge_id,
                "parent_id": "prediction",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": agent_call_id,
                "parent_id": bridge_id,
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
        ],
        "weave_authoritative_missing_call_ids": [],
        "weave_conversation_ids": ["native-session"],
        "weave_root_spans": [
            {
                "conversation_id": "native-session",
                "attempt_id": "a" * 64,
                "execution_fingerprint": "e" * 64,
                "trace_id": evaluation_trace_id,
                "span_id": "b" * 16,
                "otel_parent_span_id": "f" * 16,
                "eval_predict_and_score_call_id": "predict-and-score",
                "weave_call_id": agent_call_id,
                "weave_call_parent_id": bridge_id,
                "weave_call_project_id": "entity/project",
                "weave_call_trace_id": evaluation_trace_id,
            }
        ],
    }
    assert export._verified_evaluation_root(row, "predict-and-score") is None
    assert row["trace_link_status"] == "ancestry_unresolved"
    export._verify_authoritative_agent_graph(row)

    root = export._verified_evaluation_root(row, "predict-and-score")

    assert root is not None
    assert root["weave_ancestry_verified"] is True
    assert root["weave_parent_kind"] == "agent_execution_bridge"
    assert row["weave_authoritative_call_graph_verified"] is True
    row["weave_agent_bridge_parent_id"] = "other-prediction"
    assert export._verified_evaluation_root(row, "predict-and-score") is None
    assert row["trace_link_status"] == "ancestry_mismatch"


def test_authoritative_graph_recovers_a_transient_bridge_close_poll_timeout() -> None:
    bridge_id = "c" * 16
    evaluation_trace_id = "d" * 32
    row = {
        "harness": "claude-code",
        "trace_project": "entity/project",
        "weave_evaluation_root_call_id": "evaluation",
        "eval_predict_and_score_call_id": "predict-and-score",
        "eval_predict_and_score_trace_id": evaluation_trace_id,
        "weave_prediction_call_id": "prediction",
        "weave_agent_bridge_call_id": bridge_id,
        "weave_agent_root_call_id": "agent-root",
        "weave_agent_bridge_closed_verified": False,
        "weave_agent_bridge_close_error": "ReadTimeout",
        "weave_authoritative_call_graph": [
            {
                "call_id": "evaluation",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": "predict-and-score",
                "parent_id": "evaluation",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": "prediction",
                "parent_id": "predict-and-score",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": bridge_id,
                "parent_id": "prediction",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": "agent-root",
                "parent_id": bridge_id,
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
        ],
        "weave_authoritative_missing_call_ids": [],
    }

    export._verify_authoritative_agent_graph(row)

    assert row["weave_authoritative_call_graph_verified"] is True
    assert row["weave_agent_bridge_closed_verified"] is True
    assert (
        row["weave_agent_bridge_close_verification_source"]
        == "authoritative_call_graph"
    )
    assert row["weave_agent_bridge_close_poll_error"] == "ReadTimeout"
    assert "weave_agent_bridge_close_error" not in row


def test_authoritative_call_state_retries_a_transient_read_timeout() -> None:
    calls = 0

    def summary_fetcher(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("temporary trace API timeout")
        return {
            "run-key": {
                "weave_authoritative_call_graph": [
                    {
                        "call_id": "bridge",
                        "project_id": "entity/project",
                        "terminal": True,
                    }
                ]
            }
        }

    coordinator = object.__new__(LiveEvaluationCoordinator)
    coordinator.project = "entity/project"
    coordinator.trace_timeout_sec = 1
    coordinator.env = {}
    coordinator._summary_fetcher = summary_fetcher
    coordinator._cancellation_event = threading.Event()

    coordinator._wait_for_authoritative_call_state(
        row={"run_key": "run-key"},
        call_id="bridge",
        terminal=True,
        phase="bridge close",
    )

    assert calls == 2


def test_invalid_authoritative_graph_does_not_recover_bridge_close() -> None:
    bridge_id = "c" * 16
    evaluation_trace_id = "d" * 32
    row = {
        "harness": "claude-code",
        "trace_project": "entity/project",
        "weave_evaluation_root_call_id": "evaluation",
        "eval_predict_and_score_call_id": "predict-and-score",
        "eval_predict_and_score_trace_id": evaluation_trace_id,
        "weave_prediction_call_id": "prediction",
        "weave_agent_bridge_call_id": bridge_id,
        "weave_agent_root_call_id": "agent-root",
        "weave_agent_bridge_closed_verified": False,
        "weave_agent_bridge_close_error": "ReadTimeout",
        "weave_authoritative_call_graph": [
            {
                "call_id": "evaluation",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": "predict-and-score",
                "parent_id": "evaluation",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": "prediction",
                "parent_id": "predict-and-score",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": bridge_id,
                "parent_id": "wrong-prediction",
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
            {
                "call_id": "agent-root",
                "parent_id": bridge_id,
                "project_id": "entity/project",
                "trace_id": evaluation_trace_id,
                "terminal": True,
            },
        ],
        "weave_authoritative_missing_call_ids": [],
    }

    export._verify_authoritative_agent_graph(row)

    assert row["weave_authoritative_call_graph_verified"] is False
    assert row["weave_agent_bridge_closed_verified"] is False
    assert row["weave_agent_bridge_close_error"] == "ReadTimeout"
    assert "bridge parent" in row["weave_authoritative_call_graph_error"]


@pytest.mark.parametrize(
    ("field", "observed"),
    (
        ("attempt_id", "b" * 64),
        ("execution_fingerprint", "f" * 64),
    ),
)
def test_observed_root_rejects_attempt_or_runtime_drift(
    field: str,
    observed: str,
) -> None:
    expected = {
        "attempt_id": "a" * 64,
        "execution_fingerprint": "e" * 64,
    }
    root = {
        "agent_name": "codex",
        "run_key": "run-key",
        "task_id": "task-a",
        "candidate_id": "candidate-a",
        "comparison_example_id": "example-a",
        "trial_index": 1,
        "attempt_id": expected["attempt_id"],
        "execution_fingerprint": expected["execution_fingerprint"],
        "conversation_id": "native-session",
        "trace_id": "a" * 32,
        "span_id": "b" * 16,
    }
    root[field] = observed
    row = {
        **expected,
        "weave_agent_name": "codex",
        "run_key": "run-key",
        "task_name": "task-a",
        "candidate_id": "candidate-a",
        "comparison_example_id": "example-a",
        "trial_index": 1,
        "weave_root_spans": [root],
    }

    export._apply_observed_identity(row)

    assert row["trace_link_status"] == "missing"
    assert "no matching invoke_agent root" in row["trace_link_error"]


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
        [cell],
        repo_root=tmp_path,
        env={"PRIVATE_TOKEN": "secret-value"},
        host_evaluator=lambda row: row.update(
            {
                "comparison_evaluation_status": "scored",
                "comparison_deterministic_scores": {"fact-correct": True},
            }
        ),
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
    assert result["comparison_deterministic_scores"] == {"fact-correct": True}
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
        SimpleNamespace(
            init=lambda project, **kwargs: None,
            EvaluationLogger=FakeLogger,
        ),
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
        SimpleNamespace(
            init=lambda project, **kwargs: None,
            EvaluationLogger=object,
        ),
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


def test_authoritative_call_query_uses_exact_ids() -> None:
    requests = []

    class Response:
        status_code = 200
        text = '{"id":"evaluation"}\n{"id":"prediction"}\n'

    class Client:
        def post(self, url, *, json):
            requests.append((url, json))
            return Response()

    calls = export._fetch_call_ids(
        Client(),
        "https://trace.wandb.ai",
        "team/fugue-experiments",
        ["evaluation", "prediction", "prediction"],
    )

    assert calls == [{"id": "evaluation"}, {"id": "prediction"}]
    assert requests == [
        (
            "https://trace.wandb.ai/calls/stream_query",
            {
                "project_id": "team/fugue-experiments",
                "filter": {
                    "trace_roots_only": False,
                    "call_ids": ["evaluation", "prediction"],
                },
                "limit": 2,
            },
        )
    ]


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
    assert summary["otel_root_span_ids"] == ["turn"]
    assert "weave_root_span_ids" not in summary
    assert summary["weave_input_tokens"] == 12
    assert summary["weave_output_tokens"] == 3
    assert summary["weave_usage_source"] == "chat_sum"


def test_agent_root_accepts_external_bridge_parent_but_excludes_nested_agent() -> None:
    summary = _summarize_spans(
        [
            {
                "span_id": "a" * 16,
                "parent_span_id": "b" * 16,
                "trace_id": "c" * 32,
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.conversation.id": "conversation-1",
                },
            },
            {
                "span_id": "d" * 16,
                "parent_span_id": "a" * 16,
                "trace_id": "c" * 32,
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.conversation.id": "conversation-1",
                },
            },
        ]
    )

    assert summary["otel_root_span_ids"] == ["a" * 16]
    assert [root["span_id"] for root in summary["weave_root_spans"]] == [
        "a" * 16
    ]


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
        "fugue.attempt_id": "a" * 64,
        "fugue.execution_fingerprint": "e" * 64,
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


def test_agent_span_summary_requires_comparison_project_lineage() -> None:
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
        "fugue.attempt_id": "a" * 64,
        "fugue.execution_fingerprint": "e" * 64,
        "fugue.model_provider": "wandb",
        "fugue.model": "wandb/test-model",
        "wandb.study_id": "study-a",
    }
    summary = _summarize_spans(
        [
            {
                "span_id": "turn",
                "operation_name": "invoke_agent",
                "custom_attrs_string": fugue_attributes,
            }
        ]
    )

    assert summary["weave_attribute_status"] == "partial"
    assert summary["weave_missing_attributes"] == [
        "fugue.source_evidence_project",
        "fugue.result_evidence_project",
    ]


def test_agent_spans_are_authoritative_when_calls_repeat_same_activity() -> None:
    shared_tool = {
        "operation_name": "execute_tool",
        "tool_name": "query_project",
    }
    shared_chat = {
        "operation_name": "chat",
        "input_tokens": 12,
        "output_tokens": 3,
        "total_cost_usd": 0.25,
    }
    summary = _summarize_spans(
        [
                {
                    **shared_tool,
                    "span_id": "otel-tool",
                    "_fugue_evidence_source": export._WEAVE_AGENT_SPAN_SOURCE,
                },
                {
                    **shared_tool,
                    "id": "weave-call",
                    "_fugue_evidence_source": export._WEAVE_CALL_SOURCE,
                },
                {
                    **shared_chat,
                    "span_id": "otel-chat",
                    "_fugue_evidence_source": export._WEAVE_AGENT_SPAN_SOURCE,
                },
                {
                    **shared_chat,
                    "id": "weave-chat-call",
                    "_fugue_evidence_source": export._WEAVE_CALL_SOURCE,
                },
            ]
        )

    assert summary["weave_span_count"] == 2
    assert summary["weave_tool_call_count"] == 1
    assert summary["weave_tool_names"] == {"query_project": 1}
    assert summary["weave_input_tokens"] == 12
    assert summary["weave_output_tokens"] == 3
    assert summary["weave_total_cost_usd"] == 0.25


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


def test_interrupted_trial_recovers_exact_planned_cell_identity() -> None:
    identity = {
        "task_id": "routing-plan",
        "arm": "candidate",
        "harness": "claude-code",
        "attempt": 2,
        "candidate": "c" * 64,
        "runtime": "e" * 64,
    }
    shared = {
        "run_id": "run-1",
        "candidate_id": "c" * 64,
        "comparison_example_id": "x" * 64,
        "trial_index": 2,
        "execution_fingerprint": "e" * 64,
        "workload_id": "harbor",
        "harness": "claude-code",
        "context_system_id": "none",
        "variant_id": "candidate",
        "model_provider": "anthropic",
        "model": "anthropic/claude-sonnet-5",
    }
    trial = {
        "record_type": "trial",
        **shared,
        "task_name": "fugue/routing-plan",
    }
    cell = {
        "record_type": "cell",
        **shared,
        "cell_id": "cell-1",
        "task_id": "routing-plan",
        "attempt_id": "a" * 64,
        "attempt_identity": identity,
        "run_name": "skill-upgrade",
        "status": "interrupted",
        "runtime_outcome": "cancelled",
    }

    _merge_planned_cell_identities([trial, cell])

    assert trial["cell_id"] == "cell-1"
    assert trial["task_id"] == "routing-plan"
    assert trial["task_name"] == "routing-plan"
    assert trial["attempt_id"] == "a" * 64
    assert trial["attempt_identity"] == identity
    assert trial["status"] == "interrupted"
    assert trial["runtime_outcome"] == "cancelled"
    assert trial["planned_cell_identity_status"] == "reconciled"


def test_export_synthesizes_interrupted_trial_preserving_planned_identity(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    run_key = (
        "run-1:harbor:cell:routing-plan:claude-code:none:candidate:t001"
    )
    shared = {
        "cell_id": "cell-1",
        "run_key": run_key,
        "run_id": "run-1",
        "candidate_id": "c" * 64,
        "comparison_example_id": "x" * 64,
        "trial_index": 1,
        "execution_fingerprint": "e" * 64,
        "workload_id": "harbor",
        "harness": "claude-code",
        "context_system_id": "none",
        "variant_id": "candidate",
        "model_provider": "anthropic",
        "model": "anthropic/claude-sonnet-5",
        "task_id": "routing-plan",
        "attempt_id": "a" * 64,
        "attempt_identity": {
            "task_id": "routing-plan",
            "arm": "candidate",
            "harness": "claude-code",
            "attempt": 1,
            "candidate": "c" * 64,
            "runtime": "e" * 64,
        },
    }
    (jobs / "cells.jsonl").write_text(
        json.dumps({**shared, "status": "interrupted"}) + "\n",
        encoding="utf-8",
    )
    (jobs / "evaluation-results.jsonl").write_text(
        json.dumps(
            {
                **shared,
                "task_id": "",
                "attempt_id": "",
                "attempt_identity": None,
                "candidate_id": "",
                "execution_fingerprint": None,
                "status": "interrupted",
                "runtime_outcome": "cancelled",
                "comparison_evaluation_status": "unavailable",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = export_rows([jobs], repo_root=tmp_path)
    trials = [row for row in rows if row.get("record_type") == "trial"]

    assert len(trials) == 1
    assert trials[0]["task_id"] == "routing-plan"
    assert trials[0]["attempt_id"] == "a" * 64
    assert trials[0]["attempt_identity"] == shared["attempt_identity"]
    assert trials[0]["candidate_id"] == "c" * 64
    assert trials[0]["execution_fingerprint"] == "e" * 64
    assert trials[0]["status"] == "interrupted"
    assert trials[0]["runtime_outcome"] == "cancelled"
    assert trials[0]["planned_cell_identity_status"] == (
        "synthesized_from_live_terminal"
    )


def test_synthesized_interrupted_trial_rejects_live_identity_disagreement() -> None:
    planned = {
        "record_type": "cell",
        "cell_id": "cell-1",
        "run_key": "run-1:harbor:cell:routing-plan:claude-code:none:candidate:t001",
        "task_id": "routing-plan",
        "attempt_id": "a" * 64,
        "candidate_id": "c" * 64,
        "execution_fingerprint": "e" * 64,
        "trial_index": 1,
        "harness": "claude-code",
        "context_system_id": "none",
        "variant_id": "candidate",
    }
    live = {
        **planned,
        "record_type": "live_evaluation",
        "candidate_id": "d" * 64,
        "status": "interrupted",
    }

    with pytest.raises(ValueError, match="candidate_id"):
        _synthesize_missing_trial_rows([planned], [live])


def test_interrupted_trial_rejects_conflicting_planned_identity() -> None:
    shared = {
        "run_id": "run-1",
        "candidate_id": "c" * 64,
        "comparison_example_id": "x" * 64,
        "trial_index": 1,
        "execution_fingerprint": "e" * 64,
        "workload_id": "harbor",
        "harness": "claude-code",
        "context_system_id": "none",
        "variant_id": "candidate",
        "model_provider": "anthropic",
        "model": "anthropic/claude-sonnet-5",
    }
    trial = {"record_type": "trial", **shared, "task_id": "wrong-task"}
    cell = {
        "record_type": "cell",
        **shared,
        "cell_id": "cell-1",
        "task_id": "right-task",
        "attempt_id": "a" * 64,
        "attempt_identity": {},
        "run_name": "skill-upgrade",
    }

    with pytest.raises(ValueError, match="task_id"):
        _merge_planned_cell_identities([trial, cell])
