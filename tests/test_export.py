import json
import sys
from pathlib import Path
from types import SimpleNamespace

from fugue.bench import export
from fugue.bench.export import (
    _fetch_agents_spans,
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
    monkeypatch.setenv("WANDB_BASE_URL", "https://api.wandb.test")

    class FakeLogger:
        def __init__(self, *, model, dataset) -> None:
            assert model == "fugue"
            assert dataset == "fugue-context-evaluation"

        def log_example(self, inputs, output, scores) -> None:
            calls.append((inputs, output, scores))

    fake_weave = SimpleNamespace(
        init=lambda project: calls.append(
            ("init", project, __import__("os").environ.get("WANDB_BASE_URL"))
        ),
        EvaluationLogger=FakeLogger,
    )
    monkeypatch.setitem(sys.modules, "weave", fake_weave)
    project = f"entity/project-{tmp_path.name}"
    rows = [{"record_type": "trial", "task_name": "task", "reward": 1.0}]

    env = {
        "WANDB_API_KEY": "test-only",
        "WANDB_BASE_URL": "https://api.wandb.ai",
    }
    assert publish_to_weave(rows, project, ledger_root=tmp_path, env=env) == 1
    assert publish_to_weave(rows, project, ledger_root=tmp_path) == 0
    assert publish_to_weave(
        rows, project, ledger_root=tmp_path, republish=True
    ) == 1

    examples = [item for item in calls if isinstance(item, tuple) and len(item) == 3]
    logged_examples = [item for item in examples if item[0] != "init"]
    assert len(logged_examples) == 2
    assert logged_examples[0][0]["publication_id"] == logged_examples[1][0]["publication_id"]
    assert logged_examples[0][2] == {"reward": 1.0}
    assert ("init", project, "https://api.wandb.ai") in calls


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
        "https://api.wandb.ai",
        "team/fugue-experiments",
        ["conversation-1", "conversation-1"],
    )

    assert spans == [{"id": "turn-1"}]
    assert requests == [
        (
            "https://api.wandb.ai/agents/spans/query",
            {
                "project_id": "team/fugue-experiments",
                "filter": {"conversation_id": "conversation-1"},
            },
        )
    ]


def test_agent_span_summary_counts_logical_hierarchy_once() -> None:
    spans = [
        {
            "id": "turn",
            "trace_id": "trace-1",
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.conversation.id": "conversation-1",
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
    assert summary["weave_conversation_ids"] == ["conversation-1"]
    assert summary["weave_root_span_ids"] == ["turn"]
    assert summary["weave_input_tokens"] == 12
    assert summary["weave_output_tokens"] == 3
