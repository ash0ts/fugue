from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

import pytest
from pydantic import TypeAdapter

pytest.importorskip("fastapi")
from ag_ui.core import Event
from fastapi.testclient import TestClient
from harbor import JobConfig

from fugue.serve import runtime as serve_runtime
from fugue.serve.app import create_app
from fugue.serve.runtime import HarborWorkerBackend, WorkerRequest, render_conversation
from fugue.serve.worker import extract_final_answer


def _deployment() -> dict:
    return {
        "deployment_id": "deployment-1",
        "candidate_id": "candidate-abcdef1234567890",
        "experiment_id": "demo",
        "harness": "codex",
        "variant_id": "baseline",
        "context_system_id": "none",
        "context_version": "1",
        "context_config_hash": "context-hash",
        "agent_config_hash": "agent-hash",
        "model": "openai/gpt-5",
        "model_provider": "openai",
        "required_env": ["CUSTOM_TOKEN", "OPENAI_API_KEY", "WANDB_API_KEY"],
        "network_allowed_hosts": ["api.openai.com", "*.wandb.ai"],
        "resources": {
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 10240,
            "timeout_sec": 900,
        },
        "image": "example/fugue:test",
        "candidate": {
            "trace_content": "full",
            "model_route": {
                "provider": "openai",
                "responses_base_url": "https://locked.example/v1",
            },
            "required_env": ["CUSTOM_TOKEN", "OPENAI_API_KEY", "WANDB_API_KEY"],
            "agent": {
                "import_path": "fugue.agents:FugueCodex",
                "model_name": "openai/gpt-5",
                "env": {"CUSTOM_TOKEN": "${CUSTOM_TOKEN}"},
            },
            "environment": {},
            "extra_instruction_paths": [],
        },
    }


class FakeBackend:
    def __init__(
        self,
        *,
        answer: str = "served answer",
        delay: float = 0,
        error: Exception | None = None,
        ready: bool = True,
    ) -> None:
        self.deployment = _deployment()
        self.answer = answer
        self.delay = delay
        self.error = error
        self.ready = ready
        self.requests: list[WorkerRequest] = []

    def readiness(self):
        return (self.ready, () if self.ready else ("OPENAI_API_KEY",))

    async def run(self, request: WorkerRequest) -> str:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.answer


def _client(backend: FakeBackend, **kwargs) -> TestClient:
    app = create_app(
        backend=backend,
        env={
            "FUGUE_SERVE_API_KEY": "serve-secret",
            "FUGUE_SERVE_CORS_ORIGINS": "https://app.example",
        },
        **kwargs,
    )
    return TestClient(app)


def _auth() -> dict[str, str]:
    return {"Authorization": "Bearer serve-secret"}


def _sse_events(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: {")
    ]


def test_auth_cors_health_and_readiness() -> None:
    backend = FakeBackend(ready=False)
    client = _client(backend)

    assert client.get("/healthz").status_code == 200
    readiness = client.get("/readyz")
    assert readiness.status_code == 503
    assert readiness.json()["missing"] == ["OPENAI_API_KEY"]
    unauthorized = client.get("/v1/models")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "invalid_api_key"
    models = client.get("/v1/models", headers=_auth())
    assert models.status_code == 200
    assert models.json()["data"][0]["id"].startswith("fugue-candidate-")
    preflight = client.options(
        "/v1/responses",
        headers={
            "Origin": "https://app.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "https://app.example"


def test_responses_and_chat_sync_preserve_full_history() -> None:
    backend = FakeBackend()
    client = _client(backend)
    history = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow up"},
    ]

    response = client.post(
        "/v1/responses",
        headers=_auth(),
        json={"model": "ignored", "input": history},
    )
    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "served answer"
    chat = client.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={"messages": history},
    )
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "served answer"
    assert backend.requests[0].messages == tuple(history)
    assert backend.requests[1].messages == tuple(history)


def test_open_responses_2026_04_24_basic_and_streaming_subset() -> None:
    client = _client(FakeBackend())
    response = client.post(
        "/v1/responses", headers=_auth(), json={"input": "hello"}
    )
    assert response.status_code == 200
    resource = response.json()
    required_resource_fields = {
        "id",
        "object",
        "created_at",
        "completed_at",
        "status",
        "incomplete_details",
        "model",
        "previous_response_id",
        "instructions",
        "output",
        "error",
        "tools",
        "tool_choice",
        "truncation",
        "parallel_tool_calls",
        "text",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "top_logprobs",
        "temperature",
        "reasoning",
        "usage",
        "max_output_tokens",
        "max_tool_calls",
        "store",
        "background",
        "service_tier",
        "metadata",
        "safety_identifier",
        "prompt_cache_key",
    }
    assert required_resource_fields <= resource.keys()
    assert resource["object"] == "response"
    assert resource["status"] == "completed"

    streamed = client.post(
        "/v1/responses",
        headers=_auth(),
        json={"input": "hello", "stream": True},
    )
    assert streamed.status_code == 200
    events = _sse_events(streamed.text)
    assert [event["sequence_number"] for event in events] == list(range(9))
    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[4]["delta"] == "served answer"
    assert events[-1]["response"]["status"] == "completed"


def test_all_protocols_stream_lifecycle_heartbeat_and_one_final_delta() -> None:
    backend = FakeBackend(delay=0.02)
    client = _client(backend, heartbeat_sec=0.001)
    message = [{"role": "user", "content": "Hello"}]

    responses = client.post(
        "/v1/responses",
        headers=_auth(),
        json={"input": message, "stream": True},
    ).text
    assert "response.created" in responses
    assert ": heartbeat" in responses
    assert responses.count("response.output_text.delta") == 1
    assert "response.completed" in responses

    chat = client.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={"messages": message, "stream": True},
    ).text
    assert ": heartbeat" in chat
    assert chat.count("served answer") == 1
    assert "data: [DONE]" in chat

    ag_ui = client.post(
        "/ag-ui",
        headers=_auth(),
        json={"threadId": "thread-1", "runId": "run-1", "messages": message},
    ).text
    assert "RUN_STARTED" in ag_ui
    assert "TEXT_MESSAGE_CONTENT" in ag_ui
    assert ag_ui.count("served answer") == 1
    assert "RUN_FINISHED" in ag_ui
    ag_ui_events = _sse_events(ag_ui)
    adapter = TypeAdapter(Event)
    assert all(adapter.validate_python(event) for event in ag_ui_events)


@pytest.mark.parametrize(
    ("path", "payload", "parameter"),
    (
        ("/v1/responses", {"input": "hello", "tools": [{"type": "function"}]}, "tools"),
        ("/v1/responses", {"input": "hello", "previous_response_id": "resp_old"}, "previous_response_id"),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, "messages"),
        ("/ag-ui", {"messages": [{"role": "user", "content": "hello"}], "tools": [{}]}, "tools"),
    ),
)
def test_unsupported_client_features_are_rejected(
    path: str, payload: dict, parameter: str
) -> None:
    response = _client(FakeBackend()).post(path, headers=_auth(), json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_feature"
    assert response.json()["error"]["param"] == parameter


def test_worker_timeout_and_error_are_normalized() -> None:
    timed_out = _client(FakeBackend(delay=0.05), timeout_sec=0.001).post(
        "/v1/responses", headers=_auth(), json={"input": "hello"}
    )
    assert timed_out.status_code == 504
    assert timed_out.json()["error"]["code"] == "worker_timeout"
    failed = _client(FakeBackend(error=RuntimeError("worker broke"))).post(
        "/v1/chat/completions",
        headers=_auth(),
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "worker_error"
    assert failed.json()["error"]["message"] == "worker failed"
    assert "worker broke" not in failed.text


def test_websocket_and_compaction_are_explicitly_rejected() -> None:
    client = _client(FakeBackend())
    compact = client.post(
        "/v1/responses/compact", headers=_auth(), json={"input": "hello"}
    )
    assert compact.status_code == 400
    assert compact.json()["error"]["code"] == "unsupported_feature"
    with client.websocket_connect(
        "/v1/responses", headers=_auth()
    ) as websocket:
        event = websocket.receive_json()
    assert event["error"]["code"] == "unsupported_feature"


def test_harbor_config_preserves_candidate_and_enforces_request_policy(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "deployment.json"
    spec_path.write_text(json.dumps(_deployment()))
    backend = HarborWorkerBackend(
        spec_path,
        runtime_dir=tmp_path / "runtime",
        env={
            "OPENAI_API_KEY": "model-secret",
            "WANDB_API_KEY": "trace-secret",
            "CUSTOM_TOKEN": "candidate-secret",
            "FUGUE_SERVE_API_KEY": "serve-secret",
            "FUGUE_SERVE_HARBOR_ENVIRONMENT": "docker",
        },
    )
    task = tmp_path / "task"
    task.mkdir()
    request = WorkerRequest(
        "req-1", "open-responses", ({"role": "user", "content": "hello"},)
    )
    config = backend._job_config(request, task)
    validated = JobConfig.model_validate(config)

    assert validated.n_attempts == 1
    assert config["verifier"] == {"disable": True}
    assert config["agents"][0]["import_path"] == "fugue.agents:FugueCodex"
    assert config["agents"][0]["model_name"] == "openai/gpt-5"
    assert config["agents"][0]["env"]["OPENAI_BASE_URL"] == (
        "https://locked.example/v1"
    )
    assert config["agents"][0]["env"]["CUSTOM_TOKEN"] == "candidate-secret"
    assert config["agents"][0]["env"]["FUGUE_CANDIDATE_ID"].startswith("candidate-")
    assert config["environment"]["delete"] is True
    assert config["environment"]["override_cpus"] == 2
    task_toml = backend._task_toml()
    assert 'network_mode = "allowlist"' in task_toml
    assert 'docker_image = "example/fugue:test"' in task_toml
    assert "model-secret" not in task_toml
    assert "trace-secret" not in task_toml
    request_env = backend._request_env(request)
    assert request_env["OPENAI_BASE_URL"] == "https://locked.example/v1"
    assert "FUGUE_SERVE_API_KEY" not in request_env


def test_runtime_directory_comes_from_the_supplied_environment(tmp_path: Path) -> None:
    spec_path = tmp_path / "deployment.json"
    spec_path.write_text(json.dumps(_deployment()))
    runtime = tmp_path / "custom-runtime"

    backend = HarborWorkerBackend(
        spec_path,
        env={"FUGUE_SERVE_RUNTIME_DIR": runtime.as_posix()},
    )

    assert backend.runtime_dir == runtime


def test_one_isolated_worker_per_request_and_ephemeral_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = tmp_path / "deployment.json"
    spec_path.write_text(json.dumps(_deployment()))
    runtime = tmp_path / "runtime"
    backend = HarborWorkerBackend(
        spec_path,
        runtime_dir=runtime,
        env={
            "OPENAI_API_KEY": "model-secret",
            "WANDB_API_KEY": "trace-secret",
            "CUSTOM_TOKEN": "candidate-secret",
        },
    )
    configs: list[Path] = []

    async def spawn(*args, **kwargs):
        config_path = Path(args[-2])
        result_path = Path(args[-1])
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
        configs.append(config_path)
        result_path.write_text(
            json.dumps({"status": "completed", "answer": "isolated answer"})
        )

        class Process:
            returncode = None
            pid = 1

            async def wait(self):
                self.returncode = 0
                return 0

        return Process()

    monkeypatch.setattr(serve_runtime.asyncio, "create_subprocess_exec", spawn)

    async def run_requests():
        return await asyncio.gather(
            backend._run_isolated(
                WorkerRequest(
                    "req-one",
                    "open-responses",
                    ({"role": "user", "content": "one"},),
                )
            ),
            backend._run_isolated(
                WorkerRequest(
                    "req-two",
                    "chat-completions",
                    ({"role": "user", "content": "two"},),
                )
            ),
        )

    assert asyncio.run(run_requests()) == ["isolated answer", "isolated answer"]
    assert len(configs) == 2
    assert configs[0].parent != configs[1].parent
    assert not any(runtime.iterdir())


def test_history_rendering_and_native_final_answer_extraction(tmp_path: Path) -> None:
    rendered = render_conversation(
        (
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        )
    )
    assert rendered.index("first") < rendered.index("second") < rendered.index("third")

    job_dir = tmp_path / "job"
    logs = job_dir / "trial/agent"
    logs.mkdir(parents=True)
    (logs / "codex.jsonl").write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "final native response"},
            }
        )
        + "\n"
    )
    result = type("Result", (), {"trial_results": []})()
    assert extract_final_answer(result, job_dir) == "final native response"

    atif_dir = tmp_path / "atif-job"
    atif_logs = atif_dir / "trial/agent"
    atif_logs.mkdir(parents=True)
    (atif_logs / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "user", "message": "question"},
                    {"source": "agent", "message": "intermediate response"},
                    {"source": "agent", "message": "final ATIF response"},
                ]
            }
        )
    )
    assert extract_final_answer(result, atif_dir) == "final ATIF response"
