from __future__ import annotations

import json
import subprocess

import yaml

from fugue.bridge import (
    BRIDGE_PORT_ENV,
    LITELLM_IMAGE,
    bridge_container_base_url,
    bridge_container_name,
    bridge_port,
    bridge_project_name,
    bridge_runtime_lock_for_route,
    bridge_status,
    bridge_up,
    docker_compose_for_route,
    litellm_config_for_route,
    write_bridge_files,
)
from fugue.model_plane import resolve_model_route


def test_wandb_bridge_config_keeps_nebius_mapping() -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    config = litellm_config_for_route(
        route, env={"WANDB_ENTITY": "wandb", "WANDB_PROJECT": "billing-project"}
    )

    params = config["model_list"][0]["litellm_params"]
    assert params["model"] == "nebius/*"
    assert params["api_base"] == "https://api.inference.wandb.ai/v1"
    assert params["api_key"] == "os.environ/WANDB_API_KEY"
    assert params["extra_headers"] == {"OpenAI-Project": "wandb/billing-project"}


def test_openai_bridge_config_uses_env_reference_only() -> None:
    route = resolve_model_route("openai/gpt-5", {"OPENAI_API_KEY": "sk-secret"})
    config = litellm_config_for_route(route)
    compose = docker_compose_for_route(route)

    params = config["model_list"][0]["litellm_params"]
    assert params["model"] == "openai/*"
    assert params["api_key"] == "os.environ/OPENAI_API_KEY"
    assert "sk-secret" not in str(config)
    assert compose["services"]["bridge"]["environment"]["OPENAI_API_KEY"] == (
        "${OPENAI_API_KEY}"
    )


def test_anthropic_bridge_config_uses_messages_base_url() -> None:
    route = resolve_model_route("anthropic/claude-sonnet-4-5", {})
    config = litellm_config_for_route(route)

    params = config["model_list"][0]["litellm_params"]
    assert params["model"] == "anthropic/*"
    assert params["api_base"] == "https://api.anthropic.com"
    assert params["api_key"] == "os.environ/ANTHROPIC_API_KEY"


def test_bridge_uses_distinct_role_aliases_and_pinned_image() -> None:
    target = resolve_model_route("openai/gpt-5", {})
    builder = resolve_model_route("anthropic/claude-sonnet-4-5", {})
    judge = resolve_model_route("wandb/zai-org/GLM-5.2", {})

    config = litellm_config_for_route(
        target,
        builder_route=builder,
        judge_route=judge,
    )
    aliases = {item["model_name"]: item for item in config["model_list"]}

    assert aliases["fugue-target"]["litellm_params"]["model"] == "openai/gpt-5"
    assert aliases["fugue-builder"]["litellm_params"]["model"] == (
        "anthropic/claude-sonnet-4-5"
    )
    assert aliases["fugue-judge"]["litellm_params"]["model"] == (
        "nebius/zai-org/GLM-5.2"
    )
    compose = docker_compose_for_route(
        target,
        builder_route=builder,
        judge_route=judge,
    )
    assert compose["services"]["bridge"]["image"] == LITELLM_IMAGE
    assert "@sha256:" in LITELLM_IMAGE


def test_bridge_operator_port_is_strict_and_container_scoped() -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    env = {BRIDGE_PORT_ENV: "14017"}
    compose = docker_compose_for_route(route, env=env)
    lock = bridge_runtime_lock_for_route(route, env=env)

    assert bridge_port(env) == 14017
    assert bridge_container_name(env) == "fugue-litellm-bridge-14017"
    assert bridge_project_name(env) == "fugue-bridge-14017"
    assert bridge_container_base_url(env) == "http://host.docker.internal:14017"
    assert compose["services"]["bridge"]["container_name"] == (
        "fugue-litellm-bridge-14017"
    )
    assert compose["services"]["bridge"]["ports"] == ["127.0.0.1:14017:4000"]
    assert lock["host_port"] == 14017
    assert lock["container_name"] == "fugue-litellm-bridge-14017"
    assert lock["project_name"] == "fugue-bridge-14017"


def test_bridge_operator_port_rejects_non_loopback_indirection() -> None:
    for raw in ("https://example.com", "0", "70000", " 4000/tcp "):
        try:
            bridge_port({BRIDGE_PORT_ENV: raw})
        except ValueError:
            continue
        raise AssertionError(f"accepted unsafe bridge port {raw!r}")


def test_bridge_files_include_a_secret_free_route_and_image_lock(tmp_path) -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    env = {
        "WANDB_API_KEY": "sk-secret",
        "WANDB_ENTITY": "wandb",
        "WANDB_PROJECT": "route-proof",
    }

    files = write_bridge_files(route, repo_root=tmp_path, env=env)
    lock = json.loads(files.lock_path.read_text(encoding="utf-8"))

    assert lock == bridge_runtime_lock_for_route(route, env=env)
    assert lock["image"] == LITELLM_IMAGE
    assert "sk-secret" not in files.lock_path.read_text(encoding="utf-8")


def test_bridge_status_fails_closed_when_runtime_lock_drifted(
    monkeypatch, tmp_path
) -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    files = write_bridge_files(route, repo_root=tmp_path, env={})
    lock = json.loads(files.lock_path.read_text(encoding="utf-8"))
    lock["config_sha256"] = "0" * 64
    files.lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(
        "fugue.bridge.httpx.get",
        lambda *_args, **_kwargs: type(
            "Response",
            (),
            {"status_code": 200, "json": lambda self: {"status": "ok"}},
        )(),
    )

    status = bridge_status(repo_root=tmp_path, route=route, env={})

    assert status["ok"] is False
    assert "differs from the selected route" in status["error"]


def test_bridge_status_fails_closed_when_mounted_config_drifted(
    monkeypatch, tmp_path
) -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    files = write_bridge_files(route, repo_root=tmp_path, env={})
    config = yaml.safe_load(files.config_path.read_text(encoding="utf-8"))
    config["litellm_settings"]["drop_params"] = False
    files.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        "fugue.bridge.httpx.get",
        lambda *_args, **_kwargs: type(
            "Response",
            (),
            {"status_code": 200, "json": lambda self: {"status": "ok"}},
        )(),
    )

    status = bridge_status(repo_root=tmp_path, route=route, env={})

    assert status["ok"] is False
    assert "bridge config differs from its runtime lock" in status["error"]


def test_bridge_status_attests_running_image_command_and_mount(
    monkeypatch, tmp_path
) -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    files = write_bridge_files(route, repo_root=tmp_path, env={})
    command = docker_compose_for_route(route)["services"]["bridge"]["command"]
    container = {
        "Config": {"Image": LITELLM_IMAGE, "Cmd": command},
        "Image": "sha256:" + "f" * 64,
        "NetworkSettings": {
            "Ports": {"4000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "4000"}]}
        },
        "Mounts": [
            {
                "Destination": "/app/config.yaml",
                "Source": files.config_path.resolve().as_posix(),
                "RW": False,
            }
        ],
    }
    monkeypatch.setattr(
        "fugue.bridge.httpx.get",
        lambda *_args, **_kwargs: type(
            "Response",
            (),
            {"status_code": 200, "json": lambda self: {"status": "ok"}},
        )(),
    )
    monkeypatch.setattr(
        "fugue.bridge.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps([container]), ""
        ),
    )

    status = bridge_status(repo_root=tmp_path, route=route, env={})

    assert status["ok"] is True
    assert status["runtime_lock"]["image"] == LITELLM_IMAGE
    assert status["resolved_image_id"] == "sha256:" + "f" * 64


def test_bridge_up_reloads_generated_config(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr("fugue.bridge.subprocess.run", fake_run)
    monkeypatch.setattr("fugue.bridge.bridge_status", lambda **_kwargs: {"ok": True})

    bridge_up("wandb/zai-org/GLM-5.2", repo_root=tmp_path, env={})

    command, kwargs = calls[0]
    assert command[-3:] == ["up", "-d", "--force-recreate"]
    assert command[2:4] == ["--project-name", "fugue-bridge-4000"]
    assert kwargs["check"] is True


def test_bridge_up_waits_for_readiness(monkeypatch, tmp_path) -> None:
    statuses = iter(
        [
            {"ok": False, "error": "starting"},
            {"ok": True, "status_code": 200},
        ]
    )
    monkeypatch.setattr("fugue.bridge.subprocess.run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("fugue.bridge.bridge_status", lambda **_kwargs: next(statuses))
    monkeypatch.setattr("fugue.bridge.time.sleep", lambda _seconds: None)

    files = bridge_up("wandb/zai-org/GLM-5.2", repo_root=tmp_path, env={})

    assert files.config_path.is_file()
