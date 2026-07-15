from __future__ import annotations

from fugue.bridge import (
    LITELLM_IMAGE,
    bridge_up,
    docker_compose_for_route,
    litellm_config_for_route,
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
    assert params["extra_headers"] == {
        "OpenAI-Project": "wandb/billing-project"
    }


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
    assert "latest" not in LITELLM_IMAGE


def test_bridge_up_reloads_generated_config(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr("fugue.bridge.subprocess.run", fake_run)
    monkeypatch.setattr("fugue.bridge.bridge_status", lambda **_kwargs: {"ok": True})

    bridge_up("wandb/zai-org/GLM-5.2", repo_root=tmp_path, env={})

    command, kwargs = calls[0]
    assert command[-3:] == ["up", "-d", "--force-recreate"]
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
