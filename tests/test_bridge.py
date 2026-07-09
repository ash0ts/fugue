from __future__ import annotations

from fugue.bridge import docker_compose_for_route, litellm_config_for_route
from fugue.model_plane import resolve_model_route


def test_wandb_bridge_config_keeps_nebius_mapping() -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    config = litellm_config_for_route(route)

    params = config["model_list"][0]["litellm_params"]
    assert params["model"] == "nebius/*"
    assert params["api_base"] == "https://api.inference.wandb.ai/v1"
    assert params["api_key"] == "os.environ/WANDB_API_KEY"


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
