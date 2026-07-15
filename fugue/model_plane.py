from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Provider = Literal["wandb", "openai", "anthropic"]
ToolResultModality = Literal["text", "image"]

DEFAULT_MODEL = "wandb/zai-org/GLM-5.2"
DEFAULT_WANDB_ENTITY = "wandb"
DEFAULT_WANDB_PROJECT = "fugue-experiments"
WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
BRIDGE_BASE_URL_HOST = "http://127.0.0.1:4000"
BRIDGE_BASE_URL_CONTAINER = "http://host.docker.internal:4000"
BRIDGE_MASTER_KEY_ENV = "LITELLM_MASTER_KEY"
DEFAULT_BRIDGE_MASTER_KEY = "sk-fugue-local"
WANDB_INFERENCE_PROJECT_HEADER = "OpenAI-Project"
OPENAI_PROJECT_ENV = "OPENAI_PROJECT"
OPENAI_PROJECT_ID_ENV = "OPENAI_PROJECT_ID"

# GLM-5.2 rejected image-bearing tool results through both bridge protocols in
# the release canary. Keep this model-specific: other W&B routes may be visual.
_TEXT_ONLY_MODEL_ROUTES = {("wandb", "zai-org/GLM-5.2")}


@dataclass(frozen=True)
class ModelRoute:
    provider: Provider
    model_id: str
    display_model: str
    api_key_env: str
    chat_base_url: str | None
    responses_base_url: str | None
    messages_base_url: str | None
    litellm_model: str
    tool_result_modalities: tuple[ToolResultModality, ...]


def select_model(
    cli_model: str | None = None,
    manifest_model: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    harness_model: str | None = None,
    experiment_model: str | None = None,
) -> str:
    values = env if env is not None else os.environ
    for candidate in (
        cli_model,
        harness_model,
        experiment_model,
        manifest_model,
        values.get("FUGUE_MODEL"),
        DEFAULT_MODEL,
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return DEFAULT_MODEL


def resolve_model_route(
    model_name: str | None = None, env: Mapping[str, str] | None = None
) -> ModelRoute:
    values = env if env is not None else os.environ
    raw = select_model(model_name, env=values) if model_name is None else model_name
    raw = raw.strip()
    if "/" not in raw:
        raise ValueError(
            "model must include a provider prefix: wandb/..., openai/..., "
            "or anthropic/..."
        )
    provider_raw, model_id = raw.split("/", 1)
    if not model_id:
        raise ValueError(f"model id is empty in {raw!r}")
    provider = provider_raw.lower()

    if provider == "wandb":
        base_url = values.get("WANDB_INFERENCE_BASE_URL", WANDB_INFERENCE_BASE_URL)
        return ModelRoute(
            provider="wandb",
            model_id=model_id,
            display_model=f"wandb/{model_id}",
            api_key_env="WANDB_API_KEY",
            chat_base_url=base_url.rstrip("/"),
            responses_base_url=None,
            messages_base_url=None,
            litellm_model="nebius/*",
            tool_result_modalities=_tool_result_modalities("wandb", model_id),
        )
    if provider == "openai":
        base_url = values.get("OPENAI_BASE_URL", OPENAI_BASE_URL)
        return ModelRoute(
            provider="openai",
            model_id=model_id,
            display_model=f"openai/{model_id}",
            api_key_env="OPENAI_API_KEY",
            chat_base_url=base_url.rstrip("/"),
            responses_base_url=base_url.rstrip("/"),
            messages_base_url=None,
            litellm_model="openai/*",
            tool_result_modalities=_tool_result_modalities("openai", model_id),
        )
    if provider == "anthropic":
        base_url = values.get("ANTHROPIC_BASE_URL", ANTHROPIC_BASE_URL)
        return ModelRoute(
            provider="anthropic",
            model_id=model_id,
            display_model=f"anthropic/{model_id}",
            api_key_env="ANTHROPIC_API_KEY",
            chat_base_url=None,
            responses_base_url=None,
            messages_base_url=base_url.rstrip("/"),
            litellm_model="anthropic/*",
            tool_result_modalities=_tool_result_modalities("anthropic", model_id),
        )

    raise ValueError(
        f"unknown model provider {provider_raw!r}; expected wandb, openai, "
        "or anthropic"
    )


def model_route_identity(route: ModelRoute) -> dict[str, object]:
    return {
        "provider": route.provider,
        "model_id": route.model_id,
        "display_model": route.display_model,
        "chat_base_url": route.chat_base_url,
        "responses_base_url": route.responses_base_url,
        "messages_base_url": route.messages_base_url,
        "litellm_model": route.litellm_model,
        "tool_result_modalities": list(route.tool_result_modalities),
    }


def _tool_result_modalities(
    provider: Provider, model_id: str
) -> tuple[ToolResultModality, ...]:
    if (provider, model_id) in _TEXT_ONLY_MODEL_ROUTES:
        return ("text",)
    return ("text", "image")


def bridge_master_key(env: Mapping[str, str] | None = None) -> str:
    values = env if env is not None else os.environ
    return values.get(BRIDGE_MASTER_KEY_ENV, DEFAULT_BRIDGE_MASTER_KEY)


def missing_model_env(
    route: ModelRoute, env: Mapping[str, str] | None = None
) -> list[str]:
    values = env if env is not None else os.environ
    return [] if values.get(route.api_key_env, "").strip() else [route.api_key_env]


def missing_trace_env(env: Mapping[str, str] | None = None) -> list[str]:
    values = env if env is not None else os.environ
    return [key for key in ("WANDB_API_KEY",) if not values.get(key, "").strip()]


def trace_entity_project(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    values = env if env is not None else os.environ
    slug = values.get("WEAVE_PROJECT", "").strip()
    if slug and "/" in slug:
        entity, project = slug.split("/", 1)
        return entity, project
    entity = values.get("WANDB_ENTITY", "").strip() or DEFAULT_WANDB_ENTITY
    project = values.get("WANDB_PROJECT", "").strip() or DEFAULT_WANDB_PROJECT
    return entity, project


def trace_project_slug(env: Mapping[str, str] | None = None) -> str:
    entity, project = trace_entity_project(env)
    return f"{entity}/{project}"


def trace_env_defaults(env: Mapping[str, str] | None = None) -> dict[str, str]:
    entity, project = trace_entity_project(env)
    return {
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project,
        "WEAVE_PROJECT": f"{entity}/{project}",
    }


def provider_request_headers(
    route: ModelRoute, env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return non-secret headers required by the selected model provider."""
    if route.provider != "wandb":
        return {}
    return {WANDB_INFERENCE_PROJECT_HEADER: trace_project_slug(env)}


def provider_client_env(
    route: ModelRoute, env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return SDK environment needed for provider-specific request metadata."""
    if route.provider != "wandb":
        return {}
    project = trace_project_slug(env)
    return {
        OPENAI_PROJECT_ENV: project,
        OPENAI_PROJECT_ID_ENV: project,
    }
