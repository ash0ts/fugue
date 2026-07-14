from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Provider = Literal["wandb", "openai", "anthropic"]

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
        )

    raise ValueError(
        f"unknown model provider {provider_raw!r}; expected wandb, openai, "
        "or anthropic"
    )


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


def env_presence(
    keys: list[str], env: Mapping[str, str] | None = None
) -> dict[str, bool]:
    values = env if env is not None else os.environ
    return {key: bool(values.get(key, "").strip()) for key in keys}
