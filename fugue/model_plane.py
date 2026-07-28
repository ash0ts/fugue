from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

Provider = Literal["wandb", "openai", "anthropic"]
ToolResultModality = Literal["text", "image"]
ModelWireProtocol = Literal["chat_completions", "messages", "responses"]

DEFAULT_MODEL = "wandb/zai-org/GLM-5.2"
DEFAULT_WANDB_ENTITY = "wandb"
DEFAULT_WANDB_PROJECT = "fugue-experiments"
WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
WANDB_INFERENCE_API_KEY_ENV = "FUGUE_WANDB_INFERENCE_API_KEY"
WANDB_INFERENCE_BASE_URL_ENV = "FUGUE_WANDB_INFERENCE_BASE_URL"
WANDB_INFERENCE_PROJECT_ENV = "FUGUE_WANDB_INFERENCE_PROJECT"
WEAVE_API_KEY_ENV = "FUGUE_WEAVE_API_KEY"
WEAVE_BASE_URL_ENV = "FUGUE_WEAVE_BASE_URL"
WEAVE_PROJECT_ENV = "FUGUE_WEAVE_PROJECT"
WEAVE_TRACE_SERVER_URL_ENV = "FUGUE_WEAVE_TRACE_SERVER_URL"
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
_HARNESS_PROTOCOLS: dict[str, ModelWireProtocol] = {
    "hermes": "chat_completions",
    "openclaw": "chat_completions",
    "claude-code": "messages",
    "codex": "responses",
}


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
        base_url = (
            values.get(WANDB_INFERENCE_BASE_URL_ENV)
            or values.get("WANDB_INFERENCE_BASE_URL")
            or WANDB_INFERENCE_BASE_URL
        )
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
        f"unknown model provider {provider_raw!r}; expected wandb, openai, or anthropic"
    )


def model_route_identity(
    route: ModelRoute, env: Mapping[str, str] | None = None
) -> dict[str, object]:
    identity: dict[str, object] = {
        "provider": route.provider,
        "model_id": route.model_id,
        "display_model": route.display_model,
        "chat_base_url": route.chat_base_url,
        "responses_base_url": route.responses_base_url,
        "messages_base_url": route.messages_base_url,
        "litellm_model": route.litellm_model,
        "tool_result_modalities": list(route.tool_result_modalities),
    }
    if route.provider == "wandb":
        identity["inference_project"] = inference_project_slug(env)
    return identity


def resolve_harness_model_route(route: ModelRoute, harness: str) -> dict[str, object]:
    normalized = harness.removeprefix("fugue-").strip().lower()
    try:
        protocol = _HARNESS_PROTOCOLS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Agent harness for model routing: {harness}"
        ) from exc
    _, bridge_required = model_protocol_endpoint(route, protocol)
    return {
        "harness": normalized,
        "wire_protocol": protocol,
        "endpoint_kind": "fugue_bridge" if bridge_required else "provider_direct",
        "upstream_host": _provider_host(route),
        "bridge_required": bridge_required,
    }


def model_protocol_endpoint(
    route: ModelRoute, protocol: ModelWireProtocol
) -> tuple[str, bool]:
    gateway = os.environ.get("FUGUE_MODEL_GATEWAY_BASE_URL", "").rstrip("/")
    if gateway:
        endpoint = gateway if protocol == "messages" else f"{gateway}/v1"
        return endpoint, True
    direct = {
        "chat_completions": route.chat_base_url,
        "messages": route.messages_base_url,
        "responses": route.responses_base_url,
    }[protocol]
    if direct:
        return direct, False
    bridge = (
        BRIDGE_BASE_URL_CONTAINER
        if protocol == "messages"
        else f"{BRIDGE_BASE_URL_CONTAINER}/v1"
    )
    return bridge, True


def _provider_host(route: ModelRoute) -> str:
    endpoint = (
        route.responses_base_url or route.messages_base_url or route.chat_base_url
    )
    host = urlparse(endpoint).hostname if endpoint else None
    if not host:
        raise ValueError(f"{route.display_model} has no provider endpoint host")
    return host


def structured_assistant_options(route: ModelRoute) -> dict[str, object]:
    # GLM-5.2 spent the entire structured-output budget on reasoning in the
    # release canary. W&B Inference accepts this route-specific control and
    # still returns native tool calls; Agent execution keeps normal thinking.
    if (route.provider, route.model_id) == ("wandb", "zai-org/GLM-5.2"):
        return {"thinking": {"type": "disabled"}}
    return {}


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
    return (
        []
        if provider_api_key(route, values)
        else [
            WANDB_INFERENCE_API_KEY_ENV
            if route.provider == "wandb"
            else route.api_key_env
        ]
    )


def missing_trace_env(env: Mapping[str, str] | None = None) -> list[str]:
    values = env if env is not None else os.environ
    return [] if trace_api_key(values) else [WEAVE_API_KEY_ENV]


def provider_api_key_env(route: ModelRoute) -> str:
    """Return the provider-only credential name used by isolated runtimes."""

    return (
        WANDB_INFERENCE_API_KEY_ENV
        if route.provider == "wandb"
        else route.api_key_env
    )


def provider_api_key(
    route: ModelRoute, env: Mapping[str, str] | None = None
) -> str:
    """Resolve a model credential without treating a trace key as authoritative."""

    values = env if env is not None else os.environ
    if route.provider == "wandb":
        explicit = values.get(WANDB_INFERENCE_API_KEY_ENV, "").strip()
        if explicit:
            return explicit
        split_evidence_configured = any(
            values.get(name, "").strip()
            for name in (
                WEAVE_API_KEY_ENV,
                WEAVE_PROJECT_ENV,
                WEAVE_BASE_URL_ENV,
                WEAVE_TRACE_SERVER_URL_ENV,
            )
        )
        if split_evidence_configured:
            return ""
    return values.get(route.api_key_env, "").strip()


def trace_api_key(env: Mapping[str, str] | None = None) -> str:
    """Resolve the evidence credential, retaining WANDB_API_KEY compatibility."""

    values = env if env is not None else os.environ
    return (
        values.get(WEAVE_API_KEY_ENV, "").strip()
        or values.get("WANDB_API_KEY", "").strip()
    )


def inference_project_slug(env: Mapping[str, str] | None = None) -> str:
    """Return the immutable W&B Inference billing scope."""

    values = env if env is not None else os.environ
    return (
        values.get(WANDB_INFERENCE_PROJECT_ENV, "").strip()
        or f"{DEFAULT_WANDB_ENTITY}/{DEFAULT_WANDB_PROJECT}"
    )


def trace_entity_project(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    values = env if env is not None else os.environ
    slug = (
        values.get(WEAVE_PROJECT_ENV, "").strip()
        or values.get("WEAVE_PROJECT", "").strip()
    )
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
    values = env if env is not None else os.environ
    result = {
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project,
        "WEAVE_PROJECT": f"{entity}/{project}",
    }
    base_url = (
        values.get(WEAVE_BASE_URL_ENV, "").strip()
        or values.get("WANDB_BASE_URL", "").strip()
    )
    trace_server_url = (
        values.get(WEAVE_TRACE_SERVER_URL_ENV, "").strip()
        or values.get("WF_TRACE_SERVER_URL", "").strip()
    )
    if base_url:
        result["WANDB_BASE_URL"] = base_url
    if trace_server_url:
        result["WF_TRACE_SERVER_URL"] = trace_server_url
    for key in ("WANDB_INSECURE_DISABLE_SSL", "WEAVE_INSECURE_DISABLE_SSL"):
        value = values.get(key, "").strip()
        if value:
            result[key] = value
    return result


def trace_destination_identity(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a secret-free identity for the evidence publication destination."""

    values = env if env is not None else os.environ
    entity, project = trace_entity_project(values)
    base_url = (
        values.get(WEAVE_BASE_URL_ENV, "").strip()
        or values.get("WANDB_BASE_URL", "").strip()
        or "https://api.wandb.ai"
    )
    trace_server_url = (
        values.get(WEAVE_TRACE_SERVER_URL_ENV, "").strip()
        or values.get("WF_TRACE_SERVER_URL", "").strip()
    )
    return {
        "entity": entity,
        "project": project,
        "project_slug": f"{entity}/{project}",
        "base_url": base_url.rstrip("/"),
        **(
            {"trace_server_url": trace_server_url.rstrip("/")}
            if trace_server_url
            else {}
        ),
    }


def provider_request_headers(
    route: ModelRoute, env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return non-secret headers required by the selected model provider."""
    if route.provider != "wandb":
        return {}
    return {WANDB_INFERENCE_PROJECT_HEADER: inference_project_slug(env)}


def provider_client_env(
    route: ModelRoute, env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return SDK environment needed for provider-specific request metadata."""
    if route.provider != "wandb":
        return {}
    project = inference_project_slug(env)
    return {
        OPENAI_PROJECT_ENV: project,
        OPENAI_PROJECT_ID_ENV: project,
    }
