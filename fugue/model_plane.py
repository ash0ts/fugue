from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast
from urllib.parse import urlparse

Provider = Literal["wandb", "openai", "anthropic"]
ToolResultModality = Literal["text", "image"]
ModelWireProtocol = Literal["chat_completions", "messages", "responses"]
WBATransportProfile = Literal[
    "responses-proxy",
    "responses-inline",
    "chat-inline",
]

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
_HARNESS_PROTOCOLS: dict[str, ModelWireProtocol] = {
    "hermes": "chat_completions",
    "openclaw": "chat_completions",
    "claude-code": "messages",
    "codex": "responses",
    "wba-responses": "responses",
}

WBA_TRANSPORT_CONTRACT_VERSION = 1
WBA_DEFAULT_TRANSPORT_PROFILE: WBATransportProfile = "responses-inline"
WBA_CORE_COMPATIBILITY_REFERENCE = "wandb/core@05115fffae784aef09bc0d4167ce19a587caf839"
WBA_RUNTIME_DEPENDENCIES = {
    "python": "3.13.5",
    "litellm": "1.93.0",
    "openai": "2.24.0",
    "pydantic": "2.12.5",
    "tenacity": "9.1.4",
    "tiktoken": "0.12.0",
    "weave": "0.53.0",
}
WBA_RETRY_POLICY = {
    "attempts": 5,
    "backoff": "exponential-capped-8s-v1",
    "client_retries": 0,
}
WBA_TIMEOUT_POLICY = {
    "request_timeout_sec": 300,
    "http_timeout_sec": 120,
    "connect_timeout_sec": 30,
    "shell_timeout_sec": 120,
    "shell_timeout_max_sec": 180,
}
WBA_COMPACTION_POLICY = {
    "strategy": "summarize-middle-v1",
    "trigger_ratio": 0.8,
    "keep_head_turns": 1,
    "keep_tail_turns": 3,
}
WBA_LOOP_POLICY = {
    "max_steps_per_turn": 20,
    "tool_names": ["shell"],
    "shell_output_limit_bytes": 100_000,
}
WBA_SAMPLING_POLICY = {
    "temperature": 0.0,
}

_WBA_TRANSPORT_PROFILES: dict[WBATransportProfile, dict[str, object]] = {
    "responses-proxy": {
        "agent_wire_protocol": "responses",
        "provider_wire_protocol": "chat_completions",
        "client": "openai-responses",
        "codec": "fugue-litellm-responses-proxy-v1",
        "conversion_location": "external_proxy",
        "bridge_required": True,
    },
    "responses-inline": {
        "agent_wire_protocol": "responses",
        "provider_wire_protocol": "chat_completions",
        "client": "litellm-responses",
        "codec": "litellm-responses-to-chat-v1",
        "conversion_location": "in_process",
        "bridge_required": False,
    },
    "chat-inline": {
        "agent_wire_protocol": "chat_completions",
        "provider_wire_protocol": "chat_completions",
        "client": "litellm-chat",
        "codec": "chat-completions-native-v1",
        "conversion_location": "none",
        "bridge_required": False,
    },
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
        f"unknown model provider {provider_raw!r}; expected wandb, openai, or anthropic"
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


def normalize_wba_transport_profile(
    value: str | None,
) -> WBATransportProfile:
    selected = str(value or WBA_DEFAULT_TRANSPORT_PROFILE).strip().lower()
    if selected not in _WBA_TRANSPORT_PROFILES:
        expected = ", ".join(sorted(_WBA_TRANSPORT_PROFILES))
        raise ValueError(
            f"unsupported WBA transport profile {value!r}; expected one of {expected}"
        )
    return cast(WBATransportProfile, selected)


def resolve_wba_transport_receipt(
    route: ModelRoute,
    profile: str | None,
) -> dict[str, object]:
    selected = normalize_wba_transport_profile(profile)
    profile_spec = _WBA_TRANSPORT_PROFILES[selected]
    bridge_required = bool(profile_spec["bridge_required"])
    receipt: dict[str, object] = {
        "schema_version": WBA_TRANSPORT_CONTRACT_VERSION,
        "harness": "wba-responses",
        "profile": selected,
        "wire_protocol": profile_spec["agent_wire_protocol"],
        "agent_wire_protocol": profile_spec["agent_wire_protocol"],
        "provider_wire_protocol": profile_spec["provider_wire_protocol"],
        "client": profile_spec["client"],
        "codec": profile_spec["codec"],
        "conversion_location": profile_spec["conversion_location"],
        "endpoint_kind": "fugue_bridge" if bridge_required else "provider_direct",
        "model_provider": route.provider,
        "model_id": route.model_id,
        "provider_endpoint": route.chat_base_url,
        "upstream_host": _provider_host(route),
        "bridge_required": bridge_required,
        "runtime_dependencies": dict(WBA_RUNTIME_DEPENDENCIES),
        "compatibility_reference": WBA_CORE_COMPATIBILITY_REFERENCE,
        "retry_policy": dict(WBA_RETRY_POLICY),
        "retry_policy_digest": _wba_policy_digest(WBA_RETRY_POLICY),
        "timeout_policy": dict(WBA_TIMEOUT_POLICY),
        "timeout_policy_digest": _wba_policy_digest(WBA_TIMEOUT_POLICY),
        "compaction_policy": dict(WBA_COMPACTION_POLICY),
        "compaction_policy_digest": _wba_policy_digest(WBA_COMPACTION_POLICY),
        "loop_policy": dict(WBA_LOOP_POLICY),
        "loop_policy_digest": _wba_policy_digest(WBA_LOOP_POLICY),
        "sampling_policy": dict(WBA_SAMPLING_POLICY),
        "sampling_policy_digest": _wba_policy_digest(WBA_SAMPLING_POLICY),
    }
    receipt["route_digest"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return receipt


def wba_transport_receipt_from_dict(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Strictly reconstruct one immutable V1 WBA transport receipt."""

    raw = {str(key): item for key, item in value.items()}
    fields = {
        "schema_version",
        "harness",
        "profile",
        "wire_protocol",
        "agent_wire_protocol",
        "provider_wire_protocol",
        "client",
        "codec",
        "conversion_location",
        "endpoint_kind",
        "model_provider",
        "model_id",
        "provider_endpoint",
        "upstream_host",
        "bridge_required",
        "runtime_dependencies",
        "compatibility_reference",
        "retry_policy",
        "retry_policy_digest",
        "timeout_policy",
        "timeout_policy_digest",
        "compaction_policy",
        "compaction_policy_digest",
        "loop_policy",
        "loop_policy_digest",
        "sampling_policy",
        "sampling_policy_digest",
        "route_digest",
    }
    unknown = sorted(set(raw) - fields)
    if unknown:
        raise ValueError(
            "unknown WBA transport receipt field(s): " + ", ".join(unknown)
        )
    if isinstance(raw.get("schema_version"), bool) or raw.get(
        "schema_version"
    ) != WBA_TRANSPORT_CONTRACT_VERSION:
        raise ValueError("WBA transport receipt must use schema_version 1")
    if raw.get("harness") != "wba-responses":
        raise ValueError("WBA transport receipt has the wrong harness")
    if not isinstance(raw.get("bridge_required"), bool):
        raise ValueError("WBA transport receipt bridge_required must be a boolean")
    profile = normalize_wba_transport_profile(str(raw.get("profile") or ""))
    provider = str(raw.get("model_provider") or "")
    if provider not in {"wandb", "openai", "anthropic"}:
        raise ValueError("WBA transport receipt has an unsupported model provider")
    model_id = str(raw.get("model_id") or "").strip()
    endpoint = str(raw.get("provider_endpoint") or "").strip().rstrip("/")
    parsed = urlparse(endpoint)
    if not model_id or parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("WBA transport receipt has an invalid provider route")
    route = ModelRoute(
        provider=cast(Provider, provider),
        model_id=model_id,
        display_model=f"{provider}/{model_id}",
        api_key_env="",
        chat_base_url=endpoint,
        responses_base_url=None,
        messages_base_url=None,
        litellm_model="",
        tool_result_modalities=("text",),
    )
    expected = resolve_wba_transport_receipt(route, profile)
    if raw != expected:
        raise ValueError("WBA transport receipt does not match its locked profile")
    return expected


def _wba_policy_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_harness_model_route(
    route: ModelRoute,
    harness: str,
    *,
    transport_profile: str | None = None,
) -> dict[str, object]:
    normalized = harness.removeprefix("fugue-").strip().lower()
    if normalized == "wba-responses":
        return resolve_wba_transport_receipt(route, transport_profile)
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
