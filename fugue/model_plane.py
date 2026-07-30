from __future__ import annotations

import hashlib
import json
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
EVIDENCE_DESTINATION_SCHEMA_VERSION = 1
DEFAULT_WANDB_API_BASE_URL = "https://api.wandb.ai"
DEFAULT_WEAVE_TRACE_BASE_URL = "https://trace.wandb.ai"
DEFAULT_WANDB_APP_BASE_URL = "https://wandb.ai"
EVIDENCE_DESTINATION_DIGEST_ENV = "FUGUE_EVIDENCE_DESTINATION_DIGEST"
EVIDENCE_DESTINATION_JSON_ENV = "FUGUE_EVIDENCE_DESTINATION_JSON"

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


@dataclass(frozen=True)
class EvidenceDestinationV1:
    """One immutable, secret-free destination shared by every evidence sink."""

    entity: str
    project: str
    api_base_url: str
    trace_base_url: str
    app_base_url: str
    schema_version: int = EVIDENCE_DESTINATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_DESTINATION_SCHEMA_VERSION:
            raise ValueError("unsupported evidence destination schema version")
        if (
            not self.entity
            or not self.project
            or "/" in self.entity
            or "/" in self.project
        ):
            raise ValueError(
                "evidence destination requires an exact W&B entity/project"
            )
        for label, value in (
            ("API", self.api_base_url),
            ("trace", self.trace_base_url),
            ("application", self.app_base_url),
        ):
            _validated_evidence_base_url(value, label=label)

    @property
    def project_slug(self) -> str:
        return f"{self.entity}/{self.project}"

    @property
    def destination_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "entity": self.entity,
                    "project": self.project,
                    "api_base_url": self.api_base_url,
                    "trace_base_url": self.trace_base_url,
                    "app_base_url": self.app_base_url,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema_version": self.schema_version,
            "entity": self.entity,
            "project": self.project,
            "project_slug": self.project_slug,
            "api_base_url": self.api_base_url,
            "trace_base_url": self.trace_base_url,
            "app_base_url": self.app_base_url,
            "destination_digest": self.destination_digest,
        }

    def environment(self) -> dict[str, str]:
        identity = self.to_dict()
        return {
            WEAVE_PROJECT_ENV: self.project_slug,
            "WEAVE_PROJECT": self.project_slug,
            "WANDB_ENTITY": self.entity,
            "WANDB_PROJECT": self.project,
            WEAVE_BASE_URL_ENV: self.api_base_url,
            "WANDB_BASE_URL": self.api_base_url,
            WEAVE_TRACE_SERVER_URL_ENV: self.trace_base_url,
            "WF_TRACE_SERVER_URL": self.trace_base_url,
            "WANDB_APP_BASE_URL": self.app_base_url,
            EVIDENCE_DESTINATION_DIGEST_ENV: self.destination_digest,
            EVIDENCE_DESTINATION_JSON_ENV: json.dumps(
                identity, sort_keys=True, separators=(",", ":")
            ),
        }


def evidence_destination_from_dict(
    value: Mapping[str, object],
) -> EvidenceDestinationV1:
    """Load a destination while verifying any supplied derived identity."""

    allowed = {
        "schema_version",
        "entity",
        "project",
        "project_slug",
        "api_base_url",
        "trace_base_url",
        "app_base_url",
        "destination_digest",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            "unknown evidence destination field(s): " + ", ".join(unknown)
        )
    required = {
        "entity",
        "project",
        "api_base_url",
        "trace_base_url",
        "app_base_url",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            "missing evidence destination field(s): " + ", ".join(missing)
        )
    destination = EvidenceDestinationV1(
        entity=str(value["entity"]),
        project=str(value["project"]),
        api_base_url=str(value["api_base_url"]),
        trace_base_url=str(value["trace_base_url"]),
        app_base_url=str(value["app_base_url"]),
        schema_version=int(
            value.get(
                "schema_version",
                EVIDENCE_DESTINATION_SCHEMA_VERSION,
            )
        ),
    )
    supplied_project_slug = str(value.get("project_slug") or "")
    if supplied_project_slug and supplied_project_slug != destination.project_slug:
        raise ValueError("evidence destination project_slug does not match")
    supplied_digest = str(value.get("destination_digest") or "")
    if supplied_digest and supplied_digest != destination.destination_digest:
        raise ValueError("evidence destination digest does not match")
    return destination


def evidence_destination_environment(
    destination: EvidenceDestinationV1,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply an immutable declared destination with precedence over legacy env."""

    result = dict(env if env is not None else os.environ)
    result.update(destination.environment())
    return trace_project_environment(destination.project_slug, result)


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
    if normalized in {"direct", "sequence"}:
        return {
            "harness": normalized,
            "wire_protocol": None,
            "endpoint_kind": "not_applicable",
            "upstream_host": None,
            "bridge_required": False,
        }
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
    """Resolve a model credential without treating an evidence-only key as authoritative."""

    values = env if env is not None else os.environ
    if route.provider == "wandb":
        explicit = values.get(WANDB_INFERENCE_API_KEY_ENV, "").strip()
        if explicit:
            return explicit
        # WANDB_API_KEY is the legacy general W&B credential and can
        # authenticate both public-cloud evidence and Inference. Project and
        # endpoint routing alone must not disable that compatibility. An
        # explicit evidence-only credential, however, is never promoted into
        # the model plane.
        if values.get(WEAVE_API_KEY_ENV, "").strip():
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


def trace_project_environment(
    project_slug: str | None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Bind an explicit evidence destination without changing behavior identity."""

    result = dict(env if env is not None else os.environ)
    selected = str(project_slug or "").strip()
    if selected:
        parts = selected.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "evidence project must be an exact W&B entity/project slug"
            )
        result[WEAVE_PROJECT_ENV] = selected
        result["WEAVE_PROJECT"] = selected
        result["WANDB_ENTITY"], result["WANDB_PROJECT"] = parts
    result.update(resolve_evidence_destination(result).environment())
    for key in ("WANDB_INSECURE_DISABLE_SSL", "WEAVE_INSECURE_DISABLE_SSL"):
        value = result.get(key, "").strip()
        if value:
            result[key] = value
    return result


def trace_env_defaults(env: Mapping[str, str] | None = None) -> dict[str, str]:
    values = env if env is not None else os.environ
    result = resolve_evidence_destination(values).environment()
    for key in ("WANDB_INSECURE_DISABLE_SSL", "WEAVE_INSECURE_DISABLE_SSL"):
        value = values.get(key, "").strip()
        if value:
            result[key] = value
    return result


def trace_destination_identity(
    env: Mapping[str, str] | None = None,
) -> dict[str, str | int]:
    """Return a secret-free identity for the evidence publication destination."""

    return resolve_evidence_destination(env).to_dict()


def resolve_evidence_destination(
    env: Mapping[str, str] | None = None,
) -> EvidenceDestinationV1:
    """Resolve all evidence endpoints once so every sink receives one identity."""

    values = env if env is not None else os.environ
    entity, project = trace_entity_project(values)
    api_base_url = _validated_evidence_base_url(
        values.get(WEAVE_BASE_URL_ENV, "").strip()
        or values.get("WANDB_BASE_URL", "").strip()
        or DEFAULT_WANDB_API_BASE_URL,
        label="API",
    )
    configured_trace_base = (
        values.get(WEAVE_TRACE_SERVER_URL_ENV, "").strip()
        or values.get("WF_TRACE_SERVER_URL", "").strip()
    )
    public_base = values.get("WANDB_PUBLIC_BASE_URL", "").strip()
    trace_base_url = _validated_evidence_base_url(
        configured_trace_base
        or _derived_trace_base_url(api_base_url, public_base=public_base),
        label="trace",
    )
    app_base_url = _validated_evidence_base_url(
        values.get("WANDB_APP_BASE_URL", "").strip()
        or DEFAULT_WANDB_APP_BASE_URL,
        label="application",
    )
    return EvidenceDestinationV1(
        entity=entity,
        project=project,
        api_base_url=api_base_url,
        trace_base_url=trace_base_url,
        app_base_url=app_base_url,
    )


def _derived_trace_base_url(api_base_url: str, *, public_base: str) -> str:
    selected = public_base.rstrip("/") if public_base else api_base_url
    return (
        DEFAULT_WEAVE_TRACE_BASE_URL
        if selected == DEFAULT_WANDB_API_BASE_URL
        else f"{selected.rstrip('/')}/traces"
    )


def _validated_evidence_base_url(value: str, *, label: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"evidence {label} base URL must be an absolute, credential-free "
            "HTTP(S) URL without query or fragment"
        )
    return normalized


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
