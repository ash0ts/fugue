from __future__ import annotations

import json
from pathlib import Path

import pytest

from fugue.model_plane import (
    DEFAULT_MODEL,
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    EVIDENCE_DESTINATION_DIGEST_ENV,
    EVIDENCE_DESTINATION_JSON_ENV,
    EvidenceDestinationV1,
    default_evidence_destination,
    evidence_destination_environment,
    evidence_destination_from_dict,
    inference_project_slug,
    missing_model_env,
    missing_trace_env,
    model_route_identity,
    provider_api_key,
    provider_client_env,
    provider_request_headers,
    resolve_harness_model_route,
    resolve_model_route,
    select_model,
    structured_assistant_options,
    trace_api_key,
    trace_destination_identity,
    trace_entity_project,
    trace_env_defaults,
    trace_project_environment,
    trace_project_slug,
)
from fugue.weave_support import weave_agents_otel_headers


def test_resolve_model_route_for_supported_providers() -> None:
    wandb = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    assert wandb.provider == "wandb"
    assert wandb.model_id == "zai-org/GLM-5.2"
    assert wandb.api_key_env == "WANDB_API_KEY"
    assert wandb.chat_base_url == "https://api.inference.wandb.ai/v1"
    assert wandb.responses_base_url is None
    assert wandb.messages_base_url is None
    assert wandb.tool_result_modalities == ("text",)
    assert model_route_identity(wandb)["tool_result_modalities"] == ["text"]
    assert structured_assistant_options(wandb) == {"thinking": {"type": "disabled"}}

    openai = resolve_model_route("openai/gpt-5", {})
    assert openai.provider == "openai"
    assert structured_assistant_options(openai) == {}
    assert openai.model_id == "gpt-5"
    assert openai.api_key_env == "OPENAI_API_KEY"
    assert openai.responses_base_url == "https://api.openai.com/v1"
    assert openai.tool_result_modalities == ("text", "image")

    anthropic = resolve_model_route("anthropic/claude-sonnet-4-5", {})
    assert anthropic.provider == "anthropic"
    assert anthropic.model_id == "claude-sonnet-4-5"
    assert anthropic.api_key_env == "ANTHROPIC_API_KEY"
    assert anthropic.chat_base_url is None
    assert anthropic.messages_base_url == "https://api.anthropic.com"


@pytest.mark.parametrize(
    ("model", "harness", "protocol", "endpoint_kind", "endpoint_host"),
    [
        (
            "wandb/zai-org/GLM-5.2",
            "hermes",
            "chat_completions",
            "provider_direct",
            "api.inference.wandb.ai",
        ),
        (
            "wandb/zai-org/GLM-5.2",
            "openclaw",
            "chat_completions",
            "provider_direct",
            "api.inference.wandb.ai",
        ),
        (
            "wandb/zai-org/GLM-5.2",
            "claude-code",
            "messages",
            "fugue_bridge",
            "host.docker.internal",
        ),
        (
            "wandb/zai-org/GLM-5.2",
            "codex",
            "responses",
            "fugue_bridge",
            "host.docker.internal",
        ),
        (
            "anthropic/claude-sonnet-4-5",
            "hermes",
            "chat_completions",
            "fugue_bridge",
            "host.docker.internal",
        ),
        (
            "anthropic/claude-sonnet-4-5",
            "openclaw",
            "chat_completions",
            "fugue_bridge",
            "host.docker.internal",
        ),
        (
            "anthropic/claude-sonnet-4-5",
            "claude-code",
            "messages",
            "provider_direct",
            "api.anthropic.com",
        ),
        (
            "anthropic/claude-sonnet-4-5",
            "codex",
            "responses",
            "fugue_bridge",
            "host.docker.internal",
        ),
        (
            "openai/gpt-5",
            "hermes",
            "chat_completions",
            "provider_direct",
            "api.openai.com",
        ),
        (
            "openai/gpt-5",
            "openclaw",
            "chat_completions",
            "provider_direct",
            "api.openai.com",
        ),
        (
            "openai/gpt-5",
            "codex",
            "responses",
            "provider_direct",
            "api.openai.com",
        ),
        (
            "openai/gpt-5",
            "claude-code",
            "messages",
            "fugue_bridge",
            "host.docker.internal",
        ),
    ],
)
def test_harness_model_route_is_explicit_and_provider_aware(
    model: str,
    harness: str,
    protocol: str,
    endpoint_kind: str,
    endpoint_host: str,
) -> None:
    receipt = resolve_harness_model_route(resolve_model_route(model, {}), harness)

    assert receipt["wire_protocol"] == protocol
    assert receipt["endpoint_kind"] == endpoint_kind
    assert (
        "host.docker.internal" if receipt["bridge_required"] else receipt["upstream_host"]
    ) == endpoint_host
    assert receipt["bridge_required"] is (endpoint_kind == "fugue_bridge")


@pytest.mark.parametrize("harness", ["direct", "sequence"])
def test_non_agent_workloads_have_no_model_transport(harness: str) -> None:
    receipt = resolve_harness_model_route(
        resolve_model_route("wandb/zai-org/GLM-5.2", {}),
        harness,
    )

    assert receipt == {
        "harness": harness,
        "wire_protocol": None,
        "endpoint_kind": "not_applicable",
        "upstream_host": None,
        "bridge_required": False,
    }


@pytest.mark.parametrize("model", ["gpt-5", "local/foo", "openai/"])
def test_resolve_model_route_rejects_invalid_models(model: str) -> None:
    with pytest.raises(ValueError):
        resolve_model_route(model, {})


def test_select_model_precedence() -> None:
    env = {"FUGUE_MODEL": "anthropic/env-model"}
    assert select_model("openai/cli", "wandb/manifest", env) == "openai/cli"
    assert select_model(None, "wandb/manifest", env) == "wandb/manifest"
    assert select_model(None, None, env) == "anthropic/env-model"
    assert select_model(None, None, {}) == DEFAULT_MODEL
    assert (
        select_model(
            None,
            "wandb/manifest",
            env,
            harness_model="openai/harness",
            experiment_model="anthropic/experiment",
        )
        == "openai/harness"
    )
    assert (
        select_model(
            None,
            "wandb/manifest",
            env,
            experiment_model="anthropic/experiment",
        )
        == "anthropic/experiment"
    )


def test_missing_env_separates_model_and_trace_keys() -> None:
    route = resolve_model_route("openai/gpt-5", {})
    env = {"WANDB_API_KEY": "trace"}
    assert missing_trace_env(env) == []
    assert missing_model_env(route, env) == ["OPENAI_API_KEY"]


def test_trace_project_defaults_to_wandb_shared_project() -> None:
    assert trace_entity_project({}) == (DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT)
    assert trace_project_slug({}) == "wandb/fugue-experiments"
    defaults = trace_env_defaults({})
    assert defaults["WANDB_ENTITY"] == "wandb"
    assert defaults["WANDB_PROJECT"] == "fugue-experiments"
    assert defaults["WEAVE_PROJECT"] == "wandb/fugue-experiments"
    assert defaults["WANDB_BASE_URL"] == "https://api.wandb.ai"
    assert defaults["WF_TRACE_SERVER_URL"] == "https://trace.wandb.ai"
    assert defaults["WANDB_APP_BASE_URL"] == "https://wandb.ai"
    assert trace_entity_project({"WEAVE_PROJECT": "custom/project"}) == (
        "custom",
        "project",
    )


def test_wandb_model_requests_use_an_independent_inference_project() -> None:
    wandb = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    env = {
        "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/fugue-experiments",
        "FUGUE_WEAVE_PROJECT": "team/experiment-project",
    }

    assert provider_request_headers(wandb, env) == {
        "OpenAI-Project": "wandb/fugue-experiments"
    }
    assert provider_client_env(wandb, env) == {
        "OPENAI_PROJECT": "wandb/fugue-experiments",
        "OPENAI_PROJECT_ID": "wandb/fugue-experiments",
    }
    assert inference_project_slug(env) == "wandb/fugue-experiments"
    assert trace_project_slug(env) == "team/experiment-project"


def test_declared_evidence_project_overrides_legacy_project_environment() -> None:
    env = trace_project_environment(
        "wandb/fugue-mcp-release-qualification-v1",
        {
            "WANDB_ENTITY": "wandb",
            "WANDB_PROJECT": "news-research-agent",
            "WEAVE_PROJECT": "wandb/news-research-agent",
            "FUGUE_WEAVE_PROJECT": "wandb/news-research-agent",
            "WANDB_API_KEY": "secret",
        },
    )

    assert trace_project_slug(env) == (
        "wandb/fugue-mcp-release-qualification-v1"
    )
    assert env["WANDB_ENTITY"] == "wandb"
    assert env["WANDB_PROJECT"] == "fugue-mcp-release-qualification-v1"
    assert env["WEAVE_PROJECT"] == "wandb/fugue-mcp-release-qualification-v1"
    assert env["FUGUE_WEAVE_PROJECT"] == (
        "wandb/fugue-mcp-release-qualification-v1"
    )
    assert env["WANDB_BASE_URL"] == "https://api.wandb.ai"
    assert env["WF_TRACE_SERVER_URL"] == "https://trace.wandb.ai"
    assert env["WANDB_APP_BASE_URL"] == "https://wandb.ai"
    assert len(env[EVIDENCE_DESTINATION_DIGEST_ENV]) == 64
    assert (
        json.loads(env[EVIDENCE_DESTINATION_JSON_ENV])["project_slug"]
        == "wandb/fugue-mcp-release-qualification-v1"
    )
    assert env["WANDB_API_KEY"] == "secret"


def test_declared_destination_overrides_legacy_test_endpoints() -> None:
    destination = EvidenceDestinationV1(
        entity="wandb",
        project="fugue-mcp-release-qualification-v1",
        api_base_url="https://api.wandb.ai",
        trace_base_url="https://trace.wandb.ai",
        app_base_url="https://wandb.ai",
    )

    env = evidence_destination_environment(
        destination,
        {
            "FUGUE_WEAVE_BASE_URL": "https://api.wandb.test",
            "FUGUE_WEAVE_TRACE_SERVER_URL": "https://trace.wandb.test",
            "WANDB_APP_BASE_URL": "https://app.wandb.test",
            "FUGUE_WEAVE_PROJECT": "wandb/news-research-agent",
        },
    )

    assert trace_destination_identity(env) == destination.to_dict()
    assert env["FUGUE_WEAVE_BASE_URL"] == "https://api.wandb.ai"
    assert env["FUGUE_WEAVE_TRACE_SERVER_URL"] == "https://trace.wandb.ai"
    assert env["WANDB_APP_BASE_URL"] == "https://wandb.ai"


def test_wandb_inference_and_weave_credentials_are_independent() -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    env = {
        "FUGUE_WANDB_INFERENCE_API_KEY": "cloud-model-key",
        "FUGUE_WEAVE_API_KEY": "local-trace-key",
        "FUGUE_WANDB_INFERENCE_PROJECT": "wandb/fugue-experiments",
        "FUGUE_WEAVE_PROJECT": "ashah-weights-biases/loop-engineering-demo",
        "FUGUE_WEAVE_BASE_URL": "https://api.wandb.test",
    }

    assert provider_api_key(route, env) == "cloud-model-key"
    assert trace_api_key(env) == "local-trace-key"
    assert trace_destination_identity(env) == {
        "schema_version": 1,
        "entity": "ashah-weights-biases",
        "project": "loop-engineering-demo",
        "project_slug": "ashah-weights-biases/loop-engineering-demo",
        "api_base_url": "https://api.wandb.test",
        "trace_base_url": "https://api.wandb.test/traces",
        "app_base_url": "https://wandb.ai",
        "destination_digest": trace_destination_identity(env)[
            "destination_digest"
        ],
    }

    openai = resolve_model_route("openai/gpt-5", {})
    assert provider_request_headers(openai, env) == {}
    assert provider_client_env(openai, env) == {}


def test_local_evidence_tls_mode_is_explicitly_propagated() -> None:
    env = {
        "FUGUE_WEAVE_PROJECT": "ashah-weights-biases/loop-engineering-demo",
        "FUGUE_WEAVE_BASE_URL": "https://api.wandb.test",
        "FUGUE_WEAVE_TRACE_SERVER_URL": "http://127.0.0.1:6345",
        "WANDB_INSECURE_DISABLE_SSL": "true",
        "WEAVE_INSECURE_DISABLE_SSL": "true",
    }

    assert trace_env_defaults(env) == {
        "FUGUE_EVIDENCE_DESTINATION_DIGEST": trace_destination_identity(env)[
            "destination_digest"
        ],
        "FUGUE_EVIDENCE_DESTINATION_JSON": json.dumps(
            trace_destination_identity(env),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "FUGUE_WEAVE_BASE_URL": "https://api.wandb.test",
        "FUGUE_WEAVE_PROJECT": (
            "ashah-weights-biases/loop-engineering-demo"
        ),
        "FUGUE_WEAVE_TRACE_SERVER_URL": (
            "http://127.0.0.1:6345"
        ),
        "WANDB_ENTITY": "ashah-weights-biases",
        "WANDB_PROJECT": "loop-engineering-demo",
        "WEAVE_PROJECT": "ashah-weights-biases/loop-engineering-demo",
        "WANDB_BASE_URL": "https://api.wandb.test",
        "WANDB_APP_BASE_URL": "https://wandb.ai",
        "WF_TRACE_SERVER_URL": "http://127.0.0.1:6345",
        "WANDB_INSECURE_DISABLE_SSL": "true",
        "WEAVE_INSECURE_DISABLE_SSL": "true",
    }


def test_evidence_destination_rejects_credential_bearing_endpoint() -> None:
    with pytest.raises(ValueError, match="without credentials"):
        EvidenceDestinationV1(
            entity="wandb",
            project="fugue",
            api_base_url="https://user:secret@example.test",
            trace_base_url="https://trace.example.test",
            app_base_url="https://app.example.test",
        )


def test_evidence_destination_matches_the_console_endpoint_parity_fixture() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "evidence-destination-console-parity-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["schema_version"] == 1

    for case in fixture["accepted"]:
        destination = EvidenceDestinationV1(
            entity="wandb",
            project="endpoint-parity",
            api_base_url=case["api_base_url"],
            trace_base_url=case["trace_base_url"],
            app_base_url=case["app_base_url"],
        )
        assert {
            "api_base_url": destination.api_base_url,
            "trace_base_url": destination.trace_base_url,
            "app_base_url": destination.app_base_url,
        } == case["canonical"]

    defaults = fixture["accepted"][0]["canonical"]
    for case in fixture["rejected"]:
        endpoints = {**defaults, case["field"]: case["value"]}
        with pytest.raises(ValueError, match=case["error"]):
            EvidenceDestinationV1(
                entity="wandb",
                project="endpoint-parity",
                **endpoints,
            )

    with pytest.raises(ValueError, match="at most 2000"):
        EvidenceDestinationV1(
            entity="wandb",
            project="endpoint-parity",
            api_base_url="https://selfhost.example/" + "a" * 2000,
            trace_base_url=defaults["trace_base_url"],
            app_base_url=defaults["app_base_url"],
        )


@pytest.mark.parametrize("schema_version", (True, 1.0))
def test_evidence_destination_rejects_noninteger_schema_one(
    schema_version: object,
) -> None:
    values = default_evidence_destination("wandb/endpoint-parity").to_dict()
    values["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema version"):
        evidence_destination_from_dict(values)

    with pytest.raises(ValueError, match="schema version"):
        EvidenceDestinationV1(
            entity="wandb",
            project="endpoint-parity",
            api_base_url="https://api.wandb.ai",
            trace_base_url="https://trace.wandb.ai",
            app_base_url="https://wandb.ai",
            schema_version=schema_version,  # type: ignore[arg-type]
        )


def test_split_evidence_routing_keeps_general_wandb_key_compatibility() -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    env = {
        "WANDB_API_KEY": "general-wandb-key",
        "FUGUE_WEAVE_PROJECT": "ashah-weights-biases/loop-engineering-demo",
        "FUGUE_WEAVE_BASE_URL": "https://api.wandb.ai",
    }

    assert provider_api_key(route, env) == "general-wandb-key"
    assert missing_model_env(route, env) == []


def test_evidence_only_key_cannot_be_promoted_to_wandb_inference() -> None:
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    env = {
        "WANDB_API_KEY": "legacy-general-key",
        "FUGUE_WEAVE_API_KEY": "evidence-only-key",
        "FUGUE_WEAVE_PROJECT": "ashah-weights-biases/loop-engineering-demo",
    }

    assert provider_api_key(route, env) == ""
    assert missing_model_env(route, env) == ["FUGUE_WANDB_INFERENCE_API_KEY"]


def test_weave_agents_otel_headers_route_without_exposing_plain_key() -> None:
    headers = weave_agents_otel_headers("wandb/fugue-experiments", "test-key")

    assert headers.startswith(
        "project_id=wandb/fugue-experiments,Authorization=Basic%20"
    )
    assert "test-key" not in headers
