from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from fugue.connectivity_gateway import (
    GatewayAuthorizationError,
    GatewayServer,
    GatewayUsageLedger,
    _capability_token,
    _forward_headers,
    _PinnedHTTPSConnection,
    _request_pinned_upstream,
    authorize_gateway_request,
    gateway_policy_from_dict,
    mint_cell_capability,
    validate_upstream_resolution,
    verify_cell_capability,
)

KEY = b"s" * 32


def _policy():
    return gateway_policy_from_dict(
        {
            "schema_version": 1,
            "max_token_lifetime_seconds": 1800,
            "routes": [
                {
                    "id": "model",
                    "upstream": "https://api.inference.wandb.ai",
                    "methods": ["POST"],
                    "path_prefixes": ["/v1"],
                    "content_types": ["application/json"],
                    "static_headers": {
                        "openai-project": "wandb/fugue-experiments"
                    },
                    "credential_env": "FUGUE_WANDB_INFERENCE_API_KEY",
                }
            ],
        }
    )


def _token(*, cell_id: str = "cell-1", now: int = 1_000) -> str:
    return mint_cell_capability(
        signing_key=KEY,
        issuer="instance-1",
        cell_id=cell_id,
        execution_fingerprint="a" * 64,
        route_ids=("model",),
        lifetime_seconds=300,
        max_requests=2,
        max_request_bytes=1024,
        max_response_bytes=2048,
        now=now,
    )


def test_gateway_capability_is_signed_scoped_and_expiring() -> None:
    capability = verify_cell_capability(
        _token(), signing_key=KEY, expected_issuer="instance-1", now=1_001
    )
    assert capability.cell_id == "cell-1"
    assert capability.route_ids == ("model",)

    with pytest.raises(GatewayAuthorizationError, match="signature"):
        verify_cell_capability(
            _token(),
            signing_key=b"x" * 32,
            expected_issuer="instance-1",
            now=1_001,
        )
    with pytest.raises(GatewayAuthorizationError, match="expired"):
        verify_cell_capability(
            _token(), signing_key=KEY, expected_issuer="instance-1", now=1_301
        )


def test_gateway_allows_only_locked_method_path_and_media_type() -> None:
    capability = verify_cell_capability(
        _token(), signing_key=KEY, expected_issuer="instance-1", now=1_001
    )
    policy = _policy()
    allowed = authorize_gateway_request(
        policy=policy,
        capability=capability,
        route_id="model",
        method="POST",
        suffix="/v1/chat/completions",
        content_type="application/json",
        request_bytes=100,
    )
    assert allowed.allowed
    assert (
        allowed.upstream_url
        == "https://api.inference.wandb.ai/v1/chat/completions"
    )

    for values in (
        {"method": "CONNECT"},
        {"suffix": "/v2/admin"},
        {"content_type": "text/plain"},
        {"route_id": "weave"},
        {"request_bytes": 2_000},
    ):
        arguments = {
            "policy": policy,
            "capability": capability,
            "route_id": "model",
            "method": "POST",
            "suffix": "/v1/chat/completions",
            "content_type": "application/json",
            "request_bytes": 100,
            **values,
        }
        assert authorize_gateway_request(**arguments).allowed is False


def test_gateway_rejects_ssrf_origins_and_unsafe_policy_fields() -> None:
    raw = _policy().to_dict()
    raw["policy_sha256"] = ""
    raw["routes"][0]["upstream"] = "https://127.0.0.1"
    with pytest.raises(ValueError, match="private or special"):
        gateway_policy_from_dict(raw)


def test_gateway_rejects_dns_rebinding_to_private_addresses(monkeypatch) -> None:
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("169.254.169.254", 443)),
        ],
    )
    with pytest.raises(GatewayAuthorizationError, match="private or special"):
        validate_upstream_resolution("https://api.example.com")

    raw = _policy().to_dict()
    raw["policy_sha256"] = ""
    raw["routes"][0]["upstream"] = "https://example.com/redirect"
    with pytest.raises(ValueError, match="origin-only"):
        gateway_policy_from_dict(raw)

    raw = _policy().to_dict()
    raw["policy_sha256"] = ""
    raw["routes"][0]["methods"] = ["CONNECT"]
    with pytest.raises(ValueError, match="GET and POST"):
        gateway_policy_from_dict(raw)


def test_gateway_ledger_prevents_overuse_and_cross_cell_replay(
    tmp_path: Path,
) -> None:
    ledger = GatewayUsageLedger(tmp_path / "gateway.sqlite3")
    capability = verify_cell_capability(
        _token(), signing_key=KEY, expected_issuer="instance-1", now=1_001
    )
    assert (
        ledger.claim_request(
            capability,
            request_bytes=100,
            source_address="203.0.113.10",
        )
        == 1
    )
    assert (
        ledger.claim_request(
            capability,
            request_bytes=100,
            source_address="203.0.113.10",
        )
        == 2
    )
    with pytest.raises(GatewayAuthorizationError, match="request limit"):
        ledger.claim_request(
            capability,
            request_bytes=100,
            source_address="203.0.113.10",
        )

    changed = capability.__class__(
        **{**capability.to_dict(), "cell_id": "cell-2"}
    )
    with pytest.raises(GatewayAuthorizationError, match="across cells"):
        ledger.claim_request(
            changed,
            request_bytes=1,
            source_address="203.0.113.10",
        )

    with pytest.raises(GatewayAuthorizationError, match="another sandbox"):
        ledger.claim_request(
            capability,
            request_bytes=1,
            source_address="203.0.113.11",
        )


def test_gateway_ledger_enforces_cumulative_byte_budgets(tmp_path: Path) -> None:
    ledger = GatewayUsageLedger(tmp_path / "gateway.sqlite3")
    capability = replace(
        verify_cell_capability(
            _token(), signing_key=KEY, expected_issuer="instance-1", now=1_001
        ),
        max_request_bytes=10,
        max_response_bytes=12,
    )

    assert (
        ledger.claim_request(
            capability,
            request_bytes=6,
            source_address="203.0.113.10",
        )
        == 1
    )
    with pytest.raises(GatewayAuthorizationError, match="cumulative request"):
        ledger.claim_request(
            capability,
            request_bytes=5,
            source_address="203.0.113.10",
        )

    ledger.record_response(capability, response_bytes=7)
    with pytest.raises(GatewayAuthorizationError, match="cumulative response"):
        ledger.record_response(capability, response_bytes=6)


def test_gateway_ledger_migrates_pre_source_binding_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gateway.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE capability_usage ("
                "nonce TEXT PRIMARY KEY, cell_id TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL, request_count INTEGER NOT NULL, "
                "request_bytes INTEGER NOT NULL, response_bytes INTEGER NOT NULL)"
            )
    ledger = GatewayUsageLedger(path)
    capability = verify_cell_capability(
        _token(), signing_key=KEY, expected_issuer="instance-1", now=1_001
    )
    assert (
        ledger.claim_request(
            capability,
            request_bytes=1,
            source_address="203.0.113.10",
        )
        == 1
    )


def test_pinned_connection_uses_validated_address_and_original_tls_name(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    class RawSocket:
        def close(self) -> None:
            observed["raw_closed"] = True

    class Context:
        def wrap_socket(self, raw, *, server_hostname):
            observed["raw"] = raw
            observed["server_hostname"] = server_hostname
            return object()

    def create_connection(address, *, timeout):
        observed["address"] = address
        observed["timeout"] = timeout
        return RawSocket()

    monkeypatch.setattr("socket.create_connection", create_connection)
    connection = _PinnedHTTPSConnection(
        "api.example.com",
        "203.0.113.10",
        port=443,
        context=Context(),  # type: ignore[arg-type]
        timeout=4,
    )
    connection.connect()

    assert observed["address"] == ("203.0.113.10", 443)
    assert observed["server_hostname"] == "api.example.com"


def test_pinned_upstream_stops_reading_at_the_locked_limit(monkeypatch) -> None:
    class Response:
        status = 200

        def getheaders(self):
            return [("content-type", "application/json")]

        def read(self, amount: int):
            assert amount == 5
            return b"12345"

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(
        "fugue.connectivity_gateway._PinnedHTTPSConnection", Connection
    )
    with pytest.raises(GatewayAuthorizationError, match="byte limit"):
        _request_pinned_upstream(
            route=_policy().routes[0],
            method="POST",
            path="/v1/chat/completions",
            body=b"{}",
            headers={"content-type": "application/json"},
            addresses=("203.0.113.10",),
            verify=True,
            timeout=5,
            response_limit=4,
        )


def test_gateway_forbids_disabling_upstream_tls(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="may not be disabled"):
        GatewayServer(
            ("127.0.0.1", 0),
            policy=_policy(),
            signing_key=KEY,
            issuer="instance-1",
            ledger=GatewayUsageLedger(tmp_path / "gateway.sqlite3"),
            upstream_ca=False,
        )


def test_gateway_injects_locked_nonsecret_route_headers(monkeypatch) -> None:
    route = _policy().routes[0]
    monkeypatch.setenv("FUGUE_WANDB_INFERENCE_API_KEY", "provider-secret")
    headers = _forward_headers(
        {
            "content-type": "application/json",
            "authorization": "Bearer attacker",
            "openai-project": "attacker/project",
        },
        route,
    )

    assert headers["openai-project"] == "wandb/fugue-experiments"
    assert headers["authorization"] == "Bearer provider-secret"


def test_gateway_accepts_capability_from_wandb_basic_auth() -> None:
    token = _token()
    encoded = __import__("base64").b64encode(f"api:{token}".encode()).decode()
    assert _capability_token(f"Basic {encoded}") == token
    assert _capability_token(f"Bearer {token}") == token
    with pytest.raises(GatewayAuthorizationError, match="missing"):
        _capability_token("Basic broken")


def test_gateway_encodes_upstream_wandb_credentials(monkeypatch) -> None:
    raw = _policy().to_dict()
    raw["policy_sha256"] = ""
    raw["routes"][0]["credential_prefix"] = "Basic "
    raw["routes"][0]["credential_encoding"] = "wandb-basic"
    route = gateway_policy_from_dict(raw).routes[0]
    monkeypatch.setenv("FUGUE_WANDB_INFERENCE_API_KEY", "raw-key")

    headers = _forward_headers({}, route)
    expected = __import__("base64").b64encode(b"api:raw-key").decode()
    assert headers["authorization"] == f"Basic {expected}"


@pytest.mark.parametrize(
    "headers",
    [
        {"authorization": "literal-secret"},
        {"host": "internal.example"},
        {"x-forwarded-host": "internal.example"},
        {"safe": "value\ninjected"},
    ],
)
def test_gateway_rejects_unsafe_static_headers(headers) -> None:
    raw = _policy().to_dict()
    raw["policy_sha256"] = ""
    raw["routes"][0]["static_headers"] = headers
    with pytest.raises(ValueError, match="static header"):
        gateway_policy_from_dict(raw)


def test_gateway_rejects_upstream_redirects(monkeypatch) -> None:
    class Response:
        status = 302

        def getheaders(self):
            return [("location", "https://internal.example")]

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(
        "fugue.connectivity_gateway._PinnedHTTPSConnection", Connection
    )
    with pytest.raises(GatewayAuthorizationError, match="redirect"):
        _request_pinned_upstream(
            route=_policy().routes[0],
            method="POST",
            path="/v1/chat/completions",
            body=b"{}",
            headers={"content-type": "application/json"},
            addresses=("203.0.113.10",),
            verify=True,
            timeout=5,
            response_limit=4,
        )
