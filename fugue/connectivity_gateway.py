from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.client
import ipaddress
import json
import os
import re
import secrets
import socket
import sqlite3
import ssl
import threading
import time
from collections.abc import Mapping
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from fugue.bench.candidates import stable_digest

GATEWAY_POLICY_SCHEMA_VERSION = 1
CAPABILITY_SCHEMA_VERSION = 1
_MAX_TOKEN_BYTES = 16 * 1024
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_BLOCKED_FORWARD_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "host",
        "proxy-authorization",
        "x-api-key",
    }
)


@dataclass(frozen=True)
class GatewayRouteV1:
    id: str
    upstream: str
    methods: tuple[str, ...]
    path_prefixes: tuple[str, ...]
    content_types: tuple[str, ...]
    static_headers: tuple[tuple[str, str], ...] = ()
    credential_env: str | None = None
    credential_header: str = "authorization"
    credential_prefix: str = "Bearer "
    credential_encoding: str = "plain"
    max_request_bytes: int = 8 * 1024 * 1024
    max_response_bytes: int = 64 * 1024 * 1024

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["methods"] = list(self.methods)
        value["path_prefixes"] = list(self.path_prefixes)
        value["content_types"] = list(self.content_types)
        value["static_headers"] = dict(self.static_headers)
        return value


@dataclass(frozen=True)
class GatewayPolicyV1:
    schema_version: int
    routes: tuple[GatewayRouteV1, ...]
    max_token_lifetime_seconds: int
    policy_sha256: str = ""

    def unsigned_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["policy_sha256"] = ""
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "routes": [route.to_dict() for route in self.routes],
            "max_token_lifetime_seconds": self.max_token_lifetime_seconds,
            "policy_sha256": self.policy_sha256,
        }


@dataclass(frozen=True)
class CellCapabilityV1:
    schema_version: int
    issuer: str
    cell_id: str
    execution_fingerprint: str
    route_ids: tuple[str, ...]
    issued_at: int
    expires_at: int
    max_requests: int
    max_request_bytes: int
    max_response_bytes: int
    nonce: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GatewayDecisionV1:
    allowed: bool
    reason: str
    route_id: str | None = None
    upstream_url: str | None = None
    request_number: int | None = None


class GatewayAuthorizationError(ValueError):
    pass


def gateway_policy_from_dict(raw: Mapping[str, Any]) -> GatewayPolicyV1:
    _reject_unknown(
        raw,
        {"schema_version", "routes", "max_token_lifetime_seconds", "policy_sha256"},
        "gateway policy",
    )
    if int(raw.get("schema_version") or 0) != GATEWAY_POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported gateway policy schema")
    routes_raw = raw.get("routes")
    if not isinstance(routes_raw, list) or not routes_raw:
        raise ValueError("gateway policy routes must be a non-empty list")
    routes: list[GatewayRouteV1] = []
    seen: set[str] = set()
    for index, item in enumerate(routes_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"gateway route {index} must be an object")
        route_raw = {str(key): value for key, value in item.items()}
        _reject_unknown(
            route_raw,
            {
                "id",
                "upstream",
                "methods",
                "path_prefixes",
                "content_types",
                "static_headers",
                "credential_env",
                "credential_header",
                "credential_prefix",
                "credential_encoding",
                "max_request_bytes",
                "max_response_bytes",
            },
            "gateway route",
        )
        route_id = _identifier(route_raw.get("id"), "gateway route id")
        if route_id in seen:
            raise ValueError(f"duplicate gateway route id: {route_id}")
        seen.add(route_id)
        upstream = _locked_upstream(route_raw.get("upstream"))
        methods = tuple(
            dict.fromkeys(
                str(value).upper() for value in _nonempty_list(route_raw, "methods")
            )
        )
        if not set(methods) <= {"GET", "POST"}:
            raise ValueError("gateway routes may allow only GET and POST")
        prefixes = tuple(
            _path_prefix(value)
            for value in _nonempty_list(route_raw, "path_prefixes")
        )
        content_types = tuple(
            str(value).lower().strip()
            for value in _nonempty_list(route_raw, "content_types")
        )
        if any("*" in value for value in content_types):
            raise ValueError("gateway content types may not use wildcards")
        static_headers = _static_headers(route_raw.get("static_headers", {}))
        credential_env = route_raw.get("credential_env")
        if credential_env is not None:
            credential_env = _env_name(credential_env)
        header = str(
            route_raw.get("credential_header") or "authorization"
        ).lower()
        if (
            not re.fullmatch(r"[a-z0-9!#$%&'*+.^_`|~-]+", header)
            or header in _HOP_BY_HOP_HEADERS
            or header in {"host", "cookie"}
        ):
            raise ValueError("gateway credential header is unsafe")
        credential_prefix = str(
            route_raw.get("credential_prefix", "Bearer ")
        )
        if (
            len(credential_prefix) > 128
            or "\r" in credential_prefix
            or "\n" in credential_prefix
        ):
            raise ValueError("gateway credential prefix is unsafe")
        credential_encoding = str(
            route_raw.get("credential_encoding") or "plain"
        )
        if credential_encoding not in {"plain", "wandb-basic"}:
            raise ValueError("unsupported gateway credential encoding")
        routes.append(
            GatewayRouteV1(
                id=route_id,
                upstream=upstream,
                methods=methods,
                path_prefixes=prefixes,
                content_types=content_types,
                static_headers=static_headers,
                credential_env=credential_env,
                credential_header=header,
                credential_prefix=credential_prefix,
                credential_encoding=credential_encoding,
                max_request_bytes=_positive_int(
                    route_raw.get("max_request_bytes", 8 * 1024 * 1024),
                    "gateway request size",
                ),
                max_response_bytes=_positive_int(
                    route_raw.get("max_response_bytes", 64 * 1024 * 1024),
                    "gateway response size",
                ),
            )
        )
    policy = GatewayPolicyV1(
        schema_version=GATEWAY_POLICY_SCHEMA_VERSION,
        routes=tuple(routes),
        max_token_lifetime_seconds=_bounded_positive_int(
            raw.get("max_token_lifetime_seconds", 3600),
            "gateway token lifetime",
            maximum=3600,
        ),
    )
    digest = stable_digest(policy.unsigned_dict())
    supplied = str(raw.get("policy_sha256") or "")
    if supplied and not hmac.compare_digest(supplied, digest):
        raise ValueError("gateway policy digest does not match")
    return replace(policy, policy_sha256=digest)


def load_gateway_policy(path: Path) -> GatewayPolicyV1:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read gateway policy {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("gateway policy must be an object")
    return gateway_policy_from_dict(raw)


def mint_cell_capability(
    *,
    signing_key: bytes,
    issuer: str,
    cell_id: str,
    execution_fingerprint: str,
    route_ids: tuple[str, ...],
    lifetime_seconds: int,
    max_requests: int,
    max_request_bytes: int,
    max_response_bytes: int,
    now: int | None = None,
) -> str:
    if len(signing_key) < 32:
        raise ValueError("gateway signing key must contain at least 32 bytes")
    issued_at = int(time.time() if now is None else now)
    capability = CellCapabilityV1(
        schema_version=CAPABILITY_SCHEMA_VERSION,
        issuer=_identifier(issuer, "gateway issuer"),
        cell_id=_identifier(cell_id, "cell id"),
        execution_fingerprint=_sha256(
            execution_fingerprint, "execution fingerprint"
        ),
        route_ids=tuple(
            dict.fromkeys(_identifier(value, "gateway route id") for value in route_ids)
        ),
        issued_at=issued_at,
        expires_at=issued_at
        + _bounded_positive_int(
            lifetime_seconds, "gateway capability lifetime", maximum=3600
        ),
        max_requests=_positive_int(max_requests, "gateway request limit"),
        max_request_bytes=_positive_int(
            max_request_bytes, "gateway request size"
        ),
        max_response_bytes=_positive_int(
            max_response_bytes, "gateway response size"
        ),
        nonce=secrets.token_hex(16),
    )
    payload = _canonical_json(capability.to_dict())
    signature = hmac.new(signing_key, payload, hashlib.sha256).digest()
    return f"{_b64(payload)}.{_b64(signature)}"


def verify_cell_capability(
    token: str,
    *,
    signing_key: bytes,
    expected_issuer: str,
    now: int | None = None,
) -> CellCapabilityV1:
    if not token or len(token.encode()) > _MAX_TOKEN_BYTES:
        raise GatewayAuthorizationError("invalid capability token")
    try:
        payload_part, signature_part = token.split(".", 1)
        payload = _unb64(payload_part)
        signature = _unb64(signature_part)
        raw = json.loads(payload)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise GatewayAuthorizationError("invalid capability token") from exc
    expected = hmac.new(signing_key, payload, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise GatewayAuthorizationError("invalid capability signature")
    if not isinstance(raw, Mapping):
        raise GatewayAuthorizationError("invalid capability payload")
    _reject_unknown(
        raw,
        {
            "schema_version",
            "issuer",
            "cell_id",
            "execution_fingerprint",
            "route_ids",
            "issued_at",
            "expires_at",
            "max_requests",
            "max_request_bytes",
            "max_response_bytes",
            "nonce",
        },
        "gateway capability",
    )
    capability = CellCapabilityV1(
        schema_version=int(raw.get("schema_version") or 0),
        issuer=_identifier(raw.get("issuer"), "gateway issuer"),
        cell_id=_identifier(raw.get("cell_id"), "cell id"),
        execution_fingerprint=_sha256(
            raw.get("execution_fingerprint"), "execution fingerprint"
        ),
        route_ids=tuple(
            _identifier(value, "gateway route id")
            for value in _sequence(raw.get("route_ids"), "gateway route ids")
        ),
        issued_at=int(raw.get("issued_at") or 0),
        expires_at=int(raw.get("expires_at") or 0),
        max_requests=_positive_int(
            raw.get("max_requests"), "gateway request limit"
        ),
        max_request_bytes=_positive_int(
            raw.get("max_request_bytes"), "gateway request size"
        ),
        max_response_bytes=_positive_int(
            raw.get("max_response_bytes"), "gateway response size"
        ),
        nonce=_identifier(raw.get("nonce"), "gateway nonce"),
    )
    current = int(time.time() if now is None else now)
    if capability.schema_version != CAPABILITY_SCHEMA_VERSION:
        raise GatewayAuthorizationError("unsupported capability schema")
    if not hmac.compare_digest(capability.issuer, expected_issuer):
        raise GatewayAuthorizationError("capability issuer mismatch")
    if capability.issued_at > current + 30 or capability.expires_at <= current:
        raise GatewayAuthorizationError("capability expired or not yet valid")
    if capability.expires_at - capability.issued_at > 3600:
        raise GatewayAuthorizationError("capability lifetime exceeds policy")
    return capability


class GatewayUsageLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS capability_usage (
                        nonce TEXT PRIMARY KEY,
                        cell_id TEXT NOT NULL,
                        source_address TEXT NOT NULL,
                        expires_at INTEGER NOT NULL,
                        request_count INTEGER NOT NULL,
                        request_bytes INTEGER NOT NULL,
                        response_bytes INTEGER NOT NULL
                    );
                    """
                )
                columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(capability_usage)"
                    )
                }
                if "source_address" not in columns:
                    connection.execute(
                        "ALTER TABLE capability_usage "
                        "ADD COLUMN source_address TEXT NOT NULL DEFAULT ''"
                    )

    def claim_request(
        self,
        capability: CellCapabilityV1,
        *,
        request_bytes: int,
        source_address: str,
    ) -> int:
        if request_bytes < 0 or request_bytes > capability.max_request_bytes:
            raise GatewayAuthorizationError("request exceeds capability byte limit")
        source = _source_address(source_address)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT cell_id, source_address, request_count, request_bytes "
                    "FROM capability_usage WHERE nonce = ?",
                    (capability.nonce,),
                ).fetchone()
                if row is None:
                    request_number = 1
                    total_request_bytes = request_bytes
                    connection.execute(
                        "INSERT INTO capability_usage "
                        "(nonce, cell_id, source_address, expires_at, "
                        "request_count, request_bytes, response_bytes) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0)",
                        (
                            capability.nonce,
                            capability.cell_id,
                            source,
                            capability.expires_at,
                            request_number,
                            total_request_bytes,
                        ),
                    )
                else:
                    if not hmac.compare_digest(str(row[0]), capability.cell_id):
                        raise GatewayAuthorizationError(
                            "capability replayed across cells"
                        )
                    if not hmac.compare_digest(str(row[1]), source):
                        raise GatewayAuthorizationError(
                            "capability replayed from another sandbox"
                        )
                    request_number = int(row[2]) + 1
                    total_request_bytes = int(row[3]) + request_bytes
                    if request_number > capability.max_requests:
                        raise GatewayAuthorizationError(
                            "capability request limit exceeded"
                        )
                    if total_request_bytes > capability.max_request_bytes:
                        raise GatewayAuthorizationError(
                            "capability cumulative request byte limit exceeded"
                        )
                    connection.execute(
                        "UPDATE capability_usage SET request_count = ?, "
                        "request_bytes = ? WHERE nonce = ?",
                        (request_number, total_request_bytes, capability.nonce),
                    )
                return request_number

    def record_response(
        self, capability: CellCapabilityV1, *, response_bytes: int
    ) -> None:
        if response_bytes < 0 or response_bytes > capability.max_response_bytes:
            raise GatewayAuthorizationError("response exceeds capability byte limit")
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT response_bytes FROM capability_usage WHERE nonce = ?",
                    (capability.nonce,),
                ).fetchone()
                if row is None:
                    raise GatewayAuthorizationError("capability usage is missing")
                total = int(row[0]) + response_bytes
                if total > capability.max_response_bytes:
                    raise GatewayAuthorizationError(
                        "capability cumulative response byte limit exceeded"
                    )
                connection.execute(
                    "UPDATE capability_usage SET response_bytes = ? WHERE nonce = ?",
                    (total, capability.nonce),
                )

    def purge_expired(self, *, now: int | None = None) -> int:
        current = int(time.time() if now is None else now)
        with closing(self._connect()) as connection:
            with connection:
                result = connection.execute(
                    "DELETE FROM capability_usage WHERE expires_at < ?", (current,)
                )
                return int(result.rowcount)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection


def authorize_gateway_request(
    *,
    policy: GatewayPolicyV1,
    capability: CellCapabilityV1,
    route_id: str,
    method: str,
    suffix: str,
    content_type: str,
    request_bytes: int,
) -> GatewayDecisionV1:
    routes = {route.id: route for route in policy.routes}
    route = routes.get(route_id)
    if route is None or route_id not in capability.route_ids:
        return GatewayDecisionV1(False, "route is not authorized")
    selected_method = method.upper()
    if selected_method not in route.methods:
        return GatewayDecisionV1(False, "method is not authorized", route_id)
    path = _request_suffix(suffix)
    if not any(path == prefix or path.startswith(f"{prefix}/") for prefix in route.path_prefixes):
        return GatewayDecisionV1(False, "path is not authorized", route_id)
    media_type = content_type.partition(";")[0].strip().lower()
    if selected_method == "POST" and media_type not in route.content_types:
        return GatewayDecisionV1(
            False, "content type is not authorized", route_id
        )
    limit = min(route.max_request_bytes, capability.max_request_bytes)
    if request_bytes > limit:
        return GatewayDecisionV1(False, "request exceeds byte limit", route_id)
    upstream = f"{route.upstream.rstrip('/')}{path}"
    return GatewayDecisionV1(True, "allowed", route_id, upstream)


def _forward_headers(
    incoming: Mapping[str, str], route: GatewayRouteV1
) -> dict[str, str]:
    forwarded = dict(route.static_headers)
    for name in ("accept", "content-type", "openai-beta", "user-agent"):
        value = incoming.get(name)
        if value:
            forwarded[name] = value[:4096]
    for name in incoming:
        lowered = name.lower()
        if lowered in _HOP_BY_HOP_HEADERS or lowered in _BLOCKED_FORWARD_HEADERS:
            continue
        if lowered.startswith(("x-forwarded-", "x-real-", "proxy-")):
            continue
    if route.credential_env:
        credential = os.environ.get(route.credential_env, "")
        if not credential:
            raise GatewayAuthorizationError(
                f"credential binding unavailable for route {route.id}"
            )
        if route.credential_encoding == "wandb-basic":
            credential = base64.b64encode(
                f"api:{credential}".encode()
            ).decode()
        forwarded[route.credential_header] = (
            f"{route.credential_prefix}{credential}"
        )
    return forwarded


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        hostname: str,
        address: str,
        *,
        port: int,
        context: ssl.SSLContext,
        timeout: float,
    ) -> None:
        super().__init__(hostname, port=port, timeout=timeout, context=context)
        self._pinned_address = address

    def connect(self) -> None:
        raw = socket.create_connection(
            (self._pinned_address, self.port),
            timeout=self.timeout,
        )
        try:
            self.sock = self._context.wrap_socket(
                raw,
                server_hostname=self.host,
            )
        except BaseException:
            raw.close()
            raise


def _request_pinned_upstream(
    *,
    route: GatewayRouteV1,
    method: str,
    path: str,
    body: bytes,
    headers: Mapping[str, str],
    addresses: tuple[str, ...],
    verify: str | bool,
    timeout: float,
    response_limit: int,
) -> tuple[int, dict[str, str], bytes]:
    split = urlsplit(route.upstream)
    hostname = split.hostname
    if hostname is None:
        raise GatewayAuthorizationError("gateway upstream has no hostname")
    context = (
        ssl.create_default_context()
        if verify is True
        else ssl.create_default_context(cafile=str(verify))
    )
    errors: list[BaseException] = []
    for address in addresses:
        connection = _PinnedHTTPSConnection(
            hostname,
            address,
            port=split.port or 443,
            context=context,
            timeout=timeout,
        )
        try:
            connection.request(
                method,
                path,
                body=body or None,
                headers=dict(headers),
                encode_chunked=False,
            )
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise GatewayAuthorizationError(
                    "upstream redirects are forbidden"
                )
            response_headers = {
                str(name).lower(): str(value)
                for name, value in response.getheaders()
            }
            declared = response_headers.get("content-length")
            if declared is not None:
                try:
                    declared_bytes = int(declared)
                except ValueError as exc:
                    raise GatewayAuthorizationError(
                        "upstream returned an invalid content length"
                    ) from exc
                if declared_bytes < 0 or declared_bytes > response_limit:
                    raise GatewayAuthorizationError(
                        "upstream response exceeds byte limit"
                    )
            response_body = response.read(response_limit + 1)
            if len(response_body) > response_limit:
                raise GatewayAuthorizationError(
                    "upstream response exceeds byte limit"
                )
            return response.status, response_headers, response_body
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            errors.append(exc)
        finally:
            connection.close()
    if errors:
        raise errors[-1]
    raise GatewayAuthorizationError("gateway upstream has no validated address")


class _GatewayHandler(BaseHTTPRequestHandler):
    server: GatewayServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.client_timeout)

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_CONNECT(self) -> None:  # noqa: N802
        self.send_error(405, "CONNECT is not supported")

    def _handle(self) -> None:
        started = time.monotonic()
        route_id: str | None = None
        capability: CellCapabilityV1 | None = None
        acquired = self.server.concurrency.acquire(blocking=False)
        if not acquired:
            self.send_error(503, "gateway concurrency limit reached")
            return
        try:
            if self.headers.get("transfer-encoding"):
                raise GatewayAuthorizationError(
                    "transfer encoding is not supported"
                )
            route_id, suffix = _parse_gateway_path(self.path)
            token = _capability_token(
                self.headers.get("authorization", "")
            )
            capability = verify_cell_capability(
                token,
                signing_key=self.server.signing_key,
                expected_issuer=self.server.issuer,
            )
            if (
                capability.expires_at - capability.issued_at
                > self.server.policy.max_token_lifetime_seconds
            ):
                raise GatewayAuthorizationError(
                    "capability lifetime exceeds gateway policy"
                )
            content_length = _bounded_content_length(
                self.headers.get("content-length")
            )
            body = self.rfile.read(content_length) if content_length else b""
            decision = authorize_gateway_request(
                policy=self.server.policy,
                capability=capability,
                route_id=route_id,
                method=self.command,
                suffix=suffix,
                content_type=self.headers.get("content-type", ""),
                request_bytes=len(body),
            )
            if not decision.allowed or decision.upstream_url is None:
                raise GatewayAuthorizationError(decision.reason)
            request_number = self.server.ledger.claim_request(
                capability,
                request_bytes=len(body),
                source_address=str(self.client_address[0]),
            )
            route = next(
                item for item in self.server.policy.routes if item.id == route_id
            )
            addresses = validate_upstream_resolution(route.upstream)
            response_limit = min(
                route.max_response_bytes, capability.max_response_bytes
            )
            status_code, response_headers, response_body = _request_pinned_upstream(
                route=route,
                method=self.command,
                path=_request_suffix(suffix),
                body=body,
                headers=_forward_headers(
                    {key.lower(): value for key, value in self.headers.items()},
                    route,
                ),
                addresses=addresses,
                verify=self.server.upstream_ca,
                timeout=self.server.timeout,
                response_limit=response_limit,
            )
            if 300 <= status_code < 400:
                raise GatewayAuthorizationError("upstream redirects are forbidden")
            if len(response_body) > response_limit:
                raise GatewayAuthorizationError("upstream response exceeds byte limit")
            self.server.ledger.record_response(
                capability, response_bytes=len(response_body)
            )
            self.send_response(status_code)
            for name in ("content-type", "x-request-id"):
                value = response_headers.get(name)
                if value:
                    self.send_header(name, value[:4096])
            self.send_header("content-length", str(len(response_body)))
            self.send_header("x-fugue-gateway-route", route_id)
            self.send_header("x-fugue-gateway-request", str(request_number))
            self.end_headers()
            self.wfile.write(response_body)
            self.server.audit(
                {
                    "timestamp": int(time.time()),
                    "event": "request",
                    "allowed": True,
                    "route_id": route_id,
                    "cell_id": capability.cell_id,
                    "request_number": request_number,
                    "status_code": status_code,
                    "request_bytes": len(body),
                    "response_bytes": len(response_body),
                    "latency_ms": round(
                        (time.monotonic() - started) * 1000, 3
                    ),
                }
            )
        except GatewayAuthorizationError as exc:
            self.server.audit(
                {
                    "timestamp": int(time.time()),
                    "event": "policy_denial",
                    "allowed": False,
                    "route_id": route_id,
                    "cell_id": (
                        capability.cell_id if capability is not None else None
                    ),
                    "reason": str(exc)[:200],
                    "latency_ms": round(
                        (time.monotonic() - started) * 1000, 3
                    ),
                }
            )
            self.send_error(403, str(exc))
        except (ValueError, OSError, ssl.SSLError, http.client.HTTPException) as exc:
            self.server.audit(
                {
                    "timestamp": int(time.time()),
                    "event": "upstream_error",
                    "allowed": False,
                    "route_id": route_id,
                    "cell_id": (
                        capability.cell_id if capability is not None else None
                    ),
                    "error_type": type(exc).__name__,
                    "latency_ms": round(
                        (time.monotonic() - started) * 1000, 3
                    ),
                }
            )
            self.send_error(502, f"gateway request failed: {type(exc).__name__}")
        finally:
            self.server.concurrency.release()

    def log_message(self, format: str, *args: Any) -> None:
        # Structured audit calls above deliberately omit URLs, tokens, and bodies.
        return


class GatewayServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        *,
        policy: GatewayPolicyV1,
        signing_key: bytes,
        issuer: str,
        ledger: GatewayUsageLedger,
        upstream_ca: str | bool = True,
        timeout: float = 60,
        client_timeout: float = 15,
        max_concurrency: int = 32,
        audit_path: Path | None = None,
    ) -> None:
        if upstream_ca is False:
            raise ValueError("gateway upstream TLS verification may not be disabled")
        if max_concurrency < 1 or max_concurrency > 256:
            raise ValueError("gateway concurrency must be between 1 and 256")
        super().__init__(address, _GatewayHandler)
        self.policy = policy
        self.signing_key = signing_key
        self.issuer = issuer
        self.ledger = ledger
        self.upstream_ca = upstream_ca
        self.timeout = timeout
        self.client_timeout = client_timeout
        self.concurrency = threading.BoundedSemaphore(max_concurrency)
        self.audit_path = audit_path
        self._audit_lock = threading.Lock()

    def audit(self, record: Mapping[str, Any]) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fugue-connectivity-gateway")
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--issuer", required=True)
    parser.add_argument("--tls-cert", type=Path, required=True)
    parser.add_argument("--tls-key", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    key_text = os.environ.get("FUGUE_GATEWAY_SIGNING_KEY", "")
    try:
        signing_key = base64.b64decode(key_text, validate=True)
    except ValueError as exc:
        raise SystemExit("FUGUE_GATEWAY_SIGNING_KEY must be base64") from exc
    if len(signing_key) < 32:
        raise SystemExit("FUGUE_GATEWAY_SIGNING_KEY must decode to at least 32 bytes")
    server = GatewayServer(
        (args.host, args.port),
        policy=load_gateway_policy(args.policy),
        signing_key=signing_key,
        issuer=args.issuer,
        ledger=GatewayUsageLedger(args.ledger),
        audit_path=args.audit,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(args.tls_cert, args.tls_key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


def _parse_gateway_path(path: str) -> tuple[str, str]:
    split = urlsplit(path)
    if split.query or split.fragment:
        raise GatewayAuthorizationError("query strings are forbidden")
    parts = split.path.split("/")
    if len(parts) < 4 or parts[1] != "routes":
        raise GatewayAuthorizationError("invalid gateway route")
    return _identifier(parts[2], "gateway route id"), "/" + "/".join(parts[3:])


def _request_suffix(value: str) -> str:
    split = urlsplit(value)
    if split.scheme or split.netloc or split.query or split.fragment:
        raise GatewayAuthorizationError("request path must be relative and query-free")
    path = split.path
    if not path.startswith("/") or "\\" in path or "\x00" in path:
        raise GatewayAuthorizationError("invalid request path")
    if any(part in {".", ".."} for part in path.split("/")):
        raise GatewayAuthorizationError("request path traversal is forbidden")
    return path


def _static_headers(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise ValueError("gateway static headers must be an object")
    headers: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name).lower().strip()
        header_value = str(raw_value).strip()
        if (
            not name
            or len(name) > 128
            or not re.fullmatch(r"[a-z0-9!#$%&'*+.^_`|~-]+", name)
            or name in _HOP_BY_HOP_HEADERS
            or name in _BLOCKED_FORWARD_HEADERS
            or name.startswith(("x-forwarded-", "x-real-", "proxy-"))
        ):
            raise ValueError("gateway static header name is unsafe")
        if (
            not header_value
            or len(header_value) > 4096
            or "\r" in header_value
            or "\n" in header_value
        ):
            raise ValueError("gateway static header value is unsafe")
        headers[name] = header_value
    return tuple(sorted(headers.items()))


def _locked_upstream(value: Any) -> str:
    text = str(value or "").strip().rstrip("/")
    split = urlsplit(text)
    if (
        split.scheme != "https"
        or not split.hostname
        or split.username
        or split.password
        or split.query
        or split.fragment
        or split.path not in {"", "/"}
    ):
        raise ValueError("gateway upstream must be an origin-only HTTPS URL")
    hostname = split.hostname
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("gateway upstream may not be a private or special address")
    return text


def validate_upstream_resolution(upstream: str) -> tuple[str, ...]:
    """Reject DNS answers that could reach local, metadata, or private networks."""
    hostname = urlsplit(_locked_upstream(upstream)).hostname
    if hostname is None:
        raise GatewayAuthorizationError("gateway upstream hostname is missing")
    try:
        answers = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise GatewayAuthorizationError("gateway upstream DNS failed") from exc
    addresses = tuple(
        dict.fromkeys(str(answer[4][0]).split("%", 1)[0] for answer in answers)
    )
    if not addresses:
        raise GatewayAuthorizationError("gateway upstream DNS returned no addresses")
    for value in addresses:
        address = ipaddress.ip_address(value)
        if not address.is_global:
            raise GatewayAuthorizationError(
                "gateway upstream DNS resolved to a private or special address"
            )
    return addresses


def _capability_token(value: str) -> str:
    scheme, _, token = value.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    if scheme.lower() == "basic" and token:
        try:
            decoded = base64.b64decode(token, validate=True).decode()
            username, separator, password = decoded.partition(":")
        except (ValueError, UnicodeDecodeError) as exc:
            raise GatewayAuthorizationError(
                "missing gateway capability"
            ) from exc
        if separator and username == "api" and password:
            return password
    raise GatewayAuthorizationError("missing gateway capability")


def _bounded_content_length(value: str | None) -> int:
    if not value:
        return 0
    try:
        length = int(value)
    except ValueError as exc:
        raise GatewayAuthorizationError("invalid content length") from exc
    if length < 0 or length > 64 * 1024 * 1024:
        raise GatewayAuthorizationError("content length exceeds gateway maximum")
    return length


def _source_address(value: str) -> str:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError as exc:
        raise GatewayAuthorizationError(
            "gateway client address is invalid"
        ) from exc
    return address.compressed


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _identifier(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if (
        not text
        or len(text) > 160
        or not all(character.isalnum() or character in "._:-" for character in text)
    ):
        raise ValueError(f"invalid {label}")
    return text


def _sha256(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return text


def _env_name(value: Any) -> str:
    text = str(value or "")
    if not text or not text.replace("_", "").isalnum() or text != text.upper():
        raise ValueError("credential binding must be an uppercase environment name")
    return text


def _path_prefix(value: Any) -> str:
    path = _request_suffix(str(value))
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path


def _nonempty_list(raw: Mapping[str, Any], name: str) -> list[Any]:
    value = raw.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _positive_int(value: Any, label: str) -> int:
    try:
        selected = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if selected <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return selected


def _bounded_positive_int(value: Any, label: str, *, maximum: int) -> int:
    selected = _positive_int(value, label)
    if selected > maximum:
        raise ValueError(f"{label} may not exceed {maximum}")
    return selected


def _reject_unknown(
    raw: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
