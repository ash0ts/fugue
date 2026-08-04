"""Freeze auditable GitHub candidate-discovery receipts for public task sampling.

This module deliberately stops at *candidate discovery*. Search results are not
eligibility labels, population truth, or tasks. Trusted reviewers and lane-
specific preparation code must verify immutable repository/source identities,
classify eligibility, collapse shared lineages, and seal selection before any
Agent trial can run.

The collector is intentionally dependency-free so the exact query, pagination,
and response-byte contract can be tested offline. Generated responses belong in
``.fugue/qualification`` (or another private evidence directory), never in a
trial image or a source pull request.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SCHEMA_VERSION = 2
API_ORIGIN = "https://api.github.com"
SUPPORTED_ENDPOINTS = {"search/issues", "search/repositories", "search/code"}
SUPPORTED_DATE_QUALIFIERS = {"closed", "created", "merged", "pushed", "updated"}
_PLAN_KEYS = {
    "schema_version",
    "plan_id",
    "lane_id",
    "candidate_discovery_only",
    "api_version",
    "source_query_plan_sha256",
    "source_sampling_protocol_sha256",
    "sampling_protocol_id",
    "population_scope",
    "temporal_scope",
    "selection_source_cutoff_utc",
    "acquisition_repetitions",
    "index_stabilization_seconds",
    "repeat_separation_seconds",
    "queries",
    "completeness",
    "deduplication",
    "public_visibility",
    "credential_profile",
    "rate_limit_policy",
    "max_response_bytes",
    "collector_implementation",
}
_QUERY_KEYS = {
    "query_id",
    "endpoint",
    "query",
    "date_qualifier",
    "start_utc",
    "end_utc",
    "max_results_per_shard",
    "per_page",
    "sort",
    "order",
    "shard_granularity",
    "unsplittable_overflow_policy",
}
_COMPLETENESS_KEYS = {
    "require_incomplete_results_false",
    "require_all_pages",
    "record_response_bytes",
    "fail_if_unsplittable",
}
_DEDUP_KEYS = {
    "primary_identity_fields",
    "candidate_only_not_lineage_deduplication",
}
_PUBLIC_VISIBILITY_KEYS = {
    "required_query_qualifier",
    "independent_repository_lookup",
    "persist_search_body_before_visibility_verification",
    "persist_repository_response_body",
}
_CREDENTIAL_PROFILE_KEYS = {
    "profile_id",
    "token_env_name",
    "credential_required",
    "credential_value_serialized",
}
_RATE_LIMIT_POLICY_KEYS = {
    "required_header_names",
    "minimum_remaining",
    "maximum_wait_seconds",
}
_IMPLEMENTATION_KEYS = {
    "source_commit",
    "source_tree",
    "collector_sha256",
    "compiler_sha256",
    "files_match_commit",
    "lock_sha256",
}
_ACQUISITION_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "lane_id",
    "status",
    "candidate_discovery_only",
    "eligibility_or_population_claim",
    "plan_sha256",
    "source_query_plan_sha256",
    "source_sampling_protocol_sha256",
    "sampling_protocol_id",
    "acquisition_index",
    "qualification_mode",
    "server_start_utc",
    "server_end_utc",
    "credential_profile",
    "responses",
    "shards",
    "response_count",
    "candidate_count",
    "candidates_path",
    "candidates_sha256",
    "query_snapshots_path",
    "query_snapshots_sha256",
    "privacy_scan",
    "repository_visibility_receipts",
    "repository_visibility_receipt_count",
    "checkpoint_sha256",
    "collector_implementation",
    "receipt_sha256",
}
_RESPONSE_RECORD_KEYS = {
    "query_id",
    "acquisition_index",
    "shard_start_utc",
    "shard_end_utc",
    "page",
    "requested_url",
    "url",
    "redirect_chain",
    "request_started_at_utc",
    "request_completed_at_utc",
    "request_url_sha256",
    "status",
    "headers",
    "headers_sha256",
    "header_privacy_findings",
    "body_sha256",
    "body_path",
    "total_count",
    "item_count",
    "incomplete_results",
}
_SHARD_RECORD_KEYS = {
    "query_id",
    "acquisition_index",
    "start_utc",
    "end_utc",
    "total_count",
    "page_count",
    "identity_count",
    "ordered_identity_sha256",
}
_PRIVACY_SCAN_KEYS = {
    "scanner_revision",
    "pattern_manifest_sha256",
    "pattern_ids_checked",
    "finding_field_count",
    "finding_occurrence_count",
    "affected_candidate_count",
    "affected_candidate_references",
    "findings",
    "matched_values_serialized",
    "raw_source_redacted",
    "downstream_authoring_export_blocked",
    "status",
    "required_resolution",
}
_PRIVACY_FINDING_KEYS = {
    "pattern_id",
    "json_path",
    "matched_surface",
    "occurrence_count",
    "query_id",
    "acquisition_index",
    "response_body_sha256",
    "page",
    "candidate_reference",
}
_CANDIDATE_ROW_KEYS = {
    "candidate_identity",
    "query_ids",
    "item_sha256",
    "observed_item_sha256s",
    "item",
    "privacy_review",
}
_PRIVACY_REVIEW_KEYS = {
    "status",
    "pattern_ids",
    "matched_values_serialized",
    "raw_source_redacted",
    "downstream_authoring_export_allowed",
}
_VISIBILITY_RECEIPT_KEYS = {
    "repository_url",
    "repository_full_name",
    "repository_node_id",
    "repository_id",
    "visibility",
    "private",
    "requested_url",
    "url",
    "redirect_chain",
    "request_started_at_utc",
    "request_completed_at_utc",
    "request_url_sha256",
    "status",
    "headers",
    "headers_sha256",
    "header_privacy_findings",
    "response_body_sha256",
    "response_body_persisted",
    "receipt_sha256",
}
_CHECKPOINT_KEYS = {
    "schema_version",
    "status",
    "plan_sha256",
    "acquisition_index",
    "qualification_mode",
    "source_sampling_protocol_sha256",
    "sampling_protocol_id",
    "responses",
    "repository_visibility_receipts",
    "collector_implementation",
    "checkpoint_sha256",
}
_FINAL_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "lane_id",
    "status",
    "candidate_discovery_only",
    "eligibility_or_population_claim",
    "public_only_repository_visibility_verified",
    "plan_sha256",
    "api_origin",
    "api_version",
    "source_query_plan_sha256",
    "source_sampling_protocol_sha256",
    "sampling_protocol_id",
    "qualification_mode",
    "temporal_scope",
    "selection_source_cutoff_utc",
    "acquisition_repetitions",
    "index_stabilization_seconds",
    "repeat_separation_seconds",
    "acquisition_windows",
    "acquisition_receipts",
    "credential_profile",
    "collector_implementation",
    "public_visibility",
    "query_count",
    "shards",
    "responses",
    "response_count",
    "repository_visibility_receipts",
    "repository_visibility_receipt_count",
    "candidate_count",
    "candidates_path",
    "candidates_sha256",
    "privacy_scan",
    "limitations",
    "receipt_sha256",
}
_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_NAMES = {"authorization", "cookie", "set-cookie", "x-api-key"}
_SAFE_HEADER_NAMES = {
    "date",
    "etag",
    "last-modified",
    "link",
    "x-github-api-version-selected",
    "x-github-request-id",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-ratelimit-resource",
    "x-ratelimit-used",
}
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40,64}$")
_PUBLIC_QUALIFIED = "public_qualified"
_DIAGNOSTIC_NONQUALIFYING = "diagnostic_nonqualifying"
_LIVE_TRANSPORT_SENTINEL = object()
_PRIVACY_SCANNER_REVISION = "public-source-credential-patterns-v2"
_FINAL_LIMITATIONS = (
    "GitHub Search is candidate discovery, not a complete or representative population frame.",
    "Search-index eligibility, ranking, and update timing remain platform selection mechanisms.",
    "Every candidate still requires immutable-source verification, blinded eligibility review, lineage collapse, and externally randomized selection.",
)
_FINAL_PUBLICATION_MARKER_KEYS = {
    "schema_version",
    "target_name",
    "staging_name",
    "receipt_sha256",
}
_PRIVACY_PATTERNS = {
    "anthropic_api_key": re.compile(r"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_-]{20,}"),
    "aws_access_key_id": re.compile(
        r"(?<![A-Z0-9])(AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"
    ),
    "google_api_key": re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}"),
    "github_fine_grained_token": re.compile(
        r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{40,}"
    ),
    "github_legacy_token": re.compile(r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36,}"),
    "openai_api_key": re.compile(
        r"(?<![A-Za-z0-9_-])sk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}"
    ),
    "npm_token": re.compile(r"(?<![A-Za-z0-9_])npm_[A-Za-z0-9]{36,}"),
    "private_key_pem": re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "pypi_token": re.compile(r"(?<![A-Za-z0-9_-])pypi-[A-Za-z0-9_-]{40,}"),
    "slack_token": re.compile(r"(?<![A-Za-z0-9-])xox[baprs]-[A-Za-z0-9-]{20,}"),
    "stripe_live_secret": re.compile(r"(?<![A-Za-z0-9_])sk_live_[A-Za-z0-9]{20,}"),
    "gitlab_personal_access_token": re.compile(
        r"(?<![A-Za-z0-9_-])glpat-[A-Za-z0-9_-]{20,}"
    ),
}


class PopulationDiscoveryError(ValueError):
    """Raised when discovery cannot produce a complete, auditable receipt."""


class PublicationRecoveryRequired(PopulationDiscoveryError):
    """Raised after namespace publication when durability needs recovery."""


@dataclass(frozen=True)
class FrozenResponse:
    """One exact HTTP response returned by a collector transport."""

    url: str
    status: int
    headers: Mapping[str, str]
    body: bytes
    requested_url: str | None = None
    redirect_chain: tuple[str, ...] = ()
    request_started_at_utc: str | None = None
    request_completed_at_utc: str | None = None


Transport = Callable[[str, Mapping[str, str]], FrozenResponse]
Pacer = Callable[[FrozenResponse], None]
CheckpointWriter = Callable[[Mapping[str, Any], Mapping[str, Mapping[str, Any]]], None]


class _RejectRedirects(HTTPRedirectHandler):
    """Reject every redirect before urllib can forward a credential."""

    def redirect_request(  # type: ignore[override]
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        raise PopulationDiscoveryError(
            f"GitHub request returned an unapproved redirect (HTTP {code})"
        )


def _open_no_redirect(request: Request, *, timeout: int) -> Any:
    """Open one request with redirect following disabled.

    Kept as a narrow seam so offline tests can emulate the authoritative HTTP
    response while the qualifying transport itself remains the exact built-in
    type.
    """

    return build_opener(_RejectRedirects()).open(request, timeout=timeout)


class GitHubLiveTransportV2:
    """The only transport that may produce public-qualified acquisition evidence."""

    __slots__ = (
        "_api_version",
        "_credential_fingerprint",
        "_profile_id",
        "_token",
        "_token_env_name",
        "_max_response_bytes",
    )

    def __init__(
        self,
        sentinel: object,
        *,
        api_version: str,
        profile_id: str,
        token_env_name: str,
        token: str,
        max_response_bytes: int,
    ) -> None:
        if sentinel is not _LIVE_TRANSPORT_SENTINEL:
            raise PopulationDiscoveryError(
                "GitHub live transports must be created from the locked environment"
            )
        self._api_version = api_version
        self._profile_id = profile_id
        self._token_env_name = token_env_name
        self._token = token
        self._credential_fingerprint = _sha256_bytes(token.encode("utf-8"))
        self._max_response_bytes = max_response_bytes

    def __repr__(self) -> str:
        return (
            "GitHubLiveTransportV2("
            f"api_version={self._api_version!r}, "
            f"profile_id={self._profile_id!r}, "
            f"token_env_name={self._token_env_name!r})"
        )

    def verify_environment(self, normalized: Mapping[str, Any]) -> None:
        profile = normalized["credential_profile"]
        if (
            self._api_version != normalized["api_version"]
            or self._profile_id != profile["profile_id"]
            or self._token_env_name != profile["token_env_name"]
            or self._max_response_bytes != normalized["max_response_bytes"]
        ):
            raise PopulationDiscoveryError(
                "GitHub live transport differs from the locked plan"
            )
        current = os.environ.get(self._token_env_name)
        if (
            not current
            or _sha256_bytes(current.encode("utf-8")) != self._credential_fingerprint
        ):
            raise PopulationDiscoveryError(
                "the exact lane-bound GitHub credential is no longer present"
            )

    def __call__(self, url: str, params: Mapping[str, str]) -> FrozenResponse:
        request_url = _expected_request_url(url, params)
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "fugue-public-population-freezer-v2",
            "X-GitHub-Api-Version": self._api_version,
            "Authorization": f"Bearer {self._token}",
        }
        request = Request(request_url, headers=headers, method="GET")
        started = datetime.now(UTC)
        try:
            with _open_no_redirect(request, timeout=60) as response:
                content_length = response.headers.get("Content-Length")
                if (
                    isinstance(content_length, str)
                    and content_length.isdigit()
                    and int(content_length) > self._max_response_bytes
                ):
                    raise PopulationDiscoveryError(
                        "GitHub response exceeds the locked byte limit"
                    )
                body = response.read(self._max_response_bytes + 1)
                if len(body) > self._max_response_bytes:
                    raise PopulationDiscoveryError(
                        "GitHub response exceeds the locked byte limit"
                    )
                completed = datetime.now(UTC)
                final_url = response.geturl()
                return FrozenResponse(
                    url=final_url,
                    status=response.status,
                    headers=dict(response.headers.items()),
                    body=body,
                    requested_url=request_url,
                    redirect_chain=(),
                    request_started_at_utc=_format_utc(started),
                    request_completed_at_utc=_format_utc(completed),
                )
        except PopulationDiscoveryError:
            raise
        except (HTTPError, URLError, TimeoutError) as exc:
            raise PopulationDiscoveryError(f"GitHub request failed: {exc}") from exc


def _strict_json_loads(value: str | bytes, *, field: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PopulationDiscoveryError(
                    f"{field} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=reject_duplicates)
    except PopulationDiscoveryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PopulationDiscoveryError(f"{field} is malformed JSON") from exc


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def stable_digest(value: object) -> str:
    """Return the canonical SHA-256 identity for a JSON-compatible value."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_token_env_name(value: object) -> str:
    if not isinstance(value, str) or _ENV_NAME.fullmatch(value) is None:
        raise PopulationDiscoveryError(
            "token environment name must be an uppercase environment identifier"
        )
    return value


def _reject_symlink_ancestors(path: Path, *, field: str) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        try:
            if candidate.is_symlink():
                raise PopulationDiscoveryError(
                    f"{field} may not contain a symlink ancestor: {candidate}"
                )
        except OSError as exc:
            raise PopulationDiscoveryError(f"cannot inspect {field}: {exc}") from exc


def _ensure_private_directory(path: Path) -> None:
    _reject_symlink_ancestors(path, field="evidence directory")
    try:
        if path.exists() and not path.is_dir():
            raise PopulationDiscoveryError(f"evidence path is not a directory: {path}")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
    except PopulationDiscoveryError:
        raise
    except OSError as exc:
        raise PopulationDiscoveryError(
            f"cannot prepare evidence directory: {exc}"
        ) from exc


def _write_private(path: Path, value: bytes, *, allow_identical: bool = False) -> None:
    _reject_symlink_ancestors(path, field="evidence file")
    try:
        if path.exists():
            metadata = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise PopulationDiscoveryError(
                    f"evidence path is not a regular file: {path}"
                )
            if allow_identical and path.read_bytes() == value:
                os.chmod(path, 0o600)
                return
            raise PopulationDiscoveryError(
                f"refusing to overwrite evidence file: {path}"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        os.chmod(path, 0o600)
    except PopulationDiscoveryError:
        raise
    except OSError as exc:
        raise PopulationDiscoveryError(f"cannot write private evidence: {exc}") from exc


def _replace_private(path: Path, value: bytes) -> None:
    _reject_symlink_ancestors(path, field="checkpoint")
    _ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        if temporary.exists() or temporary.is_symlink():
            raise PopulationDiscoveryError("checkpoint temporary path already exists")
        _write_private(temporary, value)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except PopulationDiscoveryError:
        raise
    except OSError as exc:
        raise PopulationDiscoveryError(f"cannot replace checkpoint: {exc}") from exc


def _read_private(path: Path, *, name: str, maximum_bytes: int | None = None) -> bytes:
    _reject_symlink_ancestors(path, field=name)
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise PopulationDiscoveryError(f"cannot inspect {name}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise PopulationDiscoveryError(f"{name} must be a regular file")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        os.close(descriptor)
        raise PopulationDiscoveryError(
            f"{name} must not be accessible by group or other"
        )
    if maximum_bytes is not None and metadata.st_size > maximum_bytes:
        os.close(descriptor)
        raise PopulationDiscoveryError(f"{name} exceeds the locked byte limit")
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            if maximum_bytes is None:
                return handle.read()
            value = handle.read(maximum_bytes + 1)
            if len(value) > maximum_bytes or handle.read(1):
                raise PopulationDiscoveryError(f"{name} exceeds the locked byte limit")
            return value
    except PopulationDiscoveryError:
        raise
    except OSError as exc:
        raise PopulationDiscoveryError(f"cannot read {name}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _locked_artifact_path(
    directory: Path,
    value: object,
    *,
    expected: str,
    field: str,
) -> Path:
    if value != expected:
        raise PopulationDiscoveryError(f"{field} must be exactly {expected!r}")
    return directory / expected


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], field: str
) -> None:
    keys = set(value)
    if keys != expected:
        raise PopulationDiscoveryError(
            f"{field} has invalid keys; "
            f"missing={sorted(expected - keys)}, unknown={sorted(keys - expected)}"
        )


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PopulationDiscoveryError(f"{field} must be an object")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise PopulationDiscoveryError(f"{field} must be boolean")
    return value


def _require_int(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise PopulationDiscoveryError(
            f"{field} must be an integer in {minimum}{suffix}"
        )
    return value


def _require_text(value: object, field: str, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(character in value for character in "\x00\r\n")
    ):
        raise PopulationDiscoveryError(f"{field} must be non-empty single-line text")
    return value


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PopulationDiscoveryError(f"{field} must be a lowercase SHA-256")
    return value


def _require_git_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise PopulationDiscoveryError(f"{field} must be a lowercase Git object ID")
    return value


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or _ISO_UTC.fullmatch(value) is None:
        raise PopulationDiscoveryError(f"{field} must use YYYY-MM-DDTHH:MM:SSZ")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_acquisition_policy(plan: Mapping[str, Any]) -> None:
    repetitions = _require_int(
        plan["acquisition_repetitions"],
        "plan.acquisition_repetitions",
        minimum=1,
        maximum=3,
    )
    _require_int(
        plan["index_stabilization_seconds"],
        "plan.index_stabilization_seconds",
    )
    separation = _require_int(
        plan["repeat_separation_seconds"],
        "plan.repeat_separation_seconds",
    )
    if repetitions > 1 and separation < 3600:
        raise PopulationDiscoveryError(
            "repeated acquisitions require at least one hour of server-time separation"
        )


def _validate_completeness(value: object) -> None:
    completeness = _require_mapping(value, "plan.completeness")
    _require_exact_keys(completeness, _COMPLETENESS_KEYS, "plan.completeness")
    if not all(
        _require_bool(completeness[key], f"plan.completeness.{key}")
        for key in _COMPLETENESS_KEYS
    ):
        raise PopulationDiscoveryError("all completeness checks must fail closed")


def _validate_deduplication(value: object) -> None:
    deduplication = _require_mapping(value, "plan.deduplication")
    _require_exact_keys(deduplication, _DEDUP_KEYS, "plan.deduplication")
    if deduplication["primary_identity_fields"] != ["node_id", "id"]:
        raise PopulationDiscoveryError(
            "plan.deduplication.primary_identity_fields must be exactly "
            "['node_id', 'id']"
        )
    if (
        _require_bool(
            deduplication["candidate_only_not_lineage_deduplication"],
            "plan.deduplication.candidate_only_not_lineage_deduplication",
        )
        is not True
    ):
        raise PopulationDiscoveryError(
            "candidate discovery may not claim lineage-independent deduplication"
        )


def _validate_public_visibility(value: object) -> None:
    contract = _require_mapping(value, "plan.public_visibility")
    _require_exact_keys(contract, _PUBLIC_VISIBILITY_KEYS, "plan.public_visibility")
    expected = {
        "required_query_qualifier": "is:public",
        "independent_repository_lookup": True,
        "persist_search_body_before_visibility_verification": False,
        "persist_repository_response_body": False,
    }
    if dict(contract) != expected:
        raise PopulationDiscoveryError(
            "plan.public_visibility must require an independent public-only verification"
        )


def _validate_credential_profile(value: object, *, lane_id: str) -> None:
    profile = _require_mapping(value, "plan.credential_profile")
    _require_exact_keys(profile, _CREDENTIAL_PROFILE_KEYS, "plan.credential_profile")
    profile_id = _require_text(
        profile["profile_id"], "plan.credential_profile.profile_id", maximum=128
    )
    if profile_id != f"{lane_id}-github-public-read-v2":
        raise PopulationDiscoveryError("credential profile is not bound to the lane")
    if _require_token_env_name(profile["token_env_name"]) != "GITHUB_TOKEN":
        raise PopulationDiscoveryError(
            "credential environment is not the locked GitHub token"
        )
    if (
        profile["credential_required"] is not True
        or profile["credential_value_serialized"] is not False
    ):
        raise PopulationDiscoveryError(
            "credential presence is required without serialization"
        )


def _validate_rate_limit_policy(value: object) -> None:
    policy = _require_mapping(value, "plan.rate_limit_policy")
    _require_exact_keys(policy, _RATE_LIMIT_POLICY_KEYS, "plan.rate_limit_policy")
    expected_headers = [
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "x-ratelimit-resource",
        "x-ratelimit-used",
    ]
    if policy["required_header_names"] != expected_headers:
        raise PopulationDiscoveryError(
            "rate-limit policy must require every audit header"
        )
    _require_int(
        policy["minimum_remaining"], "rate_limit_policy.minimum_remaining", minimum=1
    )
    _require_int(
        policy["maximum_wait_seconds"],
        "rate_limit_policy.maximum_wait_seconds",
        minimum=1,
        maximum=60,
    )


def _validate_implementation_lock(value: object) -> None:
    lock = dict(_require_mapping(value, "plan.collector_implementation"))
    _require_exact_keys(lock, _IMPLEMENTATION_KEYS, "plan.collector_implementation")
    _require_git_sha(lock["source_commit"], "collector_implementation.source_commit")
    _require_git_sha(lock["source_tree"], "collector_implementation.source_tree")
    _require_sha256(
        lock["collector_sha256"], "collector_implementation.collector_sha256"
    )
    _require_sha256(lock["compiler_sha256"], "collector_implementation.compiler_sha256")
    _require_bool(
        lock["files_match_commit"], "collector_implementation.files_match_commit"
    )
    claimed = _require_sha256(
        lock.pop("lock_sha256"), "collector_implementation.lock_sha256"
    )
    if claimed != stable_digest(lock):
        raise PopulationDiscoveryError("collector implementation lock digest disagrees")


def _validate_query(
    raw_query: object,
    *,
    index: int,
    source_cutoff: datetime,
    seen_ids: set[str],
) -> None:
    query = _require_mapping(raw_query, f"plan.queries[{index}]")
    _require_exact_keys(query, _QUERY_KEYS, f"plan.queries[{index}]")
    query_id = _require_text(query["query_id"], f"queries[{index}].query_id")
    if _SAFE_ID.fullmatch(query_id) is None or query_id in seen_ids:
        raise PopulationDiscoveryError("query IDs must be unique safe identifiers")
    seen_ids.add(query_id)
    if query["endpoint"] not in SUPPORTED_ENDPOINTS:
        raise PopulationDiscoveryError(f"query {query_id} uses an unsupported endpoint")
    query_text = _require_text(query["query"], f"query {query_id}.query")
    query_tokens = query_text.split()
    if query_tokens.count("is:public") != 1 or any(
        token in {"is:private", "visibility:private"} for token in query_tokens
    ):
        raise PopulationDiscoveryError(
            f"query {query_id} must contain exactly one public-only qualifier"
        )
    if _matched_privacy_patterns(query_text):
        raise PopulationDiscoveryError(
            f"query {query_id} contains credential-shaped text"
        )
    date_qualifier = query["date_qualifier"]
    if date_qualifier not in SUPPORTED_DATE_QUALIFIERS:
        raise PopulationDiscoveryError(
            f"query {query_id} uses an unsupported date qualifier"
        )
    if re.search(rf"(?:^|\s){re.escape(str(date_qualifier))}:", query_text):
        raise PopulationDiscoveryError(
            f"query {query_id} must not embed its date qualifier"
        )
    start = _parse_utc(query["start_utc"], f"query {query_id}.start_utc")
    end = _parse_utc(query["end_utc"], f"query {query_id}.end_utc")
    if start > end or end > source_cutoff:
        raise PopulationDiscoveryError(
            f"query {query_id} must end on or before the selection source cutoff"
        )
    _require_int(
        query["max_results_per_shard"],
        f"query {query_id}.max_results_per_shard",
        minimum=1,
        maximum=900,
    )
    _require_int(
        query["per_page"],
        f"query {query_id}.per_page",
        minimum=1,
        maximum=100,
    )
    sort = query["sort"]
    order = query["order"]
    if sort is not None:
        _require_text(sort, f"query {query_id}.sort", maximum=64)
    if order not in {None, "asc", "desc"}:
        raise PopulationDiscoveryError(f"query {query_id}.order is invalid")
    if sort is None and order is not None:
        raise PopulationDiscoveryError(
            f"query {query_id}.order requires an explicit sort"
        )
    granularity = query["shard_granularity"]
    if granularity not in {"second", "day"}:
        raise PopulationDiscoveryError(f"query {query_id}.shard_granularity is invalid")
    if query["unsplittable_overflow_policy"] != "fail_closed_new_protocol_required":
        raise PopulationDiscoveryError(
            f"query {query_id}.unsplittable_overflow_policy must fail closed"
        )
    if granularity == "day" and (
        start.strftime("%H:%M:%S") != "00:00:00"
        or end.strftime("%H:%M:%S") != "23:59:59"
    ):
        raise PopulationDiscoveryError(
            f"query {query_id} day-granularity bounds must cover whole UTC days"
        )


def validate_query_plan(value: object) -> dict[str, Any]:
    """Validate and normalize a strict V2 candidate-discovery plan."""

    plan = dict(_require_mapping(value, "plan"))
    _require_exact_keys(plan, _PLAN_KEYS, "plan")
    if plan["schema_version"] != SCHEMA_VERSION:
        raise PopulationDiscoveryError("plan.schema_version must be 2")
    plan_id = _require_text(plan["plan_id"], "plan.plan_id", maximum=128)
    if _SAFE_ID.fullmatch(plan_id) is None:
        raise PopulationDiscoveryError("plan.plan_id has an invalid identifier")
    lane_id = _require_text(plan["lane_id"], "plan.lane_id", maximum=128)
    if _SAFE_ID.fullmatch(lane_id) is None:
        raise PopulationDiscoveryError("plan.lane_id has an invalid identifier")
    if (
        _require_bool(plan["candidate_discovery_only"], "plan.candidate_discovery_only")
        is not True
    ):
        raise PopulationDiscoveryError(
            "search output must remain candidate discovery only"
        )
    _require_text(plan["api_version"], "plan.api_version", maximum=32)
    _require_sha256(plan["source_query_plan_sha256"], "plan.source_query_plan_sha256")
    _require_sha256(
        plan["source_sampling_protocol_sha256"],
        "plan.source_sampling_protocol_sha256",
    )
    protocol_id = _require_text(
        plan["sampling_protocol_id"], "plan.sampling_protocol_id", maximum=128
    )
    if _SAFE_ID.fullmatch(protocol_id) is None:
        raise PopulationDiscoveryError("plan.sampling_protocol_id is invalid")
    _require_text(plan["population_scope"], "plan.population_scope")
    _require_text(plan["temporal_scope"], "plan.temporal_scope", maximum=256)
    source_cutoff = _parse_utc(
        plan["selection_source_cutoff_utc"], "plan.selection_source_cutoff_utc"
    )
    _validate_acquisition_policy(plan)
    _validate_completeness(plan["completeness"])
    _validate_deduplication(plan["deduplication"])
    _validate_public_visibility(plan["public_visibility"])
    _validate_credential_profile(plan["credential_profile"], lane_id=lane_id)
    _validate_rate_limit_policy(plan["rate_limit_policy"])
    _require_int(
        plan["max_response_bytes"],
        "plan.max_response_bytes",
        minimum=1024,
        maximum=64 * 1024 * 1024,
    )
    _validate_implementation_lock(plan["collector_implementation"])
    queries = plan["queries"]
    if not isinstance(queries, list) or not queries:
        raise PopulationDiscoveryError("plan.queries must be a non-empty list")
    seen_ids: set[str] = set()
    for index, raw_query in enumerate(queries):
        _validate_query(
            raw_query,
            index=index,
            source_cutoff=source_cutoff,
            seen_ids=seen_ids,
        )
    return plan


def load_query_plan(path: Path) -> dict[str, Any]:
    _reject_symlink_ancestors(path, field="query plan")
    try:
        value = _strict_json_loads(path.read_bytes(), field="query plan")
    except OSError as exc:
        raise PopulationDiscoveryError(f"cannot load query plan {path}: {exc}") from exc
    return validate_query_plan(value)


def current_implementation_lock(*, compiler_path: Path) -> dict[str, Any]:
    """Resolve exact implementation blobs without a manifest self-reference.

    The implementation commit is the latest commit touching either executable
    source file, not the checkout's current HEAD. A later metadata-only commit
    can therefore record the compiled-plan digest without changing that digest.
    """

    collector_path = Path(__file__).resolve()
    repository = collector_path.parents[3]
    paths = (collector_path, compiler_path.resolve())

    def git_output(*arguments: str, scalar: bool = False) -> bytes | None:
        try:
            completed = subprocess.run(  # noqa: S603
                ["git", "-C", str(repository), *arguments],  # noqa: S607
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.rstrip(b"\n") if scalar else completed.stdout

    relative_paths: list[str] = []
    try:
        relative_paths = [path.relative_to(repository).as_posix() for path in paths]
    except ValueError:
        relative_paths = []
    commit_bytes = (
        git_output(
            "log",
            "-1",
            "--format=%H",
            "--",
            *relative_paths,
            scalar=True,
        )
        if relative_paths
        else None
    )
    commit = commit_bytes.decode() if commit_bytes else "0" * 40
    tree_bytes = (
        git_output("rev-parse", f"{commit}^{{tree}}", scalar=True)
        if commit_bytes
        else None
    )
    tree = tree_bytes.decode() if tree_bytes else "0" * 40
    matches_commit = bool(commit_bytes and tree_bytes)
    for path, relative in zip(paths, relative_paths, strict=False):
        try:
            committed = git_output("cat-file", "blob", f"{commit}:{relative}")
            matches_commit = matches_commit and committed == path.read_bytes()
        except (OSError, ValueError):
            matches_commit = False
    lock: dict[str, Any] = {
        "source_commit": commit,
        "source_tree": tree,
        "collector_sha256": _sha256_bytes(collector_path.read_bytes()),
        "compiler_sha256": _sha256_bytes(compiler_path.resolve().read_bytes()),
        "files_match_commit": matches_commit,
    }
    lock["lock_sha256"] = stable_digest(lock)
    return lock


def _verify_live_implementation_lock(normalized: Mapping[str, Any]) -> None:
    compiler_path = Path(__file__).with_name("compile_github_collector_plan_v2.py")
    current = current_implementation_lock(compiler_path=compiler_path)
    if current != normalized["collector_implementation"]:
        raise PopulationDiscoveryError(
            "collector implementation differs from the approved plan lock"
        )
    if current["files_match_commit"] is not True:
        raise PopulationDiscoveryError(
            "live collection requires collector and compiler bytes committed at the locked tree"
        )


def _sanitized_headers(headers: Mapping[str, str]) -> dict[str, str | None]:
    normalized: dict[str, str | None] = {
        str(key).lower(): None if value is None else str(value)
        for key, value in headers.items()
    }
    if _TOKEN_NAMES & set(normalized):
        normalized = {
            key: value for key, value in normalized.items() if key not in _TOKEN_NAMES
        }
    return {
        key: (
            None
            if normalized.get(key) is not None
            and _matched_privacy_patterns(str(normalized[key]))
            else normalized.get(key)
        )
        for key in sorted(_SAFE_HEADER_NAMES)
    }


def _header_privacy_findings(headers: Mapping[str, str]) -> list[dict[str, Any]]:
    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    findings: list[dict[str, Any]] = []
    for name in sorted(_SAFE_HEADER_NAMES):
        value = normalized.get(name)
        if value is None:
            continue
        for pattern_id, occurrence_count in _matched_privacy_patterns(value).items():
            findings.append(
                {
                    "header_name": name,
                    "pattern_id": pattern_id,
                    "occurrence_count": occurrence_count,
                    "value_sha256": _sha256_bytes(value.encode()),
                }
            )
    return findings


def _require_audit_headers(response: FrozenResponse, *, query_id: str) -> None:
    headers = _sanitized_headers(response.headers)
    required = {
        "date",
        "x-github-request-id",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "x-ratelimit-resource",
        "x-ratelimit-used",
    }
    missing = sorted(key for key in required if not headers[key])
    if missing:
        raise PopulationDiscoveryError(
            f"query {query_id} response is missing audit headers {missing}"
        )
    for name in {
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "x-ratelimit-used",
    }:
        raw = headers[name]
        if not isinstance(raw, str) or not raw.isdigit():
            raise PopulationDiscoveryError(
                f"query {query_id} response has invalid rate-limit header {name}"
            )


def _expected_request_url(endpoint_url: str, params: Mapping[str, str]) -> str:
    value = f"{endpoint_url}?{urlencode(params)}" if params else endpoint_url
    if _matched_privacy_patterns(value):
        raise PopulationDiscoveryError("request URL contains credential-shaped text")
    return value


def _require_transport_provenance(
    response: FrozenResponse,
    *,
    expected_request_url: str,
    query_id: str,
) -> None:
    if _matched_privacy_patterns(response.url):
        raise PopulationDiscoveryError(
            f"query {query_id} response URL contains credential-shaped text"
        )
    if response.requested_url != expected_request_url:
        raise PopulationDiscoveryError(
            f"query {query_id} transport did not bind the exact requested URL"
        )
    if response.redirect_chain or response.url != expected_request_url:
        raise PopulationDiscoveryError(
            f"query {query_id} response used an unapproved redirect"
        )
    started = _parse_utc(
        response.request_started_at_utc,
        f"query {query_id}.request_started_at_utc",
    )
    completed = _parse_utc(
        response.request_completed_at_utc,
        f"query {query_id}.request_completed_at_utc",
    )
    if started > completed:
        raise PopulationDiscoveryError(
            f"query {query_id} request completion precedes its start"
        )


def _rate_limit_pacer(
    policy: Mapping[str, Any],
    *,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
) -> Pacer:
    minimum_remaining = int(policy["minimum_remaining"])
    maximum_wait = int(policy["maximum_wait_seconds"])

    def pace(response: FrozenResponse) -> None:
        headers = _sanitized_headers(response.headers)
        remaining = int(str(headers["x-ratelimit-remaining"]))
        if remaining > minimum_remaining:
            return
        reset_at = int(str(headers["x-ratelimit-reset"]))
        wait_seconds = max(0.0, reset_at - now() + 1.0)
        if wait_seconds > maximum_wait:
            raise PopulationDiscoveryError(
                "GitHub rate limit requires a durable checkpoint and later resume"
            )
        if wait_seconds:
            sleep(wait_seconds)

    return pace


def _require_response_identity(
    response: FrozenResponse,
    *,
    query_id: str,
    endpoint: str,
    params: Mapping[str, str],
    api_version: str,
) -> None:
    parsed = urlsplit(response.url)
    expected_request_url = _expected_request_url(f"{API_ORIGIN}/{endpoint}", params)
    _require_transport_provenance(
        response,
        expected_request_url=expected_request_url,
        query_id=query_id,
    )
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.github.com"
        or parsed.path != f"/{endpoint}"
        or parsed.fragment
    ):
        raise PopulationDiscoveryError(
            f"query {query_id} response URL left the locked GitHub endpoint"
        )
    observed_params = parse_qs(parsed.query, keep_blank_values=True)
    expected_params = {key: [value] for key, value in params.items()}
    if observed_params != expected_params:
        raise PopulationDiscoveryError(
            f"query {query_id} response URL does not match the locked request"
        )
    selected_version = _sanitized_headers(response.headers)[
        "x-github-api-version-selected"
    ]
    if selected_version != api_version:
        raise PopulationDiscoveryError(
            f"query {query_id} selected GitHub API version {selected_version!r}; "
            f"expected {api_version!r}"
        )


def _link_targets(value: str, *, query_id: str) -> dict[str, str]:
    targets: dict[str, str] = {}
    for segment in value.split(","):
        match = re.fullmatch(
            r'\s*<([^>]+)>\s*;\s*rel="([^"]+)"(?:\s*;[^,]*)?\s*', segment
        )
        if match is None:
            raise PopulationDiscoveryError(
                f"query {query_id} returned a malformed pagination Link"
            )
        url, relations = match.groups()
        for relation in relations.split():
            if relation in targets:
                raise PopulationDiscoveryError(
                    f"query {query_id} duplicated pagination relation {relation}"
                )
            targets[relation] = url
    return targets


def _require_link_target(
    url: str,
    *,
    query_id: str,
    endpoint: str,
    current_params: Mapping[str, list[str]],
    expected_page: int,
) -> None:
    parsed = urlsplit(url)
    observed = parse_qs(parsed.query, keep_blank_values=True)
    expected = {key: list(values) for key, values in current_params.items()}
    expected["page"] = [str(expected_page)]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.github.com"
        or parsed.path != f"/{endpoint}"
        or parsed.fragment
        or observed != expected
    ):
        raise PopulationDiscoveryError(
            f"query {query_id} pagination Link left the locked request"
        )


def _require_page_link(
    response: FrozenResponse,
    *,
    query_id: str,
    endpoint: str,
    page: int,
    page_count: int,
) -> None:
    link = _sanitized_headers(response.headers)["link"]
    if not link:
        if page_count > 1:
            raise PopulationDiscoveryError(
                f"query {query_id} omitted its pagination Link"
            )
        return
    targets = _link_targets(link, query_id=query_id)
    has_next = "next" in targets
    if (page < page_count) != has_next:
        raise PopulationDiscoveryError(
            f"query {query_id} pagination Link disagrees at page {page}"
        )
    current_params = parse_qs(urlsplit(response.url).query, keep_blank_values=True)
    expected_pages = {
        "first": 1,
        "prev": page - 1,
        "next": page + 1,
        "last": page_count,
    }
    for relation, target in targets.items():
        if (
            relation not in expected_pages
            or not 1 <= expected_pages[relation] <= page_count
        ):
            raise PopulationDiscoveryError(
                f"query {query_id} returned unexpected pagination relation {relation}"
            )
        _require_link_target(
            target,
            query_id=query_id,
            endpoint=endpoint,
            current_params=current_params,
            expected_page=expected_pages[relation],
        )


def github_transport(*, plan: Mapping[str, Any]) -> GitHubLiveTransportV2:
    """Build the exact live transport from the lane-bound environment value."""

    normalized = validate_query_plan(plan)
    profile = normalized["credential_profile"]
    token_env_name = str(profile["token_env_name"])
    token = os.environ.get(token_env_name)
    if not token:
        raise PopulationDiscoveryError(
            "the exact lane-bound GitHub credential is not present"
        )
    return GitHubLiveTransportV2(
        _LIVE_TRANSPORT_SENTINEL,
        api_version=str(normalized["api_version"]),
        profile_id=str(profile["profile_id"]),
        token_env_name=token_env_name,
        token=token,
        max_response_bytes=int(normalized["max_response_bytes"]),
    )


def _qualification_mode(
    normalized: Mapping[str, Any],
    *,
    transport: Transport,
    visibility_transport: Transport,
) -> str:
    search_live = type(transport) is GitHubLiveTransportV2
    visibility_live = type(visibility_transport) is GitHubLiveTransportV2
    if search_live or visibility_live:
        if not (search_live and visibility_live and transport is visibility_transport):
            raise PopulationDiscoveryError(
                "public qualification requires one exact built-in GitHub transport"
            )
        live = transport
        assert isinstance(live, GitHubLiveTransportV2)
        live.verify_environment(normalized)
        _verify_live_implementation_lock(normalized)
        return _PUBLIC_QUALIFIED
    return _DIAGNOSTIC_NONQUALIFYING


def _require_body_bound(
    response: FrozenResponse, *, maximum: int, query_id: str
) -> None:
    if len(response.body) > maximum:
        raise PopulationDiscoveryError(
            f"query {query_id} response exceeds the locked byte limit"
        )


def _parse_search_response(
    response: FrozenResponse, *, query_id: str, max_response_bytes: int
) -> dict[str, Any]:
    _require_body_bound(response, maximum=max_response_bytes, query_id=query_id)
    if response.status != 200:
        raise PopulationDiscoveryError(
            f"query {query_id} returned unexpected HTTP status {response.status}"
        )
    _require_audit_headers(response, query_id=query_id)
    payload = _strict_json_loads(response.body, field=f"query {query_id} response")
    if not isinstance(payload, dict):
        raise PopulationDiscoveryError(f"query {query_id} response must be an object")
    if not {"total_count", "incomplete_results", "items"} <= set(payload):
        raise PopulationDiscoveryError(
            f"query {query_id} response is missing search fields"
        )
    if (
        isinstance(payload["total_count"], bool)
        or not isinstance(payload["total_count"], int)
        or payload["total_count"] < 0
        or not isinstance(payload["incomplete_results"], bool)
        or not isinstance(payload["items"], list)
        or not all(isinstance(item, dict) for item in payload["items"])
    ):
        raise PopulationDiscoveryError(f"query {query_id} response fields are invalid")
    if payload["incomplete_results"]:
        raise PopulationDiscoveryError(
            f"query {query_id} returned incomplete_results=true"
        )
    return payload


def _repository_url(item: Mapping[str, Any], *, endpoint: str, query_id: str) -> str:
    if endpoint == "search/repositories":
        value = item.get("url")
    elif endpoint == "search/code":
        repository = _require_mapping(
            item.get("repository"), f"query {query_id} code item repository"
        )
        value = repository.get("url")
    else:
        value = item.get("repository_url")
    if not isinstance(value, str):
        raise PopulationDiscoveryError(
            f"query {query_id} item lacks an independently resolvable repository URL"
        )
    parsed = urlsplit(value)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.github.com"
        or len(segments) != 3
        or segments[0] != "repos"
        or parsed.query
        or parsed.fragment
        or any(segment in {".", ".."} for segment in segments)
        or _matched_privacy_patterns(value)
    ):
        raise PopulationDiscoveryError(
            f"query {query_id} item has an invalid repository visibility endpoint"
        )
    return value


def _embedded_repository_identity(
    item: Mapping[str, Any], *, endpoint: str, query_id: str
) -> dict[str, object] | None:
    if endpoint == "search/repositories":
        repository = item
    elif endpoint == "search/code":
        repository = _require_mapping(
            item.get("repository"), f"query {query_id} code item repository"
        )
    else:
        return None
    return {
        "node_id": repository.get("node_id"),
        "id": repository.get("id"),
        "full_name": repository.get("full_name"),
    }


def _require_repository_identity_correlation(
    item: Mapping[str, Any],
    *,
    endpoint: str,
    query_id: str,
    visibility_receipt: Mapping[str, Any],
) -> None:
    observed = _embedded_repository_identity(item, endpoint=endpoint, query_id=query_id)
    if observed is None:
        return
    expected = {
        "node_id": visibility_receipt["repository_node_id"],
        "id": visibility_receipt["repository_id"],
        "full_name": visibility_receipt["repository_full_name"],
    }
    if any(observed.get(field) != value for field, value in expected.items()):
        raise PopulationDiscoveryError(
            f"query {query_id} repository identity changed between search and visibility verification"
        )


def _verify_public_repository(
    repository_url: str,
    *,
    transport: Transport,
    api_version: str,
    query_id: str,
    pacer: Pacer,
    max_response_bytes: int,
) -> dict[str, Any]:
    response = transport(repository_url, {})
    _require_body_bound(
        response,
        maximum=max_response_bytes,
        query_id=f"{query_id} repository visibility",
    )
    _require_transport_provenance(
        response,
        expected_request_url=repository_url,
        query_id=f"{query_id} repository visibility",
    )
    if response.status != 200:
        raise PopulationDiscoveryError(
            f"query {query_id} repository visibility lookup returned HTTP {response.status}"
        )
    _require_audit_headers(response, query_id=f"{query_id} repository visibility")
    pacer(response)
    selected_version = _sanitized_headers(response.headers)[
        "x-github-api-version-selected"
    ]
    if selected_version != api_version:
        raise PopulationDiscoveryError(
            f"query {query_id} repository visibility used the wrong API version"
        )
    payload = _require_mapping(
        _strict_json_loads(response.body, field="repository visibility response"),
        "repository visibility response",
    )
    parsed = urlsplit(repository_url)
    expected_full_name = "/".join(
        segment for segment in parsed.path.split("/") if segment
    ).removeprefix("repos/")
    if (
        payload.get("url") != repository_url
        or payload.get("full_name") != expected_full_name
        or payload.get("private") is not False
        or payload.get("visibility") != "public"
    ):
        raise PopulationDiscoveryError(
            f"query {query_id} returned a repository not independently verified public"
        )
    node_id = _require_text(
        payload.get("node_id"), "repository visibility node_id", maximum=256
    )
    repository_id = _require_int(
        payload.get("id"), "repository visibility id", minimum=1
    )
    safe_headers = _sanitized_headers(response.headers)
    receipt: dict[str, Any] = {
        "repository_url": repository_url,
        "repository_full_name": expected_full_name,
        "repository_node_id": node_id,
        "repository_id": repository_id,
        "visibility": "public",
        "private": False,
        "requested_url": response.requested_url,
        "url": response.url,
        "redirect_chain": list(response.redirect_chain),
        "request_started_at_utc": response.request_started_at_utc,
        "request_completed_at_utc": response.request_completed_at_utc,
        "request_url_sha256": _sha256_bytes(repository_url.encode()),
        "status": response.status,
        "headers": safe_headers,
        "headers_sha256": stable_digest(safe_headers),
        "header_privacy_findings": _header_privacy_findings(response.headers),
        "response_body_sha256": _sha256_bytes(response.body),
        "response_body_persisted": False,
    }
    receipt["receipt_sha256"] = stable_digest(receipt)
    return receipt


def _verify_payload_public(
    payload: Mapping[str, Any],
    *,
    endpoint: str,
    query_id: str,
    visibility_transport: Transport,
    api_version: str,
    visibility_receipts: dict[str, dict[str, Any]],
    pacer: Pacer,
    max_response_bytes: int,
) -> None:
    for item in payload["items"]:
        repository_url = _repository_url(item, endpoint=endpoint, query_id=query_id)
        if repository_url not in visibility_receipts:
            visibility_receipts[repository_url] = _verify_public_repository(
                repository_url,
                transport=visibility_transport,
                api_version=api_version,
                query_id=query_id,
                pacer=pacer,
                max_response_bytes=max_response_bytes,
            )
        _require_repository_identity_correlation(
            item,
            endpoint=endpoint,
            query_id=query_id,
            visibility_receipt=visibility_receipts[repository_url],
        )


def _response_record(
    response: FrozenResponse,
    *,
    query_id: str,
    start: datetime,
    end: datetime,
    page: int,
    acquisition_index: int,
    payload: Mapping[str, Any],
    raw_directory: Path,
) -> dict[str, Any]:
    digest = _sha256_bytes(response.body)
    _ensure_private_directory(raw_directory)
    raw_path = raw_directory / f"{digest}.json"
    _write_private(raw_path, response.body, allow_identical=True)
    safe_headers = _sanitized_headers(response.headers)
    return {
        "query_id": query_id,
        "acquisition_index": acquisition_index,
        "shard_start_utc": _format_utc(start),
        "shard_end_utc": _format_utc(end),
        "page": page,
        "requested_url": response.requested_url,
        "url": response.url,
        "redirect_chain": list(response.redirect_chain),
        "request_started_at_utc": response.request_started_at_utc,
        "request_completed_at_utc": response.request_completed_at_utc,
        "request_url_sha256": _sha256_bytes(str(response.requested_url).encode()),
        "status": response.status,
        "headers": safe_headers,
        "headers_sha256": stable_digest(safe_headers),
        "header_privacy_findings": _header_privacy_findings(response.headers),
        "body_sha256": digest,
        "body_path": f"responses/{digest}.json",
        "total_count": payload["total_count"],
        "item_count": len(payload["items"]),
        "incomplete_results": payload["incomplete_results"],
    }


def _rehydrate_response(
    record: Mapping[str, Any], *, raw_directory: Path, max_response_bytes: int
) -> FrozenResponse:
    body = _read_private(
        raw_directory / f"{record['body_sha256']}.json",
        name="checkpoint response body",
        maximum_bytes=max_response_bytes,
    )
    if _sha256_bytes(body) != record["body_sha256"]:
        raise PopulationDiscoveryError("checkpoint response body digest disagrees")
    redirect_chain = record["redirect_chain"]
    if not isinstance(redirect_chain, list) or not all(
        isinstance(item, str) for item in redirect_chain
    ):
        raise PopulationDiscoveryError("checkpoint redirect chain is invalid")
    return FrozenResponse(
        url=str(record["url"]),
        status=int(record["status"]),
        headers=_require_mapping(record["headers"], "checkpoint response headers"),
        body=body,
        requested_url=str(record["requested_url"]),
        redirect_chain=tuple(redirect_chain),
        request_started_at_utc=str(record["request_started_at_utc"]),
        request_completed_at_utc=str(record["request_completed_at_utc"]),
    )


def _checkpoint_document(
    normalized: Mapping[str, Any],
    *,
    acquisition_index: int,
    responses: Sequence[Mapping[str, Any]],
    visibility_receipts: Mapping[str, Mapping[str, Any]],
    status: str,
    qualification_mode: str,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "plan_sha256": stable_digest(normalized),
        "acquisition_index": acquisition_index,
        "qualification_mode": qualification_mode,
        "source_sampling_protocol_sha256": normalized[
            "source_sampling_protocol_sha256"
        ],
        "sampling_protocol_id": normalized["sampling_protocol_id"],
        "responses": sorted(
            (dict(record) for record in responses),
            key=lambda record: record["request_url_sha256"],
        ),
        "repository_visibility_receipts": [
            dict(visibility_receipts[url]) for url in sorted(visibility_receipts)
        ],
        "collector_implementation": dict(normalized["collector_implementation"]),
    }
    checkpoint["checkpoint_sha256"] = stable_digest(checkpoint)
    return checkpoint


def _write_checkpoint(output_directory: Path, checkpoint: Mapping[str, Any]) -> None:
    _replace_private(
        output_directory / "acquisition-checkpoint.json",
        (json.dumps(checkpoint, indent=2, sort_keys=True) + "\n").encode(),
    )


def _load_checkpoint(
    output_directory: Path,
    *,
    normalized: Mapping[str, Any],
    acquisition_index: int,
    require_materialized: bool = False,
    qualification_mode: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    value = _strict_json_loads(
        _read_private(
            output_directory / "acquisition-checkpoint.json",
            name="acquisition checkpoint",
        ),
        field="acquisition checkpoint",
    )
    checkpoint = dict(_require_mapping(value, "acquisition checkpoint"))
    _require_exact_keys(checkpoint, _CHECKPOINT_KEYS, "acquisition checkpoint")
    claimed = _require_sha256(
        checkpoint.pop("checkpoint_sha256"), "checkpoint.checkpoint_sha256"
    )
    if claimed != stable_digest(checkpoint):
        raise PopulationDiscoveryError("acquisition checkpoint digest disagrees")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": stable_digest(normalized),
        "acquisition_index": acquisition_index,
        "qualification_mode": qualification_mode,
        "source_sampling_protocol_sha256": normalized[
            "source_sampling_protocol_sha256"
        ],
        "sampling_protocol_id": normalized["sampling_protocol_id"],
        "collector_implementation": normalized["collector_implementation"],
    }
    if any(
        checkpoint.get(field) != expected_value
        for field, expected_value in expected.items()
    ):
        raise PopulationDiscoveryError(
            "acquisition checkpoint does not match the locked run"
        )
    if checkpoint.get("status") not in {"in_progress", "materialized"}:
        raise PopulationDiscoveryError("acquisition checkpoint status is invalid")
    if require_materialized and checkpoint.get("status") != "materialized":
        raise PopulationDiscoveryError("acquisition checkpoint is not materialized")
    raw_responses = checkpoint.get("responses")
    raw_visibility = checkpoint.get("repository_visibility_receipts")
    if not isinstance(raw_responses, list) or not isinstance(raw_visibility, list):
        raise PopulationDiscoveryError("acquisition checkpoint ledgers are invalid")
    responses: dict[str, dict[str, Any]] = {}
    for raw_response in raw_responses:
        response = dict(_require_mapping(raw_response, "checkpoint response"))
        key = _require_sha256(
            response.get("request_url_sha256"), "checkpoint response request digest"
        )
        if key in responses:
            raise PopulationDiscoveryError("checkpoint contains duplicate requests")
        responses[key] = response
    visibility: dict[str, dict[str, Any]] = {}
    for raw_receipt in raw_visibility:
        receipt = dict(_require_mapping(raw_receipt, "visibility receipt"))
        url = _require_text(receipt.get("repository_url"), "repository URL")
        if url in visibility:
            raise PopulationDiscoveryError(
                "checkpoint contains duplicate visibility receipts"
            )
        visibility[url] = receipt
    return responses, visibility


def _candidate_identity(item: Mapping[str, Any], identity_fields: Sequence[str]) -> str:
    for field in identity_fields:
        value = item.get(field)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value):
            return f"{field}:{value}"
    raise PopulationDiscoveryError(
        f"candidate is missing every declared identity field {list(identity_fields)}"
    )


def _semantic_search_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Drop search-ranking decorations while retaining source/content metadata."""

    return {
        key: value
        for key, value in item.items()
        if key not in {"score", "text_matches"}
    }


def _matched_privacy_patterns(value: str) -> dict[str, int]:
    return {
        pattern_id: len(list(pattern.finditer(value)))
        for pattern_id, pattern in _PRIVACY_PATTERNS.items()
        if pattern.search(value) is not None
    }


def _privacy_pattern_manifest_sha256() -> str:
    return stable_digest(
        {
            "scanner_revision": _PRIVACY_SCANNER_REVISION,
            "patterns": {
                pattern_id: pattern.pattern
                for pattern_id, pattern in sorted(_PRIVACY_PATTERNS.items())
            },
        }
    )


def _safe_path_key(value: object) -> str:
    rendered = str(value)
    if (
        len(rendered) > 96
        or any(character in rendered for character in "\x00\r\n")
        or _matched_privacy_patterns(rendered)
    ):
        return f"key-sha256:{_sha256_bytes(rendered.encode('utf-8'))}"
    return rendered.replace("~", "~0").replace("/", "~1")


def _safe_candidate_reference(identity: str) -> str:
    if _matched_privacy_patterns(identity) or len(identity) > 256:
        return f"sha256:{_sha256_bytes(identity.encode('utf-8'))}"
    return identity


def _scan_json_privacy(
    value: object,
    *,
    path: str,
    matched_surface: str = "value",
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(value, str):
        for pattern_id, occurrence_count in _matched_privacy_patterns(value).items():
            findings.append(
                {
                    "pattern_id": pattern_id,
                    "json_path": path,
                    "matched_surface": matched_surface,
                    "occurrence_count": occurrence_count,
                }
            )
        return findings
    if isinstance(value, Mapping):
        for key, child in value.items():
            safe_key = _safe_path_key(key)
            child_path = f"{path}/{safe_key}"
            findings.extend(
                _scan_json_privacy(
                    str(key), path=child_path, matched_surface="object_key"
                )
            )
            findings.extend(_scan_json_privacy(child, path=child_path))
        return findings
    if isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_scan_json_privacy(child, path=f"{path}/{index}"))
    return findings


def _response_privacy_findings(
    payload: Mapping[str, Any],
    *,
    response_record: Mapping[str, Any],
    identity_fields: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    public_findings: list[dict[str, Any]] = []
    patterns_by_identity: dict[str, set[str]] = {}
    common = {
        "query_id": response_record["query_id"],
        "acquisition_index": response_record["acquisition_index"],
        "response_body_sha256": response_record["body_sha256"],
        "page": response_record["page"],
    }
    for raw_header_finding in response_record["header_privacy_findings"]:
        header_finding = _require_mapping(
            raw_header_finding, "response.header_privacy_finding"
        )
        public_findings.append(
            {
                **common,
                "candidate_reference": None,
                "pattern_id": header_finding["pattern_id"],
                "json_path": f"$/headers/{header_finding['header_name']}",
                "matched_surface": "value",
                "occurrence_count": header_finding["occurrence_count"],
            }
        )
    metadata = {key: value for key, value in payload.items() if key != "items"}
    for finding in _scan_json_privacy(metadata, path="$"):
        public_findings.append({**common, "candidate_reference": None, **finding})
    for index, item in enumerate(payload["items"]):
        identity = _candidate_identity(item, identity_fields)
        item_findings = _scan_json_privacy(item, path=f"$/items/{index}")
        if item_findings:
            patterns_by_identity.setdefault(identity, set()).update(
                finding["pattern_id"] for finding in item_findings
            )
        reference = _safe_candidate_reference(identity)
        public_findings.extend(
            {**common, "candidate_reference": reference, **finding}
            for finding in item_findings
        )
    return public_findings, patterns_by_identity


def _privacy_summary(
    findings: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    affected_references = {
        _safe_candidate_reference(str(row["candidate_identity"]))
        for row in candidate_rows
        if row["privacy_review"]["status"] == "blocked_pending_review"
    }
    affected_references.update(
        str(finding["candidate_reference"])
        for finding in findings
        if finding["candidate_reference"] is not None
    )
    occurrence_count = sum(int(finding["occurrence_count"]) for finding in findings)
    blocked = bool(findings)
    return {
        "scanner_revision": _PRIVACY_SCANNER_REVISION,
        "pattern_manifest_sha256": _privacy_pattern_manifest_sha256(),
        "pattern_ids_checked": sorted(_PRIVACY_PATTERNS),
        "finding_field_count": len(findings),
        "finding_occurrence_count": occurrence_count,
        "affected_candidate_count": len(affected_references),
        "affected_candidate_references": sorted(affected_references),
        "findings": sorted(
            (dict(finding) for finding in findings),
            key=lambda row: (
                row["response_body_sha256"],
                row["json_path"],
                row["pattern_id"],
                str(row["candidate_reference"]),
            ),
        ),
        "matched_values_serialized": False,
        "raw_source_redacted": False,
        "downstream_authoring_export_blocked": blocked,
        "status": "blocked_pending_review" if blocked else "clear",
        "required_resolution": (
            "Review and exclude the affected record or create a sanitized derivative "
            "that binds the immutable raw-source digest and has a new derivative digest."
            if blocked
            else None
        ),
    }


def _server_date(response_record: Mapping[str, Any]) -> datetime:
    raw = _require_mapping(response_record.get("headers"), "response headers").get(
        "date"
    )
    if not isinstance(raw, str):
        raise PopulationDiscoveryError("response receipt is missing its server Date")
    try:
        value = parsedate_to_datetime(raw)
    except (TypeError, ValueError) as exc:
        raise PopulationDiscoveryError("response server Date is invalid") from exc
    if value.tzinfo is None:
        raise PopulationDiscoveryError("response server Date must include a timezone")
    return value.astimezone(UTC)


def _render_query_boundary(value: datetime, *, granularity: str) -> str:
    return value.strftime("%Y-%m-%d") if granularity == "day" else _format_utc(value)


def _split_interval(
    start: datetime, end: datetime, *, granularity: str, query_id: str
) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    if granularity == "day":
        day_count = (end.date() - start.date()).days
        if day_count < 1:
            raise PopulationDiscoveryError(
                f"query {query_id} exceeds the cap in an unsplittable UTC day"
            )
        midpoint_date = start.date() + timedelta(days=day_count // 2)
        left_end = datetime.combine(
            midpoint_date,
            datetime.max.time().replace(microsecond=0),
            tzinfo=UTC,
        )
        return (start, left_end), (left_end + timedelta(seconds=1), end)
    if start >= end:
        raise PopulationDiscoveryError(
            f"query {query_id} exceeds the cap in an unsplittable second"
        )
    span_seconds = int((end - start).total_seconds())
    midpoint = start + timedelta(seconds=span_seconds // 2)
    if midpoint >= end:
        midpoint = start
    return (start, midpoint), (midpoint + timedelta(seconds=1), end)


def _collect_query(
    query: Mapping[str, Any],
    *,
    transport: Transport,
    raw_directory: Path,
    acquisition_index: int,
    identity_fields: Sequence[str],
    api_version: str,
    visibility_transport: Transport,
    visibility_receipts: dict[str, dict[str, Any]],
    pacer: Pacer,
    cached_responses: Mapping[str, Mapping[str, Any]],
    checkpoint_writer: CheckpointWriter,
    max_response_bytes: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, Any]],
    dict[str, set[str]],
]:
    query_id = str(query["query_id"])
    endpoint = str(query["endpoint"])
    base_query = str(query["query"])
    date_qualifier = str(query["date_qualifier"])
    maximum = int(query["max_results_per_shard"])
    per_page = int(query["per_page"])
    granularity = str(query["shard_granularity"])
    response_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    shards: list[dict[str, str]] = []
    privacy_findings: list[dict[str, Any]] = []
    privacy_patterns_by_identity: dict[str, set[str]] = {}

    def record_privacy(
        payload: Mapping[str, Any], response_record: Mapping[str, Any]
    ) -> None:
        findings, patterns_by_identity = _response_privacy_findings(
            payload,
            response_record=response_record,
            identity_fields=identity_fields,
        )
        privacy_findings.extend(findings)
        for identity, pattern_ids in patterns_by_identity.items():
            privacy_patterns_by_identity.setdefault(identity, set()).update(pattern_ids)

    def fetch(
        start: datetime, end: datetime, page: int
    ) -> tuple[FrozenResponse, dict[str, Any]]:
        dated_query = (
            f"{base_query} {date_qualifier}:"
            f"{_render_query_boundary(start, granularity=granularity)}.."
            f"{_render_query_boundary(end, granularity=granularity)}"
        )
        params = {"q": dated_query, "per_page": str(per_page), "page": str(page)}
        if query["sort"] is not None:
            params["sort"] = str(query["sort"])
            params["order"] = str(query["order"])
        request_url = _expected_request_url(f"{API_ORIGIN}/{endpoint}", params)
        request_digest = _sha256_bytes(request_url.encode())
        cached = cached_responses.get(request_digest)
        response = (
            _rehydrate_response(
                cached,
                raw_directory=raw_directory,
                max_response_bytes=max_response_bytes,
            )
            if cached is not None
            else transport(f"{API_ORIGIN}/{endpoint}", params)
        )
        _require_response_identity(
            response,
            query_id=query_id,
            endpoint=endpoint,
            params=params,
            api_version=api_version,
        )
        payload = _parse_search_response(
            response,
            query_id=query_id,
            max_response_bytes=max_response_bytes,
        )
        if cached is None:
            pacer(response)
        return response, payload

    def collect_shard(start: datetime, end: datetime) -> None:
        first_response, first_payload = fetch(start, end, 1)
        _verify_payload_public(
            first_payload,
            endpoint=endpoint,
            query_id=query_id,
            visibility_transport=visibility_transport,
            api_version=api_version,
            visibility_receipts=visibility_receipts,
            pacer=pacer,
            max_response_bytes=max_response_bytes,
        )
        total = int(first_payload["total_count"])
        first_record = _response_record(
            first_response,
            query_id=query_id,
            start=start,
            end=end,
            page=1,
            acquisition_index=acquisition_index,
            payload=first_payload,
            raw_directory=raw_directory,
        )
        checkpoint_writer(first_record, visibility_receipts)
        record_privacy(first_payload, first_record)
        if total > maximum:
            response_records.append(first_record)
            left, right = _split_interval(
                start, end, granularity=granularity, query_id=query_id
            )
            collect_shard(*left)
            collect_shard(*right)
            return

        page_count = max(1, math.ceil(total / per_page))
        _require_page_link(
            first_response,
            query_id=query_id,
            endpoint=endpoint,
            page=1,
            page_count=page_count,
        )
        page_payloads: list[dict[str, Any]] = [first_payload]
        response_records.append(first_record)
        for page in range(2, page_count + 1):
            response, payload = fetch(start, end, page)
            _verify_payload_public(
                payload,
                endpoint=endpoint,
                query_id=query_id,
                visibility_transport=visibility_transport,
                api_version=api_version,
                visibility_receipts=visibility_receipts,
                pacer=pacer,
                max_response_bytes=max_response_bytes,
            )
            if payload["total_count"] != total:
                raise PopulationDiscoveryError(
                    f"query {query_id} changed total_count while paginating"
                )
            _require_page_link(
                response,
                query_id=query_id,
                endpoint=endpoint,
                page=page,
                page_count=page_count,
            )
            page_payloads.append(payload)
            response_record = _response_record(
                response,
                query_id=query_id,
                start=start,
                end=end,
                page=page,
                acquisition_index=acquisition_index,
                payload=payload,
                raw_directory=raw_directory,
            )
            checkpoint_writer(response_record, visibility_receipts)
            record_privacy(payload, response_record)
            response_records.append(response_record)
        observed = sum(len(payload["items"]) for payload in page_payloads)
        if observed != total:
            raise PopulationDiscoveryError(
                f"query {query_id} expected {total} items but retrieved {observed}"
            )
        observed_identities: list[str] = []
        for page_index, payload in enumerate(page_payloads, start=1):
            expected_page_size = (
                per_page
                if page_index < page_count
                else total - per_page * (page_count - 1)
            )
            if len(payload["items"]) != expected_page_size:
                raise PopulationDiscoveryError(
                    f"query {query_id} page {page_index} has "
                    f"{len(payload['items'])} items; expected {expected_page_size}"
                )
            observed_identities.extend(
                _candidate_identity(item, identity_fields) for item in payload["items"]
            )
        if len(set(observed_identities)) != total:
            raise PopulationDiscoveryError(
                f"query {query_id} pagination contains duplicate or missing identities"
            )
        for payload in page_payloads:
            candidates.extend(dict(item) for item in payload["items"])
        shards.append(
            {
                "query_id": query_id,
                "acquisition_index": str(acquisition_index),
                "start_utc": _format_utc(start),
                "end_utc": _format_utc(end),
                "total_count": str(total),
                "page_count": str(page_count),
                "identity_count": str(len(observed_identities)),
                "ordered_identity_sha256": stable_digest(observed_identities),
            }
        )

    collect_shard(
        _parse_utc(query["start_utc"], f"query {query_id}.start_utc"),
        _parse_utc(query["end_utc"], f"query {query_id}.end_utc"),
    )
    return (
        candidates,
        response_records,
        shards,
        privacy_findings,
        privacy_patterns_by_identity,
    )


def _prepare_output_directory(path: Path) -> None:
    _reject_symlink_ancestors(path, field="output directory")
    try:
        if path.exists() and any(path.iterdir()):
            raise PopulationDiscoveryError(f"output directory is not empty: {path}")
    except PopulationDiscoveryError:
        raise
    except OSError as exc:
        raise PopulationDiscoveryError("cannot inspect output directory") from exc
    _ensure_private_directory(path)


def _require_private_directory(path: Path, *, name: str) -> None:
    _reject_symlink_ancestors(path, field=name)
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PopulationDiscoveryError(f"cannot inspect {name}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise PopulationDiscoveryError(f"{name} is not a directory")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PopulationDiscoveryError(
            f"{name} must not be accessible by group or other"
        )


def _private_atomic_staging_directory(target: Path) -> Path:
    _reject_symlink_ancestors(target, field="final output directory")
    try:
        if target.exists() or target.is_symlink():
            raise PopulationDiscoveryError(
                f"final output destination must be absent: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_ancestors(target.parent, field="final output parent")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
        )
        os.chmod(staging, 0o700)
        return staging
    except PopulationDiscoveryError:
        raise
    except OSError as exc:
        raise PopulationDiscoveryError(
            "cannot prepare atomic final-output staging"
        ) from exc


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(staging: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, target_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, target_bytes, 1)
    else:
        raise PopulationDiscoveryError(
            "platform lacks atomic no-replace directory publication"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PopulationDiscoveryError(
            f"final output destination appeared during preparation: {target}"
        )
    raise OSError(error_number, os.strerror(error_number), str(target))


def _final_publication_marker(target: Path) -> Path:
    return target.with_name(f".{target.name}.publication-v2.json")


def _remove_final_publication_marker(marker: Path, *, durable: bool) -> None:
    marker.unlink(missing_ok=True)
    if durable:
        _fsync_directory(marker.parent)


def _load_final_publication_marker(target: Path) -> dict[str, Any] | None:
    marker_path = _final_publication_marker(target)
    if not marker_path.exists() and not marker_path.is_symlink():
        return None
    value = _strict_json_loads(
        _read_private(marker_path, name="final publication marker"),
        field="final publication marker",
    )
    marker = dict(_require_mapping(value, "final publication marker"))
    _require_exact_keys(
        marker, _FINAL_PUBLICATION_MARKER_KEYS, "final publication marker"
    )
    if marker.get("schema_version") != SCHEMA_VERSION:
        raise PopulationDiscoveryError(
            "final publication marker schema version is invalid"
        )
    if marker.get("target_name") != target.name:
        raise PopulationDiscoveryError("final publication marker target is invalid")
    _require_sha256(marker.get("receipt_sha256"), "publication receipt digest")
    staging_name = marker.get("staging_name")
    if (
        not isinstance(staging_name, str)
        or Path(staging_name).name != staging_name
        or not staging_name.startswith(f".{target.name}.staging-")
    ):
        raise PopulationDiscoveryError(
            "final publication marker staging name is invalid"
        )
    return marker


def _finish_private_directory_publication(
    staging: Path,
    target: Path,
    *,
    marker_path: Path,
) -> None:
    try:
        _fsync_directory(staging)
        _rename_directory_no_replace(staging, target)
    except PopulationDiscoveryError:
        _remove_final_publication_marker(marker_path, durable=True)
        raise
    except OSError as exc:
        _remove_final_publication_marker(marker_path, durable=True)
        raise PopulationDiscoveryError(
            "cannot atomically publish final output"
        ) from exc
    try:
        _fsync_directory(target.parent)
    except OSError as exc:
        raise PublicationRecoveryRequired(
            "final output was renamed but publication durability requires recovery"
        ) from exc
    try:
        _remove_final_publication_marker(marker_path, durable=True)
    except OSError:
        # The namespace publication was already fsynced. A marker that survives
        # a crash is harmless and is removed by idempotent recovery.
        pass


def _commit_private_directory(
    staging: Path, target: Path, *, receipt_sha256: str
) -> None:
    _reject_symlink_ancestors(staging, field="final output staging")
    _reject_symlink_ancestors(target, field="final output destination")
    marker_path = _final_publication_marker(target)
    try:
        if (
            target.exists()
            or target.is_symlink()
            or marker_path.exists()
            or marker_path.is_symlink()
        ):
            raise PopulationDiscoveryError(
                "final output destination or publication marker appeared during preparation"
            )
        marker = {
            "schema_version": SCHEMA_VERSION,
            "target_name": target.name,
            "staging_name": staging.name,
            "receipt_sha256": _require_sha256(
                receipt_sha256, "publication receipt digest"
            ),
        }
        _write_private(
            marker_path,
            (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        _fsync_directory(target.parent)
        _finish_private_directory_publication(
            staging,
            target,
            marker_path=marker_path,
        )
    except PopulationDiscoveryError:
        raise
    except OSError as exc:
        raise PopulationDiscoveryError(
            "cannot atomically publish final output"
        ) from exc


def _candidate_rows(
    candidates_by_identity: Mapping[str, Mapping[str, Mapping[str, Any]]],
    query_ids_by_identity: Mapping[str, set[str]],
    privacy_patterns_by_identity: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity in sorted(candidates_by_identity):
        variants = candidates_by_identity[identity]
        if len(variants) != 1:
            raise PopulationDiscoveryError(
                f"candidate {identity} changed across frozen query atoms"
            )
        digest, item = next(iter(variants.items()))
        privacy_pattern_ids = sorted(privacy_patterns_by_identity.get(identity, set()))
        rows.append(
            {
                "candidate_identity": identity,
                "query_ids": sorted(query_ids_by_identity[identity]),
                "item_sha256": digest,
                "observed_item_sha256s": [digest],
                "item": item,
                "privacy_review": {
                    "status": (
                        "blocked_pending_review" if privacy_pattern_ids else "clear"
                    ),
                    "pattern_ids": privacy_pattern_ids,
                    "matched_values_serialized": False,
                    "raw_source_redacted": False,
                    "downstream_authoring_export_allowed": not privacy_pattern_ids,
                },
            }
        )
    return rows


def _run_acquisition(
    normalized: Mapping[str, Any],
    *,
    acquisition_index: int,
    transport: Transport,
    visibility_transport: Transport,
    raw_directory: Path,
    pacer: Pacer,
    cached_responses: Mapping[str, Mapping[str, Any]],
    checkpoint_writer: CheckpointWriter,
    initial_visibility_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    identity_fields = normalized["deduplication"]["primary_identity_fields"]
    responses: list[dict[str, Any]] = []
    shards: list[dict[str, str]] = []
    candidates_by_identity: dict[str, dict[str, dict[str, Any]]] = {}
    query_ids_by_identity: dict[str, set[str]] = {}
    query_snapshots: dict[str, dict[str, str]] = {}
    privacy_findings: list[dict[str, Any]] = []
    privacy_patterns_by_identity: dict[str, set[str]] = {}
    visibility_receipts = {
        url: dict(receipt) for url, receipt in initial_visibility_receipts.items()
    }

    for query in normalized["queries"]:
        query_id = str(query["query_id"])
        (
            items,
            query_responses,
            query_shards,
            query_privacy_findings,
            query_privacy_patterns,
        ) = _collect_query(
            query,
            transport=transport,
            raw_directory=raw_directory,
            acquisition_index=acquisition_index,
            identity_fields=identity_fields,
            api_version=normalized["api_version"],
            visibility_transport=visibility_transport,
            visibility_receipts=visibility_receipts,
            pacer=pacer,
            cached_responses=cached_responses,
            checkpoint_writer=checkpoint_writer,
            max_response_bytes=int(normalized["max_response_bytes"]),
        )
        responses.extend(query_responses)
        shards.extend(query_shards)
        privacy_findings.extend(query_privacy_findings)
        for identity, pattern_ids in query_privacy_patterns.items():
            privacy_patterns_by_identity.setdefault(identity, set()).update(pattern_ids)
        snapshot: dict[str, str] = {}
        for item in items:
            identity = _candidate_identity(item, identity_fields)
            semantic_item = _semantic_search_item(item)
            digest = stable_digest(semantic_item)
            if identity in snapshot:
                detail = (
                    "conflicting content"
                    if snapshot[identity] != digest
                    else "duplicates"
                )
                raise PopulationDiscoveryError(
                    f"query {query_id} returned {detail} for {identity} across shards"
                )
            snapshot[identity] = digest
            candidates_by_identity.setdefault(identity, {})[digest] = semantic_item
            query_ids_by_identity.setdefault(identity, set()).add(query_id)
        query_snapshots[query_id] = dict(sorted(snapshot.items()))

    if not responses:
        raise PopulationDiscoveryError("an acquisition produced no response receipts")
    server_dates = [_server_date(row) for row in responses]
    server_start = min(server_dates)
    server_end = max(server_dates)
    source_cutoff = _parse_utc(
        normalized["selection_source_cutoff_utc"],
        "plan.selection_source_cutoff_utc",
    )
    earliest_allowed = source_cutoff + timedelta(
        seconds=normalized["index_stabilization_seconds"]
    )
    if server_start < earliest_allowed:
        raise PopulationDiscoveryError(
            "GitHub acquisition occurred before the locked index-stabilization boundary"
        )
    candidate_rows = _candidate_rows(
        candidates_by_identity,
        query_ids_by_identity,
        privacy_patterns_by_identity,
    )
    return {
        "responses": responses,
        "shards": shards,
        "candidate_rows": candidate_rows,
        "query_snapshots": query_snapshots,
        "privacy_scan": _privacy_summary(privacy_findings, candidate_rows),
        "repository_visibility_receipts": [
            visibility_receipts[url] for url in sorted(visibility_receipts)
        ],
        "server_start_utc": _format_utc(server_start),
        "server_end_utc": _format_utc(server_end),
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def _query_snapshots_bytes(snapshots: Mapping[str, Any]) -> bytes:
    return (json.dumps(snapshots, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _acquisition_receipt_document(
    normalized: Mapping[str, Any],
    *,
    acquisition_index: int,
    qualification_mode: str,
    server_start_utc: str,
    server_end_utc: str,
    responses: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    candidate_bytes: bytes,
    query_snapshots: Mapping[str, Any],
    privacy_scan: Mapping[str, Any],
    repository_visibility_receipts: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    snapshots_bytes = _query_snapshots_bytes(query_snapshots)
    locked_profile = normalized["credential_profile"]
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_id": f"{normalized['plan_id']}-acquisition-{acquisition_index}",
        "lane_id": normalized["lane_id"],
        "status": (
            "frozen_candidate_discovery_acquisition"
            if qualification_mode == _PUBLIC_QUALIFIED
            else "diagnostic_nonqualifying_acquisition"
        ),
        "candidate_discovery_only": True,
        "eligibility_or_population_claim": False,
        "plan_sha256": stable_digest(normalized),
        "source_query_plan_sha256": normalized["source_query_plan_sha256"],
        "source_sampling_protocol_sha256": normalized[
            "source_sampling_protocol_sha256"
        ],
        "sampling_protocol_id": normalized["sampling_protocol_id"],
        "acquisition_index": acquisition_index,
        "qualification_mode": qualification_mode,
        "server_start_utc": server_start_utc,
        "server_end_utc": server_end_utc,
        "credential_profile": {
            "profile_id": locked_profile["profile_id"],
            "token_env_name": locked_profile["token_env_name"],
            "credential_present": qualification_mode == _PUBLIC_QUALIFIED,
            "credential_value_serialized": False,
        },
        "responses": sorted(
            (dict(row) for row in responses),
            key=lambda row: (
                row["query_id"],
                row["shard_start_utc"],
                row["shard_end_utc"],
                row["page"],
            ),
        ),
        "shards": sorted(
            (dict(row) for row in shards),
            key=lambda row: (row["query_id"], row["start_utc"], row["end_utc"]),
        ),
        "response_count": len(responses),
        "candidate_count": len(candidate_rows),
        "candidates_path": "candidates.jsonl",
        "candidates_sha256": _sha256_bytes(candidate_bytes),
        "query_snapshots_path": "query-snapshots.json",
        "query_snapshots_sha256": _sha256_bytes(snapshots_bytes),
        "privacy_scan": dict(privacy_scan),
        "repository_visibility_receipts": sorted(
            (dict(item) for item in repository_visibility_receipts),
            key=lambda item: item["repository_url"],
        ),
        "repository_visibility_receipt_count": len(repository_visibility_receipts),
        "checkpoint_sha256": checkpoint_sha256,
        "collector_implementation": dict(normalized["collector_implementation"]),
    }
    receipt["receipt_sha256"] = stable_digest(receipt)
    return receipt


def freeze_acquisition(
    plan: Mapping[str, Any],
    *,
    acquisition_index: int,
    transport: Transport,
    visibility_transport: Transport,
    output_directory: Path,
    resume: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Freeze one independently scheduled acquisition for later reconciliation."""

    normalized = validate_query_plan(plan)
    if not 1 <= acquisition_index <= normalized["acquisition_repetitions"]:
        raise PopulationDiscoveryError("acquisition index is outside the locked plan")
    qualification_mode = _qualification_mode(
        normalized,
        transport=transport,
        visibility_transport=visibility_transport,
    )
    if resume:
        if not output_directory.is_dir():
            raise PopulationDiscoveryError(
                "resume requires an existing checkpoint directory"
            )
        cached_responses, visibility_receipts = _load_checkpoint(
            output_directory,
            normalized=normalized,
            acquisition_index=acquisition_index,
            qualification_mode=qualification_mode,
        )
    else:
        _prepare_output_directory(output_directory)
        cached_responses = {}
        visibility_receipts = {}
        _write_checkpoint(
            output_directory,
            _checkpoint_document(
                normalized,
                acquisition_index=acquisition_index,
                responses=[],
                visibility_receipts={},
                status="in_progress",
                qualification_mode=qualification_mode,
            ),
        )

    def checkpoint_writer(
        response: Mapping[str, Any],
        current_visibility: Mapping[str, Mapping[str, Any]],
    ) -> None:
        key = str(response["request_url_sha256"])
        existing = cached_responses.get(key)
        if existing is not None and existing != response:
            raise PopulationDiscoveryError(
                "resumed request disagrees with its checkpoint"
            )
        cached_responses[key] = dict(response)
        _write_checkpoint(
            output_directory,
            _checkpoint_document(
                normalized,
                acquisition_index=acquisition_index,
                responses=list(cached_responses.values()),
                visibility_receipts=current_visibility,
                status="in_progress",
                qualification_mode=qualification_mode,
            ),
        )

    pacer = _rate_limit_pacer(normalized["rate_limit_policy"], sleep=sleep, now=now)
    result = _run_acquisition(
        normalized,
        acquisition_index=acquisition_index,
        transport=transport,
        visibility_transport=visibility_transport,
        raw_directory=output_directory / "responses",
        pacer=pacer,
        cached_responses=cached_responses,
        checkpoint_writer=checkpoint_writer,
        initial_visibility_receipts=visibility_receipts,
    )
    materialized_checkpoint = _checkpoint_document(
        normalized,
        acquisition_index=acquisition_index,
        responses=result["responses"],
        visibility_receipts={
            receipt["repository_url"]: receipt
            for receipt in result["repository_visibility_receipts"]
        },
        status="materialized",
        qualification_mode=qualification_mode,
    )
    _write_checkpoint(output_directory, materialized_checkpoint)
    candidate_bytes = _jsonl_bytes(result["candidate_rows"])
    candidate_path = output_directory / "candidates.jsonl"
    _write_private(candidate_path, candidate_bytes)
    snapshots_bytes = _query_snapshots_bytes(result["query_snapshots"])
    snapshots_path = output_directory / "query-snapshots.json"
    _write_private(snapshots_path, snapshots_bytes)
    receipt = _acquisition_receipt_document(
        normalized,
        acquisition_index=acquisition_index,
        qualification_mode=qualification_mode,
        server_start_utc=result["server_start_utc"],
        server_end_utc=result["server_end_utc"],
        responses=result["responses"],
        shards=result["shards"],
        candidate_rows=result["candidate_rows"],
        candidate_bytes=candidate_bytes,
        query_snapshots=result["query_snapshots"],
        privacy_scan=result["privacy_scan"],
        repository_visibility_receipts=result["repository_visibility_receipts"],
        checkpoint_sha256=materialized_checkpoint["checkpoint_sha256"],
    )
    _write_private(
        output_directory / "acquisition-receipt.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    reloaded = _load_verified_receipt(
        output_directory / "acquisition-receipt.json", name="acquisition receipt"
    )
    _validate_acquisition_receipt(reloaded, normalized=normalized)
    if reloaded != receipt:
        raise PopulationDiscoveryError(
            "reloaded acquisition receipt changed after serialization"
        )
    return reloaded


def _load_verified_receipt(path: Path, *, name: str) -> dict[str, Any]:
    value = _strict_json_loads(_read_private(path, name=name), field=name)
    receipt = dict(_require_mapping(value, name))
    _require_exact_keys(receipt, _ACQUISITION_RECEIPT_KEYS, name)
    claimed = _require_sha256(
        receipt.pop("receipt_sha256", None), f"{name}.receipt_sha256"
    )
    if claimed != stable_digest(receipt):
        raise PopulationDiscoveryError(f"{name} digest does not match")
    receipt["receipt_sha256"] = claimed
    return receipt


def _validate_privacy_scan(value: object, *, name: str) -> dict[str, Any]:
    scan = dict(_require_mapping(value, name))
    _require_exact_keys(scan, _PRIVACY_SCAN_KEYS, name)
    if _scan_json_privacy(scan, path="$"):
        raise PopulationDiscoveryError(f"{name} serialized a matched credential value")
    if scan["scanner_revision"] != _PRIVACY_SCANNER_REVISION:
        raise PopulationDiscoveryError(f"{name} uses an unsupported scanner revision")
    if scan["pattern_manifest_sha256"] != _privacy_pattern_manifest_sha256():
        raise PopulationDiscoveryError(f"{name} pattern manifest digest disagrees")
    if scan["pattern_ids_checked"] != sorted(_PRIVACY_PATTERNS):
        raise PopulationDiscoveryError(f"{name} did not run every governed pattern")
    findings = scan["findings"]
    if not isinstance(findings, list):
        raise PopulationDiscoveryError(f"{name}.findings must be a list")
    seen_findings: set[str] = set()
    for index, raw_finding in enumerate(findings):
        field = f"{name}.findings[{index}]"
        finding = _require_mapping(raw_finding, field)
        _require_exact_keys(finding, _PRIVACY_FINDING_KEYS, field)
        pattern_id = _require_text(
            finding["pattern_id"], f"{field}.pattern_id", maximum=64
        )
        if pattern_id not in _PRIVACY_PATTERNS:
            raise PopulationDiscoveryError(f"{field}.pattern_id is unsupported")
        _require_text(finding["json_path"], f"{field}.json_path", maximum=512)
        if finding["matched_surface"] not in {"value", "object_key"}:
            raise PopulationDiscoveryError(f"{field}.matched_surface is invalid")
        _require_int(
            finding["occurrence_count"], f"{field}.occurrence_count", minimum=1
        )
        _require_text(finding["query_id"], f"{field}.query_id", maximum=128)
        _require_int(
            finding["acquisition_index"], f"{field}.acquisition_index", minimum=1
        )
        _require_sha256(
            finding["response_body_sha256"], f"{field}.response_body_sha256"
        )
        _require_int(finding["page"], f"{field}.page", minimum=1)
        reference = finding["candidate_reference"]
        if reference is not None:
            _require_text(reference, f"{field}.candidate_reference", maximum=320)
        digest = stable_digest(finding)
        if digest in seen_findings:
            raise PopulationDiscoveryError(
                f"{name} contains duplicate privacy findings"
            )
        seen_findings.add(digest)
    finding_count = _require_int(
        scan["finding_field_count"], f"{name}.finding_field_count"
    )
    occurrence_count = _require_int(
        scan["finding_occurrence_count"], f"{name}.finding_occurrence_count"
    )
    if finding_count != len(findings) or occurrence_count != sum(
        int(finding["occurrence_count"]) for finding in findings
    ):
        raise PopulationDiscoveryError(f"{name} finding totals disagree")
    references = scan["affected_candidate_references"]
    if (
        not isinstance(references, list)
        or not all(isinstance(reference, str) and reference for reference in references)
        or references != sorted(set(references))
    ):
        raise PopulationDiscoveryError(f"{name} candidate references are invalid")
    if _require_int(
        scan["affected_candidate_count"], f"{name}.affected_candidate_count"
    ) != len(references):
        raise PopulationDiscoveryError(f"{name} affected-candidate total disagrees")
    blocked = bool(findings)
    if (
        scan["matched_values_serialized"] is not False
        or scan["raw_source_redacted"] is not False
        or scan["downstream_authoring_export_blocked"] is not blocked
        or scan["status"] != ("blocked_pending_review" if blocked else "clear")
        or (blocked and not isinstance(scan["required_resolution"], str))
        or (not blocked and scan["required_resolution"] is not None)
    ):
        raise PopulationDiscoveryError(f"{name} disposition is inconsistent")
    return scan


def _parse_candidate_rows(value: bytes, *, query_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_identities: set[str] = set()
    try:
        lines = value.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise PopulationDiscoveryError("acquisition candidates are not UTF-8") from exc
    for index, line in enumerate(lines):
        if not line:
            raise PopulationDiscoveryError("acquisition candidates contain a blank row")
        row = dict(
            _require_mapping(
                _strict_json_loads(line, field=f"candidate row {index}"),
                f"candidate row {index}",
            )
        )
        _require_exact_keys(row, _CANDIDATE_ROW_KEYS, f"candidate row {index}")
        identity = _require_text(
            row["candidate_identity"],
            f"candidate row {index}.candidate_identity",
            maximum=320,
        )
        if identity in seen_identities:
            raise PopulationDiscoveryError(
                "acquisition candidates contain duplicate identities"
            )
        seen_identities.add(identity)
        memberships = row["query_ids"]
        if (
            not isinstance(memberships, list)
            or not all(isinstance(membership, str) for membership in memberships)
            or memberships != sorted(set(memberships))
            or not memberships
            or not set(memberships) <= query_ids
        ):
            raise PopulationDiscoveryError(
                f"candidate row {index} has invalid query memberships"
            )
        item = _require_mapping(row["item"], f"candidate row {index}.item")
        digest = _require_sha256(
            row["item_sha256"], f"candidate row {index}.item_sha256"
        )
        if digest != stable_digest(_semantic_search_item(item)):
            raise PopulationDiscoveryError(
                f"candidate row {index} item digest disagrees"
            )
        if row["observed_item_sha256s"] != [digest]:
            raise PopulationDiscoveryError(
                f"candidate row {index} observed item digests are invalid"
            )
        review = _require_mapping(
            row["privacy_review"], f"candidate row {index}.privacy_review"
        )
        _require_exact_keys(
            review, _PRIVACY_REVIEW_KEYS, f"candidate row {index}.privacy_review"
        )
        patterns = review["pattern_ids"]
        if (
            not isinstance(patterns, list)
            or not all(isinstance(pattern, str) for pattern in patterns)
            or patterns != sorted(set(patterns))
            or not set(patterns) <= set(_PRIVACY_PATTERNS)
        ):
            raise PopulationDiscoveryError(
                f"candidate row {index} privacy patterns are invalid"
            )
        blocked = bool(patterns)
        if (
            review["status"] != ("blocked_pending_review" if blocked else "clear")
            or review["matched_values_serialized"] is not False
            or review["raw_source_redacted"] is not False
            or review["downstream_authoring_export_allowed"] is not (not blocked)
        ):
            raise PopulationDiscoveryError(
                f"candidate row {index} privacy state is invalid"
            )
        rows.append(row)
    if [str(row["candidate_identity"]) for row in rows] != sorted(seen_identities):
        raise PopulationDiscoveryError(
            "acquisition candidates are not in canonical identity order"
        )
    return rows


def _validate_header_privacy_findings(value: object, *, field: str) -> None:
    if not isinstance(value, list):
        raise PopulationDiscoveryError(f"{field} must be a list")
    expected_keys = {
        "header_name",
        "pattern_id",
        "occurrence_count",
        "value_sha256",
    }
    for index, raw_finding in enumerate(value):
        finding_field = f"{field}[{index}]"
        finding = _require_mapping(raw_finding, finding_field)
        _require_exact_keys(finding, expected_keys, finding_field)
        if finding["header_name"] not in _SAFE_HEADER_NAMES:
            raise PopulationDiscoveryError(f"{finding_field}.header_name is invalid")
        if finding["pattern_id"] not in _PRIVACY_PATTERNS:
            raise PopulationDiscoveryError(f"{finding_field}.pattern_id is invalid")
        _require_int(
            finding["occurrence_count"],
            f"{finding_field}.occurrence_count",
            minimum=1,
        )
        _require_sha256(finding["value_sha256"], f"{finding_field}.value_sha256")


def _validate_visibility_receipt(value: object, *, api_version: str) -> dict[str, Any]:
    receipt = dict(_require_mapping(value, "repository visibility receipt"))
    _require_exact_keys(
        receipt, _VISIBILITY_RECEIPT_KEYS, "repository visibility receipt"
    )
    claimed = _require_sha256(
        receipt.pop("receipt_sha256"), "repository visibility receipt digest"
    )
    if claimed != stable_digest(receipt):
        raise PopulationDiscoveryError("repository visibility receipt digest disagrees")
    if _scan_json_privacy(receipt, path="$"):
        raise PopulationDiscoveryError(
            "repository visibility receipt serialized credential-shaped text"
        )
    repository_url = _require_text(receipt["repository_url"], "repository URL")
    parsed = urlsplit(repository_url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.github.com"
        or len(path_parts) != 3
        or path_parts[0] != "repos"
        or parsed.query
        or parsed.fragment
    ):
        raise PopulationDiscoveryError(
            "repository visibility receipt URL is not a GitHub repository endpoint"
        )
    expected_full_name = "/".join(path_parts[1:])
    if (
        receipt["requested_url"] != repository_url
        or receipt["url"] != repository_url
        or receipt["redirect_chain"] != []
        or receipt["status"] != 200
        or receipt["visibility"] != "public"
        or receipt["private"] is not False
        or receipt["response_body_persisted"] is not False
        or receipt["repository_full_name"] != expected_full_name
    ):
        raise PopulationDiscoveryError(
            "repository visibility receipt is not public-only"
        )
    _require_text(
        receipt["repository_node_id"], "repository visibility node ID", maximum=256
    )
    _require_int(
        receipt["repository_id"], "repository visibility repository ID", minimum=1
    )
    started = _parse_utc(receipt["request_started_at_utc"], "visibility request start")
    completed = _parse_utc(
        receipt["request_completed_at_utc"], "visibility request end"
    )
    if started > completed:
        raise PopulationDiscoveryError("visibility request interval is invalid")
    if receipt["request_url_sha256"] != _sha256_bytes(repository_url.encode()):
        raise PopulationDiscoveryError("visibility request URL digest disagrees")
    headers = _require_mapping(receipt["headers"], "visibility headers")
    if (
        set(headers) != _SAFE_HEADER_NAMES
        or any(
            header_value is not None and not isinstance(header_value, str)
            for header_value in headers.values()
        )
        or receipt["headers_sha256"] != stable_digest(headers)
    ):
        raise PopulationDiscoveryError("visibility header digest disagrees")
    frozen = FrozenResponse(
        url=repository_url,
        status=200,
        headers=headers,
        body=b"",
        requested_url=repository_url,
        redirect_chain=(),
        request_started_at_utc=str(receipt["request_started_at_utc"]),
        request_completed_at_utc=str(receipt["request_completed_at_utc"]),
    )
    _require_transport_provenance(
        frozen,
        expected_request_url=repository_url,
        query_id="repository visibility receipt",
    )
    _require_audit_headers(frozen, query_id="repository visibility receipt")
    if headers.get("x-github-api-version-selected") != api_version:
        raise PopulationDiscoveryError("visibility receipt API version disagrees")
    _validate_header_privacy_findings(
        receipt["header_privacy_findings"], field="visibility header privacy findings"
    )
    _require_sha256(receipt["response_body_sha256"], "visibility response body digest")
    receipt["receipt_sha256"] = claimed
    return receipt


def _validate_acquisition_receipt(  # noqa: C901 - one exact receipt boundary
    receipt: Mapping[str, Any],
    *,
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    index = _require_int(
        receipt.get("acquisition_index"), "acquisition.acquisition_index", minimum=1
    )
    qualification_mode = receipt.get("qualification_mode")
    if qualification_mode not in {
        _PUBLIC_QUALIFIED,
        _DIAGNOSTIC_NONQUALIFYING,
    }:
        raise PopulationDiscoveryError(
            "acquisition receipt qualification mode is invalid"
        )
    expected_values = {
        "schema_version": SCHEMA_VERSION,
        "receipt_id": f"{normalized['plan_id']}-acquisition-{index}",
        "lane_id": normalized["lane_id"],
        "status": (
            "frozen_candidate_discovery_acquisition"
            if qualification_mode == _PUBLIC_QUALIFIED
            else "diagnostic_nonqualifying_acquisition"
        ),
        "candidate_discovery_only": True,
        "eligibility_or_population_claim": False,
        "plan_sha256": stable_digest(normalized),
        "source_query_plan_sha256": normalized["source_query_plan_sha256"],
        "source_sampling_protocol_sha256": normalized[
            "source_sampling_protocol_sha256"
        ],
        "sampling_protocol_id": normalized["sampling_protocol_id"],
        "qualification_mode": qualification_mode,
        "collector_implementation": normalized["collector_implementation"],
    }
    for field, expected in expected_values.items():
        if receipt.get(field) != expected:
            raise PopulationDiscoveryError(f"acquisition receipt has invalid {field}")
    profile = _require_mapping(receipt.get("credential_profile"), "credential_profile")
    _require_exact_keys(
        profile,
        {
            "profile_id",
            "token_env_name",
            "credential_present",
            "credential_value_serialized",
        },
        "credential_profile",
    )
    if (
        profile["profile_id"] != normalized["credential_profile"]["profile_id"]
        or _require_token_env_name(profile["token_env_name"])
        != normalized["credential_profile"]["token_env_name"]
        or profile["credential_present"]
        is not (qualification_mode == _PUBLIC_QUALIFIED)
        or profile["credential_value_serialized"] is not False
    ):
        raise PopulationDiscoveryError(
            "acquisition receipt credential profile is invalid"
        )
    responses = receipt.get("responses")
    shards = receipt.get("shards")
    if not isinstance(responses, list) or not responses:
        raise PopulationDiscoveryError("acquisition receipt must contain responses")
    if not isinstance(shards, list) or not shards:
        raise PopulationDiscoveryError(
            "acquisition receipt must contain terminal shards"
        )
    if _require_int(receipt.get("response_count"), "response_count") != len(responses):
        raise PopulationDiscoveryError("acquisition response total disagrees")
    query_ids = {str(query["query_id"]) for query in normalized["queries"]}
    server_dates: list[datetime] = []
    for response_index, raw_response in enumerate(responses):
        field = f"responses[{response_index}]"
        response = _require_mapping(raw_response, field)
        _require_exact_keys(response, _RESPONSE_RECORD_KEYS, field)
        response_query_id = _require_text(
            response["query_id"], f"{field}.query_id", maximum=128
        )
        if response_query_id not in query_ids or response["acquisition_index"] != index:
            raise PopulationDiscoveryError(f"{field} identity is invalid")
        _require_sha256(response["body_sha256"], f"{field}.body_sha256")
        _require_sha256(response["request_url_sha256"], f"{field}.request_url_sha256")
        _require_sha256(response["headers_sha256"], f"{field}.headers_sha256")
        if response["headers_sha256"] != stable_digest(response["headers"]):
            raise PopulationDiscoveryError(f"{field} header digest disagrees")
        _validate_header_privacy_findings(
            response["header_privacy_findings"],
            field=f"{field}.header_privacy_findings",
        )
        if (
            response["requested_url"] != response["url"]
            or response["redirect_chain"] != []
            or response["request_url_sha256"]
            != _sha256_bytes(str(response["requested_url"]).encode())
        ):
            raise PopulationDiscoveryError(f"{field} redirect provenance is invalid")
        started = _parse_utc(
            response["request_started_at_utc"], f"{field}.request_started_at_utc"
        )
        completed = _parse_utc(
            response["request_completed_at_utc"], f"{field}.request_completed_at_utc"
        )
        if started > completed:
            raise PopulationDiscoveryError(f"{field} request interval is invalid")
        if response["body_path"] != f"responses/{response['body_sha256']}.json":
            raise PopulationDiscoveryError(
                f"{field}.body_path is not content addressed"
            )
        if response["status"] != 200 or response["incomplete_results"] is not False:
            raise PopulationDiscoveryError(
                f"{field} is not a complete successful response"
            )
        _require_int(response["page"], f"{field}.page", minimum=1)
        _require_int(response["total_count"], f"{field}.total_count")
        _require_int(response["item_count"], f"{field}.item_count")
        server_dates.append(_server_date(response))
    for shard_index, raw_shard in enumerate(shards):
        field = f"shards[{shard_index}]"
        shard = _require_mapping(raw_shard, field)
        _require_exact_keys(shard, _SHARD_RECORD_KEYS, field)
        shard_query_id = _require_text(
            shard["query_id"], f"{field}.query_id", maximum=128
        )
        if shard_query_id not in query_ids or shard["acquisition_index"] != str(index):
            raise PopulationDiscoveryError(f"{field} identity is invalid")
        _parse_utc(shard["start_utc"], f"{field}.start_utc")
        _parse_utc(shard["end_utc"], f"{field}.end_utc")
        for numeric in ("total_count", "page_count"):
            raw_numeric = shard[numeric]
            if not isinstance(raw_numeric, str) or not raw_numeric.isdigit():
                raise PopulationDiscoveryError(f"{field}.{numeric} is invalid")
            parsed = int(raw_numeric)
            if parsed < (1 if numeric == "page_count" else 0):
                raise PopulationDiscoveryError(f"{field}.{numeric} is invalid")
        identity_count = shard["identity_count"]
        if not isinstance(identity_count, str) or not identity_count.isdigit():
            raise PopulationDiscoveryError(f"{field}.identity_count is invalid")
        if identity_count != shard["total_count"]:
            raise PopulationDiscoveryError(f"{field} identity total disagrees")
        _require_sha256(
            shard["ordered_identity_sha256"], f"{field}.ordered_identity_sha256"
        )
    start = _parse_utc(receipt.get("server_start_utc"), "acquisition.server_start_utc")
    end = _parse_utc(receipt.get("server_end_utc"), "acquisition.server_end_utc")
    if start != min(server_dates) or end != max(server_dates):
        raise PopulationDiscoveryError(
            "acquisition server window disagrees with responses"
        )
    source_cutoff = _parse_utc(
        normalized["selection_source_cutoff_utc"], "selection_source_cutoff_utc"
    )
    if start < source_cutoff + timedelta(
        seconds=normalized["index_stabilization_seconds"]
    ):
        raise PopulationDiscoveryError(
            "acquisition predates the stabilization boundary"
        )
    visibility = receipt.get("repository_visibility_receipts")
    if not isinstance(visibility, list):
        raise PopulationDiscoveryError(
            "repository visibility receipt ledger is invalid"
        )
    validated_visibility = [
        _validate_visibility_receipt(item, api_version=normalized["api_version"])
        for item in visibility
    ]
    if _require_int(
        receipt.get("repository_visibility_receipt_count"),
        "repository_visibility_receipt_count",
    ) != len(validated_visibility) or len(
        {item["repository_url"] for item in validated_visibility}
    ) != len(validated_visibility):
        raise PopulationDiscoveryError("repository visibility receipt total disagrees")
    _require_sha256(receipt.get("checkpoint_sha256"), "checkpoint_sha256")
    return _validate_privacy_scan(receipt.get("privacy_scan"), name="privacy_scan")


def _validate_query_snapshots(
    snapshots: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    query_ids: set[str],
) -> None:
    if set(snapshots) != query_ids:
        raise PopulationDiscoveryError("acquisition snapshots do not cover every query")
    memberships: dict[str, set[str]] = {}
    candidate_digests = {
        str(row["candidate_identity"]): str(row["item_sha256"])
        for row in candidate_rows
    }
    for query_id, raw_snapshot in snapshots.items():
        snapshot = _require_mapping(raw_snapshot, f"snapshot {query_id}")
        for identity, raw_digest in snapshot.items():
            if not isinstance(identity, str) or identity not in candidate_digests:
                raise PopulationDiscoveryError(
                    "snapshot references an unknown candidate"
                )
            digest = _require_sha256(raw_digest, f"snapshot {query_id} digest")
            if digest != candidate_digests[identity]:
                raise PopulationDiscoveryError("snapshot candidate digest disagrees")
            memberships.setdefault(identity, set()).add(query_id)
    for row in candidate_rows:
        identity = str(row["candidate_identity"])
        if memberships.get(identity) != set(row["query_ids"]):
            raise PopulationDiscoveryError("snapshot query memberships disagree")


def _validate_shard_partitions(
    receipt: Mapping[str, Any], normalized: Mapping[str, Any]
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    terminal: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    shards_by_query: dict[str, list[Mapping[str, Any]]] = {}
    for raw_shard in receipt["shards"]:
        shard = _require_mapping(raw_shard, "shard")
        key = (str(shard["query_id"]), str(shard["start_utc"]), str(shard["end_utc"]))
        if key in terminal:
            raise PopulationDiscoveryError("terminal shard ledger contains duplicates")
        terminal[key] = shard
        shards_by_query.setdefault(key[0], []).append(shard)
    for query in normalized["queries"]:
        query_id = str(query["query_id"])
        shards = sorted(
            shards_by_query.get(query_id, []), key=lambda row: row["start_utc"]
        )
        if not shards:
            raise PopulationDiscoveryError(f"query {query_id} has no terminal shards")
        expected_start = _parse_utc(query["start_utc"], f"query {query_id}.start_utc")
        query_end = _parse_utc(query["end_utc"], f"query {query_id}.end_utc")
        for shard in shards:
            start = _parse_utc(shard["start_utc"], "shard.start_utc")
            end = _parse_utc(shard["end_utc"], "shard.end_utc")
            if start != expected_start or end < start or end > query_end:
                raise PopulationDiscoveryError(
                    f"query {query_id} terminal shards are not a closed partition"
                )
            total = int(shard["total_count"])
            expected_pages = max(1, math.ceil(total / int(query["per_page"])))
            if (
                total > int(query["max_results_per_shard"])
                or int(shard["page_count"]) != expected_pages
            ):
                raise PopulationDiscoveryError(
                    f"query {query_id} terminal shard is invalid"
                )
            expected_start = end + timedelta(seconds=1)
        if expected_start != query_end + timedelta(seconds=1):
            raise PopulationDiscoveryError(
                f"query {query_id} terminal shards do not cover the locked interval"
            )
    return terminal


def _validate_exact_split_topology(
    receipt: Mapping[str, Any],
    normalized: Mapping[str, Any],
    *,
    terminal: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> None:
    responses: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for raw_response in receipt["responses"]:
        response = _require_mapping(raw_response, "split topology response")
        key = (
            str(response["query_id"]),
            str(response["shard_start_utc"]),
            str(response["shard_end_utc"]),
            int(response["page"]),
        )
        if key in responses:
            raise PopulationDiscoveryError(
                "split topology contains duplicate response coordinates"
            )
        responses[key] = response

    consumed_responses: set[tuple[str, str, str, int]] = set()
    consumed_terminal: set[tuple[str, str, str]] = set()

    for query in normalized["queries"]:
        query_id = str(query["query_id"])
        maximum = int(query["max_results_per_shard"])
        granularity = str(query["shard_granularity"])

        def visit(
            start: datetime,
            end: datetime,
            *,
            current_query_id: str = query_id,
            current_maximum: int = maximum,
            current_granularity: str = granularity,
        ) -> None:
            interval = (current_query_id, _format_utc(start), _format_utc(end))
            first_key = (*interval, 1)
            first = responses.get(first_key)
            if first is None:
                raise PopulationDiscoveryError(
                    f"query {current_query_id} split topology is missing its page-one probe"
                )
            total = int(first["total_count"])
            consumed_responses.add(first_key)
            if total > current_maximum:
                if interval in terminal:
                    raise PopulationDiscoveryError(
                        f"query {current_query_id} over-cap probe was marked terminal"
                    )
                left, right = _split_interval(
                    start,
                    end,
                    granularity=current_granularity,
                    query_id=current_query_id,
                )
                visit(*left)
                visit(*right)
                return
            shard = terminal.get(interval)
            if shard is None:
                raise PopulationDiscoveryError(
                    f"query {current_query_id} split topology omitted a terminal shard"
                )
            consumed_terminal.add(interval)
            for page in range(2, int(shard["page_count"]) + 1):
                page_key = (*interval, page)
                if page_key not in responses:
                    raise PopulationDiscoveryError(
                        f"query {current_query_id} split topology omitted page {page}"
                    )
                consumed_responses.add(page_key)

        visit(
            _parse_utc(query["start_utc"], f"query {query_id}.start_utc"),
            _parse_utc(query["end_utc"], f"query {query_id}.end_utc"),
        )

    if set(responses) != consumed_responses:
        raise PopulationDiscoveryError(
            "response ledger contains a probe outside the exact recursive split tree"
        )
    if set(terminal) != consumed_terminal:
        raise PopulationDiscoveryError(
            "terminal shard ledger differs from the exact recursive split tree"
        )


def _response_request_params(
    response: Mapping[str, Any], query: Mapping[str, Any]
) -> dict[str, str]:
    start = _parse_utc(response["shard_start_utc"], "response.shard_start_utc")
    end = _parse_utc(response["shard_end_utc"], "response.shard_end_utc")
    query_start = _parse_utc(query["start_utc"], "query.start_utc")
    query_end = _parse_utc(query["end_utc"], "query.end_utc")
    if start < query_start or end > query_end or start > end:
        raise PopulationDiscoveryError("response shard left the locked query interval")
    params = {
        "q": (
            f"{query['query']} {query['date_qualifier']}:"
            f"{_render_query_boundary(start, granularity=str(query['shard_granularity']))}.."
            f"{_render_query_boundary(end, granularity=str(query['shard_granularity']))}"
        ),
        "per_page": str(query["per_page"]),
        "page": str(response["page"]),
    }
    if query["sort"] is not None:
        params["sort"] = str(query["sort"])
        params["order"] = str(query["order"])
    return params


def _validate_raw_response(
    directory: Path,
    response: Mapping[str, Any],
    *,
    query: Mapping[str, Any],
    identity_fields: Sequence[str],
    api_version: str,
    max_response_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, set[str]]]:
    body_path = _locked_artifact_path(
        directory,
        response["body_path"],
        expected=f"responses/{response['body_sha256']}.json",
        field="response.body_path",
    )
    body = _read_private(
        body_path,
        name="raw acquisition response",
        maximum_bytes=max_response_bytes,
    )
    if _sha256_bytes(body) != response["body_sha256"]:
        raise PopulationDiscoveryError("raw acquisition response digest mismatch")
    frozen = FrozenResponse(
        url=str(response["url"]),
        status=int(response["status"]),
        headers=_require_mapping(response["headers"], "response.headers"),
        body=body,
        requested_url=str(response["requested_url"]),
        redirect_chain=tuple(response["redirect_chain"]),
        request_started_at_utc=str(response["request_started_at_utc"]),
        request_completed_at_utc=str(response["request_completed_at_utc"]),
    )
    params = _response_request_params(response, query)
    _require_response_identity(
        frozen,
        query_id=str(query["query_id"]),
        endpoint=str(query["endpoint"]),
        params=params,
        api_version=api_version,
    )
    payload = _parse_search_response(
        frozen,
        query_id=str(query["query_id"]),
        max_response_bytes=int(max_response_bytes),
    )
    if (
        payload["total_count"] != response["total_count"]
        or len(payload["items"]) != response["item_count"]
        or payload["incomplete_results"] != response["incomplete_results"]
    ):
        raise PopulationDiscoveryError("raw response fields disagree with its receipt")
    findings, patterns = _response_privacy_findings(
        payload,
        response_record=response,
        identity_fields=identity_fields,
    )
    return payload, findings, patterns


def _validate_response_ledger(
    directory: Path,
    receipt: Mapping[str, Any],
    *,
    normalized: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Any],
) -> dict[str, Any]:
    queries = {str(query["query_id"]): query for query in normalized["queries"]}
    terminal = _validate_shard_partitions(receipt, normalized)
    pages_by_terminal: dict[tuple[str, str, str], set[int]] = {}
    identities_by_terminal: dict[tuple[str, str, str], dict[int, list[str]]] = {}
    snapshots_from_raw: dict[str, dict[str, str]] = {
        query_id: {} for query_id in queries
    }
    findings: list[dict[str, Any]] = []
    seen_responses: set[tuple[str, str, str, int]] = set()
    visibility_by_url = {
        str(item["repository_url"]): item
        for item in receipt["repository_visibility_receipts"]
    }
    visibility_urls = set(visibility_by_url)
    observed_repository_urls: set[str] = set()
    for raw_response in receipt["responses"]:
        response = _require_mapping(raw_response, "response")
        key = (
            str(response["query_id"]),
            str(response["shard_start_utc"]),
            str(response["shard_end_utc"]),
        )
        response_identity = (*key, int(response["page"]))
        if response_identity in seen_responses:
            raise PopulationDiscoveryError("response ledger contains duplicates")
        seen_responses.add(response_identity)
        query = queries.get(key[0])
        if query is None:
            raise PopulationDiscoveryError("response references an unknown query")
        payload, response_findings, _ = _validate_raw_response(
            directory,
            response,
            query=query,
            identity_fields=normalized["deduplication"]["primary_identity_fields"],
            api_version=str(normalized["api_version"]),
            max_response_bytes=int(normalized["max_response_bytes"]),
        )
        findings.extend(response_findings)
        for item in payload["items"]:
            repository_url = _repository_url(
                item, endpoint=str(query["endpoint"]), query_id=key[0]
            )
            observed_repository_urls.add(repository_url)
            if repository_url not in visibility_urls:
                raise PopulationDiscoveryError(
                    "raw search item lacks an independent public visibility receipt"
                )
            _require_repository_identity_correlation(
                item,
                endpoint=str(query["endpoint"]),
                query_id=key[0],
                visibility_receipt=visibility_by_url[repository_url],
            )
        if key not in terminal:
            if int(response["page"]) != 1 or int(payload["total_count"]) <= int(
                query["max_results_per_shard"]
            ):
                raise PopulationDiscoveryError(
                    "nonterminal response is not a split probe"
                )
            continue
        shard = terminal[key]
        if int(payload["total_count"]) != int(shard["total_count"]):
            raise PopulationDiscoveryError(
                "terminal response total disagrees with its shard"
            )
        page_count = int(shard["page_count"])
        if not 1 <= int(response["page"]) <= page_count:
            raise PopulationDiscoveryError(
                "terminal response page is outside its shard"
            )
        _require_page_link(
            FrozenResponse(
                url=str(response["url"]),
                status=int(response["status"]),
                headers=_require_mapping(response["headers"], "response.headers"),
                body=b"",
            ),
            query_id=key[0],
            endpoint=str(query["endpoint"]),
            page=int(response["page"]),
            page_count=page_count,
        )
        pages_by_terminal.setdefault(key, set()).add(int(response["page"]))
        expected_page_size = (
            int(query["per_page"])
            if int(response["page"]) < page_count
            else int(shard["total_count"]) - int(query["per_page"]) * (page_count - 1)
        )
        if len(payload["items"]) != expected_page_size:
            raise PopulationDiscoveryError(
                "terminal response page size disagrees with the locked pagination"
            )
        page_identities = [
            _candidate_identity(
                item, normalized["deduplication"]["primary_identity_fields"]
            )
            for item in payload["items"]
        ]
        identities_by_terminal.setdefault(key, {})[int(response["page"])] = (
            page_identities
        )
        snapshot = snapshots_from_raw[key[0]]
        for item, identity in zip(payload["items"], page_identities, strict=True):
            if identity in snapshot:
                raise PopulationDiscoveryError(
                    f"query {key[0]} raw terminal shards contain a duplicate identity"
                )
            snapshot[identity] = stable_digest(_semantic_search_item(item))
    _validate_exact_split_topology(receipt, normalized, terminal=terminal)
    for key, shard in terminal.items():
        expected_pages = set(range(1, int(shard["page_count"]) + 1))
        if pages_by_terminal.get(key) != expected_pages:
            raise PopulationDiscoveryError("terminal shard page ledger is incomplete")
        ordered_identities = [
            identity
            for page in range(1, int(shard["page_count"]) + 1)
            for identity in identities_by_terminal.get(key, {}).get(page, [])
        ]
        if (
            len(ordered_identities) != int(shard["total_count"])
            or len(set(ordered_identities)) != len(ordered_identities)
            or str(shard["identity_count"]) != str(len(ordered_identities))
            or shard["ordered_identity_sha256"] != stable_digest(ordered_identities)
        ):
            raise PopulationDiscoveryError(
                "terminal shard ordered identity ledger does not recompute"
            )
    if observed_repository_urls != visibility_urls:
        raise PopulationDiscoveryError(
            "repository visibility receipts do not exactly match raw terminal records"
        )
    if snapshots_from_raw != snapshots:
        raise PopulationDiscoveryError(
            "raw terminal responses disagree with query snapshots"
        )
    return _privacy_summary(findings, candidate_rows)


def _reload_final_receipt(  # noqa: C901 - one complete final reconciliation boundary
    output_directory: Path, *, normalized: Mapping[str, Any]
) -> dict[str, Any]:
    value = _strict_json_loads(
        _read_private(output_directory / "query-receipt.json", name="final receipt"),
        field="final receipt",
    )
    receipt = dict(_require_mapping(value, "final receipt"))
    _require_exact_keys(receipt, _FINAL_RECEIPT_KEYS, "final receipt")
    claimed = _require_sha256(
        receipt.pop("receipt_sha256"), "final receipt.receipt_sha256"
    )
    if claimed != stable_digest(receipt):
        raise PopulationDiscoveryError("reloaded final receipt digest disagrees")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "receipt_id": f"{normalized['plan_id']}-receipt",
        "lane_id": normalized["lane_id"],
        "candidate_discovery_only": True,
        "eligibility_or_population_claim": False,
        "public_only_repository_visibility_verified": True,
        "plan_sha256": stable_digest(normalized),
        "source_query_plan_sha256": normalized["source_query_plan_sha256"],
        "source_sampling_protocol_sha256": normalized[
            "source_sampling_protocol_sha256"
        ],
        "sampling_protocol_id": normalized["sampling_protocol_id"],
        "qualification_mode": _PUBLIC_QUALIFIED,
        "collector_implementation": normalized["collector_implementation"],
        "public_visibility": normalized["public_visibility"],
        "api_origin": API_ORIGIN,
        "api_version": normalized["api_version"],
        "temporal_scope": normalized["temporal_scope"],
        "selection_source_cutoff_utc": normalized["selection_source_cutoff_utc"],
        "acquisition_repetitions": normalized["acquisition_repetitions"],
        "index_stabilization_seconds": normalized["index_stabilization_seconds"],
        "repeat_separation_seconds": normalized["repeat_separation_seconds"],
        "query_count": len(normalized["queries"]),
    }
    if any(
        receipt.get(field) != expected_value
        for field, expected_value in expected.items()
    ):
        raise PopulationDiscoveryError("reloaded final receipt identity disagrees")
    if receipt.get("limitations") != list(_FINAL_LIMITATIONS):
        raise PopulationDiscoveryError("reloaded final limitations disagree")
    responses = receipt.get("responses")
    visibility = receipt.get("repository_visibility_receipts")
    shards = receipt.get("shards")
    if (
        not isinstance(responses, list)
        or not isinstance(visibility, list)
        or not isinstance(shards, list)
    ):
        raise PopulationDiscoveryError("reloaded final receipt ledgers are invalid")
    if _require_int(receipt.get("response_count"), "final response count") != len(
        responses
    ):
        raise PopulationDiscoveryError("reloaded final response count disagrees")
    if _require_int(
        receipt.get("repository_visibility_receipt_count"),
        "final visibility receipt count",
    ) != len(visibility):
        raise PopulationDiscoveryError("reloaded final visibility count disagrees")
    for raw_response in responses:
        response = _require_mapping(raw_response, "final response")
        _require_exact_keys(response, _RESPONSE_RECORD_KEYS, "final response")
        body = _read_private(
            _locked_artifact_path(
                output_directory,
                response.get("body_path"),
                expected=f"responses/{response.get('body_sha256')}.json",
                field="final response.body_path",
            ),
            name="final raw response",
            maximum_bytes=int(normalized["max_response_bytes"]),
        )
        if _sha256_bytes(body) != response.get("body_sha256"):
            raise PopulationDiscoveryError("reloaded final response digest disagrees")
    visibility_by_acquisition: dict[int, list[dict[str, Any]]] = {}
    for index, raw_entry in enumerate(visibility):
        entry = _require_mapping(raw_entry, f"final visibility[{index}]")
        _require_exact_keys(
            entry, {"acquisition_index", "receipt"}, f"final visibility[{index}]"
        )
        acquisition_index = _require_int(
            entry["acquisition_index"],
            f"final visibility[{index}].acquisition_index",
            minimum=1,
        )
        validated_visibility = _validate_visibility_receipt(
            entry["receipt"], api_version=str(normalized["api_version"])
        )
        visibility_by_acquisition.setdefault(acquisition_index, []).append(
            validated_visibility
        )
    candidate_bytes = _read_private(
        _locked_artifact_path(
            output_directory,
            receipt.get("candidates_path"),
            expected="candidates.jsonl",
            field="final candidates_path",
        ),
        name="final candidates",
    )
    candidate_rows = _parse_candidate_rows(
        candidate_bytes,
        query_ids={str(query["query_id"]) for query in normalized["queries"]},
    )
    if _sha256_bytes(candidate_bytes) != receipt.get("candidates_sha256") or len(
        candidate_rows
    ) != receipt.get("candidate_count"):
        raise PopulationDiscoveryError("reloaded final candidates disagree")

    expected_profile = {
        "profile_id": normalized["credential_profile"]["profile_id"],
        "token_env_name": normalized["credential_profile"]["token_env_name"],
        "credential_present": True,
        "credential_value_serialized": False,
    }
    if receipt.get("credential_profile") != expected_profile:
        raise PopulationDiscoveryError("final credential profile disagrees")

    acquisition_receipts = receipt.get("acquisition_receipts")
    windows = receipt.get("acquisition_windows")
    if not isinstance(acquisition_receipts, list) or not isinstance(windows, list):
        raise PopulationDiscoveryError("final acquisition ledgers are invalid")
    expected_indices = list(range(1, int(normalized["acquisition_repetitions"]) + 1))
    receipt_indices: list[int] = []
    claimed_acquisition_receipts: dict[int, str] = {}
    for index, raw_entry in enumerate(acquisition_receipts):
        field = f"acquisition_receipts[{index}]"
        entry = _require_mapping(raw_entry, field)
        _require_exact_keys(entry, {"acquisition_index", "receipt_sha256"}, field)
        acquisition_index = _require_int(
            entry["acquisition_index"], f"{field}.acquisition_index", minimum=1
        )
        receipt_indices.append(acquisition_index)
        claimed_acquisition_receipts[acquisition_index] = _require_sha256(
            entry["receipt_sha256"], f"{field}.receipt_sha256"
        )
    if receipt_indices != expected_indices:
        raise PopulationDiscoveryError("final acquisition receipt indices disagree")

    queries = {str(query["query_id"]): query for query in normalized["queries"]}
    derived_snapshots: dict[str, dict[str, str]] = {
        query_id: {} for query_id in queries
    }
    for row in candidate_rows:
        for query_id in row["query_ids"]:
            derived_snapshots[str(query_id)][str(row["candidate_identity"])] = str(
                row["item_sha256"]
            )

    responses_by_acquisition: dict[int, list[Mapping[str, Any]]] = {}
    for raw_response in responses:
        response = _require_mapping(raw_response, "final response")
        index = _require_int(
            response["acquisition_index"], "final response acquisition index", minimum=1
        )
        responses_by_acquisition.setdefault(index, []).append(response)
    shards_by_acquisition: dict[int, list[Mapping[str, Any]]] = {}
    for raw_shard in shards:
        shard = _require_mapping(raw_shard, "final shard")
        _require_exact_keys(shard, _SHARD_RECORD_KEYS, "final shard")
        raw_index = shard["acquisition_index"]
        if not isinstance(raw_index, str) or not raw_index.isdigit():
            raise PopulationDiscoveryError("final shard acquisition index is invalid")
        shards_by_acquisition.setdefault(int(raw_index), []).append(shard)
    if (
        sorted(responses_by_acquisition) != expected_indices
        or sorted(shards_by_acquisition) != expected_indices
        or any(index not in expected_indices for index in visibility_by_acquisition)
    ):
        raise PopulationDiscoveryError("final ledgers do not cover every acquisition")
    for acquisition_index in expected_indices:
        visibility_by_acquisition.setdefault(acquisition_index, [])

    expected_response_order = sorted(
        responses,
        key=lambda row: (
            row["query_id"],
            row["acquisition_index"],
            row["shard_start_utc"],
            row["shard_end_utc"],
            row["page"],
        ),
    )
    expected_shard_order = sorted(
        shards,
        key=lambda row: (
            row["query_id"],
            row["acquisition_index"],
            row["start_utc"],
            row["end_utc"],
        ),
    )
    expected_visibility_order = sorted(
        visibility,
        key=lambda row: (
            row["acquisition_index"],
            row["receipt"]["repository_url"],
        ),
    )
    if (
        responses != expected_response_order
        or shards != expected_shard_order
        or visibility != expected_visibility_order
    ):
        raise PopulationDiscoveryError("final ledgers are not in canonical order")

    recomputed_windows: list[dict[str, Any]] = []
    combined_findings: list[dict[str, Any]] = []
    first_shard_snapshot: list[dict[str, Any]] | None = None
    previous_end: datetime | None = None
    for acquisition_index in expected_indices:
        acquisition_visibility = visibility_by_acquisition[acquisition_index]
        if len({item["repository_url"] for item in acquisition_visibility}) != len(
            acquisition_visibility
        ):
            raise PopulationDiscoveryError(
                "final visibility ledger contains duplicates"
            )
        synthetic_receipt = {
            "responses": responses_by_acquisition[acquisition_index],
            "shards": shards_by_acquisition[acquisition_index],
            "repository_visibility_receipts": acquisition_visibility,
        }
        scan = _validate_response_ledger(
            output_directory,
            synthetic_receipt,
            normalized=normalized,
            candidate_rows=candidate_rows,
            snapshots=derived_snapshots,
        )
        combined_findings.extend(scan["findings"])
        server_dates = [
            _server_date(response)
            for response in responses_by_acquisition[acquisition_index]
        ]
        start = min(server_dates)
        end = max(server_dates)
        visibility_map = {
            str(item["repository_url"]): item for item in acquisition_visibility
        }
        materialized_checkpoint = _checkpoint_document(
            normalized,
            acquisition_index=acquisition_index,
            responses=responses_by_acquisition[acquisition_index],
            visibility_receipts=visibility_map,
            status="materialized",
            qualification_mode=_PUBLIC_QUALIFIED,
        )
        recomputed_acquisition = _acquisition_receipt_document(
            normalized,
            acquisition_index=acquisition_index,
            qualification_mode=_PUBLIC_QUALIFIED,
            server_start_utc=_format_utc(start),
            server_end_utc=_format_utc(end),
            responses=responses_by_acquisition[acquisition_index],
            shards=shards_by_acquisition[acquisition_index],
            candidate_rows=candidate_rows,
            candidate_bytes=candidate_bytes,
            query_snapshots=derived_snapshots,
            privacy_scan=scan,
            repository_visibility_receipts=acquisition_visibility,
            checkpoint_sha256=materialized_checkpoint["checkpoint_sha256"],
        )
        if (
            claimed_acquisition_receipts[acquisition_index]
            != recomputed_acquisition["receipt_sha256"]
        ):
            raise PopulationDiscoveryError(
                "final acquisition receipt lineage does not recompute"
            )
        if previous_end is not None and start < previous_end + timedelta(
            seconds=int(normalized["repeat_separation_seconds"])
        ):
            raise PopulationDiscoveryError(
                "final acquisition windows violate repeat separation"
            )
        recomputed_windows.append(
            {
                "acquisition_index": acquisition_index,
                "server_start_utc": _format_utc(start),
                "server_end_utc": _format_utc(end),
                "response_count": len(responses_by_acquisition[acquisition_index]),
            }
        )
        previous_end = end
        shard_snapshot = [
            {key: value for key, value in shard.items() if key != "acquisition_index"}
            for shard in shards_by_acquisition[acquisition_index]
        ]
        if first_shard_snapshot is None:
            first_shard_snapshot = shard_snapshot
        elif shard_snapshot != first_shard_snapshot:
            raise PopulationDiscoveryError(
                "final terminal shard identities drifted across acquisitions"
            )
    if windows != recomputed_windows:
        raise PopulationDiscoveryError("final acquisition windows do not recompute")

    recomputed_privacy = _privacy_summary(combined_findings, candidate_rows)
    privacy_scan = _validate_privacy_scan(
        receipt.get("privacy_scan"), name="final privacy_scan"
    )
    if privacy_scan != recomputed_privacy:
        raise PopulationDiscoveryError("final privacy scan does not recompute")
    expected_status = (
        "blocked_pending_privacy_review"
        if privacy_scan["downstream_authoring_export_blocked"]
        else "frozen_candidate_discovery"
    )
    if receipt.get("status") != expected_status:
        raise PopulationDiscoveryError(
            "final status disagrees with privacy disposition"
        )
    if _scan_json_privacy(receipt, path="$"):
        raise PopulationDiscoveryError(
            "final receipt serialized credential-shaped text"
        )
    receipt["receipt_sha256"] = claimed
    return receipt


def _recover_final_publication(
    target: Path,
    *,
    normalized: Mapping[str, Any],
    expected_acquisition_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    marker = _load_final_publication_marker(target)
    if marker is None:
        return None
    marker_path = _final_publication_marker(target)
    staging = target.parent / str(marker["staging_name"])
    target_exists = target.exists() or target.is_symlink()
    staging_exists = staging.exists() or staging.is_symlink()
    if target_exists and staging_exists:
        raise PopulationDiscoveryError(
            "publication recovery found both target and staging directories"
        )
    if not target_exists and not staging_exists:
        _remove_final_publication_marker(marker_path, durable=True)
        return None
    candidate = target if target_exists else staging
    _require_private_directory(candidate, name="publication recovery directory")
    receipt = _reload_final_receipt(candidate, normalized=normalized)
    if receipt["receipt_sha256"] != marker["receipt_sha256"] or receipt[
        "acquisition_receipts"
    ] != [dict(item) for item in expected_acquisition_receipts]:
        raise PopulationDiscoveryError(
            "publication recovery artifact differs from its locked inputs"
        )
    if not target_exists:
        _finish_private_directory_publication(
            staging,
            target,
            marker_path=marker_path,
        )
    else:
        try:
            _fsync_directory(target)
            _fsync_directory(target.parent)
        except OSError as exc:
            raise PublicationRecoveryRequired(
                "final publication recovery could not establish durability"
            ) from exc
        try:
            _remove_final_publication_marker(marker_path, durable=True)
        except OSError:
            pass
    return receipt


def _materialize_final_output(
    normalized: Mapping[str, Any],
    *,
    loaded: Sequence[
        tuple[Path, dict[str, Any], bytes, dict[str, Any], list[dict[str, Any]]]
    ],
    candidate_bytes: bytes,
    candidate_rows: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> dict[str, Any]:
    raw_directory = output_directory / "responses"
    _ensure_private_directory(raw_directory)
    responses: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    acquisition_receipts: list[dict[str, Any]] = []
    visibility_receipts: list[dict[str, Any]] = []
    combined_privacy_findings: list[dict[str, Any]] = []
    for directory, receipt, _, _, _ in loaded:
        combined_privacy_findings.extend(receipt["privacy_scan"]["findings"])
        acquisition_receipts.append(
            {
                "acquisition_index": receipt["acquisition_index"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
        )
        visibility_receipts.extend(
            {
                "acquisition_index": receipt["acquisition_index"],
                "receipt": visibility_receipt,
            }
            for visibility_receipt in receipt["repository_visibility_receipts"]
        )
        shards.extend(receipt["shards"])
        for response in receipt["responses"]:
            source = _locked_artifact_path(
                directory,
                response["body_path"],
                expected=f"responses/{response['body_sha256']}.json",
                field="response.body_path",
            )
            body = _read_private(
                source,
                name="raw acquisition response",
                maximum_bytes=int(normalized["max_response_bytes"]),
            )
            if _sha256_bytes(body) != response["body_sha256"]:
                raise PopulationDiscoveryError(
                    "raw acquisition response digest mismatch"
                )
            destination = raw_directory / f"{response['body_sha256']}.json"
            _write_private(destination, body, allow_identical=True)
            responses.append(dict(response))

    _write_private(output_directory / "candidates.jsonl", candidate_bytes)
    response_manifest = sorted(
        responses,
        key=lambda row: (
            row["query_id"],
            row["acquisition_index"],
            row["shard_start_utc"],
            row["shard_end_utc"],
            row["page"],
        ),
    )
    privacy_scan = _privacy_summary(combined_privacy_findings, candidate_rows)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_id": f"{normalized['plan_id']}-receipt",
        "lane_id": normalized["lane_id"],
        "status": (
            "blocked_pending_privacy_review"
            if privacy_scan["downstream_authoring_export_blocked"]
            else "frozen_candidate_discovery"
        ),
        "candidate_discovery_only": True,
        "eligibility_or_population_claim": False,
        "public_only_repository_visibility_verified": True,
        "plan_sha256": stable_digest(normalized),
        "api_origin": API_ORIGIN,
        "api_version": normalized["api_version"],
        "source_query_plan_sha256": normalized["source_query_plan_sha256"],
        "source_sampling_protocol_sha256": normalized[
            "source_sampling_protocol_sha256"
        ],
        "sampling_protocol_id": normalized["sampling_protocol_id"],
        "qualification_mode": _PUBLIC_QUALIFIED,
        "temporal_scope": normalized["temporal_scope"],
        "selection_source_cutoff_utc": normalized["selection_source_cutoff_utc"],
        "acquisition_repetitions": normalized["acquisition_repetitions"],
        "index_stabilization_seconds": normalized["index_stabilization_seconds"],
        "repeat_separation_seconds": normalized["repeat_separation_seconds"],
        "acquisition_windows": [dict(window) for window in windows],
        "acquisition_receipts": acquisition_receipts,
        "credential_profile": dict(loaded[0][1]["credential_profile"]),
        "collector_implementation": dict(normalized["collector_implementation"]),
        "public_visibility": dict(normalized["public_visibility"]),
        "query_count": len(normalized["queries"]),
        "shards": sorted(
            shards,
            key=lambda row: (
                row["query_id"],
                row["acquisition_index"],
                row["start_utc"],
                row["end_utc"],
            ),
        ),
        "responses": response_manifest,
        "response_count": len(response_manifest),
        "repository_visibility_receipts": sorted(
            visibility_receipts,
            key=lambda row: (
                row["acquisition_index"],
                row["receipt"]["repository_url"],
            ),
        ),
        "repository_visibility_receipt_count": len(visibility_receipts),
        "candidate_count": len(candidate_rows),
        "candidates_path": "candidates.jsonl",
        "candidates_sha256": _sha256_bytes(candidate_bytes),
        "privacy_scan": privacy_scan,
        "limitations": list(_FINAL_LIMITATIONS),
    }
    receipt["receipt_sha256"] = stable_digest(receipt)
    _write_private(
        output_directory / "query-receipt.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    reloaded = _reload_final_receipt(output_directory, normalized=normalized)
    if reloaded != receipt:
        raise PopulationDiscoveryError(
            "reloaded final receipt changed after serialization"
        )
    return reloaded


def finalize_discovery(  # noqa: C901 - one atomic reconciliation boundary
    plan: Mapping[str, Any],
    *,
    acquisition_directories: Sequence[Path],
    output_directory: Path,
) -> dict[str, Any]:
    """Reconcile separately scheduled acquisitions into one candidate receipt."""

    normalized = validate_query_plan(plan)
    if len(acquisition_directories) != normalized["acquisition_repetitions"]:
        raise PopulationDiscoveryError(
            "finalization requires every locked acquisition directory"
        )
    query_ids = {str(query["query_id"]) for query in normalized["queries"]}
    loaded: list[
        tuple[Path, dict[str, Any], bytes, dict[str, Any], list[dict[str, Any]]]
    ] = []
    for directory in acquisition_directories:
        _require_private_directory(directory, name="acquisition directory")
        receipt = _load_verified_receipt(
            directory / "acquisition-receipt.json", name="acquisition receipt"
        )
        privacy_scan = _validate_acquisition_receipt(receipt, normalized=normalized)
        if receipt["qualification_mode"] != _PUBLIC_QUALIFIED:
            raise PopulationDiscoveryError(
                "diagnostic acquisitions cannot produce a public final receipt"
            )
        checkpoint_responses, checkpoint_visibility = _load_checkpoint(
            directory,
            normalized=normalized,
            acquisition_index=int(receipt["acquisition_index"]),
            require_materialized=True,
            qualification_mode=str(receipt["qualification_mode"]),
        )
        checkpoint = _checkpoint_document(
            normalized,
            acquisition_index=int(receipt["acquisition_index"]),
            responses=list(checkpoint_responses.values()),
            visibility_receipts=checkpoint_visibility,
            status="materialized",
            qualification_mode=str(receipt["qualification_mode"]),
        )
        receipt_response_map = {
            str(response["request_url_sha256"]): response
            for response in receipt["responses"]
        }
        receipt_visibility_map = {
            str(item["repository_url"]): item
            for item in receipt["repository_visibility_receipts"]
        }
        if (
            checkpoint["checkpoint_sha256"] != receipt["checkpoint_sha256"]
            or checkpoint_responses != receipt_response_map
            or checkpoint_visibility != receipt_visibility_map
        ):
            raise PopulationDiscoveryError(
                "materialized checkpoint does not match the acquisition receipt"
            )
        candidate_path = _locked_artifact_path(
            directory,
            receipt.get("candidates_path"),
            expected="candidates.jsonl",
            field="acquisition.candidates_path",
        )
        candidate_bytes = _read_private(candidate_path, name="acquisition candidates")
        if _sha256_bytes(candidate_bytes) != receipt.get("candidates_sha256"):
            raise PopulationDiscoveryError(
                "acquisition candidates digest does not match"
            )
        candidate_rows = _parse_candidate_rows(candidate_bytes, query_ids=query_ids)
        if len(candidate_rows) != receipt.get("candidate_count"):
            raise PopulationDiscoveryError("acquisition candidate total disagrees")
        snapshots_path = _locked_artifact_path(
            directory,
            receipt.get("query_snapshots_path"),
            expected="query-snapshots.json",
            field="acquisition.query_snapshots_path",
        )
        snapshots_bytes = _read_private(snapshots_path, name="acquisition snapshots")
        if _sha256_bytes(snapshots_bytes) != receipt.get("query_snapshots_sha256"):
            raise PopulationDiscoveryError(
                "acquisition snapshots digest does not match"
            )
        snapshots = _require_mapping(
            _strict_json_loads(snapshots_bytes, field="acquisition query snapshots"),
            "acquisition query snapshots",
        )
        normalized_snapshots = dict(snapshots)
        _validate_query_snapshots(
            normalized_snapshots, candidate_rows, query_ids=query_ids
        )
        recomputed_privacy = _validate_response_ledger(
            directory,
            receipt,
            normalized=normalized,
            candidate_rows=candidate_rows,
            snapshots=normalized_snapshots,
        )
        if recomputed_privacy != privacy_scan:
            raise PopulationDiscoveryError(
                "acquisition privacy scan does not recompute"
            )
        loaded.append(
            (directory, receipt, candidate_bytes, normalized_snapshots, candidate_rows)
        )
    loaded.sort(key=lambda item: int(item[1]["acquisition_index"]))
    indices = [int(item[1]["acquisition_index"]) for item in loaded]
    expected_indices = list(range(1, normalized["acquisition_repetitions"] + 1))
    if indices != expected_indices:
        raise PopulationDiscoveryError("acquisition indices are missing or duplicated")
    credential_profiles = {
        json.dumps(item[1].get("credential_profile"), sort_keys=True) for item in loaded
    }
    if len(credential_profiles) != 1:
        raise PopulationDiscoveryError(
            "acquisitions used different credential profiles"
        )

    first_candidates = loaded[0][2]
    first_snapshots = loaded[0][3]
    first_candidate_rows = loaded[0][4]
    first_shard_snapshot = [
        {key: value for key, value in shard.items() if key != "acquisition_index"}
        for shard in loaded[0][1]["shards"]
    ]
    previous_end: datetime | None = None
    windows: list[dict[str, Any]] = []
    for _, receipt, candidate_bytes, snapshots, _ in loaded:
        if candidate_bytes != first_candidates or snapshots != first_snapshots:
            raise PopulationDiscoveryError(
                "candidate identities or semantic content drifted across acquisitions"
            )
        shard_snapshot = [
            {key: value for key, value in shard.items() if key != "acquisition_index"}
            for shard in receipt["shards"]
        ]
        if shard_snapshot != first_shard_snapshot:
            raise PopulationDiscoveryError(
                "terminal shard identities drifted across acquisitions"
            )
        start = _parse_utc(receipt["server_start_utc"], "acquisition.server_start_utc")
        end = _parse_utc(receipt["server_end_utc"], "acquisition.server_end_utc")
        if start > end:
            raise PopulationDiscoveryError("acquisition server window is invalid")
        if previous_end is not None and start < previous_end + timedelta(
            seconds=normalized["repeat_separation_seconds"]
        ):
            raise PopulationDiscoveryError(
                "frozen acquisitions do not satisfy the server-time separation gate"
            )
        windows.append(
            {
                "acquisition_index": receipt["acquisition_index"],
                "server_start_utc": receipt["server_start_utc"],
                "server_end_utc": receipt["server_end_utc"],
                "response_count": receipt["response_count"],
            }
        )
        previous_end = end

    acquisition_receipts = [
        {
            "acquisition_index": receipt["acquisition_index"],
            "receipt_sha256": receipt["receipt_sha256"],
        }
        for _, receipt, _, _, _ in loaded
    ]
    recovered = _recover_final_publication(
        output_directory,
        normalized=normalized,
        expected_acquisition_receipts=acquisition_receipts,
    )
    if recovered is not None:
        return recovered

    requested_output_directory = output_directory
    staging_directory = _private_atomic_staging_directory(requested_output_directory)
    try:
        receipt = _materialize_final_output(
            normalized,
            loaded=loaded,
            candidate_bytes=first_candidates,
            candidate_rows=first_candidate_rows,
            windows=windows,
            output_directory=staging_directory,
        )
        _commit_private_directory(
            staging_directory,
            requested_output_directory,
            receipt_sha256=receipt["receipt_sha256"],
        )
        return receipt
    except PublicationRecoveryRequired:
        raise
    except Exception:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)
        marker_path = _final_publication_marker(requested_output_directory)
        if not requested_output_directory.exists() and (
            marker_path.exists() or marker_path.is_symlink()
        ):
            try:
                _remove_final_publication_marker(marker_path, durable=True)
            except OSError:
                pass
        raise


def freeze_discovery(
    plan: Mapping[str, Any],
    *,
    transport: Transport,
    visibility_transport: Transport,
    output_directory: Path,
) -> dict[str, Any]:
    """Convenience path for a one-acquisition pilot or diagnostic plan."""

    normalized = validate_query_plan(plan)
    if normalized["acquisition_repetitions"] != 1:
        raise PopulationDiscoveryError(
            "multi-acquisition plans require separate --acquisition-index runs"
        )
    import tempfile

    _reject_symlink_ancestors(output_directory.parent, field="output parent")
    try:
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        if not output_directory.parent.is_dir():
            raise PopulationDiscoveryError("output parent is not a directory")
    except PopulationDiscoveryError:
        raise
    except OSError as exc:
        raise PopulationDiscoveryError("cannot prepare output parent") from exc
    try:
        with tempfile.TemporaryDirectory(
            prefix="fugue-github-acquisition-", dir=output_directory.parent
        ) as temporary:
            acquisition_directory = Path(temporary) / "acquisition-1"
            freeze_acquisition(
                normalized,
                acquisition_index=1,
                transport=transport,
                visibility_transport=visibility_transport,
                output_directory=acquisition_directory,
            )
            return finalize_discovery(
                normalized,
                acquisition_directories=[acquisition_directory],
                output_directory=output_directory,
            )
    except PopulationDiscoveryError:
        raise
    except OSError as exc:
        raise PopulationDiscoveryError(
            "cannot prepare or clean diagnostic acquisition directory"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze auditable GitHub candidate-discovery responses."
    )
    parser.add_argument("query_plan", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--acquisition-index", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--finalize-acquisition",
        action="append",
        type=Path,
        default=[],
        help="A separately frozen acquisition directory; repeat for every index.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_query_plan(args.query_plan)
        token_env_name = _require_token_env_name(args.token_env)
        if token_env_name != plan["credential_profile"]["token_env_name"]:
            raise PopulationDiscoveryError(
                "credential environment differs from the lane lock"
            )
        if args.acquisition_index is not None and args.finalize_acquisition:
            raise PopulationDiscoveryError(
                "acquisition and finalization modes are mutually exclusive"
            )
        if args.resume and args.acquisition_index is None:
            raise PopulationDiscoveryError("--resume requires --acquisition-index")
        if args.finalize_acquisition:
            receipt = finalize_discovery(
                plan,
                acquisition_directories=args.finalize_acquisition,
                output_directory=args.output_directory,
            )
        else:
            transport = github_transport(plan=plan)
            if args.acquisition_index is not None:
                receipt = freeze_acquisition(
                    plan,
                    acquisition_index=args.acquisition_index,
                    transport=transport,
                    visibility_transport=transport,
                    output_directory=args.output_directory,
                    resume=args.resume,
                )
            else:
                receipt = freeze_discovery(
                    plan,
                    transport=transport,
                    visibility_transport=transport,
                    output_directory=args.output_directory,
                )
    except PopulationDiscoveryError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "frozen_candidate_discovery", **receipt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
