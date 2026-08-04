from __future__ import annotations

import importlib.util
import json
import math
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "examples"
    / "comparisons"
    / "community-skill-upgrades"
    / "github_population_v2.py"
)
SPEC = importlib.util.spec_from_file_location("github_population_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _plan(*, maximum: int = 2) -> dict[str, Any]:
    implementation = {
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "collector_sha256": "3" * 64,
        "compiler_sha256": "4" * 64,
        "files_match_commit": False,
    }
    implementation["lock_sha256"] = MODULE.stable_digest(implementation)
    return {
        "schema_version": 2,
        "plan_id": "test-public-population-v2",
        "lane_id": "test-public-population",
        "candidate_discovery_only": True,
        "api_version": "2022-11-28",
        "source_query_plan_sha256": "a" * 64,
        "source_sampling_protocol_sha256": "b" * 64,
        "sampling_protocol_id": "test-public-population-protocol-v2",
        "population_scope": "Public records discoverable by the frozen query plan.",
        "temporal_scope": "pre-treatment public maintenance records",
        "selection_source_cutoff_utc": "2026-01-04T00:00:00Z",
        "acquisition_repetitions": 1,
        "index_stabilization_seconds": 0,
        "repeat_separation_seconds": 0,
        "queries": [
            {
                "query_id": "security-prs",
                "endpoint": "search/issues",
                "query": '"security" is:pr is:merged is:public',
                "date_qualifier": "merged",
                "start_utc": "2026-01-01T00:00:00Z",
                "end_utc": "2026-01-01T00:00:03Z",
                "max_results_per_shard": maximum,
                "per_page": 100,
                "sort": "created",
                "order": "asc",
                "shard_granularity": "second",
                "unsplittable_overflow_policy": ("fail_closed_new_protocol_required"),
            }
        ],
        "completeness": {
            "require_incomplete_results_false": True,
            "require_all_pages": True,
            "record_response_bytes": True,
            "fail_if_unsplittable": True,
        },
        "deduplication": {
            "primary_identity_fields": ["node_id", "id"],
            "candidate_only_not_lineage_deduplication": True,
        },
        "public_visibility": {
            "required_query_qualifier": "is:public",
            "independent_repository_lookup": True,
            "persist_search_body_before_visibility_verification": False,
            "persist_repository_response_body": False,
        },
        "credential_profile": {
            "profile_id": "test-public-population-github-public-read-v2",
            "token_env_name": "GITHUB_TOKEN",
            "credential_required": True,
            "credential_value_serialized": False,
        },
        "rate_limit_policy": {
            "required_header_names": [
                "x-ratelimit-limit",
                "x-ratelimit-remaining",
                "x-ratelimit-reset",
                "x-ratelimit-resource",
                "x-ratelimit-used",
            ],
            "minimum_remaining": 1,
            "maximum_wait_seconds": 60,
        },
        "max_response_bytes": 16 * 1024 * 1024,
        "collector_implementation": implementation,
    }


def _response(
    module: Any,
    url: str,
    params: dict[str, str],
    *,
    total: int,
    items: list[dict[str, Any]],
    incomplete: bool = False,
    response_date: str = "Mon, 05 Jan 2026 00:00:00 GMT",
    header_overrides: dict[str, str] | None = None,
    request_started_at_utc: str = "2026-01-05T00:00:00Z",
    request_completed_at_utc: str = "2026-01-05T00:00:01Z",
) -> Any:
    normalized_items = [
        {
            **item,
            "repository_url": item.get(
                "repository_url", "https://api.github.com/repos/acme/public-repo"
            ),
        }
        for item in items
    ]
    body = json.dumps(
        {
            "total_count": total,
            "incomplete_results": incomplete,
            "items": normalized_items,
        },
        sort_keys=True,
    ).encode()
    page = int(params["page"])
    page_count = max(1, math.ceil(total / int(params["per_page"])))
    links: list[str] = []
    if page < page_count:
        next_params = {**params, "page": str(page + 1)}
        links.append(f'<{url}?{urlencode(next_params)}>; rel="next"')
    if page > 1:
        previous_params = {**params, "page": str(page - 1)}
        links.append(f'<{url}?{urlencode(previous_params)}>; rel="prev"')
    headers = {
        "Date": response_date,
        "ETag": '"frozen"',
        "X-GitHub-Request-Id": "request-1",
        "X-GitHub-Api-Version-Selected": "2022-11-28",
        "X-RateLimit-Limit": "5000",
        "X-RateLimit-Remaining": "4999",
        "X-RateLimit-Reset": "1893456000",
        "X-RateLimit-Resource": "search",
        "X-RateLimit-Used": "1",
        "Link": ", ".join(links),
        "Authorization": "must-not-be-recorded",
    }
    headers.update(header_overrides or {})
    return module.FrozenResponse(
        url=f"{url}?{urlencode(params)}",
        status=200,
        headers=headers,
        body=body,
        requested_url=f"{url}?{urlencode(params)}",
        redirect_chain=(),
        request_started_at_utc=request_started_at_utc,
        request_completed_at_utc=request_completed_at_utc,
    )


def _visibility_transport(url: str, params: dict[str, str]) -> Any:
    assert params == {}
    assert url == "https://api.github.com/repos/acme/public-repo"
    body = json.dumps(
        {
            "url": url,
            "full_name": "acme/public-repo",
            "node_id": "R_public",
            "id": 1001,
            "private": False,
            "visibility": "public",
        },
        sort_keys=True,
    ).encode()
    return MODULE.FrozenResponse(
        url=url,
        status=200,
        headers={
            "Date": "Mon, 05 Jan 2026 00:00:00 GMT",
            "ETag": '"visibility"',
            "X-GitHub-Request-Id": "visibility-request-1",
            "X-GitHub-Api-Version-Selected": "2022-11-28",
            "X-RateLimit-Limit": "5000",
            "X-RateLimit-Remaining": "4998",
            "X-RateLimit-Reset": "1893456000",
            "X-RateLimit-Resource": "core",
            "X-RateLimit-Used": "2",
        },
        body=body,
        requested_url=url,
        redirect_chain=(),
        request_started_at_utc="2026-01-05T00:00:00Z",
        request_completed_at_utc="2026-01-05T00:00:01Z",
    )


class _HTTPResponseAdapter:
    def __init__(self, response: Any) -> None:
        self.status = response.status
        self.headers = dict(response.headers)
        self._body = response.body
        self._url = response.url

    def __enter__(self) -> _HTTPResponseAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, maximum: int) -> bytes:
        return self._body[:maximum]


def _live_transport(
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, Any],
    search_transport: Any,
    visibility_transport: Any = _visibility_transport,
) -> Any:
    monkeypatch.setenv("GITHUB_TOKEN", "test-only-public-read-credential")
    monkeypatch.setattr(MODULE, "_verify_live_implementation_lock", lambda _: None)

    def open_request(request: Any, *, timeout: int) -> _HTTPResponseAdapter:
        assert timeout == 60
        parsed = urlsplit(request.full_url)
        url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        params = {
            key: values[0]
            for key, values in parse_qs(parsed.query, keep_blank_values=True).items()
        }
        response = (
            visibility_transport(url, params)
            if parsed.path.startswith("/repos/")
            else search_transport(url, params)
        )
        return _HTTPResponseAdapter(response)

    monkeypatch.setattr(MODULE, "_open_no_redirect", open_request)
    return MODULE.github_transport(plan=plan)


def _freeze_live_discovery(
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, Any],
    *,
    search_transport: Any,
    output_directory: Path,
    visibility_transport: Any = _visibility_transport,
) -> dict[str, Any]:
    live = _live_transport(
        monkeypatch,
        plan,
        search_transport,
        visibility_transport,
    )
    return MODULE.freeze_discovery(
        plan,
        transport=live,
        visibility_transport=live,
        output_directory=output_directory,
    )


def test_freeze_discovery_splits_large_ranges_and_records_every_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def transport(url: str, params: dict[str, str]) -> Any:
        query = params["q"]
        calls.append(query)
        if "00:00:00Z..2026-01-01T00:00:03Z" in query:
            return _response(
                MODULE,
                url,
                params,
                total=4,
                items=[{"id": value, "node_id": f"PR_{value}"} for value in range(4)],
            )
        if "00:00:00Z..2026-01-01T00:00:01Z" in query:
            return _response(
                MODULE,
                url,
                params,
                total=2,
                items=[{"id": 1, "node_id": "PR_1"}, {"id": 2, "node_id": "PR_2"}],
            )
        assert "00:00:02Z..2026-01-01T00:00:03Z" in query
        return _response(
            MODULE,
            url,
            params,
            total=2,
            items=[{"id": 3, "node_id": "PR_3"}, {"id": 4, "node_id": "PR_4"}],
        )

    output = tmp_path / "discovery"
    plan = _plan()
    live = _live_transport(monkeypatch, plan, transport)
    receipt = MODULE.freeze_discovery(
        plan,
        transport=live,
        output_directory=output,
        visibility_transport=live,
    )

    assert len(calls) == 3
    assert receipt["response_count"] == 3
    assert receipt["candidate_count"] == 4
    assert receipt["eligibility_or_population_claim"] is False
    assert receipt["credential_profile"] == {
        "profile_id": "test-public-population-github-public-read-v2",
        "token_env_name": "GITHUB_TOKEN",
        "credential_present": True,
        "credential_value_serialized": False,
    }
    assert receipt["public_only_repository_visibility_verified"] is True
    assert receipt["repository_visibility_receipt_count"] == 1
    assert all("authorization" not in row["headers"] for row in receipt["responses"])
    assert all((output / row["body_path"]).is_file() for row in receipt["responses"])
    rows = [
        json.loads(line)
        for line in (output / "candidates.jsonl").read_text().splitlines()
    ]
    assert [row["candidate_identity"] for row in rows] == [
        "node_id:PR_1",
        "node_id:PR_2",
        "node_id:PR_3",
        "node_id:PR_4",
    ]
    frozen = json.loads((output / "query-receipt.json").read_text())
    digest = frozen.pop("receipt_sha256")
    assert digest == MODULE.stable_digest(frozen)


def test_freeze_discovery_fails_closed_on_unsplittable_search_cap(
    tmp_path: Path,
) -> None:
    plan = _plan(maximum=2)
    plan["queries"][0]["end_utc"] = plan["queries"][0]["start_utc"]

    def transport(url: str, params: dict[str, str]) -> Any:
        return _response(MODULE, url, params, total=3, items=[])

    with pytest.raises(MODULE.PopulationDiscoveryError, match="unsplittable second"):
        MODULE.freeze_discovery(
            plan,
            transport=transport,
            output_directory=tmp_path / "blocked",
            visibility_transport=_visibility_transport,
        )


def test_day_granularity_splits_only_at_utc_days_and_single_day_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(maximum=2)
    query = plan["queries"][0]
    query["shard_granularity"] = "day"
    query["start_utc"] = "2026-01-01T00:00:00Z"
    query["end_utc"] = "2026-01-02T23:59:59Z"

    def transport(url: str, params: dict[str, str]) -> Any:
        rendered = params["q"]
        if "merged:2026-01-01..2026-01-02" in rendered:
            items = [{"id": value, "node_id": f"PR_{value}"} for value in range(1, 5)]
        elif "merged:2026-01-01..2026-01-01" in rendered:
            items = [
                {"id": 1, "node_id": "PR_1"},
                {"id": 2, "node_id": "PR_2"},
            ]
        else:
            assert "merged:2026-01-02..2026-01-02" in rendered
            items = [
                {"id": 3, "node_id": "PR_3"},
                {"id": 4, "node_id": "PR_4"},
            ]
        return _response(MODULE, url, params, total=len(items), items=items)

    live = _live_transport(monkeypatch, plan, transport)
    result = MODULE.freeze_discovery(
        plan,
        transport=live,
        visibility_transport=live,
        output_directory=tmp_path / "day-split",
    )
    assert [(shard["start_utc"], shard["end_utc"]) for shard in result["shards"]] == [
        ("2026-01-01T00:00:00Z", "2026-01-01T23:59:59Z"),
        ("2026-01-02T00:00:00Z", "2026-01-02T23:59:59Z"),
    ]

    one_day = _plan(maximum=2)
    one_day_query = one_day["queries"][0]
    one_day_query["shard_granularity"] = "day"
    one_day_query["start_utc"] = "2026-01-01T00:00:00Z"
    one_day_query["end_utc"] = "2026-01-01T23:59:59Z"
    with pytest.raises(MODULE.PopulationDiscoveryError, match="unsplittable UTC day"):
        MODULE.freeze_discovery(
            one_day,
            transport=lambda url, params: _response(
                MODULE, url, params, total=3, items=[]
            ),
            visibility_transport=_visibility_transport,
            output_directory=tmp_path / "one-day-overflow",
        )


def test_freeze_discovery_rejects_incomplete_search_results(tmp_path: Path) -> None:
    def transport(url: str, params: dict[str, str]) -> Any:
        return _response(MODULE, url, params, total=1, items=[], incomplete=True)

    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="incomplete_results=true"
    ):
        MODULE.freeze_discovery(
            _plan(),
            transport=transport,
            output_directory=tmp_path / "blocked",
            visibility_transport=_visibility_transport,
        )


def test_freeze_discovery_rejects_missing_search_response_fields(
    tmp_path: Path,
) -> None:
    def transport(url: str, params: dict[str, str]) -> Any:
        response = _response(MODULE, url, params, total=0, items=[])
        return MODULE.FrozenResponse(
            url=response.url,
            status=response.status,
            headers=response.headers,
            body=json.dumps({"total_count": 0, "incomplete_results": False}).encode(),
            requested_url=response.requested_url,
            redirect_chain=response.redirect_chain,
            request_started_at_utc=response.request_started_at_utc,
            request_completed_at_utc=response.request_completed_at_utc,
        )

    with pytest.raises(MODULE.PopulationDiscoveryError, match="missing search fields"):
        MODULE.freeze_discovery(
            _plan(),
            transport=transport,
            output_directory=tmp_path / "missing-fields",
            visibility_transport=_visibility_transport,
        )


def test_repeated_acquisition_rejects_semantic_candidate_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    plan["acquisition_repetitions"] = 2
    plan["repeat_separation_seconds"] = 3600
    calls = 0

    def first_transport(url: str, params: dict[str, str]) -> Any:
        nonlocal calls
        calls += 1
        return _response(
            MODULE,
            url,
            params,
            total=1,
            items=[
                {
                    "id": 1,
                    "node_id": "PR_1",
                    "title": "before",
                    "score": calls,
                }
            ],
            response_date="Mon, 05 Jan 2026 00:00:00 GMT",
        )

    def second_transport(url: str, params: dict[str, str]) -> Any:
        return _response(
            MODULE,
            url,
            params,
            total=1,
            items=[{"id": 1, "node_id": "PR_1", "title": "after"}],
            response_date="Mon, 05 Jan 2026 02:00:00 GMT",
        )

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_live = _live_transport(monkeypatch, plan, first_transport)
    MODULE.freeze_acquisition(
        plan,
        acquisition_index=1,
        transport=first_live,
        output_directory=first,
        visibility_transport=first_live,
    )
    second_live = _live_transport(monkeypatch, plan, second_transport)
    MODULE.freeze_acquisition(
        plan,
        acquisition_index=2,
        transport=second_live,
        output_directory=second,
        visibility_transport=second_live,
    )
    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="semantic content drifted"
    ):
        MODULE.finalize_discovery(
            plan,
            acquisition_directories=[first, second],
            output_directory=tmp_path / "drifted",
        )


def test_query_plan_rejects_post_treatment_or_embedded_date_queries() -> None:
    plan = _plan()
    plan["queries"][0]["end_utc"] = "2026-01-05T00:00:00Z"
    with pytest.raises(MODULE.PopulationDiscoveryError, match="source cutoff"):
        MODULE.validate_query_plan(plan)

    plan = _plan()
    plan["queries"][0]["query"] += " merged:2025-01-01..2025-12-31"
    with pytest.raises(MODULE.PopulationDiscoveryError, match="must not embed"):
        MODULE.validate_query_plan(plan)


def test_query_plan_rejects_unknown_fields_and_population_claims() -> None:
    plan = _plan()
    plan["credential"] = "secret"
    with pytest.raises(MODULE.PopulationDiscoveryError, match="invalid keys"):
        MODULE.validate_query_plan(plan)

    plan = _plan()
    plan["candidate_discovery_only"] = False
    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="candidate discovery only"
    ):
        MODULE.validate_query_plan(plan)


def test_pagination_rejects_duplicate_identity_hidden_by_correct_page_lengths(
    tmp_path: Path,
) -> None:
    plan = _plan(maximum=10)
    plan["queries"][0]["per_page"] = 1

    def transport(url: str, params: dict[str, str]) -> Any:
        return _response(
            MODULE,
            url,
            params,
            total=2,
            items=[{"id": 1, "node_id": "PR_1"}],
        )

    with pytest.raises(MODULE.PopulationDiscoveryError, match="duplicate or missing"):
        MODULE.freeze_discovery(
            plan,
            transport=transport,
            output_directory=tmp_path / "duplicate-pages",
            visibility_transport=_visibility_transport,
        )


def test_collector_rejects_semantic_drift_across_query_atoms(tmp_path: Path) -> None:
    plan = _plan(maximum=10)
    plan["queries"].append(
        {
            **plan["queries"][0],
            "query_id": "privacy-prs",
            "query": '"privacy" is:pr is:merged is:public',
        }
    )

    def transport(url: str, params: dict[str, str]) -> Any:
        title = "security version" if '"security"' in params["q"] else "privacy version"
        return _response(
            MODULE,
            url,
            params,
            total=1,
            items=[{"id": 1, "node_id": "PR_1", "title": title}],
        )

    with pytest.raises(MODULE.PopulationDiscoveryError, match="frozen query atoms"):
        MODULE.freeze_discovery(
            plan,
            transport=transport,
            output_directory=tmp_path / "cross-query-drift",
            visibility_transport=_visibility_transport,
        )


def test_collector_rejects_duplicate_identity_across_split_shards(
    tmp_path: Path,
) -> None:
    def transport(url: str, params: dict[str, str]) -> Any:
        if "00:00:00Z..2026-01-01T00:00:03Z" in params["q"]:
            return _response(MODULE, url, params, total=4, items=[])
        return _response(
            MODULE,
            url,
            params,
            total=1,
            items=[{"id": 1, "node_id": "PR_1"}],
        )

    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="duplicates.*across shards"
    ):
        MODULE.freeze_discovery(
            _plan(),
            transport=transport,
            output_directory=tmp_path / "cross-shard-duplicate",
            visibility_transport=_visibility_transport,
        )


@pytest.mark.parametrize("tamper", ["url", "api_version", "next_link"])
def test_collector_rejects_response_provenance_drift(
    tmp_path: Path, tamper: str
) -> None:
    plan = _plan(maximum=10)
    if tamper == "next_link":
        plan["queries"][0]["per_page"] = 1

    def transport(url: str, params: dict[str, str]) -> Any:
        total = 2 if tamper == "next_link" else 1
        response = _response(
            MODULE,
            url,
            params,
            total=total,
            items=[{"id": int(params["page"]), "node_id": f"PR_{params['page']}"}],
        )
        headers = dict(response.headers)
        response_url = response.url
        if tamper == "url":
            response_url = response_url.replace("page=1", "page=9")
        elif tamper == "api_version":
            headers["X-GitHub-Api-Version-Selected"] = "2024-01-01"
        else:
            headers["Link"] = headers["Link"].replace("api.github.com", "example.com")
        return MODULE.FrozenResponse(
            url=response_url,
            status=response.status,
            headers=headers,
            body=response.body,
            requested_url=response.requested_url,
            redirect_chain=response.redirect_chain,
            request_started_at_utc=response.request_started_at_utc,
            request_completed_at_utc=response.request_completed_at_utc,
        )

    match = {
        "url": "unapproved redirect",
        "api_version": "API version",
        "next_link": "pagination Link",
    }[tamper]
    with pytest.raises(MODULE.PopulationDiscoveryError, match=match):
        MODULE.freeze_discovery(
            plan,
            transport=transport,
            output_directory=tmp_path / tamper,
            visibility_transport=_visibility_transport,
        )


def test_acquisition_enforces_stabilization_and_repeat_separation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(maximum=10)
    plan["index_stabilization_seconds"] = 86400

    def too_early(url: str, params: dict[str, str]) -> Any:
        return _response(
            MODULE,
            url,
            params,
            total=0,
            items=[],
            response_date="Sun, 04 Jan 2026 23:59:59 GMT",
        )

    with pytest.raises(MODULE.PopulationDiscoveryError, match="stabilization boundary"):
        MODULE.freeze_acquisition(
            plan,
            acquisition_index=1,
            transport=too_early,
            output_directory=tmp_path / "early",
            visibility_transport=_visibility_transport,
        )

    plan["acquisition_repetitions"] = 2
    plan["repeat_separation_seconds"] = 3600

    def at(server_date: str) -> Any:
        def transport(url: str, params: dict[str, str]) -> Any:
            return _response(
                MODULE,
                url,
                params,
                total=1,
                items=[{"id": 1, "node_id": "PR_1"}],
                response_date=server_date,
            )

        return transport

    first = tmp_path / "separation-first"
    second = tmp_path / "separation-second"
    first_live = _live_transport(monkeypatch, plan, at("Mon, 05 Jan 2026 00:00:00 GMT"))
    MODULE.freeze_acquisition(
        plan,
        acquisition_index=1,
        transport=first_live,
        output_directory=first,
        visibility_transport=first_live,
    )
    second_live = _live_transport(
        monkeypatch, plan, at("Mon, 05 Jan 2026 00:59:59 GMT")
    )
    MODULE.freeze_acquisition(
        plan,
        acquisition_index=2,
        transport=second_live,
        output_directory=second,
        visibility_transport=second_live,
    )
    with pytest.raises(MODULE.PopulationDiscoveryError, match="separation gate"):
        MODULE.finalize_discovery(
            plan,
            acquisition_directories=[first, second],
            output_directory=tmp_path / "too-close",
        )


def test_live_transport_requires_locked_environment_and_diagnostic_cannot_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(maximum=10)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(MODULE.PopulationDiscoveryError, match="is not present"):
        MODULE.github_transport(plan=plan)

    diagnostic = tmp_path / "diagnostic"
    receipt = MODULE.freeze_acquisition(
        plan,
        acquisition_index=1,
        transport=lambda url, params: _response(MODULE, url, params, total=0, items=[]),
        output_directory=diagnostic,
        visibility_transport=_visibility_transport,
    )
    assert receipt["qualification_mode"] == "diagnostic_nonqualifying"
    assert receipt["credential_profile"]["credential_present"] is False
    with pytest.raises(
        MODULE.PopulationDiscoveryError,
        match="diagnostic acquisitions cannot produce a public final receipt",
    ):
        MODULE.finalize_discovery(
            plan,
            acquisition_directories=[diagnostic],
            output_directory=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()

    monkeypatch.setenv("GITHUB_TOKEN", "test-only-public-read-credential")
    monkeypatch.setattr(MODULE, "_verify_live_implementation_lock", lambda _: None)
    live = MODULE.github_transport(plan=plan)
    monkeypatch.delenv("GITHUB_TOKEN")
    with pytest.raises(MODULE.PopulationDiscoveryError, match="no longer present"):
        MODULE.freeze_acquisition(
            plan,
            acquisition_index=1,
            transport=live,
            output_directory=tmp_path / "removed-after-construction",
            visibility_transport=live,
        )


def test_privacy_scan_preserves_private_raw_bytes_and_blocks_affected_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_shaped_source_text = "ghp_" + "A" * 36

    def transport(url: str, params: dict[str, str]) -> Any:
        return _response(
            MODULE,
            url,
            params,
            total=1,
            items=[
                {
                    "id": 1,
                    "node_id": "PR_1",
                    "body": f"public incident example {token_shaped_source_text}",
                }
            ],
        )

    output = tmp_path / "privacy"
    plan = _plan(maximum=10)
    receipt = _freeze_live_discovery(
        monkeypatch,
        plan,
        search_transport=transport,
        output_directory=output,
    )

    serialized_receipt = json.dumps(receipt, sort_keys=True)
    assert token_shaped_source_text not in serialized_receipt
    assert receipt["status"] == "blocked_pending_privacy_review"
    assert receipt["privacy_scan"]["status"] == "blocked_pending_review"
    assert receipt["privacy_scan"]["affected_candidate_references"] == ["node_id:PR_1"]
    assert receipt["privacy_scan"]["matched_values_serialized"] is False
    candidates = (output / "candidates.jsonl").read_text()
    assert token_shaped_source_text in candidates
    candidate = json.loads(candidates)
    assert candidate["privacy_review"]["status"] == "blocked_pending_review"
    assert candidate["privacy_review"]["downstream_authoring_export_allowed"] is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in output.rglob("*")
        if path.is_file()
    )
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o700
        for path in output.rglob("*")
        if path.is_dir()
    )


def test_finalization_rejects_content_address_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition = tmp_path / "acquisition"
    plan = _plan(maximum=10)
    live = _live_transport(
        monkeypatch,
        plan,
        lambda url, params: _response(MODULE, url, params, total=0, items=[]),
    )
    MODULE.freeze_acquisition(
        plan,
        acquisition_index=1,
        transport=live,
        output_directory=acquisition,
        visibility_transport=live,
    )
    receipt_path = acquisition / "acquisition-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["candidates_path"] = "../outside.jsonl"
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = MODULE.stable_digest(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(MODULE.PopulationDiscoveryError, match="must be exactly"):
        MODULE.finalize_discovery(
            plan,
            acquisition_directories=[acquisition],
            output_directory=tmp_path / "traversal-result",
        )


def test_private_repository_is_rejected_before_search_body_persistence(
    tmp_path: Path,
) -> None:
    marker = "TRANSIENT_PRIVATE_SEARCH_BODY_MUST_NOT_PERSIST"

    def search(url: str, params: dict[str, str]) -> Any:
        return _response(
            MODULE,
            url,
            params,
            total=1,
            items=[{"id": 1, "node_id": "PR_private", "title": marker}],
        )

    def private_visibility(url: str, params: dict[str, str]) -> Any:
        response = _visibility_transport(url, params)
        return MODULE.FrozenResponse(
            url=response.url,
            status=response.status,
            headers=response.headers,
            body=json.dumps(
                {
                    "url": url,
                    "full_name": "acme/public-repo",
                    "node_id": "R_private",
                    "private": True,
                    "visibility": "private",
                }
            ).encode(),
            requested_url=response.requested_url,
            redirect_chain=response.redirect_chain,
            request_started_at_utc=response.request_started_at_utc,
            request_completed_at_utc=response.request_completed_at_utc,
        )

    output = tmp_path / "private-blocked"
    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="not independently verified public"
    ):
        MODULE.freeze_acquisition(
            _plan(maximum=10),
            acquisition_index=1,
            transport=search,
            visibility_transport=private_visibility,
            output_directory=output,
        )

    assert not (output / "responses").exists()
    assert all(marker.encode() not in path.read_bytes() for path in output.rglob("*"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda plan: plan["credential_profile"].update(
                {"profile_id": "another-lane-github-public-read-v2"}
            ),
            "not bound to the lane",
        ),
        (
            lambda plan: plan["credential_profile"].update(
                {"token_env_name": "OTHER_TOKEN"}
            ),
            "credential environment",
        ),
        (
            lambda plan: plan["credential_profile"].update(
                {"credential_required": False}
            ),
            "credential presence is required",
        ),
    ],
)
def test_lane_credential_profile_contract_fails_closed(
    mutate: Any,
    message: str,
) -> None:
    plan = _plan(maximum=10)
    mutate(plan)
    with pytest.raises(MODULE.PopulationDiscoveryError, match=message):
        MODULE.validate_query_plan(plan)


def test_rate_limit_block_preserves_checkpoint_and_resume_skips_completed_request(
    tmp_path: Path,
) -> None:
    plan = _plan(maximum=10)
    plan["queries"].append(
        {
            **plan["queries"][0],
            "query_id": "privacy-prs",
            "query": '"privacy" is:pr is:merged is:public',
        }
    )
    calls: list[str] = []

    def blocked_transport(url: str, params: dict[str, str]) -> Any:
        calls.append(params["q"])
        headers = (
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "120"}
            if '"privacy"' in params["q"]
            else None
        )
        return _response(
            MODULE,
            url,
            params,
            total=0,
            items=[],
            header_overrides=headers,
        )

    output = tmp_path / "rate-resume"
    with pytest.raises(MODULE.PopulationDiscoveryError, match="later resume"):
        MODULE.freeze_acquisition(
            plan,
            acquisition_index=1,
            transport=blocked_transport,
            visibility_transport=_visibility_transport,
            output_directory=output,
            now=lambda: 0.0,
        )
    checkpoint = json.loads((output / "acquisition-checkpoint.json").read_text())
    assert checkpoint["status"] == "in_progress"
    assert len(checkpoint["responses"]) == 1

    def resumed_transport(url: str, params: dict[str, str]) -> Any:
        calls.append(params["q"])
        assert '"privacy"' in params["q"]
        return _response(MODULE, url, params, total=0, items=[])

    receipt = MODULE.freeze_acquisition(
        plan,
        acquisition_index=1,
        transport=resumed_transport,
        visibility_transport=_visibility_transport,
        output_directory=output,
        resume=True,
        now=lambda: 0.0,
    )
    assert receipt["response_count"] == 2
    assert sum('"security"' in query for query in calls) == 1
    assert sum('"privacy"' in query for query in calls) == 2


def test_finalizer_recomputes_ordered_shard_identity_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(maximum=10)
    acquisition = tmp_path / "shard-acquisition"
    live = _live_transport(
        monkeypatch,
        plan,
        lambda url, params: _response(
            MODULE,
            url,
            params,
            total=2,
            items=[{"id": 1, "node_id": "PR_1"}, {"id": 2, "node_id": "PR_2"}],
        ),
    )
    MODULE.freeze_acquisition(
        plan,
        acquisition_index=1,
        transport=live,
        visibility_transport=live,
        output_directory=acquisition,
    )
    receipt_path = acquisition / "acquisition-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["shards"][0]["ordered_identity_sha256"] = "f" * 64
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = MODULE.stable_digest(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(MODULE.PopulationDiscoveryError, match="does not recompute"):
        MODULE.finalize_discovery(
            plan,
            acquisition_directories=[acquisition],
            output_directory=tmp_path / "tampered-shard-result",
        )


def test_finalizer_requires_exact_cross_acquisition_shard_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(maximum=2)
    plan["acquisition_repetitions"] = 2
    plan["repeat_separation_seconds"] = 3600

    def at(server_date: str) -> Any:
        def transport(url: str, params: dict[str, str]) -> Any:
            query = params["q"]
            if "00:00:00Z..2026-01-01T00:00:03Z" in query:
                items = [
                    {"id": value, "node_id": f"PR_{value}"} for value in range(1, 5)
                ]
            elif "00:00:00Z..2026-01-01T00:00:01Z" in query:
                items = [
                    {"id": 1, "node_id": "PR_1"},
                    {"id": 2, "node_id": "PR_2"},
                ]
            else:
                items = [
                    {"id": 3, "node_id": "PR_3"},
                    {"id": 4, "node_id": "PR_4"},
                ]
            return _response(
                MODULE,
                url,
                params,
                total=len(items),
                items=items,
                response_date=server_date,
            )

        return transport

    first = tmp_path / "shards-first"
    second = tmp_path / "shards-second"
    for index, directory, server_date in (
        (1, first, "Mon, 05 Jan 2026 00:00:00 GMT"),
        (2, second, "Mon, 05 Jan 2026 02:00:00 GMT"),
    ):
        live = _live_transport(monkeypatch, plan, at(server_date))
        MODULE.freeze_acquisition(
            plan,
            acquisition_index=index,
            transport=live,
            visibility_transport=live,
            output_directory=directory,
        )
    receipt_path = second / "acquisition-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["shards"].reverse()
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = MODULE.stable_digest(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="shard identities drifted"
    ):
        MODULE.finalize_discovery(
            plan,
            acquisition_directories=[first, second],
            output_directory=tmp_path / "cross-shard-result",
        )


def test_strict_json_symlink_ancestors_and_filesystem_errors_are_blocked(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(_plan()).replace(
            '"schema_version": 2',
            '"schema_version": 2, "schema_version": 2',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(MODULE.PopulationDiscoveryError, match="duplicate JSON key"):
        MODULE.load_query_plan(duplicate)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(MODULE.PopulationDiscoveryError, match="symlink ancestor"):
        MODULE.freeze_acquisition(
            _plan(maximum=10),
            acquisition_index=1,
            transport=lambda url, params: _response(
                MODULE, url, params, total=0, items=[]
            ),
            visibility_transport=_visibility_transport,
            output_directory=linked_parent / "evidence",
        )

    regular_file = tmp_path / "not-a-directory"
    regular_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="cannot inspect output directory"
    ):
        MODULE.freeze_acquisition(
            _plan(maximum=10),
            acquisition_index=1,
            transport=lambda url, params: _response(
                MODULE, url, params, total=0, items=[]
            ),
            visibility_transport=_visibility_transport,
            output_directory=regular_file,
        )


def test_privacy_scanner_governs_queries_urls_and_safe_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert MODULE._PRIVACY_SCANNER_REVISION.endswith("v2")
    assert {
        "anthropic_api_key",
        "aws_access_key_id",
        "github_fine_grained_token",
        "openai_api_key",
        "private_key_pem",
        "slack_token",
        "stripe_live_secret",
    } <= set(MODULE._PRIVACY_PATTERNS)
    assert len(MODULE._privacy_pattern_manifest_sha256()) == 64

    query_secret = _plan(maximum=10)
    query_secret["queries"][0]["query"] += " ghp_" + "A" * 36
    with pytest.raises(MODULE.PopulationDiscoveryError, match="credential-shaped"):
        MODULE.validate_query_plan(query_secret)

    token = "ghp_" + "B" * 36

    def secret_url_transport(url: str, params: dict[str, str]) -> Any:
        response = _response(MODULE, url, params, total=0, items=[])
        return MODULE.FrozenResponse(
            url=f"{response.url}&marker={token}",
            status=response.status,
            headers=response.headers,
            body=response.body,
            requested_url=response.requested_url,
            redirect_chain=response.redirect_chain,
            request_started_at_utc=response.request_started_at_utc,
            request_completed_at_utc=response.request_completed_at_utc,
        )

    with pytest.raises(MODULE.PopulationDiscoveryError, match="URL contains"):
        MODULE.freeze_acquisition(
            _plan(maximum=10),
            acquisition_index=1,
            transport=secret_url_transport,
            visibility_transport=_visibility_transport,
            output_directory=tmp_path / "secret-url",
        )

    def transport(url: str, params: dict[str, str]) -> Any:
        return _response(
            MODULE,
            url,
            params,
            total=1,
            items=[{"id": 1, "node_id": "PR_1"}],
            header_overrides={"ETag": token},
        )

    output = tmp_path / "header-privacy"
    plan = _plan(maximum=10)
    result = _freeze_live_discovery(
        monkeypatch,
        plan,
        search_transport=transport,
        output_directory=output,
    )
    assert result["status"] == "blocked_pending_privacy_review"
    assert result["privacy_scan"]["matched_values_serialized"] is False
    assert all(
        token.encode() not in path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    )


@pytest.mark.parametrize(
    ("header_overrides", "started", "completed", "message"),
    [
        (
            {"X-RateLimit-Reset": ""},
            "2026-01-05T00:00:00Z",
            "2026-01-05T00:00:01Z",
            "audit headers",
        ),
        (None, "2026-01-05T00:00:02Z", "2026-01-05T00:00:01Z", "precedes its start"),
    ],
)
def test_request_audit_headers_and_times_are_required(
    tmp_path: Path,
    header_overrides: dict[str, str] | None,
    started: str,
    completed: str,
    message: str,
) -> None:
    with pytest.raises(MODULE.PopulationDiscoveryError, match=message):
        MODULE.freeze_acquisition(
            _plan(maximum=10),
            acquisition_index=1,
            transport=lambda url, params: _response(
                MODULE,
                url,
                params,
                total=0,
                items=[],
                header_overrides=header_overrides,
                request_started_at_utc=started,
                request_completed_at_utc=completed,
            ),
            visibility_transport=_visibility_transport,
            output_directory=tmp_path / message.replace(" ", "-"),
        )


def test_reloaded_final_receipt_rechecks_content_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(maximum=10)
    output = tmp_path / "reloaded-final"
    _freeze_live_discovery(
        monkeypatch,
        plan,
        search_transport=lambda url, params: _response(
            MODULE,
            url,
            params,
            total=1,
            items=[{"id": 1, "node_id": "PR_1"}],
        ),
        output_directory=output,
    )
    candidates = output / "candidates.jsonl"
    candidates.write_bytes(candidates.read_bytes() + b"\n")
    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="candidates contain a blank row"
    ):
        MODULE._reload_final_receipt(
            output, normalized=MODULE.validate_query_plan(plan)
        )


def test_exact_git_blob_lock_ignores_metadata_head_and_detects_source_drift(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    source_directory = (
        repository / "examples" / "comparisons" / "community-skill-upgrades"
    )
    source_directory.mkdir(parents=True)
    collector = source_directory / "github_population_v2.py"
    compiler = source_directory / "compile_github_collector_plan_v2.py"
    shutil.copy2(MODULE_PATH, collector)
    shutil.copy2(MODULE_PATH.with_name(compiler.name), compiler)

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
        )

    git("init")
    git("config", "user.email", "collector-test@example.invalid")
    git("config", "user.name", "Collector Test")
    git("add", ".")
    git("commit", "-m", "collector implementation")

    name = "github_population_v2_exact_blob_test"
    spec = importlib.util.spec_from_file_location(name, collector)
    assert spec is not None and spec.loader is not None
    copied_module = importlib.util.module_from_spec(spec)
    sys.modules[name] = copied_module
    spec.loader.exec_module(copied_module)
    first = copied_module.current_implementation_lock(compiler_path=compiler)
    assert first["files_match_commit"] is True

    (repository / "manifest.json").write_text('{"digest":"later"}\n')
    git("add", "manifest.json")
    git("commit", "-m", "record compiled digest")
    after_metadata = copied_module.current_implementation_lock(compiler_path=compiler)
    assert after_metadata == first

    collector.write_bytes(collector.read_bytes() + b"\n")
    drifted = copied_module.current_implementation_lock(compiler_path=compiler)
    assert drifted["source_commit"] == first["source_commit"]
    assert drifted["files_match_commit"] is False
    assert drifted["collector_sha256"] != first["collector_sha256"]


def test_fake_callable_cannot_self_assert_public_qualification(tmp_path: Path) -> None:
    plan = _plan(maximum=10)

    def fake(url: str, params: dict[str, str]) -> Any:
        return _response(MODULE, url, params, total=0, items=[])

    fake._fugue_public_qualified = True  # type: ignore[attr-defined]
    fake.credential_present = True  # type: ignore[attr-defined]
    acquisition = tmp_path / "fake"
    receipt = MODULE.freeze_acquisition(
        plan,
        acquisition_index=1,
        transport=fake,
        visibility_transport=fake,
        output_directory=acquisition,
    )
    assert receipt["qualification_mode"] == "diagnostic_nonqualifying"
    assert receipt["credential_profile"]["credential_present"] is False
    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="diagnostic acquisitions"
    ):
        MODULE.finalize_discovery(
            plan,
            acquisition_directories=[acquisition],
            output_directory=tmp_path / "fake-final",
        )


def test_repository_search_identity_must_match_visibility_lookup(
    tmp_path: Path,
) -> None:
    plan = _plan(maximum=10)
    plan["queries"][0]["endpoint"] = "search/repositories"

    def search(url: str, params: dict[str, str]) -> Any:
        return _response(
            MODULE,
            url,
            params,
            total=1,
            items=[
                {
                    "id": 9999,
                    "node_id": "R_stale",
                    "full_name": "acme/public-repo",
                    "url": "https://api.github.com/repos/acme/public-repo",
                }
            ],
        )

    with pytest.raises(MODULE.PopulationDiscoveryError, match="identity changed"):
        MODULE.freeze_acquisition(
            plan,
            acquisition_index=1,
            transport=search,
            visibility_transport=_visibility_transport,
            output_directory=tmp_path / "identity-mismatch",
        )


def test_response_size_and_redirects_fail_closed(tmp_path: Path) -> None:
    plan = _plan(maximum=10)
    plan["max_response_bytes"] = 1024
    endpoint = "https://api.github.com/search/issues"
    response = _response(
        MODULE,
        endpoint,
        {
            "q": (
                '"security" is:pr is:merged is:public '
                "merged:2026-01-01T00:00:00Z..2026-01-01T00:00:03Z"
            ),
            "per_page": "100",
            "page": "1",
            "sort": "created",
            "order": "asc",
        },
        total=0,
        items=[],
    )
    oversized = MODULE.FrozenResponse(
        url=response.url,
        status=response.status,
        headers=response.headers,
        body=b"x" * 1025,
        requested_url=response.requested_url,
        redirect_chain=(),
        request_started_at_utc=response.request_started_at_utc,
        request_completed_at_utc=response.request_completed_at_utc,
    )
    with pytest.raises(MODULE.PopulationDiscoveryError, match="locked byte limit"):
        MODULE.freeze_acquisition(
            plan,
            acquisition_index=1,
            transport=lambda _url, _params: oversized,
            visibility_transport=_visibility_transport,
            output_directory=tmp_path / "oversized",
        )

    with pytest.raises(MODULE.PopulationDiscoveryError, match="unapproved redirect"):
        MODULE._RejectRedirects().redirect_request(
            MODULE.Request(endpoint),
            None,
            302,
            "Found",
            {},
            "https://example.com/credential-forward",
        )


def test_final_publication_is_atomic_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(maximum=10)
    live = _live_transport(
        monkeypatch,
        plan,
        lambda url, params: _response(MODULE, url, params, total=0, items=[]),
    )
    acquisition = tmp_path / "acquisition-for-retry"
    MODULE.freeze_acquisition(
        plan,
        acquisition_index=1,
        transport=live,
        visibility_transport=live,
        output_directory=acquisition,
    )
    target = tmp_path / "atomic-final"
    original_reload = MODULE._reload_final_receipt

    def fail_reload(*args: Any, **kwargs: Any) -> Any:
        raise MODULE.PopulationDiscoveryError("injected final reload failure")

    monkeypatch.setattr(MODULE, "_reload_final_receipt", fail_reload)
    with pytest.raises(MODULE.PopulationDiscoveryError, match="injected"):
        MODULE.finalize_discovery(
            plan,
            acquisition_directories=[acquisition],
            output_directory=target,
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".atomic-final.staging-*"))

    monkeypatch.setattr(MODULE, "_reload_final_receipt", original_reload)
    receipt = MODULE.finalize_discovery(
        plan,
        acquisition_directories=[acquisition],
        output_directory=target,
    )
    assert target.is_dir()
    assert receipt["status"] == "frozen_candidate_discovery"


def test_final_reload_recomputes_raw_page_and_shard_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(maximum=10)
    output = tmp_path / "tampered-final-ledger"
    _freeze_live_discovery(
        monkeypatch,
        plan,
        search_transport=lambda url, params: _response(
            MODULE,
            url,
            params,
            total=1,
            items=[{"id": 1, "node_id": "PR_1"}],
        ),
        output_directory=output,
    )
    receipt_path = output / "query-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["shards"][0]["page_count"] = 2
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = MODULE.stable_digest(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="terminal shard is invalid"
    ):
        MODULE._reload_final_receipt(
            output, normalized=MODULE.validate_query_plan(plan)
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda receipt: receipt["acquisition_receipts"][0].update(
                {"receipt_sha256": "f" * 64}
            ),
            "acquisition receipt lineage does not recompute",
        ),
        (
            lambda receipt: receipt.update(
                {"limitations": ["This is a complete population frame."]}
            ),
            "final limitations disagree",
        ),
    ],
)
def test_final_reload_rejects_rehashed_lineage_or_claim_boundary_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    message: str,
) -> None:
    plan = _plan(maximum=10)
    output = tmp_path / message.replace(" ", "-")
    _freeze_live_discovery(
        monkeypatch,
        plan,
        search_transport=lambda url, params: _response(
            MODULE,
            url,
            params,
            total=1,
            items=[{"id": 1, "node_id": "PR_1"}],
        ),
        output_directory=output,
    )
    receipt_path = output / "query-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutate(receipt)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = MODULE.stable_digest(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)

    with pytest.raises(MODULE.PopulationDiscoveryError, match=message):
        MODULE._reload_final_receipt(
            output, normalized=MODULE.validate_query_plan(plan)
        )


def test_final_reload_requires_the_exact_recursive_split_probe_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(maximum=2)

    def transport(url: str, params: dict[str, str]) -> Any:
        query = params["q"]
        if "00:00:00Z..2026-01-01T00:00:03Z" in query:
            items = [{"id": value, "node_id": f"PR_{value}"} for value in range(1, 5)]
        elif "00:00:00Z..2026-01-01T00:00:01Z" in query:
            items = [
                {"id": 1, "node_id": "PR_1"},
                {"id": 2, "node_id": "PR_2"},
            ]
        else:
            items = [
                {"id": 3, "node_id": "PR_3"},
                {"id": 4, "node_id": "PR_4"},
            ]
        return _response(MODULE, url, params, total=len(items), items=items)

    output = tmp_path / "missing-root-probe"
    _freeze_live_discovery(
        monkeypatch,
        plan,
        search_transport=transport,
        output_directory=output,
    )
    receipt_path = output / "query-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    root = next(
        response
        for response in receipt["responses"]
        if response["shard_start_utc"] == "2026-01-01T00:00:00Z"
        and response["shard_end_utc"] == "2026-01-01T00:00:03Z"
    )
    receipt["responses"].remove(root)
    receipt["response_count"] -= 1
    receipt["acquisition_windows"][0]["response_count"] -= 1
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = MODULE.stable_digest(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)

    with pytest.raises(
        MODULE.PopulationDiscoveryError, match="missing its page-one probe"
    ):
        MODULE._reload_final_receipt(
            output, normalized=MODULE.validate_query_plan(plan)
        )


def test_final_publication_recovers_after_post_rename_parent_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(maximum=10)
    acquisition = tmp_path / "post-rename-acquisition"
    live = _live_transport(
        monkeypatch,
        plan,
        lambda url, params: _response(MODULE, url, params, total=0, items=[]),
    )
    MODULE.freeze_acquisition(
        plan,
        acquisition_index=1,
        transport=live,
        visibility_transport=live,
        output_directory=acquisition,
    )
    target = tmp_path / "post-rename-final"
    original_fsync_directory = MODULE._fsync_directory
    calls = 0

    def fail_after_rename(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected post-rename parent fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(MODULE, "_fsync_directory", fail_after_rename)
    with pytest.raises(MODULE.PublicationRecoveryRequired):
        MODULE.finalize_discovery(
            plan,
            acquisition_directories=[acquisition],
            output_directory=target,
        )
    assert target.is_dir()
    assert MODULE._final_publication_marker(target).is_file()

    monkeypatch.setattr(MODULE, "_fsync_directory", original_fsync_directory)
    receipt = MODULE.finalize_discovery(
        plan,
        acquisition_directories=[acquisition],
        output_directory=target,
    )
    assert receipt["status"] == "frozen_candidate_discovery"
    assert not MODULE._final_publication_marker(target).exists()
    assert not list(tmp_path.glob(".post-rename-final.staging-*"))


def test_persisted_raw_response_size_is_checked_before_reading(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "oversized-private-response.json"
    raw.write_bytes(b"x" * 1025)
    raw.chmod(0o600)
    with pytest.raises(MODULE.PopulationDiscoveryError, match="locked byte limit"):
        MODULE._read_private(
            raw,
            name="persisted raw response",
            maximum_bytes=1024,
        )
