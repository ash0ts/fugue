#!/usr/bin/env python3
"""Verify that Fugue can resolve one seeded Northstar call through Weave."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

RESEARCH_ID = "aria-support-data-authority-v1"
REVIEW_CALL_ID = "d4123e2f-c414-57f9-9080-86642302b838"


def _env_value(path: Path, key: str) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'\"")
    raise RuntimeError(f"{key} is required in {path}")


def _request(
    base_url: str,
    api_key: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=(json.dumps(body).encode() if body is not None else None),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("Fugue readiness response was not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--entity")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    entity = str(args.entity or "").strip()
    if not entity and args.env_file is not None:
        entity = _env_value(args.env_file, "WANDB_ENTITY")
    if not entity:
        parser.error("--entity or an --env-file containing WANDB_ENTITY is required")
    project = f"{entity}/northstar-support-agent"
    try:
        _request(args.base_url, api_key, "GET", f"/v1/research/{RESEARCH_ID}")
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read()) if exc.code == 400 else {}
        error = body.get("error") if isinstance(body, dict) else None
        is_missing = exc.code == 404 or (
            exc.code == 400
            and isinstance(error, dict)
            and error.get("code") == "study_not_found"
        )
        if not is_missing:
            raise
        _request(
            args.base_url,
            api_key,
            "POST",
            "/v1/research",
            {
                "research_id": RESEARCH_ID,
                "title": "Northstar support readiness probe",
                "campaign_id": "support-data-authority-v1",
                "question": "Can Fugue query the seeded synthetic support project?",
                "background": "Read-only local-demo readiness check.",
                "idempotency_key": f"create-{RESEARCH_ID}",
            },
        )
    draft = {
        "schema_version": 1,
        "study_id": RESEARCH_ID,
        "source_id": "northstar-support-agent",
        "objective": "Verify one seeded synthetic call is queryable.",
        "fields": ["status", "operation"],
        "filters": {},
        "max_traces": 1,
        "selection": {
            "schema_version": 1,
            "project": project,
            "mode": "selected",
            "call_ids": [REVIEW_CALL_ID],
            "filters": {},
            "max_traces": 1,
        },
    }
    preview = _request(
        args.base_url,
        api_key,
        "POST",
        f"/v1/research/{RESEARCH_ID}/trace-audits:preview",
        {"draft": draft},
    )
    audit = _request(
        args.base_url,
        api_key,
        "POST",
        f"/v1/research/{RESEARCH_ID}/trace-audits",
        {"preview": preview, "idempotency_key": f"query-{RESEARCH_ID}"},
    )
    if audit.get("cohort_count") != 1:
        raise RuntimeError("Fugue did not resolve exactly one seeded support trace")
    print(
        json.dumps(
            {
                "status": "ready",
                "research_id": RESEARCH_ID,
                "project": project,
                "trace_audit_id": audit["id"],
                "source_snapshot_digest": audit["source_snapshot_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
