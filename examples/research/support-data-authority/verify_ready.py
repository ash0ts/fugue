#!/usr/bin/env python3
"""Verify that Fugue can resolve one seeded Northstar call through Weave."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

STUDY_ID = "northstar-support-ready-v1"


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
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    entity = _env_value(args.env_file, "WANDB_ENTITY")
    project = f"{entity}/northstar-support-agent"
    try:
        _request(args.base_url, api_key, "GET", f"/v1/studies/{STUDY_ID}")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        _request(
            args.base_url,
            api_key,
            "POST",
            "/v1/studies",
            {
                "study_id": STUDY_ID,
                "title": "Northstar support readiness probe",
                "campaign_id": "support-data-authority-v1",
                "question": "Can Fugue query the seeded synthetic support project?",
                "background": "Read-only local-demo readiness check.",
                "idempotency_key": "create-northstar-support-ready-v1",
            },
        )
    draft = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "source_id": "northstar-support-agent",
        "objective": "Verify one seeded synthetic call is queryable.",
        "fields": ["status", "operation"],
        "filters": {},
        "max_traces": 1,
        "selection": {
            "schema_version": 1,
            "project": project,
            "mode": "selected",
            "call_ids": ["northstar-support-07-root"],
            "filters": {},
            "max_traces": 1,
        },
    }
    preview = _request(
        args.base_url,
        api_key,
        "POST",
        f"/v1/studies/{STUDY_ID}/trace-audits:preview",
        {"draft": draft},
    )
    audit = _request(
        args.base_url,
        api_key,
        "POST",
        f"/v1/studies/{STUDY_ID}/trace-audits",
        {"preview": preview, "idempotency_key": "query-northstar-support-ready-v1"},
    )
    if audit.get("cohort_count") != 1:
        raise RuntimeError("Fugue did not resolve exactly one seeded support trace")
    print(
        json.dumps(
            {
                "status": "ready",
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
