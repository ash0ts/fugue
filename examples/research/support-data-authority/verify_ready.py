#!/usr/bin/env python3
"""Verify that Fugue can resolve one seeded Northstar call through Weave."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fugue.bench.candidates import stable_digest
from fugue.bench.library import validate_id
from fugue.research.agent_contracts import trace_audit_draft_from_dict
from fugue.research.traces import TraceSourceRegistry

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-sources-file", type=Path, required=True)
    parser.add_argument("--trace-api-key-file", type=Path, required=True)
    parser.add_argument("--trace-server-url", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--research-id", required=True)
    args = parser.parse_args()
    research_id = validate_id(args.research_id, kind="research id")
    entity = str(args.entity or "").strip()
    if not entity and args.env_file is not None:
        entity = _env_value(args.env_file, "WANDB_ENTITY")
    if not entity:
        parser.error("--entity or an --env-file containing WANDB_ENTITY is required")
    project = f"{entity}/northstar-support-agent"
    registry = TraceSourceRegistry.from_file(
        args.trace_sources_file,
        env={
            "WANDB_ENTITY": entity,
            "WANDB_API_KEY_FILE": str(args.trace_api_key_file),
            "WF_TRACE_SERVER_URL": args.trace_server_url,
        },
    )
    source = registry.get("northstar-support-agent")
    draft = trace_audit_draft_from_dict(
        {
            "schema_version": 1,
            "study_id": research_id,
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
        },
        require_digest=False,
    )
    records = source.read(draft)
    if len(records) != 1:
        raise RuntimeError("Fugue did not resolve exactly one seeded support trace")
    print(
        json.dumps(
            {
                "status": "ready",
                "research_id": research_id,
                "project": project,
                "source_digest": source.source.digest,
                "source_snapshot_digest": stable_digest(list(records)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
