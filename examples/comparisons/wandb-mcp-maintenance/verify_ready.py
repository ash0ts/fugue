#!/usr/bin/env python3
"""Verify the exact hosted MCP evidence is queryable through Fugue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fugue.bench.candidates import stable_digest
from fugue.bench.library import validate_id
from fugue.research.agent_contracts import trace_audit_draft_from_dict
from fugue.research.traces import TraceSourceRegistry

SOURCE_ID = "wandb-mcp-release-hosted-evidence"
PROJECT = "wandb/fugue-mcp-release-qualification-v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-sources-file", type=Path, required=True)
    parser.add_argument("--trace-api-key-file", type=Path, required=True)
    parser.add_argument("--trace-server-url", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="fugue-mcp-release-qualification-v1")
    parser.add_argument("--research-id", required=True)
    args = parser.parse_args()
    research_id = validate_id(args.research_id, kind="research id")
    project = f"{args.entity}/{args.project}"
    if project != PROJECT:
        raise RuntimeError(f"hosted evidence project must remain {PROJECT}")
    lock_path = Path(__file__).with_name("evidence.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("project") != PROJECT:
        raise RuntimeError("evidence lock project drifted")
    call_ids = [
        str(item.get("call_id") or "")
        for item in lock.get("objects", {}).get("source_conversations", [])
        if isinstance(item, dict) and item.get("call_id")
    ]
    if len(call_ids) != int(lock.get("counts", {}).get("source_conversations") or 0):
        raise RuntimeError("evidence lock conversation census is incomplete")
    registry = TraceSourceRegistry.from_file(
        args.trace_sources_file,
        env={
            "FUGUE_WEAVE_PROJECT": project,
            "WANDB_API_KEY_FILE": str(args.trace_api_key_file),
            "WF_TRACE_SERVER_URL": args.trace_server_url,
        },
    )
    source = registry.get(SOURCE_ID)
    draft = trace_audit_draft_from_dict(
        {
            "schema_version": 1,
            "study_id": research_id,
            "source_id": SOURCE_ID,
            "objective": "Verify one immutable hosted conversation is queryable.",
            "fields": ["status", "operation"],
            "filters": {},
            "max_traces": 1,
            "selection": {
                "schema_version": 1,
                "project": project,
                "mode": "selected",
                "call_ids": [call_ids[0]],
                "filters": {},
                "max_traces": 1,
            },
        },
        require_digest=False,
    )
    records = source.read(draft)
    if len(records) != 1:
        raise RuntimeError("the immutable hosted probe call was not queryable")
    print(
        json.dumps(
            {
                "status": "ready",
                "research_id": research_id,
                "project": project,
                "probe_call_id": call_ids[0],
                "source_digest": source.source.source_digest,
                "probe_snapshot_digest": stable_digest(list(records)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
