#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://fugue-context:8001"
MAX_RESPONSE_BYTES = 1_048_576


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fugue-context")
    parser.add_argument(
        "--url",
        default=os.environ.get("FUGUE_CONTEXT_QUERY_URL", DEFAULT_URL),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe")
    query = subparsers.add_parser("query")
    query.add_argument("--text", required=True)
    query.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        if args.command == "probe":
            payload = _request(f"{args.url.rstrip('/')}/health")
        else:
            started = time.perf_counter()
            payload = _request(
                f"{args.url.rstrip('/')}/query",
                {
                    "query": args.text[:4_000],
                    "top_k": max(1, min(args.top_k, 50)),
                },
            )
            _record_query(payload, args.text, args.top_k, started)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


def _request(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("context response exceeded 1 MiB")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("context response was not an object")
    return value


def _record_query(
    payload: dict[str, Any], query: str, top_k: int, started: float
) -> None:
    path = Path(
        os.environ.get(
            "FUGUE_CONTEXT_EVENTS_PATH",
            "/logs/artifacts/fugue-context-events.jsonl",
        )
    )
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    hits = payload.get("hits") if isinstance(payload.get("hits"), list) else []
    event = {
        "event": "retrieve",
        "layer": "portable_client",
        "logical_request_id": uuid.uuid4().hex,
        "elapsed_ms": (time.perf_counter() - started) * 1_000,
        "context_system_id": payload.get("context_system_id"),
        "query": _redact(query[:1_000]),
        "top_k": max(1, min(top_k, 50)),
        "metrics": metrics,
        "hits": [
            {key: hit.get(key) for key in ("path", "score") if hit.get(key) is not None}
            for hit in hits[:20]
            if isinstance(hit, dict)
        ],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    except OSError:
        pass


def _redact(value: str) -> str:
    secrets = {
        item
        for key, item in os.environ.items()
        if item and any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET"))
    }
    for secret in sorted(secrets, key=len, reverse=True):
        value = value.replace(secret, "[redacted]")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
