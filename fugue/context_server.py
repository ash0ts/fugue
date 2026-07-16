from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fugue.bench.context import (
    ContextRuntime,
    PreparedContext,
    RetrievalQuery,
    get_context_system,
    query_context,
)
from fugue.redaction import redact_text, secrets_from_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fugue.context_server")
    parser.add_argument("--system", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError('install context dependencies with: uv pip install -e ".[context]"') from exc

    repo_root = args.repo_root.resolve()
    spec = get_context_system(args.system, _fugue_repo_root(repo_root))
    manifest_path = args.prepared / "context-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    prepared = PreparedContext(
        system_id=spec.id,
        cache_key=str(manifest.get("cache_key") or "trial"),
        path=args.prepared.resolve(),
        manifest=manifest,
        metrics=dict(manifest.get("metrics") or {}),
    )
    runtime = ContextRuntime(
        repo_root=repo_root,
        cache_root=args.prepared.resolve().parent,
        env=dict(os.environ),
    )
    events_path = Path(
        os.environ.get(
            "FUGUE_CONTEXT_EVENTS_PATH",
            "/logs/artifacts/fugue-context-events.jsonl",
        )
    )
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.touch(exist_ok=True)
    server = FastMCP(
        f"Fugue context: {spec.title}",
        host=args.host,
        port=args.port,
    )

    def retrieve(query: str, top_k: int) -> dict[str, Any]:
        started_at = time.perf_counter()
        bounded_top_k = max(1, min(top_k, 50))
        bounded_query = query[:4_000]
        hits, metrics = asyncio.run(
            query_context(
                spec,
                RetrievalQuery(id="agent", text=bounded_query, top_k=bounded_top_k),
                prepared,
                runtime,
            )
        )
        remaining = 64_000
        payload: list[dict[str, Any]] = []
        for hit in hits:
            text = (hit.text or "")[: min(8_000, remaining)]
            remaining -= len(text)
            payload.append(
                {
                    "path": hit.path,
                    "start_line": hit.start_line,
                    "end_line": hit.end_line,
                    "score": hit.score,
                    "text": text,
                    "metadata": hit.metadata,
                }
            )
            if remaining <= 0:
                break
        _record_event(
            events_path,
            {
                "event": "retrieve",
                "layer": "provider",
                "logical_request_id": uuid.uuid4().hex,
                "elapsed_ms": (time.perf_counter() - started_at) * 1_000,
                "context_system_id": spec.id,
                "query": redact_text(bounded_query[:1_000], secrets_from_env(os.environ)),
                "top_k": bounded_top_k,
                "metrics": metrics,
                "hits": [
                    {"path": hit.path, "score": hit.score} for hit in hits[:20]
                ],
            },
        )
        return {
            "ok": True,
            "context_system_id": spec.id,
            "hits": payload,
            "metrics": metrics,
        }

    @server.tool(
        name="context_search",
        description="Search the configured repository context system for relevant evidence.",
    )
    def context_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return retrieve(query, top_k)["hits"]

    if args.transport == "streamable-http":
        _start_portable_server(args.host, args.port + 1, spec.id, retrieve)
    server.run(transport=args.transport)
    return 0


def _record_event(path: Path, event: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    except OSError:
        return


def _start_portable_server(
    host: str,
    port: int,
    system_id: str,
    retrieve: Any,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            self._send(
                HTTPStatus.OK,
                {"ok": True, "context_system_id": system_id, "delivery": "portable"},
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/query":
                self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 16_384:
                    raise ValueError("query body must be between 1 byte and 16 KiB")
                value = json.loads(self.rfile.read(size))
                query = str(value.get("query") or "").strip()
                if not query:
                    raise ValueError("query is required")
                result = retrieve(query, int(value.get("top_k") or 10))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                )
                return
            self._send(HTTPStatus.OK, result)

        def _send(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            body = json.dumps(value, sort_keys=True, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _fugue_repo_root(cwd: Path) -> Path:
    configured = os.environ.get("FUGUE_REPO_ROOT", "").strip()
    if configured:
        return Path(configured)
    package_root = Path(__file__).resolve().parent.parent
    if (package_root / "configs" / "fugue" / "context-systems").is_dir():
        return package_root
    return cwd


if __name__ == "__main__":
    raise SystemExit(main())
