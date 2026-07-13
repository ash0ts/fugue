from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO

from fugue.redaction import redact_text, secrets_from_env

_MAX_TEXT = 1_000
_MAX_EVENT_BYTES = 16_384
_ENV_SECRETS = secrets_from_env(os.environ)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fugue.mcp_proxy")
    parser.add_argument("--name", required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an upstream command is required after --")

    process = subprocess.Popen(
        command,
        cwd=args.cwd,
        env=dict(os.environ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("failed to open MCP proxy pipes")

    recorder = _Recorder(
        name=args.name,
        path=Path(
            os.environ.get(
                "FUGUE_CONTEXT_EVENTS_PATH",
                "/logs/artifacts/fugue-context-events.jsonl",
            )
        ),
    )
    request_thread = threading.Thread(
        target=_relay_requests,
        args=(sys.stdin.buffer, process.stdin, recorder),
        daemon=True,
    )
    response_thread = threading.Thread(
        target=_relay_responses,
        args=(process.stdout, sys.stdout.buffer, recorder),
        daemon=True,
    )
    request_thread.start()
    response_thread.start()
    return_code = process.wait()
    request_thread.join(timeout=1)
    response_thread.join(timeout=1)
    return return_code


class _Recorder:
    def __init__(self, *, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.started_at = time.perf_counter()
        self.pending: dict[str, tuple[float, str | None]] = {}
        self.lock = threading.Lock()

    def request(self, payload: dict[str, Any], size: int) -> None:
        request_id = _request_id(payload)
        method = str(payload.get("method") or "")
        tool = None
        if method == "tools/call":
            params = payload.get("params") or {}
            tool = str(params.get("name") or "") if isinstance(params, dict) else None
        if request_id:
            with self.lock:
                self.pending[request_id] = (time.perf_counter(), tool)
        if method == "tools/call":
            self.write(
                {
                    "event": "mcp_tool_request",
                    "layer": "proxy",
                    "server": self.name,
                    "tool": tool,
                    "request_id": request_id,
                    "request_bytes": size,
                    "arguments": _sanitize((payload.get("params") or {}).get("arguments")),
                }
            )

    def response(self, payload: dict[str, Any], size: int) -> None:
        request_id = _request_id(payload)
        with self.lock:
            pending = self.pending.pop(request_id, None) if request_id else None
        if pending is None:
            return
        started, tool = pending
        if not tool:
            return
        self.write(
            {
                "event": "mcp_tool_response",
                "layer": "upstream",
                "server": self.name,
                "tool": tool,
                "request_id": request_id,
                "response_bytes": size,
                "latency_ms": (time.perf_counter() - started) * 1_000,
                "error": _sanitize(payload.get("error")),
            }
        )

    def write(self, event: dict[str, Any]) -> None:
        event.setdefault("elapsed_ms", (time.perf_counter() - self.started_at) * 1_000)
        line = json.dumps(event, sort_keys=True, default=str)
        if len(line.encode()) > _MAX_EVENT_BYTES:
            event = {
                key: value
                for key, value in event.items()
                if key not in {"arguments", "error"}
            }
            event["payload_truncated"] = True
            line = json.dumps(event, sort_keys=True, default=str)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


def _relay_requests(source: BinaryIO, target: BinaryIO, recorder: _Recorder) -> None:
    _relay(source, target, recorder.request)


def _relay_responses(source: BinaryIO, target: BinaryIO, recorder: _Recorder) -> None:
    _relay(source, target, recorder.response)


def _relay(
    source: BinaryIO,
    target: BinaryIO,
    observe: Any,
) -> None:
    while line := source.readline():
        target.write(line)
        target.flush()
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict):
            observe(payload, len(line))


def _request_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("id")
    return str(value) if value is not None else None


def _sanitize(value: Any, *, key: str = "") -> Any:
    if _sensitive_key(key):
        return "[redacted]"
    if isinstance(value, dict):
        return {str(name): _sanitize(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, key=key) for item in value[:50]]
    if isinstance(value, str):
        return redact_text(value[:_MAX_TEXT], _ENV_SECRETS)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:_MAX_TEXT]


def _sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return (
        normalized in {"authorization", "password", "secret", "token", "apikey"}
        or "api_key" in normalized
        or normalized.endswith(
            ("_access_token", "_refresh_token", "_auth_token", "_password", "_secret")
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
