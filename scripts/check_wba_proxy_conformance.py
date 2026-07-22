#!/usr/bin/env python3
"""Qualify one immutable LiteLLM proxy image against Fugue's Responses contract."""

from __future__ import annotations

import argparse
import json
import runpy
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import yaml

RUNNER = runpy.run_path("configs/fugue/runtime/wba-responses/wba-runner")


class _Provider(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        payload = json.loads(self.rfile.read(length))
        if not self.path.endswith("/chat/completions"):
            self.send_error(400)
            return
        if payload.get("stream") is not True:
            body = json.dumps(
                {
                    "id": "chatcmpl-conformance",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "shell",
                                            "arguments": '{"command":"pwd"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        chunks = [
            _chat_chunk({"role": "assistant", "content": None}),
            _chat_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"comm'},
                        }
                    ]
                }
            ),
            _chat_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": 'and":"pwd"}'},
                        }
                    ]
                }
            ),
            _chat_chunk({}, finish_reason="tool_calls"),
            {
                "id": "chatcmpl-conformance",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        body += "data: [DONE]\n\n"
        encoded = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *args: Any) -> None:
        del args


def _chat_chunk(
    delta: dict[str, Any], *, finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-conformance",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _events(base_url: str) -> list[dict[str, Any]]:
    payload = {
        "model": "test-model",
        "input": "Use the shell tool once.",
        "tools": [
            {
                "type": "function",
                "name": "shell",
                "description": "Run one command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            }
        ],
        "stream": True,
    }
    events: list[dict[str, Any]] = []
    with httpx.stream(
        "POST",
        f"{base_url}/v1/responses",
        headers={"Authorization": "Bearer test-key"},
        json=payload,
        timeout=30,
    ) as response:
        if not response.is_success:
            detail = response.read().decode(errors="replace")
            raise RuntimeError(
                f"proxy returned HTTP {response.status_code}: {detail[:4000]}"
            )
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            value = line.removeprefix("data:").strip()
            if value and value != "[DONE]":
                event = json.loads(value)
                if not isinstance(event, dict):
                    raise RuntimeError("proxy emitted a non-object event")
                events.append(event)
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="immutable image reference with @sha256 digest")
    args = parser.parse_args()
    if "@sha256:" not in args.image:
        raise SystemExit("image must be pinned by digest")
    provider_port = _free_port()
    proxy_port = _free_port()
    provider = ThreadingHTTPServer(("127.0.0.1", provider_port), _Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    container = f"fugue-wba-conformance-{proxy_port}"
    process: subprocess.Popen[str] | None = None
    try:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "litellm.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "model_list": [
                            {
                                "model_name": "test-model",
                                "litellm_params": {
                                    "model": "nebius/test-model",
                                    "api_base": (
                                        "http://host.docker.internal:"
                                        f"{provider_port}/v1"
                                    ),
                                    "api_key": "test-provider-key",
                                },
                            }
                        ],
                        "general_settings": {"master_key": "test-key"},
                    },
                    sort_keys=False,
                )
            )
            process = subprocess.Popen(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    container,
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    "-p",
                    f"127.0.0.1:{proxy_port}:4000",
                    "-v",
                    f"{config}:/app/config.yaml:ro",
                    args.image,
                    "--config",
                    "/app/config.yaml",
                    "--port",
                    "4000",
                    "--num_workers",
                    "1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            base_url = f"http://127.0.0.1:{proxy_port}"
            for _ in range(120):
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout else ""
                    raise RuntimeError(
                        f"proxy exited before readiness:\n{output[-4000:]}"
                    )
                try:
                    if httpx.get(f"{base_url}/health/liveliness", timeout=1).is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError("proxy did not become ready")
            events = _events(base_url)
            inspector = RUNNER["ResponsesStreamInspector"]()
            for event in events:
                inspector.observe(event)
            inspector.finish()
            result = {
                "image": args.image,
                "events": len(events),
                "anomaly_kinds": inspector.anomaly_kinds,
                "conformant": not inspector.anomaly_kinds,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["conformant"] else 1
    finally:
        provider.shutdown()
        provider.server_close()
        subprocess.run(
            ["docker", "rm", "-f", container],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
