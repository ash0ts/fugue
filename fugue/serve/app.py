from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from ag_ui.core import (
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from fugue.serve.runtime import (
    HarborWorkerBackend,
    WorkerBackend,
    WorkerRequest,
    new_request_id,
)


def create_app(
    *,
    deployment_path: Path | None = None,
    backend: WorkerBackend | None = None,
    env: dict[str, str] | None = None,
    heartbeat_sec: float = 15.0,
    timeout_sec: float | None = None,
) -> FastAPI:
    selected_env = dict(os.environ if env is None else env)
    if backend is None:
        selected_path = deployment_path or Path(
            selected_env.get(
                "FUGUE_DEPLOYMENT_SPEC",
                "/opt/fugue/deployment/deployment.json",
            )
        )
        backend = HarborWorkerBackend(selected_path, env=selected_env)
    deployment = backend.deployment
    timeout = timeout_sec or float(
        selected_env.get(
            "FUGUE_SERVE_TIMEOUT_SEC",
            deployment.get("resources", {}).get("timeout_sec", 900),
        )
    )
    model_id = selected_env.get("FUGUE_SERVE_MODEL_ID") or (
        f"fugue-{str(deployment['candidate_id'])[:16]}"
    )
    api_key = selected_env.get("FUGUE_SERVE_API_KEY", "")
    cors_origins = _csv(selected_env.get("FUGUE_SERVE_CORS_ORIGINS"))

    app = FastAPI(
        title="Fugue Candidate Service",
        version=str(deployment.get("deployment_id") or "unknown"),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.method != "OPTIONS" and request.url.path not in {
            "/healthz",
            "/readyz",
        }:
            authorization = request.headers.get("authorization", "")
            expected = f"Bearer {api_key}"
            if not api_key or not hmac.compare_digest(authorization, expected):
                return _error(
                    "invalid or missing bearer token",
                    code="invalid_api_key",
                    status=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "deployment_id": deployment["deployment_id"]}

    @app.get("/readyz")
    async def readyz():
        ready, missing = backend.readiness()
        return JSONResponse(
            {
                "status": "ready" if ready else "not_ready",
                "deployment_id": deployment["deployment_id"],
                "missing": list(missing),
            },
            status_code=200 if ready else 503,
        )

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "fugue",
                }
            ],
        }

    @app.post("/v1/responses/compact")
    async def compact_response():
        return _unsupported("compact responses are not supported", "compact")

    @app.websocket("/v1/responses")
    async def responses_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        authorization = websocket.headers.get("authorization", "")
        if not api_key or not hmac.compare_digest(
            authorization, f"Bearer {api_key}"
        ):
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                        "message": "invalid or missing bearer token",
                        "param": None,
                    },
                }
            )
        else:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "unsupported_feature",
                        "message": "WebSocket responses are not supported",
                        "param": "websocket",
                    },
                }
            )
        await websocket.close(code=1008)

    @app.post("/v1/responses")
    async def responses(request: Request):
        payload = await _json_body(request)
        error = _responses_unsupported(payload)
        if error:
            return error
        try:
            messages = _responses_messages(payload.get("input"))
            instructions = payload.get("instructions")
            if instructions is not None and not isinstance(instructions, str):
                raise ValueError("instructions must be text")
            if isinstance(instructions, str) and instructions.strip():
                messages = (
                    {"role": "developer", "content": instructions},
                    *messages,
                )
        except ValueError as exc:
            return _unsupported(str(exc), "input")
        request_id = new_request_id("resp")
        worker_request = WorkerRequest(request_id, "open-responses", messages)
        if payload.get("stream") is True:
            return StreamingResponse(
                _responses_stream(
                    backend,
                    worker_request,
                    model_id=model_id,
                    heartbeat_sec=heartbeat_sec,
                    timeout_sec=timeout,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            answer = await _run_backend(backend, worker_request, timeout)
        except TimeoutError:
            return _error("worker timed out", code="worker_timeout", status=504)
        except Exception:
            return _error("worker failed", code="worker_error", status=500)
        return _response_object(request_id, model_id, answer)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await _json_body(request)
        error = _chat_unsupported(payload)
        if error:
            return error
        try:
            messages = _message_list(payload.get("messages"))
        except ValueError as exc:
            return _unsupported(str(exc), "messages")
        request_id = new_request_id("chatcmpl")
        worker_request = WorkerRequest(request_id, "chat-completions", messages)
        if payload.get("stream") is True:
            return StreamingResponse(
                _chat_stream(
                    backend,
                    worker_request,
                    model_id=model_id,
                    heartbeat_sec=heartbeat_sec,
                    timeout_sec=timeout,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            answer = await _run_backend(backend, worker_request, timeout)
        except TimeoutError:
            return _error("worker timed out", code="worker_timeout", status=504)
        except Exception:
            return _error("worker failed", code="worker_error", status=500)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.post("/ag-ui")
    async def ag_ui(request: Request):
        payload = await _json_body(request)
        if payload.get("tools"):
            return _unsupported("client-provided tools are not supported", "tools")
        try:
            messages = _message_list(payload.get("messages"))
        except ValueError as exc:
            return _unsupported(str(exc), "messages")
        run_id = str(payload.get("runId") or new_request_id("run"))
        thread_id = str(payload.get("threadId") or new_request_id("thread"))
        worker_request = WorkerRequest(
            new_request_id("agui"), "ag-ui", messages
        )
        return StreamingResponse(
            _ag_ui_stream(
                backend,
                worker_request,
                run_id=run_id,
                thread_id=thread_id,
                heartbeat_sec=heartbeat_sec,
                timeout_sec=timeout,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


async def _run_backend(
    backend: WorkerBackend, request: WorkerRequest, timeout_sec: float
) -> str:
    async with asyncio.timeout(timeout_sec):
        return await backend.run(request)


async def _wait_for_answer(
    backend: WorkerBackend,
    request: WorkerRequest,
    *,
    heartbeat_sec: float,
    timeout_sec: float,
) -> AsyncIterator[str | None]:
    async def bounded() -> str:
        return await _run_backend(backend, request, timeout_sec)

    task = asyncio.create_task(bounded())
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=heartbeat_sec)
            if not done:
                yield None
        yield task.result()
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def _responses_stream(
    backend: WorkerBackend,
    request: WorkerRequest,
    *,
    model_id: str,
    heartbeat_sec: float,
    timeout_sec: float,
) -> AsyncIterator[str]:
    response = _response_object(request.request_id, model_id, "", status="in_progress")
    yield _sse(
        {"type": "response.created", "sequence_number": 0, "response": response}
    )
    yield _sse(
        {
            "type": "response.in_progress",
            "sequence_number": 1,
            "response": response,
        }
    )
    try:
        async for value in _wait_for_answer(
            backend,
            request,
            heartbeat_sec=heartbeat_sec,
            timeout_sec=timeout_sec,
        ):
            if value is None:
                yield ": heartbeat\n\n"
                continue
            answer = value
            message_id = f"msg_{request.request_id.split('_', 1)[-1]}"
            pending_item = {
                "id": message_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": "final_answer",
                "content": [],
            }
            yield _sse(
                {
                    "type": "response.output_item.added",
                    "sequence_number": 2,
                    "output_index": 0,
                    "item": pending_item,
                }
            )
            yield _sse(
                {
                    "type": "response.content_part.added",
                    "sequence_number": 3,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                    },
                }
            )
            yield _sse(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 4,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": answer,
                }
            )
            yield _sse(
                {
                    "type": "response.output_text.done",
                    "sequence_number": 5,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": answer,
                }
            )
            completed_part = {
                "type": "output_text",
                "text": answer,
                "annotations": [],
            }
            completed_item = {
                **pending_item,
                "status": "completed",
                "content": [completed_part],
            }
            yield _sse(
                {
                    "type": "response.content_part.done",
                    "sequence_number": 6,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": completed_part,
                }
            )
            yield _sse(
                {
                    "type": "response.output_item.done",
                    "sequence_number": 7,
                    "output_index": 0,
                    "item": completed_item,
                }
            )
            yield _sse(
                {
                    "type": "response.completed",
                    "sequence_number": 8,
                    "response": _response_object(request.request_id, model_id, answer),
                }
            )
    except Exception as exc:
        code = "worker_timeout" if isinstance(exc, TimeoutError) else "worker_error"
        yield _sse(
            {
                "type": "error",
                "sequence_number": 2,
                "error": {
                    "type": "server_error",
                    "code": code,
                    "message": "worker timed out" if code == "worker_timeout" else "worker failed",
                },
            }
        )


async def _chat_stream(
    backend: WorkerBackend,
    request: WorkerRequest,
    *,
    model_id: str,
    heartbeat_sec: float,
    timeout_sec: float,
) -> AsyncIterator[str]:
    created = int(time.time())
    base = {
        "id": request.request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
    }
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    try:
        async for value in _wait_for_answer(
            backend,
            request,
            heartbeat_sec=heartbeat_sec,
            timeout_sec=timeout_sec,
        ):
            if value is None:
                yield ": heartbeat\n\n"
                continue
            yield _sse(
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": {"content": value}, "finish_reason": None}
                    ],
                }
            )
            yield _sse(
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                }
            )
            yield "data: [DONE]\n\n"
    except Exception as exc:
        code = "worker_timeout" if isinstance(exc, TimeoutError) else "worker_error"
        yield _sse(
            {
                "error": {
                    "type": "server_error",
                    "code": code,
                    "message": "worker timed out" if code == "worker_timeout" else "worker failed",
                }
            }
        )


async def _ag_ui_stream(
    backend: WorkerBackend,
    request: WorkerRequest,
    *,
    run_id: str,
    thread_id: str,
    heartbeat_sec: float,
    timeout_sec: float,
) -> AsyncIterator[str]:
    message_id = new_request_id("msg")
    encoder = EventEncoder()
    yield encoder.encode(RunStartedEvent(threadId=thread_id, runId=run_id))
    try:
        async for value in _wait_for_answer(
            backend,
            request,
            heartbeat_sec=heartbeat_sec,
            timeout_sec=timeout_sec,
        ):
            if value is None:
                yield ": heartbeat\n\n"
                continue
            yield encoder.encode(
                TextMessageStartEvent(messageId=message_id, role="assistant")
            )
            yield encoder.encode(
                TextMessageContentEvent(messageId=message_id, delta=value)
            )
            yield encoder.encode(TextMessageEndEvent(messageId=message_id))
            yield encoder.encode(
                RunFinishedEvent(threadId=thread_id, runId=run_id)
            )
    except Exception as exc:
        yield encoder.encode(
            RunErrorEvent(
                message=(
                    "worker timed out"
                    if isinstance(exc, TimeoutError)
                    else "worker failed"
                ),
                code=(
                    "worker_timeout" if isinstance(exc, TimeoutError) else "worker_error"
                ),
            )
        )


def _response_object(
    request_id: str,
    model_id: str,
    answer: str,
    *,
    status: str = "completed",
) -> dict[str, Any]:
    message_id = f"msg_{request_id.split('_', 1)[-1]}"
    output = []
    if answer or status == "completed":
        output = [
            {
                "id": message_id,
                "type": "message",
                "status": status,
                "role": "assistant",
                "phase": "final_answer",
                "content": [
                    {"type": "output_text", "text": answer, "annotations": []}
                ],
            }
        ]
    return {
        "id": request_id,
        "object": "response",
        "created_at": int(time.time()),
        "completed_at": int(time.time()) if status == "completed" else None,
        "status": status,
        "incomplete_details": None,
        "model": model_id,
        "previous_response_id": None,
        "instructions": None,
        "output": output,
        "error": None,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
        "truncation": "disabled",
        "text": {"format": {"type": "text"}},
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_logprobs": 0,
        "temperature": 1,
        "reasoning": {"effort": None, "summary": None},
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
        "max_output_tokens": None,
        "max_tool_calls": None,
        "store": False,
        "background": False,
        "service_tier": "default",
        "metadata": {},
        "safety_identifier": None,
        "prompt_cache_key": None,
    }


def _responses_messages(value: Any) -> tuple[dict[str, str], ...]:
    if isinstance(value, str) and value.strip():
        return ({"role": "user", "content": value},)
    return _message_list(value)


def _message_list(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("a non-empty text message history is required")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("message history must contain objects")
        role = str(item.get("role") or "")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role or 'missing'}")
        content = _text_content(item.get("content"))
        messages.append({"role": role, "content": content})
    return tuple(messages)


def _text_content(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value:
        parts: list[str] = []
        for item in value:
            if not isinstance(item, dict) or item.get("type") not in {
                "text",
                "input_text",
                "output_text",
            }:
                raise ValueError("media and non-text message content are not supported")
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError("text message content must contain text")
            parts.append(text)
        if any(part.strip() for part in parts):
            return "\n".join(parts)
    raise ValueError("messages must contain non-empty text")


def _responses_unsupported(payload: dict[str, Any]) -> JSONResponse | None:
    checks = (
        (bool(payload.get("tools")), "client-provided tools are not supported", "tools"),
        (payload.get("store") not in {None, False}, "persisted responses are not supported", "store"),
        (bool(payload.get("previous_response_id")), "previous_response_id is not supported", "previous_response_id"),
        (bool(payload.get("context_management")), "compact responses are not supported", "context_management"),
        (payload.get("background") is True, "background responses are not supported", "background"),
    )
    for rejected, message, parameter in checks:
        if rejected:
            return _unsupported(message, parameter)
    text = payload.get("text") or {}
    if not isinstance(text, dict):
        return _unsupported("text configuration must be an object", "text")
    text_format = text.get("format") or {}
    if not isinstance(text_format, dict):
        return _unsupported("text format must be an object", "text.format")
    if text_format and text_format.get("type") not in {None, "text"}:
        return _unsupported("structured response formats are not supported", "text.format")
    return None


def _chat_unsupported(payload: dict[str, Any]) -> JSONResponse | None:
    if payload.get("tools"):
        return _unsupported("client-provided tools are not supported", "tools")
    if payload.get("functions"):
        return _unsupported("client-provided functions are not supported", "functions")
    response_format = payload.get("response_format") or {}
    if not isinstance(response_format, dict):
        return _unsupported("response format must be an object", "response_format")
    if response_format and response_format.get("type") not in {None, "text"}:
        return _unsupported("structured response formats are not supported", "response_format")
    return None


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        value = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _unsupported(message: str, parameter: str) -> JSONResponse:
    return _error(
        message,
        code="unsupported_feature",
        status=400,
        parameter=parameter,
    )


def _error(
    message: str,
    *,
    code: str,
    status: int,
    parameter: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "param": parameter,
                "code": code,
            }
        },
        status_code=status,
        headers=headers,
    )


def _sse(value: dict[str, Any]) -> str:
    return f"data: {json.dumps(value, separators=(',', ':'))}\n\n"


def _csv(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]
