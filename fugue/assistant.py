from __future__ import annotations

import inspect
import json
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from fugue.model_plane import (
    ModelRoute,
    provider_request_headers,
    resolve_model_route,
    select_model,
    trace_project_slug,
)
from fugue.redaction import redact_value
from fugue.weave_support import initialize_weave

AssistantRole = Literal["composer", "analyst"]
ToolHandler = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class AssistantToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: tuple[AssistantToolCall, ...] = ()


@dataclass(frozen=True)
class AssistantTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler | None = None
    terminal: bool = False


@dataclass(frozen=True)
class AssistantUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class AssistantResponse:
    text: str
    tool_calls: tuple[AssistantToolCall, ...] = ()
    usage: AssistantUsage = field(default_factory=AssistantUsage)
    raw_id: str | None = None


@dataclass(frozen=True)
class AssistantRunResult:
    payload: dict[str, Any]
    messages: tuple[AssistantMessage, ...]
    usage: AssistantUsage
    model: str
    provider: str
    session_id: str


def select_assistant_model(
    role: AssistantRole,
    *,
    cli_model: str | None = None,
    saved_model: str | None = None,
    experiment_model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    values = dict(env or {})
    role_key = "FUGUE_COMPOSER_MODEL" if role == "composer" else "FUGUE_ANALYST_MODEL"
    for candidate in (
        cli_model,
        saved_model,
        values.get(role_key),
        experiment_model,
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return select_model(env=values)


class AssistantModelClient:
    def __init__(
        self,
        model: str,
        env: Mapping[str, str],
        *,
        timeout_sec: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.env = dict(env)
        self.route = resolve_model_route(model, self.env)
        self.timeout_sec = timeout_sec
        self.transport = transport

    async def complete(
        self,
        messages: Sequence[AssistantMessage],
        *,
        tools: Sequence[AssistantTool] = (),
        max_tokens: int = 4_096,
        temperature: float = 0,
    ) -> AssistantResponse:
        api_key = self.env.get(self.route.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{self.route.api_key_env} is required for assistant model "
                f"{self.route.display_model}"
            )
        async with httpx.AsyncClient(
            timeout=self.timeout_sec,
            transport=self.transport,
        ) as client:
            provider_headers = provider_request_headers(self.route, self.env)
            if self.route.provider == "openai" and self.route.responses_base_url:
                response = await client.post(
                    f"{self.route.responses_base_url}/responses",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        **provider_headers,
                    },
                    json=_responses_payload(
                        self.route,
                        messages,
                        tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                response.raise_for_status()
                return _parse_responses(response.json())
            if self.route.messages_base_url:
                response = await client.post(
                    f"{self.route.messages_base_url}/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                        **provider_headers,
                    },
                    json=_messages_payload(
                        self.route,
                        messages,
                        tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                response.raise_for_status()
                return _parse_messages(response.json())
            if not self.route.chat_base_url:
                raise RuntimeError(
                    f"assistant model route has no supported endpoint: {self.route.display_model}"
                )
            response = await client.post(
                f"{self.route.chat_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    **provider_headers,
                },
                json=_chat_payload(
                    self.route,
                    messages,
                    tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            response.raise_for_status()
            return _parse_chat(response.json())


class AssistantAgent:
    def __init__(
        self,
        client: AssistantModelClient,
        *,
        role: AssistantRole,
        tools: Sequence[AssistantTool],
        env: Mapping[str, str],
        trace_content: str = "full",
        session_id: str | None = None,
        max_rounds: int = 8,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.role = role
        self.tools = tuple(tools)
        self.env = dict(env)
        self.trace_content = trace_content
        self.session_id = session_id or uuid.uuid4().hex
        self.max_rounds = max_rounds
        self.attributes = dict(attributes or {})

    async def run(
        self,
        messages: Sequence[AssistantMessage],
    ) -> AssistantRunResult:
        history = list(messages)
        input_tokens = 0
        output_tokens = 0
        tracer = _AssistantTrace(
            role=self.role,
            route=self.client.route,
            env=self.env,
            trace_content=self.trace_content,
            session_id=self.session_id,
            attributes=self.attributes,
        )
        user_message = next(
            (message.content for message in reversed(history) if message.role == "user"),
            "",
        )
        tracer.start(user_message)
        try:
            for _ in range(self.max_rounds):
                llm = tracer.start_llm(history)
                try:
                    response = await self.client.complete(history, tools=self.tools)
                except BaseException as exc:
                    tracer.finish_llm(llm, error=exc)
                    raise
                tracer.finish_llm(llm, response=response)
                if response.usage.input_tokens is not None:
                    input_tokens += response.usage.input_tokens
                if response.usage.output_tokens is not None:
                    output_tokens += response.usage.output_tokens
                history.append(
                    AssistantMessage(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )
                if not response.tool_calls:
                    try:
                        payload = _json_object(response.text)
                    except ValueError:
                        terminal_tools = [tool.name for tool in self.tools if tool.terminal]
                        if not terminal_tools:
                            raise
                        history.append(
                            AssistantMessage(
                                role="user",
                                content=(
                                    "Your response did not satisfy the required structured "
                                    "output contract. Call exactly one terminal tool: "
                                    f"{', '.join(terminal_tools)}."
                                ),
                            )
                        )
                        continue
                    tracer.finish(payload)
                    return self._result(payload, history, input_tokens, output_tokens)
                for call in response.tool_calls:
                    tool = next((item for item in self.tools if item.name == call.name), None)
                    if tool is None:
                        raise RuntimeError(f"assistant requested unknown tool: {call.name}")
                    _validate_tool_arguments(tool, call.arguments)
                    if tool.terminal:
                        payload = redact_value(call.arguments)
                        if not isinstance(payload, dict):
                            raise ValueError(f"terminal tool {tool.name} returned invalid payload")
                        tracer.finish(payload)
                        return self._result(payload, history, input_tokens, output_tokens)
                    tool_span = tracer.start_tool(call)
                    try:
                        value = await _call_tool(tool, call.arguments)
                    except BaseException as exc:
                        tracer.finish_tool(tool_span, error=exc)
                        value = {"error": f"{type(exc).__name__}: {exc}"}
                    else:
                        tracer.finish_tool(tool_span, result=value)
                    history.append(
                        AssistantMessage(
                            role="tool",
                            tool_call_id=call.id,
                            content=json.dumps(redact_value(value), sort_keys=True, default=str),
                        )
                    )
            raise RuntimeError(f"assistant exceeded {self.max_rounds} tool rounds")
        except BaseException as exc:
            tracer.finish(error=exc)
            raise

    def _result(
        self,
        payload: dict[str, Any],
        history: list[AssistantMessage],
        input_tokens: int,
        output_tokens: int,
    ) -> AssistantRunResult:
        return AssistantRunResult(
            payload=payload,
            messages=tuple(history),
            usage=AssistantUsage(input_tokens, output_tokens),
            model=self.client.route.display_model,
            provider=self.client.route.provider,
            session_id=self.session_id,
        )


async def _call_tool(tool: AssistantTool, arguments: dict[str, Any]) -> Any:
    if tool.handler is None:
        raise RuntimeError(f"tool has no handler: {tool.name}")
    value = tool.handler(arguments)
    return await value if inspect.isawaitable(value) else value


def _validate_tool_arguments(tool: AssistantTool, arguments: dict[str, Any]) -> None:
    schema = tool.input_schema
    required = {str(item) for item in schema.get("required") or []}
    missing = sorted(required - set(arguments))
    if missing:
        raise ValueError(
            f"tool {tool.name} is missing required argument(s): {', '.join(missing)}"
        )
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError(
                f"tool {tool.name} has unknown argument(s): {', '.join(unknown)}"
            )
    expected_types = {
        "string": str,
        "object": dict,
        "array": list,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }
    for key, value in arguments.items():
        expected = (properties.get(key) or {}).get("type")
        if isinstance(expected, list) or expected not in expected_types:
            continue
        if not isinstance(value, expected_types[expected]):
            raise ValueError(f"tool {tool.name} argument {key} must be {expected}")


def _chat_payload(
    route: ModelRoute,
    messages: Sequence[AssistantMessage],
    tools: Sequence[AssistantTool],
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": route.model_id,
        "messages": [_chat_message(message) for message in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]
        payload["tool_choice"] = "auto"
    return payload


def _chat_message(message: AssistantMessage) -> dict[str, Any]:
    value: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_call_id:
        value["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        value["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, sort_keys=True),
                },
            }
            for call in message.tool_calls
        ]
    return value


def _responses_payload(
    route: ModelRoute,
    messages: Sequence[AssistantMessage],
    tools: Sequence[AssistantTool],
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    input_items: list[dict[str, Any]] = []
    instructions: list[str] = []
    for message in messages:
        if message.role == "system":
            instructions.append(message.content)
            continue
        if message.role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
            continue
        if message.content:
            input_items.append(
                {
                    "role": message.role,
                    "content": message.content,
                }
            )
        input_items.extend(
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": json.dumps(call.arguments, sort_keys=True),
            }
            for call in message.tool_calls
        )
    payload: dict[str, Any] = {
        "model": route.model_id,
        "input": input_items,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in tools
        ]
    return payload


def _messages_payload(
    route: ModelRoute,
    messages: Sequence[AssistantMessage],
    tools: Sequence[AssistantTool],
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    system = "\n\n".join(message.content for message in messages if message.role == "system")
    values: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            values.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                        }
                    ],
                }
            )
            continue
        content: list[dict[str, Any]] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        content.extend(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
            for call in message.tool_calls
        )
        values.append({"role": message.role, "content": content})
    payload: dict[str, Any] = {
        "model": route.model_id,
        "messages": values,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
    return payload


def _parse_chat(body: dict[str, Any]) -> AssistantResponse:
    message = ((body.get("choices") or [{}])[0].get("message") or {})
    calls = tuple(
        AssistantToolCall(
            id=str(item.get("id") or uuid.uuid4().hex),
            name=str((item.get("function") or {}).get("name") or ""),
            arguments=_arguments((item.get("function") or {}).get("arguments")),
        )
        for item in message.get("tool_calls") or []
    )
    usage = body.get("usage") or {}
    return AssistantResponse(
        text=str(message.get("content") or ""),
        tool_calls=calls,
        usage=AssistantUsage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
        raw_id=body.get("id"),
    )


def _parse_responses(body: dict[str, Any]) -> AssistantResponse:
    texts: list[str] = []
    calls: list[AssistantToolCall] = []
    for item in body.get("output") or []:
        if item.get("type") == "function_call":
            calls.append(
                AssistantToolCall(
                    id=str(item.get("call_id") or item.get("id") or uuid.uuid4().hex),
                    name=str(item.get("name") or ""),
                    arguments=_arguments(item.get("arguments")),
                )
            )
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"}:
                texts.append(str(content.get("text") or ""))
    usage = body.get("usage") or {}
    return AssistantResponse(
        text="\n".join(texts),
        tool_calls=tuple(calls),
        usage=AssistantUsage(usage.get("input_tokens"), usage.get("output_tokens")),
        raw_id=body.get("id"),
    )


def _parse_messages(body: dict[str, Any]) -> AssistantResponse:
    texts: list[str] = []
    calls: list[AssistantToolCall] = []
    for item in body.get("content") or []:
        if item.get("type") == "text":
            texts.append(str(item.get("text") or ""))
        elif item.get("type") == "tool_use":
            calls.append(
                AssistantToolCall(
                    id=str(item.get("id") or uuid.uuid4().hex),
                    name=str(item.get("name") or ""),
                    arguments=_arguments(item.get("input")),
                )
            )
    usage = body.get("usage") or {}
    return AssistantResponse(
        text="\n".join(texts),
        tool_calls=tuple(calls),
        usage=AssistantUsage(usage.get("input_tokens"), usage.get("output_tokens")),
        raw_id=body.get("id"),
    )


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant returned invalid tool arguments: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("assistant tool arguments must be an object")
    return parsed


def _json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("assistant returned neither a terminal tool call nor a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("assistant response must be a JSON object")
    return value


class _AssistantTrace:
    def __init__(
        self,
        *,
        role: AssistantRole,
        route: ModelRoute,
        env: Mapping[str, str],
        trace_content: str,
        session_id: str,
        attributes: Mapping[str, Any],
    ) -> None:
        self.role = role
        self.route = route
        self.env = dict(env)
        self.include_content = trace_content == "full"
        self.session_id = session_id
        self.attributes = dict(attributes)
        self.conversation: Any = None
        self.turn: Any = None

    def start(self, user_message: str) -> None:
        if (
            not self.env.get("WANDB_API_KEY", "").strip()
            or self.env.get("FUGUE_DISABLE_WEAVE") == "1"
        ):
            return
        try:
            weave = initialize_weave(trace_project_slug(self.env), self.env)
            start = getattr(weave, "start_conversation", None)
            if start is None:
                return
            self.conversation = start(
                agent_name=f"fugue-experiment-{self.role}" if self.role == "composer" else "fugue-analysis-agent",
                conversation_id=self.session_id,
                conversation_name=f"Fugue {self.role}",
                model=self.route.display_model,
                include_content=self.include_content,
                attributes=redact_value(
                    {
                        "fugue.ai.role": self.role,
                        "fugue.ai.session_id": self.session_id,
                        "fugue.ai.model": self.route.display_model,
                        **self.attributes,
                    }
                ),
            )
            _enter(self.conversation)
            self.turn = self.conversation.start_turn(
                user_message=user_message if self.include_content else "",
                agent_name=(
                    "fugue-experiment-composer"
                    if self.role == "composer"
                    else "fugue-analysis-agent"
                ),
                model=self.route.display_model,
            )
            _enter(self.turn)
        except Exception:
            self.conversation = None
            self.turn = None

    def start_llm(self, messages: Sequence[AssistantMessage]) -> Any:
        if self.turn is None:
            return None
        try:
            import weave

            span = weave.start_llm(
                model=self.route.model_id,
                provider_name=self.route.provider,
            )
            _enter(span)
            if self.include_content:
                span.input_messages = _weave_messages(
                    message for message in messages if message.role != "tool"
                )
            return span
        except Exception:
            return None

    def finish_llm(
        self,
        span: Any,
        *,
        response: AssistantResponse | None = None,
        error: BaseException | None = None,
    ) -> None:
        if span is None:
            return
        try:
            if response is not None:
                if self.include_content:
                    span.output_messages = _weave_messages(
                        [AssistantMessage("assistant", response.text)]
                    )
                _record_llm_usage(span, response.usage)
            if error is not None:
                span.error = f"{type(error).__name__}: {error}"
        finally:
            _exit(span, error)

    def start_tool(self, call: AssistantToolCall) -> Any:
        if self.turn is None:
            return None
        try:
            import weave

            span = weave.start_tool(
                name=call.name,
                arguments=json.dumps(call.arguments) if self.include_content else "{}",
            )
            _enter(span)
            return span
        except Exception:
            return None

    def finish_tool(
        self,
        span: Any,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        if span is None:
            return
        try:
            if self.include_content and error is None:
                span.result = redact_value(result)
            if error is not None:
                span.error = f"{type(error).__name__}: {error}"
        finally:
            _exit(span, error)

    def finish(
        self,
        output: Any = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        if self.turn is not None:
            if self.include_content and error is None:
                try:
                    self.turn.output = redact_value(output)
                except Exception:
                    pass
            _exit(self.turn, error)
            self.turn = None
        if self.conversation is not None:
            _exit(self.conversation, error)
            self.conversation = None


def _enter(value: Any) -> None:
    enter = getattr(value, "__enter__", None)
    if enter is not None:
        enter()


def _record_llm_usage(span: Any, usage: AssistantUsage) -> None:
    # Weave owns this concrete Usage type. Replacing it with a dict makes the
    # span fail during publication after the model call has already completed.
    target = getattr(span, "usage", None)
    if target is None:
        return
    for name in ("input_tokens", "output_tokens"):
        value = getattr(usage, name)
        if value is not None:
            setattr(target, name, int(value))


def _weave_messages(messages: Iterable[AssistantMessage]) -> list[Any]:
    import weave

    return [
        weave.Message(role=message.role, content=message.content)
        for message in messages
    ]


def _exit(value: Any, error: BaseException | None) -> None:
    exit_method = getattr(value, "__exit__", None)
    if exit_method is not None:
        if error is None:
            exit_method(None, None, None)
        else:
            exit_method(type(error), error, error.__traceback__)
        return
    end = getattr(value, "end", None)
    if end is not None:
        end()
