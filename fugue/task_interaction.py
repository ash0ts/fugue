from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from fugue.bench.candidates import stable_digest
from fugue.bench.evaluations import _post_judge
from fugue.model_plane import model_route_identity, resolve_model_route

_MAX_TRANSCRIPT_BYTES = 48_000
_MAX_TURN_BYTES = 16_000


@dataclass(frozen=True)
class InteractionPlan:
    kind: str
    profile_id: str | None
    profile_digest: str | None
    model: str | None
    scripted_turns: tuple[str, ...]
    directions: tuple[str, ...]
    max_user_turns: int
    max_agent_turns: int
    timeout_sec: int
    input_cost_per_million: float
    output_cost_per_million: float
    reserve_cost_usd: float

    @property
    def follow_up_count(self) -> int:
        if self.kind == "single_turn":
            return 0
        if self.kind == "scripted":
            return len(self.scripted_turns)
        return self.max_user_turns


@dataclass
class TaskInteractionController:
    """Reveal locked follow-ups one turn at a time outside the Agent container."""

    plan: InteractionPlan
    logs_dir: Path
    initial_instruction: str
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    _messages: list[dict[str, str]] = field(default_factory=list)
    _observed_cost_usd: float = 0.0
    _unmeasured_calls: int = 0

    @classmethod
    def from_environment(
        cls,
        *,
        logs_dir: Path,
        initial_instruction: str,
        env: Mapping[str, str] | None = None,
    ) -> TaskInteractionController:
        selected_env = env or os.environ
        authored_raw = str(selected_env.get("FUGUE_TASK_AUTHORING") or "").strip()
        interaction_raw = str(selected_env.get("FUGUE_TASK_INTERACTION") or "").strip()
        metadata = json.loads(authored_raw) if authored_raw else {}
        if not isinstance(metadata, dict):
            raise ValueError("FUGUE_TASK_AUTHORING must contain one JSON object")
        controller = (
            json.loads(interaction_raw)
            if interaction_raw
            else metadata.get("interaction_controller")
        ) or {
            "type": "single_turn",
            "max_user_turns": 1,
            "max_agent_turns": 1,
            "timeout_sec": 900,
        }
        if not isinstance(controller, dict):
            raise ValueError("authored task interaction controller must be an object")
        value = cls(
            plan=_interaction_plan(controller),
            logs_dir=logs_dir,
            initial_instruction=initial_instruction,
            env=selected_env,
        )
        value._messages.append({"role": "user", "content": initial_instruction})
        return value

    @property
    def enabled(self) -> bool:
        return self.plan.follow_up_count > 0

    def observe_agent(self, content: str) -> None:
        bounded = _bounded_observation(content)
        self._messages.append({"role": "assistant", "content": bounded})
        self._record(
            "agent_turn",
            {
                "agent_turn": sum(
                    item["role"] == "assistant" for item in self._messages
                ),
                **_content_record(bounded, self.env),
            },
        )

    def next_follow_up(self, index: int) -> str:
        if not 0 <= index < self.plan.follow_up_count:
            raise IndexError("interaction follow-up index is outside the locked plan")
        if self.plan.kind == "scripted":
            prompt = self.plan.scripted_turns[index]
            route_receipt = None
        elif self.plan.kind == "model":
            prompt, route_receipt = self._model_follow_up(index)
        else:
            raise RuntimeError("single-turn tasks do not have follow-ups")
        prompt = _bounded_turn(prompt, "interactor follow-up")
        self._messages.append({"role": "user", "content": prompt})
        self._record(
            "interactor_turn",
            {
                "user_turn": index + 1,
                "kind": self.plan.kind,
                "profile_id": self.plan.profile_id,
                "profile_digest": self.plan.profile_digest,
                "route_receipt": route_receipt,
                **_content_record(prompt, self.env),
            },
        )
        return prompt

    def summary(self) -> dict[str, Any]:
        evidence = self.logs_dir / "task-interaction.jsonl"
        receipts = self.logs_dir / "interactor-route-receipts.jsonl"
        return {
            "status": "completed" if self.enabled else "single_turn",
            "type": self.plan.kind,
            "profile_id": self.plan.profile_id,
            "profile_digest": self.plan.profile_digest,
            "planned_agent_turns": self.plan.max_agent_turns,
            "observed_agent_turns": sum(
                item["role"] == "assistant" for item in self._messages
            ),
            "planned_interactor_calls": (
                self.plan.follow_up_count if self.plan.kind == "model" else 0
            ),
            "observed_interactor_cost_usd": self._observed_cost_usd,
            "unmeasured_paid_calls": self._unmeasured_calls,
            "accounted_interactor_cost_usd": self._observed_cost_usd
            + self._unmeasured_calls * self.plan.reserve_cost_usd,
            "evidence_path": evidence.name if evidence.is_file() else None,
            "route_receipts_path": receipts.name if receipts.is_file() else None,
        }

    def _model_follow_up(self, index: int) -> tuple[str, dict[str, Any]]:
        if not self.plan.model:
            raise ValueError("model interaction has no locked model route")
        route = resolve_model_route(self.plan.model, self.env)
        api_key = str(self.env.get(route.api_key_env) or "").strip()
        if not api_key:
            raise RuntimeError(f"{route.api_key_env} is required for interactor calls")
        payload = {
            "directions": list(self.plan.directions),
            "turn": index + 1,
            "remaining_turns": self.plan.follow_up_count - index - 1,
            "conversation": self._messages,
        }
        rendered = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if len(rendered.encode()) > _MAX_TRANSCRIPT_BYTES:
            rendered = rendered.encode()[-_MAX_TRANSCRIPT_BYTES:].decode(
                "utf-8", errors="ignore"
            )
        prompt = (
            "Act as the evaluation user in an ongoing Agent task. Using only the "
            "Agent-visible conversation and the directions below, produce the next "
            "user follow-up. Do not score the Agent, reveal hidden criteria, invent "
            "reference answers, or mention model, harness, variant, or treatment. "
            'Return only JSON: {"follow_up": "..."}.\n\n' + rendered
        )
        with httpx.Client(timeout=self.plan.timeout_sec) as client:
            response, usage = _post_judge(client, route, api_key, self.env, prompt)
        follow_up = str(response.get("follow_up") or "").strip()
        if not follow_up:
            raise ValueError("interactor returned no follow_up")
        cost = _usage_cost(
            usage,
            self.plan.input_cost_per_million,
            self.plan.output_cost_per_million,
        )
        if cost is None:
            self._unmeasured_calls += 1
        else:
            self._observed_cost_usd += cost
        receipt = {
            "schema_version": 1,
            "call_id": stable_digest(
                {
                    "profile_digest": self.plan.profile_digest,
                    "conversation": self._messages,
                    "turn": index + 1,
                }
            ),
            "role": "interactor",
            "profile_id": self.plan.profile_id,
            "profile_digest": self.plan.profile_digest,
            "route": model_route_identity(route),
            "usage": usage,
            "cost_usd": cost,
            "trace_scope": "separate_from_agent",
        }
        self._append_jsonl(self.logs_dir / "interactor-route-receipts.jsonl", receipt)
        return follow_up, receipt

    def _record(self, event: str, value: Mapping[str, Any]) -> None:
        self._append_jsonl(
            self.logs_dir / "task-interaction.jsonl",
            {
                "schema_version": 1,
                "event": event,
                "conversation_digest": stable_digest(self._messages),
                **value,
            },
        )

    def _append_jsonl(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")


def _interaction_plan(raw: Mapping[str, Any]) -> InteractionPlan:
    allowed = {
        "type",
        "profile_id",
        "profile_digest",
        "model",
        "scripted_turns",
        "directions",
        "max_user_turns",
        "max_agent_turns",
        "timeout_sec",
        "input_cost_per_million",
        "output_cost_per_million",
        "reserve_cost_usd",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "unknown interaction controller field(s): " + ", ".join(unknown)
        )
    kind = str(raw.get("type") or "single_turn")
    if kind not in {"single_turn", "scripted", "model"}:
        raise ValueError(f"unsupported task interaction type: {kind}")
    max_user_turns = _positive_int(raw.get("max_user_turns", 1), "max_user_turns")
    max_agent_turns = _positive_int(raw.get("max_agent_turns", 1), "max_agent_turns")
    scripted = tuple(
        _bounded_turn(str(item), "scripted follow-up")
        for item in raw.get("scripted_turns") or []
    )
    if kind == "single_turn" and max_agent_turns != 1:
        raise ValueError("single-turn interaction must have one Agent turn")
    if kind == "scripted" and len(scripted) != max_user_turns:
        raise ValueError("scripted interaction count differs from max_user_turns")
    if kind != "single_turn" and max_agent_turns != max_user_turns + 1:
        raise ValueError(
            "multi-turn interaction must end with one Agent reply per user turn"
        )
    profile_digest = str(raw.get("profile_digest") or "").strip() or None
    if kind != "single_turn" and (
        not profile_digest
        or len(profile_digest) != 64
        or any(character not in "0123456789abcdef" for character in profile_digest)
    ):
        raise ValueError("multi-turn interaction requires a locked profile digest")
    return InteractionPlan(
        kind=kind,
        profile_id=str(raw.get("profile_id") or "").strip() or None,
        profile_digest=profile_digest,
        model=str(raw.get("model") or "").strip() or None,
        scripted_turns=scripted,
        directions=tuple(str(item) for item in raw.get("directions") or []),
        max_user_turns=max_user_turns,
        max_agent_turns=max_agent_turns,
        timeout_sec=_positive_int(raw.get("timeout_sec", 900), "timeout_sec"),
        input_cost_per_million=_non_negative_number(
            raw.get("input_cost_per_million", 0), "input cost"
        ),
        output_cost_per_million=_non_negative_number(
            raw.get("output_cost_per_million", 0), "output cost"
        ),
        reserve_cost_usd=_non_negative_number(
            raw.get("reserve_cost_usd", 0), "reserve cost"
        ),
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value or 0)
    if result < 1:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _non_negative_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative number")
    result = float(value or 0)
    if result < 0:
        raise ValueError(f"{label} must be a non-negative number")
    return result


def _usage_cost(
    usage: Mapping[str, Any], input_rate: float, output_rate: float
) -> float | None:
    if not input_rate and not output_rate:
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _bounded_turn(value: str, label: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError(f"{label} must be non-empty")
    if len(result.encode()) > _MAX_TURN_BYTES:
        raise ValueError(f"{label} exceeds {_MAX_TURN_BYTES} bytes")
    return result


def _bounded_observation(value: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError("Agent response must be non-empty")
    encoded = result.encode()
    if len(encoded) <= _MAX_TURN_BYTES:
        return result
    tail = encoded[-_MAX_TURN_BYTES:].decode("utf-8", errors="ignore")
    return "[Earlier Agent output omitted by the interactor boundary.]\n" + tail


def _content_record(content: str, env: Mapping[str, str]) -> dict[str, Any]:
    encoded = content.encode()
    result: dict[str, Any] = {
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "content_bytes": len(encoded),
    }
    if str(env.get("FUGUE_TRACE_CONTENT") or "full") == "full":
        result["content"] = content
    return result
