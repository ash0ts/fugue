from __future__ import annotations

import json
import shlex
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

AGENT_NAMES = {
    "hermes": "hermes-agent",
    "openclaw": "openclaw",
    "claude-code": "claude-code",
    "codex": "codex",
}
CONVERSATION_NAMESPACE = uuid.UUID("218f38ca-7fe1-4db2-96e0-30f9b62c20eb")

def stable_agent_name(harness: str) -> str:
    return AGENT_NAMES.get(harness, harness)


def conversation_id(run_or_cohort_key: str) -> str:
    if not run_or_cohort_key:
        raise ValueError("conversation key cannot be empty")
    return str(uuid.uuid5(CONVERSATION_NAMESPACE, run_or_cohort_key))


def openclaw_agent_id(fugue_conversation_id: str) -> str:
    """Return the per-trial OpenClaw agent id used to isolate its trace."""
    return f"fugue-{fugue_conversation_id}"


def openclaw_conversation_id(fugue_conversation_id: str) -> str:
    """Return the conversation id emitted by OpenClaw for its main session."""
    return f"agent:{openclaw_agent_id(fugue_conversation_id)}:main"


def agent_conversation_id(harness: str, run_or_cohort_key: str) -> str:
    """Resolve the conversation identity emitted by a typed harness adapter."""
    resolved = conversation_id(run_or_cohort_key)
    if harness == "openclaw":
        return openclaw_conversation_id(resolved)
    return resolved


def agent_conversation_name(
    *, run_name: str, task_id: str, variant_id: str, trial_index: int
) -> str:
    labels = [
        value.strip()
        for value in (run_name, task_id, variant_id)
        if value and value.strip()
    ]
    labels.append(f"t{max(1, trial_index):03d}")
    return " · ".join(labels)[:256]


def normalize_trace_content(value: str | None) -> str:
    selected = str(value or "full").strip().lower()
    if selected not in {"full", "metadata"}:
        raise ValueError("trace content must be 'full' or 'metadata'")
    return selected


def codex_skill_instruction(
    instruction: str,
    *,
    skills: list[str],
    directory: str,
    provenance: Any = (),
) -> str:
    """Make assigned skill use explicit enough to prove from Codex events."""
    if not skills:
        return instruction
    commands: list[str] = []
    root = PurePosixPath(directory)
    runtime_names = skill_runtime_name_map(skills, provenance)
    for skill_id in skills:
        if not skill_id or PurePosixPath(skill_id).name != skill_id:
            raise ValueError(f"invalid skill id for Codex delivery: {skill_id!r}")
        path = root / runtime_names[skill_id] / "SKILL.md"
        commands.append(f"cat {shlex.quote(path.as_posix())}")
    return (
        "Assigned reviewed skills are behavior-affecting inputs. Before any task "
        "work, run each command below, then follow the instructions you read:\n"
        + "\n".join(commands)
        + "\n\n"
        + instruction
    )


def skill_runtime_name_map(
    assigned: list[str],
    provenance: Any,
) -> dict[str, str]:
    """Map stable Fugue Skill ids to the names registered by the harness.

    Imported Skills may be given an operator-facing alias while retaining the
    upstream ``name`` from their SKILL.md frontmatter. Runtime discovery uses
    that declared name; Fugue evidence continues to use the stable lock id.
    """
    mapping = {skill_id: skill_id for skill_id in assigned}
    if isinstance(provenance, list):
        for item in provenance:
            if not isinstance(item, dict):
                continue
            skill_id = str(item.get("id") or "")
            declared_name = str(item.get("declared_name") or "")
            if skill_id not in mapping or not declared_name:
                continue
            if (
                PurePosixPath(declared_name).name != declared_name
                or declared_name in {".", ".."}
            ):
                raise ValueError(
                    f"invalid declared runtime name for Skill {skill_id!r}"
                )
            mapping[skill_id] = declared_name
    runtime_names = list(mapping.values())
    if len(runtime_names) != len(set(runtime_names)):
        raise ValueError("assigned Skills must have unique declared runtime names")
    return mapping


def skill_invocation_evidence(
    logs_dir: Path,
    harness: str,
    registration: dict[str, Any],
) -> dict[str, Any]:
    assigned = [str(item) for item in registration.get("skills_assigned") or []]
    provenance = registration.get("skill_provenance") or []
    if not assigned:
        return {"status": "not_applicable", "skills_invoked": []}
    if harness not in {"codex", "claude-code"}:
        return {
            "status": "unavailable",
            "skills_invoked": [],
            "reason": f"{harness} does not emit structured skill-read events",
        }
    directory = str(registration.get("directory") or "").rstrip("/")
    if not directory:
        return {
            "status": "unavailable",
            "skills_invoked": [],
            "reason": "Codex skill registration did not record its isolated directory",
        }
    events: list[dict[str, str]] = []
    path = logs_dir / (
        "codex.txt" if harness == "codex" else "claude-code.txt"
    )
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    payloads: list[dict[str, Any]] = []
    for line in lines:
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    if harness == "claude-code":
        events.extend(
            _claude_skill_events(
                payloads,
                assigned=assigned,
                directory=directory,
                provenance=provenance,
            )
        )
    else:
        for payload in payloads:
            item = payload.get("item") or {}
            if (
                payload.get("type") != "item.completed"
                or item.get("type") != "command_execution"
                or item.get("status") != "completed"
                or item.get("exit_code") != 0
            ):
                continue
            skill_id = _skill_read_from_command(
                str(item.get("command") or ""),
                assigned=assigned,
                directory=directory,
                provenance=provenance,
            )
            if skill_id:
                events.append(
                    {
                        "item_id": str(item.get("id") or ""),
                        "operation": "read_skill_instructions",
                        "skill_id": skill_id,
                    }
                )
    invoked = list(dict.fromkeys(event["skill_id"] for event in events))
    missing = [skill_id for skill_id in assigned if skill_id not in invoked]
    return {
        "status": "observed" if invoked else "not_observed",
        "skills_invoked": invoked,
        "missing_skills": missing,
        "events": events,
    }


def _claude_skill_events(
    payloads: list[dict[str, Any]],
    *,
    assigned: list[str],
    directory: str,
    provenance: Any,
) -> list[dict[str, str]]:
    declared_to_id = {
        str(item.get("declared_name") or ""): str(item.get("id") or "")
        for item in provenance
        if isinstance(item, dict)
        and str(item.get("id") or "") in assigned
        and item.get("declared_name")
    }
    pending: dict[str, tuple[str, str]] = {}
    succeeded: set[str] = set()
    for payload in payloads:
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_id = str(block.get("id") or "")
                tool_name = str(block.get("name") or "")
                tool_input = block.get("input") or {}
                if not tool_id or not isinstance(tool_input, dict):
                    continue
                if tool_name == "Skill":
                    declared = str(
                        tool_input.get("skill")
                        or tool_input.get("name")
                        or ""
                    )
                    skill_id = declared_to_id.get(declared)
                    if skill_id:
                        pending[tool_id] = ("invoke_skill", skill_id)
                elif tool_name in {"Bash", "Shell"}:
                    skill_id = _skill_read_from_command(
                        str(tool_input.get("command") or ""),
                        assigned=assigned,
                        directory=directory,
                        provenance=provenance,
                    )
                    if skill_id:
                        pending[tool_id] = ("read_skill_instructions", skill_id)
            elif block.get("type") == "tool_result":
                tool_id = str(block.get("tool_use_id") or "")
                if tool_id and block.get("is_error") is not True:
                    succeeded.add(tool_id)
    return [
        {
            "item_id": tool_id,
            "operation": operation,
            "skill_id": skill_id,
        }
        for tool_id, (operation, skill_id) in pending.items()
        if tool_id in succeeded
    ]


def _skill_read_from_command(
    command: str,
    *,
    assigned: list[str],
    directory: str,
    provenance: Any = (),
) -> str | None:
    try:
        outer = shlex.split(command)
    except ValueError:
        return None
    argv = outer
    if len(outer) == 3 and Path(outer[0]).name in {"bash", "sh"} and outer[1] in {
        "-c",
        "-lc",
    }:
        try:
            argv = shlex.split(outer[2])
        except ValueError:
            return None
    if not argv or Path(argv[0]).name not in {"cat", "head", "sed", "tail"}:
        return None
    runtime_names = skill_runtime_name_map(assigned, provenance)
    for skill_id, runtime_name in runtime_names.items():
        if f"{directory}/{runtime_name}/SKILL.md" in argv:
            return skill_id
    return None
