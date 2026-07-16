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
    "letta": "letta",
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
) -> str:
    """Make assigned skill use explicit enough to prove from Codex events."""
    if not skills:
        return instruction
    commands: list[str] = []
    root = PurePosixPath(directory)
    for skill_id in skills:
        if not skill_id or PurePosixPath(skill_id).name != skill_id:
            raise ValueError(f"invalid skill id for Codex delivery: {skill_id!r}")
        path = root / skill_id / "SKILL.md"
        commands.append(f"cat {shlex.quote(path.as_posix())}")
    return (
        "Assigned reviewed skills are behavior-affecting inputs. Before any task "
        "work, run each command below, then follow the instructions you read:\n"
        + "\n".join(commands)
        + "\n\n"
        + instruction
    )


def skill_invocation_evidence(
    logs_dir: Path,
    harness: str,
    registration: dict[str, Any],
) -> dict[str, Any]:
    assigned = [str(item) for item in registration.get("skills_assigned") or []]
    if not assigned:
        return {"status": "not_applicable", "skills_invoked": []}
    if harness != "codex":
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
    path = logs_dir / "codex.txt"
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = payload.get("item") or {}
        if (
            payload.get("type") != "item.completed"
            or item.get("type") != "command_execution"
            or item.get("status") != "completed"
            or item.get("exit_code") != 0
        ):
            continue
        command = str(item.get("command") or "")
        try:
            outer = shlex.split(command)
        except ValueError:
            continue
        argv = outer
        if len(outer) == 3 and Path(outer[0]).name in {"bash", "sh"} and outer[1] in {
            "-c",
            "-lc",
        }:
            try:
                argv = shlex.split(outer[2])
            except ValueError:
                continue
        if not argv or Path(argv[0]).name not in {"cat", "head", "sed", "tail"}:
            continue
        for skill_id in assigned:
            expected = f"{directory}/{skill_id}/SKILL.md"
            if expected in argv:
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
