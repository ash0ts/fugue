from __future__ import annotations

import json
import shlex
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

AGENT_NAMES = {
    "hermes": "hermes-agent",
    "openclaw": "openclaw",
    "claude-code": "claude-code",
    "codex": "codex",
}
CONVERSATION_NAMESPACE = uuid.UUID("218f38ca-7fe1-4db2-96e0-30f9b62c20eb")
SKILL_MECHANISM_EVIDENCE_SCHEMA_VERSION = 2
_MAX_SKILL_EVIDENCE_EVENTS = 256
_MAX_SKILL_RELATIVE_PATH = 500


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
    registered = [str(item) for item in registration.get("skills_registered") or []]
    if not assigned:
        return _skill_evidence_payload(
            status="not_applicable",
            assigned=assigned,
            registered=registered,
        )
    if harness not in {"codex", "claude-code"}:
        return _skill_evidence_payload(
            status="unavailable",
            assigned=assigned,
            registered=registered,
            reason=f"{harness} does not emit structured skill-read events",
        )
    directory = str(registration.get("directory") or "").rstrip("/")
    if not directory:
        return _skill_evidence_payload(
            status="unavailable",
            assigned=assigned,
            registered=registered,
            reason="skill registration did not record its isolated directory",
        )
    events: list[dict[str, str]] = []
    path = logs_dir / ("codex.txt" if harness == "codex" else "claude-code.txt")
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
                provenance=registration.get("skill_provenance") or [],
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
            skill_read = _skill_read_from_command(
                str(item.get("command") or ""),
                assigned=assigned,
                directory=directory,
            )
            if skill_read:
                skill_id, relative_path = skill_read
                events.append(
                    {
                        "item_id": str(item.get("id") or ""),
                        "operation": (
                            "read_skill_instructions"
                            if relative_path == "SKILL.md"
                            else "read_skill_file"
                        ),
                        "skill_id": skill_id,
                        "relative_path": relative_path,
                    }
                )
    return _skill_evidence_payload(
        status="observed" if events else "not_observed",
        assigned=assigned,
        registered=registered,
        events=events[:_MAX_SKILL_EVIDENCE_EVENTS],
    )


def normalize_skill_mechanism_evidence(value: Any) -> dict[str, Any]:
    """Read V2 Skill evidence and older merged-use evidence without guessing.

    Historical evidence called both an instruction-file read and a native Skill
    tool call ``skills_invoked``.  Event operations can disambiguate those rows.
    Rows without operations remain explicitly unclassified instead of being
    promoted to either mechanism.
    """

    if not isinstance(value, Mapping):
        return {
            "status": "unavailable",
            "contract_status": "unavailable",
            "skills_opened": [],
            "skill_files_opened": [],
            "skills_native_invoked": [],
            "legacy_unclassified_skill_use": [],
        }
    events = [
        dict(item) for item in value.get("events") or [] if isinstance(item, Mapping)
    ][:_MAX_SKILL_EVIDENCE_EVENTS]
    opened: list[str] = []
    native: list[str] = []
    files: list[dict[str, str]] = []
    for event in events:
        skill_id = str(event.get("skill_id") or "")
        operation = str(event.get("operation") or "")
        if not skill_id:
            continue
        if operation == "invoke_skill":
            native.append(skill_id)
        elif operation in {"read_skill_instructions", "read_skill_file"}:
            relative_path = str(event.get("relative_path") or "")
            if not relative_path and operation == "read_skill_instructions":
                relative_path = "SKILL.md"
            if _safe_skill_relative_path(relative_path) is None:
                continue
            files.append({"skill_id": skill_id, "relative_path": relative_path})
            if relative_path == "SKILL.md":
                opened.append(skill_id)
    if value.get("schema_version") == SKILL_MECHANISM_EVIDENCE_SCHEMA_VERSION:
        explicit_opened = [
            str(item) for item in value.get("skills_opened") or [] if str(item)
        ]
        explicit_native = [
            str(item) for item in value.get("skills_native_invoked") or [] if str(item)
        ]
        explicit_files: list[dict[str, str]] = []
        for item in value.get("skill_files_opened") or []:
            if not isinstance(item, Mapping):
                continue
            skill_id = str(item.get("skill_id") or "")
            relative_path = str(item.get("relative_path") or "")
            if skill_id and _safe_skill_relative_path(relative_path) is not None:
                explicit_files.append(
                    {"skill_id": skill_id, "relative_path": relative_path}
                )
        derived_opened = list(dict.fromkeys(opened))
        derived_native = list(dict.fromkeys(native))
        derived_files = _unique_skill_files(files)
        mismatched = any(
            name in value and set(explicit) != set(derived)
            for name, explicit, derived in (
                ("skills_opened", explicit_opened, derived_opened),
                ("skills_native_invoked", explicit_native, derived_native),
                (
                    "skills_invoked",
                    [str(item) for item in value.get("skills_invoked") or []],
                    derived_native,
                ),
            )
        ) or (
            "skill_files_opened" in value
            and {(item["skill_id"], item["relative_path"]) for item in explicit_files}
            != {(item["skill_id"], item["relative_path"]) for item in derived_files}
        )
        opened = derived_opened
        native = derived_native
        files = derived_files
        unclassified: list[str] = []
        contract_status = "invalid" if mismatched else "valid"
    else:
        classified = set(opened) | set(native)
        unclassified = [
            str(item)
            for item in value.get("skills_invoked") or []
            if str(item) and str(item) not in classified
        ]
        contract_status = "legacy"
    return {
        "status": str(value.get("status") or "unavailable"),
        "contract_status": contract_status,
        "skills_opened": list(dict.fromkeys(opened)),
        "skill_files_opened": _unique_skill_files(files),
        "skills_native_invoked": list(dict.fromkeys(native)),
        "legacy_unclassified_skill_use": list(dict.fromkeys(unclassified)),
    }


def _skill_evidence_payload(
    *,
    status: str,
    assigned: list[str],
    registered: list[str],
    events: list[dict[str, str]] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_skill_mechanism_evidence(
        {
            "schema_version": SKILL_MECHANISM_EVIDENCE_SCHEMA_VERSION,
            "status": status,
            "events": events or [],
        }
    )
    opened = normalized["skills_opened"]
    native = normalized["skills_native_invoked"]
    result: dict[str, Any] = {
        "schema_version": SKILL_MECHANISM_EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "skills_assigned": list(dict.fromkeys(assigned)),
        "skills_registered": list(dict.fromkeys(registered)),
        "skills_opened": opened,
        "skill_files_opened": normalized["skill_files_opened"],
        "skills_native_invoked": native,
        # Backward field retained with precise V2 semantics: native tool only.
        "skills_invoked": native,
        "missing_skill_instructions": [
            skill_id for skill_id in assigned if skill_id not in opened
        ],
        "missing_skills": [skill_id for skill_id in assigned if skill_id not in opened],
        "events": events or [],
    }
    if reason:
        result["reason"] = reason
    return result


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
                        tool_input.get("skill") or tool_input.get("name") or ""
                    )
                    skill_id = declared_to_id.get(declared)
                    if skill_id:
                        pending[tool_id] = ("invoke_skill", skill_id)
                elif tool_name in {"Bash", "Shell"}:
                    skill_read = _skill_read_from_command(
                        str(tool_input.get("command") or ""),
                        assigned=assigned,
                        directory=directory,
                    )
                    if skill_read:
                        skill_id, relative_path = skill_read
                        operation = (
                            "read_skill_instructions"
                            if relative_path == "SKILL.md"
                            else "read_skill_file"
                        )
                        pending[tool_id] = (
                            f"{operation}\0{relative_path}",
                            skill_id,
                        )
                elif tool_name in {"Read", "read_file"}:
                    skill_read = _skill_file_read_from_path(
                        str(
                            tool_input.get("file_path") or tool_input.get("path") or ""
                        ),
                        assigned=assigned,
                        directory=directory,
                    )
                    if skill_read:
                        skill_id, relative_path = skill_read
                        operation = (
                            "read_skill_instructions"
                            if relative_path == "SKILL.md"
                            else "read_skill_file"
                        )
                        pending[tool_id] = (
                            f"{operation}\0{relative_path}",
                            skill_id,
                        )
            elif block.get("type") == "tool_result":
                tool_id = str(block.get("tool_use_id") or "")
                if tool_id and block.get("is_error") is not True:
                    succeeded.add(tool_id)
    result: list[dict[str, str]] = []
    for tool_id, (encoded_operation, skill_id) in pending.items():
        if tool_id not in succeeded:
            continue
        operation, _, relative_path = encoded_operation.partition("\0")
        event = {
            "item_id": tool_id,
            "operation": operation,
            "skill_id": skill_id,
        }
        if relative_path:
            event["relative_path"] = relative_path
        result.append(event)
    return result[:_MAX_SKILL_EVIDENCE_EVENTS]


def _skill_read_from_command(
    command: str,
    *,
    assigned: list[str],
    directory: str,
) -> tuple[str, str] | None:
    try:
        outer = shlex.split(command)
    except ValueError:
        return None
    argv = outer
    if (
        len(outer) == 3
        and Path(outer[0]).name in {"bash", "sh"}
        and outer[1]
        in {
            "-c",
            "-lc",
        }
    ):
        try:
            argv = shlex.split(outer[2])
        except ValueError:
            return None
    if not argv or Path(argv[0]).name not in {"cat", "head", "sed", "tail"}:
        return None
    for token in argv[1:]:
        observed = _skill_file_read_from_path(
            token,
            assigned=assigned,
            directory=directory,
        )
        if observed is not None:
            return observed
    return None


def _skill_file_read_from_path(
    value: str,
    *,
    assigned: list[str],
    directory: str,
) -> tuple[str, str] | None:
    if not value or len(value) > 2_000:
        return None
    root = PurePosixPath(directory)
    candidate = PurePosixPath(value)
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) < 2 or relative.parts[0] not in assigned:
        return None
    skill_id = relative.parts[0]
    skill_relative = PurePosixPath(*relative.parts[1:]).as_posix()
    if _safe_skill_relative_path(skill_relative) is None:
        return None
    return skill_id, skill_relative


def _safe_skill_relative_path(value: str) -> str | None:
    if not value or len(value) > _MAX_SKILL_RELATIVE_PATH:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _unique_skill_files(values: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for value in values:
        key = (value["skill_id"], value["relative_path"])
        if key in seen:
            continue
        seen.add(key)
        result.append({"skill_id": key[0], "relative_path": key[1]})
    return result
