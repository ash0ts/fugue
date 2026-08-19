from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from fugue.bench.candidates import stable_digest
from fugue.redaction import redact_value

TASK_PRESENTATION_SCHEMA_VERSION = 1
TASK_RESULT_SCHEMA_VERSION = 1
UNDECLARED_REQUIRED_OUTPUT = "Not declared in the public task."
UNDECLARED_ACCEPTANCE_CRITERION = (
    "No separate public acceptance criteria were declared."
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_AGENT_EXECUTION_STATUSES = frozenset(
    {
        "completed",
        "timed_out",
        "failed",
        "cancelled",
        "interrupted",
        "not_started",
        "not_applicable",
    }
)
_EVIDENCE_INTEGRITY_STATUSES = frozenset({"verified", "incomplete", "invalid"})
_PRIVATE_FIELDS = frozenset(
    {
        "answer_key",
        "expected",
        "gold",
        "gold_output",
        "private",
        "private_expected_values",
        "private_labels",
        "reference_answer",
    }
)


@dataclass(frozen=True)
class PublicPromptPartV1:
    order: int
    text: str

    def __post_init__(self) -> None:
        if type(self.order) is not int or self.order < 1:
            raise ValueError("public prompt part order must be positive")
        _public_text(self.text, "public prompt part", maximum=64_000)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskPresentationV1:
    """A public, digest-bound description of one logical task.

    This contract contains only information that the Agent already receives or
    that the author explicitly classified as public. Host-only expected values,
    grader configuration, and credentials are never accepted here.
    """

    task_id: str
    title: str
    public_prompt: tuple[PublicPromptPartV1, ...]
    required_output: str
    public_acceptance_criteria: tuple[str, ...]
    scenario: str | None = None
    tags: tuple[str, ...] = ()
    partition: str | None = None
    safe_resource_references: tuple[dict[str, str], ...] = ()
    schema_version: Literal[1] = TASK_PRESENTATION_SCHEMA_VERSION
    task_definition_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != TASK_PRESENTATION_SCHEMA_VERSION:
            raise ValueError("unsupported task presentation schema version")
        _public_text(self.task_id, "task presentation task id", maximum=300)
        _public_text(self.title, "task presentation title", maximum=500)
        if not self.public_prompt:
            raise ValueError("task presentation requires a public prompt")
        expected_orders = tuple(range(1, len(self.public_prompt) + 1))
        if tuple(item.order for item in self.public_prompt) != expected_orders:
            raise ValueError(
                "task presentation public prompt parts must be ordered from one"
            )
        _public_text(
            self.required_output,
            "task presentation required output",
            maximum=4_000,
        )
        if not self.public_acceptance_criteria:
            raise ValueError("task presentation requires public acceptance criteria")
        for criterion in self.public_acceptance_criteria:
            _public_text(
                criterion,
                "task presentation public acceptance criterion",
                maximum=2_000,
            )
        if len(set(self.public_acceptance_criteria)) != len(
            self.public_acceptance_criteria
        ):
            raise ValueError(
                "task presentation public acceptance criteria must be unique"
            )
        if self.scenario is not None:
            _public_text(self.scenario, "task presentation scenario", maximum=300)
        if self.partition is not None:
            _public_text(self.partition, "task presentation partition", maximum=100)
        for tag in self.tags:
            _public_text(tag, "task presentation tag", maximum=200)
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("task presentation tags must be unique")
        for reference in self.safe_resource_references:
            _safe_resource_reference(reference)
        if len(
            {
                stable_digest(reference)
                for reference in self.safe_resource_references
            }
        ) != len(self.safe_resource_references):
            raise ValueError("task presentation resource references must be unique")
        unsigned = self._unsigned_dict()
        _assert_public(unsigned, "task presentation")
        computed = stable_digest(unsigned)
        if self.task_definition_digest and self.task_definition_digest != computed:
            raise ValueError("task presentation digest does not match its definition")
        if not self.task_definition_digest:
            object.__setattr__(self, "task_definition_digest", computed)

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "title": self.title,
            "public_prompt": [item.to_dict() for item in self.public_prompt],
            "required_output": self.required_output,
            "public_acceptance_criteria": list(self.public_acceptance_criteria),
            "scenario": self.scenario,
            "tags": list(self.tags),
            "partition": self.partition,
            "safe_resource_references": [
                dict(item) for item in self.safe_resource_references
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._unsigned_dict(),
            "task_definition_digest": self.task_definition_digest,
        }


@dataclass(frozen=True)
class FailedRequiredCheckV1:
    id: str
    label: str
    explanation: str | None = None
    critical: bool = True

    def __post_init__(self) -> None:
        _public_text(self.id, "failed required check id", maximum=300)
        _public_text(self.label, "failed required check label", maximum=500)
        if self.explanation is not None:
            _public_text(
                self.explanation,
                "failed required check explanation",
                maximum=2_000,
            )
        if not isinstance(self.critical, bool):
            raise ValueError("failed required check critical must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {key: item for key, item in value.items() if item is not None}


@dataclass(frozen=True)
class TaskResultV1:
    task_passed: bool | None
    outcome_summary: str
    failed_required_checks: tuple[FailedRequiredCheckV1, ...]
    agent_execution_status: Literal[
        "completed",
        "timed_out",
        "failed",
        "cancelled",
        "interrupted",
        "not_started",
        "not_applicable",
    ]
    evidence_integrity_status: Literal["verified", "incomplete", "invalid"]
    answer_digest: str | None = None
    schema_version: Literal[1] = TASK_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TASK_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported task result schema version")
        if self.task_passed is not None and not isinstance(self.task_passed, bool):
            raise ValueError("task result task_passed must be a boolean or null")
        _public_text(self.outcome_summary, "task result outcome summary", maximum=2_000)
        if len({item.id for item in self.failed_required_checks}) != len(
            self.failed_required_checks
        ):
            raise ValueError("failed required check ids must be unique")
        if self.task_passed is True and self.failed_required_checks:
            raise ValueError("a passing task cannot have failed required checks")
        if self.task_passed is False and not self.failed_required_checks:
            raise ValueError("a failing task requires at least one failed check")
        if self.agent_execution_status not in _AGENT_EXECUTION_STATUSES:
            raise ValueError("task result has an invalid Agent execution status")
        if self.evidence_integrity_status not in _EVIDENCE_INTEGRITY_STATUSES:
            raise ValueError("task result has an invalid evidence integrity status")
        if self.evidence_integrity_status == "invalid" and self.task_passed is not None:
            raise ValueError("invalid evidence cannot publish a task verdict")
        if self.answer_digest is not None and not _DIGEST.fullmatch(
            self.answer_digest
        ):
            raise ValueError("task result answer digest must be a SHA-256")
        _assert_public(self.to_dict(), "task result")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_passed": self.task_passed,
            "outcome_summary": self.outcome_summary,
            "failed_required_checks": [
                item.to_dict() for item in self.failed_required_checks
            ],
            "answer_digest": self.answer_digest,
            "agent_execution_status": self.agent_execution_status,
            "evidence_integrity_status": self.evidence_integrity_status,
        }


def task_presentation_from_public_case(
    *,
    task_id: str,
    public_case: Mapping[str, Any] | None,
    title: str | None = None,
    scenario: str | None = None,
    tags: Sequence[str] = (),
    partition: str | None = None,
) -> TaskPresentationV1 | None:
    """Build a presentation only when the exact public prompt is available."""

    if public_case is None:
        return None
    _reject_private_fields(public_case, "public task case")
    prompt_texts = _public_prompt_texts(public_case)
    if not prompt_texts:
        return None
    selected_title = str(public_case.get("title") or title or "").strip()
    if not selected_title:
        selected_title = task_id.replace("-", " ").replace("_", " ").title()
    required_output = str(public_case.get("required_output") or "").strip()
    if not required_output:
        required_output = UNDECLARED_REQUIRED_OUTPUT
    criteria_raw = public_case.get("public_acceptance_criteria")
    criteria = (
        tuple(str(item).strip() for item in criteria_raw if str(item).strip())
        if isinstance(criteria_raw, list | tuple)
        else ()
    )
    if not criteria:
        criteria = (UNDECLARED_ACCEPTANCE_CRITERION,)
    selected_tags = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in (public_case.get("tags") or tags)
            if str(item).strip()
        )
    )
    selected_scenario = str(
        public_case.get("scenario_id") or public_case.get("scenario") or scenario or ""
    ).strip() or None
    selected_partition = str(
        public_case.get("partition") or partition or ""
    ).strip() or None
    return TaskPresentationV1(
        task_id=task_id,
        title=selected_title,
        public_prompt=tuple(
            PublicPromptPartV1(order=index, text=text)
            for index, text in enumerate(prompt_texts, start=1)
        ),
        required_output=required_output,
        public_acceptance_criteria=criteria,
        scenario=selected_scenario,
        tags=selected_tags,
        partition=selected_partition,
        safe_resource_references=_safe_resource_references(public_case),
    )


def task_presentation_from_dict(raw: Mapping[str, Any]) -> TaskPresentationV1:
    allowed = {
        "schema_version",
        "task_id",
        "title",
        "public_prompt",
        "required_output",
        "public_acceptance_criteria",
        "scenario",
        "tags",
        "partition",
        "safe_resource_references",
        "task_definition_digest",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "unknown task presentation field(s): " + ", ".join(unknown)
        )
    if type(raw.get("schema_version")) is not int or raw.get("schema_version") != 1:
        raise ValueError("unsupported task presentation schema version")
    digest = raw.get("task_definition_digest")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ValueError("task presentation requires a SHA-256 definition digest")
    for field_name in ("task_id", "title", "required_output"):
        if not isinstance(raw.get(field_name), str):
            raise ValueError(f"task presentation {field_name} must be text")
    for field_name in ("scenario", "partition"):
        if raw.get(field_name) is not None and not isinstance(
            raw.get(field_name), str
        ):
            raise ValueError(f"task presentation {field_name} must be text or null")
    prompt = raw.get("public_prompt")
    if not isinstance(prompt, list | tuple):
        raise ValueError("task presentation public_prompt must be an array")
    parts: list[PublicPromptPartV1] = []
    for item in prompt:
        if not isinstance(item, Mapping) or set(item) != {"order", "text"}:
            raise ValueError("public prompt parts require order and text")
        order = item.get("order")
        if type(order) is not int:
            raise ValueError("public prompt part order must be an integer")
        if not isinstance(item.get("text"), str):
            raise ValueError("public prompt part text must be text")
        parts.append(PublicPromptPartV1(order=order, text=item["text"]))
    references = raw.get("safe_resource_references") or ()
    if not isinstance(references, list | tuple) or not all(
        isinstance(item, Mapping) for item in references
    ):
        raise ValueError("task presentation resource references must be objects")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for item in references
        for key, value in item.items()
    ):
        raise ValueError("task presentation resource reference values must be text")
    criteria = raw.get("public_acceptance_criteria")
    if not isinstance(criteria, list | tuple) or not all(
        isinstance(item, str) for item in criteria
    ):
        raise ValueError("task presentation acceptance criteria must be an array")
    tags = raw.get("tags") or ()
    if not isinstance(tags, list | tuple) or not all(
        isinstance(item, str) for item in tags
    ):
        raise ValueError("task presentation tags must be an array")
    return TaskPresentationV1(
        schema_version=raw["schema_version"],  # type: ignore[arg-type]
        task_id=raw["task_id"],
        title=raw["title"],
        public_prompt=tuple(parts),
        required_output=raw["required_output"],
        public_acceptance_criteria=tuple(criteria),
        scenario=raw.get("scenario"),
        tags=tuple(tags),
        partition=raw.get("partition"),
        safe_resource_references=tuple(
            dict(item) for item in references
        ),
        task_definition_digest=digest,
    )


def task_result_from_dict(raw: Mapping[str, Any]) -> TaskResultV1:
    allowed = {
        "schema_version",
        "task_passed",
        "outcome_summary",
        "failed_required_checks",
        "answer_digest",
        "agent_execution_status",
        "evidence_integrity_status",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError("unknown task result field(s): " + ", ".join(unknown))
    if type(raw.get("schema_version")) is not int or raw.get("schema_version") != 1:
        raise ValueError("unsupported task result schema version")
    if not isinstance(raw.get("outcome_summary"), str):
        raise ValueError("task result outcome_summary must be text")
    for field_name in ("agent_execution_status", "evidence_integrity_status"):
        if not isinstance(raw.get(field_name), str):
            raise ValueError(f"task result {field_name} must be text")
    if raw.get("answer_digest") is not None and not isinstance(
        raw.get("answer_digest"), str
    ):
        raise ValueError("task result answer_digest must be text or null")
    checks = raw.get("failed_required_checks") or ()
    if not isinstance(checks, list | tuple):
        raise ValueError("task result failed checks must be an array")
    parsed: list[FailedRequiredCheckV1] = []
    for item in checks:
        if not isinstance(item, Mapping):
            raise ValueError("failed required check must be an object")
        unknown_check = sorted(set(item) - {"id", "label", "explanation", "critical"})
        if unknown_check:
            raise ValueError(
                "unknown failed required check field(s): " + ", ".join(unknown_check)
            )
        for field_name in ("id", "label"):
            if not isinstance(item.get(field_name), str):
                raise ValueError(f"failed required check {field_name} must be text")
        if item.get("explanation") is not None and not isinstance(
            item.get("explanation"), str
        ):
            raise ValueError("failed required check explanation must be text or null")
        if "critical" in item and not isinstance(item.get("critical"), bool):
            raise ValueError("failed required check critical must be a boolean")
        parsed.append(
            FailedRequiredCheckV1(
                id=item["id"],
                label=item["label"],
                explanation=item.get("explanation"),
                critical=item.get("critical", True),
            )
        )
    task_passed = raw.get("task_passed")
    if task_passed is not None and not isinstance(task_passed, bool):
        raise ValueError("task result task_passed must be a boolean or null")
    return TaskResultV1(
        schema_version=raw["schema_version"],  # type: ignore[arg-type]
        task_passed=task_passed,
        outcome_summary=raw["outcome_summary"],
        failed_required_checks=tuple(parsed),
        answer_digest=raw.get("answer_digest"),
        agent_execution_status=raw["agent_execution_status"],  # type: ignore[arg-type]
        evidence_integrity_status=raw["evidence_integrity_status"],  # type: ignore[arg-type]
    )


def candidate_treatment_summary(definition: Mapping[str, Any]) -> str:
    """Return an allowlisted, public summary of one behavior definition."""

    harness = str(definition.get("harness") or "Agent").strip()
    route = definition.get("model_route")
    model = ""
    if isinstance(route, Mapping):
        model = str(
            route.get("display_model") or route.get("model") or route.get("id") or ""
        ).strip()
    parts = [harness]
    if model:
        parts.append(f"model {model}")
    skills = _component_descriptions(
        definition.get("skills"),
        revision_fields=("resolved_commit", "digest", "version"),
    )
    if skills:
        parts.append("Skills " + ", ".join(skills))
    integrations = _component_descriptions(
        definition.get("integrations"),
        revision_fields=("version_identity", "version", "behavior_hash"),
    )
    if integrations:
        parts.append("integrations " + ", ".join(integrations))
    context = definition.get("context")
    if isinstance(context, Mapping):
        context_id = str(context.get("id") or "").strip()
        if context_id and context_id != "none":
            delivery = str(context.get("delivery") or "").strip()
            parts.append(
                f"context {context_id}" + (f" via {delivery}" if delivery else "")
            )
    prompt_digest = str(definition.get("prompt_digest") or "").strip()
    if prompt_digest:
        parts.append(f"prompt {prompt_digest[:12]}")
    summary = "; ".join(parts) + "."
    _public_text(summary, "candidate treatment summary", maximum=2_000)
    return summary


def _public_prompt_texts(public_case: Mapping[str, Any]) -> tuple[str, ...]:
    prompt = public_case.get("public_prompt") or public_case.get("prompt")
    values: list[str] = []
    if isinstance(prompt, str):
        if prompt.strip():
            values.append(prompt)
    elif isinstance(prompt, list | tuple):
        for item in prompt:
            text = (
                str(item.get("text") or "")
                if isinstance(item, Mapping)
                else str(item)
            ).strip()
            if text:
                values.append(text)
    if not values:
        instruction = str(public_case.get("instruction") or "").strip()
        if not instruction:
            input_value = public_case.get("input")
            if isinstance(input_value, Mapping) and isinstance(
                input_value.get("question"), str
            ):
                instruction = str(input_value["question"]).strip()
        if instruction:
            values.append(instruction)
    return tuple(values)


def _safe_resource_references(
    public_case: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    result: list[dict[str, str]] = []
    resources = [
        *(public_case.get("attachments") or ()),
        *(public_case.get("resources") or ()),
    ]
    for index, item in enumerate(resources, start=1):
        if not isinstance(item, Mapping):
            raise ValueError("public task resource must be an object")
        selected: dict[str, str] = {}
        target = str(item.get("target") or "").strip()
        resource_id = str(
            item.get("resource_profile_id")
            or item.get("id")
            or (target.rsplit("/", maxsplit=1)[-1] if target else "")
            or f"resource-{index}"
        ).strip()
        selected["id"] = resource_id
        for key in ("sha256", "target", "media_type", "title"):
            value = str(item.get(key) or "").strip()
            if value:
                selected[key] = value
        result.append(selected)
    repository = public_case.get("repository")
    if isinstance(repository, Mapping):
        digest = str(repository.get("sha256") or "").strip()
        if digest:
            result.append(
                {
                    "id": "task-repository",
                    "sha256": digest,
                    "media_type": "application/vnd.fugue.repository",
                }
            )
    return tuple(result)


def _safe_resource_reference(value: Mapping[str, str]) -> None:
    allowed = {"id", "sha256", "target", "media_type", "title"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            "unknown safe resource reference field(s): " + ", ".join(unknown)
        )
    if not value.get("id"):
        raise ValueError("safe resource reference id is required")
    for key, item in value.items():
        _public_text(str(item), f"safe resource reference {key}", maximum=1_000)
    digest = value.get("sha256")
    if digest is not None and not _DIGEST.fullmatch(digest):
        raise ValueError("safe resource reference sha256 is invalid")
    _assert_public(dict(value), "safe resource reference")


def _component_descriptions(
    raw: Any, *, revision_fields: tuple[str, ...]
) -> tuple[str, ...]:
    if not isinstance(raw, list | tuple):
        return ()
    values: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            value = str(item.get("id") or item.get("name") or "").strip()
            revisions = _component_revisions(item, fields=revision_fields)
            if value and revisions:
                value = f"{value} at {', '.join(revisions)}"
        else:
            value = str(item).strip()
        if value:
            values.append(value)
    return tuple(dict.fromkeys(values))


def _component_revisions(
    item: Mapping[str, Any], *, fields: tuple[str, ...]
) -> tuple[str, ...]:
    result: list[str] = []
    for field_name in fields:
        raw = str(item.get(field_name) or "").strip()
        if not raw:
            continue
        if field_name == "resolved_commit":
            label = "commit " + raw[:12]
        elif field_name == "digest":
            label = "digest " + _short_digest(raw)
        elif field_name == "behavior_hash":
            label = "behavior " + _short_digest(raw)
        elif field_name == "version_identity" and raw.startswith("git:"):
            label = "commit " + raw.removeprefix("git:")[:12]
        else:
            label = "version " + _short_digest(raw)
        if label not in result:
            result.append(label)
    return tuple(result)


def _short_digest(value: str) -> str:
    if value.startswith("sha256:"):
        return "sha256:" + value.removeprefix("sha256:")[:12]
    return value[:12]


def _reject_private_fields(value: Mapping[str, Any], label: str) -> None:
    leaked = sorted(str(key) for key in value if str(key).lower() in _PRIVATE_FIELDS)
    if leaked:
        raise ValueError(f"{label} contains private field(s): " + ", ".join(leaked))


def _assert_public(value: Any, label: str) -> None:
    if redact_value(value) != value:
        raise ValueError(f"{label} contains credential-like content")


def _public_text(value: str, label: str, *, maximum: int) -> str:
    text = " ".join(value.split())
    if not text or len(value) > maximum:
        raise ValueError(f"{label} must be 1..{maximum} characters")
    if redact_value(value) != value:
        raise ValueError(f"{label} contains credential-like content")
    return value
