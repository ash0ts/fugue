"""Strict, lazy dispatch for optional reference-study adapters.

Reference studies are examples of Fugue's generic comparison surface.  They
may contribute preparation and readiness validation, but they never replace
candidate resolution, planning, execution, scoring, or result construction.
Keeping the registry dependency-light also means importing the comparison
library does not import a product-specific adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from fugue.bench.library import validate_id

WANDB_MCP_REFERENCE_STUDY_ID = "wandb-mcp-release"
WANDB_MCP_REFERENCE_STUDY_VERSION = 1
WANDB_MCP_REFERENCE_STUDY_INTENT = "python-package-release-qualification"


@dataclass(frozen=True)
class ReferenceStudyBindingV1:
    """Identity of one reviewed, package-owned reference-study adapter."""

    id: str
    version: int
    intent: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReferenceStudyAdapter(Protocol):
    """Narrow adapter boundary consumed by comparison readiness."""

    def qualification_input_readiness(
        self,
        spec: Any,
        *,
        repo_root: Path,
    ) -> tuple[dict[str, str], list[str]]: ...

    def bound_release_note_coverage(
        self,
        spec: Any,
        *,
        readiness: Mapping[str, Any],
        repo_root: Path,
    ) -> tuple[dict[str, Any], ...]: ...

    def verify_source_drift(
        self,
        spec: Any,
        *,
        readiness: Mapping[str, Any],
        repo_root: Path,
        env: Mapping[str, str],
    ) -> Any: ...


def reference_study_binding_from_dict(raw: Any) -> ReferenceStudyBindingV1:
    if not isinstance(raw, Mapping):
        raise ValueError("execution reference_study must be a mapping")
    unknown = sorted(set(raw) - {"id", "version", "intent"})
    if unknown:
        raise ValueError(
            "execution reference_study has unknown fields: "
            + ", ".join(unknown)
        )
    adapter_id = validate_id(
        str(raw.get("id") or ""),
        kind="reference study adapter id",
    )
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("reference study adapter version must be a positive integer")
    intent = validate_id(
        str(raw.get("intent") or ""),
        kind="reference study intent",
    )
    binding = ReferenceStudyBindingV1(
        id=adapter_id,
        version=version,
        intent=intent,
    )
    # Fail at parse time for unsupported explicit adapters.  This is strict
    # authoring validation and does not import the adapter implementation.
    _adapter_module(binding)
    return binding


def infer_legacy_reference_study(execution: Any) -> ReferenceStudyBindingV1 | None:
    """Infer the historical W&B MCP adapter without changing old documents.

    The legacy quartet already uniquely identifies this reference study.  The
    inferred binding is used only for dispatch; callers must not serialize it
    back into the comparison or add a new digest to historical previews.
    """

    legacy_fields = (
        "evidence_lock",
        "source_conformance_receipt",
        "release_notes_lock",
        "mechanism_receipt",
    )
    if not any(getattr(execution, field, None) for field in legacy_fields):
        return None
    return ReferenceStudyBindingV1(
        id=WANDB_MCP_REFERENCE_STUDY_ID,
        version=WANDB_MCP_REFERENCE_STUDY_VERSION,
        intent=WANDB_MCP_REFERENCE_STUDY_INTENT,
    )


def resolve_reference_study_adapter(
    binding: ReferenceStudyBindingV1,
) -> ReferenceStudyAdapter:
    module_name = _adapter_module(binding)
    module = import_module(module_name)
    callback = getattr(module, "qualification_input_readiness", None)
    if not callable(callback):
        raise RuntimeError(
            f"reference study adapter {binding.id!r} does not expose readiness"
        )
    return module  # type: ignore[return-value]


def _adapter_module(binding: ReferenceStudyBindingV1) -> str:
    supported = {
        (
            WANDB_MCP_REFERENCE_STUDY_ID,
            WANDB_MCP_REFERENCE_STUDY_VERSION,
            WANDB_MCP_REFERENCE_STUDY_INTENT,
        ): "fugue.reference_studies.wandb_mcp_qualification",
    }
    key = (binding.id, binding.version, binding.intent)
    try:
        return supported[key]
    except KeyError as exc:
        raise ValueError(
            "unsupported reference study adapter binding: "
            f"{binding.id}@{binding.version} ({binding.intent})"
        ) from exc
