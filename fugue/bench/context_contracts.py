from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, cast

ContextCapability = Literal[
    "prepare", "retrieve", "bind", "ingest", "sequence", "serve"
]
ContextDelivery = Literal["portable", "native_mcp"]
SupportLevel = Literal["supported", "experimental", "not_applicable", "disabled"]
WorkloadRunner = Literal["harbor", "retrieval", "sequence"]
CapabilityResolutionCode = Literal[
    "applicable", "unsupported_system", "unsupported_delivery", "missing_capabilities"
]

CONTEXT_CAPABILITIES = frozenset(
    {"prepare", "retrieve", "bind", "ingest", "sequence", "serve"}
)
CONTEXT_DELIVERIES = frozenset({"portable", "native_mcp"})
WORKLOAD_RUNNERS = frozenset({"harbor", "retrieval", "sequence"})

_RUNNER_CAPABILITIES: dict[WorkloadRunner, frozenset[ContextCapability]] = {
    "harbor": frozenset({"bind"}),
    "retrieval": frozenset({"retrieve"}),
    "sequence": frozenset({"ingest", "sequence"}),
}


class ContextCapabilitySource(Protocol):
    id: str
    capabilities: frozenset[ContextCapability]
    deliveries: frozenset[ContextDelivery]
    support: SupportLevel


@dataclass(frozen=True)
class ContextCapabilityResolution:
    applicable: bool
    code: CapabilityResolutionCode
    required: frozenset[ContextCapability]
    missing: tuple[ContextCapability, ...] = ()
    reason: str | None = None


def context_capabilities(
    values: Iterable[object], *, kind: str
) -> frozenset[ContextCapability]:
    selected = frozenset(str(value).strip() for value in values if str(value).strip())
    invalid = sorted(selected - CONTEXT_CAPABILITIES)
    if invalid:
        raise ValueError(f"{kind}: unknown context capabilities: {', '.join(invalid)}")
    return cast(frozenset[ContextCapability], selected)


def required_context_capabilities(
    runner: str,
    additional: Iterable[object] = (),
) -> frozenset[ContextCapability]:
    if runner not in WORKLOAD_RUNNERS:
        raise ValueError(f"unknown workload runner: {runner}")
    selected_runner = cast(WorkloadRunner, runner)
    declared = context_capabilities(additional, kind=f"{runner} workload")
    return _RUNNER_CAPABILITIES[selected_runner] | declared


def resolve_context_capabilities(
    spec: ContextCapabilitySource,
    *,
    delivery: ContextDelivery,
    runner: str,
    additional: Iterable[object] = (),
) -> ContextCapabilityResolution:
    required = required_context_capabilities(runner, additional)
    if spec.support in {"not_applicable", "disabled"}:
        return ContextCapabilityResolution(
            applicable=False,
            code="unsupported_system",
            required=required,
            reason=f"context system {spec.id} is {spec.support}",
        )
    if delivery not in spec.deliveries:
        return ContextCapabilityResolution(
            applicable=False,
            code="unsupported_delivery",
            required=required,
            reason=f"context system {spec.id} does not support {delivery} delivery",
        )
    missing = tuple(sorted(required - spec.capabilities))
    if missing:
        return ContextCapabilityResolution(
            applicable=False,
            code="missing_capabilities",
            required=required,
            missing=missing,
            reason=f"missing context capabilities: {', '.join(missing)}",
        )
    return ContextCapabilityResolution(
        applicable=True,
        code="applicable",
        required=required,
    )
