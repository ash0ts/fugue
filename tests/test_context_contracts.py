from fugue.bench.context import ContextSystemSpec
from fugue.bench.context_contracts import (
    ContextCapability,
    ContextDelivery,
    required_context_capabilities,
    resolve_context_capabilities,
)


def _spec(
    *,
    capabilities: frozenset[ContextCapability],
    deliveries: frozenset[ContextDelivery] = frozenset({"portable"}),
) -> ContextSystemSpec:
    return ContextSystemSpec(
        id="example",
        title="Example",
        provider="fugue.bench.context:EmptyContextProvider",
        version="1",
        capabilities=capabilities,
        deliveries=deliveries,
    )


def test_runner_requirements_are_typed_and_declared_requirements_are_additive() -> None:
    assert required_context_capabilities("harbor") == frozenset({"bind"})
    assert required_context_capabilities("retrieval", ["prepare"]) == frozenset(
        {"prepare", "retrieve"}
    )
    assert required_context_capabilities("sequence") == frozenset(
        {"ingest", "sequence"}
    )


def test_resolution_reports_delivery_before_capability_mismatch() -> None:
    resolution = resolve_context_capabilities(
        _spec(capabilities=frozenset({"prepare"})),
        delivery="native_mcp",
        runner="retrieval",
    )

    assert resolution.applicable is False
    assert resolution.code == "unsupported_delivery"
    assert resolution.missing == ()
    assert resolution.reason == (
        "context system example does not support native_mcp delivery"
    )


def test_resolution_exposes_complete_missing_capability_set() -> None:
    resolution = resolve_context_capabilities(
        _spec(capabilities=frozenset({"prepare"})),
        delivery="portable",
        runner="sequence",
        additional=["retrieve"],
    )

    assert resolution.applicable is False
    assert resolution.code == "missing_capabilities"
    assert resolution.required == frozenset({"ingest", "retrieve", "sequence"})
    assert resolution.missing == ("ingest", "retrieve", "sequence")
    assert resolution.reason == (
        "missing context capabilities: ingest, retrieve, sequence"
    )
