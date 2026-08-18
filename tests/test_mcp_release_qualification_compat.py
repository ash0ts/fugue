from __future__ import annotations

import importlib


def test_legacy_mcp_qualification_import_is_the_reference_implementation(
    monkeypatch,
) -> None:
    legacy = importlib.import_module("fugue.bench.mcp_release_qualification")
    implementation = importlib.import_module(
        "fugue.reference_studies.wandb_mcp_qualification_core"
    )

    assert legacy is implementation

    sentinel = object()
    monkeypatch.setattr(legacy, "_legacy_import_probe", sentinel, raising=False)
    assert implementation._legacy_import_probe is sentinel
