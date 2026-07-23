from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


def _module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "research"
        / "support-data-authority"
        / "verify_ready.py"
    )
    spec = importlib.util.spec_from_file_location("support_data_authority_ready", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readiness_probe_uses_the_explicit_research_identity(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    module = _module()
    api_key = tmp_path / "research-key"
    api_key.write_text("secret\n", encoding="utf-8")
    sources = tmp_path / "trace-sources.yaml"
    sources.write_text("version: 1\nsources: []\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    class Adapter:
        source = SimpleNamespace(source_digest="source-digest")

        def read(self, draft: Any) -> tuple[dict[str, Any], ...]:
            captured["draft"] = draft
            return ({"status": "success", "operation": "support"},)

    class Registry:
        def get(self, source_id: str) -> Adapter:
            assert source_id == "northstar-support-agent"
            return Adapter()

    def from_file(path: Path, *, env: dict[str, str]) -> Registry:
        assert path == sources
        captured["env"] = env
        return Registry()

    monkeypatch.setattr(module.TraceSourceRegistry, "from_file", from_file)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_ready.py",
            "--trace-sources-file",
            str(sources),
            "--trace-api-key-file",
            str(api_key),
            "--trace-server-url",
            "http://trace.test",
            "--entity",
            "demo-entity",
            "--research-id",
            "aria-support-data-authority-local-01",
        ],
    )

    assert module.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["research_id"] == "aria-support-data-authority-local-01"
    assert output["source_digest"] == "source-digest"
    assert captured["draft"].study_id == "aria-support-data-authority-local-01"
    assert captured["draft"].selection.project == (
        "demo-entity/northstar-support-agent"
    )
    assert captured["env"]["WANDB_API_KEY_FILE"] == str(api_key)
    assert captured["env"]["WF_TRACE_SERVER_URL"] == "http://trace.test"
