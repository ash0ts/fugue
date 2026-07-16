from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_AGENTS = {
    "fugue.agents:FugueHermes",
    "fugue.agents:FugueOpenClaw",
    "fugue.agents:FugueClaudeCode",
    "fugue.agents:FugueCodex",
}


def test_checked_in_dynamic_imports_resolve_and_agent_allowlist_is_exact() -> None:
    imports: set[str] = set()
    agents: set[str] = set()
    for root in (Path("configs"), Path("datasets")):
        for path in root.rglob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            for key, value in _walk(payload):
                if key in {"agent", "provider", "materializer"} and _import_path(value):
                    imports.add(value)
                if key == "agent" and _import_path(value):
                    agents.add(value)

    assert agents == SUPPORTED_AGENTS
    for import_path in sorted(imports):
        module_name, object_name = import_path.split(":", 1)
        module = importlib.import_module(module_name)
        assert getattr(module, object_name, None) is not None, import_path


def _walk(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _import_path(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("fugue.") and ":" in value
