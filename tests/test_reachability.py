from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

SUPPORTED_AGENTS = {
    "fugue.agents:FugueHermes",
    "fugue.agents:FugueOpenClaw",
    "fugue.agents:FugueClaudeCode",
    "fugue.agents:FugueCodex",
}


def test_checked_in_dynamic_imports_are_declared_and_agent_allowlist_is_exact() -> None:
    imports, agents = _checked_in_imports()

    assert agents == SUPPORTED_AGENTS
    for import_path in sorted(imports):
        module_name, object_name = import_path.split(":", 1)
        spec = importlib.util.find_spec(module_name)
        assert spec is not None and spec.origin is not None, import_path
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        declared = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        declared.update(
            name.id
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for name in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(name, ast.Name)
        )
        declared.update(
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        )
        assert object_name in declared, import_path


@pytest.mark.skipif(
    importlib.util.find_spec("harbor") is None,
    reason="Harbor is a Python 3.13 execution dependency",
)
def test_checked_in_dynamic_imports_resolve_with_execution_dependencies() -> None:
    imports, _ = _checked_in_imports()
    for import_path in sorted(imports):
        module_name, object_name = import_path.split(":", 1)
        module = importlib.import_module(module_name)
        assert getattr(module, object_name, None) is not None, import_path


def _checked_in_imports() -> tuple[set[str], set[str]]:
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

    return imports, agents


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
