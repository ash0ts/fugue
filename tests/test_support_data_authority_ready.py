from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from types import ModuleType
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
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(
        _base_url: str,
        supplied_key: str,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert supplied_key == "secret"
        calls.append((method, path, body))
        if method == "GET":
            raise urllib.error.HTTPError(path, 404, "missing", {}, None)
        if path == "/v1/research":
            return {"id": body["research_id"] if body else ""}
        if path.endswith(":preview"):
            return {"preview_digest": "preview"}
        return {
            "id": "audit-1",
            "cohort_count": 1,
            "source_snapshot_digest": "source-digest",
        }

    monkeypatch.setattr(module, "_request", request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_ready.py",
            "--base-url",
            "http://fugue.test",
            "--api-key-file",
            str(api_key),
            "--entity",
            "demo-entity",
            "--research-id",
            "aria-support-data-authority-local-01",
        ],
    )

    assert module.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["research_id"] == "aria-support-data-authority-local-01"
    readiness_id = output["readiness_research_id"]
    assert readiness_id.startswith("readiness-")
    assert calls[0][1] == f"/v1/research/{readiness_id}"
    assert calls[1][2]["research_id"] == readiness_id
    assert all(
        "aria-support-data-authority-local-01" not in path
        for _method, path, _body in calls
    )
    assert all(
        "aria-support-data-authority-v1" not in path for _method, path, _body in calls
    )
