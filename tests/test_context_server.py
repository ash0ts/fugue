from pathlib import Path

import pytest

from fugue.context_server import _selected_context_system


def test_context_server_loads_the_explicit_bound_config(tmp_path: Path) -> None:
    config = tmp_path / "study-context.yaml"
    config.write_text(
        """
id: study-rag
title: Study RAG
provider: fugue.bench.context:EmptyContextProvider
version: study-v1
capabilities: [prepare, retrieve, bind]
deliveries: [portable]
config: {mode: bm25}
"""
    )

    spec = _selected_context_system("study-rag", config, tmp_path / "not-fugue")

    assert spec.id == "study-rag"
    assert spec.version == "study-v1"
    assert spec.path == config


def test_context_server_rejects_bound_config_for_another_system(
    tmp_path: Path,
) -> None:
    config = tmp_path / "wrong-context.yaml"
    config.write_text(
        """
id: another-system
title: Another system
provider: fugue.bench.context:EmptyContextProvider
version: study-v1
capabilities: [prepare]
deliveries: [portable]
"""
    )

    with pytest.raises(ValueError, match="does not match requested system"):
        _selected_context_system("expected-system", config, tmp_path)
