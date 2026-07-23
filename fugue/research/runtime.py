from __future__ import annotations

import os
from pathlib import Path


def load_secret_file_environment() -> None:
    """Load operator-configured *_FILE secrets into this process only."""

    for key, path_value in tuple(os.environ.items()):
        if not key.endswith("_FILE") or not path_value.strip():
            continue
        target = key.removesuffix("_FILE")
        if target in os.environ:
            continue
        if not (
            target.endswith("_API_KEY")
            or target.endswith("_TOKEN")
            or target in {"FUGUE_RESEARCH_API_KEY"}
        ):
            continue
        path = Path(path_value)
        if path.stat().st_size > 65_536:
            raise RuntimeError(f"secret file for {target} exceeds 64 KiB")
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise RuntimeError(f"secret file for {target} is empty")
        os.environ[target] = value
