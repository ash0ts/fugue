from __future__ import annotations

import os
import secrets
from pathlib import Path


def bootstrap_container_secrets(
    repo_root: Path,
    *,
    wandb_api_key_file: Path | None = None,
) -> dict[str, str]:
    root = repo_root.resolve()
    secret_dir = root / ".fugue" / "secrets"
    secret_dir.mkdir(parents=True, exist_ok=True)
    (root / ".fugue" / "trace-data").mkdir(parents=True, exist_ok=True)

    agent_token = secret_dir / "research_api_key"
    if not agent_token.exists():
        _write_secret(agent_token, secrets.token_urlsafe(32))

    wandb_token = secret_dir / "wandb_api_key"
    if not wandb_token.exists():
        value = os.environ.get("WANDB_API_KEY", "").strip()
        if wandb_api_key_file is not None:
            value = wandb_api_key_file.read_text(encoding="utf-8").strip()
        if not value:
            raise RuntimeError(
                "WANDB_API_KEY or --wandb-api-key-file is required for bootstrap"
            )
        _write_secret(wandb_token, value)

    return {
        "research_api_key_file": str(agent_token),
        "wandb_api_key_file": str(wandb_token),
        "trace_data_directory": str(root / ".fugue" / "trace-data"),
    }


def _write_secret(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"{value}\n".encode())
    finally:
        os.close(descriptor)
