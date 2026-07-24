from __future__ import annotations

import hmac
import json
import os
import secrets
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fugue.research.access import AGENT_SCOPES, token_digest
from fugue.research.contracts import validate_id

_DEFAULT_AGENT_RESEARCH_IDS = (
    "aria-enterprise-evidence-use-v1",
    "aria-support-data-authority-v1",
)


def bootstrap_container_secrets(
    repo_root: Path,
    *,
    wandb_api_key_file: Path | None = None,
    env_file: Path | None = None,
) -> dict[str, str]:
    root = repo_root.resolve()
    secret_dir = root / ".fugue" / "secrets"
    secret_dir.mkdir(parents=True, exist_ok=True)
    secret_dir.chmod(0o700)
    (root / ".fugue" / "trace-data").mkdir(parents=True, exist_ok=True)
    (root / ".fugue" / "runtime").mkdir(parents=True, exist_ok=True)
    (root / ".fugue" / "cache").mkdir(parents=True, exist_ok=True)

    agent_token = secret_dir / "research_api_key"
    if not agent_token.exists():
        _write_secret(agent_token, secrets.token_urlsafe(32))
    else:
        _make_compose_readable(agent_token)
    access_grants = secret_dir / "research_access_grants.json"
    _ensure_access_grants(access_grants, agent_token, root)

    record_token = secret_dir / "research_record_ingest_key"
    if not record_token.exists():
        _write_secret(record_token, secrets.token_urlsafe(32))
    else:
        _make_compose_readable(record_token)

    wandb_token = secret_dir / "wandb_api_key"
    if not wandb_token.exists():
        value = os.environ.get("WANDB_API_KEY", "").strip()
        if env_file is not None:
            value = _read_env_value(env_file, "WANDB_API_KEY")
        if wandb_api_key_file is not None:
            value = _read_secret_file(wandb_api_key_file)
        if not value:
            raise RuntimeError(
                "WANDB_API_KEY, --env-file, or --wandb-api-key-file is required "
                "for bootstrap"
            )
        _write_secret(wandb_token, value)
    else:
        _make_compose_readable(wandb_token)

    compose_environment = root / ".fugue" / "compose.env"
    _write_compose_environment(compose_environment, root)

    return {
        "compose_environment_file": str(compose_environment),
        "research_api_key_file": str(agent_token),
        "research_access_grants_file": str(access_grants),
        "research_record_ingest_key_file": str(record_token),
        "wandb_api_key_file": str(wandb_token),
        "trace_data_directory": str(root / ".fugue" / "trace-data"),
    }


def _write_compose_environment(path: Path, repo_root: Path) -> None:
    git_common_dir = repo_root / ".git"
    try:
        result = subprocess.run(
            (
                "git",
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        candidate = Path(result.stdout.strip()).resolve(strict=True)
        if candidate.is_dir():
            git_common_dir = candidate
    except (OSError, subprocess.SubprocessError):
        pass
    deployment_mode = os.environ.get("FUGUE_RESEARCH_DEPLOYMENT_MODE", "local").strip()
    docker_host = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock").strip()
    if not docker_host.startswith("unix://"):
        raise RuntimeError("research workers support only a local Unix Docker endpoint")
    configured_socket = Path(docker_host.removeprefix("unix://"))
    if deployment_mode != "local" and configured_socket == Path(
        "/var/run/docker.sock"
    ):
        raise RuntimeError(
            "non-local research workers require an explicitly configured rootless "
            "Docker endpoint"
        )
    socket = configured_socket.resolve()
    # Docker Desktop and OrbStack expose the socket as root:root inside their
    # Linux VM even when the macOS symlink target has the host user's group.
    # Native Linux preserves the socket's real Docker group.
    docker_gid = (
        0
        if sys.platform == "darwin"
        else socket.stat().st_gid
        if socket.exists()
        else os.getgid()
    )
    values = {
        "FUGUE_DOCKER_GID": str(docker_gid),
        "FUGUE_DOCKER_SOCKET": str(socket),
        "FUGUE_GIT_COMMON_DIR": str(git_common_dir.resolve()),
        "FUGUE_HOST_GID": str(os.getgid()),
        "FUGUE_HOST_REPO_ROOT": str(repo_root),
        "FUGUE_HOST_UID": str(os.getuid()),
        "FUGUE_RESEARCH_DEPLOYMENT_MODE": deployment_mode,
    }
    lines = [f"{key}={_dotenv_value(value)}" for key, value in sorted(values.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _ensure_access_grants(path: Path, token_path: Path, repo_root: Path) -> None:
    token = _read_secret_file(token_path)
    instance_path = repo_root / ".fugue" / "instance-id"
    if not instance_path.exists():
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(instance_path, flags, 0o600)
        try:
            os.write(descriptor, f"fugue-{secrets.token_hex(16)}\n".encode())
        finally:
            os.close(descriptor)
    instance_id = _read_secret_file(instance_path)
    if path.exists():
        raw = json.loads(_read_secret_file(path))
        digests = {
            str(item.get("token_digest") or "")
            for item in raw.get("grants", [])
            if isinstance(item, dict)
        }
        if not any(
            hmac.compare_digest(value, token_digest(token)) for value in digests
        ):
            raise RuntimeError(
                "research access grants do not match the existing Agent credential"
            )
        _make_compose_readable(path)
        return
    expires = (datetime.now(UTC) + timedelta(days=366)).isoformat()
    configured_ids = os.environ.get("FUGUE_RESEARCH_AGENT_RESEARCH_IDS", "").strip()
    research_ids = (
        tuple(
            validate_id(value.strip(), kind="research access id")
            for value in configured_ids.split(",")
            if value.strip()
        )
        if configured_ids
        else _DEFAULT_AGENT_RESEARCH_IDS
    )
    if not research_ids or len(set(research_ids)) != len(research_ids):
        raise RuntimeError("Agent Research IDs must be a unique non-empty list")
    value = {
        "schema_version": 1,
        "instance_id": instance_id,
        "grants": [
            {
                "schema_version": 1,
                "token_digest": token_digest(token),
                "subject": "external-research-agent",
                "instance_id": instance_id,
                "research_ids": list(research_ids),
                "scopes": sorted(AGENT_SCOPES),
                "expires_at": expires,
            }
        ],
    }
    _write_secret(path, json.dumps(value, sort_keys=True, separators=(",", ":")))


def _dotenv_value(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise RuntimeError("container path contains a newline")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$")
    return f'"{escaped}"'


def _write_secret(path: Path, value: str) -> None:
    # Compose implements file-backed secrets as bind mounts, so the container's
    # non-root control process needs read permission on the mounted inode.  The
    # containing directory remains host-private while the mounted file itself is
    # read-only to every container user.
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, f"{value}\n".encode())
    finally:
        os.close(descriptor)
    path.chmod(0o444)


def _make_compose_readable(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"container secret must be a regular file: {path}")
    path.chmod(0o444)


def _read_secret_file(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file() or resolved.stat().st_size > 65_536:
        raise RuntimeError("credential source must be a small regular file")
    return resolved.read_text(encoding="utf-8").strip()


def _read_env_value(path: Path, key: str) -> str:
    """Read one allowlisted dotenv value without evaluating shell syntax."""

    if key != "WANDB_API_KEY":
        raise ValueError("bootstrap credential is not allowlisted")
    text = _read_secret_file(path)
    selected = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()
        name, separator, raw_value = stripped.partition("=")
        if not separator or name.strip() != key:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not value:
            raise RuntimeError(f"{key} is empty in the credential environment file")
        if selected and selected != value:
            raise RuntimeError(
                f"{key} is declared more than once with different values"
            )
        selected = value
    return selected
