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
    trace_wandb_api_key_file: Path | None = None,
    env_file: Path | None = None,
) -> dict[str, str]:
    root = repo_root.resolve()
    explicit_value: str | None = None
    if env_file is not None:
        explicit_value = _read_env_value(env_file, "WANDB_API_KEY")
    if wandb_api_key_file is not None:
        explicit_value = _read_secret_file(wandb_api_key_file)
    explicit_trace_value = (
        _read_secret_file(trace_wandb_api_key_file)
        if trace_wandb_api_key_file is not None
        else None
    )
    if trace_wandb_api_key_file is not None and not explicit_trace_value:
        raise RuntimeError("the explicit Weave credential file is empty")
    if (
        explicit_value
        and explicit_trace_value
        and hmac.compare_digest(explicit_value, explicit_trace_value)
    ):
        raise RuntimeError(
            "the cloud inference key and local evidence key must be different"
        )
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
    if explicit_value is not None:
        if not explicit_value:
            raise RuntimeError(
                "the explicit W&B credential source contains no WANDB_API_KEY"
            )
        _sync_secret(wandb_token, explicit_value)
    elif not wandb_token.exists():
        value = os.environ.get("WANDB_API_KEY", "").strip()
        if not value:
            raise RuntimeError(
                "WANDB_API_KEY, --env-file, or --wandb-api-key-file is required "
                "for bootstrap"
            )
        _write_secret(wandb_token, value)
    else:
        _make_compose_readable(wandb_token)

    trace_token = secret_dir / "trace_wandb_api_key"
    trace_value = (
        explicit_trace_value
        if explicit_trace_value is not None
        else os.environ.get("FUGUE_WEAVE_API_KEY", "").strip()
    )
    inference_value = _read_secret_file(wandb_token)
    if trace_wandb_api_key_file is not None and hmac.compare_digest(
        inference_value, trace_value
    ):
        raise RuntimeError(
            "the cloud inference key and local evidence key must be different"
        )
    if trace_value:
        _sync_secret(trace_token, trace_value)
    elif not trace_token.exists():
        # Preserve the single-key local experience for existing users. The
        # demo supplies an explicit local trace credential and therefore never
        # copies the hosted inference key into its evidence runtime.
        _write_secret(trace_token, _read_secret_file(wandb_token))
    else:
        _make_compose_readable(trace_token)

    compose_environment = root / ".fugue" / "compose.env"
    _write_compose_environment(compose_environment, root)
    credential_routing_receipt = root / ".fugue" / "credential-routing.json"
    _write_credential_routing_receipt(
        credential_routing_receipt,
        credentials_distinct=not hmac.compare_digest(
            _read_secret_file(wandb_token),
            _read_secret_file(trace_token),
        ),
    )

    return {
        "compose_environment_file": str(compose_environment),
        "research_api_key_file": str(agent_token),
        "research_access_grants_file": str(access_grants),
        "research_record_ingest_key_file": str(record_token),
        "wandb_api_key_file": str(wandb_token),
        "trace_wandb_api_key_file": str(trace_token),
        "credential_routing_receipt": str(credential_routing_receipt),
        "trace_data_directory": str(root / ".fugue" / "trace-data"),
    }


def _write_credential_routing_receipt(
    path: Path,
    *,
    credentials_distinct: bool,
) -> None:
    inference_project = os.environ.get(
        "FUGUE_WANDB_INFERENCE_PROJECT", "wandb/fugue-experiments"
    ).strip()
    evidence_project = os.environ.get(
        "FUGUE_WEAVE_PROJECT", "wandb/fugue-experiments"
    ).strip()
    receipt = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "inference": {
            "credential": "wandb_api_key",
            "base_url": os.environ.get(
                "FUGUE_WANDB_INFERENCE_BASE_URL",
                "https://api.inference.wandb.ai/v1",
            ).strip(),
            "project": inference_project,
            "required_model": os.environ.get(
                "FUGUE_WANDB_INFERENCE_REQUIRED_MODEL", "zai-org/GLM-5.2"
            ).strip(),
        },
        "evidence": {
            "credential": "trace_wandb_api_key",
            "base_url": os.environ.get(
                "FUGUE_WEAVE_BASE_URL", "https://api.wandb.ai"
            ).strip(),
            "project": evidence_project,
        },
        "credentials_distinct": credentials_distinct,
    }
    validation_time = os.environ.get(
        "FUGUE_WANDB_INFERENCE_VALIDATED_AT", ""
    ).strip()
    model_available = os.environ.get(
        "FUGUE_WANDB_INFERENCE_MODEL_AVAILABLE", ""
    ).strip()
    if validation_time:
        receipt["inference"]["validated_at"] = validation_time
    if model_available:
        if model_available not in {"true", "false"}:
            raise RuntimeError(
                "FUGUE_WANDB_INFERENCE_MODEL_AVAILABLE must be true or false"
            )
        receipt["inference"]["model_available"] = model_available == "true"
    evidence_validation_time = os.environ.get(
        "FUGUE_WEAVE_VALIDATED_AT", ""
    ).strip()
    if evidence_validation_time:
        receipt["evidence"]["validated_at"] = evidence_validation_time
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            os.write(
                descriptor,
                (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode(),
            )
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


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
        grants = [item for item in raw.get("grants", []) if isinstance(item, dict)]
        matching = next(
            (
                item
                for item in grants
                if hmac.compare_digest(
                    str(item.get("token_digest") or ""), token_digest(token)
                )
            ),
            None,
        )
        if matching is None:
            raise RuntimeError(
                "research access grants do not match the existing Agent credential"
            )
        configured_ids = _agent_research_ids()
        merged_ids = sorted(
            {
                *(
                    str(value)
                    for value in matching.get("research_ids", [])
                    if str(value).strip()
                ),
                *configured_ids,
            }
        )
        if merged_ids != matching.get("research_ids"):
            matching["research_ids"] = merged_ids
            _sync_secret(
                path,
                json.dumps(raw, sort_keys=True, separators=(",", ":")),
            )
        else:
            _make_compose_readable(path)
        return
    expires = (datetime.now(UTC) + timedelta(days=366)).isoformat()
    research_ids = _agent_research_ids()
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


def _agent_research_ids() -> tuple[str, ...]:
    configured = os.environ.get("FUGUE_RESEARCH_AGENT_RESEARCH_IDS", "").strip()
    values = (
        tuple(
            validate_id(value.strip(), kind="research access id")
            for value in configured.split(",")
            if value.strip()
        )
        if configured
        else _DEFAULT_AGENT_RESEARCH_IDS
    )
    if not values or len(set(values)) != len(values):
        raise RuntimeError("Agent Research IDs must be a unique non-empty list")
    return values


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


def _sync_secret(path: Path, value: str) -> None:
    """Atomically make an explicitly supplied credential authoritative."""

    if path.exists():
        current = _read_secret_file(path)
        if hmac.compare_digest(current, value):
            _make_compose_readable(path)
            return
    elif path.is_symlink():
        raise RuntimeError(f"container secret must not be a symlink: {path}")

    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        _write_secret(temporary, value)
        os.replace(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


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
