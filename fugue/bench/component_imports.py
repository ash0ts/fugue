from __future__ import annotations

import asyncio
import configparser
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import yaml

from fugue.bench.integrations import (
    IMPORTED_INTEGRATION_ROOT,
    MANAGED_MCP_RUNTIME_ROOT,
    load_integration,
)
from fugue.bench.sources import (
    SKILL_CACHE_ROOT,
    ResolvedSkill,
    digest_local_skill,
)

MCP_DRAFT_ROOT = Path(".fugue") / "imports" / "mcp" / "drafts"
MCP_LOCK_ROOT = Path(".fugue") / "imports" / "mcp" / "locks"
SKILL_DRAFT_ROOT = Path(".fugue") / "imports" / "skills" / "drafts"
SKILL_LOCK_ROOT = Path(".fugue") / "imports" / "skills" / "locks"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PYTHON_PACKAGE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9_.+!-]*$"
)
_NPM_PACKAGE_RE = re.compile(
    r"^(?:@[A-Za-z0-9_.-]+/)?[A-Za-z0-9_.-]+@[A-Za-z0-9][A-Za-z0-9_.+-]*$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_SECRET_NAME_RE = re.compile(
    r"(?:api.?key|credential|password|secret|token)", re.IGNORECASE
)


@dataclass(frozen=True)
class MCPImportDraftV1:
    schema_version: int
    id: str
    transport: str
    command: tuple[str, ...] = ()
    url: str | None = None
    required_env: tuple[str, ...] = ()
    source: str = "manual"
    server_name: str | None = None
    version_identity: str | None = None
    allowed_tools: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["command"] = list(self.command)
        value["required_env"] = list(self.required_env)
        value["allowed_tools"] = list(self.allowed_tools)
        return value


@dataclass(frozen=True)
class MCPImportLockV1:
    schema_version: int
    id: str
    transport: str
    source: str
    source_digest: str
    runtime_digest: str | None
    command: tuple[str, ...] = ()
    url: str | None = None
    version_identity: str | None = None
    required_env: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    tool_manifest: tuple[dict[str, Any], ...] = ()
    tool_manifest_digest: str | None = None
    server_info: dict[str, Any] | None = None
    support: str = "experimental"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["command"] = list(self.command)
        value["required_env"] = list(self.required_env)
        value["allowed_tools"] = list(self.allowed_tools)
        value["tool_manifest"] = list(self.tool_manifest)
        return value


@dataclass(frozen=True)
class SkillImportDraftV1:
    schema_version: int
    id: str
    kind: str
    source: str
    git_url: str | None = None
    commit: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImportedSkillLockV1:
    schema_version: int
    id: str
    declared_name: str
    digest: str
    source: str
    source_kind: str
    resolved_commit: str | None
    source_path: str | None
    executable_files: tuple[str, ...]
    total_files: int
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["executable_files"] = list(self.executable_files)
        return value


def import_mcp_config(
    config_path: Path,
    *,
    server: str,
    import_id: str,
    repo_root: Path,
) -> MCPImportDraftV1:
    _validate_id(import_id)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = _load_mcp_config(config_path)
    servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        raise ValueError(f"{config_path}: expected an mcpServers mapping")
    if server not in servers:
        raise ValueError(f"{config_path}: MCP server not found: {server}")
    selected = servers[server]
    if not isinstance(selected, dict):
        raise ValueError(f"{config_path}: MCP server {server!r} must be a mapping")
    draft = _draft_from_server(
        selected,
        import_id=import_id,
        source=config_path.resolve().as_posix(),
        server_name=server,
    )
    _write_json(_mcp_draft_path(repo_root, import_id), draft.to_dict())
    return draft


def add_mcp_command(
    import_id: str,
    command: list[str],
    *,
    repo_root: Path,
    required_env: tuple[str, ...] = (),
) -> MCPImportDraftV1:
    _validate_id(import_id)
    _validate_argv(command)
    _validate_env_names(required_env)
    draft = MCPImportDraftV1(
        schema_version=1,
        id=import_id,
        transport="stdio",
        command=tuple(command),
        required_env=tuple(required_env),
        source="argv",
        server_name=import_id,
    )
    _write_json(_mcp_draft_path(repo_root, import_id), draft.to_dict())
    return draft


def inspect_mcp_import(import_id: str, repo_root: Path) -> dict[str, Any]:
    draft = load_mcp_draft(import_id, repo_root)
    lock_path = _mcp_lock_path(repo_root, import_id)
    lock = _load_json(lock_path) if lock_path.is_file() else None
    return {"draft": draft.to_dict(), "lock": lock}


def load_mcp_draft(import_id: str, repo_root: Path) -> MCPImportDraftV1:
    _validate_id(import_id)
    path = _mcp_draft_path(repo_root, import_id)
    raw = _load_json(path)
    _reject_unknown(
        raw,
        {
            "schema_version",
            "id",
            "transport",
            "command",
            "url",
            "required_env",
            "source",
            "server_name",
            "version_identity",
            "allowed_tools",
        },
        path,
    )
    if raw.get("schema_version") != 1 or raw.get("id") != import_id:
        raise ValueError(f"{path}: unsupported or mismatched MCP draft")
    command = _string_tuple(raw.get("command"))
    required_env = _string_tuple(raw.get("required_env"))
    allowed_tools = _string_tuple(raw.get("allowed_tools"))
    _validate_env_names(required_env)
    transport = str(raw.get("transport") or "")
    if transport == "stdio":
        _validate_argv(list(command))
    elif transport not in {"streamable-http", "sse"}:
        raise ValueError(f"{path}: unsupported MCP transport {transport!r}")
    return MCPImportDraftV1(
        schema_version=1,
        id=import_id,
        transport=transport,
        command=command,
        url=str(raw["url"]) if raw.get("url") else None,
        required_env=required_env,
        source=str(raw.get("source") or ""),
        server_name=str(raw["server_name"]) if raw.get("server_name") else None,
        version_identity=(
            str(raw["version_identity"]) if raw.get("version_identity") else None
        ),
        allowed_tools=allowed_tools,
    )


def lock_mcp_import(
    import_id: str,
    repo_root: Path,
    *,
    acknowledge_package_code: bool = False,
) -> MCPImportLockV1:
    draft = load_mcp_draft(import_id, repo_root)
    source_digest = _stable_digest(draft.to_dict())
    if draft.transport == "stdio":
        manifest, server_info = _probe_stdio_manifest(draft)
        discovered = tuple(str(item["name"]) for item in manifest)
        missing = sorted(set(draft.allowed_tools) - set(discovered))
        if missing:
            raise ValueError(
                "declared allowed MCP tools were not advertised: " + ", ".join(missing)
            )
        allowed_tools = draft.allowed_tools or discovered
        manifest_digest = _stable_digest(list(manifest))
        runtime_digest, command = _materialize_stdio_runtime(
            draft,
            repo_root,
            acknowledge_package_code=acknowledge_package_code,
        )
        support = "supported"
        runtime = {
            "type": "managed",
            "source": (
                MANAGED_MCP_RUNTIME_ROOT
                / import_id
                / runtime_digest.removeprefix("sha256:")
            ).as_posix(),
            "digest": runtime_digest,
            "command": list(command),
        }
        interface: dict[str, Any] = {
            "type": "mcp",
            "name": import_id,
            "transport": "stdio",
            "allowed_tools": list(allowed_tools),
        }
    else:
        if not draft.url:
            raise ValueError("remote MCP import requires a URL")
        _validate_remote_mcp_url(draft.url)
        runtime_digest = None
        command = ()
        manifest = ()
        server_info = None
        manifest_digest = None
        allowed_tools = draft.allowed_tools
        support = "supported" if draft.version_identity else "experimental"
        runtime = {"type": "external", "url": draft.url}
        interface = {
            "type": "mcp",
            "name": import_id,
            "transport": draft.transport,
            "url": draft.url,
        }
    lock = MCPImportLockV1(
        schema_version=1,
        id=import_id,
        transport=draft.transport,
        source=draft.source,
        source_digest=source_digest,
        runtime_digest=runtime_digest,
        command=tuple(command),
        url=draft.url,
        version_identity=draft.version_identity,
        required_env=draft.required_env,
        allowed_tools=allowed_tools,
        tool_manifest=manifest,
        tool_manifest_digest=manifest_digest,
        server_info=server_info,
        support=support,
    )
    _write_json(_mcp_lock_path(repo_root, import_id), lock.to_dict())
    lock_digest = _stable_digest(lock.to_dict())
    integration = {
        "id": import_id,
        "version": lock_digest,
        "support": support,
        "runtime": runtime,
        "interfaces": [interface],
        "capabilities": ["mcp"],
        "required_env": list(draft.required_env),
        "allowed_hosts": (
            [str(urlparse(draft.url or "").hostname)]
            if draft.transport != "stdio"
            else []
        ),
    }
    integration_path = repo_root / IMPORTED_INTEGRATION_ROOT / f"{import_id}.yaml"
    integration_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        integration_path,
        yaml.safe_dump(integration, sort_keys=False).encode(),
        mode=0o600,
    )
    load_integration(import_id, repo_root)
    return lock


def _probe_stdio_manifest(
    draft: MCPImportDraftV1,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    async def probe() -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "MCP conformance inspection requires "
                "`uv sync --extra research-worker`"
            ) from exc

        environment = {
            name: os.environ[name]
            for name in draft.required_env
            if os.environ.get(name)
        }
        missing = sorted(set(draft.required_env) - set(environment))
        if missing:
            raise ValueError(
                "MCP inspection requires runtime credential environment: "
                + ", ".join(missing)
            )
        parameters = StdioServerParameters(
            command=draft.command[0],
            args=list(draft.command[1:]),
            env=environment or None,
        )
        async with stdio_client(parameters) as streams:
            async with ClientSession(*streams) as session:
                initialized = await asyncio.wait_for(
                    session.initialize(), timeout=30
                )
                listed = await asyncio.wait_for(session.list_tools(), timeout=30)
        tools = tuple(
            sorted(
                (
                    {
                        "name": str(tool.name),
                        "description": str(tool.description or ""),
                        "input_schema": dict(tool.inputSchema or {}),
                    }
                    for tool in listed.tools
                ),
                key=lambda value: value["name"],
            )
        )
        if not tools:
            raise ValueError("MCP server initialized but advertised no tools")
        server = getattr(initialized, "serverInfo", None)
        info = {
            "name": str(getattr(server, "name", "") or ""),
            "version": str(getattr(server, "version", "") or ""),
        }
        return tools, info

    try:
        return asyncio.run(probe())
    except TimeoutError as exc:
        raise RuntimeError("MCP initialization or tools/list timed out") from exc


def import_skill(
    source: str,
    *,
    repo_root: Path,
    import_id: str | None = None,
) -> SkillImportDraftV1:
    if source.startswith("git+"):
        git_url, commit, source_path = _parse_git_skill_source(source)
        skill_id = _validate_id(import_id or PurePosixPath(source_path).name)
        draft = SkillImportDraftV1(
            schema_version=1,
            id=skill_id,
            kind="git",
            source=source,
            git_url=git_url,
            commit=commit,
            path=source_path,
        )
    else:
        path = Path(source).expanduser().resolve()
        if not (path / "SKILL.md").is_file():
            raise ValueError(f"skill source must contain SKILL.md: {path}")
        _, declared_name = digest_local_skill(path, fallback_name=path.name)
        skill_id = _validate_id(import_id or declared_name)
        draft = SkillImportDraftV1(
            schema_version=1,
            id=skill_id,
            kind="local",
            source=path.as_posix(),
        )
    _write_json(_skill_draft_path(repo_root, skill_id), draft.to_dict())
    return draft


def inspect_skill_import(skill_id: str, repo_root: Path) -> dict[str, Any]:
    draft = load_skill_draft(skill_id, repo_root)
    source_path, cleanup = _skill_source_path(draft, repo_root)
    try:
        digest, declared_name = digest_local_skill(
            source_path, fallback_name=skill_id
        )
        inventory = _skill_inventory(source_path)
        return {
            "draft": draft.to_dict(),
            "declared_name": declared_name,
            "digest": digest,
            **inventory,
        }
    finally:
        cleanup()


def load_skill_draft(skill_id: str, repo_root: Path) -> SkillImportDraftV1:
    _validate_id(skill_id)
    path = _skill_draft_path(repo_root, skill_id)
    raw = _load_json(path)
    _reject_unknown(
        raw,
        {"schema_version", "id", "kind", "source", "git_url", "commit", "path"},
        path,
    )
    if raw.get("schema_version") != 1 or raw.get("id") != skill_id:
        raise ValueError(f"{path}: unsupported or mismatched Skill draft")
    kind = str(raw.get("kind") or "")
    if kind not in {"local", "git"}:
        raise ValueError(f"{path}: unsupported Skill source kind {kind!r}")
    return SkillImportDraftV1(
        schema_version=1,
        id=skill_id,
        kind=kind,
        source=str(raw.get("source") or ""),
        git_url=str(raw["git_url"]) if raw.get("git_url") else None,
        commit=str(raw["commit"]) if raw.get("commit") else None,
        path=str(raw["path"]) if raw.get("path") else None,
    )


def lock_skill_import(skill_id: str, repo_root: Path) -> ImportedSkillLockV1:
    draft = load_skill_draft(skill_id, repo_root)
    source_path, cleanup = _skill_source_path(draft, repo_root)
    try:
        digest, declared_name = digest_local_skill(
            source_path, fallback_name=skill_id
        )
        inventory = _skill_inventory(source_path)
        cache = (
            repo_root
            / SKILL_CACHE_ROOT
            / digest.removeprefix("sha256:")
            / declared_name
        )
        _copy_skill_bundle(source_path, cache)
        actual = digest_local_skill(cache, fallback_name=declared_name)
        if actual != (digest, declared_name):
            raise ValueError("materialized Skill bundle does not match its review")
        lock = ImportedSkillLockV1(
            schema_version=1,
            id=skill_id,
            declared_name=declared_name,
            digest=digest,
            source=draft.source,
            source_kind=draft.kind,
            resolved_commit=draft.commit,
            source_path=draft.path,
            executable_files=tuple(inventory["executable_files"]),
            total_files=int(inventory["total_files"]),
            total_bytes=int(inventory["total_bytes"]),
        )
        _write_json(_skill_lock_path(repo_root, skill_id), lock.to_dict())
        return lock
    finally:
        cleanup()


def resolve_imported_skill(skill_id: str, repo_root: Path) -> ResolvedSkill | None:
    path = _skill_lock_path(repo_root, skill_id)
    if not path.is_file():
        return None
    raw = _load_json(path)
    _reject_unknown(
        raw,
        {
            "schema_version",
            "id",
            "declared_name",
            "digest",
            "source",
            "source_kind",
            "resolved_commit",
            "source_path",
            "executable_files",
            "total_files",
            "total_bytes",
        },
        path,
    )
    if raw.get("schema_version") != 1 or raw.get("id") != skill_id:
        raise ValueError(f"{path}: unsupported or mismatched Skill lock")
    declared_name = str(raw["declared_name"])
    digest = str(raw["digest"])
    cache = repo_root / SKILL_CACHE_ROOT / digest.removeprefix("sha256:") / declared_name
    actual = digest_local_skill(cache, fallback_name=declared_name)
    if actual != (digest, declared_name):
        raise ValueError(f"imported Skill {skill_id} cache drifted from its lock")
    return ResolvedSkill(
        id=skill_id,
        declared_name=declared_name,
        path=cache,
        digest=digest,
        source_url=str(raw["source"]),
        requested_ref=(
            str(raw["resolved_commit"]) if raw.get("resolved_commit") else None
        ),
        resolved_commit=(
            str(raw["resolved_commit"]) if raw.get("resolved_commit") else None
        ),
        source_path=str(raw["source_path"]) if raw.get("source_path") else None,
        license_status="reviewed-import",
        policy_version="import-v1",
    )


def _load_mcp_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        servers = raw.get("mcp_servers")
        if servers is None:
            servers = raw.get("mcpServers")
        return {"mcpServers": servers}
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: MCP config must be an object")
        return raw
    raise ValueError("MCP import supports Codex TOML or mcpServers JSON")


def _draft_from_server(
    raw: dict[str, Any],
    *,
    import_id: str,
    source: str,
    server_name: str,
) -> MCPImportDraftV1:
    allowed = {
        "command",
        "args",
        "env",
        "url",
        "transport",
        "type",
        "version",
        "allowed_tools",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"MCP server {server_name!r} has unsupported field(s): {', '.join(unknown)}"
        )
    env = raw.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("MCP env must be a mapping of names to references")
    env_names: list[str] = []
    for name, value in env.items():
        name = str(name)
        if not _ENV_RE.fullmatch(name):
            raise ValueError(f"invalid MCP environment name: {name!r}")
        if not _is_env_reference(value, name):
            raise ValueError(
                f"MCP environment {name} contains a literal value; only references are importable"
            )
        env_names.append(name)
    url = str(raw["url"]) if raw.get("url") else None
    if url:
        transport = str(raw.get("transport") or raw.get("type") or "streamable-http")
        if transport in {"http", "streamable_http"}:
            transport = "streamable-http"
        _validate_remote_mcp_url(url)
        command: tuple[str, ...] = ()
    else:
        executable = raw.get("command")
        args = raw.get("args") or []
        if not isinstance(executable, str) or not isinstance(args, list):
            raise ValueError("stdio MCP requires command as a string and args as an array")
        command = (executable, *(str(value) for value in args))
        _validate_argv(list(command))
        transport = "stdio"
    allowed_tools = _string_tuple(raw.get("allowed_tools"))
    return MCPImportDraftV1(
        schema_version=1,
        id=import_id,
        transport=transport,
        command=command,
        url=url,
        required_env=tuple(sorted(env_names)),
        source=source,
        server_name=server_name,
        version_identity=str(raw["version"]) if raw.get("version") else None,
        allowed_tools=allowed_tools,
    )


def _materialize_stdio_runtime(
    draft: MCPImportDraftV1,
    repo_root: Path,
    *,
    acknowledge_package_code: bool,
) -> tuple[str, tuple[str, ...]]:
    command = list(draft.command)
    kind, package, executable, extra = _classify_stdio_command(command)
    if extra:
        raise ValueError(
            "MCP package commands may not include undeclared runtime arguments in V1"
        )
    if kind in {"python", "node"} and not acknowledge_package_code:
        raise ValueError(
            "locking a package-backed MCP server requires --acknowledge-package-code"
        )
    parent = repo_root / MANAGED_MCP_RUNTIME_ROOT / draft.id
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=parent))
    try:
        if kind == "python":
            _install_python_tool(package, executable, staging)
        elif kind == "node":
            _install_node_tool(package, executable, staging)
        elif kind == "oci":
            raise ValueError(
                "digest-pinned OCI MCP imports require an HTTP/SSE gateway in V1"
            )
        else:
            raise ValueError(
                "stdio MCP commands must be pinned uvx or npx package declarations"
            )
        digest = _runtime_directory_digest(staging)
        destination = parent / digest.removeprefix("sha256:")
        if destination.exists():
            shutil.rmtree(staging)
        else:
            os.replace(staging, destination)
        return digest, (f"/fugue-components/{draft.id}/bin/server",)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _classify_stdio_command(
    command: list[str],
) -> tuple[str, str, str, list[str]]:
    executable = PurePosixPath(command[0]).name
    if executable == "uvx":
        if len(command) < 4 or command[1] != "--from":
            raise ValueError("uvx MCP declarations must use: uvx --from PACKAGE==VERSION COMMAND")
        package = command[2]
        if not _PYTHON_PACKAGE_RE.fullmatch(package):
            raise ValueError("uvx MCP packages must use an exact name==version")
        return "python", package, command[3], command[4:]
    if executable == "npx":
        values = command[1:]
        if values[:1] in (["-y"], ["--yes"]):
            values = values[1:]
        if not values or not _NPM_PACKAGE_RE.fullmatch(values[0]):
            raise ValueError("npx MCP packages must use an exact package@version")
        package = values[0]
        return "node", package, "", values[1:]
    if executable in {"docker", "podman"}:
        images = [value for value in command[1:] if _DIGEST_IMAGE_RE.fullmatch(value)]
        return "oci", images[0] if len(images) == 1 else "", "", []
    return "unknown", "", "", command[1:]


def _install_python_tool(package: str, executable: str, destination: Path) -> None:
    site = destination / "site"
    site.mkdir()
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            site.as_posix(),
            "--python-platform",
            "x86_64-manylinux_2_28",
            "--python-version",
            "3.12",
            "--only-binary",
            ":all:",
            package,
        ],
        check=True,
        timeout=180,
    )
    entrypoint = _python_entrypoint(site, executable)
    bin_dir = destination / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / "server"
    module, function = entrypoint
    launcher.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.path.insert(0, '/fugue-components/"
        f"{destination.parent.name}/site')\n"
        f"from {module} import {function} as _entry\n"
        "raise SystemExit(_entry())\n",
        encoding="utf-8",
    )
    launcher.chmod(0o555)
    _make_tree_read_only(destination)


def _python_entrypoint(site: Path, executable: str) -> tuple[str, str]:
    matches: list[str] = []
    for path in site.glob("*.dist-info/entry_points.txt"):
        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        if parser.has_option("console_scripts", executable):
            matches.append(parser.get("console_scripts", executable))
    if len(matches) != 1:
        raise ValueError(
            f"Python package must expose exactly one {executable!r} console script"
        )
    target = matches[0].split("[", 1)[0].strip()
    if ":" not in target:
        raise ValueError(f"unsupported Python console entry point: {target}")
    module, function = target.split(":", 1)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", function
    ):
        raise ValueError(f"unsupported Python console entry point: {target}")
    return module, function


def _install_node_tool(package: str, executable: str, destination: Path) -> None:
    package_root = destination / "package"
    package_root.mkdir()
    subprocess.run(
        [
            "npm",
            "install",
            "--ignore-scripts",
            "--omit=dev",
            "--no-audit",
            "--no-fund",
            "--prefix",
            package_root.as_posix(),
            package,
        ],
        check=True,
        timeout=180,
    )
    if list(package_root.rglob("*.node")):
        raise ValueError("native Node addons are not portable into Harbor attempts")
    package_name = package.rsplit("@", 1)[0]
    manifest = package_root / "node_modules" / package_name / "package.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    bin_value = raw.get("bin")
    if isinstance(bin_value, str):
        relative = bin_value
    elif isinstance(bin_value, dict):
        choices = (
            [bin_value[executable]]
            if executable and executable in bin_value
            else list(bin_value.values())
        )
        if len(choices) != 1:
            raise ValueError("Node package must expose one unambiguous executable")
        relative = str(choices[0])
    else:
        raise ValueError("Node MCP package does not expose a bin entry")
    script = (manifest.parent / relative).resolve()
    if manifest.parent.resolve() not in script.parents or not script.is_file():
        raise ValueError("Node package bin entry escaped the installed package")
    bin_dir = destination / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / "server"
    launcher.write_text(
        "#!/bin/sh\n"
        f"exec node /fugue-components/{destination.parent.name}/package/"
        f"node_modules/{package_name}/{PurePosixPath(relative).as_posix()} \"$@\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o555)
    _remove_symlinks(package_root)
    _make_tree_read_only(destination)


def _skill_source_path(
    draft: SkillImportDraftV1, repo_root: Path
) -> tuple[Path, Any]:
    if draft.kind == "local":
        source = Path(draft.source)
        return source, lambda: None
    assert draft.git_url and draft.commit and draft.path
    temporary = Path(tempfile.mkdtemp(prefix="fugue-skill-"))
    checkout = temporary / "repo"
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", draft.git_url, checkout],
        check=True,
        timeout=120,
    )
    subprocess.run(
        ["git", "-C", checkout.as_posix(), "checkout", "--detach", draft.commit],
        check=True,
        timeout=60,
    )
    source = checkout / draft.path
    return source, lambda: shutil.rmtree(temporary)


def _parse_git_skill_source(value: str) -> tuple[str, str, str]:
    raw = value.removeprefix("git+")
    base, marker, fragment = raw.partition("#path=")
    if not marker or not fragment:
        raise ValueError("Git Skill sources require #path=RELATIVE_PATH")
    url, separator, commit = base.rpartition("@")
    if not separator or not _SHA_RE.fullmatch(commit):
        raise ValueError("Git Skill sources require a full 40-character commit SHA")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Git Skill sources must use public https://github.com URLs")
    selected = PurePosixPath(fragment)
    if selected.is_absolute() or any(
        part in {"", ".", ".."} for part in selected.parts
    ):
        raise ValueError("Git Skill path must be a safe relative path")
    return url, commit, selected.as_posix()


def _skill_inventory(root: Path) -> dict[str, Any]:
    executable: list[str] = []
    count = 0
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Skill bundles may not contain symlinks: {path}")
        if not path.is_file():
            continue
        count += 1
        total += path.stat().st_size
        relative = path.relative_to(root).as_posix()
        if path.stat().st_mode & 0o111:
            executable.append(relative)
        if path.stat().st_size <= 1024 * 1024:
            content = path.read_bytes()
            if _looks_like_secret(relative, content):
                raise ValueError(f"Skill bundle may contain a secret: {relative}")
    return {
        "total_files": count,
        "total_bytes": total,
        "executable_files": executable,
    }


def _looks_like_secret(relative: str, content: bytes) -> bool:
    if _SECRET_NAME_RE.search(PurePosixPath(relative).name):
        return True
    text = content.decode("utf-8", errors="ignore")
    return bool(
        re.search(r"(?i)(?:api.?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_-]{16,}", text)
        or re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", text)
    )


def _copy_skill_bundle(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    staging = destination.parent / f".{destination.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging, symlinks=False)
    _make_tree_read_only(staging)
    os.replace(staging, destination)


def _runtime_directory_digest(root: Path) -> str:
    hasher = hashlib.sha256()
    found = False
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"managed runtime may not contain symlinks: {path}")
        if not path.is_file():
            continue
        found = True
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        hasher.update(f"{relative}\0{mode}\0{len(content)}\0".encode())
        hasher.update(content)
    if not found:
        raise ValueError("managed runtime is empty")
    return f"sha256:{hasher.hexdigest()}"


def _remove_symlinks(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            path.unlink()


def _make_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise ValueError(f"prepared components may not contain symlinks: {path}")
        if path.is_file():
            executable = bool(path.stat().st_mode & 0o111)
            path.chmod(0o555 if executable else 0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _validate_argv(command: list[str]) -> None:
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise ValueError("MCP command must be a non-empty argv array")
    forbidden = {"sh", "bash", "zsh", "fish", "cmd", "powershell", "pwsh"}
    if PurePosixPath(command[0]).name.lower() in forbidden:
        raise ValueError("shell-backed MCP commands are not importable")
    if any(
        token in value
        for value in command
        for token in (";", "&&", "||", "\n", "\r", "`", "$(")
    ):
        raise ValueError("MCP argv may not contain shell composition")


def _validate_env_names(values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError("required MCP environment names must be unique")
    for name in values:
        if not _ENV_RE.fullmatch(name):
            raise ValueError(f"invalid MCP environment name: {name!r}")


def _is_env_reference(value: Any, name: str) -> bool:
    return isinstance(value, str) and value in {
        f"${{{name}}}",
        f"${name}",
        f"env:{name}",
    }


def _validate_remote_mcp_url(value: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("remote MCP URLs must be credential-free HTTPS origins or paths")


def _validate_id(value: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"invalid component id: {value!r}")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("expected an array of strings")
    return tuple(value)


def _mcp_draft_path(repo_root: Path, import_id: str) -> Path:
    return repo_root / MCP_DRAFT_ROOT / f"{import_id}.json"


def _mcp_lock_path(repo_root: Path, import_id: str) -> Path:
    return repo_root / MCP_LOCK_ROOT / f"{import_id}.json"


def _skill_draft_path(repo_root: Path, skill_id: str) -> Path:
    return repo_root / SKILL_DRAFT_ROOT / f"{skill_id}.json"


def _skill_lock_path(repo_root: Path, skill_id: str) -> Path:
    return repo_root / SKILL_LOCK_ROOT / f"{skill_id}.json"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
        mode=0o600,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object")
    return raw


def _atomic_write(path: Path, content: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            handle = -1
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if handle >= 0:
            os.close(handle)
        if os.path.exists(temporary):
            os.unlink(temporary)


def _reject_unknown(
    raw: dict[str, Any], allowed: set[str], path: Path
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown field(s): {', '.join(unknown)}")


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"
