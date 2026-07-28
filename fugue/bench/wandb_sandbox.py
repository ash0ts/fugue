from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from fugue.bench.agent_runtime import (
    AGENT_RUNTIME_MOUNT,
)
from fugue.bench.agent_runtime import prepare_runtime as prepare_agent_runtime
from fugue.bench.agent_runtime import (
    read_runtime_lock as read_agent_runtime_lock,
)
from fugue.bench.agent_runtime import (
    runtime_spec as agent_runtime_spec,
)
from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.integrations import (
    MANAGED_MCP_RUNTIME_ROOT,
    load_integration,
)

WANDB_RUNTIME_SCHEMA_VERSION = 1
WANDB_RUNTIME_LOCK_PATH = Path(".fugue/wandb-serverless-runtime.lock.json")
WANDB_ATTESTATION_NAME = "wandb-serverless-attestation.json"
WANDB_ENVIRONMENT_IMPORT = (
    "fugue.bench.wandb_sandbox:FugueWandbEnvironment"
)
_BASE_IMAGE = (
    "python:3.13.14-slim-trixie@"
    "sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91"
)
_UV_VERSION = "0.11.27"
_IMAGE_DIGEST = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SECRET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,126}$")
_SECRET_ENV_NAME = re.compile(
    r"(?:api.?key|credential|password|secret|token)",
    re.IGNORECASE,
)
_SUPPORTED_HARNESSES = frozenset({"claude-code", "openclaw"})
_SECRET_ENV_NAMES = frozenset({"ANTHROPIC_API_KEY", "WANDB_API_KEY"})
_ALLOWED_EXECUTION_FIELDS = frozenset(
    {
        "type",
        "runtime_lock",
        "cpus",
        "memory_mb",
        "storage_mb",
        "network_mode",
        "max_timeout_seconds",
        "max_lifetime_seconds",
        "delete",
    }
)


@dataclass(frozen=True)
class WandbRuntimeImageV1:
    harness: str
    platform: str
    image: str
    published: bool
    public_pull_verified: bool
    agent_runtime: dict[str, Any]
    assets: tuple[dict[str, Any], ...]
    probes: tuple[str, ...]
    sbom: dict[str, Any]
    scan: dict[str, Any]


@dataclass(frozen=True)
class WandbRuntimeManifestV1:
    schema_version: int
    backend: str
    created_at: str
    source: dict[str, Any]
    comparisons: tuple[dict[str, str], ...]
    images: tuple[WandbRuntimeImageV1, ...]
    required_secrets: dict[str, str]
    manifest_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WandbRuntimeLockV1:
    schema_version: int
    backend: str
    manifest: dict[str, Any]
    manifest_digest: str
    lock_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_wandb_runtime(
    *,
    comparisons: Sequence[Path],
    repo_root: Path,
    platform: str,
    image: str,
    push: bool,
    output_manifest: Path,
    sbom_dir: Path | None = None,
) -> WandbRuntimeManifestV1:
    from fugue.bench.comparison import check_comparison, load_comparison

    root = repo_root.resolve()
    if platform != "linux/amd64":
        raise ValueError("W&B Serverless qualification requires linux/amd64")
    if not comparisons:
        raise ValueError("at least one comparison is required")
    _require_command("docker")
    _require_command("syft")
    _require_command("grype")
    comparison_records: list[dict[str, str]] = []
    harnesses: set[str] = set()
    integration_ids: set[str] = set()
    for path in comparisons:
        spec = load_comparison(path, repo_root=root)
        readiness = check_comparison(spec, repo_root=root)
        if readiness.task_count < 1 or readiness.estimated_cells < 1:
            raise ValueError(f"comparison {spec.id} has no planned work")
        if readiness.actual_changes != ("integrations",):
            raise ValueError(
                f"comparison {spec.id} must vary only the MCP integration"
            )
        harnesses.update(spec.execution.harnesses)
        integration_ids.update(
            item["id"]
            for candidate in (spec.baseline, spec.candidate)
            for item in candidate.integrations
        )
        comparison_records.append(
            {
                "id": spec.id,
                "path": path.as_posix(),
                "spec_digest": spec.spec_digest,
                "taskset_digest": readiness.taskset_digest,
            }
        )
    unsupported = sorted(harnesses - _SUPPORTED_HARNESSES)
    if unsupported:
        raise ValueError(
            "W&B runtime supports only the direct flagship harnesses: "
            + ", ".join(unsupported)
        )
    integrations = _resolved_integrations(integration_ids, root)
    output = output_manifest.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = (sbom_dir or output.parent / "runtime-reports").resolve()
    reports.mkdir(parents=True, exist_ok=True)
    images: list[WandbRuntimeImageV1] = []
    for harness in sorted(harnesses):
        agent_lock = read_agent_runtime_lock(harness, root, "amd64")
        if agent_lock is None:
            agent_lock = prepare_agent_runtime(
                harness,
                repo_root=root,
                architecture="amd64",
            )
        runtime = agent_runtime_spec(harness)
        assert runtime is not None
        local_image = str(agent_lock.get("image") or "")
        locked_image_id = str(agent_lock.get("image_id") or "")
        inspected = _inspect_image(local_image)
        if inspected.get("Id") != locked_image_id:
            raise RuntimeError(f"{harness} Agent runtime image drifted from its lock")
        target = _harness_image_tag(image, harness)
        with tempfile.TemporaryDirectory(
            prefix=f"fugue-wandb-{harness}-"
        ) as temporary:
            context = Path(temporary)
            assets = _write_build_context(
                context,
                repo_root=root,
                harness=harness,
                agent_image=local_image,
                agent_image_id=locked_image_id,
                integrations=integrations,
            )
            metadata = context / "build-metadata.json"
            command = [
                "docker",
                "buildx",
                "build",
                "--platform",
                platform,
                "--provenance=true",
                "--sbom=true",
                "--metadata-file",
                metadata.as_posix(),
                "--tag",
                target,
                "--push" if push else "--load",
                context.as_posix(),
            ]
            subprocess.run(command, cwd=root, check=True, timeout=3600)
            build_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        digest = str(build_metadata.get("containerimage.digest") or "")
        published_image = (
            f"{target.split('@', 1)[0]}@{digest}"
            if push and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            else target
        )
        public_verified = False
        if push:
            if not _IMAGE_DIGEST.fullmatch(published_image):
                raise RuntimeError(
                    f"registry did not return an immutable digest for {target}"
                )
            _verify_anonymous_pull(published_image)
            public_verified = True
        sbom_path = reports / f"{harness}.cyclonedx.json"
        scan_path = reports / f"{harness}.grype.json"
        _write_sbom(published_image, sbom_path)
        _scan_image(published_image, scan_path)
        probes = (
            f"{AGENT_RUNTIME_MOUNT}/bin/{_harness_command(harness)} --version",
            "python -c 'import fugue, fugue.mcp_proxy'",
            *(
                f"test -x /fugue-components/{item['id']}/bin/server"
                for item in integrations
            ),
        )
        _run_image_probes(published_image, probes, platform)
        images.append(
            WandbRuntimeImageV1(
                harness=harness,
                platform=platform,
                image=published_image,
                published=push,
                public_pull_verified=public_verified,
                agent_runtime=dict(agent_lock),
                assets=tuple(assets),
                probes=tuple(probes),
                sbom={
                    "format": "cyclonedx-json",
                    "path": _relative_or_absolute(sbom_path, root),
                    "sha256": _file_digest(sbom_path),
                },
                scan={
                    "scanner": "grype",
                    "fail_on": "high",
                    "status": "passed",
                    "path": _relative_or_absolute(scan_path, root),
                    "sha256": _file_digest(scan_path),
                },
            )
        )
    source = _source_identity(root)
    unsigned = WandbRuntimeManifestV1(
        schema_version=WANDB_RUNTIME_SCHEMA_VERSION,
        backend="wandb-serverless",
        created_at=_now(),
        source=source,
        comparisons=tuple(
            sorted(comparison_records, key=lambda item: item["id"])
        ),
        images=tuple(images),
        required_secrets={
            "ANTHROPIC_API_KEY": "fugue-anthropic-api-key",
            "WANDB_API_KEY": "fugue-wandb-api-key",
        },
    )
    manifest = WandbRuntimeManifestV1(
        **{
            **asdict(unsigned),
            "manifest_digest": stable_digest(
                {
                    **unsigned.to_dict(),
                    "manifest_digest": "",
                }
            ),
        }
    )
    atomic_write_json(output, manifest.to_dict())
    return validate_wandb_runtime_manifest(manifest.to_dict(), require_published=push)


def validate_wandb_runtime_manifest(
    raw: Mapping[str, Any],
    *,
    require_published: bool,
) -> WandbRuntimeManifestV1:
    value = dict(raw)
    expected_fields = {
        "schema_version",
        "backend",
        "created_at",
        "source",
        "comparisons",
        "images",
        "required_secrets",
        "manifest_digest",
    }
    if set(value) != expected_fields:
        raise ValueError("W&B runtime manifest has an invalid shape")
    supplied = str(value.get("manifest_digest") or "")
    unsigned = dict(value)
    unsigned["manifest_digest"] = ""
    if not _SHA256.fullmatch(supplied) or stable_digest(unsigned) != supplied:
        raise ValueError("W&B runtime manifest digest does not match")
    if (
        value["schema_version"] != WANDB_RUNTIME_SCHEMA_VERSION
        or value["backend"] != "wandb-serverless"
    ):
        raise ValueError("W&B runtime manifest identity is unsupported")
    comparisons = value["comparisons"]
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("W&B runtime manifest must lock comparisons")
    comparison_ids: set[str] = set()
    for item in comparisons:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {"id", "path", "spec_digest", "taskset_digest"}
            or not str(item["id"])
            or str(item["id"]) in comparison_ids
            or Path(str(item["path"])).is_absolute()
            or ".." in Path(str(item["path"])).parts
            or not _SHA256.fullmatch(str(item["spec_digest"]))
            or not _SHA256.fullmatch(str(item["taskset_digest"]))
        ):
            raise ValueError("W&B comparison manifest entry is invalid")
        comparison_ids.add(str(item["id"]))
    source = value["source"]
    if (
        not isinstance(source, Mapping)
        or set(source) != {"commit", "tree", "fugue_tree_sha256"}
        or not _GIT_SHA.fullmatch(str(source["commit"]))
        or not _GIT_SHA.fullmatch(str(source["tree"]))
        or not _SHA256.fullmatch(str(source["fugue_tree_sha256"]))
    ):
        raise ValueError("W&B runtime source identity is invalid")
    secrets = value["required_secrets"]
    if (
        not isinstance(secrets, Mapping)
        or set(secrets) != _SECRET_ENV_NAMES
        or any(
            not _SECRET_NAME.fullmatch(str(name))
            for name in secrets.values()
        )
    ):
        raise ValueError("W&B runtime secret mapping is invalid")
    image_values = value["images"]
    if not isinstance(image_values, list) or not image_values:
        raise ValueError("W&B runtime manifest must contain images")
    images: list[WandbRuntimeImageV1] = []
    harnesses: set[str] = set()
    for item in image_values:
        if not isinstance(item, Mapping) or set(item) != {
            "harness",
            "platform",
            "image",
            "published",
            "public_pull_verified",
            "agent_runtime",
            "assets",
            "probes",
            "sbom",
            "scan",
        }:
            raise ValueError("W&B runtime image entry is invalid")
        image = WandbRuntimeImageV1(
            harness=str(item.get("harness") or ""),
            platform=str(item.get("platform") or ""),
            image=str(item.get("image") or ""),
            published=bool(item.get("published")),
            public_pull_verified=bool(item.get("public_pull_verified")),
            agent_runtime=dict(_mapping(item.get("agent_runtime"), "agent runtime")),
            assets=tuple(
                dict(_mapping(asset, "runtime asset"))
                for asset in _sequence(item.get("assets"), "runtime assets")
            ),
            probes=tuple(
                str(probe)
                for probe in _sequence(item.get("probes"), "runtime probes")
            ),
            sbom=dict(_mapping(item.get("sbom"), "runtime SBOM")),
            scan=dict(_mapping(item.get("scan"), "runtime scan")),
        )
        if (
            image.harness not in _SUPPORTED_HARNESSES
            or image.harness in harnesses
            or image.platform != "linux/amd64"
        ):
            raise ValueError("W&B runtime image harness or platform is invalid")
        if require_published and (
            not image.published
            or not image.public_pull_verified
            or not _IMAGE_DIGEST.fullmatch(image.image)
        ):
            raise ValueError("W&B runtime image is not publicly digest-locked")
        if (
            image.sbom.get("format") != "cyclonedx-json"
            or set(image.sbom) != {"format", "path", "sha256"}
            or not _SHA256.fullmatch(str(image.sbom.get("sha256") or ""))
            or image.scan.get("status") != "passed"
            or image.scan.get("fail_on") != "high"
            or set(image.scan)
            != {"scanner", "fail_on", "status", "path", "sha256"}
            or image.scan.get("scanner") != "grype"
            or not _SHA256.fullmatch(str(image.scan.get("sha256") or ""))
            or not image.probes
        ):
            raise ValueError("W&B runtime image qualification evidence is invalid")
        for asset in image.assets:
            if (
                set(asset) != {"kind", "source", "target", "sha256"}
                or not str(asset["target"]).startswith("/")
                or not _SHA256.fullmatch(str(asset["sha256"]))
            ):
                raise ValueError("W&B runtime asset is invalid")
        harnesses.add(image.harness)
        images.append(image)
    return WandbRuntimeManifestV1(
        schema_version=WANDB_RUNTIME_SCHEMA_VERSION,
        backend="wandb-serverless",
        created_at=str(value["created_at"]),
        source=dict(_mapping(value["source"], "runtime source")),
        comparisons=tuple(dict(item) for item in comparisons),
        images=tuple(images),
        required_secrets={str(k): str(v) for k, v in secrets.items()},
        manifest_digest=supplied,
    )


def lock_wandb_runtime(
    manifest_path: Path,
    *,
    output: Path = WANDB_RUNTIME_LOCK_PATH,
) -> WandbRuntimeLockV1:
    manifest = validate_wandb_runtime_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        require_published=True,
    )
    unsigned = WandbRuntimeLockV1(
        schema_version=WANDB_RUNTIME_SCHEMA_VERSION,
        backend="wandb-serverless",
        manifest=manifest.to_dict(),
        manifest_digest=manifest.manifest_digest,
    )
    lock = WandbRuntimeLockV1(
        **{
            **asdict(unsigned),
            "lock_digest": stable_digest(
                {
                    **unsigned.to_dict(),
                    "lock_digest": "",
                }
            ),
        }
    )
    atomic_write_json(output.resolve(), lock.to_dict())
    return read_wandb_runtime_lock(output)


def read_wandb_runtime_lock(path: Path) -> WandbRuntimeLockV1:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "backend",
        "manifest",
        "manifest_digest",
        "lock_digest",
    }:
        raise ValueError("W&B runtime lock has an invalid shape")
    supplied = str(raw.get("lock_digest") or "")
    unsigned = dict(raw)
    unsigned["lock_digest"] = ""
    if not _SHA256.fullmatch(supplied) or stable_digest(unsigned) != supplied:
        raise ValueError("W&B runtime lock digest does not match")
    manifest = validate_wandb_runtime_manifest(
        _mapping(raw["manifest"], "runtime manifest"),
        require_published=True,
    )
    if (
        raw["schema_version"] != WANDB_RUNTIME_SCHEMA_VERSION
        or raw["backend"] != "wandb-serverless"
        or raw["manifest_digest"] != manifest.manifest_digest
    ):
        raise ValueError("W&B runtime lock manifest digest does not match")
    return WandbRuntimeLockV1(
        schema_version=int(raw["schema_version"]),
        backend=str(raw["backend"]),
        manifest=manifest.to_dict(),
        manifest_digest=manifest.manifest_digest,
        lock_digest=supplied,
    )


def wandb_execution_identity(
    environment: Mapping[str, Any],
    *,
    harness: str,
    repo_root: Path,
) -> dict[str, Any] | None:
    if str(environment.get("type") or "docker") != "wandb":
        return None
    _reject_unknown_environment(environment)
    lock_path = _runtime_lock_path(environment, repo_root)
    lock = read_wandb_runtime_lock(lock_path)
    image = _image_for_harness(lock, harness)
    return {
        "backend": "wandb-serverless",
        "lock_digest": lock.lock_digest,
        "manifest_digest": lock.manifest_digest,
        "runtime_image": image["image"],
        "harness": harness,
        "resources": {
            key: environment[key]
            for key in ("cpus", "memory_mb", "storage_mb")
            if key in environment
        },
        "network_mode": str(environment.get("network_mode") or "public"),
        "delete": bool(environment.get("delete", True)),
    }


def bind_wandb_job_environment(
    environment: Mapping[str, Any],
    *,
    harness: str,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    identity = wandb_execution_identity(
        environment,
        harness=harness,
        repo_root=repo_root,
    )
    if identity is None:
        return dict(environment), None
    if str(environment.get("network_mode") or "public") != "public":
        raise ValueError(
            "the MCP qualification needs W&B public egress; Serverless does "
            "not enforce a host allowlist, so this is an explicit limitation"
        )
    if not bool(environment.get("delete", True)):
        raise ValueError("W&B qualification sandboxes must be deleted")
    harbor = {
        "docker_image": identity["runtime_image"],
        "network_mode": "public",
        "cpus": int(environment.get("cpus") or 2),
        "memory_mb": int(environment.get("memory_mb") or 4096),
        "storage_mb": int(environment.get("storage_mb") or 10240),
        "gpus": 0,
    }
    return harbor, identity


def wandb_harbor_command(
    config_path: Path,
    *,
    environment: Mapping[str, Any],
    repo_root: Path,
) -> list[str]:
    if str(environment.get("type") or "docker") != "wandb":
        return ["harbor", "run", "--config", config_path.as_posix()]
    lock_path = _runtime_lock_path(environment, repo_root)
    read_wandb_runtime_lock(lock_path)
    return [
        "harbor",
        "run",
        "--config",
        config_path.as_posix(),
        "--env",
        WANDB_ENVIRONMENT_IMPORT,
        "--no-force-build",
        "--delete",
        "--cpus",
        "guarantee",
        "--memory",
        "guarantee",
        "--ek",
        f"fugue_wandb_lock_path={lock_path.as_posix()}",
        "--ek",
        f"max_timeout_seconds={int(environment.get('max_timeout_seconds') or 1800)}",
        "--ek",
        (
            "max_lifetime_seconds="
            f"{int(environment.get('max_lifetime_seconds') or 2400)}"
        ),
    ]


def wandb_doctor(
    lock: WandbRuntimeLockV1,
    *,
    env_file: Path,
) -> dict[str, Any]:
    from fugue.bench.operator import load_env

    env = load_env(env_file)
    missing = [
        name for name in ("WANDB_API_KEY",)
        if not str(env.get(name) or "").strip()
    ]
    if missing:
        raise RuntimeError("W&B doctor requires " + ", ".join(missing))
    previous = os.environ.get("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = str(env["WANDB_API_KEY"])
    try:
        from harbor.environments.wandb import WandbEnvironment

        WandbEnvironment.preflight()
        import wandb.sandbox as sdk

        image = _image_for_harness(lock, "openclaw")["image"]
        tag = f"fugue-doctor-{lock.lock_digest[:20]}"
        sandbox = sdk.Sandbox(
            container_image=image,
            tags=["fugue", "fugue-doctor", tag],
            network=sdk.NetworkOptions(egress_mode="internet"),
            max_timeout_seconds=300,
            max_lifetime_seconds=600,
            resources={
                "requests": {"cpu": "1", "memory": "1024Mi"},
                "limits": {"cpu": "1", "memory": "1024Mi"},
            },
        )
        sandbox.start().result()
        sandbox.wait(timeout=300)
        try:
            result = sandbox.exec(
                [
                    "bash",
                    "-lc",
                    (
                        "test -x /opt/fugue-agent-runtime/bin/openclaw && "
                        "python -c 'import fugue, fugue.mcp_proxy'"
                    ),
                ],
                check=True,
                timeout_seconds=60,
            ).result()
            sandbox_id = str(sandbox.sandbox_id)
        finally:
            sandbox.stop(missing_ok=True).result()
            sdk.Sandbox.delete(
                str(sandbox.sandbox_id),
                missing_ok=True,
            ).result()
        remaining = [
            item
            for item in sdk.Sandbox.list(
                tags=[tag],
                include_stopped=True,
            ).result()
            if str(item.sandbox_id) == sandbox_id
        ]
        if remaining:
            raise RuntimeError("W&B doctor left a sandbox behind")
        return {
            "backend": "wandb-serverless",
            "lock_digest": lock.lock_digest,
            "sandbox_id": sandbox_id,
            "probe_return_code": int(result.returncode),
            "deleted": True,
            "orphans": 0,
        }
    finally:
        if previous is None:
            os.environ.pop("WANDB_API_KEY", None)
        else:
            os.environ["WANDB_API_KEY"] = previous


try:
    from harbor.environments.wandb import WandbEnvironment as _WandbEnvironment
except ImportError:  # pragma: no cover - exercised by optional-extra checks
    _WandbEnvironment = object  # type: ignore[assignment,misc]


class FugueWandbEnvironment(_WandbEnvironment):  # type: ignore[misc]
    """W&B Serverless with exact runtime and secret-inheritance attestations."""

    _provider_label: ClassVar[str] = "fugue-wandb"

    def __init__(
        self,
        *args: Any,
        fugue_wandb_lock_path: str,
        secrets: Sequence[Any] | None = None,
        tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if secrets:
            raise ValueError(
                "Fugue W&B secrets come only from the reviewed runtime lock"
            )
        self._fugue_lock = read_wandb_runtime_lock(
            Path(fugue_wandb_lock_path)
        )
        manifest = self._fugue_lock.manifest
        secret_specs = [
            {
                "name": name,
                "env_var": env_name,
            }
            for env_name, name in sorted(
                manifest["required_secrets"].items()
            )
        ]
        locked_tags = [
            "fugue",
            "fugue-wandb-serverless",
            f"fl-{self._fugue_lock.lock_digest[:20]}",
        ]
        super().__init__(
            *args,
            secrets=secret_specs,
            tags=[*locked_tags, *(tags or ())],
            **kwargs,
        )
        expected = {
            str(item["image"]): str(item["harness"])
            for item in manifest["images"]
        }
        selected = str(self.task_env_config.docker_image or "")
        if selected not in expected:
            raise ValueError(
                "effective W&B runtime image is absent from the runtime lock"
            )
        self._fugue_harness = expected[selected]
        self._fugue_sandbox_id: str | None = None

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        self._fugue_sandbox_id = self._sb_id(self._sandbox)
        self._write_attestation(state="running", deleted=False)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any:
        clean, exports = _remote_secret_env(env or {})
        wrapped = command
        if exports:
            wrapped = " && ".join([*exports, command])
        return await super().exec(
            wrapped,
            cwd=cwd,
            env=clean,
            timeout_sec=timeout_sec,
            user=user,
        )

    def prepare_managed_secret_environment(
        self,
        values: Mapping[str, str],
    ) -> tuple[dict[str, str], list[str]]:
        """Resolve per-command secret aliases from remote named secrets only."""

        return _remote_secret_env(values, reject_unknown=True)

    async def stop(self, delete: bool) -> None:
        if not delete:
            raise ValueError("Fugue W&B sandboxes must be deleted")
        sandbox_id = self._fugue_sandbox_id
        await super().stop(delete=True)
        if sandbox_id:
            import wandb.sandbox as sdk

            remaining: list[Any] = []
            for attempt in range(3):
                remaining = await asyncio.to_thread(
                    lambda: [
                        item
                        for item in sdk.Sandbox.list(
                            tags=[
                                f"fl-{self._fugue_lock.lock_digest[:20]}"
                            ],
                            include_stopped=True,
                        ).result()
                        if str(item.sandbox_id) == sandbox_id
                    ]
                )
                if not remaining:
                    break
                if attempt < 2:
                    await asyncio.sleep(1)
            if remaining:
                raise RuntimeError(
                    f"W&B Serverless sandbox {sandbox_id} was not deleted"
                )
        self._write_attestation(
            state="deleted",
            deleted=True,
            sandbox_id=sandbox_id,
            orphans=0,
        )

    def _write_attestation(
        self,
        *,
        state: str,
        deleted: bool,
        sandbox_id: str | None = None,
        orphans: int | None = None,
    ) -> None:
        self.trial_paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "backend": "wandb-serverless",
            "lock_digest": self._fugue_lock.lock_digest,
            "manifest_digest": self._fugue_lock.manifest_digest,
            "harness": self._fugue_harness,
            "runtime_image": str(self.task_env_config.docker_image),
            "sandbox_id": sandbox_id or self._fugue_sandbox_id,
            "state": state,
            "deleted": deleted,
            "orphans": orphans,
            "secret_delivery": "wandb-secrets-manager",
            "secret_env_names": sorted(_SECRET_ENV_NAMES),
            "raw_secret_overlays_forwarded": False,
            "recorded_at": _now(),
        }
        record["attestation_digest"] = stable_digest(record)
        atomic_write_json(
            self.trial_paths.artifacts_dir / WANDB_ATTESTATION_NAME,
            record,
        )


def _remote_secret_env(
    values: Mapping[str, str],
    *,
    reject_unknown: bool = False,
) -> tuple[dict[str, str], list[str]]:
    clean: dict[str, str] = {}
    exports: list[str] = []
    secret_values = {
        name: os.environ.get(name, "")
        for name in _SECRET_ENV_NAMES
        if os.environ.get(name)
    }
    for key, raw_value in values.items():
        value = str(raw_value)
        matching = [
            name
            for name, secret in secret_values.items()
            if value == secret
        ]
        if key in _SECRET_ENV_NAMES and key in matching:
            continue
        template = value
        for name, secret in secret_values.items():
            template = template.replace(secret, f"${{{name}}}")
            encoded = base64.b64encode(f"api:{secret}".encode()).decode()
            template = template.replace(
                encoded,
                "$("
                f"printf %s \"api:${{{name}}}\""
                " | base64 | tr -d '\\n'"
                ")",
            )
        if template == value and reject_unknown:
            raise ValueError(
                f"W&B Serverless cannot derive secret environment {key} "
                "from a reviewed named secret"
            )
        if template == value and _SECRET_ENV_NAME.search(str(key)):
            raise ValueError(
                f"W&B Serverless refuses unreviewed secret environment {key}"
            )
        if template == value:
            clean[str(key)] = value
            continue
        escaped = (
            template.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("`", "\\`")
        )
        exports.append(f"export {shlex.quote(str(key))}=\"{escaped}\"")
    return clean, exports


def _resolved_integrations(
    integration_ids: set[str],
    repo_root: Path,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for integration_id in sorted(integration_ids):
        spec = load_integration(integration_id, repo_root)
        if (
            spec.runtime.type != "managed"
            or spec.runtime.platform != "linux/amd64"
            or not spec.runtime.source
            or not spec.runtime.digest
        ):
            raise ValueError(
                "W&B runtime requires a managed linux/amd64 MCP lock: "
                f"{integration_id}"
            )
        source = (repo_root / spec.runtime.source).resolve()
        allowed_root = (repo_root / MANAGED_MCP_RUNTIME_ROOT).resolve()
        if allowed_root not in source.parents or not source.is_dir():
            raise ValueError(f"managed MCP runtime is unavailable: {integration_id}")
        digest = _managed_runtime_digest(source)
        if f"sha256:{digest}" != spec.runtime.digest:
            raise ValueError(f"managed MCP runtime drifted: {integration_id}")
        result.append(
            {
                "id": integration_id,
                "source": source,
                "digest": digest,
            }
        )
    if not result:
        raise ValueError("W&B runtime requires locked MCP integrations")
    return result


def _write_build_context(
    context: Path,
    *,
    repo_root: Path,
    harness: str,
    agent_image: str,
    agent_image_id: str,
    integrations: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    source = context / "source"
    source.mkdir()
    for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
        shutil.copy2(repo_root / name, source / name)
    shutil.copytree(repo_root / "fugue", source / "fugue")
    integration_root = context / "integrations"
    integration_root.mkdir()
    assets = [
        {
            "kind": "fugue-source",
            "source": "fugue",
            "target": "/fugue-src/fugue",
            "sha256": _tree_digest(repo_root / "fugue"),
        },
        {
            "kind": "dependency-lock",
            "source": "uv.lock",
            "target": "/fugue-src/uv.lock",
            "sha256": _file_digest(repo_root / "uv.lock"),
        },
        {
            "kind": "project-metadata",
            "source": "pyproject.toml",
            "target": "/fugue-src/pyproject.toml",
            "sha256": _file_digest(repo_root / "pyproject.toml"),
        },
    ]
    for item in integrations:
        destination = integration_root / str(item["id"])
        shutil.copytree(Path(item["source"]), destination)
        assets.append(
            {
                "kind": "mcp-runtime",
                "source": _relative_or_absolute(Path(item["source"]), repo_root),
                "target": f"/fugue-components/{item['id']}",
                "sha256": str(item["digest"]),
            }
        )
    agent_digest = _tree_digest_from_image(agent_image, AGENT_RUNTIME_MOUNT)
    assets.append(
        {
            "kind": "agent-runtime",
            "source": agent_image_id,
            "target": AGENT_RUNTIME_MOUNT,
            "sha256": agent_digest,
        }
    )
    copies = "\n".join(
        f"COPY integrations/{item['id']} /fugue-components/{item['id']}"
        for item in integrations
    )
    probes = " && \\\n    ".join(
        [
            f"test -x {AGENT_RUNTIME_MOUNT}/bin/{_harness_command(harness)}",
            "python -c \"import fugue, fugue.mcp_proxy\"",
            *(
                f"test -x /fugue-components/{item['id']}/bin/server"
                for item in integrations
            ),
        ]
    )
    dockerfile = (
        f"FROM {_BASE_IMAGE} AS fugue-builder\n"
        "COPY source /fugue-src\n"
        f'RUN python -m pip install --no-cache-dir "uv=={_UV_VERSION}" && '
        "cd /fugue-src && "
        "uv sync --frozen --no-dev --no-editable "
        "--python /usr/local/bin/python\n"
        f"FROM {agent_image} AS agent-runtime\n"
        f"FROM {_BASE_IMAGE}\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "bash ca-certificates curl git jq procps ripgrep && "
        "rm -rf /var/lib/apt/lists/*\n"
        f"COPY --from=agent-runtime {AGENT_RUNTIME_MOUNT} {AGENT_RUNTIME_MOUNT}\n"
        "COPY --from=fugue-builder /fugue-src /fugue-src\n"
        f"{copies}\n"
        'ENV PATH="/fugue-src/.venv/bin:$PATH" PYTHONPATH=/fugue-src\n'
        f"RUN {probes}\n"
        "WORKDIR /workspace\n"
    )
    (context / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    return assets


def _tree_digest_from_image(image: str, path: str) -> str:
    container = subprocess.run(
        ["docker", "container", "create", image],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()
    try:
        with tempfile.TemporaryDirectory(
            prefix="fugue-agent-runtime-attestation-"
        ) as temporary:
            destination = Path(temporary) / "runtime"
            subprocess.run(
                [
                    "docker",
                    "container",
                    "cp",
                    f"{container}:{path}/.",
                    destination.as_posix(),
                ],
                check=True,
                timeout=300,
            )
            return _tree_digest(destination, allow_symlinks=True)
    finally:
        subprocess.run(
            ["docker", "container", "rm", "--force", container],
            capture_output=True,
            check=False,
            timeout=60,
        )


def _run_image_probes(
    image: str,
    probes: Sequence[str],
    platform: str,
) -> None:
    command = " && ".join(probes)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            platform,
            "--network",
            "none",
            image,
            "bash",
            "-lc",
            command,
        ],
        check=True,
        timeout=300,
    )


def _write_sbom(image: str, path: Path) -> None:
    subprocess.run(
        ["syft", "scan", image, "-o", f"cyclonedx-json={path.as_posix()}"],
        check=True,
        timeout=600,
    )
    if not path.is_file() or path.stat().st_size < 100:
        raise RuntimeError("SBOM generation did not produce an artifact")


def _scan_image(image: str, path: Path) -> None:
    result = subprocess.run(
        [
            "grype",
            image,
            "--fail-on",
            "high",
            "--output",
            "json",
            "--file",
            path.as_posix(),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "W&B runtime vulnerability scan failed at severity high: "
            + (result.stderr.strip() or result.stdout.strip() or "unknown error")
        )


def _verify_anonymous_pull(image: str) -> None:
    with tempfile.TemporaryDirectory(prefix="fugue-public-pull-") as temporary:
        config = Path(temporary) / "config.json"
        config.write_text('{"auths":{}}\n', encoding="utf-8")
        subprocess.run(
            ["docker", "--config", temporary, "pull", image],
            check=True,
            timeout=900,
        )


def _inspect_image(image: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    values = json.loads(result.stdout)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError("Docker image inspection returned an invalid result")
    return dict(values[0])


def _image_for_harness(
    lock: WandbRuntimeLockV1,
    harness: str,
) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in lock.manifest["images"]
        if item["harness"] == harness
    ]
    if len(matches) != 1:
        raise ValueError(f"W&B runtime lock has no exact {harness} image")
    return matches[0]


def _runtime_lock_path(
    environment: Mapping[str, Any],
    repo_root: Path,
) -> Path:
    raw = str(environment.get("runtime_lock") or WANDB_RUNTIME_LOCK_PATH)
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("W&B runtime lock must be repository-relative")
    path = (repo_root / relative).resolve()
    if repo_root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"W&B runtime lock is unavailable: {relative}")
    return path


def _reject_unknown_environment(environment: Mapping[str, Any]) -> None:
    unknown = sorted(set(environment) - _ALLOWED_EXECUTION_FIELDS)
    if unknown:
        raise ValueError(
            "unsupported W&B execution field(s): " + ", ".join(unknown)
        )


def _harness_image_tag(image: str, harness: str) -> str:
    if "@" in image or "/" not in image:
        raise ValueError("runtime image must be a registry repository with a tag")
    slash = image.rfind("/")
    colon = image.rfind(":")
    if colon <= slash:
        raise ValueError("runtime image must include a qualification tag")
    return f"{image[:colon]}:{image[colon + 1:]}-{harness}"


def _harness_command(harness: str) -> str:
    return {"claude-code": "claude", "openclaw": "openclaw"}[harness]


def _source_identity(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("runtime images must be built from a clean source tree")
    return {
        "commit": commit,
        "tree": tree,
        "fugue_tree_sha256": _tree_digest(repo_root / "fugue"),
    }


def _tree_digest(root: Path, *, allow_symlinks: bool = False) -> str:
    if not root.is_dir():
        raise ValueError(f"runtime asset directory is missing: {root}")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            if not allow_symlinks:
                raise ValueError(
                    f"runtime asset may not contain symlinks: {path}"
                )
            count += 1
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0symlink\0")
            digest.update(os.readlink(path).encode())
            continue
        if not path.is_file():
            continue
        count += 1
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    if count == 0:
        raise ValueError(f"runtime asset directory is empty: {root}")
    return digest.hexdigest()


def _managed_runtime_digest(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"managed runtime must be a real directory: {root}")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(
        root.rglob("*"),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        if path.is_symlink():
            raise ValueError(f"managed runtime may not contain symlinks: {path}")
        if not path.is_file():
            continue
        count += 1
        content = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        digest.update(f"{relative}\0{mode}\0{len(content)}\0".encode())
        digest.update(content)
    if count == 0:
        raise ValueError(f"managed runtime is empty: {root}")
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} is required for W&B runtime qualification")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be an array")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
