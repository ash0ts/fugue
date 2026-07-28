from __future__ import annotations

import base64
import json
from copy import deepcopy
from pathlib import Path

import pytest

import fugue.bench.wandb_sandbox as wandb_sandbox
from fugue.bench.candidates import resolve_candidate, stable_digest
from fugue.bench.cli import _parser
from fugue.bench.comparison import (
    analyze_comparison_rows,
    check_comparison,
    compile_comparison,
    load_comparison,
)
from fugue.bench.export import _wandb_serverless_evidence
from fugue.bench.job_config import _candidate_agent_configuration
from fugue.bench.wandb_sandbox import (
    WANDB_ENVIRONMENT_IMPORT,
    _remote_secret_env,
    _write_build_context,
    bind_wandb_job_environment,
    lock_wandb_runtime,
    read_wandb_runtime_lock,
    validate_wandb_runtime_manifest,
    wandb_execution_identity,
    wandb_harbor_command,
)

EXAMPLE = Path("examples/comparisons/wandb-mcp-maintenance")
SHA256 = "a" * 64
GIT_SHA = "b" * 40


def test_runtime_build_context_uses_frozen_lock_and_locked_agent_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    context = tmp_path / "context"
    runtime = repo / ".fugue" / "runtime" / "mcp" / "wandb" / "runtime"
    (repo / "fugue").mkdir(parents=True)
    context.mkdir()
    runtime.mkdir(parents=True)
    (repo / "fugue" / "__init__.py").write_text("")
    (runtime / "server").write_text("runtime")
    for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
        (repo / name).write_text(name)
    monkeypatch.setattr(
        wandb_sandbox,
        "_tree_digest_from_image",
        lambda image, path: SHA256,
    )

    assets = _write_build_context(
        context,
        repo_root=repo,
        harness="claude-code",
        agent_image="fugue-agent-claude-code:locked",
        agent_image_id="sha256:" + SHA256,
        integrations=(
            {
                "id": "wandb",
                "source": runtime,
                "digest": SHA256,
            },
        ),
    )

    dockerfile = (context / "Dockerfile").read_text()
    assert "uv==0.11.27" in dockerfile
    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "FROM fugue-agent-claude-code:locked AS agent-runtime" in dockerfile
    assert "python -m pip install --no-cache-dir /fugue-src" not in dockerfile
    assert {item["kind"] for item in assets} == {
        "fugue-source",
        "dependency-lock",
        "project-metadata",
        "mcp-runtime",
        "agent-runtime",
    }
    agent_asset = next(item for item in assets if item["kind"] == "agent-runtime")
    assert agent_asset["source"] == "sha256:" + SHA256


def _manifest() -> dict[str, object]:
    images = []
    for harness in ("claude-code", "openclaw"):
        images.append(
            {
                "harness": harness,
                "platform": "linux/amd64",
                "image": (
                    f"docker.io/fugue/runtime-{harness}@sha256:"
                    + ("c" if harness == "claude-code" else "d") * 64
                ),
                "published": True,
                "public_pull_verified": True,
                "agent_runtime": {"image_id": "sha256:" + "e" * 64},
                "assets": [
                    {
                        "kind": "fugue-source",
                        "source": "fugue",
                        "target": "/fugue-src/fugue",
                        "sha256": SHA256,
                    },
                    {
                        "kind": "agent-runtime",
                        "source": "agent",
                        "target": "/opt/fugue-agent-runtime",
                        "sha256": SHA256,
                    },
                    {
                        "kind": "mcp-runtime",
                        "source": "mcp",
                        "target": "/fugue-components/wandb-mcp",
                        "sha256": SHA256,
                    },
                ],
                "probes": [
                    f"/opt/fugue-agent-runtime/bin/{harness} --version",
                    "python -c 'import fugue'",
                ],
                "sbom": {
                    "format": "cyclonedx-json",
                    "path": f"reports/{harness}.sbom.json",
                    "sha256": SHA256,
                },
                "scan": {
                    "scanner": "grype",
                    "fail_on": "high",
                    "status": "passed",
                    "path": f"reports/{harness}.grype.json",
                    "sha256": SHA256,
                },
            }
        )
    value: dict[str, object] = {
        "schema_version": 1,
        "backend": "wandb-serverless",
        "created_at": "2026-07-28T00:00:00Z",
        "source": {
            "commit": GIT_SHA,
            "tree": GIT_SHA,
            "fugue_tree_sha256": SHA256,
        },
        "comparisons": [
            {
                "id": "mcp-primary",
                "path": "examples/comparison.yaml",
                "spec_digest": SHA256,
                "taskset_digest": SHA256,
            }
        ],
        "images": images,
        "required_secrets": {
            "ANTHROPIC_API_KEY": "fugue-anthropic-api-key",
            "WANDB_API_KEY": "fugue-wandb-api-key",
        },
        "manifest_digest": "",
    }
    value["manifest_digest"] = stable_digest(value)
    return value


def _write_lock(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    lock_path = tmp_path / "runtime.lock.json"
    lock_wandb_runtime(manifest_path, output=lock_path)
    return lock_path


def _environment(lock_name: str = "runtime.lock.json") -> dict[str, object]:
    return {
        "type": "wandb",
        "runtime_lock": lock_name,
        "cpus": 2,
        "memory_mb": 4096,
        "storage_mb": 10240,
        "network_mode": "public",
        "max_timeout_seconds": 1800,
        "max_lifetime_seconds": 2400,
        "delete": True,
    }


def test_runtime_manifest_and_lock_reject_drift(tmp_path: Path) -> None:
    manifest = _manifest()
    validated = validate_wandb_runtime_manifest(
        manifest,
        require_published=True,
    )
    assert validated.backend == "wandb-serverless"
    assert {item.harness for item in validated.images} == {
        "claude-code",
        "openclaw",
    }

    lock_path = _write_lock(tmp_path)
    lock = read_wandb_runtime_lock(lock_path)
    assert lock.manifest_digest == manifest["manifest_digest"]
    assert len(lock.lock_digest) == 64

    raw = json.loads(lock_path.read_text())
    raw["manifest"]["images"][0]["scan"]["status"] = "failed"
    lock_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="lock digest"):
        read_wandb_runtime_lock(lock_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["images"][0].update({"published": False}),
            "not publicly digest-locked",
        ),
        (
            lambda value: value["images"][0]["scan"].update(
                {"status": "failed"}
            ),
            "qualification evidence",
        ),
        (
            lambda value: value["comparisons"].append(
                deepcopy(value["comparisons"][0])
            ),
            "comparison manifest entry",
        ),
    ],
)
def test_runtime_manifest_requires_public_scanned_unique_inputs(
    mutation,
    message: str,
) -> None:
    value = _manifest()
    mutation(value)
    value["manifest_digest"] = ""
    value["manifest_digest"] = stable_digest(value)
    with pytest.raises(ValueError, match=message):
        validate_wandb_runtime_manifest(value, require_published=True)


def test_wandb_binding_uses_exact_image_and_no_secret_values(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    environment = _environment(lock_path.name)
    identity = wandb_execution_identity(
        environment,
        harness="claude-code",
        repo_root=tmp_path,
    )
    assert identity is not None
    assert identity["runtime_image"].startswith(
        "docker.io/fugue/runtime-claude-code@sha256:"
    )
    rendered, bound = bind_wandb_job_environment(
        environment,
        harness="claude-code",
        repo_root=tmp_path,
    )
    assert bound == identity
    assert rendered == {
        "docker_image": identity["runtime_image"],
        "network_mode": "public",
        "cpus": 2,
        "memory_mb": 4096,
        "storage_mb": 10240,
        "gpus": 0,
    }
    command = wandb_harbor_command(
        tmp_path / "job.json",
        environment=environment,
        repo_root=tmp_path,
    )
    assert WANDB_ENVIRONMENT_IMPORT in command
    assert "--no-force-build" in command
    assert "--delete" in command
    assert all("secret" not in item.lower() for item in command)


def test_wandb_binding_rejects_uncontrolled_execution_fields(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    environment = _environment(lock_path.name)
    with pytest.raises(ValueError, match="unsupported W&B execution field"):
        wandb_execution_identity(
            {**environment, "mounts": []},
            harness="claude-code",
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="public egress"):
        bind_wandb_job_environment(
            {**environment, "network_mode": "none"},
            harness="claude-code",
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="must be deleted"):
        bind_wandb_job_environment(
            {**environment, "delete": False},
            harness="claude-code",
            repo_root=tmp_path,
        )


def test_named_secret_boundary_derives_aliases_and_headers_remotely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_secret = "unit-wandb-secret"
    anthropic_secret = "unit-anthropic-secret"
    monkeypatch.setenv("WANDB_API_KEY", wandb_secret)
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_secret)
    encoded = base64.b64encode(f"api:{wandb_secret}".encode()).decode()
    clean, exports = _remote_secret_env(
        {
            "WANDB_API_KEY": wandb_secret,
            "ANTHROPIC_API_KEY": anthropic_secret,
            "OPENAI_API_KEY": wandb_secret,
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS": (
                f"project_id=wandb/demo,Authorization=Basic%20{encoded}"
            ),
        },
        reject_unknown=True,
    )
    rendered = "\n".join(exports)
    assert clean == {}
    assert wandb_secret not in rendered
    assert anthropic_secret not in rendered
    assert "${WANDB_API_KEY}" in rendered
    assert "base64" in rendered
    with pytest.raises(ValueError, match="cannot derive secret environment"):
        _remote_secret_env(
            {"UNRELATED_TOKEN": "not-reviewed"},
            reject_unknown=True,
        )
    with pytest.raises(ValueError, match="refuses unreviewed secret environment"):
        _remote_secret_env({"UNRELATED_TOKEN": "not-reviewed"})
    assert _remote_secret_env({"PUBLIC_SETTING": "reviewed"}) == (
        {"PUBLIC_SETTING": "reviewed"},
        [],
    )


def test_serverless_specs_preserve_behavior_and_total_eighty_cells() -> None:
    pairs = [
        ("discovery.yaml", "discovery-serverless.yaml"),
        ("discovery-wandb.yaml", "discovery-wandb-serverless.yaml"),
        ("primary.yaml", "primary-serverless.yaml"),
        ("wandb-replication.yaml", "wandb-replication-serverless.yaml"),
    ]
    total = 0
    for local_name, serverless_name in pairs:
        local = load_comparison(EXAMPLE / local_name, repo_root=Path.cwd())
        serverless = load_comparison(
            EXAMPLE / serverless_name,
            repo_root=Path.cwd(),
        )
        local_experiment, _, _ = compile_comparison(local, repo_root=Path.cwd())
        remote_experiment, _, _ = compile_comparison(
            serverless,
            repo_root=Path.cwd(),
        )
        assert local.baseline == serverless.baseline
        assert local.candidate == serverless.candidate
        assert local.execution.environment == {}
        assert serverless.execution.environment["type"] == "wandb"
        for local_variant, remote_variant in zip(
            local_experiment.variants,
            remote_experiment.variants,
            strict=True,
        ):
            assert _candidate_agent_configuration(
                local_experiment,
                local_variant,
            ) == _candidate_agent_configuration(
                remote_experiment,
                remote_variant,
            )
            local_resolved = resolve_candidate(
                harness=local.execution.harnesses[0],
                harness_version="unit",
                model_route={"provider": "unit", "model": local.execution.model},
                prompt_digest=None,
                skills=[],
                context={},
                integrations=[
                    {"id": item.id} for item in local_variant.integrations
                ],
                agent=_candidate_agent_configuration(
                    local_experiment,
                    local_variant,
                ),
                execution={"environment": local.execution.environment},
            )
            remote_resolved = resolve_candidate(
                harness=serverless.execution.harnesses[0],
                harness_version="unit",
                model_route={
                    "provider": "unit",
                    "model": serverless.execution.model,
                },
                prompt_digest=None,
                skills=[],
                context={},
                integrations=[
                    {"id": item.id} for item in remote_variant.integrations
                ],
                agent=_candidate_agent_configuration(
                    remote_experiment,
                    remote_variant,
                ),
                execution={"environment": serverless.execution.environment},
            )
            assert local_resolved.candidate_id == remote_resolved.candidate_id
            assert (
                local_resolved.execution_fingerprint
                != remote_resolved.execution_fingerprint
            )
        total += check_comparison(
            serverless,
            repo_root=Path.cwd(),
        ).estimated_cells
    assert total == 80
    combined = "\n".join(
        (EXAMPLE / serverless).read_text()
        for _, serverless in pairs
    ).lower()
    assert "coreweave" not in combined
    assert "openai" not in combined


def test_wandb_attestation_is_required_and_must_match_runtime(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial"
    artifacts = trial / "artifacts"
    artifacts.mkdir(parents=True)
    expected = {
        "backend": "wandb-serverless",
        "lock_digest": SHA256,
        "manifest_digest": SHA256,
        "runtime_image": "docker.io/fugue/runtime@sha256:" + "c" * 64,
        "harness": "claude-code",
    }
    meta = {
        "harbor_environment": "wandb",
        "sandbox_runtime": expected,
    }
    missing = _wandb_serverless_evidence(trial, meta)
    assert missing["wandb_serverless_eligible"] is False
    assert missing["wandb_serverless_attestation_status"] == "missing"

    record = {
        "schema_version": 1,
        "backend": "wandb-serverless",
        "lock_digest": SHA256,
        "manifest_digest": SHA256,
        "harness": "claude-code",
        "runtime_image": expected["runtime_image"],
        "sandbox_id": "sandbox-123",
        "state": "deleted",
        "deleted": True,
        "orphans": 0,
        "secret_delivery": "wandb-secrets-manager",
        "secret_env_names": ["ANTHROPIC_API_KEY", "WANDB_API_KEY"],
        "raw_secret_overlays_forwarded": False,
        "recorded_at": "2026-07-28T00:00:00Z",
    }
    record["attestation_digest"] = stable_digest(record)
    (artifacts / "wandb-serverless-attestation.json").write_text(
        json.dumps(record)
    )
    verified = _wandb_serverless_evidence(trial, meta)
    assert verified["wandb_serverless_eligible"] is True
    assert verified["wandb_serverless_orphans"] == 0

    record["runtime_image"] = "docker.io/fugue/other@sha256:" + "d" * 64
    record["attestation_digest"] = ""
    record["attestation_digest"] = stable_digest(record)
    (artifacts / "wandb-serverless-attestation.json").write_text(
        json.dumps(record)
    )
    assert (
        _wandb_serverless_evidence(trial, meta)[
            "wandb_serverless_eligible"
        ]
        is False
    )


def test_ineligible_serverless_attestation_makes_aligned_pair_incomplete() -> None:
    rows = [
        {
            "variant_id": "baseline",
            "task_id": "task",
            "harness": "claude-code",
            "trial_index": 1,
            "pass": False,
            "wandb_serverless_eligible": True,
        },
        {
            "variant_id": "candidate",
            "task_id": "task",
            "harness": "claude-code",
            "trial_index": 1,
            "pass": True,
            "wandb_serverless_eligible": False,
        },
    ]
    result = analyze_comparison_rows(
        comparison_id="serverless",
        preview_digest=SHA256,
        rows=rows,
        source="unit",
    )
    assert result.incomplete == 1
    assert result.improved == 0
    assert result.operational_summary["wandb_serverless"] == {
        "rows": 2,
        "eligible": 1,
        "ineligible": 1,
    }


def test_cli_exposes_wandb_serverless_runtime_workflow() -> None:
    args = _parser().parse_args(
        [
            "sandbox",
            "wandb",
            "build-runtime",
            "--comparison",
            "comparison.yaml",
            "--image",
            "docker.io/fugue/runtime:qualified",
            "--output-manifest",
            "runtime.json",
        ]
    )
    assert args.sandbox_backend == "wandb"
    assert args.wandb_action == "build-runtime"
    assert args.comparisons == [Path("comparison.yaml")]
