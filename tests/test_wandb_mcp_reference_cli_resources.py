from __future__ import annotations

import json
import re
import tomllib
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fugue.bench.cli import main

_RESOURCE_ROOT = files("fugue").joinpath(
    "resources", "reference-studies", "wandb-mcp"
)
_CANDIDATE = "b50534a0df9586f7189b3f43f1c71696a7db2a90"


def _resource_text(name: str) -> str:
    return _RESOURCE_ROOT.joinpath(name).read_text(encoding="utf-8")


def test_cli_lazily_prepares_runnable_wandb_mcp_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.reference_studies import wandb_mcp

    env_file = tmp_path / "operator.env"
    env_file.write_text("WANDB_API_KEY=not-serialized\n", encoding="utf-8")
    env_file.chmod(0o600)
    captured: dict[str, object] = {}

    def prepare(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            to_dict=lambda: {
                "schema_version": 1,
                "source_commit": _CANDIDATE,
                "destination": (
                    ".fugue/reference-studies/wandb-mcp/" + _CANDIDATE
                ),
                "materialization": {
                    "artifacts": [
                        {"path": "comparison.yaml"},
                        {"path": "private-labels.jsonl"},
                    ]
                },
            }
        )

    monkeypatch.setattr(wandb_mcp, "prepare_wandb_mcp_reference_study", prepare)

    assert (
        main(
            [
                "mcp",
                "prepare-wandb-release",
                "--repo-root",
                tmp_path.as_posix(),
                "--env-file",
                env_file.as_posix(),
                "--platform",
                "linux/arm64",
            ]
        )
        == 0
    )
    assert captured == {
        "repo_root": tmp_path.resolve(),
        "env_file": env_file.resolve(),
        "platform": "linux/arm64",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_sha"] == _CANDIDATE
    assert payload["comparison_path"] == (
        tmp_path
        / ".fugue/reference-studies/wandb-mcp"
        / _CANDIDATE
        / "comparison.yaml"
    ).as_posix()
    assert "not-serialized" not in json.dumps(payload)


def test_cli_reference_defaults_do_not_require_an_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fugue.reference_studies import wandb_mcp

    captured: dict[str, object] = {}

    def prepare(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            to_dict=lambda: {
                "source_commit": _CANDIDATE,
                "destination": (
                    ".fugue/reference-studies/wandb-mcp/" + _CANDIDATE
                ),
                "materialization": {
                    "artifacts": [{"path": "comparison.yaml"}]
                },
            }
        )

    monkeypatch.setattr(wandb_mcp, "prepare_wandb_mcp_reference_study", prepare)
    assert (
        main(
            [
                "mcp",
                "prepare-wandb-release",
                "--repo-root",
                tmp_path.as_posix(),
            ]
        )
        == 0
    )
    assert captured == {
        "repo_root": tmp_path.resolve(),
        "env_file": None,
        "platform": "linux/amd64",
    }
    capsys.readouterr()


def test_cli_fails_when_preparation_did_not_materialize_a_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fugue.reference_studies import wandb_mcp

    monkeypatch.setattr(
        wandb_mcp,
        "prepare_wandb_mcp_reference_study",
        lambda **_kwargs: SimpleNamespace(
            to_dict=lambda: {
                "source_commit": _CANDIDATE,
                "destination": (
                    ".fugue/reference-studies/wandb-mcp/" + _CANDIDATE
                ),
                "materialization": None,
            }
        ),
    )
    with pytest.raises(RuntimeError, match="runnable comparison"):
        main(
            [
                "mcp",
                "prepare-wandb-release",
                "--repo-root",
                tmp_path.as_posix(),
            ]
        )


@pytest.mark.parametrize(
    ("resource_name", "example_name"),
    (
        (
            "tasks.jsonl",
            "tool-surface-confirmation-tasks-v8.jsonl",
        ),
        (
            "private-labels.jsonl",
            "tool-surface-confirmation-private-v8.jsonl",
        ),
        ("tool_surface_scorer_v7.py", "tool_surface_scorer_v7.py"),
    ),
)
def test_packaged_v8_v7_assets_are_exact_copies(
    resource_name: str,
    example_name: str,
) -> None:
    source = (
        Path(__file__).parents[1]
        / "examples/comparisons/wandb-mcp-maintenance"
        / example_name
    )
    assert _RESOURCE_ROOT.joinpath(resource_name).read_bytes() == source.read_bytes()


def test_packaged_comparison_is_local_and_reference_bound() -> None:
    template = yaml.safe_load(_resource_text("comparison.yaml.template"))
    assert template["schema_version"] == 3
    assert template["id"] == (
        "mcp-main-vs-0-4-{{CANDIDATE_SHORT}}-harbor-canary-v11"
    )
    assert template["execution"]["evidence_mode"] == "local"
    assert template["execution"]["environment"] == {"type": "docker"}
    assert template["execution"]["attempts"] == 1
    assert template["execution"]["concurrency"] == 1
    assert template["execution"]["evidence_checkpoint_cells"] == 2
    assert template["execution"]["research_id"] == (
        "fugue-mcp-release-qualification-v1"
    )
    assert template["execution"]["study_console_base_url"] == (
        "http://127.0.0.1:18080"
    )
    assert template["supersedes"] == [
        {
            "result_digest": (
                "e062f5b392a36d9ebd97adc3ab58b6e253cdd9dd943381342d51d76303bbcf38"
            ),
            "reason": (
                "V10 compared the same locked tasks against the earlier 5c6cc1c9 "
                "staging candidate; this Study freezes the current staging head "
                "under a new candidate, preview, approval, and Study identity."
            ),
        }
    ]
    assert template["execution"]["reference_study"] == {
        "id": "wandb-mcp-release",
        "version": 1,
        "intent": "python-package-release-qualification",
    }
    assert "evidence_project" not in template["execution"]
    assert "evidence_destination" not in template["execution"]
    assert template["execution"]["source_evidence_project"] == (
        "wandb/fugue-mcp-release-source-v2"
    )
    assert template["execution"]["source_evidence_destination"] == {
        "entity": "wandb",
        "project": "fugue-mcp-release-source-v2",
        "api_base_url": "https://api.wandb.ai",
        "trace_base_url": "https://trace.wandb.ai",
        "app_base_url": "https://wandb.ai",
    }
    assert template["execution"]["evidence_lock"] == (
        "source-evidence.lock.json"
    )
    assert template["execution"]["source_conformance_receipt"] == (
        "source-conformance-receipt.json"
    )
    assert template["decision_policy"]["candidate_sha"] == "{{CANDIDATE_SHA}}"
    assert template["baseline"]["integrations"] == [
        "{{MCP_BASELINE_LOCK_ID}}"
    ]
    assert template["candidate"]["integrations"] == [
        "{{MCP_CANDIDATE_LOCK_ID}}"
    ]
    assert len(json.loads(_resource_text("tasks.jsonl").splitlines()[0])) > 0
    assert len(_resource_text("tasks.jsonl").splitlines()) == 4
    assert len(_resource_text("private-labels.jsonl").splitlines()) == 4


def test_release_contract_and_wbaf_provenance_are_exact_and_bounded() -> None:
    from fugue.reference_studies.wandb_mcp import (
        WBAF_TASK_DESIGN_PROVENANCE,
        ReviewedTaskDesignProvenanceV1,
    )

    contract = json.loads(_resource_text("release-contract-v1.json"))
    assert contract["repository"]["baseline"] == {
        "commit": "53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0",
        "tree": "1f38bd7791184ba14545feb69728773c341bb780",
    }
    assert contract["repository"]["candidate"]["ref"] == (
        "refs/heads/staging/0.4.0"
    )
    assert contract["study"]["logical_cell_count"] == 8
    assert contract["study"]["result_evidence_mode"] == "local"
    assert contract["study"]["result_project"] is None

    provenance = json.loads(
        _resource_text("wbaf-task-design-provenance-v1.json")
    )
    assert provenance["source_commit"] == (
        "e2d8d670017bc426b68a311c5777c3b9084023f3"
    )
    assert provenance["source_tree"] == (
        "7776ac62bbe32da5c6824809893d70ea6725a42e"
    )
    assert provenance["runtime_dependency"] is False
    assert provenance["role"] == "task_design_reference"
    assert (
        ReviewedTaskDesignProvenanceV1.from_dict(provenance)
        == WBAF_TASK_DESIGN_PROVENANCE
    )
    assert [(item["path"], item["git_blob"], item["sha256"]) for item in provenance["files"]] == [
        (
            "data/evals/mcp-all.yaml",
            "6165211befa7b5af60468f41b195d29c7c29a7ed",
            "d0dc3ea830cb9ccb2e5d57bbef54712f46e335ca731bb873072201c43a305624",
        ),
        (
            "data/evals/mcp-ci.yaml",
            "201a0f0cb1b77845378e834e4e2db08b89570d2d",
            "777985c511d405795e93b2df75ba6c6d6f7d723e6a43123b0d483fa84f91ba6e",
        ),
        (
            "docs/tasks.md",
            "52bf9f093503ae3ab022942302a54d5d1df142a7",
            "133bd8924378dede4ba73a30a9cd5298a9beb741f6d8bdd9e57e8a4939b59033",
        ),
    ]


def test_mcp_template_freezes_baseline_and_requires_candidate_substitution() -> None:
    raw = _resource_text("mcp.json.template")
    assert raw.count("{{CANDIDATE_SHA}}") == 2
    rendered = json.loads(raw.replace("{{CANDIDATE_SHA}}", _CANDIDATE))
    baseline = rendered["mcpServers"]["wandb-main"]
    candidate = rendered["mcpServers"]["wandb-0-4-staging"]
    assert baseline["version"] == (
        "git:53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0"
    )
    assert candidate["version"] == f"git:{_CANDIDATE}"
    assert baseline["allowed_tools"] == candidate["allowed_tools"]
    assert baseline["env"]["WANDB_MCP_READ_ONLY"] == "true"
    assert candidate["env"]["WANDB_MCP_READ_ONLY"] == "true"
    assert "WANDB_API_KEY" in candidate["env"]
    assert "ANTHROPIC_API_KEY" not in raw


def test_reference_ci_is_no_spend_by_default_and_protected_when_live() -> None:
    workflow = files("fugue").joinpath(
        "resources", "ci", "wandb-mcp-reference.yml"
    ).read_text(encoding="utf-8")
    assert "schedule:" in workflow
    assert "no-spend-preview:" in workflow
    assert "protected-execute:" in workflow
    assert "environment: wandb-mcp-release-qualification" in workflow
    assert "inputs.run_cells" in workflow
    assert "inputs.candidate_sha" in workflow
    assert "inputs.preview_digest" in workflow
    assert "fugue approve" in workflow
    assert "--max-cells 8" in workflow
    assert "--max-usd 10" in workflow
    assert "env -u OPENAI_API_KEY" in workflow
    assert "/current/" not in workflow
    assert "--project" not in workflow
    assert "FUGUE_WEAVE_PROJECT" not in workflow
    assert "pip wheel --no-deps" in workflow
    assert '"${wheel}[local-runner,mcp,weave]"' in workflow
    assert "python -I -c 'import fugue" in workflow
    assert "fugue doctor --workspace" in workflow
    assert "--require local-runner" in workflow
    assert "!.fugue/reference-studies/wandb-mcp/**/private-labels.jsonl" in workflow
    assert "wandb-mcp-preview-failure.json" in workflow
    assert "wandb-mcp-execution-failure.json" in workflow
    assert "if: success()" in workflow
    assert "if: ${{ always() }}" not in workflow
    assert 'python -m pip install ".[local-runner,mcp]"' not in workflow
    no_spend, protected = workflow.split("  protected-execute:", maxsplit=1)
    assert "environment: wandb-mcp-release-qualification" in no_spend
    assert "secrets.ANTHROPIC_API_KEY" not in no_spend
    assert "secrets.WANDB_API_KEY" in no_spend
    assert "secrets.ANTHROPIC_API_KEY" in protected
    assert "secrets.WANDB_API_KEY" in protected
    assert "private-labels.jsonl\n" not in "\n".join(
        line for line in workflow.splitlines() if line.startswith("            ")
        and not line.lstrip().startswith("!")
    )
    uses = re.findall(r"uses:\s*[^@\s]+@([^\s#]+)", workflow)
    assert uses and all(re.fullmatch(r"[0-9a-f]{40}", value) for value in uses)


def test_reference_resources_and_mcp_extra_are_packaged() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["optional-dependencies"]["mcp"] == [
        "mcp==1.28.1"
    ]
    assert "resources/reference-studies/**/*" in (
        project["tool"]["setuptools"]["package-data"]["fugue"]
    )
    expected = {
        "README.md",
        "comparison.yaml.template",
        "mcp.json.template",
        "private-labels.jsonl",
        "release-contract-v1.json",
        "tasks.jsonl",
        "tool_surface_scorer_v7.py",
        "wbaf-task-design-provenance-v1.json",
    }
    assert {item.name for item in _RESOURCE_ROOT.iterdir() if item.is_file()} == expected
