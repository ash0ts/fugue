from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from fugue.bench import operator as operator_module
from fugue.bench.comparison import (
    check_comparison,
    compile_comparison,
    load_comparison,
)
from fugue.bench.job_config import _snapshot_for_task
from fugue.bench.manifest import (
    FixtureRepositorySpec,
    fixture_repository_digest,
    load_manifest,
)
from fugue.bench.operator import ExperimentRequest, OperatorService
from fugue.bench.templates import (
    get_standalone_template,
    scaffold_standalone_template,
    standalone_template_ids,
    standalone_templates,
)

TEMPLATE_IDS = (
    "prompt-change",
    "skill-change",
    "mcp-change",
    "memory-change",
    "harness-change",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _run_packaged_scorer(*, source, evidence, reference, profile, limits):
    namespace: dict[str, Any] = {"__name__": "fugue_template_scorer_test"}
    exec(source, namespace)
    scorer_evidence = {**evidence, "expected": reference["expected"]}
    details = namespace["score"](
        reference["task"],
        reference["output"],
        scorer_evidence,
    )
    values = [1.0 if value is True else float(value) for value in details.values()]
    return {
        "score": min(values),
        "reason": "packaged deterministic scorer",
        "details": details,
    }


@pytest.mark.parametrize("template_id", TEMPLATE_IDS)
def test_packaged_template_is_a_complete_local_eight_cell_study(
    template_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fugue.bench.task_authoring.run_inline_scorer",
        _run_packaged_scorer,
    )
    root = tmp_path / template_id
    comparison_path = scaffold_standalone_template(
        root,
        template_id=template_id,
    )

    assert comparison_path == root / "comparison.yaml"
    assert json.loads((root / ".fugue-study.json").read_text()) == {
        "schema_version": 1,
        "kind": "fugue_standalone_study",
        "template": template_id,
    }
    assert (root / ".github/workflows/fugue-comparison.yml").is_file()
    assert stat.S_IMODE((root / "private-labels.jsonl").stat().st_mode) == 0o600
    assert not any(path.is_symlink() for path in root.rglob("*"))

    gitignore = set((root / ".gitignore").read_text().splitlines())
    assert {".env", ".fugue/", "private-labels.jsonl"} <= gitignore
    assert (root / ".env.example").read_text().splitlines()[-1] == (
        "ANTHROPIC_API_KEY="
    )
    readme = (root / "README.md").read_text(encoding="utf-8")
    create_env = "install -m 600 .env.example .env"
    set_key = "Set `ANTHROPIC_API_KEY` in `.env`."
    doctor = (
        "fugue doctor --require local-runner "
        "--model anthropic/claude-sonnet-5 --env-file .env"
    )
    assert readme.index(create_env) < readme.index(set_key) < readme.index(doctor)
    assert doctor in readme
    assert readme.index(doctor) < readme.index(
        "fugue compare comparison.yaml --run"
    )

    raw = yaml.safe_load(comparison_path.read_text())
    assert raw["execution"] == {
        "model": "anthropic/claude-sonnet-5",
        "harnesses": ["claude-code"],
        "attempts": 2,
        "concurrency": 1,
        "evidence_checkpoint_cells": 2,
        "max_cost_usd": 10,
        "reserve_per_attempt_usd": 1.25,
        "approval_required": True,
        "preparation_required": True,
        "trace_content": "full",
        "evidence_mode": "local",
        "environment": {"type": "docker"},
    }
    serialized = json.dumps(raw, sort_keys=True).lower()
    assert "wandb" not in serialized
    assert "weave" not in serialized

    tasks = _jsonl(root / "tasks.jsonl")
    labels = _jsonl(root / "private-labels.jsonl")
    assert len(tasks) == len(labels) == 2
    assert {item["id"] for item in tasks} == {item["id"] for item in labels}
    assert all("expected" not in task for task in tasks)
    assert all(task.get("resources") for task in tasks)
    for task in tasks:
        for resource in task["resources"]:
            source = root / resource["path"]
            assert source.is_file() and not source.is_symlink()
            assert resource["target"].startswith("/workspace/resources/")
    assert all(
        {"expected", "base_output", "gold_output"} <= set(label) for label in labels
    )
    [evaluator] = raw["evaluators"]
    assert evaluator["scorer"] == "scorer.py"
    assert evaluator["runtime"] == "python312-sandbox-v1"
    assert evaluator["dimensions"] == ["answer_present", "expected_values"]
    assert (root / evaluator["scorer"]).is_file()
    profiles = yaml.safe_load(
        (root / "configs/fugue/task-authoring/profiles.yaml").read_text()
    )
    [scorer_runtime] = profiles["scorer_runtimes"]
    assert scorer_runtime["id"] == evaluator["runtime"]
    assert "@sha256:" in scorer_runtime["image"]
    assert scorer_runtime["platform"] in {"linux/amd64", "linux/arm64"}
    assert "{{" not in json.dumps(profiles)

    spec = load_comparison(comparison_path, repo_root=root)
    readiness = check_comparison(spec, repo_root=root)
    template = get_standalone_template(template_id)
    assert readiness.task_count == 2
    assert readiness.estimated_cells == 8
    assert readiness.estimated_cost_usd == 10
    assert readiness.base_failures == 2
    assert readiness.gold_passes == 2
    assert readiness.actual_changes == spec.changed
    assert template.changed_dimension == spec.changed[0].split(".", 1)[0]
    assert not any(
        "not locked and usable" in blocker
        or "declared candidate changes" in blocker
        or "local comparison runtime cannot be resolved" in blocker
        for blocker in readiness.blockers
    )


def test_standalone_scorer_source_is_part_of_the_evaluator_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fugue.bench.task_authoring.run_inline_scorer",
        _run_packaged_scorer,
    )
    root = tmp_path / "prompt-change"
    comparison_path = scaffold_standalone_template(
        root,
        template_id="prompt-change",
    )
    spec = load_comparison(comparison_path, repo_root=root)
    before = check_comparison(spec, repo_root=root)

    scorer = root / "scorer.py"
    scorer.write_text(scorer.read_text() + "\n# pinned revision change\n")
    after = check_comparison(spec, repo_root=root)

    assert before.evaluator_digests != after.evaluator_digests


@pytest.mark.parametrize(
    ("machine", "expected"),
    [("x86_64", "linux/amd64"), ("aarch64", "linux/arm64")],
)
def test_standalone_scorer_platform_is_resolved_per_study(
    machine: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fugue.bench.templates.platform.machine", lambda: machine)
    root = tmp_path / machine
    scaffold_standalone_template(root, template_id="prompt-change")
    profiles = yaml.safe_load(
        (root / "configs/fugue/task-authoring/profiles.yaml").read_text()
    )
    assert profiles["scorer_runtimes"][0]["platform"] == expected


def test_standalone_scorer_rejects_unsupported_host_architecture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fugue.bench.templates.platform.machine", lambda: "riscv64")
    with pytest.raises(RuntimeError, match="supports only amd64 and arm64"):
        scaffold_standalone_template(tmp_path / "study", template_id="prompt-change")


def test_standalone_check_fails_closed_without_private_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prompt-change"
    comparison_path = scaffold_standalone_template(
        root,
        template_id="prompt-change",
    )
    (root / "private-labels.jsonl").unlink()

    spec = load_comparison(comparison_path, repo_root=root)
    readiness = check_comparison(spec, repo_root=root)

    assert readiness.status == "blocked"
    assert readiness.base_failures == 0
    assert readiness.gold_passes == 0
    assert any(
        "host-only private labels are unavailable" in blocker
        for blocker in readiness.blockers
    )


def test_standalone_template_catalogue_is_stable_and_unique() -> None:
    assert standalone_template_ids() == TEMPLATE_IDS
    templates = standalone_templates()
    assert len({item.id for item in templates}) == len(templates)
    assert templates[-1].title == "Harness configuration change"


def test_scaffold_rejects_unknown_templates_and_nonempty_destinations(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unknown standalone template"):
        scaffold_standalone_template(
            tmp_path / "escaped",
            template_id="../prompt-change",
        )

    root = tmp_path / "existing"
    root.mkdir()
    (root / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        scaffold_standalone_template(root, template_id="prompt-change")
    assert (root / "user.txt").read_text() == "keep"


def test_scaffold_ignores_release_gate_bytecode_in_packaged_resources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    scaffold_standalone_template(root, template_id="memory-change")

    assert not any(path.suffix == ".pyc" for path in root.rglob("*"))
    assert not any(path.name == "__pycache__" for path in root.rglob("*"))


def test_force_scaffold_refuses_destination_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "study"
    scaffold_standalone_template(root, template_id="prompt-change")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    readme = root / "README.md"
    readme.unlink()
    readme.symlink_to(outside)

    with pytest.raises(ValueError, match="may not contain symlinks"):
        scaffold_standalone_template(
            root,
            template_id="prompt-change",
            force=True,
        )
    assert outside.read_text() == "outside"


def test_offline_mcp_fixture_runs_both_locked_revisions(tmp_path: Path) -> None:
    root = tmp_path / "mcp"
    scaffold_standalone_template(root, template_id="mcp-change")
    script = root / "fixtures/offline_catalog_mcp.py"

    observed: dict[str, dict[str, Any]] = {}
    for revision in ("legacy", "current"):
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "lookup_setting",
                "arguments": {"setting": "refund_window_days"},
            },
        }
        process = subprocess.run(
            [sys.executable, script.as_posix(), "--revision", revision],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            check=True,
            env={"PATH": os.environ.get("PATH", "")},
            timeout=5,
        )
        response = json.loads(process.stdout)
        observed[revision] = response["result"]["structuredContent"]

    assert observed == {
        "legacy": {
            "setting": "refund_window_days",
            "value": 14,
            "unit": "days",
            "revision": "legacy",
            "source": "offline-catalog",
        },
        "current": {
            "setting": "refund_window_days",
            "value": 30,
            "unit": "days",
            "revision": "current",
            "source": "offline-catalog",
        },
    }
    configs = root / "configs/fugue/integrations"
    legacy = yaml.safe_load((configs / "offline-catalog-legacy.yaml").read_text())
    current = yaml.safe_load((configs / "offline-catalog-current.yaml").read_text())
    assert legacy["runtime"]["command"][-1] == "legacy"
    assert current["runtime"]["command"][-1] == "current"


def test_harness_template_is_explicitly_a_configuration_experiment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "harness"
    comparison = scaffold_standalone_template(
        root,
        template_id="harness-change",
    )
    raw = yaml.safe_load(comparison.read_text())

    assert raw["execution"]["harnesses"] == ["claude-code"]
    assert raw["baseline"]["agent_kwargs"] == {"max_turns": 3}
    assert raw["candidate"]["agent_kwargs"] == {"max_turns": 8}
    assert "does **not** rank" in (root / "README.md").read_text()


def test_memory_template_prepares_and_binds_the_same_locked_fixture_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "memory"
    comparison_path = scaffold_standalone_template(
        root,
        template_id="memory-change",
    )
    spec = load_comparison(comparison_path, repo_root=root)
    experiment, raw_manifest, _public_rows = compile_comparison(spec, repo_root=root)
    manifest_path = root / experiment.manifest
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(yaml.safe_dump(raw_manifest, sort_keys=False))
    manifest = load_manifest(manifest_path)

    expected_checkout = (root / "fixtures").resolve()
    expected_digest = fixture_repository_digest(expected_checkout)
    assert len(manifest.tasks) == 2
    for task in manifest.tasks:
        assert task.repository == FixtureRepositorySpec(
            type="fixture",
            path="fixtures",
            sha256=expected_digest,
        )
        snapshot = _snapshot_for_task(task, root, manifest.dataset.harbor_ref)
        assert snapshot.checkout == expected_checkout
        assert snapshot.fixture_digest == expected_digest

    monkeypatch.setattr(
        operator_module,
        "materialize_manifest_dataset",
        lambda *_args, **_kwargs: None,
    )
    prepared = OperatorService(root).prepare_context(
        ExperimentRequest(
            experiment_id=experiment.id,
            n_tasks=2,
            n_attempts=2,
            n_concurrent=1,
        ),
        experiment=experiment,
        run_id="memory-template-prepare",
    )
    agentsmd = [item for item in prepared if item.system_id == "agentsmd"]
    assert len(agentsmd) == 2
    for item in agentsmd:
        assert item.path is not None
        instruction = item.path / "artifact" / "AGENTS.md"
        assert instruction.is_file()
        assert "`ledger_cli.py`" in instruction.read_text()


def test_memory_template_fixture_drift_fails_before_context_binding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    comparison_path = scaffold_standalone_template(
        root,
        template_id="memory-change",
    )
    spec = load_comparison(comparison_path, repo_root=root)
    _experiment, raw_manifest, _public_rows = compile_comparison(spec, repo_root=root)
    manifest_path = root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(raw_manifest, sort_keys=False))
    [task, *_rest] = load_manifest(manifest_path).tasks

    (root / "fixtures" / "config.py").write_text("DRIFTED = True\n")

    with pytest.raises(ValueError, match="fixture repository digest changed"):
        _snapshot_for_task(task, root, "comparison-fixture")
