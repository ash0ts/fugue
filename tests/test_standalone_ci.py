from __future__ import annotations

import re
from importlib.metadata import version
from importlib.resources import files
from typing import Any

import yaml


def _workflow_text() -> str:
    return (
        files("fugue")
        .joinpath("resources", "ci", "standalone-comparison.yml")
        .read_text(encoding="utf-8")
    )


def _workflow() -> dict[str, Any]:
    workflow = yaml.safe_load(_workflow_text())
    assert isinstance(workflow, dict)
    return workflow


def _events(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML keeps YAML 1.1 boolean-key behavior while GitHub uses YAML 1.2.
    events = workflow.get("on", workflow.get(True))
    assert isinstance(events, dict)
    return events


def _run_commands(job: dict[str, Any]) -> str:
    return "\n".join(
        str(step.get("run", ""))
        for step in job["steps"]
        if isinstance(step, dict)
    )


def test_standalone_workflow_plans_automatically_but_executes_only_when_gated(
) -> None:
    workflow = _workflow()
    events = _events(workflow)
    inputs = events["workflow_dispatch"]["inputs"]
    jobs = workflow["jobs"]

    assert "pull_request" in events
    assert inputs["study_root"]["default"] == "."
    assert inputs["comparison"]["default"] == "comparison.yaml"
    assert inputs["execute"]["default"] is False
    assert inputs["fugue_requirement"]["default"] == (
        f"fugue[local-runner]=={version('fugue')}"
    )

    assert jobs["plan"].get("environment") is None
    assert "secrets." not in yaml.safe_dump(jobs["plan"], sort_keys=True)
    assert jobs["execute"]["environment"] == "fugue-local-experiment"
    execution_gate = str(jobs["execute"]["if"])
    assert "workflow_dispatch" in execution_gate
    assert "inputs.execute == true" in execution_gate

    credential_step = next(
        step
        for step in jobs["execute"]["steps"]
        if step.get("name") == "Require the protected model credential"
    )
    assert credential_step["env"]["ANTHROPIC_API_KEY"] == (
        "${{ secrets.ANTHROPIC_API_KEY }}"
    )
    assert "test -n" in credential_step["run"]
    assert "ANTHROPIC_API_KEY" in credential_step["run"]
    assert "GITHUB_ENV" not in credential_step["run"]

    run_step = next(
        step
        for step in jobs["execute"]["steps"]
        if step.get("name") == "Run the approved local comparison"
    )
    assert run_step["env"]["ANTHROPIC_API_KEY"] == (
        "${{ secrets.ANTHROPIC_API_KEY }}"
    )


def test_standalone_workflow_uses_wheel_and_the_canonical_local_cli() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    plan_commands = _run_commands(jobs["plan"])
    execute_commands = _run_commands(jobs["execute"])

    for job in jobs.values():
        setup = next(
            step
            for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/setup-python@")
        )
        assert setup["with"]["python-version"] == "3.13"
        commands = _run_commands(job)
        assert "python -m pip wheel" in commands
        assert "python -m pip install" in commands
        assert "--no-index" in commands
        assert "$FUGUE_REQUIREMENT" in commands
        assert "fugue doctor" in commands
        assert "fugue check" in commands
        assert 'test "$?" -eq 2' in commands
        assert "--prepare" in commands
        assert "--preview" in commands

    assert "fugue approve" not in plan_commands
    assert "--run" not in plan_commands
    assert "fugue approve" in execute_commands
    assert "--run --approval" in execute_commands
    assert "fugue result latest" in execute_commands
    assert "--fetch-weave" not in _workflow_text()
    assert "WANDB" not in _workflow_text().upper()
    assert "WEAVE" not in _workflow_text().upper()


def test_standalone_workflow_preserves_and_uploads_the_local_ledger() -> None:
    workflow = _workflow()

    for job in workflow["jobs"].values():
        uploads = [
            step
            for step in job["steps"]
            if "actions/upload-artifact@" in str(step.get("uses", ""))
        ]
        assert len(uploads) == 1
        upload = uploads[0]
        assert upload["if"] == "always()"
        assert upload["with"]["path"] == "${{ env.STUDY_ROOT }}/.fugue"
        assert upload["with"]["include-hidden-files"] is True
        assert upload["with"]["if-no-files-found"] == "error"

    for job in workflow["jobs"].values():
        for step in job["steps"]:
            uses = str(step.get("uses", ""))
            if not uses:
                continue
            revision = uses.rsplit("@", 1)[-1].split()[0]
            assert re.fullmatch(r"[0-9a-f]{40}", revision), uses
