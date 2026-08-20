from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from fugue.bench.candidates import resolve_candidate
from fugue.bench.comparison import _comparison_dimension_roles, load_comparison
from fugue.bench.execution import PlannedCell, plan_cells
from fugue.bench.job_config import render_jobs
from fugue.bench.library import ExperimentSpec, FeatureVariant
from fugue.bench.local_evidence import (
    LocalAttemptPlanV1,
    local_attempt_plan_from_dict,
)
from fugue.bench.manifest import load_manifest
from fugue.bench.task_presentation import (
    UNDECLARED_ACCEPTANCE_CRITERION,
    UNDECLARED_REQUIRED_OUTPUT,
    FailedRequiredCheckV1,
    PublicPromptPartV1,
    TaskPresentationV1,
    TaskResultV1,
    candidate_treatment_summary,
    task_presentation_from_dict,
    task_presentation_from_public_case,
)


def _presentation(prompt: str = "Inspect the locked project.") -> TaskPresentationV1:
    return TaskPresentationV1(
        task_id="locked-project-audit",
        title="Audit the locked project",
        public_prompt=(PublicPromptPartV1(order=1, text=prompt),),
        required_output="Return one bounded JSON maintenance brief.",
        public_acceptance_criteria=(
            "Use only the locked project.",
            "State when required evidence is unavailable.",
        ),
        scenario="maintenance",
        tags=("mcp", "bounded-read"),
        partition="qualification",
        safe_resource_references=(
            {
                "id": "evidence-lock",
                "sha256": "a" * 64,
                "target": "/workspace/resources/evidence.lock.json",
            },
        ),
    )


def _cell(presentation: TaskPresentationV1) -> PlannedCell:
    return PlannedCell(
        id="cell-1",
        run_id="run-1",
        run_name="Readable evidence",
        workload_id="harbor",
        task_id=presentation.task_id,
        harness="claude-code",
        context_system_id="none",
        variant_id="candidate",
        model_provider="anthropic",
        model="anthropic/claude-sonnet-5",
        trial_index=1,
        comparison_example_id="b" * 64,
        candidate_id="c" * 64,
        execution_fingerprint="d" * 64,
        config_path=Path("job.json"),
        result_path=Path("result.json"),
        command=("harbor", "run"),
        env={},
        n_attempts=1,
        arm_label="Candidate 2.0",
        treatment_summary="claude-code; Skills candidate-skill.",
        task_presentation=presentation,
    )


def test_task_presentation_digest_binds_only_public_task_definition() -> None:
    first = _presentation()
    assert task_presentation_from_dict(first.to_dict()) == first
    assert len(first.task_definition_digest) == 64

    changed = _presentation("Inspect the locked project and report missing evidence.")
    assert changed.task_definition_digest != first.task_definition_digest

    with pytest.raises(ValueError, match="ordered from one"):
        TaskPresentationV1(
            task_id="task",
            title="Task",
            public_prompt=(PublicPromptPartV1(order=2, text="Do the task."),),
            required_output="Return a result.",
            public_acceptance_criteria=("Complete the task.",),
        )


def test_task_presentation_rejects_private_and_credential_content() -> None:
    with pytest.raises(ValueError, match="private field"):
        task_presentation_from_public_case(
            task_id="task",
            public_case={
                "id": "task",
                "instruction": "Answer the public question.",
                "expected": {"answer": 42},
            },
        )

    with pytest.raises(ValueError, match="credential-like"):
        _presentation("Use api_key=sk-abcdefghijklmnop to inspect the project.")

    presentation = task_presentation_from_public_case(
        task_id="task",
        public_case={
            "id": "task",
            "input": {"question": "Inspect the supplied policy."},
            "resources": [
                {
                    "path": "private/host/layout/policy.md",
                    "target": "/workspace/resources/policy.md",
                    "title": "Public policy",
                }
            ],
        },
    )
    assert presentation is not None
    assert presentation.safe_resource_references == (
        {
            "id": "policy.md",
            "target": "/workspace/resources/policy.md",
            "title": "Public policy",
        },
    )
    assert "private/host/layout" not in str(presentation.to_dict())


def test_missing_public_success_contract_is_explicitly_unavailable() -> None:
    presentation = task_presentation_from_public_case(
        task_id="legacy-task",
        public_case={
            "id": "legacy-task",
            "prompt": "Return the observed result.",
        },
    )

    assert presentation is not None
    assert presentation.public_prompt == (
        PublicPromptPartV1(order=1, text="Return the observed result."),
    )
    assert presentation.required_output == UNDECLARED_REQUIRED_OUTPUT
    assert presentation.public_acceptance_criteria == (
        UNDECLARED_ACCEPTANCE_CRITERION,
    )
    assert "Satisfy every public" not in str(presentation.to_dict())


def test_pre_run_task_presentation_excludes_future_scripted_turns() -> None:
    presentation = task_presentation_from_public_case(
        task_id="scripted-task",
        public_case={
            "id": "scripted-task",
            "prompt": "Inspect the repository and report the current finding.",
            "interaction": {
                "type": "scripted",
                "scripted_turns": [
                    "Now revise the finding after reading the hidden follow-up."
                ],
            },
        },
    )

    assert presentation is not None
    assert presentation.public_prompt == (
        PublicPromptPartV1(
            order=1,
            text="Inspect the repository and report the current finding.",
        ),
    )
    assert "hidden follow-up" not in json.dumps(presentation.to_dict())


def test_presentation_fields_do_not_change_canonical_attempt_identity() -> None:
    first = _cell(_presentation())
    changed = replace(
        first,
        arm_label="Release candidate",
        treatment_summary="A clearer public summary.",
        task_presentation=_presentation("A clearer public prompt."),
    )

    assert changed.attempt_identity == first.attempt_identity
    assert changed.attempt_id == first.attempt_id
    assert changed.record("pending")["task_presentation"] == (
        changed.task_presentation.to_dict()
    )
    assert changed.record("pending")["arm_label"] == "Release candidate"


def test_public_task_card_flows_from_rendered_job_to_planned_cell(
    tmp_path: Path,
) -> None:
    task_row = {
        "id": "locked-project-audit",
        "input": {"question": "Inspect only the locked public project."},
        "title": "Audit one locked project",
        "required_output": "Return one bounded JSON maintenance brief.",
        "public_acceptance_criteria": [
            "Use only the locked project.",
            "State when evidence is unavailable.",
        ],
        "tags": ["mcp", "bounded-read"],
        "partition": "qualification",
        "resources": [
            {
                "path": "host-only-layout/evidence.lock.json",
                "target": "/workspace/resources/evidence.lock.json",
            }
        ],
    }
    source = json.dumps(task_row, sort_keys=True) + "\n"
    (tmp_path / "tasks.jsonl").write_text(source)
    source_digest = hashlib.sha256(source.encode()).hexdigest()
    manifest_path = tmp_path / "benchmark.yaml"
    manifest_path.write_text(
        f"""
dataset:
  path: tasks.jsonl
  materializer: fugue.bench.task_authoring:AuthoredTaskMaterializer
  source:
    path: tasks.jsonl
    sha256: {source_digest}
harnesses:
  - {{name: claude-code, agent: fugue.agents:FugueClaudeCode}}
tasks:
  - {{id: locked-project-audit}}
"""
    )
    experiment = ExperimentSpec(
        id="readable-task",
        title="Readable task",
        variants=[FeatureVariant(id="candidate", label="Candidate 2.0")],
    )

    [job] = render_jobs(
        experiment=experiment,
        manifest=load_manifest(manifest_path),
        manifest_path=manifest_path,
        repo_root=tmp_path,
        env={},
        model="anthropic/claude-sonnet-5",
        run_id="readable-task",
    )
    [cell] = plan_cells([job], run_id="readable-task", run_name="Readable task")

    assert job.task_presentation is not None
    assert cell.task_presentation == job.task_presentation
    assert cell.task_presentation.title == "Audit one locked project"
    assert cell.task_presentation.public_prompt[0].text == (
        "Inspect only the locked public project."
    )
    assert cell.arm_label == "Candidate 2.0"
    assert cell.treatment_summary == job.treatment_summary
    assert "host-only-layout" not in str(cell.task_presentation.to_dict())


def test_local_attempt_plan_preserves_new_copy_and_reads_legacy_plan() -> None:
    cell = _cell(_presentation())
    current = LocalAttemptPlanV1(
        run_id=cell.run_id,
        cell_id=cell.id,
        attempt_id=cell.attempt_id,
        attempt_identity=cell.attempt_identity,
        prediction_id="e" * 64,
        evaluation_scope_id="f" * 64,
        dataset_id="1" * 64,
        arm_label=cell.arm_label,
        treatment_summary=cell.treatment_summary,
        task_presentation=cell.task_presentation,
    )
    assert local_attempt_plan_from_dict(current.to_dict()) == current

    legacy = {
        key: value
        for key, value in current.to_dict().items()
        if key not in {"arm_label", "treatment_summary", "task_presentation"}
    }
    parsed = local_attempt_plan_from_dict(legacy)
    assert parsed.to_dict() == legacy
    assert parsed.task_presentation is None


def test_task_result_separates_execution_task_and_evidence_status() -> None:
    result = TaskResultV1(
        task_passed=False,
        outcome_summary="The Agent completed, but the package was invalid.",
        failed_required_checks=(
            FailedRequiredCheckV1(
                id="package-contract-valid",
                label="Package contract is valid",
                explanation="The required manifest was missing.",
            ),
        ),
        answer_digest="2" * 64,
        agent_execution_status="completed",
        evidence_integrity_status="verified",
    )

    assert result.agent_execution_status == "completed"
    assert result.task_passed is False
    with pytest.raises(ValueError, match="passing task"):
        replace(result, task_passed=True)
    with pytest.raises(ValueError, match="invalid Agent execution status"):
        replace(result, agent_execution_status="unknown")  # type: ignore[arg-type]


def test_treatment_summary_uses_only_allowlisted_candidate_components() -> None:
    summary = candidate_treatment_summary(
        {
            "harness": "claude-code",
            "model_route": {"display_model": "anthropic/claude-sonnet-5"},
            "skills": [{"id": "writing-plans", "private_config": "do-not-copy"}],
            "integrations": [{"id": "wandb-mcp", "token": "do-not-copy"}],
            "context": {"id": "none", "config": {"secret": "do-not-copy"}},
            "prompt_digest": "3" * 64,
        }
    )

    assert "writing-plans" in summary
    assert "wandb-mcp" in summary
    assert "do-not-copy" not in summary


def test_treatment_summary_distinguishes_exact_revisions_without_new_identity() -> None:
    def resolved(commit: str):
        return resolve_candidate(
            harness="claude-code",
            harness_version="claude-code@1",
            model_route={"display_model": "anthropic/claude-sonnet-5"},
            prompt_digest=None,
            skills=[
                {
                    "id": "writing-plans",
                    "resolved_commit": commit,
                    "digest": "a" * 64,
                }
            ],
            context={"id": "none"},
            integrations=[
                {
                    "id": "wandb-mcp",
                    "version": "sha256:" + commit,
                    "behavior_hash": "b" * 64,
                }
            ],
            agent={},
            execution={"backend": "harbor"},
        )

    baseline = resolved("1" * 40)
    candidate = resolved("2" * 40)
    baseline_summary = candidate_treatment_summary(baseline.definition)
    candidate_summary = candidate_treatment_summary(candidate.definition)

    assert baseline.candidate_id != candidate.candidate_id
    assert baseline_summary != candidate_summary
    assert "commit 111111111111" in baseline_summary
    assert "commit 222222222222" in candidate_summary
    assert "digest aaaaaaaaaaaa" in baseline_summary
    assert "version sha256:111111111111" in baseline_summary
    assert "treatment_summary" not in baseline.definition


def test_comparison_dimension_roles_match_exact_emitted_score_keys() -> None:
    root = Path.cwd()
    built_in = load_comparison(
        root / "examples/comparisons/source-use-replay/comparison.yaml",
        repo_root=root,
    )
    custom = load_comparison(
        root
        / "examples/comparisons/wandb-mcp-maintenance/"
        "tool-surface-confirmation-local-v10.yaml",
        repo_root=root,
    )

    assert _comparison_dimension_roles(built_in) == {
        "answer_present": "outcome",
        "expected_values": "outcome",
    }
    roles = _comparison_dimension_roles(custom)
    assert roles["tool-surface.answer_correct"] == "outcome"
    assert roles["tool-surface.actual_query_scope"] == "safety_gate"
    assert roles["tool-surface.release_mechanism_used"] == "mechanism"
