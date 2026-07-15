from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fugue.bench.curation import (
    CandidateRecord,
    CurationPolicy,
    evaluate_candidate,
    existing_source_keys,
    main,
    validate_context_proposal,
    validate_skill_bundle,
    validate_skill_proposal,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 7, 14, 14, tzinfo=UTC)
COMMIT = "a" * 40


def _policy() -> CurationPolicy:
    return CurationPolicy.load(REPO_ROOT / "configs/fugue/curation.yaml")


def _skill(**overrides: object) -> CandidateRecord:
    values: dict[str, object] = {
        "kind": "skill",
        "repository": "community/example-skills",
        "path": "skills/repository-guide",
        "commit": COMMIT,
        "stars": 1200,
        "last_push": NOW - timedelta(days=30),
        "archived": False,
        "license": "Apache-2.0",
        "install_reference": (
            "https://github.com/community/example-skills/tree/"
            f"{COMMIT}/skills/repository-guide"
        ),
        "capabilities": ("instruction", "reference"),
        "target_experiment": "pilot",
    }
    values.update(overrides)
    return CandidateRecord(**values)  # type: ignore[arg-type]


def _context(**overrides: object) -> CandidateRecord:
    values: dict[str, object] = {
        "kind": "context_system",
        "repository": "community/example-context",
        "path": None,
        "commit": COMMIT,
        "stars": 600,
        "last_push": NOW - timedelta(days=30),
        "archived": False,
        "license": "MIT",
        "install_reference": "example-context==1.2.3",
        "capabilities": ("bind",),
        "target_experiment": "repo-memory-impact",
    }
    values.update(overrides)
    return CandidateRecord(**values)  # type: ignore[arg-type]


def test_policy_parsing_preserves_checked_in_thresholds() -> None:
    policy = _policy()

    assert policy.maximum_inactive_days == 180
    assert policy.minimum_stars == {"skill": 1000, "context_system": 500}
    assert policy.allowed_licenses == {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
    }
    assert policy.skill_experiments == {"pilot", "skillsbench-pdf-ab"}
    assert policy.context_experiment == "repo-memory-impact"


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (_skill(archived=True), "repository_archived"),
        (
            _skill(last_push=NOW - timedelta(days=181)),
            "repository_inactive",
        ),
        (_skill(license="GPL-3.0"), "license_not_allowed"),
        (_skill(stars=999), "popularity_below_threshold"),
        (
            _skill(commit="main", install_reference="community/example-skills@main"),
            "commit_not_immutable",
        ),
        (_context(install_reference="example-context@latest"), "install_reference_not_immutable"),
        (_context(requires_new_dependencies=True), "new_dependencies_required"),
        (_context(requires_custom_provider=True), "custom_provider_required"),
        (_context(requires_new_dataset=True), "new_dataset_required"),
        (_skill(has_executable_files=True), "executable_skill_bundle"),
        (
            _context(target_experiment="pilot"),
            "target_experiment_not_allowed",
        ),
    ],
)
def test_candidate_gates_are_deterministic(
    candidate: CandidateRecord, reason: str
) -> None:
    decision = evaluate_candidate(candidate, _policy(), evaluated_at=NOW)

    assert decision.eligible is False
    assert reason in decision.reasons
    assert decision.evaluated_at == NOW


def test_activity_boundary_uses_the_explicit_evaluation_time() -> None:
    candidate = _skill(last_push=NOW - timedelta(days=180))

    at_boundary = evaluate_candidate(candidate, _policy(), evaluated_at=NOW)
    one_second_later = evaluate_candidate(
        candidate, _policy(), evaluated_at=NOW + timedelta(seconds=1)
    )

    assert at_boundary.eligible is True
    assert one_second_later.reasons == ("repository_inactive",)


def test_verified_owner_bypasses_only_the_popularity_gate() -> None:
    official = _skill(repository="github/awesome-copilot", stars=1)
    decision = evaluate_candidate(official, _policy(), evaluated_at=NOW)

    assert decision.eligible is True
    assert decision.official_popularity_exception is True
    assert decision.warnings == ("verified_owner_popularity_exception",)

    bad_evidence = replace(
        official,
        license="UNKNOWN",
        install_reference="github/awesome-copilot@main",
    )
    rejected = evaluate_candidate(bad_evidence, _policy(), evaluated_at=NOW)
    assert rejected.eligible is False
    assert set(rejected.reasons) == {
        "install_reference_not_immutable",
        "license_not_allowed",
    }


def test_existing_skill_and_context_provenance_are_deduplicated(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "configs/fugue/skills/repository-guide"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: repository-guide
description: Use when orienting in a repository.
metadata:
  fugue-source-repository: community/example-skills
  fugue-source-path: skills/repository-guide
  fugue-source-commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  fugue-source-license: Apache-2.0
---
# Repository guide
"""
    )
    context_dir = tmp_path / "configs/fugue/context-systems"
    context_dir.mkdir(parents=True)
    (context_dir / "example.yaml").write_text(
        "source_url: https://github.com/community/example-context.git\n"
    )

    assert existing_source_keys(tmp_path) == {
        "skill:community/example-skills:skills/repository-guide",
        "context_system:community/example-context",
    }
    skill_decision = evaluate_candidate(
        _skill(), _policy(), repo_root=tmp_path, evaluated_at=NOW
    )
    context_decision = evaluate_candidate(
        _context(), _policy(), repo_root=tmp_path, evaluated_at=NOW
    )
    assert skill_decision.reasons == ("source_already_present",)
    assert context_decision.reasons == ("source_already_present",)


def test_prior_curator_marker_is_deduplicated() -> None:
    candidate = _skill()
    decision = evaluate_candidate(
        candidate,
        _policy(),
        prior_markers=[f"PR body\n<!-- {candidate.marker} -->\n"],
        evaluated_at=NOW,
    )

    assert decision.reasons == ("prior_curator_pr",)


def test_candidate_record_parses_json_evidence() -> None:
    candidate = CandidateRecord.from_data(
        {
            "kind": "context_system",
            "repository": "community/example-context",
            "path": None,
            "commit": COMMIT,
            "stars": 500,
            "last_push": "2026-07-01T12:30:00Z",
            "archived": False,
            "license": "BSD-3-Clause",
            "install_reference": "example-context==2.0.0",
            "capabilities": ["bind"],
            "target_experiment": "repo-memory-impact",
        }
    )

    assert candidate.last_push == datetime(
        2026, 7, 1, 12, 30, tzinfo=UTC
    )
    assert candidate.marker.endswith(f":-@{COMMIT}")


def test_internal_evaluate_command_outputs_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidate_path = tmp_path / "candidate.json"
    candidate = _skill()
    candidate_path.write_text(
        json.dumps(
            {
                **candidate.__dict__,
                "last_push": candidate.last_push.isoformat(),
            }
        )
    )

    result = main(
        [
            "evaluate",
            "--candidate",
            str(candidate_path),
            "--policy",
            str(REPO_ROOT / "configs/fugue/curation.yaml"),
            "--repo-root",
            str(tmp_path),
            "--as-of",
            NOW.isoformat(),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["eligible"] is True
    assert payload["evaluated_at"] == "2026-07-14T14:00:00Z"


def test_checked_in_skills_conform_to_agent_skills_frontmatter() -> None:
    skills_root = REPO_ROOT / "configs/fugue/skills"
    for skill_dir in sorted(path for path in skills_root.iterdir() if path.is_dir()):
        assert validate_skill_bundle(skill_dir, require_provenance=False) == ()


def test_imported_skill_validation_and_preview_are_side_effect_free(
    tmp_path: Path,
) -> None:
    (tmp_path / "configs/fugue/context-systems").mkdir(parents=True)
    (tmp_path / "configs/fugue/experiments").mkdir(parents=True)
    skill_dir = tmp_path / "configs/fugue/skills/repository-guide"
    skill_dir.mkdir(parents=True)
    (tmp_path / "datasets").mkdir()
    (tmp_path / "configs/fugue/context-systems/none.yaml").write_text(
        """id: none
title: No context
provider: fugue.bench.context:EmptyContextProvider
version: "1"
capabilities: [prepare, retrieve, bind, ingest, sequence]
"""
    )
    (tmp_path / "datasets/demo.yaml").write_text(
        """dataset: {ref: demo/tasks, version: v1}
model: openai/gpt-5
k: 1
n_concurrent: 1
jobs_dir: jobs/demo
harnesses:
  - {name: codex, agent: fugue.agents:FugueCodex}
tasks:
  - {id: task-one, repo: test/repo, base_commit: abc123}
"""
    )
    base = tmp_path / "configs/fugue/experiments/pilot.yaml"
    base.write_text(
        """id: pilot
title: Pilot
manifest: datasets/demo.yaml
model: openai/gpt-5
harnesses: [codex]
variants:
  - {id: baseline, label: Baseline, context: {system_id: none}}
"""
    )
    proposal = tmp_path / "configs/fugue/experiments/repository-guide-ab.yaml"
    proposal.write_text(
        """id: repository-guide-ab
title: Repository guide A/B
manifest: datasets/demo.yaml
model: openai/gpt-5
harnesses: [codex]
variants:
  - {id: baseline, label: No skill, context: {system_id: none}}
  - id: with-repository-guide
    label: Repository guide
    skill_ids: [repository-guide]
    context: {system_id: none}
"""
    )
    (skill_dir / "SKILL.md").write_text(
        """---
name: repository-guide
description: Use when orienting in an unfamiliar repository.
metadata:
  fugue-source-repository: community/example-skills
  fugue-source-path: skills/repository-guide
  fugue-source-commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  fugue-source-license: Apache-2.0
---
# Repository guide

Inspect the repository entry points before editing.
"""
    )

    errors = validate_skill_proposal(
        _skill(),
        skill_dir=skill_dir,
        experiment_path=proposal,
        repo_root=tmp_path,
    )

    assert errors == ()
    assert not (tmp_path / ".fugue").exists()


def test_context_proposal_parses_preflights_and_binds_without_execution(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / "example-context.yaml"
    context_path.write_text(
        """id: example-context
title: Example context
provider: fugue.bench.context:CommandContextProvider
version: example-context@1.2.3
capabilities: [bind]
license: MIT
source_url: https://github.com/community/example-context
enabled_by_default: false
config:
  binding:
    mcp_servers:
      - name: example-context
        command: uvx
        args: [--from, "example-context==1.2.3", example-context]
"""
    )
    experiment_path = tmp_path / "repo-memory-impact.yaml"
    experiment_path.write_text(
        """id: repo-memory-impact
title: Repository memory
model: openai/gpt-5
harnesses: [codex]
workloads:
  - id: coding
    runner: harbor
    manifest: datasets/pilot.yaml
    required_capabilities: [bind]
    systems: [none, example-context]
presets:
  smoke:
    workloads: [coding]
    systems: [none]
variants:
  - {id: none, label: None, context: {system_id: none}}
  - id: example-context
    label: Example context
    context: {system_id: example-context}
"""
    )

    errors = asyncio.run(
        validate_context_proposal(
            _context(), context_path, experiment_path, repo_root=tmp_path
        )
    )

    assert errors == ()
    assert not (tmp_path / ".fugue").exists()
