from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.mcp_release_qualification import (
    _mcp_release_qualification_receipt,
    release_note_coverage_v3,
    tool_surface_coverage_v1,
    validate_release_notes_lock,
)

BASELINE = (
    "wandb-mcp-main",
    "git:53b199a5f4af29aa82077e2c7f1e2c5e5e0c2ca0",
)
CURRENT_CANDIDATE = (
    "wandb-mcp-0-4-current",
    "git:5c6cc1c9a1079296daf6613ea6d12daebdd8bcba",
)
CURRENT_CANDIDATES = (BASELINE, CURRENT_CANDIDATE)


def _observation(import_id: str, version_identity: str) -> dict[str, object]:
    return {
        "id": import_id,
        "version_identity": version_identity,
        "runtime_digest": "sha256:" + "a" * 64,
        "tool_manifest_digest": "b" * 64,
        "server": {},
        "initialized_tools": [],
        "initialized_tool_schema_digest": "",
        "locked_tools": [],
        "locked_tool_schema_digest": "",
        "initialized_schema_matches_lock": False,
        "release_capabilities": {},
        "calls": {},
        "evaluation_child_ops": {},
        "profile_probes": {},
    }


def test_mechanism_receipt_digest_binds_explicit_current_candidates() -> None:
    evidence_lock = {
        "source_project": "wandb/fugue-mcp-release-source-v2",
        "result_project": "wandb/fugue-mcp-release-qualification-v1",
        "evidence_lock_digest": "c" * 64,
        "counts": {"runs": 6},
        "objects": {"evaluations": []},
    }
    receipt = _mcp_release_qualification_receipt(
        evidence_lock,
        [_observation(*candidate) for candidate in CURRENT_CANDIDATES],
        candidates=CURRENT_CANDIDATES,
    )

    expected_bindings = [
        {
            "role": "baseline",
            "import_id": BASELINE[0],
            "version_identity": BASELINE[1],
        },
        {
            "role": "candidate",
            "import_id": CURRENT_CANDIDATE[0],
            "version_identity": CURRENT_CANDIDATE[1],
        },
    ]
    assert receipt["candidate_bindings"] == expected_bindings
    assert receipt["candidate_bindings_digest"] == stable_digest(expected_bindings)
    unsigned = {**receipt, "receipt_digest": ""}
    assert receipt["receipt_digest"] == stable_digest(unsigned)

    tampered = json.loads(json.dumps(unsigned))
    tampered["candidate_bindings"][1]["version_identity"] = "git:" + "d" * 40
    assert stable_digest(tampered) != receipt["receipt_digest"]


def test_current_release_notes_are_exact_and_fully_classified() -> None:
    path = (
        Path.cwd()
        / "examples/comparisons/wandb-mcp-maintenance/"
        "release-notes.current.lock.json"
    )
    release_notes = validate_release_notes_lock(
        json.loads(path.read_text(encoding="utf-8")),
        expected_commit=CURRENT_CANDIDATE[1].removeprefix("git:"),
    )
    with pytest.raises(ValueError, match="selected candidate commit"):
        validate_release_notes_lock(
            release_notes,
            expected_commit="0" * 40,
        )
    evidence_lock = {
        "source_project": "wandb/fugue-mcp-release-source-v2",
        "result_project": "wandb/fugue-mcp-release-qualification-v1",
        "evidence_lock_digest": "c" * 64,
        "counts": {"runs": 6},
        "objects": {"evaluations": []},
    }
    observations = [_observation(*candidate) for candidate in CURRENT_CANDIDATES]
    observations[1]["release_capabilities"] = {"cursor_continuation": True}
    receipt = _mcp_release_qualification_receipt(
        evidence_lock,
        observations,
        candidates=CURRENT_CANDIDATES,
        release_notes=release_notes,
    )

    classifications = {
        item["release_note"]: item
        for item in receipt["release_note_classification"]
    }
    assert set(classifications) == set(release_notes["behaviors"])
    assert classifications["cursor-continuation-pagination"]["status"] == (
        "observed_branch_delta"
    )
    coverage = release_note_coverage_v3(
        receipt,
        task_ids=(
            "run-inventory-projection",
            "filtered-failure-triage",
            "evaluation-summary-accuracy",
            "exact-history-target",
            "selective-run-comparison",
            "missing-cost-honesty",
        ),
        dimension_ids=(
            "tool-surface.answer_correct",
            "tool-surface.actual_query_scope",
            "tool-surface.reported_project_identity",
            "tool-surface.bounded_evidence",
            "tool-surface.evidence_honesty",
            "tool-surface.release_mechanism_used",
        ),
    )
    assert len(coverage) == 25
    by_behavior = {item["release_note"]: item for item in coverage}
    assert by_behavior["cursor-continuation-pagination"]["task_ids"] == [
        "filtered-failure-triage"
    ]
    assert by_behavior["runtime-safety-boundaries"]["status"] == (
        "infrastructure_only"
    )
    assert by_behavior["exact-report-name-or-display-title"]["status"] == (
        "unqualified"
    )


def test_tool_coverage_uses_receipt_bound_current_candidate_identity() -> None:
    tools = (
        "compare_artifact_versions_tool",
        "compare_runs_tool",
        "count_weave_traces_tool",
        "diagnose_run_tool",
        "get_artifact_details_tool",
        "get_run_history_tool",
        "infer_trace_schema_tool",
        "list_artifact_versions_tool",
        "list_entities_tool",
        "list_registries_tool",
        "list_registry_collections_tool",
        "list_wandb_automations_tool",
        "list_wandb_integrations_tool",
        "probe_project_tool",
        "query_wandb_entity_projects",
        "query_wandb_tool",
        "query_weave_traces_tool",
        "resolve_trace_roots_tool",
        "search_wandb_docs_tool",
        "summarize_evaluation_tool",
    )
    bindings = [
        {
            "role": role,
            "import_id": import_id,
            "version_identity": version_identity,
        }
        for role, (import_id, version_identity) in zip(
            ("baseline", "candidate"),
            CURRENT_CANDIDATES,
            strict=True,
        )
    ]
    receipt = {
        "candidate_bindings": bindings,
        "candidate_bindings_digest": stable_digest(bindings),
        "candidates": [
            {
                "id": import_id,
                "version_identity": version_identity,
                "initialized_tools": list(tools),
                "initialized_tool_schema_digest": "e" * 64,
                "locked_tool_schema_digest": "e" * 64,
                "initialized_schema_matches_lock": True,
            }
            for import_id, version_identity in CURRENT_CANDIDATES
        ],
    }

    coverage = tool_surface_coverage_v1(
        receipt,
        task_ids=(
            "run-inventory-projection",
            "filtered-failure-triage",
            "evaluation-summary-accuracy",
            "exact-history-target",
            "selective-run-comparison",
            "artifact-provenance",
            "trace-source-use",
            "missing-cost-honesty",
        ),
    )

    assert coverage["taskset_comprehensive"] is True
    assert coverage["total_tools"] == 20

    receipt["candidates"][1]["version_identity"] = "git:" + "f" * 40
    with pytest.raises(ValueError, match="candidate versions do not match"):
        tool_surface_coverage_v1(receipt, task_ids=())


def test_qualifier_cli_threads_explicit_candidate_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = (
        Path.cwd()
        / "examples/comparisons/wandb-mcp-maintenance/"
        "qualify_locked_revisions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qualify_locked_revisions_under_test",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured: dict[str, object] = {}

    def fake_qualify(**kwargs):
        captured.update(kwargs)
        bindings = [
            {
                "role": role,
                "import_id": import_id,
                "version_identity": version_identity,
            }
            for role, (import_id, version_identity) in zip(
                ("baseline", "candidate"),
                kwargs["candidates"],
                strict=True,
            )
        ]
        return {
            "source_project": "wandb/source",
            "result_project": "wandb/result",
            "candidate_bindings": bindings,
            "findings": {},
            "recommendation": "mechanism only",
            "receipt_digest": "0" * 64,
        }

    monkeypatch.setattr(module, "qualify_locked_mcp_revisions", fake_qualify)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--evidence-lock",
            str(tmp_path / "evidence.json"),
            "--env-file",
            str(tmp_path / ".env"),
            "--output",
            str(tmp_path / "receipt.json"),
            "--baseline-import-id",
            BASELINE[0],
            "--baseline-version",
            BASELINE[1],
            "--candidate-import-id",
            CURRENT_CANDIDATE[0],
            "--candidate-version",
            CURRENT_CANDIDATE[1],
            "--release-notes-lock",
            "examples/comparisons/wandb-mcp-maintenance/"
            "release-notes.current.lock.json",
        ],
    )

    assert module.main() == 0
    assert captured["candidates"] == CURRENT_CANDIDATES
    assert captured["release_notes_lock"] == Path(
        "examples/comparisons/wandb-mcp-maintenance/"
        "release-notes.current.lock.json"
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["candidate_bindings"][1]["import_id"] == CURRENT_CANDIDATE[0]
