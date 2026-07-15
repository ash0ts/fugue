from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_operator import make_operator_repo

from fugue.bench import evaluations
from fugue.bench.ai import AssetDraft
from fugue.bench.datasets import DATASET_MANIFEST, materialize_manifest_dataset
from fugue.bench.evaluations import (
    CASE_FILE,
    MANIFEST_FILE,
    RUBRIC_FILE,
    apply_generated_evaluation,
    build_evaluation_draft,
    source_catalog,
)
from fugue.bench.library import experiment_from_data
from fugue.bench.manifest import load_manifest


def _experiment(*, size: int = 8):
    return experiment_from_data(
        {
            "id": "capability-eval",
            "title": "Capability evaluation",
            "model": "openai/gpt-5",
            "judge_model": "openai/gpt-5-mini",
            "harnesses": ["codex"],
            "variants": [
                {"id": "baseline", "label": "Baseline"},
                {"id": "with-skill", "label": "With skill"},
            ],
            "workloads": [{"id": "capabilities", "runner": "harbor"}],
            "evaluation_generation": {
                "size": size,
                "sources": [
                    {
                        "kind": "seed",
                        "text": "Fugue evaluates controlled agent capability variants.",
                    }
                ],
            },
        }
    )


def _cases(count: int = 8) -> list[dict]:
    strata = ["easy", "boundary", "failure", "integration"]
    return [
        {
            "id": f"case-{index + 1:02d}",
            "instruction": f"Explain capability behavior for scenario {index + 1}.",
            "family": "skill" if index % 2 else "agent",
            "source_refs": ["seed:1"],
            "expected": {"facts": ["controlled agent capability variants"]},
            "tags": [strata[index % len(strata)]],
        }
        for index in range(count)
    ]


def _rubric() -> dict:
    return {
        "dimensions": [
            {
                "id": "task_completion",
                "criterion": "The requested task is completed.",
            },
            {
                "id": "correctness",
                "criterion": "The answer includes all asserted facts.",
                "threshold": 0.7,
            },
            {
                "id": "groundedness",
                "criterion": "Claims are grounded in the cited sources.",
            },
        ]
    }


def _draft(tmp_path: Path):
    experiment = _experiment()
    sources = source_catalog(experiment, tmp_path)
    return build_evaluation_draft(
        {
            "suite_id": "capability-suite",
            "cases": _cases(),
            "rubric": _rubric(),
        },
        experiment,
        generator_model="openai/gpt-5-mini",
        source_catalog=sources,
        repo_root=tmp_path,
    )


def test_evaluation_draft_is_stratified_grounded_and_reviewable(
    tmp_path: Path,
) -> None:
    experiment, draft = _draft(tmp_path)

    assert len(draft.cases) == 8
    assert draft.coverage == {"agent": 4, "skill": 4}
    assert {tag for case in draft.cases for tag in case["tags"]} == {
        "easy",
        "boundary",
        "failure",
        "integration",
    }
    assert {item.path.name for item in draft.files} == {
        CASE_FILE,
        RUBRIC_FILE,
        MANIFEST_FILE,
    }
    assert {item["threshold"] for item in draft.rubric["dimensions"]} == {0.7}
    workload = experiment.workloads[0]
    assert workload.manifest == Path(
        "configs/fugue/evaluations/capability-suite/manifest.yaml"
    )
    assert workload.scorers == [
        "builtin:harbor-outcome",
        "configs/fugue/evaluations/capability-suite/rubric.yaml",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cases: cases[:7], "exactly 8 cases"),
        (
            lambda cases: [
                *cases[:7],
                {**cases[7], "id": cases[0]["id"]},
            ],
            "duplicate evaluation case",
        ),
        (
            lambda cases: [
                {**cases[0], "expected": {"reference_answer": "optional"}},
                *cases[1:],
            ],
            "needs a fact, tool, or artifact assertion",
        ),
        (
            lambda cases: [
                {**case, "tags": ["easy"]} for case in cases
            ],
            "missing case strata",
        ),
    ],
)
def test_evaluation_draft_rejects_invalid_case_sets(
    tmp_path: Path, mutate, message: str
) -> None:
    experiment = _experiment()
    with pytest.raises(ValueError, match=message):
        build_evaluation_draft(
            {
                "suite_id": "invalid-suite",
                "cases": mutate(_cases()),
                "rubric": _rubric(),
            },
            experiment,
            generator_model="openai/gpt-5-mini",
            source_catalog=source_catalog(experiment, tmp_path),
            repo_root=tmp_path,
        )


def test_evaluation_rejects_unknown_sources_and_invalid_thresholds(
    tmp_path: Path,
) -> None:
    experiment = _experiment()
    invalid_cases = _cases()
    invalid_cases[0]["source_refs"] = ["seed:missing"]
    with pytest.raises(ValueError, match="unknown source ref"):
        build_evaluation_draft(
            {
                "suite_id": "ungrounded",
                "cases": invalid_cases,
                "rubric": _rubric(),
            },
            experiment,
            generator_model="openai/gpt-5-mini",
            source_catalog=source_catalog(experiment, tmp_path),
        )

    rubric = _rubric()
    rubric["dimensions"][0]["threshold"] = 1.1
    with pytest.raises(ValueError, match="threshold must be 0..1"):
        build_evaluation_draft(
            {
                "suite_id": "bad-threshold",
                "cases": _cases(),
                "rubric": rubric,
            },
            experiment,
            generator_model="openai/gpt-5-mini",
            source_catalog=source_catalog(experiment, tmp_path),
        )

    unsafe_cases = _cases()
    unsafe_cases[0]["attachments"] = [
        {"path": "../secret", "target": "secret", "sha256": "0" * 64}
    ]
    with pytest.raises(ValueError, match="repository-relative"):
        build_evaluation_draft(
            {
                "suite_id": "unsafe-attachment",
                "cases": unsafe_cases,
                "rubric": _rubric(),
            },
            experiment,
            generator_model="openai/gpt-5-mini",
            source_catalog=source_catalog(experiment, tmp_path),
        )

    unsafe_artifact_cases = _cases()
    unsafe_artifact_cases[0]["expected"] = {
        "artifacts": [{"path": "/tmp/result.txt"}]
    }
    with pytest.raises(ValueError, match="artifact path must start"):
        build_evaluation_draft(
            {
                "suite_id": "unsafe-artifact",
                "cases": unsafe_artifact_cases,
                "rubric": _rubric(),
            },
            experiment,
            generator_model="openai/gpt-5-mini",
            source_catalog=source_catalog(experiment, tmp_path),
        )


def test_generated_evaluation_requires_feature_omission_baseline(
    tmp_path: Path,
) -> None:
    experiment = experiment_from_data(
        {
            **_experiment().to_dict(),
            "variants": [
                {
                    "id": "one",
                    "label": "One",
                    "skill_ids": ["always-on"],
                },
                {
                    "id": "two",
                    "label": "Two",
                    "skill_ids": ["always-on"],
                },
            ],
        }
    )
    with pytest.raises(ValueError, match="baseline that omits skill always-on"):
        build_evaluation_draft(
            {
                "suite_id": "no-baseline",
                "cases": _cases(),
                "rubric": _rubric(),
            },
            experiment,
            generator_model="openai/gpt-5-mini",
            source_catalog=source_catalog(experiment, tmp_path),
        )

def test_partial_generation_fills_saved_gaps_and_detects_source_drift(
    tmp_path: Path,
) -> None:
    experiment, initial = _draft(tmp_path)
    for item in initial.files:
        path = tmp_path / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item.body)

    updated = {**_cases()[0], "instruction": "Updated boundary instruction."}
    _, repaired = build_evaluation_draft(
        {
            "suite_id": "capability-suite",
            "cases": [updated],
            "rubric": _rubric(),
        },
        _experiment(),
        generator_model="openai/gpt-5-mini",
        source_catalog=source_catalog(_experiment(), tmp_path),
        repo_root=tmp_path,
    )

    assert len(repaired.cases) == 8
    assert repaired.cases[0]["instruction"] == "Updated boundary instruction."

    cases_path = tmp_path / initial.files[0].path
    rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
    rows[1]["source_refs"][0]["sha256"] = "0" * 64
    cases_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="source drifted"):
        build_evaluation_draft(
            {
                "suite_id": "capability-suite",
                "cases": [updated],
                "rubric": _rubric(),
            },
            _experiment(),
            generator_model="openai/gpt-5-mini",
            source_catalog=source_catalog(_experiment(), tmp_path),
            repo_root=tmp_path,
        )


def test_generated_harbor_dataset_is_atomic_reusable_and_checksum_pinned(
    tmp_path: Path,
) -> None:
    _, draft = _draft(tmp_path)
    for item in draft.files:
        path = tmp_path / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item.body)
    manifest = load_manifest(
        tmp_path / "configs/fugue/evaluations/capability-suite/manifest.yaml"
    )

    first = materialize_manifest_dataset(manifest, tmp_path)
    second = materialize_manifest_dataset(manifest, tmp_path)

    assert first == second
    assert first is not None
    assert json.loads((first / DATASET_MANIFEST).read_text())["metrics"] == {
        "suite_id": "capability-suite",
        "tasks": 8,
    }
    task = first / "case-01"
    assert (task / "task.toml").is_file()
    assert (task / "environment/Dockerfile").is_file()
    assert (task / "tests/test.sh").stat().st_mode & 0o111
    assert "controlled agent capability variants" not in (
        task / "instruction.md"
    ).read_text()

    cases_path = tmp_path / "configs/fugue/evaluations/capability-suite/cases.jsonl"
    cases_path.write_text(cases_path.read_text() + "\n")
    drifted = replace(
        manifest,
        dataset=replace(
            manifest.dataset,
            path=Path(".fugue/cache/datasets/generated/drifted"),
        ),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        materialize_manifest_dataset(drifted, tmp_path)


def test_generated_evaluation_lifecycle_preview_save_prepare_and_render(
    tmp_path: Path,
) -> None:
    service = make_operator_repo(tmp_path)
    raw = service.experiment("demo").to_dict()
    raw.update(
        {
            "id": "generated-lifecycle",
            "title": "Generated lifecycle",
            "judge_model": "openai/gpt-5-mini",
            "evaluation_generation": {
                "size": 8,
                "sources": [
                    {
                        "kind": "seed",
                        "text": "The demo skill uses focused repository search.",
                    }
                ],
            },
            "workloads": [{"id": "capabilities", "runner": "harbor"}],
            "variants": [
                {"id": "baseline", "label": "Baseline"},
                {
                    "id": "with-skill",
                    "label": "With skill",
                    "skill_ids": ["demo-skill"],
                },
            ],
        }
    )
    experiment = experiment_from_data(raw)
    updated, draft = build_evaluation_draft(
        {
            "suite_id": "lifecycle-suite",
            "cases": _cases(),
            "rubric": _rubric(),
        },
        experiment,
        generator_model="openai/gpt-5-mini",
        source_catalog=source_catalog(experiment, tmp_path),
        repo_root=tmp_path,
    )
    assets = tuple(
        AssetDraft(
            kind=item.kind,
            id=item.suite_id,
            title=item.path.name,
            body=item.body,
        )
        for item in draft.files
    )
    request = service.request_for_experiment(updated)

    preview = service.preview_experiment(
        updated,
        request=request,
        asset_overlay=draft.overlay,
    )

    assert preview.cells == 16
    assert preview.estimated_trials == 16
    assert not (tmp_path / "configs/fugue/evaluations/lifecycle-suite").exists()
    assert not (tmp_path / ".fugue").exists()

    saved = service.save_working_experiment(
        updated,
        request,
        experiment_id="saved-generated-lifecycle",
        assets=assets,
    )
    saved_request = service.request_for_experiment(saved)
    preparation = service.prepare_context(saved_request, experiment=saved)

    assert preparation == ()
    manifest = load_manifest(
        tmp_path / "configs/fugue/evaluations/lifecycle-suite/manifest.yaml"
    )
    dataset = tmp_path / manifest.dataset.path
    assert (dataset / DATASET_MANIFEST).is_file()

    rendered = service.rendered_jobs(
        saved_request,
        run_id="lifecycle-run",
        experiment=saved,
    )

    assert len(rendered) == 16
    assert all(job.config_path.is_file() for job in rendered)
    assert all(job.evaluation_case is not None for job in rendered)
    for task_id in {job.task_id for job in rendered}:
        task_jobs = [job for job in rendered if job.task_id == task_id]
        assert {job.variant_id for job in task_jobs} == {"baseline", "with-skill"}
        assert len({job.comparison_example_id for job in task_jobs}) == 1


def test_attachment_checksum_and_repository_boundary_are_enforced(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("safe fixture")
    cases = _cases()
    cases[0]["attachments"] = [
        {
            "path": "fixture.txt",
            "target": "fixture.txt",
            "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        }
    ]
    experiment = _experiment()
    _, draft = build_evaluation_draft(
        {
            "suite_id": "attachment-suite",
            "cases": cases,
            "rubric": _rubric(),
        },
        experiment,
        generator_model="openai/gpt-5-mini",
        source_catalog=source_catalog(experiment, tmp_path),
        repo_root=tmp_path,
    )
    for item in draft.files:
        path = tmp_path / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item.body)
    fixture.write_text("changed")
    manifest = load_manifest(
        tmp_path / "configs/fugue/evaluations/attachment-suite/manifest.yaml"
    )
    with pytest.raises(ValueError, match="attachment checksum mismatch"):
        materialize_manifest_dataset(manifest, tmp_path)
    assert not (tmp_path / manifest.dataset.path).exists()


def test_generated_scoring_is_separate_supports_na_and_preserves_outcome(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial"
    artifacts = trial / "agent/artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "fugue-answer.md").write_text(
        "This compares controlled agent capability variants."
    )
    case = {
        **_cases(1)[0],
        "scorer_dimensions": [
            "task_completion",
            "correctness",
            "groundedness",
            "tool_use",
        ],
        "expected": {
            "facts": ["controlled agent capability variants"],
            "tool_calls": [{"tool": "search", "arguments_subset": {}}],
            "artifacts": [],
        },
    }
    rubric = {
        "dimensions": [
            {"id": dimension, "criterion": dimension, "threshold": 0.7}
            for dimension in case["scorer_dimensions"]
        ]
    }
    calls = 0

    def judge(**kwargs):
        nonlocal calls
        calls += 1
        return (
            {
                "scores": {
                    "task_completion": 1,
                    "correctness": 0.9,
                    "groundedness": 0.8,
                    "tool_use": 1,
                },
                "reasons": {
                    "correctness": "Grounded; never expose sk-abcdefghijklmnop."
                },
            },
            {"input_tokens": 10, "output_tokens": 5},
        )

    row = {"status": "passed", "pass": True}
    apply_generated_evaluation(
        row,
        case=case,
        rubrics=[rubric],
        judge_model="openai/gpt-5-mini",
        env={},
        trial_dir=trial,
        judge_request=judge,
    )

    assert calls == 1
    assert row["pass"] is True
    assert row["evaluation_task_completion"] == 1
    assert row["evaluation_correctness"] == 0.9
    assert "evaluation_tool_use" not in row
    assert row["evaluation_na_dimensions"] == ["tool_use"]
    assert "sk-abcdefghijklmnop" not in json.dumps(
        row["evaluation_judge_reasons"]
    )
    assert "evaluation_overall" not in row


def test_judge_failure_is_an_evaluation_error_not_a_harbor_failure(
    tmp_path: Path,
) -> None:
    row = {"status": "passed", "pass": True}

    def failed(**kwargs):
        raise RuntimeError("provider unavailable")

    apply_generated_evaluation(
        row,
        case={
            **_cases(1)[0],
            "scorer_dimensions": ["task_completion"],
        },
        rubrics=[
            {
                "dimensions": [
                    {
                        "id": "task_completion",
                        "criterion": "complete",
                        "threshold": 0.7,
                    }
                ]
            }
        ],
        judge_model="openai/gpt-5-mini",
        env={},
        trial_dir=tmp_path,
        judge_request=failed,
    )

    assert row["pass"] is True
    assert "provider unavailable" in row["evaluation_error"]
    assert "evaluation_task_completion" not in row


def test_artifact_assertions_bound_the_structured_judge_score(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial"
    artifact = trial / "logs" / "artifacts" / "report.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"status": "complete"}')
    case = {
        **_cases(1)[0],
        "scorer_dimensions": ["task_completion", "artifact_quality"],
        "expected": {
            "facts": [],
            "tool_calls": [],
            "artifacts": [
                {
                    "path": "/logs/artifacts/report.json",
                    "checks": ["exists", "nonempty", "json"],
                }
            ],
        },
    }
    rubric = {
        "dimensions": [
            {"id": "task_completion", "criterion": "complete", "threshold": 0.7},
            {"id": "artifact_quality", "criterion": "valid", "threshold": 0.7},
        ]
    }
    deterministic_inputs: list[dict] = []

    def judge(**kwargs):
        deterministic_inputs.append(kwargs["deterministic"])
        return (
            {
                "scores": {"task_completion": 1, "artifact_quality": 1},
                "reasons": {},
            },
            {},
        )

    complete = {"status": "passed", "pass": True}
    apply_generated_evaluation(
        complete,
        case=case,
        rubrics=[rubric],
        judge_model="openai/gpt-5-mini",
        env={},
        trial_dir=trial,
        judge_request=judge,
    )

    artifact.unlink()
    missing = {"status": "passed", "pass": True}
    apply_generated_evaluation(
        missing,
        case=case,
        rubrics=[rubric],
        judge_model="openai/gpt-5-mini",
        env={},
        trial_dir=trial,
        judge_request=judge,
    )

    assert deterministic_inputs[0]["artifact_quality"] == 1
    assert complete["evaluation_artifact_quality"] == 1
    assert deterministic_inputs[1]["artifact_quality"] == 0
    assert missing["evaluation_artifact_quality"] == 0
    assert missing["pass"] is True


def test_preview_source_resolution_never_opens_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = experiment_from_data(
        {
            "id": "mcp-preview",
            "mcp_servers": [{"name": "docs", "command": "docs-server"}],
            "evaluation_generation": {
                "sources": [
                    {
                        "kind": "mcp",
                        "server": "docs",
                        "tools": ["search"],
                        "resources": ["docs://schema"],
                    }
                ]
            },
        }
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("preview attempted MCP I/O")

    monkeypatch.setattr(evaluations, "_discover_mcp_source", forbidden)
    sources = source_catalog(experiment, tmp_path, allow_mcp_io=False)

    assert len(sources) == 1
    assert sources[0].metadata["discovery"] == "declared"
    assert "search" in sources[0].content


def test_generation_discovers_only_mcp_schemas_and_explicit_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operations: list[tuple[str, str | None]] = []

    class AsyncContext:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    class FakeSession:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def initialize(self):
            operations.append(("initialize", None))

        async def list_tools(self):
            operations.append(("list_tools", None))
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="search",
                        description="Search documentation",
                        inputSchema={"type": "object"},
                    ),
                    SimpleNamespace(
                        name="mutate",
                        description="Must not be selected",
                        inputSchema={"type": "object"},
                    ),
                ]
            )

        async def read_resource(self, uri):
            operations.append(("read_resource", str(uri)))
            return SimpleNamespace(
                contents=[SimpleNamespace(text="Explicit schema resource")]
            )

    mcp_module = types.ModuleType("mcp")
    mcp_module.ClientSession = FakeSession
    mcp_module.StdioServerParameters = lambda **kwargs: SimpleNamespace(**kwargs)
    client_module = types.ModuleType("mcp.client")
    stdio_module = types.ModuleType("mcp.client.stdio")
    stdio_module.stdio_client = lambda params: AsyncContext((object(), object()))
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.client", client_module)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_module)

    experiment = experiment_from_data(
        {
            "id": "mcp-generation",
            "mcp_servers": [
                {
                    "name": "docs",
                    "command": "docs-server",
                    "env": {"DOCS_TOKEN": "secret-value"},
                }
            ],
            "evaluation_generation": {
                "sources": [
                    {
                        "kind": "mcp",
                        "server": "docs",
                        "tools": ["search"],
                        "resources": ["docs://explicit"],
                    }
                ]
            },
        }
    )

    async def discover():
        return source_catalog(experiment, tmp_path, allow_mcp_io=True)

    sources = asyncio.run(discover())

    assert operations == [
        ("initialize", None),
        ("list_tools", None),
        ("read_resource", "docs://explicit"),
    ]
    assert {source.id for source in sources} == {
        "mcp:docs",
        "mcp:docs:tools",
        "mcp:docs:resource:docs://explicit",
    }
    tools = next(source for source in sources if source.id == "mcp:docs:tools")
    assert "search" in tools.content
    assert "mutate" not in tools.content
    assert "secret-value" not in json.dumps([source.public() for source in sources])
