from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
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
from fugue.model_plane import resolve_model_route


def _experiment(*, size: int = 8):
    return experiment_from_data(
        {
            "id": "capability-eval",
            "title": "Capability evaluation",
            "model": "openai/gpt-5",
            "judge_model": "openai/gpt-5-mini",
            "harnesses": ["codex"],
            "variants": [
                {
                    "id": "baseline",
                    "label": "Baseline",
                    "context": {"system_id": "none", "delivery": "portable"},
                },
                {
                    "id": "with-skill",
                    "label": "With skill",
                    "context": {"system_id": "none", "delivery": "portable"},
                },
            ],
            "workloads": [{"id": "capabilities", "runner": "harbor"}],
            "evaluation_generation": {
                "suite_id": "capability-suite",
                "workload_id": "capabilities",
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


def _write_docs_integration(repo_root: Path) -> None:
    root = repo_root / "configs/fugue/integrations"
    root.mkdir(parents=True, exist_ok=True)
    (root / "docs.yaml").write_text(
        """id: docs
version: docs-server@1.0.0
runtime: {type: builtin, command: [docs-server]}
interfaces:
  - {type: mcp, name: docs, transport: stdio, allowed_tools: [search]}
"""
    )


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
    assert [item.type for item in workload.scorers] == ["builtin", "rubric"]
    assert workload.scorers[0].id == "harbor-outcome"
    assert workload.scorers[1].path == (
        "configs/fugue/evaluations/capability-suite/rubric.yaml"
    )


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
        (
            lambda cases: [
                {**cases[0], "instruction": "x" * 13_000},
                *cases[1:],
            ],
            "evaluation case exceeds 12000 serialized bytes",
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
                "suite_id": "capability-suite",
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
                "suite_id": "capability-suite",
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
                "suite_id": "capability-suite",
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
                "suite_id": "capability-suite",
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
                "suite_id": "capability-suite",
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
    skill = tmp_path / "configs/fugue/skills/always-on/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Always on\n")
    experiment = experiment_from_data(
        {
            **_experiment().to_dict(),
            "variants": [
                {
                    "id": "one",
                    "label": "One",
                    "skills": ["always-on"],
                    "context": {"system_id": "none", "delivery": "portable"},
                },
                {
                    "id": "two",
                    "label": "Two",
                    "skills": ["always-on"],
                    "context": {"system_id": "none", "delivery": "portable"},
                },
            ],
        }
    )
    with pytest.raises(ValueError, match="baseline that omits skill always-on"):
        build_evaluation_draft(
            {
                "suite_id": "capability-suite",
                "cases": _cases(),
                "rubric": _rubric(),
            },
            experiment,
            generator_model="openai/gpt-5-mini",
            source_catalog=source_catalog(experiment, tmp_path),
        )

def test_partial_generation_never_merges_hidden_saved_cases(
    tmp_path: Path,
) -> None:
    experiment, initial = _draft(tmp_path)
    for item in initial.files:
        path = tmp_path / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item.body)

    updated = {**_cases()[0], "instruction": "Updated boundary instruction."}
    with pytest.raises(ValueError, match="requires exactly 8 cases"):
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
                "suite_id": "lifecycle-suite",
                "workload_id": "capabilities",
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
                {
                    "id": "baseline",
                    "label": "Baseline",
                    "context": {"system_id": "none", "delivery": "portable"},
                },
                {
                    "id": "with-skill",
                    "label": "With skill",
                    "skills": ["demo-skill"],
                    "context": {"system_id": "none", "delivery": "portable"},
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
            "suite_id": "capability-suite",
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
        tmp_path / "configs/fugue/evaluations/capability-suite/manifest.yaml"
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


def test_case_assertions_do_not_select_a_rubric_scorer(tmp_path: Path) -> None:
    row = {"status": "passed", "pass": True}

    apply_generated_evaluation(
        row,
        case={
            **_cases(1)[0],
            "scorer_dimensions": ["task_completion", "artifact_quality"],
        },
        rubrics=[],
        judge_model="openai/gpt-5-mini",
        env={},
        trial_dir=tmp_path,
    )

    assert row["evaluation_assertions"]["task_completion"] == 1
    assert row["evaluation_judge_status"] == "not_requested"
    assert "evaluation_error" not in row


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
    _write_docs_integration(tmp_path)
    experiment = experiment_from_data(
        {
            "id": "mcp-preview",
            "integrations": [{"id": "docs"}],
            "evaluation_generation": {
                "suite_id": "mcp-preview-suite",
                "workload_id": "capabilities",
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
    _write_docs_integration(tmp_path)
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
            "integrations": [{"id": "docs"}],
            "evaluation_generation": {
                "suite_id": "mcp-generation-suite",
                "workload_id": "capabilities",
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


class _RecordingJudgeClient:
    def __init__(self) -> None:
        self.request: dict[str, Any] = {}

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.request = {"url": url, **kwargs}
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": '{"conditions": []}'}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 8, "output_tokens": 4},
            },
            request=httpx.Request("POST", url),
        )


def test_anthropic_judge_request_omits_deprecated_temperature() -> None:
    client = _RecordingJudgeClient()
    route = resolve_model_route("anthropic/claude-sonnet-5", {})

    payload, usage = evaluations._post_judge(
        client,  # type: ignore[arg-type]
        route,
        "test-key",
        {},
        "Return JSON.",
    )

    assert "temperature" not in client.request["json"]
    assert client.request["json"]["thinking"] == {"type": "disabled"}
    assert payload == {"conditions": []}
    assert usage == {"input_tokens": 8, "output_tokens": 4}


def test_anthropic_judge_request_uses_exact_json_schema_output_config() -> None:
    client = _RecordingJudgeClient()
    route = resolve_model_route("anthropic/claude-sonnet-5", {})
    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["conditions"],
        "properties": {
            "conditions": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
    }

    evaluations._post_judge(
        client,  # type: ignore[arg-type]
        route,
        "test-key",
        {},
        "Return JSON.",
        response_schema=response_schema,
    )

    assert client.request["json"]["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": response_schema,
        }
    }
    assert client.request["json"]["thinking"] == {"type": "disabled"}


def test_anthropic_judge_max_tokens_is_a_body_free_protocol_error() -> None:
    client = _RecordingJudgeClient()
    route = resolve_model_route("anthropic/claude-sonnet-5", {})
    response_text = '{"conditions": ['

    def truncated_response(url: str, **kwargs: Any) -> httpx.Response:
        client.request = {"url": url, **kwargs}
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": response_text}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 42, "output_tokens": 1_200},
            },
            request=httpx.Request("POST", url),
        )

    client.post = truncated_response  # type: ignore[method-assign]
    with pytest.raises(evaluations.JudgeResponseError) as caught:
        evaluations._post_judge(
            client,  # type: ignore[arg-type]
            route,
            "test-key",
            {},
            "Return JSON.",
        )

    error = caught.value
    assert error.stage == "response_generation"
    assert error.code == "max_tokens"
    assert error.safe_message == (
        "judge response reached the bounded output-token ceiling"
    )
    assert error.response_sha256 == hashlib.sha256(response_text.encode()).hexdigest()
    assert error.response_characters == len(response_text)
    assert error.usage == {"input_tokens": 42, "output_tokens": 1_200}
    assert response_text not in vars(error).values()


class _RecordingChatJudgeClient:
    def __init__(self) -> None:
        self.request: dict[str, Any] = {}

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.request = {"url": url, **kwargs}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"scores": {"useful": 1}}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            },
            request=httpx.Request("POST", url),
        )


def test_wandb_json_judge_request_uses_versioned_structured_controls() -> None:
    client = _RecordingChatJudgeClient()
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})

    payload, usage = evaluations._post_judge(
        client,  # type: ignore[arg-type]
        route,
        "test-key",
        {},
        "Return JSON.",
    )

    request_json = client.request["json"]
    assert request_json["max_tokens"] == evaluations.JUDGE_JSON_MAX_OUTPUT_TOKENS
    assert request_json["thinking"] == {"type": "disabled"}
    assert request_json["temperature"] == 0
    assert evaluations.JUDGE_JSON_REQUEST_POLICY_SCHEMA_VERSION == 2
    assert payload == {"scores": {"useful": 1}}
    assert usage == {"input_tokens": 12, "output_tokens": 7}


def test_json_judge_no_object_error_retains_only_safe_response_metadata() -> None:
    client = _RecordingChatJudgeClient()
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    response_text = "No structured answer was emitted."

    def no_json_response(url: str, **kwargs: Any) -> httpx.Response:
        client.request = {"url": url, **kwargs}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": response_text}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1_200},
            },
            request=httpx.Request("POST", url),
        )

    client.post = no_json_response  # type: ignore[method-assign]
    with pytest.raises(evaluations.JudgeResponseError) as caught:
        evaluations._post_judge(
            client,  # type: ignore[arg-type]
            route,
            "test-key",
            {},
            "Return JSON.",
        )

    error = caught.value
    assert error.stage == "response_extraction"
    assert error.code == "no_json_object"
    assert error.safe_message == "judge returned no JSON object"
    assert error.response_sha256 == hashlib.sha256(response_text.encode()).hexdigest()
    assert error.response_characters == len(response_text)
    assert error.usage == {"input_tokens": 12, "output_tokens": 1_200}
    assert response_text not in vars(error).values()


def test_json_judge_rejects_response_above_the_bounded_envelope() -> None:
    client = _RecordingChatJudgeClient()
    route = resolve_model_route("wandb/zai-org/GLM-5.2", {})
    response_text = "x" * (evaluations.JUDGE_JSON_MAX_RESPONSE_CHARACTERS + 1)

    def oversized_response(url: str, **kwargs: Any) -> httpx.Response:
        client.request = {"url": url, **kwargs}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": response_text}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1_200},
            },
            request=httpx.Request("POST", url),
        )

    client.post = oversized_response  # type: ignore[method-assign]
    with pytest.raises(evaluations.JudgeResponseError) as caught:
        evaluations._post_judge(
            client,  # type: ignore[arg-type]
            route,
            "test-key",
            {},
            "Return JSON.",
        )

    error = caught.value
    assert error.stage == "response_validation"
    assert error.code == "response_too_large"
    assert error.response_characters == len(response_text)
    assert error.response_sha256 == hashlib.sha256(response_text.encode()).hexdigest()
    assert error.usage == {"input_tokens": 12, "output_tokens": 1_200}
    assert response_text not in vars(error).values()
