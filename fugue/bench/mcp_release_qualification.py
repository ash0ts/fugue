from __future__ import annotations

import asyncio
import json
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import weave

from fugue.bench.candidates import stable_digest
from fugue.bench.files import atomic_write_json
from fugue.bench.operator import load_env

QUALIFICATION_PROJECT = (
    "wandb/fugue-mcp-release-qualification-v1"
)
EVIDENCE_LOCK_SCHEMA_VERSION = 1
_REQUIRED_COUNTS = {
    "runs": 6,
    "source_conversations": 24,
    "tool_spans": 48,
    "dataset_rows": 8,
    "aligned_evaluation_pairs": 8,
    "evaluation_prediction_rows": 16,
}

_RUNS: tuple[dict[str, Any], ...] = (
    {
        "id": "maint-r18-01",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "inventory-openclaw",
        "latency_ms": 820,
        "observed_cost_usd": 0.11,
        "deterministic_pass": True,
        "projected_reads": 3,
        "broad_reads": 0,
        "source_returned": 4,
        "source_opened": 4,
    },
    {
        "id": "maint-r18-02",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "coverage-claude",
        "latency_ms": 860,
        "observed_cost_usd": 0.14,
        "deterministic_pass": True,
        "projected_reads": 4,
        "broad_reads": 0,
        "source_returned": 4,
        "source_opened": 4,
    },
    {
        "id": "maint-r18-03",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "regression-openclaw",
        "latency_ms": 900,
        "observed_cost_usd": 0.16,
        "deterministic_pass": True,
        "projected_reads": 2,
        "broad_reads": 1,
        "source_returned": 4,
        "source_opened": 3,
    },
    {
        "id": "maint-r18-04",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "partial-evidence-claude",
        "latency_ms": 940,
        "observed_cost_usd": 0.19,
        "deterministic_pass": False,
        "projected_reads": 2,
        "broad_reads": 1,
        "source_returned": 4,
        "source_opened": 1,
    },
    {
        "id": "maint-r18-05",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "cost-audit-openclaw",
        "latency_ms": 980,
        "observed_cost_usd": 0.36,
        "deterministic_pass": True,
        "projected_reads": 3,
        "broad_reads": 0,
        "source_returned": 4,
        "source_opened": 4,
    },
    {
        "id": "maint-r18-06",
        "candidate_revision": "maintainer-r18",
        "attempt_label": "broad-history-scan-claude",
        "latency_ms": 4200,
        "observed_cost_usd": None,
        "deterministic_pass": False,
        "projected_reads": 0,
        "broad_reads": 4,
        "source_returned": 4,
        "source_opened": 1,
    },
)

_EVALUATION_ROWS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "coverage-inventory",
        "question": "Count only inspected evidence.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
    {
        "case_id": "aligned-regression",
        "question": "Compare aligned evaluation revisions.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
    {
        "case_id": "latency-anomaly",
        "question": "Identify the observed latency outlier.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
    {
        "case_id": "cost-coverage",
        "question": "Do not replace missing observed cost with zero.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
    {
        "case_id": "source-opened",
        "question": "Separate source return from source use.",
        "baseline_pass": False,
        "candidate_pass": True,
    },
    {
        "case_id": "partial-evidence",
        "question": "Return a structured incomplete-evidence result.",
        "baseline_pass": True,
        "candidate_pass": False,
    },
    {
        "case_id": "maintenance-priority",
        "question": "Recommend one evidence-linked maintenance action.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
    {
        "case_id": "release-completeness",
        "question": "Refuse an unsupported release-wide claim.",
        "baseline_pass": True,
        "candidate_pass": True,
    },
)


def qualification_seed() -> dict[str, Any]:
    latencies = [int(item["latency_ms"]) for item in _RUNS]
    observed_costs = [
        float(item["observed_cost_usd"])
        for item in _RUNS
        if item["observed_cost_usd"] is not None
    ]
    baseline_passes = {
        str(item["case_id"]): bool(item["baseline_pass"])
        for item in _EVALUATION_ROWS
    }
    candidate_passes = {
        str(item["case_id"]): bool(item["candidate_pass"])
        for item in _EVALUATION_ROWS
    }
    regressions = sorted(
        case_id
        for case_id, passed in baseline_passes.items()
        if passed and not candidate_passes[case_id]
    )
    return {
        "schema_version": 1,
        "project": QUALIFICATION_PROJECT,
        "runs": [dict(item) for item in _RUNS],
        "evaluation_rows": [dict(item) for item in _EVALUATION_ROWS],
        "facts": {
            "run_count": len(_RUNS),
            "source_conversation_count": len(_RUNS) * 4,
            "tool_span_count": len(_RUNS) * 8,
            "aligned_evaluation_pairs": len(_EVALUATION_ROWS),
            "evaluation_prediction_rows": len(_EVALUATION_ROWS) * 2,
            "latency_anomaly": {
                "attempt_label": "broad-history-scan-claude",
                "latency_ms": 4200,
                "cohort_median_ms": int(statistics.median(latencies)),
                "ratio": round(4200 / statistics.median(latencies), 4),
            },
            "cost_coverage": {
                "attempts": len(_RUNS),
                "attempts_with_observed_cost": len(observed_costs),
                "total_observed_usd": round(sum(observed_costs), 6),
                "complete": False,
            },
            "reviewed_failure_cohort": {
                "reviewed_examples": 4,
                "current_source_returned": 4,
                "current_source_opened": 1,
                "failure_class": "current-source-returned-mostly-not-opened",
            },
            "regressions": regressions,
            "missing_evidence": "maintainer-r18-partial-evaluation",
        },
    }


def qualification_seed_digest() -> str:
    return stable_digest(qualification_seed())


def validate_evidence_lock(
    value: Mapping[str, Any],
    *,
    expected_project: str = QUALIFICATION_PROJECT,
) -> dict[str, Any]:
    raw = dict(value)
    supplied = str(raw.get("evidence_lock_digest") or "")
    unsigned = dict(raw)
    unsigned["evidence_lock_digest"] = ""
    if len(supplied) != 64 or stable_digest(unsigned) != supplied:
        raise ValueError("evidence lock digest does not match")
    if raw.get("schema_version") != EVIDENCE_LOCK_SCHEMA_VERSION:
        raise ValueError("evidence lock schema is unsupported")
    if raw.get("project") != expected_project:
        raise ValueError("evidence lock project does not match")
    if raw.get("seed_digest") != qualification_seed_digest():
        raise ValueError("evidence lock seed digest does not match")
    counts = raw.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("evidence lock counts are missing")
    for name, expected in _REQUIRED_COUNTS.items():
        if int(counts.get(name) or 0) != expected:
            raise ValueError(
                f"evidence lock count {name} must equal {expected}"
            )
    objects = raw.get("objects")
    if not isinstance(objects, Mapping):
        raise ValueError("evidence lock objects are missing")
    runs = objects.get("runs")
    conversations = objects.get("source_conversations")
    evaluations = objects.get("evaluations")
    if not isinstance(runs, list) or len(runs) != _REQUIRED_COUNTS["runs"]:
        raise ValueError("evidence lock must contain every W&B Run")
    if (
        not isinstance(conversations, list)
        or len(conversations) != _REQUIRED_COUNTS["source_conversations"]
    ):
        raise ValueError("evidence lock must contain every source conversation")
    if not isinstance(evaluations, list) or len(evaluations) != 2:
        raise ValueError("evidence lock must contain both evaluation revisions")
    dataset = objects.get("dataset")
    if not isinstance(dataset, Mapping) or not str(dataset.get("ref") or "").startswith(
        "weave:///"
    ):
        raise ValueError("evidence lock Dataset reference is not immutable")
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise ValueError("evidence lock Evaluation entry is invalid")
        if not str(evaluation.get("ref") or "").startswith("weave:///"):
            raise ValueError("evidence lock Evaluation reference is not immutable")
        if not str(evaluation.get("call_ref") or "").startswith("weave:///"):
            raise ValueError("evidence lock Evaluation call is not immutable")
    return raw


def prepare_hosted_project(
    *,
    project: str,
    output: Path,
    env_file: Path,
) -> dict[str, Any]:
    if project != QUALIFICATION_PROJECT:
        raise ValueError(
            f"qualification project must be {QUALIFICATION_PROJECT}"
        )
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        return validate_evidence_lock(existing, expected_project=project)
    env = load_env(env_file)
    required = ("WANDB_API_KEY",)
    missing = [name for name in required if not str(env.get(name) or "").strip()]
    if missing:
        raise RuntimeError(
            "hosted-project preparation requires: " + ", ".join(missing)
        )
    entity, project_name = project.split("/", 1)
    selected_env = {
        "WANDB_API_KEY": str(env["WANDB_API_KEY"]),
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project_name,
        "WANDB_BASE_URL": str(
            env.get("WANDB_BASE_URL") or "https://api.wandb.ai"
        ),
        "WANDB_SILENT": "true",
    }
    with _temporary_environment(selected_env):
        run_receipts = _prepare_wandb_runs(entity, project_name)
        weave_receipts = _prepare_weave_evidence(project)
    value = _evidence_lock(project, run_receipts, weave_receipts)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, value)
    return validate_evidence_lock(value, expected_project=project)


def _prepare_wandb_runs(
    entity: str,
    project: str,
) -> list[dict[str, Any]]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "install fugue[research-worker] with the W&B Sandbox extra"
        ) from exc
    api = wandb.Api()
    receipts: list[dict[str, Any]] = []
    seed_digest = qualification_seed_digest()
    for item in _RUNS:
        run_path = f"{entity}/{project}/{item['id']}"
        try:
            existing = api.run(run_path)
        except (wandb.errors.CommError, ValueError):
            existing = None
        if existing is not None:
            if existing.config.get("fugue_seed_digest") != seed_digest:
                raise RuntimeError(
                    f"existing W&B Run has another seed identity: {run_path}"
                )
            receipts.append(
                {
                    "id": item["id"],
                    "name": existing.name,
                    "url": existing.url,
                    "ref": f"wandb-run:///{run_path}",
                    "seed_digest": seed_digest,
                    "state": existing.state,
                }
            )
            continue
        settings = wandb.Settings(
            console="off",
            disable_git=True,
            silent=True,
        )
        run = wandb.init(
            entity=entity,
            project=project,
            id=str(item["id"]),
            name=str(item["attempt_label"]),
            group="hosted-evidence-v1",
            job_type="maintenance-evidence",
            tags=("fugue", "qualification", "mcp-release"),
            config={
                "fugue_seed_digest": seed_digest,
                "candidate_revision": item["candidate_revision"],
                "attempt_label": item["attempt_label"],
                "evidence_snapshot": "qualification-v1",
                "contains_sensitive_data": False,
            },
            resume="never",
            reinit="create_new",
            settings=settings,
        )
        if run is None:
            raise RuntimeError(f"failed to create W&B Run {item['id']}")
        try:
            _log_run_history(run, item)
            artifact_receipt = _log_run_artifact(run, item, seed_digest)
            run.summary.update(
                {
                    "fugue_seed_digest": seed_digest,
                    "evidence_lock_status": "prepared",
                    "artifact_digest": artifact_receipt["digest"],
                }
            )
            url = run.url
        finally:
            run.finish()
        receipts.append(
            {
                "id": item["id"],
                "name": item["attempt_label"],
                "url": url,
                "ref": f"wandb-run:///{run_path}",
                "seed_digest": seed_digest,
                "state": "finished",
                "artifact": artifact_receipt,
            }
        )
    return receipts


def _log_run_history(run: Any, item: Mapping[str, Any]) -> None:
    latency = int(item["latency_ms"])
    for step, multiplier in enumerate((0.7, 0.85, 1.0), start=1):
        payload: dict[str, Any] = {
            "step": step,
            "latency_ms": round(latency * multiplier, 3),
            "deterministic_pass": int(bool(item["deterministic_pass"])),
            "projected_reads": int(item["projected_reads"]),
            "broad_reads": int(item["broad_reads"]),
            "source_returned": int(item["source_returned"]),
            "source_opened": int(item["source_opened"]),
        }
        if item["observed_cost_usd"] is not None:
            payload["observed_cost_usd"] = round(
                float(item["observed_cost_usd"]) * multiplier,
                6,
            )
        run.log(payload, step=step)


def _log_run_artifact(
    run: Any,
    item: Mapping[str, Any],
    seed_digest: str,
) -> dict[str, Any]:
    import wandb

    payload = {
        "schema_version": 1,
        "seed_digest": seed_digest,
        "run_id": item["id"],
        "attempt_label": item["attempt_label"],
        "evidence": {
            key: value
            for key, value in item.items()
            if key not in {"id", "attempt_label"}
        },
    }
    with tempfile.TemporaryDirectory(prefix="fugue-wandb-evidence-") as temp:
        path = Path(temp) / "attempt-evidence.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact = wandb.Artifact(
            name=f"qualification-evidence-{item['id']}",
            type="fugue-qualification-evidence",
            metadata={"seed_digest": seed_digest},
        )
        artifact.add_file(path.as_posix(), name="attempt-evidence.json")
        logged = run.log_artifact(
            artifact,
            aliases=("qualification-v1",),
        )
        logged.wait()
    return {
        "name": logged.name,
        "version": logged.version,
        "digest": logged.digest,
        "qualified_name": logged.qualified_name,
    }


def _prepare_weave_evidence(project: str) -> dict[str, Any]:
    client = weave.init(project)
    dataset = weave.Dataset(
        name="mcp-release-maintenance-cases",
        description=(
            "Reviewed non-sensitive maintenance cases for Fugue MCP release "
            "qualification."
        ),
        rows=[dict(item) for item in _EVALUATION_ROWS],
    )
    dataset_ref = weave.publish(
        dataset,
        name="mcp-release-maintenance-cases",
        aliases=["qualification-v1"],
    )
    conversations = _existing_source_conversations(client, project)
    if not conversations:
        for run in _RUNS:
            for index in range(1, 5):
                _, call = _seed_agent_conversation.call(
                    run_id=str(run["id"]),
                    attempt_label=str(run["attempt_label"]),
                    conversation_index=index,
                    source_returned=int(run["source_returned"]),
                    source_opened=int(run["source_opened"]),
                    latency_ms=int(run["latency_ms"]),
                    __weave={
                        "display_name": (
                            f"qualification {run['attempt_label']} source {index}"
                        ),
                        "attributes": {
                            "fugue.seed_digest": qualification_seed_digest(),
                            "fugue.evidence_kind": "source_conversation",
                        },
                    },
                )
                conversations.append(
                    {
                        "run_id": run["id"],
                        "conversation_index": index,
                        "call_id": call.id,
                        "ref": f"weave:///{project}/call/{call.id}",
                    }
                )
    evaluations = asyncio.run(_run_evaluations(project, dataset))
    return {
        "dataset": {
            "name": "mcp-release-maintenance-cases",
            "ref": str(dataset_ref.uri()),
            "rows": len(_EVALUATION_ROWS),
            "content_digest": stable_digest(
                [dict(item) for item in _EVALUATION_ROWS]
            ),
        },
        "source_conversations": conversations,
        "evaluations": evaluations,
    }


def _existing_source_conversations(
    client: Any,
    project: str,
) -> list[dict[str, Any]]:
    seed_digest = qualification_seed_digest()
    expected = {
        (str(run["id"]), index)
        for run in _RUNS
        for index in range(1, 5)
    }
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for call in client.get_calls(limit=500):
        if "/op/fugue.qualification.maintenance_agent:" not in str(
            call.op_name
        ):
            continue
        attributes = call.attributes
        if (
            not isinstance(attributes, Mapping)
            or attributes.get("fugue.seed_digest") != seed_digest
        ):
            continue
        inputs = call.inputs
        if not isinstance(inputs, Mapping):
            continue
        coordinate = (
            str(inputs.get("run_id") or ""),
            int(inputs.get("conversation_index") or 0),
        )
        if coordinate in found:
            raise RuntimeError(
                "hosted project contains duplicate seeded source conversations"
            )
        found[coordinate] = {
            "run_id": coordinate[0],
            "conversation_index": coordinate[1],
            "call_id": call.id,
            "ref": f"weave:///{project}/call/{call.id}",
        }
    if not found:
        return []
    if set(found) != expected:
        raise RuntimeError(
            "hosted project contains a partial seeded conversation cohort"
        )
    return [found[key] for key in sorted(found)]


async def _run_evaluations(
    project: str,
    dataset: weave.Dataset,
) -> list[dict[str, Any]]:
    values = []
    for revision, model in (
        ("maintainer-r17", _baseline_evidence_model),
        ("maintainer-r18", _candidate_evidence_model),
    ):
        evaluation = weave.Evaluation(
            name=f"mcp-release-{revision}",
            evaluation_name=f"MCP release {revision}",
            description=(
                "Aligned deterministic maintenance evidence. This measures the "
                "seeded maintainer workflow, not the Fugue MCP comparison."
            ),
            dataset=dataset,
            scorers=[_evidence_alignment_scorer],
            metadata={
                "fugue_seed_digest": qualification_seed_digest(),
                "revision": revision,
            },
        )
        evaluation_ref = weave.publish(
            evaluation,
            name=f"mcp-release-{revision}",
            aliases=["qualification-v1"],
        )
        summary, call = await evaluation.evaluate.call(
            evaluation,
            model,
            __weave={
                "display_name": f"MCP release {revision} qualification",
                "attributes": {
                    "fugue.seed_digest": qualification_seed_digest(),
                    "fugue.evaluation_revision": revision,
                },
            },
        )
        values.append(
            {
                "revision": revision,
                "ref": str(evaluation_ref.uri()),
                "call_id": call.id,
                "call_ref": f"weave:///{project}/call/{call.id}",
                "summary_digest": stable_digest(summary),
                "prediction_rows": len(_EVALUATION_ROWS),
            }
        )
    return values


@weave.op(name="fugue.qualification.wandb_mcp_tool")
def _seed_tool_span(
    *,
    tool_name: str,
    run_id: str,
    public_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "run_id": run_id,
        "result": dict(public_result),
    }


@weave.op(name="fugue.qualification.maintenance_agent")
def _seed_agent_conversation(
    *,
    run_id: str,
    attempt_label: str,
    conversation_index: int,
    source_returned: int,
    source_opened: int,
    latency_ms: int,
) -> dict[str, Any]:
    question = (
        f"Inspect public maintenance evidence for {attempt_label}; source "
        f"segment {conversation_index}."
    )
    with weave.Conversation(
        conversation_id=f"{run_id}-source-{conversation_index}",
        conversation_name="Fugue MCP release qualification evidence",
        agent_name="maintenance-evidence-agent",
        agent_id="maintenance-evidence-agent-v1",
        agent_version="1",
        model="deterministic-reviewed-policy-v1",
        include_content=True,
        continue_parent_trace=True,
        attributes={
            "fugue.seed_digest": qualification_seed_digest(),
            "fugue.run_id": run_id,
        },
    ) as conversation:
        with conversation.start_turn(
            user_message=question,
            system_instructions=[
                "Inspect evidence, separate observations from causes, and state "
                "when measurements are incomplete."
            ],
        ) as turn:
            with turn.tool(
                name="wandb_projected_run_read",
                arguments=json.dumps(
                    {"run_id": run_id, "fields": ["latency_ms", "source_opened"]}
                ),
                tool_call_id=f"{run_id}-projected-{conversation_index}",
            ) as tool:
                projected = _seed_tool_span(
                    tool_name="wandb_projected_run_read",
                    run_id=run_id,
                    public_result={
                        "latency_ms": latency_ms,
                        "source_returned": source_returned,
                        "source_opened": source_opened,
                    },
                )
                tool.result = projected
            with turn.tool(
                name="weave_call_evidence_read",
                arguments=json.dumps(
                    {
                        "run_id": run_id,
                        "conversation_index": conversation_index,
                    }
                ),
                tool_call_id=f"{run_id}-weave-{conversation_index}",
            ) as tool:
                traced = _seed_tool_span(
                    tool_name="weave_call_evidence_read",
                    run_id=run_id,
                    public_result={
                        "evidence_status": "observed",
                        "conversation_index": conversation_index,
                    },
                )
                tool.result = traced
            turn.record(
                messages=[
                    *turn.messages,
                    {
                        "role": "assistant",
                        "content": (
                            f"Observed latency {latency_ms} ms; opened "
                            f"{source_opened} of {source_returned} returned sources."
                        ),
                    },
                ]
            )
    return {
        "run_id": run_id,
        "attempt_label": attempt_label,
        "conversation_index": conversation_index,
        "evidence_status": "observed",
    }


@weave.op(name="fugue.qualification.baseline_evidence_model")
def _baseline_evidence_model(
    *,
    case_id: str,
    question: str,
    baseline_pass: bool,
    candidate_pass: bool,
) -> dict[str, Any]:
    del question, candidate_pass
    return {
        "case_id": case_id,
        "revision": "maintainer-r17",
        "pass": baseline_pass,
    }


@weave.op(name="fugue.qualification.candidate_evidence_model")
def _candidate_evidence_model(
    *,
    case_id: str,
    question: str,
    baseline_pass: bool,
    candidate_pass: bool,
) -> dict[str, Any]:
    del question, baseline_pass
    return {
        "case_id": case_id,
        "revision": "maintainer-r18",
        "pass": candidate_pass,
    }


@weave.op(name="fugue.qualification.evidence_alignment_scorer")
def _evidence_alignment_scorer(
    *,
    output: Mapping[str, Any],
    baseline_pass: bool,
    candidate_pass: bool,
) -> dict[str, Any]:
    expected = (
        baseline_pass
        if output.get("revision") == "maintainer-r17"
        else candidate_pass
    )
    return {
        "deterministic_pass": bool(output.get("pass")),
        "aligned_with_reviewed_row": output.get("pass") is expected,
    }


def _evidence_lock(
    project: str,
    runs: Sequence[Mapping[str, Any]],
    weave_receipts: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": EVIDENCE_LOCK_SCHEMA_VERSION,
        "project": project,
        "seed_digest": qualification_seed_digest(),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "counts": dict(_REQUIRED_COUNTS),
        "objects": {
            "runs": [dict(item) for item in runs],
            "dataset": dict(weave_receipts["dataset"]),
            "source_conversations": [
                dict(item) for item in weave_receipts["source_conversations"]
            ],
            "evaluations": [
                dict(item) for item in weave_receipts["evaluations"]
            ],
        },
        "facts_digest": stable_digest(qualification_seed()["facts"]),
        "supported_scope": (
            "Non-sensitive hosted qualification evidence only; not customer "
            "data and not a release-wide product claim."
        ),
        "evidence_lock_digest": "",
    }
    return {
        **unsigned,
        "evidence_lock_digest": stable_digest(unsigned),
    }


@contextmanager
def _temporary_environment(values: Mapping[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, prior in previous.items():
            if prior is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior
