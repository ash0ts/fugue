from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from fugue.bench.comparison import load_comparison
from fugue.bench.loop_failure import (
    build_comparison_failure_lock,
    write_comparison_failure_lock,
)

COMPARISON_ID = "mcp-main-vs-0-4-tool-surface-confirmation-v10"
COMPARISON_PATH = Path(
    "examples/comparisons/wandb-mcp-maintenance/"
    "tool-surface-confirmation-local-v10.yaml"
)
SOURCE_PROJECT = "wandb/fugue-mcp-release-source-v2"
RESULT_PROJECT = "wandb/fugue-mcp-release-qualification-v1"
HARNESS = "claude-code"
TASKS = 4
ATTEMPTS = 2
BASELINE_SOURCE = "wandb-mcp-main"
CANDIDATE_SOURCE = "wandb-mcp-0-4-current"
CANDIDATE_SHA = "5c6cc1c9a1079296daf6613ea6d12daebdd8bcba"
RESULT_DIGEST = "e062f5b392a36d9ebd97adc3ab58b6e253cdd9dd943381342d51d76303bbcf38"
FAILURE_TASK_ID = "exact-history-target"
FAILURE_ARM = "baseline"


def _locked_spec(repo_root: Path) -> tuple[str, tuple[str, ...]]:
    spec = load_comparison(repo_root / COMPARISON_PATH, repo_root=repo_root)
    if (
        spec.schema_version != 3
        or spec.id != COMPARISON_ID
        or spec.execution.source_evidence_project != SOURCE_PROJECT
        or spec.execution.evidence_project != RESULT_PROJECT
        or spec.execution.harnesses != (HARNESS,)
        or spec.execution.attempts != ATTEMPTS
        or spec.execution.concurrency != 1
        or spec.execution.environment.get("type") != "docker"
        or spec.baseline.integrations != ({"id": BASELINE_SOURCE},)
        or spec.candidate.integrations != ({"id": CANDIDATE_SOURCE},)
        or spec.decision_policy is None
        or spec.decision_policy.candidate_sha != CANDIDATE_SHA
    ):
        raise ValueError(
            "checked-in MCP confirmation no longer matches the reviewed loop source"
        )
    tasks_path = (repo_root / spec.taskset.tasks).resolve()
    task_count = sum(
        bool(line.strip())
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
    )
    if task_count != TASKS:
        raise ValueError(
            "checked-in MCP confirmation no longer contains exactly four tasks"
        )
    return spec.spec_digest, (CANDIDATE_SOURCE,)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Lock one reviewed repeated failure from the source-isolated "
            "MCP maintainer canary."
        )
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--preview",
        type=Path,
        required=True,
        help="Exact prepared preview used to produce the result.",
    )
    parser.add_argument(
        "--task-id",
        choices=(FAILURE_TASK_ID,),
        required=True,
        help="The reviewed repeated V10 failure task.",
    )
    parser.add_argument(
        "--arm",
        choices=(FAILURE_ARM,),
        required=True,
        help="The exact V10 arm that reproduced the failure twice.",
    )
    parser.add_argument("--primary-attempt-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--reviewed",
        action="store_true",
        help="Confirm a human reviewed both repeated attempts and safe links.",
    )
    args = parser.parse_args()
    if not args.reviewed:
        raise ValueError("--reviewed is required before a failure can be locked")
    repo_root = args.repo_root.resolve()
    spec_digest, required_sources = _locked_spec(repo_root)
    lock = build_comparison_failure_lock(
        result_path=args.result.resolve(),
        preview_path=args.preview.resolve(),
        task_id=args.task_id,
        arm=args.arm,
        primary_attempt_id=args.primary_attempt_id,
        expected_comparison_id=COMPARISON_ID,
        expected_source_project=SOURCE_PROJECT,
        expected_result_project=RESULT_PROJECT,
        expected_harness=HARNESS,
        expected_tasks=TASKS,
        expected_attempts=ATTEMPTS,
        spec_digest=spec_digest,
        locked_at=datetime.now(UTC).isoformat(),
        required_source_ids=required_sources,
    )
    if lock["source"]["result_digest"] != RESULT_DIGEST:
        raise ValueError(
            "comparison result is not the reviewed authoritative V10 result"
        )
    output = write_comparison_failure_lock(args.output, lock)
    candidate_sources = {
        str(item["id"]): str(item["version_identity"])
        for item in lock["locks"]["candidate_sources"]
    }
    print(
        json.dumps(
            {
                "comparison_id": COMPARISON_ID,
                "failure_task_id": lock["failure"]["task_id"],
                "failure_arm": lock["failure"]["arm"],
                "repeated_attempt_ids": lock["failure"][
                    "repeated_attempt_ids"
                ],
                "failed_critical_dimensions": lock["failure"][
                    "failed_critical_dimensions"
                ],
                "baseline_candidate_identity": lock["locks"]["arm_candidates"][
                    "baseline"
                ],
                "candidate_source": candidate_sources.get(CANDIDATE_SOURCE),
                "lock_sha256": lock["lock_sha256"],
                "output": output.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
