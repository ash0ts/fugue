from __future__ import annotations

import argparse
import json
from pathlib import Path

from fugue.bench.mcp_release_qualification import (
    QUALIFICATION_RESULT_PROJECT,
    QUALIFICATION_SOURCE_PROJECT,
    evidence_result_project,
    evidence_source_project,
    prepare_hosted_project,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently prepare the non-sensitive hosted W&B/Weave "
            "qualification evidence and write its immutable lock."
        )
    )
    parser.add_argument(
        "--source-project",
        help=(
            "Immutable hosted evidence project (default: "
            f"{QUALIFICATION_SOURCE_PROJECT})."
        ),
    )
    parser.add_argument(
        "--result-project",
        help=(
            "Later experiment result destination (default: "
            f"{QUALIFICATION_RESULT_PROJECT})."
        ),
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = prepare_hosted_project(
        source_project=args.source_project,
        result_project=args.result_project,
        output=args.output,
        env_file=args.env_file,
    )
    print(
        json.dumps(
            {
                "source_project": evidence_source_project(lock),
                "result_project": evidence_result_project(lock),
                "counts": lock["counts"],
                "evidence_lock_digest": lock["evidence_lock_digest"],
                "output": args.output.resolve().as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
