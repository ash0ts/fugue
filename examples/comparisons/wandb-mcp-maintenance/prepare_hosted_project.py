from __future__ import annotations

import argparse
import json
from pathlib import Path

from fugue.bench.mcp_release_qualification import (
    QUALIFICATION_PROJECT,
    prepare_hosted_project,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently prepare the non-sensitive hosted W&B/Weave "
            "qualification evidence and write its immutable lock."
        )
    )
    parser.add_argument("--project", default=QUALIFICATION_PROJECT)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = prepare_hosted_project(
        project=args.project,
        output=args.output,
        env_file=args.env_file,
    )
    print(
        json.dumps(
            {
                "project": lock["project"],
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
