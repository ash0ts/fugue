from __future__ import annotations

import argparse
import json
from pathlib import Path

from fugue.bench.mcp_release_qualification import (
    QUALIFICATION_RESULT_PROJECT,
    QUALIFICATION_SOURCE_PROJECT,
    verify_hosted_source_conformance,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read only the immutable hosted MCP qualification source and emit "
            "a zero-model drift/conformance receipt."
        )
    )
    parser.add_argument("--evidence-lock", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-project",
        default=QUALIFICATION_SOURCE_PROJECT,
    )
    parser.add_argument(
        "--result-project",
        default=QUALIFICATION_RESULT_PROJECT,
    )
    args = parser.parse_args()
    receipt = verify_hosted_source_conformance(
        evidence_lock=args.evidence_lock,
        env_file=args.env_file,
        output=args.output,
        source_project=args.source_project,
        result_project=args.result_project,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "source_project": receipt["source_project"],
                "result_project": receipt["result_project"],
                "observed": receipt["observed"],
                "blockers": receipt["blockers"],
                "receipt_digest": receipt["receipt_digest"],
                "output": args.output.resolve().as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
