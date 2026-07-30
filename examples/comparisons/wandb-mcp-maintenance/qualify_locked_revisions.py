from __future__ import annotations

import argparse
import json
from pathlib import Path

from fugue.bench.mcp_release_qualification import qualify_locked_mcp_revisions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded real MCP calls from both exact revision locks against "
            "the immutable hosted qualification evidence."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-lock", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-project")
    parser.add_argument("--result-project")
    args = parser.parse_args()
    receipt = qualify_locked_mcp_revisions(
        repo_root=args.repo_root,
        evidence_lock=args.evidence_lock,
        env_file=args.env_file,
        output=args.output,
        source_project=args.source_project,
        result_project=args.result_project,
    )
    print(
        json.dumps(
            {
                "source_project": receipt["source_project"],
                "result_project": receipt["result_project"],
                "findings": receipt["findings"],
                "recommendation": receipt["recommendation"],
                "receipt_digest": receipt["receipt_digest"],
                "output": args.output.resolve().as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
