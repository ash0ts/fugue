from __future__ import annotations

import argparse
import json
from pathlib import Path

from fugue.bench.mcp_release_qualification import (
    DEFAULT_MCP_RELEASE_CANDIDATES,
    MCP_RELEASE_NOTES_LOCK,
    qualify_locked_mcp_revisions,
)


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
    parser.add_argument(
        "--release-notes-lock",
        type=Path,
        default=MCP_RELEASE_NOTES_LOCK,
    )
    parser.add_argument(
        "--baseline-import-id",
        default=DEFAULT_MCP_RELEASE_CANDIDATES[0][0],
    )
    parser.add_argument(
        "--baseline-version",
        default=DEFAULT_MCP_RELEASE_CANDIDATES[0][1],
    )
    parser.add_argument(
        "--candidate-import-id",
        default=DEFAULT_MCP_RELEASE_CANDIDATES[1][0],
    )
    parser.add_argument(
        "--candidate-version",
        default=DEFAULT_MCP_RELEASE_CANDIDATES[1][1],
    )
    args = parser.parse_args()
    candidates = (
        (args.baseline_import_id, args.baseline_version),
        (args.candidate_import_id, args.candidate_version),
    )
    receipt = qualify_locked_mcp_revisions(
        repo_root=args.repo_root,
        evidence_lock=args.evidence_lock,
        env_file=args.env_file,
        output=args.output,
        source_project=args.source_project,
        result_project=args.result_project,
        candidates=candidates,
        release_notes_lock=args.release_notes_lock,
    )
    print(
        json.dumps(
            {
                "source_project": receipt["source_project"],
                "result_project": receipt["result_project"],
                "candidate_bindings": receipt["candidate_bindings"],
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
