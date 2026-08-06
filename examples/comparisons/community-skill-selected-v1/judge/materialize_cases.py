from __future__ import annotations

import argparse
import json
from pathlib import Path

from fugue.bench.files import atomic_write_json
from fugue.bench.judge_calibration_run import materialize_cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and materialize locked calibration cases privately."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(".fugue/private/community-skill-selected-v1/cases.jsonl"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(".fugue/private/community-skill-selected-v1/cases.receipt.json"),
    )
    args = parser.parse_args()
    result = materialize_cases(
        repo_root=args.repo_root,
        manifest_path=args.manifest,
        source_path=args.source,
        destination=args.destination,
    )
    receipt = args.receipt if args.receipt.is_absolute() else args.repo_root / args.receipt
    private_root = (args.repo_root / ".fugue/private").resolve()
    if receipt.is_symlink() or not receipt.resolve().is_relative_to(private_root):
        raise SystemExit("materialization receipt must remain under .fugue/private")
    atomic_write_json(receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
