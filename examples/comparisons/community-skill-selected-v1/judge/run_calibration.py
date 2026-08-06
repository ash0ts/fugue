from __future__ import annotations

import argparse
import json
from pathlib import Path

from fugue.bench.files import atomic_write_json
from fugue.bench.judge_calibration import (
    build_blinded_packet,
    load_case_set,
    validate_rubric,
)
from fugue.bench.judge_calibration_run import (
    build_generation_preview,
    default_requester,
    run_generation,
)
from fugue.bench.operator import load_env


def _calibration_env(path: Path) -> dict[str, str]:
    """Load only the credentials allowed by this Anthropic-only campaign."""

    env = load_env(path, override=True)
    env.pop("OPENAI_API_KEY", None)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview or run the governed 48-request calibration generation."
    )
    parser.add_argument("action", choices=("preview", "run"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--preview-out", type=Path, required=True)
    parser.add_argument("--approval")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(".fugue/private/community-skill-selected-v1/model-outputs.jsonl"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(".fugue/private/community-skill-selected-v1/model-output-receipt.json"),
    )
    args = parser.parse_args()

    cases = load_case_set(args.manifest, args.cases)
    rubric, rubric_digest = validate_rubric(args.rubric)
    packet = build_blinded_packet(cases, rubric_digest)
    if args.packet.exists() and json.loads(args.packet.read_text()) != packet:
        raise SystemExit("existing packet disagrees with the locked cases or rubric")
    atomic_write_json(args.packet, packet)
    preview = build_generation_preview(
        case_set=cases, rubric=rubric, rubric_digest=rubric_digest, packet=packet
    )
    atomic_write_json(args.preview_out, preview)
    if args.action == "preview":
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    if not args.approval:
        raise SystemExit("run requires --approval for this exact preview digest")
    env = _calibration_env(args.env_file)
    result = run_generation(
        repo_root=args.repo_root,
        case_set=cases,
        rubric=rubric,
        rubric_digest=rubric_digest,
        packet=packet,
        approval_digest=args.approval,
        output_path=args.outputs,
        receipt_path=args.receipt,
        requester=default_requester(model=str(preview["profile"]), env=env),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
