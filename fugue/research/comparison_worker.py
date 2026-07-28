from __future__ import annotations

import argparse
from pathlib import Path

from fugue.research.comparisons import run_comparison_worker


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one approved comparison")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    return run_comparison_worker(args.input)


if __name__ == "__main__":
    raise SystemExit(main())
