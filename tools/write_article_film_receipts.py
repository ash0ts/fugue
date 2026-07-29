#!/usr/bin/env python3
"""Write deterministic hashes and video metadata for every article film."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "docs" / "articles" / "fugue-agentic-software-factory"


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def main() -> int:
    manifest = json.loads((SERIES / "series.json").read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        film = SERIES / entry["slug"] / "media" / "film"
        spec = json.loads((film / "film-spec.json").read_text(encoding="utf-8"))
        video = film / f"{entry['slug']}.mp4"
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,profile,width,height,pix_fmt,color_range,r_frame_rate",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        checkpoints = {
            checkpoint["name"]: _digest(
                film / "checkpoints" / f"{checkpoint['name']}.png"
            )
            for checkpoint in spec["checkpoints"]
        }
        receipt = {
            "schema_version": 1,
            "slug": entry["slug"],
            "html_digest": _digest(film / f"{entry['slug']}.html"),
            "mp4_digest": _digest(video),
            "poster_digest": _digest(film / f"{entry['slug']}-poster.png"),
            "contact_sheet_digest": _digest(film / "contact-sheet.png"),
            "checkpoint_frame_digests": checkpoints,
            "video": json.loads(probe.stdout)["streams"][0],
        }
        (film / "render-receipt.json").write_text(
            f"{json.dumps(receipt, indent=2)}\n", encoding="utf-8"
        )
        print(f"wrote film receipt: {entry['slug']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
