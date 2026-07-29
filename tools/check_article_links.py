#!/usr/bin/env python3
"""Verify structured article references without publishing review-only notes."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _probe(url: str) -> tuple[str, int]:
    request = Request(
        url,
        headers={
            "User-Agent": "Fugue-Article-Link-Check/1.0",
            "Range": "bytes=0-2047",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=20) as response:
                return url, int(response.status)
        except HTTPError as error:
            if error.code in {401, 403}:
                return url, error.code
            if error.code == 429 and attempt < 2:
                time.sleep(2**attempt)
                continue
            return url, error.code
        except URLError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def check(root: Path) -> None:
    urls: set[str] = set()
    for path in root.glob("*/sources.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        urls.update(source["url"] for source in document.get("sources", []))
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        requests = {pool.submit(_probe, url): url for url in sorted(urls)}
        for future in as_completed(requests):
            url = requests[future]
            try:
                _, status = future.result()
            except Exception as error:  # network failure is a failed qualification
                failures.append(f"{url}: {error}")
                continue
            if status >= 400 and status not in {401, 403}:
                failures.append(f"{url}: HTTP {status}")
    if failures:
        raise ValueError("broken article references:\n" + "\n".join(sorted(failures)))
    print(f"article references verified: {len(urls)} unique URLs")


def main(argv: list[str]) -> int:
    root = (
        Path(argv[1])
        if len(argv) > 1
        else Path("docs/articles/fugue-agentic-software-factory")
    )
    check(root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
