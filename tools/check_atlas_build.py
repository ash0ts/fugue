#!/usr/bin/env python3
"""Fail when the built atlas crosses its static or accessibility boundary."""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

PAGES = {"index.html", "experiment.html", "compare.html", "methods.html"}
NETWORK_APIS = ("fetch(", "XMLHttpRequest", "WebSocket(", "EventSource(", "sendBeacon(")
UNSAFE_DOM_APIS = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write")


class _Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[tuple[str, str]] = []
        self.landmarks: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.landmarks.add(tag)
        for field in ("href", "src"):
            if values.get(field):
                self.urls.append((field, str(values[field])))


def check_build(root: Path) -> None:
    missing = PAGES - {path.name for path in root.glob("*.html")}
    if missing:
        raise ValueError(f"atlas build is missing pages: {sorted(missing)}")
    for path in sorted(root.glob("*.html")):
        parser = _Document()
        body = path.read_text(encoding="utf-8")
        parser.feed(body)
        if not {"header", "nav", "main", "footer"} <= parser.landmarks:
            raise ValueError(f"{path.name} is missing semantic landmarks")
        for field, url in parser.urls:
            if url.startswith("/") and not url.startswith("/fugue/"):
                raise ValueError(f"{path.name} has a base-unsafe {field}: {url}")
            if field == "src" and url.startswith(("http://", "https://")):
                raise ValueError(f"{path.name} loads an external resource: {url}")
    scripts = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.rglob("*.js"))
    )
    for token in (*NETWORK_APIS, *UNSAFE_DOM_APIS):
        if token in scripts:
            raise ValueError(f"atlas bundle contains forbidden browser API: {token}")
    css = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.rglob("*.css"))
    )
    for label, pattern in (
        ("visible focus", r":focus-visible"),
        ("reduced motion", r"prefers-reduced-motion"),
        ("responsive layout", r"@media\s*\((?:max-width:|width<=)"),
    ):
        if not re.search(pattern, css):
            raise ValueError(f"atlas CSS is missing {label}")
    if re.search(r"url\([\"']?https?://", css):
        raise ValueError("atlas CSS loads an external asset")


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("atlas/dist")
    check_build(root)
    print(f"atlas build verified: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
