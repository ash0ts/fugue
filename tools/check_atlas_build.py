#!/usr/bin/env python3
"""Fail when the built atlas crosses its static or accessibility boundary."""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

PAGES = {
    "index.html",
    "experiments.html",
    "experiment.html",
    "compare.html",
    "methods.html",
}
NETWORK_APIS = ("fetch(", "XMLHttpRequest", "WebSocket(", "EventSource(", "sendBeacon(")
UNSAFE_DOM_APIS = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write")


class _Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[tuple[str, str]] = []
        self.landmarks: set[str] = set()
        self.canonical_urls: list[str] = []
        self.ids: set[str] = set()
        self.nav_links: list[tuple[str, str]] = []
        self._in_primary_nav = False
        self._nav_href: str | None = None
        self._nav_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.landmarks.add(tag)
        if tag == "nav" and values.get("aria-label") == "Primary navigation":
            self._in_primary_nav = True
        elif tag == "a" and self._in_primary_nav:
            self._nav_href = str(values.get("href") or "")
            self._nav_text = []
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "link" and values.get("rel") == "canonical" and values.get("href"):
            self.canonical_urls.append(str(values["href"]))
        for field in ("href", "src"):
            if values.get(field):
                self.urls.append((field, str(values[field])))

    def handle_data(self, data: str) -> None:
        if self._nav_href is not None:
            self._nav_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._nav_href is not None:
            self.nav_links.append(
                (" ".join("".join(self._nav_text).split()), self._nav_href)
            )
            self._nav_href = None
            self._nav_text = []
        elif tag == "nav" and self._in_primary_nav:
            self._in_primary_nav = False


def _internal_target(root: Path, page: Path, url: str) -> Path | None:
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or url.startswith(("#", "mailto:")):
        return None
    clean = unquote(parsed.path)
    if not clean:
        return page
    if clean.startswith("/fugue/"):
        clean = clean.removeprefix("/fugue/")
        target = root / clean
    elif clean == "/fugue":
        target = root
    elif clean.startswith("/"):
        return None
    else:
        target = page.parent / clean
    if clean.endswith("/") or target.is_dir():
        target = target / "index.html"
    return target.resolve()


def check_build(root: Path) -> None:
    missing = PAGES - {path.name for path in root.glob("*.html")}
    if missing:
        raise ValueError(f"atlas build is missing pages: {sorted(missing)}")
    html_paths = sorted(root.rglob("*.html"))
    if not html_paths:
        raise ValueError("atlas build contains no HTML")
    for path in html_paths:
        parser = _Document()
        body = path.read_text(encoding="utf-8")
        parser.feed(body)
        is_film = "media/film" in path.as_posix()
        required = {"main"} if is_film else {"header", "nav", "main", "footer"}
        if not required <= parser.landmarks:
            raise ValueError(
                f"{path.relative_to(root)} is missing semantic landmarks: "
                f"{sorted(required - parser.landmarks)}"
            )
        if path.relative_to(root).parts[:1] == ("articles",) and not is_film:
            if len(parser.canonical_urls) != 1:
                raise ValueError(
                    f"{path.relative_to(root)} must declare exactly one canonical URL"
                )
        if not is_film:
            nav_labels = [label for label, _ in parser.nav_links]
            if nav_labels != ["Product", "Studies", "Articles", "GitHub"]:
                raise ValueError(
                    f"{path.relative_to(root)} has inconsistent primary navigation: "
                    f"{nav_labels}"
                )
            if any(
                url.endswith(("compare.html", "methods.html"))
                for _, url in parser.nav_links
            ):
                raise ValueError(
                    f"{path.relative_to(root)} exposes a deprecated primary route"
                )
        for field, url in parser.urls:
            if url.startswith("/") and not url.startswith("/fugue/"):
                raise ValueError(
                    f"{path.relative_to(root)} has a base-unsafe {field}: {url}"
                )
            if field == "src" and url.startswith(("http://", "https://")):
                raise ValueError(
                    f"{path.relative_to(root)} loads an external resource: {url}"
                )
            target = _internal_target(root, path, url)
            if target is not None and not target.exists():
                raise ValueError(
                    f"{path.relative_to(root)} has a missing internal {field}: {url}"
                )
            if url.startswith("#") and url[1:] not in parser.ids:
                raise ValueError(
                    f"{path.relative_to(root)} has a missing fragment target: {url}"
                )
    scripts = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.rglob("*.js"))
    )
    scripts += "\n" + "\n".join(path.read_text(encoding="utf-8") for path in html_paths)
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
    _check_product_contract(root)


def _check_product_contract(root: Path) -> None:
    import json

    path = root / "data" / "product.json"
    if not path.is_file():
        raise ValueError("atlas build is missing data/product.json")
    product = json.loads(path.read_text(encoding="utf-8"))
    if product.get("maturity") != "open_source_preview":
        raise ValueError("product maturity must remain open_source_preview")
    quickstart = product.get("quickstart", {})
    revision = quickstart.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("preview quickstart must pin a full commit SHA")
    if quickstart.get("availability") == "released":
        raise ValueError("a preview quickstart cannot be labeled released")
    if revision not in quickstart.get("source_url", ""):
        raise ValueError("preview source URL must point at the pinned revision")
    if not any(revision in command for command in quickstart.get("commands", [])):
        raise ValueError("preview commands must check out the pinned revision")
    references = product.get("references", [])
    if len(references) < 6:
        raise ValueError("product contract has too few research references")
    for reference in references:
        if not all(reference.get(field) for field in ("author", "title", "url", "relevance")):
            raise ValueError("product reference is missing required metadata")
        if not reference["url"].startswith("https://"):
            raise ValueError("product reference URL must use HTTPS")


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("atlas/dist")
    check_build(root)
    print(f"atlas build verified: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
