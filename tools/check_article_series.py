#!/usr/bin/env python3
"""Validate the standalone article, publication, and film contracts."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

REQUIRED_ENTRY_FIELDS = {
    "id",
    "part",
    "title",
    "slug",
    "context",
    "publication_state",
    "evidence_state",
    "published_at",
    "updated_at",
    "accepted_digest",
    "accepted_at",
    "preview_digest",
    "previous_id",
    "next_id",
    "source_digest",
    "animation",
    "results_appendices",
}
REQUIRED_ARTICLE_HEADINGS = {
    "Scope and terms",
    "Try this in 15 minutes",
    "What this does not show",
    "References",
}
REQUIRED_FILM_FILES = {
    "contact-sheet.png",
    "transcript.md",
    "claim-ledger.json",
    "film-spec.json",
    "render-receipt.json",
}
ALLOWED_PUBLICATION_STATES = {"planned", "working_draft", "published"}
ALLOWED_EVIDENCE_STATES = {
    "concept",
    "draft_preregistration",
    "preregistered",
    "results_appended",
}
ALLOWED_CLAIM_KINDS = {
    "cited background",
    "audited Fugue observation",
    "illustrative example",
    "preregistered design",
    "pending result",
}
REQUIRED_ANIMATION_FIELDS = {
    "html",
    "mp4",
    "poster",
    "transcript",
    "after_heading",
    "bridge_to_heading",
    "poster_scene",
    "reduced_motion_scene",
    "duration_seconds",
    "what_to_watch",
    "chapters",
}
RELATIVE_MARKDOWN_LINK = re.compile(r"\]\((?!https?://|mailto:|#|/)([^)]+)\)")
CITATION = re.compile(r"\[@([a-z0-9][a-z0-9-]*)\]")
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:WANDB|ANTHROPIC|OPENAI)_API_KEY\s*=\s*(?![$<{])[^\s]+"),
)


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(str(href))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(source: str | bytes) -> str:
    payload = source.encode() if isinstance(source, str) else source
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _code_fences(source: str) -> str:
    return "\n".join(re.findall(r"```[^\n]*\n(.*?)```", source, flags=re.DOTALL))


def _heading_ids(article: str) -> set[str]:
    ids: set[str] = set()
    seen: dict[str, int] = {}
    for label in re.findall(r"^#{1,6} (.+)$", article, flags=re.MULTILINE):
        normalized = unicodedata.normalize("NFKD", label).lower()
        base = re.sub(r"[^\w\s-]", "", normalized, flags=re.ASCII).strip()
        base = re.sub(r"[\s_]+", "-", base)
        base = re.sub(r"-+", "-", base) or "section"
        count = seen.get(base, 0)
        seen[base] = count + 1
        ids.add(base if count == 0 else f"{base}-{count + 1}")
    return ids


def _heading_positions(article: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    seen: dict[str, int] = {}
    for index, label in enumerate(
        re.findall(r"^#{1,6} (.+)$", article, flags=re.MULTILINE)
    ):
        normalized = unicodedata.normalize("NFKD", label).lower()
        base = re.sub(r"[^\w\s-]", "", normalized, flags=re.ASCII).strip()
        base = re.sub(r"[\s_]+", "-", base)
        base = re.sub(r"-+", "-", base) or "section"
        count = seen.get(base, 0)
        seen[base] = count + 1
        positions[base if count == 0 else f"{base}-{count + 1}"] = index
    return positions


def _fail(message: str) -> None:
    raise ValueError(message)


def _check_article_source(entry: dict[str, Any], article_path: Path) -> None:
    article = article_path.read_text(encoding="utf-8")
    if _sha256(article) != entry["source_digest"]:
        _fail(f"{entry['id']} source_digest does not match article.md")
    if "standalone" not in article[:1_200].lower():
        _fail(f"{entry['id']} does not declare its standalone audience")
    headings = {
        match.group(1).strip()
        for match in re.finditer(r"^## (.+)$", article, flags=re.MULTILINE)
    }
    missing_headings = REQUIRED_ARTICLE_HEADINGS - headings
    if missing_headings:
        _fail(f"{entry['id']} is missing headings: {sorted(missing_headings)}")
    if not any(heading.startswith("When ") for heading in headings):
        _fail(f"{entry['id']} is missing its insufficiency boundary")
    relative_links = RELATIVE_MARKDOWN_LINK.findall(article)
    if relative_links:
        _fail(f"{entry['id']} has checkout-relative links: {relative_links}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(article):
            _fail(f"{entry['id']} appears to contain a credential value")
    runtime = _code_fences(article).lower()
    if "coreweave" in runtime or "openai_api_key" in runtime:
        _fail(f"{entry['id']} contains a forbidden runtime dependency")
    if (
        entry["evidence_state"] in {"draft_preregistration", "preregistered"}
        and "no result" not in article[:1_500].lower()
    ):
        _fail(f"{entry['id']} must visibly declare that it has no result")
    sources_path = article_path.parent / "sources.json"
    review_path = article_path.parent / "editorial-review.md"
    if not sources_path.exists() or not review_path.exists():
        _fail(f"{entry['id']} is missing sources.json or editorial-review.md")
    sources_document = _read_json(sources_path)
    if sources_document.get("schema_version") != 1:
        _fail(f"{entry['id']} has an unsupported sources schema")
    sources = sources_document.get("sources", [])
    if len(sources) < 4:
        _fail(f"{entry['id']} bibliography is too thin")
    ids = [source.get("id") for source in sources]
    if len(set(ids)) != len(ids) or None in ids:
        _fail(f"{entry['id']} has duplicate or missing source IDs")
    for source in sources:
        missing = {
            "id", "author", "title", "publication", "date", "url", "supports"
        } - set(source)
        if missing:
            _fail(f"{entry['id']} source {source.get('id')} is missing {missing}")
        if not str(source["url"]).startswith("https://"):
            _fail(f"{entry['id']} source {source['id']} is not HTTPS")
    cited = set(CITATION.findall(article))
    declared = set(ids)
    if cited != declared:
        _fail(
            f"{entry['id']} structured citations differ: "
            f"missing={sorted(declared - cited)} unknown={sorted(cited - declared)}"
        )
    for paragraph in re.split(r"\n\s*\n", article):
        quantitative = (
            ("percentage point" in paragraph.lower() or "github reports" in paragraph.lower())
            and re.search(r"\d", paragraph)
        )
        if quantitative and not CITATION.search(paragraph):
            _fail(f"{entry['id']} has an uncited quantitative claim")


def _check_film_files(entry: dict[str, Any], film_path: Path) -> None:
    required = set(REQUIRED_FILM_FILES)
    required.update(
        {
            f"{entry['slug']}.html",
            f"{entry['slug']}.mp4",
            f"{entry['slug']}-poster.png",
        }
    )
    missing_film = [name for name in sorted(required) if not (film_path / name).exists()]
    if missing_film:
        _fail(f"{entry['id']} is missing film artifacts: {missing_film}")


def _check_film_spec(entry: dict[str, Any], film_path: Path) -> None:
    spec = _read_json(film_path / "film-spec.json")
    expected = {
        "slug": entry["slug"],
        "width": 1280,
        "height": 720,
        "fps": 15,
        "silent": True,
    }
    for field, value in expected.items():
        if spec.get(field) != value:
            _fail(f"{entry['id']} film spec {field} must equal {value!r}")
    if spec.get("duration") != entry["animation"].get("duration_seconds"):
        _fail(f"{entry['id']} film duration differs from the article manifest")
    if not 84 <= spec.get("duration", 0) <= 110:
        _fail(f"{entry['id']} film duration must be between 84 and 110 seconds")
    checkpoints = spec.get("checkpoints", [])
    if not 7 <= len(checkpoints) <= 8:
        _fail(f"{entry['id']} film must have seven or eight checkpoints")
    if [item["name"] for item in checkpoints] != entry["animation"].get("chapters"):
        _fail(f"{entry['id']} film chapters differ from its checkpoints")
    placement = spec.get("placement", {})
    for field in ("after_heading", "bridge_to_heading"):
        if placement.get(field) != entry["animation"].get(field):
            _fail(f"{entry['id']} film placement {field} differs from its manifest")
    for field, manifest_field in (
        ("posterScene", "poster_scene"),
        ("reducedMotionScene", "reduced_motion_scene"),
    ):
        if spec.get(field) != entry["animation"].get(manifest_field):
            _fail(f"{entry['id']} film {field} differs from its manifest")
    checkpoint_names = {checkpoint["name"] for checkpoint in checkpoints}
    if spec.get("posterScene") not in checkpoint_names:
        _fail(f"{entry['id']} poster scene is not a checkpoint")
    if spec.get("reducedMotionScene") not in checkpoint_names:
        _fail(f"{entry['id']} reduced-motion scene is not a checkpoint")
    if spec.get("safeArea") != {"left": 54, "top": 40, "right": 54, "bottom": 100}:
        _fail(f"{entry['id']} film safe area drifted")
    if spec.get("typeScale") != {"essential": 28, "supporting": 24, "compact": 18}:
        _fail(f"{entry['id']} film type scale drifted")
    if spec.get("transitionSeconds") != 0.35:
        _fail(f"{entry['id']} film crossfade duration drifted")
    for checkpoint in checkpoints:
        name = checkpoint["name"]
        if not (film_path / "checkpoints" / f"{name}.png").exists():
            _fail(f"{entry['id']} is missing checkpoint image: {name}")


def _check_claim_ledger(
    entry: dict[str, Any],
    film_path: Path,
    article_path: Path,
) -> None:
    ledger = _read_json(film_path / "claim-ledger.json")
    if ledger.get("schema_version") != 2:
        _fail(f"{entry['id']} claim ledger must use schema version 2")
    claims = ledger.get("claims", [])
    article = article_path.read_text(encoding="utf-8")
    article_ids = _heading_ids(article)
    positions = _heading_positions(article)
    after_position = positions.get(entry["animation"]["after_heading"])
    if after_position is None:
        _fail(f"{entry['id']} film placement does not resolve")
    if not claims:
        _fail(f"{entry['id']} claim ledger is empty")
    for claim in claims:
        if claim.get("kind") not in ALLOWED_CLAIM_KINDS:
            _fail(f"{entry['id']} has invalid claim kind: {claim.get('kind')}")
        if not claim.get("claim") or not claim.get("source"):
            _fail(f"{entry['id']} has an ungrounded film claim")
        if not str(claim["source"]).startswith("article.md#"):
            _fail(f"{entry['id']} film claim is not anchored to an article section")
        if str(claim["source"]).split("#", 1)[1] not in article_ids:
            _fail(f"{entry['id']} film claim source does not resolve: {claim['source']}")
        source_id = str(claim["source"]).split("#", 1)[1]
        if positions[source_id] > after_position:
            _fail(f"{entry['id']} film cites future section: {claim['source']}")
        if not claim.get("visual_relationship"):
            _fail(f"{entry['id']} film claim lacks a visual relationship")
    if entry["evidence_state"] in {"draft_preregistration", "preregistered"}:
        observed = [
            claim for claim in claims if claim["kind"] == "audited Fugue observation"
        ]
        if observed:
            out_of_scope = str(ledger.get("out_of_scope", "")).lower()
            forbidden_outcome_terms = (
                "agent improved",
                "agent regressed",
                "behavioral winner",
                "package go",
                "release go",
            )
            observed_text = " ".join(
                str(value)
                for claim in observed
                for value in (
                    claim.get("claim", ""),
                    *claim.get("support", []),
                )
            ).lower()
            if (
                entry["publication_state"] != "working_draft"
                or "zero-model" not in article.lower()
                or "not an agent behavioral result" not in out_of_scope
                or any(term in observed_text for term in forbidden_outcome_terms)
                or any(
                    str(claim.get("caveat", "")).lower() != out_of_scope
                    for claim in observed
                )
                or any(
                    "zero-model"
                    not in " ".join(
                        (
                            str(claim.get("claim", "")),
                            *(str(item) for item in claim.get("support", [])),
                        )
                    ).lower()
                    and "no model"
                    not in " ".join(
                        (
                            str(claim.get("claim", "")),
                            *(str(item) for item in claim.get("support", [])),
                        )
                    ).lower()
                    for claim in observed
                )
            ):
                _fail(f"{entry['id']} preregistration film claims an observed result")


def _check_film_runtime(entry: dict[str, Any], film_path: Path) -> None:
    html = (film_path / f"{entry['slug']}.html").read_text(encoding="utf-8")
    for hook in (
        "window.__filmSpec",
        "window.__animationReady",
        "window.setAnimationTime",
        "window.releaseAnimation",
    ):
        if hook not in html:
            _fail(f"{entry['id']} film is missing runtime hook {hook}")
    if (
        entry["evidence_state"] in {"draft_preregistration", "preregistered"}
        and "NO RESULT YET" not in html
        and "NO AGENT RESULT YET" not in html
    ):
        _fail(f"{entry['id']} film is missing its preregistration label")
    if entry["publication_state"] == "working_draft" and (
        "WORKING DRAFT" not in html and "DRAFT DESIGN" not in html
    ):
        _fail(f"{entry['id']} film is missing its draft label")


def _check_film_receipt(entry: dict[str, Any], film_path: Path) -> None:
    receipt = _read_json(film_path / "render-receipt.json")
    if receipt.get("slug") != entry["slug"]:
        _fail(f"{entry['id']} film receipt has the wrong slug")
    frame_hashes = receipt.get("checkpoint_frame_digests", {})
    spec = _read_json(film_path / "film-spec.json")
    expected_names = {item["name"] for item in spec["checkpoints"]}
    if set(frame_hashes) != expected_names:
        _fail(f"{entry['id']} film receipt is missing checkpoint hashes")
    for name, digest in frame_hashes.items():
        actual = _sha256((film_path / "checkpoints" / f"{name}.png").read_bytes())
        if digest != actual:
            _fail(f"{entry['id']} checkpoint hash drifted: {name}")
    video = receipt.get("video", {})
    expected_video = {
        "codec_name": "h264",
        "width": 1280,
        "height": 720,
        "pix_fmt": "yuv420p",
        "color_range": "tv",
        "r_frame_rate": "15/1",
    }
    for field, value in expected_video.items():
        if video.get(field) != value:
            _fail(f"{entry['id']} video receipt {field} must equal {value!r}")


def _check_output_state(
    entry: dict[str, Any],
    output_root: Path,
) -> None:
    output_page = output_root / "articles" / entry["slug"] / "index.html"
    if entry["publication_state"] in {"working_draft", "published"} and not output_page.exists():
        _fail(f"{entry['id']} is visible but absent from the built site")
    if entry["publication_state"] == "planned" and output_page.exists():
        _fail(f"{entry['id']} planned article leaked into the built site")
    if output_page.exists():
        html = output_page.read_text(encoding="utf-8")
        if entry["publication_state"] == "working_draft":
            if 'name="robots" content="noindex, nofollow"' not in html:
                _fail(f"{entry['id']} mutable draft is indexable")
            if "WORKING DRAFT" not in html.upper() and "DRAFT PREREGISTRATION" not in html.upper():
                _fail(f"{entry['id']} mutable draft lacks a visible banner")
        if entry["publication_state"] == "published" and 'name="robots"' in html:
            _fail(f"{entry['id']} published article is noindex")
        for contract in (
            'class="article-toc"',
            'class="mobile-toc"',
            'class="reading-progress"',
            'href="article.md"',
            'class="film-player-controls"',
            'class="film-dialog"',
            'data-film-open=',
        ):
            if contract not in html:
                _fail(f"{entry['id']} built page is missing {contract}")
        film = entry["animation"]
        placement_marker = (
            f'data-after-heading="{film["after_heading"]}" '
            f'data-bridge-heading="{film["bridge_to_heading"]}"'
        )
        after_heading = f'<h2 id="{film["after_heading"]}">'
        bridge_heading = f'<h2 id="{film["bridge_to_heading"]}">'
        if placement_marker not in html:
            _fail(f"{entry['id']} built page lacks contextual film placement metadata")
        if not (
            html.find(after_heading)
            < html.find(placement_marker)
            < html.find(bridge_heading)
        ):
            _fail(f"{entry['id']} film does not follow its declared article section")
        if html.count("data-film-seek=") != len(film["chapters"]):
            _fail(f"{entry['id']} built page lacks one control per film chapter")
        if re.search(r"<video[^>]+\scontrols(?:\s|>)", html):
            _fail(f"{entry['id']} film uses overlaying native video controls")
        if "```mermaid" in html:
            _fail(f"{entry['id']} leaked an unrendered Mermaid fence")
        if "<img src=\"media/figures/" not in html:
            _fail(f"{entry['id']} has no rendered local SVG figures")


def _check_entry(
    source_root: Path,
    output_root: Path,
    entry: dict[str, Any],
) -> None:
    missing = REQUIRED_ENTRY_FIELDS - set(entry)
    if missing:
        _fail(f"{entry.get('id', '<unknown>')} is missing fields: {sorted(missing)}")
    if entry["publication_state"] not in ALLOWED_PUBLICATION_STATES:
        _fail(f"{entry['id']} has invalid publication_state")
    if entry["evidence_state"] not in ALLOWED_EVIDENCE_STATES:
        _fail(f"{entry['id']} has invalid evidence_state")
    missing_animation = REQUIRED_ANIMATION_FIELDS - set(entry["animation"])
    if missing_animation:
        _fail(f"{entry['id']} animation is missing fields: {sorted(missing_animation)}")
    acceptance = (
        entry["accepted_digest"],
        entry["accepted_at"],
        entry["preview_digest"],
    )
    if entry["evidence_state"] in {"preregistered", "results_appended"}:
        if not all(acceptance):
            _fail(f"{entry['id']} accepted evidence state lacks acceptance fields")
    elif any(acceptance):
        _fail(f"{entry['id']} mutable evidence state contains acceptance fields")

    package = source_root / entry["slug"]
    article_path = package / "article.md"
    results_path = package / "results"
    film_path = package / "media" / "film"
    for path in (article_path, results_path, film_path):
        if not path.exists():
            _fail(f"{entry['id']} is missing package path: {path}")

    _check_article_source(entry, article_path)
    _check_film_files(entry, film_path)
    _check_film_spec(entry, film_path)
    _check_claim_ledger(entry, film_path, article_path)
    _check_film_runtime(entry, film_path)
    _check_film_receipt(entry, film_path)
    _check_output_state(entry, output_root)


def check_series(source_root: Path, output_root: Path) -> None:
    manifest = _read_json(source_root / "series.json")
    if manifest.get("schema_version") != 2:
        _fail("unsupported article manifest schema")
    entries: list[dict[str, Any]] = manifest.get("entries", [])
    if len(entries) != 9:
        _fail(f"the series must declare exactly nine entries, found {len(entries)}")
    ids = [entry["id"] for entry in entries]
    slugs = [entry["slug"] for entry in entries]
    if len(set(ids)) != len(ids) or len(set(slugs)) != len(slugs):
        _fail("article IDs and slugs must be unique")

    for index, entry in enumerate(entries):
        expected_previous = entries[index - 1]["id"] if index else None
        expected_next = entries[index + 1]["id"] if index + 1 < len(entries) else None
        if entry["previous_id"] != expected_previous or entry["next_id"] != expected_next:
            _fail(f"{entry['id']} breaks the declared sequential chain")
        _check_entry(source_root, output_root, entry)

    index_path = output_root / "articles" / "index.html"
    if not index_path.exists():
        _fail("built article index is missing")
    parser = _Links()
    parser.feed(index_path.read_text(encoding="utf-8"))
    for entry in entries:
        route = f"/fugue/articles/{entry['slug']}/"
        linked = route in parser.hrefs
        if entry["publication_state"] in {"working_draft", "published"} and not linked:
            _fail(f"visible article {entry['id']} is not linked from the index")
        if entry["publication_state"] == "planned" and linked:
            _fail(f"planned article {entry['id']} is linked from the index")

    publication = _read_json(output_root / "articles" / "publication.json")
    expected_visible = {
        entry["id"] for entry in entries
        if entry["publication_state"] in {"working_draft", "published"}
    }
    actual_visible = {entry["id"] for entry in publication.get("visible", [])}
    expected_released = {
        entry["id"] for entry in entries if entry["publication_state"] == "published"
    }
    actual_released = {entry["id"] for entry in publication.get("released", [])}
    if actual_visible != expected_visible or actual_released != expected_released:
        _fail("publication receipt does not match manifest visibility/release state")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: check_article_series.py SOURCE_ROOT ATLAS_DIST", file=sys.stderr)
        return 2
    check_series(Path(argv[1]).resolve(), Path(argv[2]).resolve())
    print("article series verified: nine visible routes, one released article")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
