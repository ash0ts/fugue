from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ATLAS = REPO_ROOT / "atlas"


def test_atlas_uses_reviewed_bundled_data_without_browser_apis() -> None:
    data_source = (ATLAS / "src/data.js").read_text(encoding="utf-8")
    javascript = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ATLAS / "src").glob("*.js"))
    )

    assert "import.meta.glob" in data_source
    assert "public/data/experiments" in data_source
    assert "fetch(" not in javascript
    assert "XMLHttpRequest" not in javascript
    assert "WebSocket(" not in javascript
    assert "innerHTML" not in javascript
    assert "insertAdjacentHTML" not in javascript
    assert "textContent" in javascript


def test_atlas_pages_are_semantic_keyboard_and_mobile_ready() -> None:
    pages = [
        ATLAS / name
        for name in (
            "index.html",
            "experiments.html",
            "experiment.html",
            "compare.html",
            "methods.html",
        )
    ]
    css = (ATLAS / "src/site.css").read_text(encoding="utf-8")

    for path in pages:
        body = path.read_text(encoding="utf-8")
        assert '<a class="skip-link"' in body
        assert (
            "<header" in body
            and "<nav" in body
            and "<main" in body
            and "<footer" in body
        )
        assert "http://" not in body
        assert "Product" in body
        assert "Studies" in body
        assert "Articles" in body
        assert "GitHub" in body
        nav = body.split("<nav", 1)[1].split("</nav>", 1)[0]
        assert "compare.html" not in nav
        assert "methods.html" not in nav
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css
    assert css.count("@media (max-width:") >= 2


def test_study_analysis_has_svg_text_alternatives_and_complete_denominators() -> None:
    source = (ATLAS / "src/analysis.js").read_text(encoding="utf-8")

    assert 'setAttribute("role", "img")' in source
    assert 'setAttribute("aria-label"' in source
    assert "Text alternative for task outcomes" in source
    assert "Lift against the exact control" in source
    assert "Text alternative for cost and latency" in source
    assert "view.groups.every(hasCompleteEfficiency)" in source
    assert "paired.length !== expectedPairs" in source
    assert "No compatible cohort is defined" in source


def test_experiment_detail_exposes_safe_task_evidence_and_weave_links() -> None:
    source = (ATLAS / "src/experiment.js").read_text(encoding="utf-8")

    assert "taskEvidence(study.cells)" in source
    assert "raw Agent content remains in Weave" in source
    assert "study.links.evaluations.entries()" in source
    assert "Route proof" in source
    assert "Runtime attested" in source
    assert "Unavailable" in source
    assert "No supported winner or pass-rate conclusion" in source


def test_methods_explains_native_harness_routing_and_attestation() -> None:
    body = (ATLAS / "methods.html").read_text(encoding="utf-8")

    assert "The harness keeps its own wire protocol" in body
    assert "Chat Completions" in body
    assert "Messages" in body
    assert "Responses" in body
    assert "pinned image digest" in body
    assert "exact locked config mounted read-only" in body
    assert "Model routing and MCP transport remain separate" in body
    assert "Configured only" in body


def test_product_contract_is_an_exact_commit_preview() -> None:
    product = json.loads(
        (ATLAS / "public/data/product.json").read_text(encoding="utf-8")
    )
    quickstart = product["quickstart"]

    assert product["maturity"] == "open_source_preview"
    assert product["featured_study_id"] == "kimi-k2-7-code-baseline"
    assert quickstart["availability"] == "preview_checkout"
    assert re.fullmatch(r"[0-9a-f]{40}", quickstart["revision"])
    assert quickstart["revision"] in quickstart["source_url"]
    assert any(quickstart["revision"] in command for command in quickstart["commands"])
    assert quickstart["pr_url"] == "https://github.com/ash0ts/fugue/pull/36"
    assert quickstart["proves"]
    assert quickstart["does_not_prove"]
    assert len(product["references"]) == 7
    assert all(
        set(reference) >= {"author", "title", "url", "relevance"}
        for reference in product["references"]
    )


def test_studies_are_grouped_without_editorial_ranking() -> None:
    source = (ATLAS / "src/studies.js").read_text(encoding="utf-8")
    home = (ATLAS / "index.html").read_text(encoding="utf-8")

    assert "Results available" in source
    assert "Contract qualification" in source
    assert "Active or incomplete" in source
    assert "Open study" in source
    assert "decision_value" not in source
    assert "Read the score" not in source
    assert "ordered by decision value" not in home.lower()


def test_compare_is_a_compatibility_gateway_to_embedded_analysis() -> None:
    source = (ATLAS / "src/compare.js").read_text(encoding="utf-8")

    assert 'target.hash = "analysis"' in source
    assert 'new URL("./experiment.html"' in source
    assert "window.location.replace(target)" in source
