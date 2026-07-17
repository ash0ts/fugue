from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ATLAS = REPO_ROOT / "atlas"


def test_atlas_uses_reviewed_bundled_data_without_browser_apis() -> None:
    data_source = (ATLAS / "src/data.js").read_text(encoding="utf-8")
    javascript = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ATLAS / "src").glob("*.js"))
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
    pages = [ATLAS / name for name in ("index.html", "experiment.html", "compare.html", "methods.html")]
    css = (ATLAS / "src/site.css").read_text(encoding="utf-8")

    for path in pages:
        body = path.read_text(encoding="utf-8")
        assert '<a class="skip-link"' in body
        assert "<header" in body and "<nav" in body and "<main" in body and "<footer" in body
        assert "http://" not in body and "https://" not in body
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css
    assert css.count("@media (max-width:") >= 2


def test_counterpoint_chart_has_svg_and_text_alternative() -> None:
    source = (ATLAS / "src/compare.js").read_text(encoding="utf-8")

    assert 'setAttribute("role", "img")' in source
    assert 'setAttribute("aria-label"' in source
    assert "Text alternative for counterpoint ribbon" in source
    assert "Memory lift against the exact baseline" in source
    assert "Cost and latency frontier" in source
    assert "Text alternative for cost and latency frontier" in source
    assert "experiment.matrix.workload_id" in source


def test_experiment_detail_exposes_safe_task_evidence_and_weave_links() -> None:
    source = (ATLAS / "src/experiment.js").read_text(encoding="utf-8")

    assert "taskEvidence(experiment.cells)" in source
    assert "raw Agent content remains in Weave" in source
    assert "experiment.links.evaluations.map" in source
    assert "Refusals" in source
