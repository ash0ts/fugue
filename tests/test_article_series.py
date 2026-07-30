from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERIES_ROOT = (
    REPO_ROOT / "docs" / "articles" / "fugue-agentic-software-factory"
)
MANIFEST = json.loads((SERIES_ROOT / "series.json").read_text(encoding="utf-8"))


def test_series_manifest_is_sequential_and_exposes_review_drafts() -> None:
    entries = MANIFEST["entries"]
    assert len(entries) == 9
    assert [entry["id"] for entry in entries] == [
        "fugue-0a",
        "fugue-0b",
        "fugue-1",
        "fugue-2a",
        "fugue-2b",
        "fugue-3",
        "fugue-4a",
        "fugue-4b",
        "fugue-extra",
    ]
    assert [
        entry["id"]
        for entry in entries
        if entry["publication_state"] == "published"
    ] == ["fugue-0a"]
    assert [
        entry["id"]
        for entry in entries
        if entry["publication_state"] == "working_draft"
    ] == [
        "fugue-0b",
        "fugue-1",
        "fugue-2a",
        "fugue-2b",
        "fugue-3",
        "fugue-4a",
        "fugue-4b",
        "fugue-extra",
    ]
    for index, entry in enumerate(entries):
        assert entry["previous_id"] == (entries[index - 1]["id"] if index else None)
        assert entry["next_id"] == (
            entries[index + 1]["id"] if index + 1 < len(entries) else None
        )


def test_every_article_is_standalone_and_content_locked() -> None:
    for entry in MANIFEST["entries"]:
        article = (
            SERIES_ROOT / entry["slug"] / "article.md"
        ).read_text(encoding="utf-8")
        digest = f"sha256:{hashlib.sha256(article.encode()).hexdigest()}"
        assert entry["source_digest"] == digest
        assert "standalone" in article[:1_200].lower()
        assert "## Scope and terms" in article
        assert "## Try this in 15 minutes" in article
        assert "## When " in article
        assert "## What this does not show" in article
        assert "## References" in article
        assert (SERIES_ROOT / entry["slug"] / "sources.json").is_file()
        assert (SERIES_ROOT / entry["slug"] / "editorial-review.md").is_file()


def test_research_articles_are_unaccepted_draft_preregistrations() -> None:
    research = {"fugue-2a", "fugue-2b", "fugue-3", "fugue-4b"}
    for entry in MANIFEST["entries"]:
        if entry["id"] in research:
            assert entry["evidence_state"] == "draft_preregistration"
        assert entry["accepted_digest"] is None
        assert entry["accepted_at"] is None
        assert entry["preview_digest"] is None


def test_every_article_has_a_local_deterministic_film_package() -> None:
    runtime_hooks = (
        "window.__filmSpec",
        "window.__animationReady",
        "window.setAnimationTime",
        "window.releaseAnimation",
    )
    for entry in MANIFEST["entries"]:
        root = SERIES_ROOT / entry["slug"] / "media" / "film"
        spec = json.loads((root / "film-spec.json").read_text(encoding="utf-8"))
        ledger = json.loads((root / "claim-ledger.json").read_text(encoding="utf-8"))
        html = (root / f"{entry['slug']}.html").read_text(encoding="utf-8")

        assert spec["slug"] == entry["slug"]
        assert (spec["width"], spec["height"], spec["fps"]) == (1280, 720, 15)
        assert spec["duration"] == entry["animation"]["duration_seconds"]
        assert 84 <= spec["duration"] <= 110
        assert spec["silent"] is True
        assert 7 <= len(spec["checkpoints"]) <= 8
        assert len(ledger["claims"]) == len(spec["checkpoints"])
        assert ledger["schema_version"] == 2
        assert [item["name"] for item in spec["checkpoints"]] == entry["animation"]["chapters"]
        assert spec["placement"]["after_heading"] == entry["animation"]["after_heading"]
        assert spec["placement"]["bridge_to_heading"] == entry["animation"]["bridge_to_heading"]
        assert spec["posterScene"] == entry["animation"]["poster_scene"]
        assert spec["reducedMotionScene"] == entry["animation"]["reduced_motion_scene"]
        assert spec["posterScene"] in entry["animation"]["chapters"]
        assert spec["reducedMotionScene"] in entry["animation"]["chapters"]
        assert spec["safeArea"] == {
            "left": 54,
            "top": 40,
            "right": 54,
            "bottom": 100,
        }
        assert spec["typeScale"] == {
            "essential": 28,
            "supporting": 24,
            "compact": 18,
        }
        assert spec["transitionSeconds"] == 0.35
        assert all(hook in html for hook in runtime_hooks)
        assert (root / f"{entry['slug']}.mp4").is_file()
        assert (root / f"{entry['slug']}-poster.png").is_file()
        assert (root / "contact-sheet.png").is_file()
        assert (root / "transcript.md").is_file()
        receipt = json.loads(
            (root / "render-receipt.json").read_text(encoding="utf-8")
        )
        assert receipt["slug"] == entry["slug"]
        assert receipt["video"]["codec_name"] == "h264"
        assert receipt["video"]["pix_fmt"] == "yuv420p"
        assert receipt["video"]["color_range"] == "tv"
        for checkpoint in spec["checkpoints"]:
            assert (root / "checkpoints" / f"{checkpoint['name']}.png").is_file()
            assert checkpoint["name"] in receipt["checkpoint_frame_digests"]


def test_film_stories_are_data_bearing_and_article_local() -> None:
    stories = json.loads(
        (REPO_ROOT / "tools" / "article_film_stories.json").read_text(
            encoding="utf-8"
        )
    )
    assert stories["schema_version"] == 3
    assert len(stories["films"]) == 9
    generic_visuals = {"sequence", "roles", "grid", "split", "loop"}
    for story in stories["films"]:
        assert sum(scene["duration"] for scene in story["scenes"]) == story["duration"]
        assert story["poster_scene"] in {
            scene["id"] for scene in story["scenes"]
        }
        assert story["reduced_motion_scene"] in {
            scene["id"] for scene in story["scenes"]
        }
        assert story["scenes"][-1]["visual"]["type"] == "series_map"
        assert story["scenes"][-1]["duration"] / story["duration"] < 0.3
        for scene in story["scenes"]:
            assert scene["visual"]["type"] not in generic_visuals
            assert scene["source"].startswith("article.md#")
            assert len(scene["support"]) <= 2


def test_specialized_visuals_preserve_denominators_and_authority() -> None:
    stories = json.loads(
        (REPO_ROOT / "tools" / "article_film_stories.json").read_text(
            encoding="utf-8"
        )
    )
    scenes = {
        (story["part"], scene["id"]): scene["visual"]
        for story in stories["films"]
        for scene in story["scenes"]
    }
    funnel = scenes[("FUGUE 2B", "mechanism-funnel")]
    assert funnel["type"] == "mechanism_funnel"
    assert [stage["value"] for stage in funnel["sources"]] == [12, 3, 1]
    assert funnel["outcome"] == {
        "label": "TASKS PASSED",
        "value": 0,
        "denominator": 4,
    }
    assert scenes[("FUGUE 2A", "candidate-lattice")]["type"] == "candidate_assembly"
    assert scenes[("FUGUE 2A", "harness-reversal")]["type"] == "interaction_plot"
    mcp_stages = scenes[("FUGUE 3", "staged-study")]
    assert mcp_stages["type"] == "staged_study"
    assert [stage["cells"] for stage in mcp_stages["stages"]] == [8, 16]
    package_boundary = scenes[("FUGUE 3", "judge-gate")]
    assert package_boundary["type"] == "boundary"
    assert package_boundary["rule"] == "BEHAVIOR ≠ PACKAGE AUTHORITY"
    flagship_lifecycle = scenes[("FUGUE 4B", "attempt-lifecycle")]
    assert flagship_lifecycle["type"] == "attempt_lifecycle"
    assert flagship_lifecycle["lifecycle"] == [
        "PREPARE",
        "ATTEST",
        "EXECUTE",
        "PUBLISH",
        "CLEANUP",
    ]
    authority = scenes[("FUGUE 4A", "authority-lanes")]
    assert authority["lanes"] == [
        "CLAUDE CODE",
        "HUMAN",
        "FUGUE",
        "LOCAL HARBOR",
    ]
    assert authority["forbidden"].startswith("FORBIDDEN:")


def test_mcp_film_separates_zero_model_conformance_from_agent_results() -> None:
    ledger = json.loads(
        (
            SERIES_ROOT
            / "fugue-3-api-compatibility-is-not-agent-compatibility"
            / "media"
            / "film"
            / "claim-ledger.json"
        ).read_text(encoding="utf-8")
    )
    observed = [
        claim
        for claim in ledger["claims"]
        if claim["kind"] == "audited Fugue observation"
    ]
    assert {claim["id"] for claim in observed} == {
        "exact-revisions",
        "prepared-evidence",
    }
    assert "not an Agent behavioral result" in ledger["out_of_scope"]
    for claim in observed:
        claim_text = " ".join((claim["claim"], *claim["support"])).lower()
        assert "zero-model" in claim_text or "no model" in claim_text
        assert claim["caveat"] == ledger["out_of_scope"]


def test_film_state_foreground_background_pairs_meet_wcag_aa() -> None:
    def luminance(value: str) -> float:
        channels = [
            int(value[index : index + 2], 16) / 255
            for index in (1, 3, 5)
        ]
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def contrast(first: str, second: str) -> float:
        high, low = sorted(
            (luminance(first), luminance(second)), reverse=True
        )
        return (high + 0.05) / (low + 0.05)

    state_pairs = [
        ("#13272F", "#C8E4EB"),
        ("#13272F", "#CFE3D6"),
        ("#13272F", "#F2CBC8"),
        ("#13272F", "#F2DCB8"),
        ("#13272F", "#D5E0E1"),
        ("#13272F", "#DDD4E9"),
        ("#FFFFFF", "#176D89"),
    ]
    assert all(contrast(foreground, background) >= 4.5 for foreground, background in state_pairs)


def test_pages_workflow_builds_and_checks_article_sources() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "pages.yml"
    ).read_text(encoding="utf-8")
    assert '"docs/articles/**"' in workflow
    assert '"tools/check_article_series.py"' in workflow
    assert "python tools/check_article_series.py" in workflow
