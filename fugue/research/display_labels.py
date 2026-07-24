from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fugue.bench.library import get_experiment


def governed_display_labels(
    repo_root: Path,
    draft: Mapping[str, Any],
) -> dict[str, str]:
    """Return public labels owned by the registered experiment.

    Labels supplied by an accepted draft win.  Missing labels are filled from
    the registered experiment so a legacy immutable run can be projected with
    the same names as a newly previewed run without rewriting either artifact.
    """

    supplied = _labels(draft.get("display_labels"))
    experiment_id = str(draft.get("experiment_id") or "").strip()
    if not experiment_id:
        return supplied
    try:
        experiment = get_experiment(experiment_id, repo_root)
    except (FileNotFoundError, ValueError):
        return supplied

    defaults = {
        "research": f"Agent eval · {experiment.title}",
        "study": experiment.title,
        "harness": "Harness",
        "variant": "Loop design",
        "loop design": "Loop design",
    }
    defaults.update(
        {harness: _humanize(harness) for harness in experiment.harnesses if harness}
    )
    defaults.update(
        {
            variant.id: variant.label
            for variant in experiment.variants
            if variant.id and variant.label
        }
    )
    defaults.update(supplied)
    return defaults


def preview_with_governed_display_labels(
    repo_root: Path,
    preview: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy a preview and fill only its public display metadata."""

    value = dict(preview)
    raw_draft = preview.get("draft")
    if not isinstance(raw_draft, Mapping):
        return value
    draft = dict(raw_draft)
    labels = governed_display_labels(repo_root, draft)
    if labels:
        draft["display_labels"] = labels
    value["draft"] = draft
    return value


def _labels(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if isinstance(key, str)
        and key.strip()
        and isinstance(value, str)
        and value.strip()
    }


def _humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()
