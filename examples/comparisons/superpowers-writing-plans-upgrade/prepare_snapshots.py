"""Prepare deterministic historical Fugue source archives for the Skill canary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (
    REPO_ROOT
    / ".fugue/comparison-resources/superpowers-writing-plans-upgrade"
)
SNAPSHOTS = {
    "credential-rotation": {
        "commit": "c729d68f8c2e0f1b8ebc93428b1579973f81ac4a",
        "paths": (
            "fugue/research/bootstrap.py",
            "tests/test_research_agent_interface.py",
        ),
    },
    "routing-separation": {
        "commit": "02f75cd953e5286389f5f2b6712ef95670a42a5f",
        "paths": (
            "fugue/agents/model_plane.py",
            "fugue/assistant.py",
            "fugue/bench/ai.py",
            "fugue/bench/campaign_lifecycle.py",
            "fugue/bench/cli.py",
            "fugue/bench/evaluations.py",
            "fugue/bench/export.py",
            "fugue/bench/job_config.py",
            "fugue/bench/operator.py",
            "fugue/bench/reproducibility.py",
            "fugue/bench/task_authoring.py",
            "fugue/bridge.py",
            "fugue/model_plane.py",
            "fugue/preflight.py",
            "fugue/research/__init__.py",
            "fugue/research/bootstrap.py",
            "fugue/research/contracts.py",
            "fugue/research/experiment_views.py",
            "fugue/research/service.py",
            "fugue/research/traces.py",
            "fugue/serve/runtime.py",
            "fugue/task_interaction.py",
            "fugue/tui.py",
            "fugue/weave_support.py",
            "tests/test_bridge.py",
            "tests/test_campaigns.py",
            "tests/test_candidates.py",
            "tests/test_enterprise_evidence_use.py",
            "tests/test_experiment_atlas.py",
            "tests/test_experiment_views.py",
            "tests/test_model_plane.py",
            "tests/test_preflight.py",
            "tests/test_research_agent_interface.py",
            "tests/test_research_service.py",
        ),
    },
}


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, snapshot in SNAPSHOTS.items():
        commit = str(snapshot["commit"])
        resolved = _git("rev-parse", f"{commit}^{{commit}}")
        if resolved != commit:
            raise RuntimeError(f"snapshot {name} did not resolve to its exact commit")
        paths = tuple(str(value) for value in snapshot["paths"])
        archive = OUTPUT / f"{name}-{commit}.tar"
        subprocess.run(
            (
                "git",
                "archive",
                "--format=tar",
                "--prefix=repo/",
                f"--output={archive}",
                commit,
                *paths,
            ),
            cwd=REPO_ROOT,
            check=True,
        )
        records.append(
            {
                "id": name,
                "commit": commit,
                "archive": archive.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(archive),
                "paths": list(paths),
            }
        )
    manifest = {
        "schema_version": 1,
        "repository": "https://github.com/ash0ts/fugue",
        "snapshots": records,
    }
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = OUTPUT / "snapshots.lock.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
