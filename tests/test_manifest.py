from pathlib import Path

from fugue.bench.manifest import load_manifest, write_lock


def test_manifest_round_trip_lock(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pilot.yaml"
    manifest_path.write_text(
        """
dataset:
  ref: swe-bench/swe-bench-verified
memory_variants: [none, agentsmd]
harnesses:
  - name: hermes
    agent: fugue.agents:FugueHermes
tasks:
  - id: astropy__astropy-12907
    repo: astropy/astropy
    base_commit: d16bfe05a744909de4b27f5875fe0d4ed41ce607
"""
    )

    manifest = load_manifest(manifest_path)
    assert manifest.dataset.harbor_ref == "swe-bench/swe-bench-verified"
    assert manifest.select_memory_variants(["agentsmd"]) == ["agentsmd"]

    lock_path = tmp_path / "artifacts" / "lock.json"
    write_lock(
        path=lock_path,
        manifest_path=manifest_path,
        manifest=manifest,
        artifacts=[],
    )
    text = lock_path.read_text()
    assert '"schema_version": 1' in text
    assert "astropy__astropy-12907" in text
