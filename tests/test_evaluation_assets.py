from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fugue.bench import evaluation_assets
from fugue.bench.manifest import (
    BenchmarkManifest,
    DatasetSpec,
    HarnessSpec,
    TaskSpec,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_swe_bench_gold_paths_are_prepared_only_in_a_private_host_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = evaluation_assets._source_path(tmp_path)
    source.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "instance_id": ["org__repo-1"],
                "patch": ["diff --git a/src/old.py b/src/new.py\n"],
                "test_patch": ["diff --git a/tests/a.py b/tests/test_new.py\n"],
            }
        ),
        source,
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        evaluation_assets, "SWE_BENCH_VERIFIED_PARQUET_SHA256", digest
    )
    manifest = BenchmarkManifest(
        dataset=DatasetSpec(
            ref="swe-bench/swe-bench-verified",
            version=f"sha256:{digest}",
            source={
                "name": "princeton-nlp/SWE-bench_Verified",
                "revision": evaluation_assets.SWE_BENCH_VERIFIED_REVISION,
                "parquet_sha256": digest,
            },
        ),
        tasks=[TaskSpec(id="org__repo-1")],
        harnesses=[HarnessSpec("codex", "fugue.agents:FugueCodex")],
    )

    lock_path = evaluation_assets.prepare_evaluation_assets(manifest, tmp_path)
    assert lock_path is not None
    assert lock_path.stat().st_mode & 0o777 == 0o600
    prepared = evaluation_assets.attach_evaluation_assets(
        manifest, tmp_path, required=True
    )
    assert prepared.tasks[0].expected_paths == ("src/new.py", "tests/test_new.py")
    assert manifest.tasks[0].expected_paths == ()

    payload = json.loads(lock_path.read_text())
    payload["tasks"]["org__repo-1"]["expected_evidence_paths"] = ["leaked.py"]
    lock_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        evaluation_assets.attach_evaluation_assets(manifest, tmp_path, required=True)


def test_live_render_requires_explicit_evaluation_asset_preparation(
    tmp_path: Path,
) -> None:
    manifest = BenchmarkManifest(
        dataset=DatasetSpec(
            ref="swe-bench/swe-bench-verified",
            version=(
                "sha256:"
                + evaluation_assets.SWE_BENCH_VERIFIED_PARQUET_SHA256
            ),
            source={
                "name": "princeton-nlp/SWE-bench_Verified",
                "revision": evaluation_assets.SWE_BENCH_VERIFIED_REVISION,
                "parquet_sha256": evaluation_assets.SWE_BENCH_VERIFIED_PARQUET_SHA256,
            },
        ),
        tasks=[TaskSpec(id="org__repo-1")],
        harnesses=[HarnessSpec("codex", "fugue.agents:FugueCodex")],
    )

    assert evaluation_assets.attach_evaluation_assets(
        manifest, tmp_path, required=False
    ) == manifest
    with pytest.raises(ValueError, match="setup --prepare"):
        evaluation_assets.attach_evaluation_assets(manifest, tmp_path, required=True)
