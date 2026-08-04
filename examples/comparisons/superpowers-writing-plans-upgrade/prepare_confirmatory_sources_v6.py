#!/usr/bin/env python3
"""Prepare the V6 source archive with the exact complete-artifact scorer.

The historical preparer remains byte-stable and defaults to the V3 scorer used
by V5.  This trusted wrapper reuses its audited Git/archive implementation,
selects the V4 scorer before any preparation work, and emits a V6-specific
receipt.  It is never mounted or executed inside an Agent trial.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import runpy
import shutil
import tempfile
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent
PREPARER = EXAMPLE / "prepare_confirmatory_sources.py"
SCORER = EXAMPLE / "plan_quality_scorer_v4.py"
SPEC = EXAMPLE / "confirmatory-v6.yaml"
AMENDMENT = EXAMPLE / "preregistration-confirmatory-v6-amendment.json"
REPO_ROOT = EXAMPLE.parents[2]
OUTPUT = (
    REPO_ROOT
    / ".fugue/comparison-resources/superpowers-writing-plans-conference-v1"
)
V6_BASE_RECEIPT = OUTPUT / "preparation-v6.base.receipt.json"
V6_RECEIPT = OUTPUT / "preparation-v6.receipt.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_consistent_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe V6 preparation output: {path}")
    if path.is_file():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"existing V6 preparation output drifted: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _install_archive_without_rewriting(
    staged_archive: Path,
    final_archive: Path,
) -> None:
    expected_sha256 = _sha256(staged_archive)
    if final_archive.is_symlink() or (
        final_archive.exists() and not final_archive.is_file()
    ):
        raise RuntimeError("unsafe shared confirmatory source archive")
    if final_archive.is_file():
        if _sha256(final_archive) != expected_sha256:
            raise RuntimeError("shared confirmatory source archive drifted")
        return
    final_archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_archive.with_name(
        f".{final_archive.name}.{os.getpid()}.v6.tmp"
    )
    try:
        shutil.copyfile(staged_archive, temporary)
        temporary.chmod(0o444)
        os.replace(temporary, final_archive)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(*, output: Path = OUTPUT) -> dict[str, object]:
    module = runpy.run_path(PREPARER.as_posix())
    main = module["main"]
    globals_ = main.__globals__
    historical_scorer = Path(globals_["SCORER"])
    if historical_scorer.name != "plan_quality_scorer_v3.py":
        raise RuntimeError("historical confirmatory preparer scorer drifted")
    repository_root = Path(globals_["REPO_ROOT"]).resolve()
    resolved_output = output.resolve()
    try:
        archive_relative = resolved_output.relative_to(repository_root) / str(
            globals_["ARCHIVE_NAME"]
        )
    except ValueError as exc:
        raise RuntimeError("V6 preparation output must stay inside the repository") from exc

    # Run the audited historical implementation in an isolated staging
    # directory. Its scorer and archive path are rebound only in that private
    # module instance, so V1-V5's shared preparation receipt is never opened or
    # rewritten by V6.
    with tempfile.TemporaryDirectory(prefix="fugue-superpowers-v6-") as temporary:
        staging_output = Path(temporary) / "prepared"
        globals_["SCORER"] = SCORER
        globals_["OUTPUT"] = staging_output
        globals_["ARCHIVE_RELATIVE"] = archive_relative
        with contextlib.redirect_stdout(io.StringIO()):
            main()
        staged_base_receipt = staging_output / "preparation.receipt.json"
        base = json.loads(staged_base_receipt.read_text(encoding="utf-8"))
        _install_archive_without_rewriting(
            staging_output / str(globals_["ARCHIVE_NAME"]),
            resolved_output / str(globals_["ARCHIVE_NAME"]),
        )

    if (
        base.get("artifacts", {}).get("scorer_sha256") != _sha256(SCORER)
        or base.get("design", {}).get("planned_cells") != 192
        or base.get("source", {}).get("contains_task_oracle") is not False
    ):
        raise RuntimeError("V6 base preparation receipt is inconsistent")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    unsigned_amendment = {
        key: value for key, value in amendment.items() if key != "amendment_digest"
    }
    if amendment.get("amendment_digest") != _stable_digest(unsigned_amendment):
        raise RuntimeError("V6 amendment digest is invalid")

    v6_base_receipt = resolved_output / V6_BASE_RECEIPT.name
    v6_receipt = resolved_output / V6_RECEIPT.name
    _write_consistent_json(v6_base_receipt, base)

    receipt: dict[str, object] = {
        "schema_version": 1,
        "id": "superpowers-writing-plans-confirmatory-preparation-v6",
        "study_id": "superpowers-writing-plans-confirmatory-v6",
        "study_class": "measurement_development_descriptive",
        "conference_claim_eligible": False,
        "population_claim_eligible": False,
        "private_inputs_serialized": False,
        "base_preparation_manifest_digest": base["manifest_digest"],
        "base_preparation_receipt_sha256": _sha256(v6_base_receipt),
        "comparison_spec_sha256": _sha256(SPEC),
        "historical_preparer_sha256": _sha256(PREPARER),
        "scorer_sha256": _sha256(SCORER),
        "amendment_sha256": _sha256(AMENDMENT),
        "amendment_digest": amendment["amendment_digest"],
    }
    receipt["receipt_digest"] = _stable_digest(receipt)
    _write_consistent_json(v6_receipt, receipt)
    return receipt


def main() -> None:
    print(json.dumps(prepare(), sort_keys=True))


if __name__ == "__main__":
    main()
