from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import ComparisonResultV3, read_comparison_result
from fugue.bench.files import atomic_write_json
from fugue.bench.scientific_reports import (
    CampaignReportIndexV1,
    ScientificReportError,
    ScientificReportV1,
    VisualAssetManifestV1,
    build_campaign_report_index,
    build_scientific_report,
    campaign_membership_from_dict,
    campaign_report_index_from_dict,
    render_report_markdown,
    scientific_report_from_dict,
    study_report_index_from_dict,
    visual_asset_manifest_from_dict,
)
from fugue.model_plane import EvidenceDestinationV1
from fugue.redaction import redact_text, redact_value


@dataclass(frozen=True)
class ReportBundleFileV1:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _bundle_path(self.path)
        _digest(self.sha256, "report bundle file digest")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 1:
            raise ScientificReportError("report bundle file size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificReportBundleV1:
    schema_version: Literal[1]
    kind: Literal["scientific_report_bundle"]
    source_result_digest: str
    report_digest: str
    visual_manifest_digest: str | None
    files: tuple[ReportBundleFileV1, ...]
    bundle_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "scientific_report_bundle":
            raise ScientificReportError("unsupported report bundle schema")
        _digest(self.source_result_digest, "report bundle result digest")
        _digest(self.report_digest, "report bundle report digest")
        if self.visual_manifest_digest is not None:
            _digest(self.visual_manifest_digest, "report visual manifest digest")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ScientificReportError(
                "report bundle file paths must be sorted and unique"
            )
        required = {"report.json", "report.md", "result.json"}
        if not required.issubset(paths):
            raise ScientificReportError(
                "report bundle requires report.json, report.md, and result.json"
            )
        if ("visual-assets.json" in paths) != (
            self.visual_manifest_digest is not None
        ):
            raise ScientificReportError(
                "report bundle visual manifest presence disagrees with its digest"
            )
        computed = stable_digest(self.unsigned_dict())
        if self.bundle_digest and self.bundle_digest != computed:
            raise ScientificReportError("report bundle digest does not match")
        if not self.bundle_digest:
            object.__setattr__(self, "bundle_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source_result_digest": self.source_result_digest,
            "report_digest": self.report_digest,
            "visual_manifest_digest": self.visual_manifest_digest,
            "files": [item.to_dict() for item in self.files],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "bundle_digest": self.bundle_digest}


@dataclass(frozen=True)
class LoadedScientificReportBundle:
    root: Path
    manifest: ScientificReportBundleV1
    result: ComparisonResultV3
    report: ScientificReportV1
    artifact_files: dict[str, bytes]


@dataclass(frozen=True)
class VisualDataPanelV1:
    id: str
    role: Literal[
        "experiment_matrix",
        "paired_dimension_heatmap",
        "task_difference_plot",
        "behavior_integrity",
        "judge_mechanism_efficiency",
        "provenance",
        "next_stage",
    ]
    evidence_state: Literal["observed", "planned", "unavailable"]
    claim_ids: tuple[str, ...]
    data: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.id or not self.claim_ids:
            raise ScientificReportError("visual data panel identity is incomplete")
        if tuple(self.claim_ids) != tuple(sorted(set(self.claim_ids))):
            raise ScientificReportError("visual data panel claim ids are not canonical")
        if self.evidence_state == "planned" and set(self.data) != {"description"}:
            raise ScientificReportError(
                "planned visual panels may contain only a description"
            )
        if self.evidence_state == "unavailable" and set(self.data) != {"reason"}:
            raise ScientificReportError(
                "unavailable visual panels may contain only a reason"
            )
        if self.evidence_state == "observed" and not self.data:
            raise ScientificReportError("observed visual panels require data")
        _privacy_check(_json_bytes(self.data), f"visual-data:{self.id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "evidence_state": self.evidence_state,
            "claim_ids": list(self.claim_ids),
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class ScientificVisualDataManifestV1:
    schema_version: Literal[1]
    kind: Literal["scientific_visual_data_manifest"]
    source_result_digest: str
    source_report_digest: str
    panels: tuple[VisualDataPanelV1, ...]
    manifest_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "scientific_visual_data_manifest":
            raise ScientificReportError("unsupported visual data manifest schema")
        _digest(self.source_result_digest, "visual data result digest")
        _digest(self.source_report_digest, "visual data report digest")
        ids = tuple(item.id for item in self.panels)
        if ids != tuple(sorted(set(ids))):
            raise ScientificReportError("visual data panel ids must be sorted and unique")
        states = {item.evidence_state for item in self.panels}
        if "observed" not in states or "planned" not in states:
            raise ScientificReportError(
                "visual data must distinguish observed evidence from planned work"
            )
        computed = stable_digest(self.unsigned_dict())
        if self.manifest_digest and self.manifest_digest != computed:
            raise ScientificReportError("visual data manifest digest does not match")
        if not self.manifest_digest:
            object.__setattr__(self, "manifest_digest", computed)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source_result_digest": self.source_result_digest,
            "source_report_digest": self.source_report_digest,
            "panels": [item.to_dict() for item in self.panels],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "manifest_digest": self.manifest_digest}


def build_scientific_report_bundle(
    result_path: Path,
    output_dir: Path,
    *,
    visual_manifest_path: Path | None = None,
) -> LoadedScientificReportBundle:
    """Build one public-safe offline bundle from a canonical V3 result."""

    source = result_path.resolve()
    parsed = read_comparison_result(source)
    if not isinstance(parsed, ComparisonResultV3):
        raise ScientificReportError("scientific reports require ComparisonResultV3")
    visuals = (
        _read_visual_manifest(visual_manifest_path)
        if visual_manifest_path is not None
        else None
    )
    if visuals is not None and visuals.source_result_digest != parsed.result_digest:
        raise ScientificReportError("visual assets target a different result digest")
    report = build_scientific_report(parsed, visual_assets=visuals)
    files: dict[str, bytes] = {
        "report.json": _json_bytes(report.to_dict()),
        "report.md": render_report_markdown(report).encode("utf-8"),
        "result.json": _json_bytes(parsed.to_dict()),
    }
    if visuals is not None:
        assert visual_manifest_path is not None
        files["visual-assets.json"] = _json_bytes(visuals.to_dict())
        visual_root = visual_manifest_path.resolve().parent
        for asset in visuals.assets:
            asset_path = _resolved_child(visual_root, asset.path)
            if asset_path.is_symlink() or not asset_path.is_file():
                raise ScientificReportError(
                    f"visual asset is not a regular file: {asset.path}"
                )
            content = asset_path.read_bytes()
            if len(content) != asset.size_bytes or _sha256(content) != asset.sha256:
                raise ScientificReportError(
                    f"visual asset bytes disagree with its manifest: {asset.path}"
                )
            _privacy_check(content, asset.path, media_type=asset.media_type)
            files[asset.path] = content
    for name, content in files.items():
        _privacy_check(content, name)
    records = tuple(
        ReportBundleFileV1(
            path=name,
            sha256=_sha256(content),
            size_bytes=len(content),
        )
        for name, content in sorted(files.items())
    )
    manifest = ScientificReportBundleV1(
        schema_version=1,
        kind="scientific_report_bundle",
        source_result_digest=parsed.result_digest,
        report_digest=report.report_digest,
        visual_manifest_digest=(visuals.manifest_digest if visuals else None),
        files=records,
    )
    destination = output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        existing = read_scientific_report_bundle(destination)
        if existing.manifest != manifest:
            raise ScientificReportError(
                "report bundle destination contains a different immutable bundle"
            )
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        _atomic_bytes(destination / name, content)
    atomic_write_json(destination / "bundle.json", manifest.to_dict())
    return read_scientific_report_bundle(destination)


def read_scientific_report_bundle(root: Path) -> LoadedScientificReportBundle:
    destination = root.resolve()
    raw = _json_mapping(destination / "bundle.json", "report bundle")
    manifest = scientific_report_bundle_from_dict(raw)
    expected_paths = {item.path for item in manifest.files}
    observed_paths = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "bundle.json"
    }
    if observed_paths != expected_paths:
        raise ScientificReportError("report bundle files do not match its manifest")
    artifact_files: dict[str, bytes] = {}
    by_path = {item.path: item for item in manifest.files}
    for name in sorted(expected_paths):
        path = _resolved_child(destination, name)
        if path.is_symlink() or not path.is_file():
            raise ScientificReportError(f"report bundle file is unsafe: {name}")
        content = path.read_bytes()
        record = by_path[name]
        if len(content) != record.size_bytes or _sha256(content) != record.sha256:
            raise ScientificReportError(f"report bundle file drifted: {name}")
        _privacy_check(content, name)
        artifact_files[name] = content
    result = read_comparison_result(destination / "result.json")
    if not isinstance(result, ComparisonResultV3):
        raise ScientificReportError("report bundle result is not V3")
    report = scientific_report_from_dict(
        _json_mapping(destination / "report.json", "scientific report")
    )
    bundle_bytes = (destination / "bundle.json").read_bytes()
    _privacy_check(bundle_bytes, "bundle.json")
    artifact_files["bundle.json"] = bundle_bytes
    validated = validate_scientific_report_bundle_files(
        artifact_files,
        result=result,
        report=report,
    )
    if validated != manifest:
        raise ScientificReportError("report bundle manifest does not recompute")
    return LoadedScientificReportBundle(
        root=destination,
        manifest=manifest,
        result=result,
        report=report,
        artifact_files=artifact_files,
    )


def validate_scientific_report_bundle_files(  # noqa: C901 - one bounded bundle audit
    artifact_files: Mapping[str, bytes],
    *,
    result: ComparisonResultV3,
    report: ScientificReportV1,
) -> ScientificReportBundleV1:
    """Recompute one complete in-memory Study bundle at publication time."""

    if type(result) is not ComparisonResultV3:
        raise ScientificReportError("report bundle requires a canonical ComparisonResultV3")
    bundle_bytes = artifact_files.get("bundle.json")
    if not isinstance(bundle_bytes, bytes) or not bundle_bytes:
        raise ScientificReportError("report bundle manifest is missing")
    try:
        raw_manifest = json.loads(bundle_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScientificReportError("report bundle manifest is not valid JSON") from exc
    if not isinstance(raw_manifest, Mapping):
        raise ScientificReportError("report bundle manifest must be an object")
    manifest = scientific_report_bundle_from_dict(raw_manifest)
    if bundle_bytes != _json_bytes(manifest.to_dict()):
        raise ScientificReportError("report bundle manifest is not canonical JSON")
    expected_paths = {item.path for item in manifest.files}
    if set(artifact_files) != {*expected_paths, "bundle.json"}:
        raise ScientificReportError("report bundle files do not match its manifest")
    by_path = {item.path: item for item in manifest.files}
    for name in sorted(expected_paths):
        content = artifact_files[name]
        record = by_path[name]
        if len(content) != record.size_bytes or _sha256(content) != record.sha256:
            raise ScientificReportError(f"report bundle file drifted: {name}")
        _privacy_check(content, name)
    if result.result_digest != manifest.source_result_digest:
        raise ScientificReportError("report bundle result digest disagrees")
    if report.report_digest != manifest.report_digest:
        raise ScientificReportError("report bundle report digest disagrees")
    if report.source_result_digest != result.result_digest:
        raise ScientificReportError("scientific report binds a different result")
    if artifact_files["result.json"] != _json_bytes(result.to_dict()):
        raise ScientificReportError("report bundle result.json is not canonical")
    if artifact_files["report.json"] != _json_bytes(report.to_dict()):
        raise ScientificReportError("report bundle report.json is not canonical")
    if artifact_files["report.md"] != render_report_markdown(report).encode("utf-8"):
        raise ScientificReportError("report Markdown does not recompute")
    expected_report = build_scientific_report(
        result,
        visual_assets=report.visual_assets,
    )
    if expected_report != report:
        raise ScientificReportError("scientific report does not recompute from result")
    if report.visual_assets is None:
        if manifest.visual_manifest_digest is not None:
            raise ScientificReportError("unexpected visual manifest digest")
    else:
        visual_bytes = artifact_files.get("visual-assets.json")
        if visual_bytes is None:
            raise ScientificReportError("report visual manifest is missing")
        try:
            raw_visual = json.loads(visual_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScientificReportError("report visual manifest is invalid") from exc
        if not isinstance(raw_visual, Mapping):
            raise ScientificReportError("report visual manifest must be an object")
        visual = visual_asset_manifest_from_dict(raw_visual)
        if visual != report.visual_assets:
            raise ScientificReportError("visual manifest disagrees with report")
        if visual.manifest_digest != manifest.visual_manifest_digest:
            raise ScientificReportError("visual manifest digest disagrees")
        for asset in visual.assets:
            content = artifact_files.get(asset.path)
            if content is None or _sha256(content) != asset.sha256:
                raise ScientificReportError("visual asset does not recompute")
    _privacy_check(bundle_bytes, "bundle.json")
    return manifest


def write_campaign_report_index(
    *,
    campaign_id: str,
    publication_project: str,
    membership_path: Path,
    study_index_paths: Sequence[Path],
    output_path: Path,
) -> CampaignReportIndexV1:
    studies = tuple(
        study_report_index_from_dict(
            _json_mapping(path.resolve(), "Study report index")
        )
        for path in study_index_paths
    )
    membership = campaign_membership_from_dict(
        _json_mapping(membership_path.resolve(), "campaign membership lock")
    )
    publication_destination = _campaign_publication_destination(
        publication_project,
        studies,
    )
    index = build_campaign_report_index(
        campaign_id,
        publication_destination,
        membership,
        studies,
    )
    if output_path.exists():
        existing = campaign_report_index_from_dict(
            _json_mapping(output_path, "campaign report index")
        )
        if existing != index:
            raise ScientificReportError(
                "campaign index path contains a different immutable index"
            )
        return existing
    atomic_write_json(output_path, index.to_dict())
    return campaign_report_index_from_dict(
        _json_mapping(output_path, "campaign report index")
    )


def _campaign_publication_destination(
    publication_project: str,
    studies: Sequence[Any],
) -> EvidenceDestinationV1:
    if publication_project.count("/") != 1:
        raise ScientificReportError("campaign publication project is invalid")
    if not studies:
        raise ScientificReportError("campaign destination requires Study indexes")
    receipts = [receipt for study in studies for receipt in study.reports]
    if not receipts:
        raise ScientificReportError("campaign destination requires Study receipts")
    endpoint_identities = {
        (
            receipt.publication_destination.entity,
            receipt.publication_destination.api_base_url,
            receipt.publication_destination.trace_base_url,
            receipt.publication_destination.app_base_url,
        )
        for receipt in receipts
    }
    if len(endpoint_identities) != 1:
        raise ScientificReportError("campaign Study destinations disagree")
    entity, project = publication_project.split("/", 1)
    base_entity, api_base_url, trace_base_url, app_base_url = next(
        iter(endpoint_identities)
    )
    if entity != base_entity:
        raise ScientificReportError("campaign entity disagrees with its Studies")
    return EvidenceDestinationV1(
        entity=entity,
        project=project,
        api_base_url=api_base_url,
        trace_base_url=trace_base_url,
        app_base_url=app_base_url,
    )


def campaign_publication_files(
    index: CampaignReportIndexV1,
) -> dict[str, bytes]:
    accepted = campaign_report_index_from_dict(index.to_dict())
    return {
        "campaign-index.json": _json_bytes(accepted.to_dict()),
        "report.md": render_report_markdown(accepted).encode("utf-8"),
    }


def build_visual_data_manifest(
    bundle: LoadedScientificReportBundle,
) -> ScientificVisualDataManifestV1:
    """Derive graphic inputs while keeping future work explicitly planned."""

    result = bundle.result
    report = bundle.report
    claim_status = {item.id: item.status for item in report.claim_ledger}
    pair_rows = [
        {
            "task_id": pair.task_id,
            "attempt": pair.attempt,
            "status": pair.status,
            "dimensions": [
                {
                    "id": change.id,
                    "label": change.label,
                    "role": change.role,
                    "critical": change.critical,
                    "status": change.status,
                }
                for change in pair.dimension_changes
            ],
        }
        for pair in result.paired_cases
    ]
    task_validity = [
        {
            "task_id": item.task_id,
            "status": item.status,
            "blockers": list(item.blockers),
        }
        for item in result.task_validity
    ]
    evidence_claims = (
        "efficiency-evidence",
        "judge-evidence",
        "mechanism-evidence",
    )
    evidence_available = any(
        claim_status.get(item) != "unavailable" for item in evidence_claims
    )
    evidence_data = (
        {
            "judge": _selected_summary(
                result.judge_summary,
                keys=("status", "claim_status", "unavailable_attempts", "by_variant"),
            ),
            "mechanism": _selected_summary(result.mechanism_summary),
            "efficiency": _selected_summary(
                result.operational_summary,
                keys=(
                    "observed_cost_usd",
                    "agent_observed_cost_usd",
                    "total_cost_status",
                    "latency_ms",
                    "latency_rows",
                    "input_tokens",
                    "output_tokens",
                    "usage_rows",
                ),
            ),
        }
        if evidence_available
        else {"reason": "Judge, mechanism, and efficiency evidence are unavailable."}
    )
    panels = (
        VisualDataPanelV1(
            id="behavior-integrity",
            role="behavior_integrity",
            evidence_state="observed",
            claim_ids=("deterministic-outcome", "evidence-integrity"),
            data={
                "behavioral_status": report.behavioral_status,
                "task_validity_status": report.task_validity_status,
                "evidence_grade": report.evidence_grade,
                "integrity_status": str(result.integrity.get("status") or "unknown"),
                "named_blockers": list(report.named_blockers),
                "task_validity": task_validity,
            },
        ),
        VisualDataPanelV1(
            id="experiment-matrix",
            role="experiment_matrix",
            evidence_state="observed",
            claim_ids=("deterministic-outcome", "evidence-integrity"),
            data={
                "comparison_id": report.comparison_id,
                "pair_count": report.pair_count,
                "exact_revisions": list(report.exact_revisions),
                "runtime_digests": list(report.runtime_digests),
                "source_project": report.source_project,
                "result_project": report.result_project,
            },
        ),
        VisualDataPanelV1(
            id="judge-mechanism-efficiency",
            role="judge_mechanism_efficiency",
            evidence_state="observed" if evidence_available else "unavailable",
            claim_ids=tuple(sorted(evidence_claims)),
            data=evidence_data,
        ),
        VisualDataPanelV1(
            id="next-stage",
            role="next_stage",
            evidence_state="planned",
            claim_ids=("deterministic-outcome",),
            data={"description": report.narrative["next_action"]},
        ),
        VisualDataPanelV1(
            id="paired-dimensions",
            role="paired_dimension_heatmap",
            evidence_state="observed",
            claim_ids=("deterministic-blockers", "deterministic-outcome"),
            data={"pairs": pair_rows},
        ),
        VisualDataPanelV1(
            id="provenance",
            role="provenance",
            evidence_state="observed",
            claim_ids=("evidence-integrity",),
            data={
                "result_digest": result.result_digest,
                "report_digest": report.report_digest,
                "bundle_digest": bundle.manifest.bundle_digest,
                "chain": [
                    "ComparisonResultV3",
                    "ScientificReportV1",
                    "ScientificReportBundleV1",
                ],
            },
        ),
        VisualDataPanelV1(
            id="task-differences",
            role="task_difference_plot",
            evidence_state="observed",
            claim_ids=("deterministic-outcome",),
            data={
                "improved": result.improved,
                "regressed": result.regressed,
                "mixed": result.mixed,
                "unchanged": result.unchanged,
                "incomplete": result.incomplete,
                "pairs": [
                    {
                        "task_id": pair.task_id,
                        "attempt": pair.attempt,
                        "status": pair.status,
                    }
                    for pair in result.paired_cases
                ],
            },
        ),
    )
    manifest = ScientificVisualDataManifestV1(
        schema_version=1,
        kind="scientific_visual_data_manifest",
        source_result_digest=result.result_digest,
        source_report_digest=report.report_digest,
        panels=tuple(sorted(panels, key=lambda item: item.id)),
    )
    claim_ids = {item.id for item in report.claim_ledger}
    if any(
        claim_id not in claim_ids
        for panel in manifest.panels
        for claim_id in panel.claim_ids
    ):
        raise ScientificReportError("visual data references an unknown report claim")
    return manifest


def write_visual_data_manifest(
    bundle: LoadedScientificReportBundle,
    output_path: Path,
) -> ScientificVisualDataManifestV1:
    manifest = build_visual_data_manifest(bundle)
    write_json_once(
        output_path,
        manifest.to_dict(),
        label="scientific visual data manifest",
    )
    return visual_data_manifest_from_dict(
        _json_mapping(output_path, "scientific visual data manifest")
    )


def visual_data_manifest_from_dict(
    raw: Mapping[str, Any],
) -> ScientificVisualDataManifestV1:
    expected = {
        "schema_version",
        "kind",
        "source_result_digest",
        "source_report_digest",
        "panels",
        "manifest_digest",
    }
    if set(raw) != expected:
        raise ScientificReportError("visual data manifest has invalid fields")
    panels = raw.get("panels")
    if not isinstance(panels, list):
        raise ScientificReportError("visual data panels must be an array")
    parsed: list[VisualDataPanelV1] = []
    for raw_panel in panels:
        if not isinstance(raw_panel, Mapping) or set(raw_panel) != {
            "id",
            "role",
            "evidence_state",
            "claim_ids",
            "data",
        }:
            raise ScientificReportError("visual data panel has invalid fields")
        claim_ids = raw_panel.get("claim_ids")
        data = raw_panel.get("data")
        if not isinstance(claim_ids, list) or not isinstance(data, Mapping):
            raise ScientificReportError("visual data panel payload is invalid")
        value = dict(raw_panel)
        value["claim_ids"] = tuple(claim_ids)
        value["data"] = dict(data)
        parsed.append(VisualDataPanelV1(**value))
    value = dict(raw)
    value["panels"] = tuple(parsed)
    return ScientificVisualDataManifestV1(**value)


def write_json_once(path: Path, value: Mapping[str, Any], *, label: str) -> Path:
    if path.exists():
        existing = _json_mapping(path, label)
        if existing != dict(value):
            raise ScientificReportError(f"{label} path contains different data")
        return path
    return atomic_write_json(path, dict(value))


def _read_visual_manifest(path: Path) -> VisualAssetManifestV1:
    return visual_asset_manifest_from_dict(
        _json_mapping(path.resolve(), "visual asset manifest")
    )


def _selected_summary(
    raw: Mapping[str, Any],
    *,
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = dict(raw) if keys is None else {key: raw.get(key) for key in keys}
    value = json.loads(json.dumps(selected, sort_keys=True, default=str))
    if not isinstance(value, dict):  # pragma: no cover - fixed by construction
        raise ScientificReportError("report summary is not an object")
    _privacy_check(_json_bytes(value), "visual-data-summary.json")
    return value


def scientific_report_bundle_from_dict(
    raw: Mapping[str, Any],
) -> ScientificReportBundleV1:
    expected = {
        "schema_version",
        "kind",
        "source_result_digest",
        "report_digest",
        "visual_manifest_digest",
        "files",
        "bundle_digest",
    }
    if set(raw) != expected:
        raise ScientificReportError("report bundle has invalid fields")
    files = raw.get("files")
    if not isinstance(files, list):
        raise ScientificReportError("report bundle files must be an array")
    parsed: list[ReportBundleFileV1] = []
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ScientificReportError("report bundle file entry is invalid")
        parsed.append(ReportBundleFileV1(**dict(item)))
    value = dict(raw)
    value["files"] = tuple(parsed)
    return ScientificReportBundleV1(**value)


def _json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScientificReportError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ScientificReportError(f"{label} must be an object")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _resolved_child(root: Path, value: str) -> Path:
    _bundle_path(value)
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ScientificReportError("report bundle path escapes its root") from exc
    return resolved


def _bundle_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ScientificReportError("report bundle path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ScientificReportError("report bundle path is invalid")


def _privacy_check(
    content: bytes,
    label: str,
    *,
    media_type: str | None = None,
) -> None:
    text_types = {
        "application/json",
        "image/svg+xml",
        "text/markdown",
        "text/plain",
    }
    inferred_text = Path(label).suffix in {".json", ".md", ".svg", ".txt"}
    if not ((media_type in text_types) if media_type is not None else inferred_text):
        return
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScientificReportError(f"public report text is not UTF-8: {label}") from exc
    if redact_text(text) != text:
        raise ScientificReportError(f"public report file contains sensitive text: {label}")
    if (media_type == "application/json" or Path(label).suffix == ".json"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScientificReportError(
                f"public report JSON is malformed: {label}"
            ) from exc
        if redact_value(parsed) != parsed:
            raise ScientificReportError(
                f"public report file contains sensitive text: {label}"
            )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _digest(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ScientificReportError(f"{label} is invalid")
