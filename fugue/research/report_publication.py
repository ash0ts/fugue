from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import ComparisonResultV3, _verify_v2_result_integrity
from fugue.bench.export import publish_scoped_once
from fugue.bench.reporting import validate_scientific_report_bundle_files
from fugue.bench.scientific_reports import (
    CampaignReportIndexV1,
    ReportPublicationReceiptV1,
    ScientificReportError,
    ScientificReportV1,
    StudyReportIndexV1,
    build_scientific_report,
    campaign_report_index_from_dict,
    render_report_markdown,
    scientific_report_from_dict,
    study_report_index_from_dict,
)
from fugue.model_plane import EvidenceDestinationV1, evidence_destination_from_dict
from fugue.research.experiment_views import (
    ExperimentViewV3,
    experiment_view_from_dict,
)
from fugue.research.records import ResearchEvidenceRefV1, ResearchLogEventV1
from fugue.research.store import StudyStore

_SUMMARY_KEY = "fugue.report_publication_v1"
_PHASE_SCHEMA_VERSION = 3


class ReportRemotePublisher(Protocol):
    def publish(
        self,
        *,
        request: Mapping[str, Any],
        content: bytes,
        markdown: str,
        artifact_files: Mapping[str, bytes],
    ) -> Mapping[str, Any]: ...


def publish_report(
    document: ScientificReportV1 | CampaignReportIndexV1,
    *,
    publisher: ReportRemotePublisher,
    artifact_files: Mapping[str, bytes],
    source_result: ComparisonResultV3 | None = None,
    ledger_root: Path | None = None,
) -> ReportPublicationReceiptV1:
    """Publish one Study or campaign document through one ledger transaction."""

    if isinstance(document, ScientificReportV1):
        accepted: ScientificReportV1 | CampaignReportIndexV1 = (
            scientific_report_from_dict(document.to_dict())
        )
        scope_kind = "study"
        scope_id = accepted.comparison_id
        document_digest = accepted.report_digest
        source_projects = (accepted.source_project,)
        source_artifact_digests = (
            accepted.source_result_digest,
            accepted.report_digest,
        )
        if source_result is None:
            raise ScientificReportError(
                "Study report publication requires its canonical ComparisonResultV3"
            )
        canonical_result = _canonical_source_result(source_result)
        expected_report = build_scientific_report(
            canonical_result,
            visual_assets=accepted.visual_assets,
        )
        if expected_report != accepted:
            raise ScientificReportError(
                "scientific report does not recompute from ComparisonResultV3"
            )
        publication_destination = canonical_result.evidence_topology.result_destination
        if publication_destination.project_slug != accepted.result_project:
            raise ScientificReportError("report result destination disagrees")
    else:
        accepted = campaign_report_index_from_dict(document.to_dict())
        scope_kind = "campaign"
        scope_id = accepted.campaign_id
        document_digest = accepted.index_digest
        if not accepted.complete:
            raise ScientificReportError("incomplete campaign index cannot be published")
        publication_destination = accepted.publication_destination
        source_projects = tuple(
            sorted(
                {
                    project
                    for study in accepted.studies
                    for receipt in study.reports
                    for project in receipt.task_source_projects
                }
            )
        )
        source_artifact_digests = tuple(
            sorted(item.index_digest for item in accepted.studies)
        )
        if source_result is not None:
            raise ScientificReportError(
                "campaign report publication cannot bind one Study result"
            )
    project = publication_destination.project_slug
    if project in source_projects:
        raise ScientificReportError("report-only Run cannot enter task inputs")
    content = _json_bytes(accepted.to_dict())
    markdown = render_report_markdown(accepted)
    artifact_manifest = _artifact_manifest(artifact_files)
    artifact_manifest_digest = stable_digest(artifact_manifest)
    publication_bundle_digest = _validate_artifact_files(
        accepted,
        artifact_files,
        content=content,
        markdown=markdown,
        source_result=source_result,
    )
    request = {
        "schema_version": 2,
        "scope_kind": scope_kind,
        "scope_id": scope_id,
        "document_kind": (
            accepted.kind
            if isinstance(accepted, ScientificReportV1)
            else "campaign_report_index"
        ),
        "document_digest": document_digest,
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "publication_bundle_digest": publication_bundle_digest,
        "artifact_manifest_digest": artifact_manifest_digest,
        "source_artifact_digests": list(sorted(source_artifact_digests)),
        "publication_destination": publication_destination.to_dict(),
        "task_source_projects": list(source_projects),
        "run_kind": "report_only",
        "excluded_from_task_inputs": True,
        "excluded_from_evaluation_counts": True,
        "artifact_manifest": artifact_manifest,
    }
    publication_id = stable_digest(_publication_identity_from_request(request))
    response, _created = publish_scoped_once(
        project=project,
        publication_id=publication_id,
        scope_kind=scope_kind,
        source_digest=document_digest,
        ledger_root=ledger_root,
        publish=lambda: publisher.publish(
            request=request,
            content=content,
            markdown=markdown,
            artifact_files=artifact_files,
        ),
    )
    return _receipt(request, response)


class WandbReportPublisher:
    """Late-bound W&B adapter for one report-only Run, artifact, and Report."""

    def __init__(
        self,
        *,
        wandb_module: Any | None = None,
        reports_module: Any | None = None,
        now: Any | None = None,
        phase_observer: Callable[[str], None] | None = None,
    ) -> None:
        self._wandb_module = wandb_module
        self._reports_module = reports_module
        self._now = now or (lambda: datetime.now(UTC))
        self._phase_observer = phase_observer

    def publish(
        self,
        *,
        request: Mapping[str, Any],
        content: bytes,
        markdown: str,
        artifact_files: Mapping[str, bytes],
    ) -> Mapping[str, Any]:
        _validate_request(request, content, artifact_files)
        destination = _request_destination(request)
        entity, project = destination.entity, destination.project
        publication_id = stable_digest(_publication_identity_from_request(request))
        run_id = f"fugue-report-{publication_id[:20]}"
        wandb = self._wandb()
        reports = self._reports()
        settings_factory = getattr(wandb, "Settings", None)
        if not callable(settings_factory):
            raise ScientificReportError(
                "W&B report publication requires destination-aware Settings"
            )
        run = wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            name=f"Fugue report · {request['scope_id']}",
            job_type="scientific-report",
            resume="allow",
            reinit="create_new",
            config={"fugue": dict(request)},
            settings=settings_factory(base_url=destination.api_base_url),
        )
        if run is None:
            raise ScientificReportError("W&B did not create the report-only Run")
        _validate_run_destination(run, destination)
        try:
            prior = run.summary.get(_SUMMARY_KEY)
            if prior is not None:
                completed = _completed_response(prior, publication_id, required=False)
                if completed is not None:
                    run.finish(exit_code=0)
                    return completed
            artifact_type = (
                "fugue-scientific-report"
                if request["scope_kind"] == "study"
                else "fugue-campaign-report-index"
            )
            artifact_name = f"fugue-report-{publication_id[:20]}"
            marker = (
                f"fugue-report:{publication_id}:"
                f"sha256:{request['content_sha256']}"
            )
            report_title = f"Fugue · {request['scope_id']} · {publication_id[:12]}"
            phase = _publication_phase_ledger(
                prior,
                publication_id=publication_id,
                request=request,
                run_id=str(run.id),
                artifact_name=artifact_name,
                artifact_type=artifact_type,
                report_title=report_title,
                report_marker=marker,
                published_at=self._now().astimezone(UTC).isoformat(),
            )
            _persist_publication_phase(run, phase)
            self._observe_phase("run")

            artifact_phase = dict(phase["artifact"])
            artifact_url = str(artifact_phase.get("artifact_url") or "")
            if artifact_phase["status"] != "completed":
                artifact_phase["status"] = "reserved"
                phase["artifact"] = artifact_phase
                _persist_publication_phase(run, phase)
                recovered = _recover_artifact(
                    wandb,
                    destination=destination,
                    entity=entity,
                    project=project,
                    name=artifact_name,
                    artifact_type=artifact_type,
                    publication_id=publication_id,
                    artifact_manifest_digest=str(
                        request["artifact_manifest_digest"]
                    ),
                    publication_bundle_digest=str(
                        request["publication_bundle_digest"]
                    ),
                )
                if recovered is None:
                    artifact = wandb.Artifact(
                        name=artifact_name,
                        type=artifact_type,
                        metadata={
                            "schema_version": 1,
                            "scope_kind": request["scope_kind"],
                            "scope_id": request["scope_id"],
                            "document_digest": request["document_digest"],
                            "content_sha256": request["content_sha256"],
                            "publication_id": publication_id,
                            "artifact_manifest_digest": request[
                                "artifact_manifest_digest"
                            ],
                            "publication_bundle_digest": request[
                                "publication_bundle_digest"
                            ],
                            "destination_digest": destination.destination_digest,
                            "run_kind": "report_only",
                        },
                    )
                    for name, value in sorted(artifact_files.items()):
                        with artifact.new_file(name, mode="wb") as stream:
                            stream.write(value)
                    recovered = run.log_artifact(artifact)
                    if hasattr(recovered, "wait"):
                        waited = recovered.wait()
                        if waited is not None:
                            recovered = waited
                artifact_identity = _wandb_artifact_identity(
                    recovered,
                    destination=destination,
                    expected_name=artifact_name,
                )
                artifact_url = artifact_identity["artifact_url"]
                artifact_phase.update(status="completed", **artifact_identity)
                phase["artifact"] = artifact_phase
                _persist_publication_phase(run, phase)
            self._observe_phase("artifact")

            report_phase = dict(phase["report"])
            report_url = str(report_phase.get("url") or "")
            if report_phase["status"] != "completed":
                report_phase["status"] = "reserved"
                phase["report"] = report_phase
                _persist_publication_phase(run, phase)
                recovered_report = _recover_report(
                    wandb,
                    destination=destination,
                    entity=entity,
                    project=project,
                    title=report_title,
                    marker=marker,
                )
                if recovered_report is None:
                    recovered_report = reports.Report(
                        entity=entity,
                        project=project,
                        title=report_title,
                        description=f"Digest-bound Fugue report. {marker}",
                        blocks=[
                            reports.TableOfContents(),
                            reports.MarkdownBlock(text=markdown),
                        ],
                    )
                    recovered_report.save()
                report_url = str(getattr(recovered_report, "url", "") or "")
                if not report_url:
                    raise ScientificReportError(
                        "W&B publication did not return a report link"
                    )
                report_phase.update(status="completed", url=report_url)
                phase["report"] = report_phase
                _persist_publication_phase(run, phase)
            self._observe_phase("report")

            if not artifact_url or not report_url:
                raise ScientificReportError(
                    "W&B publication did not return report links"
                )
            response = {
                "publication_run_id": str(run.id),
                "artifact_url": artifact_url,
                "artifact_ref": str(phase["artifact"]["artifact_ref"]),
                "artifact_version": str(phase["artifact"]["artifact_version"]),
                "artifact_digest": str(phase["artifact"]["artifact_digest"]),
                "report_url": report_url,
                "publisher_id": "wandb-workspaces-reports-v2",
                "published_at": str(phase["published_at"]),
            }
            phase["receipt"] = {"status": "completed", "response": response}
            _persist_publication_phase(run, phase)
            self._observe_phase("receipt")
            run.finish(exit_code=0)
            return response
        except Exception:
            run.finish(exit_code=1)
            raise

    def _observe_phase(self, phase: str) -> None:
        if self._phase_observer is not None:
            self._phase_observer(phase)

    def _wandb(self) -> Any:
        if self._wandb_module is not None:
            return self._wandb_module
        try:
            return importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            raise ScientificReportError(
                "W&B report publication requires the optional wandb package"
            ) from exc

    def _reports(self) -> Any:
        if self._reports_module is not None:
            return self._reports_module
        try:
            return importlib.import_module("wandb_workspaces.reports.v2")
        except ModuleNotFoundError as exc:
            raise ScientificReportError(
                "W&B Report publication requires the optional wandb-workspaces package"
            ) from exc


def attach_study_report_index(
    view: ExperimentViewV3,
    index: StudyReportIndexV1,
) -> ExperimentViewV3:
    accepted = study_report_index_from_dict(index.to_dict())
    projected = replace(view, report_index=accepted.to_dict())
    parsed = experiment_view_from_dict(projected.to_dict())
    if not isinstance(parsed, ExperimentViewV3):
        raise ScientificReportError("report index projection did not remain V3")
    return parsed


def record_study_report_index(
    *,
    store: StudyStore,
    research_id: str,
    view: ExperimentViewV3,
    index: StudyReportIndexV1,
) -> tuple[ExperimentViewV3, ResearchLogEventV1]:
    """Append one idempotent typed report index to the existing Research log."""

    projected = attach_study_report_index(view, index)
    publication = index.reports[0]
    evidence = (
        ResearchEvidenceRefV1(
            system="wandb",
            kind="report",
            ref=publication.publication.url,
            uri=publication.publication.url,
            digest=publication.report_sha256,
        ),
        ResearchEvidenceRefV1(
            system="wandb",
            kind="artifact",
            ref=next(
                item.url for item in publication.related_links if item.kind == "artifact"
            ),
            uri=next(
                item.url for item in publication.related_links if item.kind == "artifact"
            ),
            digest=publication.artifact_manifest_digest,
        ),
    )
    event = store.record_experiment_view_event(
        research_id=research_id,
        experiment_id=index.study_id,
        producer_event_id=f"fugue:{index.study_id}:report:{index.index_digest}",
        classification="result",
        state="completed",
        message="Digest-bound scientific report published.",
        view=projected,
        evidence=evidence,
    )
    return projected, event


def _canonical_source_result(result: ComparisonResultV3) -> ComparisonResultV3:
    """Reject subclasses and envelopes whose canonical digests do not verify."""

    if type(result) is not ComparisonResultV3:
        raise ScientificReportError(
            "Study report publication requires an exact ComparisonResultV3"
        )
    try:
        _verify_v2_result_integrity(result, has_qualification_digest=True)
    except ValueError as exc:
        raise ScientificReportError(
            "Study report source result failed canonical integrity verification"
        ) from exc
    return result


def _artifact_manifest(
    artifact_files: Mapping[str, bytes],
) -> dict[str, dict[str, str | int]]:
    names = set(artifact_files)
    if not names or any(
        not isinstance(name, str)
        or not name
        or name.startswith("/")
        or "\\" in name
        or ".." in name.split("/")
        for name in names
    ):
        raise ScientificReportError("report artifact file paths are invalid")
    if any(
        not isinstance(value, bytes) or not value
        for value in artifact_files.values()
    ):
        raise ScientificReportError("report artifact files must be nonempty bytes")
    return {
        name: {
            "sha256": hashlib.sha256(value).hexdigest(),
            "size_bytes": len(value),
        }
        for name, value in sorted(artifact_files.items())
    }


def _publication_identity_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "scope_kind",
        "scope_id",
        "document_kind",
        "document_digest",
        "content_sha256",
        "publication_bundle_digest",
        "artifact_manifest_digest",
        "source_artifact_digests",
        "publication_destination",
        "task_source_projects",
        "run_kind",
        "excluded_from_task_inputs",
        "excluded_from_evaluation_counts",
    )
    missing = [key for key in keys if key not in request]
    if missing:
        raise ScientificReportError(
            "report publication identity is incomplete: " + ", ".join(missing)
        )
    return {key: request[key] for key in keys}


def _request_destination(request: Mapping[str, Any]) -> EvidenceDestinationV1:
    raw = request.get("publication_destination")
    if not isinstance(raw, Mapping):
        raise ScientificReportError("report publication destination is invalid")
    try:
        destination = evidence_destination_from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ScientificReportError("report publication destination is invalid") from exc
    if destination.to_dict() != dict(raw):
        raise ScientificReportError(
            "report publication destination must carry its full derived identity"
        )
    return destination


def _validate_run_destination(run: Any, destination: EvidenceDestinationV1) -> None:
    observed_entity = str(getattr(run, "entity", "") or "")
    observed_project = str(getattr(run, "project", "") or "")
    if observed_entity and observed_entity != destination.entity:
        raise ScientificReportError("W&B report Run entity disagrees")
    if observed_project and observed_project != destination.project:
        raise ScientificReportError("W&B report Run project disagrees")


def _validated_destination_url(
    value: str,
    destination: EvidenceDestinationV1,
    *,
    label: str,
) -> str:
    parsed = urlsplit(value)
    app = urlsplit(destination.app_base_url)
    try:
        origin = (parsed.scheme, parsed.hostname, parsed.port)
        expected_origin = (app.scheme, app.hostname, app.port)
    except ValueError as exc:
        raise ScientificReportError(
            f"W&B {label} URL has an invalid origin"
        ) from exc
    project_path = (
        app.path.rstrip("/")
        + f"/{destination.entity}/{destination.project}"
    )
    if (
        origin != expected_origin
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not (
            parsed.path == project_path
            or parsed.path.startswith(project_path + "/")
        )
    ):
        raise ScientificReportError(
            f"W&B {label} URL is outside the publication destination"
        )
    return value


def _receipt(
    request: Mapping[str, Any], response: Mapping[str, Any]
) -> ReportPublicationReceiptV1:
    expected = {
        "publication_run_id",
        "artifact_url",
        "artifact_ref",
        "artifact_version",
        "artifact_digest",
        "report_url",
        "publisher_id",
        "published_at",
    }
    if set(response) != expected:
        raise ScientificReportError("report publisher returned an invalid response")
    from fugue.bench.scientific_reports import ReportLinkV1

    destination = _request_destination(request)
    report_id = str(request["document_digest"])
    report_sha256 = str(request["content_sha256"])
    report_url = _validated_destination_url(
        str(response["report_url"]),
        destination,
        label="report",
    )
    artifact_url = _validated_destination_url(
        str(response["artifact_url"]),
        destination,
        label="artifact",
    )
    artifact_ref = str(response["artifact_ref"])
    artifact_version = str(response["artifact_version"])
    artifact_digest = str(response["artifact_digest"])
    publication_id = stable_digest(_publication_identity_from_request(request))
    expected_artifact_name = f"fugue-report-{publication_id[:20]}"
    expected_ref = (
        f"wandb-artifact://{destination.project_slug}/"
        f"{expected_artifact_name}:{artifact_version}"
    )
    if (
        not re.fullmatch(r"v[0-9]+", artifact_version)
        or artifact_ref != expected_ref
        or not artifact_digest
    ):
        raise ScientificReportError("report publisher returned invalid artifact identity")
    publication = ReportLinkV1(
        schema_version=1,
        id=report_id,
        kind="report",
        label="Scientific report",
        url=report_url,
        content_sha256=report_sha256,
    )
    artifact = ReportLinkV1(
        schema_version=1,
        id=f"{report_id}:artifact:{artifact_version}",
        kind="artifact",
        label="Report artifact",
        url=artifact_url,
        content_sha256=str(request["artifact_manifest_digest"]),
    )
    return ReportPublicationReceiptV1(
        schema_version=1,
        scope_kind=str(request["scope_kind"]),  # type: ignore[arg-type]
        scope_id=str(request["scope_id"]),
        document_kind=str(request["document_kind"]),  # type: ignore[arg-type]
        publication_id=publication_id,
        report_id=report_id,
        report_sha256=report_sha256,
        publication_bundle_digest=str(request["publication_bundle_digest"]),
        artifact_manifest_digest=str(request["artifact_manifest_digest"]),
        source_artifact_digests=tuple(request["source_artifact_digests"]),
        task_source_projects=tuple(request["task_source_projects"]),
        publication_destination=destination,
        publication=publication,
        related_links=(artifact,),
        artifact_ref=artifact_ref,
        artifact_version=artifact_version,
        artifact_digest=artifact_digest,
        publication_run_id=str(response["publication_run_id"]),
        publication_run_kind="report_only",
        publisher_id=str(response["publisher_id"]),
        published_at=str(response["published_at"]),
    )


def _validate_request(
    request: Mapping[str, Any],
    content: bytes,
    artifact_files: Mapping[str, bytes],
) -> None:
    required = {
        "schema_version",
        "scope_kind",
        "scope_id",
        "document_kind",
        "document_digest",
        "content_sha256",
        "publication_bundle_digest",
        "artifact_manifest_digest",
        "source_artifact_digests",
        "publication_destination",
        "task_source_projects",
        "run_kind",
        "excluded_from_task_inputs",
        "excluded_from_evaluation_counts",
        "artifact_manifest",
    }
    if set(request) != required or request.get("schema_version") != 2:
        raise ScientificReportError("report publication request is invalid")
    if request.get("scope_kind") not in {"study", "campaign"}:
        raise ScientificReportError("report publication scope is invalid")
    expected_kind = (
        "scientific_report"
        if request.get("scope_kind") == "study"
        else "campaign_report_index"
    )
    if request.get("document_kind") != expected_kind:
        raise ScientificReportError("report publication document kind is invalid")
    if not isinstance(request.get("scope_id"), str) or not request["scope_id"]:
        raise ScientificReportError("report publication scope identity is invalid")
    document_digest = request.get("document_digest")
    if (
        not isinstance(document_digest, str)
        or len(document_digest) != 64
        or any(char not in "0123456789abcdef" for char in document_digest)
    ):
        raise ScientificReportError("report publication digest is invalid")
    destination = _request_destination(request)
    sources = request.get("task_source_projects")
    if (
        not isinstance(sources, list)
        or not sources
        or sources != sorted(set(sources))
        or any(not isinstance(item, str) or item.count("/") != 1 for item in sources)
    ):
        raise ScientificReportError("report task-source projects are invalid")
    source_digests = request.get("source_artifact_digests")
    if (
        not isinstance(source_digests, list)
        or not source_digests
        or source_digests != sorted(set(source_digests))
        or any(
            not isinstance(item, str)
            or len(item) != 64
            or any(char not in "0123456789abcdef" for char in item)
            for item in source_digests
        )
    ):
        raise ScientificReportError("report source artifact digests are invalid")
    if request.get("run_kind") != "report_only" or not (
        request.get("excluded_from_task_inputs")
        and request.get("excluded_from_evaluation_counts")
    ):
        raise ScientificReportError("report publication entered evaluation scope")
    if request.get("content_sha256") != hashlib.sha256(content).hexdigest():
        raise ScientificReportError("report publication bytes do not match")
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScientificReportError("report publication document is not JSON") from exc
    digest_key = "report_digest" if expected_kind == "scientific_report" else "index_digest"
    if (
        not isinstance(document, Mapping)
        or document.get("kind", expected_kind) != expected_kind
        or document.get(digest_key) != document_digest
    ):
        raise ScientificReportError("report publication document identity disagrees")
    if destination.project_slug in request.get("task_source_projects", ()):
        raise ScientificReportError("report publication targets a task-source project")
    expected_manifest = _artifact_manifest(artifact_files)
    if request.get("artifact_manifest") != expected_manifest:
        raise ScientificReportError("report artifact files do not match their request")
    expected_manifest_digest = stable_digest(expected_manifest)
    if request.get("artifact_manifest_digest") != expected_manifest_digest:
        raise ScientificReportError("report artifact manifest digest does not match")
    bundle_digest = request.get("publication_bundle_digest")
    if not isinstance(bundle_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", bundle_digest
    ):
        raise ScientificReportError("report publication bundle digest is invalid")
    expected_publication_id = stable_digest(_publication_identity_from_request(request))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_publication_id):
        raise ScientificReportError("report publication identity is invalid")


def _validate_artifact_files(
    document: ScientificReportV1 | CampaignReportIndexV1,
    artifact_files: Mapping[str, bytes],
    *,
    content: bytes,
    markdown: str,
    source_result: ComparisonResultV3 | None,
) -> str:
    names = set(artifact_files)
    if not names or any(
        not isinstance(name, str)
        or not name
        or name.startswith("/")
        or "\\" in name
        or ".." in name.split("/")
        for name in names
    ):
        raise ScientificReportError("report artifact file paths are invalid")
    if any(not isinstance(value, bytes) or not value for value in artifact_files.values()):
        raise ScientificReportError("report artifact files must be nonempty bytes")
    encoded_markdown = markdown.encode("utf-8")
    if artifact_files.get("report.md") != encoded_markdown:
        raise ScientificReportError("report artifact Markdown does not recompute")
    if isinstance(document, ScientificReportV1):
        assert source_result is not None
        manifest = validate_scientific_report_bundle_files(
            artifact_files,
            result=source_result,
            report=document,
        )
        if artifact_files["report.json"] != content:
            raise ScientificReportError("report artifact JSON does not recompute")
        return manifest.bundle_digest
    else:
        if names != {"campaign-index.json", "report.md"}:
            raise ScientificReportError(
                "campaign report artifact contains unexpected files"
            )
        if artifact_files["campaign-index.json"] != content:
            raise ScientificReportError("campaign report index does not recompute")
        return stable_digest(
            {
                "schema_version": 1,
                "kind": "campaign_report_bundle",
                "document_digest": document.index_digest,
                "files": _artifact_manifest(artifact_files),
            }
        )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _completed_response(
    value: Any, publication_id: str, *, required: bool = True
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping) or value.get("publication_id") != publication_id:
        raise ScientificReportError("stored W&B report receipt conflicts")
    if value.get("schema_version") == 1:
        response = value.get("response")
    elif value.get("schema_version") == _PHASE_SCHEMA_VERSION:
        receipt = value.get("receipt")
        response = receipt.get("response") if isinstance(receipt, Mapping) else None
        if isinstance(receipt, Mapping) and receipt.get("status") != "completed":
            response = None
    else:
        raise ScientificReportError("stored W&B report receipt conflicts")
    if response is None and not required:
        return None
    if not isinstance(response, Mapping):
        raise ScientificReportError("stored W&B report receipt conflicts")
    return dict(response)


def _publication_phase_ledger(
    value: Any,
    *,
    publication_id: str,
    request: Mapping[str, Any],
    run_id: str,
    artifact_name: str,
    artifact_type: str,
    report_title: str,
    report_marker: str,
    published_at: str,
) -> dict[str, Any]:
    request_digest = stable_digest(dict(request))
    if value is None:
        return {
            "schema_version": _PHASE_SCHEMA_VERSION,
            "publication_id": publication_id,
            "request_digest": request_digest,
            "run": {"status": "completed", "id": run_id},
            "artifact": {
                "status": "pending",
                "name": artifact_name,
                "type": artifact_type,
                "artifact_url": None,
                "artifact_ref": None,
                "artifact_version": None,
                "artifact_digest": None,
            },
            "report": {
                "status": "pending",
                "title": report_title,
                "marker": report_marker,
                "url": None,
            },
            "receipt": {"status": "pending", "response": None},
            "published_at": published_at,
        }
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != _PHASE_SCHEMA_VERSION
    ):
        raise ScientificReportError("stored W&B report phase ledger conflicts")
    phase = json.loads(json.dumps(dict(value)))
    expected = {
        "schema_version",
        "publication_id",
        "request_digest",
        "run",
        "artifact",
        "report",
        "receipt",
        "published_at",
    }
    if set(phase) != expected or (
        phase.get("publication_id") != publication_id
        or phase.get("request_digest") != request_digest
        or phase.get("run") != {"status": "completed", "id": run_id}
    ):
        raise ScientificReportError("stored W&B report phase ledger conflicts")
    artifact = phase.get("artifact")
    report = phase.get("report")
    receipt = phase.get("receipt")
    if not isinstance(artifact, dict) or set(artifact) != {
        "status",
        "name",
        "type",
        "artifact_url",
        "artifact_ref",
        "artifact_version",
        "artifact_digest",
    }:
        raise ScientificReportError("stored W&B artifact phase conflicts")
    if (
        artifact.get("status") not in {"pending", "reserved", "completed"}
        or artifact.get("name") != artifact_name
        or artifact.get("type") != artifact_type
        or (
            artifact.get("status") == "completed"
            and any(
                not artifact.get(key)
                for key in (
                    "artifact_url",
                    "artifact_ref",
                    "artifact_version",
                    "artifact_digest",
                )
            )
        )
    ):
        raise ScientificReportError("stored W&B artifact phase conflicts")
    if not isinstance(report, dict) or set(report) != {
        "status",
        "title",
        "marker",
        "url",
    }:
        raise ScientificReportError("stored W&B Report phase conflicts")
    if (
        report.get("status") not in {"pending", "reserved", "completed"}
        or report.get("title") != report_title
        or report.get("marker") != report_marker
        or (report.get("status") == "completed" and not report.get("url"))
    ):
        raise ScientificReportError("stored W&B Report phase conflicts")
    if not isinstance(receipt, dict) or set(receipt) != {"status", "response"}:
        raise ScientificReportError("stored W&B receipt phase conflicts")
    if receipt.get("status") not in {"pending", "completed"} or (
        receipt.get("status") == "completed"
        and not isinstance(receipt.get("response"), Mapping)
    ):
        raise ScientificReportError("stored W&B receipt phase conflicts")
    return phase


def _persist_publication_phase(run: Any, phase: Mapping[str, Any]) -> None:
    run.summary[_SUMMARY_KEY] = json.loads(json.dumps(dict(phase)))


def _recover_artifact(
    wandb: Any,
    *,
    destination: EvidenceDestinationV1,
    entity: str,
    project: str,
    name: str,
    artifact_type: str,
    publication_id: str,
    artifact_manifest_digest: str,
    publication_bundle_digest: str,
) -> Any | None:
    api_factory = getattr(wandb, "Api", None)
    if not callable(api_factory):
        return None
    api = api_factory(overrides={"base_url": destination.api_base_url})
    ref = f"{entity}/{project}/{name}:latest"
    exists = getattr(api, "artifact_exists", None)
    fetch = getattr(api, "artifact", None)
    if not callable(exists) or not callable(fetch):
        raise ScientificReportError(
            "W&B artifact recovery API is unavailable after reservation"
        )
    if not exists(ref, type=artifact_type):
        return None
    artifact = fetch(ref, type=artifact_type)
    metadata = getattr(artifact, "metadata", None)
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("publication_id") != publication_id
        or metadata.get("artifact_manifest_digest") != artifact_manifest_digest
        or metadata.get("publication_bundle_digest") != publication_bundle_digest
        or metadata.get("destination_digest") != destination.destination_digest
    ):
        raise ScientificReportError("recovered W&B artifact identity metadata drifted")
    _wandb_artifact_identity(
        artifact,
        destination=destination,
        expected_name=name,
    )
    return artifact


def _recover_report(
    wandb: Any,
    *,
    destination: EvidenceDestinationV1,
    entity: str,
    project: str,
    title: str,
    marker: str,
) -> Any | None:
    api_factory = getattr(wandb, "Api", None)
    if not callable(api_factory):
        return None
    api = api_factory(overrides={"base_url": destination.api_base_url})
    reports = getattr(api, "reports", None)
    if not callable(reports):
        raise ScientificReportError(
            "W&B Report recovery API is unavailable after reservation"
        )
    matches = [
        report
        for report in reports(f"{entity}/{project}", name=title)
        if marker in str(getattr(report, "description", "") or "")
    ]
    if len(matches) > 1:
        raise ScientificReportError("W&B Report identity is duplicated")
    return matches[0] if matches else None


def _wandb_artifact_identity(
    artifact: Any,
    *,
    destination: EvidenceDestinationV1,
    expected_name: str,
) -> dict[str, str]:
    artifact_url = str(getattr(artifact, "url", "") or "")
    version = str(getattr(artifact, "version", "") or "")
    qualified = str(
        getattr(artifact, "qualified_name", "")
        or getattr(artifact, "name", "")
        or ""
    )
    digest = str(getattr(artifact, "digest", "") or "")
    if not version and ":" in qualified:
        version = qualified.rsplit(":", 1)[1]
    if not re.fullmatch(r"v[0-9]+", version):
        raise ScientificReportError("W&B artifact has no immutable version")
    if ":" not in qualified:
        qualified = f"{qualified}:{version}"
    if qualified.startswith(f"{expected_name}:"):
        qualified = f"{destination.project_slug}/{qualified}"
    expected_prefix = f"{destination.project_slug}/{expected_name}:"
    if not qualified.startswith(expected_prefix) or not qualified.endswith(f":{version}"):
        raise ScientificReportError("W&B artifact qualified identity disagrees")
    if not digest:
        raise ScientificReportError("W&B artifact has no immutable digest")
    _validated_destination_url(artifact_url, destination, label="artifact")
    if version not in urlsplit(artifact_url).path.split("/"):
        raise ScientificReportError("W&B artifact URL is not version-qualified")
    return {
        "artifact_url": artifact_url,
        "artifact_ref": f"wandb-artifact://{qualified}",
        "artifact_version": version,
        "artifact_digest": digest,
    }
